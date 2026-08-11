#!/usr/bin/env python3
"""Run identical ChatGLM3 FP16 generations before and after attaching the sleep LoRA."""
import base64
import gc
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "zai-org/chatglm3-6b"
ADAPTER_DIR = Path("/content/eval_bundle/adapter/checkpoint-160")
BENCHMARK = Path("/content/eval_bundle/benchmark.jsonl")
OUT_DIR = Path("/content/fp16_pair_results")
SELECTED_IDS = {
    "S01", "S04", "S07", "S10", "S13", "S14", "S15", "S16",
    "S17", "S18", "S19", "S20", "M01", "M05",
}
GENERATION = {
    "temperature": 0.2,
    "top_p": 0.8,
    "max_new_tokens": 256,
    "do_sample": True,
    "seed": 42,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_cases(model, tokenizer, rows, model_name):
    output = []
    model.eval()
    for index, item in enumerate(rows, 1):
        history = []
        answers = []
        total_tokens = 0
        case_start = time.perf_counter()
        for turn_index, prompt in enumerate(item["turns"]):
            set_seed(GENERATION["seed"] + turn_index)
            torch.cuda.synchronize()
            started = time.perf_counter()
            response, history = model.chat(
                tokenizer,
                prompt,
                history=history,
                temperature=GENERATION["temperature"],
                top_p=GENERATION["top_p"],
                max_new_tokens=GENERATION["max_new_tokens"],
                do_sample=GENERATION["do_sample"],
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            token_count = len(tokenizer.encode(response, add_special_tokens=False))
            total_tokens += token_count
            answers.append({
                "prompt": prompt,
                "response": response,
                "latency_s": elapsed,
                "completion_tokens": token_count,
                "tokens_per_s": token_count / elapsed if elapsed else None,
            })
        case_elapsed = time.perf_counter() - case_start
        record = {
            **item,
            "model": model_name,
            "generation": GENERATION,
            "answers": answers,
            "latency_s": case_elapsed,
            "prompt_tokens": None,
            "completion_tokens": total_tokens,
            "tokens_per_s": total_tokens / case_elapsed if case_elapsed else None,
            "execution_status": "executed_on_colab_t4_fp16",
        }
        output.append(record)
        print(
            f"PROGRESS {model_name} {index}/{len(rows)} {item['id']} "
            f"{case_elapsed:.2f}s {record['tokens_per_s']:.2f}tok/s",
            flush=True,
        )
    return output


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["id"] in SELECTED_IDS]
    print(f"CASES {len(rows)} TURNS {sum(len(row['turns']) for row in rows)}", flush=True)

    adapter_hash = hashlib.sha256((ADAPTER_DIR / "adapter_model.safetensors").read_bytes()).hexdigest()
    print(f"ADAPTER_SHA256 {adapter_hash}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    print(f"BASE_LOADED {torch.cuda.get_device_name(0)}", flush=True)

    baseline = run_cases(base_model, tokenizer, rows, "chatglm3-6b-fp16")
    write_jsonl(OUT_DIR / "baseline_fp16_raw.jsonl", baseline)

    sleep_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).eval()
    print("ADAPTER_LOADED", flush=True)
    sleep = run_cases(sleep_model, tokenizer, rows, "chatglm3-6b-sleep-lora-checkpoint-160")
    write_jsonl(OUT_DIR / "sleep_lora_raw.jsonl", sleep)

    metadata = {
        "hardware": torch.cuda.get_device_name(0),
        "base_model": BASE_MODEL,
        "adapter_sha256": adapter_hash,
        "adapter_checkpoint": 160,
        "case_count": len(rows),
        "turn_count": sum(len(row["turns"]) for row in rows),
        "generation": GENERATION,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "torch_version": torch.__version__,
    }
    (OUT_DIR / "execution_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result_zip = Path("/content/fp16_pair_results.zip")
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in OUT_DIR.iterdir():
            archive.write(path, path.name)
    payload = base64.b64encode(result_zip.read_bytes()).decode("ascii")
    print(f"RESULT_SHA256 {hashlib.sha256(result_zip.read_bytes()).hexdigest()}", flush=True)
    print(f"RESULT_B64:{payload}", flush=True)
    del sleep_model, base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
