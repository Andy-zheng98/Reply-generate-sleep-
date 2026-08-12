#!/usr/bin/env python3
"""Run identical ChatGLM3 FP16 generations before and after attaching the sleep LoRA."""
import argparse
import base64
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

BASE_MODEL = "zai-org/chatglm3-6b"
ROOT = Path(__file__).resolve().parents[3]
SUITE = ROOT / "睡眠测试" / "睡眠模型与ChatGLM3对比测试"
ADAPTER_DIR = ROOT / "sleep_lora" / "checkpoint-160"
BENCHMARK = SUITE / "benchmark" / "benchmark.jsonl"
OUT_DIR = Path("/content/fp16_pair_results")
EXPECTED_ADAPTER_SHA256 = "207588c4401877ce68a1ed8006ba7ec165ecfa8bcc18308b7db375e78d4ae3c5"
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


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def install_runtime():
    required = {"transformers": "4.40.2", "peft": "0.10.0", "accelerate": "0.30.1"}
    needs_install = False
    try:
        import accelerate
        import peft
        import transformers
        current = {
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
        }
        needs_install = current != required
    except ImportError:
        needs_install = True
    if needs_install:
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements-sleep-inference.txt")])
        if os.environ.get("SLEEP_PAIR_REEXECED") == "1":
            raise RuntimeError("Pinned inference dependencies still do not match after reinstall")
        os.environ["SLEEP_PAIR_REEXECED"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def load_cases():
    rows = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["id"] in SELECTED_IDS]
    assert len(rows) == 14
    assert sum(len(row["turns"]) for row in rows) == 18
    return rows


def preflight():
    required = [
        ADAPTER_DIR / "adapter_config.json",
        ADAPTER_DIR / "adapter_model.safetensors",
        BENCHMARK,
        ROOT / "requirements-sleep-inference.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    adapter_hash = hashlib.sha256((ADAPTER_DIR / "adapter_model.safetensors").read_bytes()).hexdigest()
    assert adapter_hash == EXPECTED_ADAPTER_SHA256, (adapter_hash, EXPECTED_ADAPTER_SHA256)
    config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == BASE_MODEL
    rows = load_cases()
    print(f"PREFLIGHT_OK cases={len(rows)} turns={sum(len(row['turns']) for row in rows)} adapter_sha256={adapter_hash}")
    return rows, adapter_hash


def set_seed(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_cases(model, tokenizer, rows, model_name, np, torch):
    output = []
    model.eval()
    for index, item in enumerate(rows, 1):
        history = []
        answers = []
        total_tokens = 0
        case_start = time.perf_counter()
        for turn_index, prompt in enumerate(item["turns"]):
            set_seed(GENERATION["seed"] + turn_index, np, torch)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="只校验仓库文件、adapter 和测试集，不下载或加载模型")
    args = parser.parse_args()
    rows, adapter_hash = preflight()
    if args.check_only:
        return

    install_runtime()
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    assert torch.cuda.is_available(), "CUDA GPU runtime is required"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"CASES {len(rows)} TURNS {sum(len(row['turns']) for row in rows)}", flush=True)

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

    baseline = run_cases(base_model, tokenizer, rows, "chatglm3-6b-fp16", np, torch)
    write_jsonl(OUT_DIR / "baseline_fp16_raw.jsonl", baseline)

    sleep_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).eval()
    print("ADAPTER_LOADED", flush=True)
    sleep = run_cases(sleep_model, tokenizer, rows, "chatglm3-6b-sleep-lora-checkpoint-160", np, torch)
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
