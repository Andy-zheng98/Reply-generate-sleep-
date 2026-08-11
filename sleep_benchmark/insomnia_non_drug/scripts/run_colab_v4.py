#!/usr/bin/env python3
"""Run the held-out insomnia/non-drug suite on ChatGLM3-6B + bundled Sleep LoRA."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUITE = ROOT / "sleep_benchmark" / "insomnia_non_drug"
BENCHMARK_PATH = SUITE / "benchmark" / "benchmark_v4.jsonl"
RESULTS_DIR = SUITE / "results"
RAW_PATH = RESULTS_DIR / "sleep_lora_raw_v4.jsonl"
METADATA_PATH = RESULTS_DIR / "execution_metadata_v4.json"
ITEM_PATH = RESULTS_DIR / "sleep_lora_item_scores_v4.csv"
SUMMARY_PATH = RESULTS_DIR / "sleep_lora_summary_v4.json"
ARCHIVE_PATH = Path("/content/insomnia_non_drug_results_v4.zip")
BASE_MODEL = "zai-org/chatglm3-6b"
EXPECTED_ADAPTER_SHA256 = "207588c4401877ce68a1ed8006ba7ec165ecfa8bcc18308b7db375e78d4ae3c5"
GENERATION = {"temperature": 0.2, "top_p": 0.8, "max_new_tokens": 320, "seed": 4242}


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_runtime() -> None:
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


def main() -> None:
    started = time.time()
    install_runtime()

    import accelerate
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    assert torch.cuda.is_available(), "CUDA GPU runtime is required"
    adapter_dir = ROOT / "sleep_lora" / "checkpoint-160"
    adapter_file = adapter_dir / "adapter_model.safetensors"
    actual_hash = sha256(adapter_file)
    assert actual_hash == EXPECTED_ADAPTER_SHA256, (actual_hash, EXPECTED_ADAPTER_SHA256)
    cases = [
        json.loads(line)
        for line in BENCHMARK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cases) == 20
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {BASE_MODEL} + verified Sleep LoRA on {torch.cuda.get_device_name(0)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()
    model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True).eval()

    records: list[dict] = []
    total_tokens = 0
    total_latency = 0.0
    global_turn = 0
    for case in cases:
        history = []
        answers = []
        for prompt in case["turns"]:
            seed = GENERATION["seed"] + global_turn
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize()
            turn_started = time.perf_counter()
            with torch.inference_mode():
                response, history = model.chat(
                    tokenizer,
                    prompt,
                    history=history,
                    temperature=GENERATION["temperature"],
                    top_p=GENERATION["top_p"],
                    max_new_tokens=GENERATION["max_new_tokens"],
                    do_sample=True,
                )
            torch.cuda.synchronize()
            latency = time.perf_counter() - turn_started
            completion_tokens = len(tokenizer.encode(response, add_special_tokens=False))
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            answers.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "latency_s": latency,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_s": completion_tokens / latency if latency else None,
                    "seed": seed,
                }
            )
            total_tokens += completion_tokens
            total_latency += latency
            global_turn += 1

        record = {
            **case,
            "model": "ChatGLM3-6B + Sleep LoRA checkpoint-160",
            "base_model": BASE_MODEL,
            "adapter_sha256": actual_hash,
            "generation": GENERATION,
            "answers": answers,
            "execution_status": "executed",
        }
        records.append(record)
        case_latency = sum(answer["latency_s"] for answer in answers)
        case_tokens = sum(answer["completion_tokens"] for answer in answers)
        print(
            f"CASE {case['id']} {len(records)}/{len(cases)} turns={len(answers)} "
            f"latency={case_latency:.2f}s tokens={case_tokens} tok/s={case_tokens / case_latency:.2f}",
            flush=True,
        )
        RAW_PATH.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
            encoding="utf-8",
        )

    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "execution_status": "executed",
        "hardware": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "repository_commit": commit,
        "base_model": BASE_MODEL,
        "adapter_sha256": actual_hash,
        "generation": GENERATION,
        "n_cases": len(records),
        "n_turns": sum(len(record["answers"]) for record in records),
        "completion_tokens": total_tokens,
        "generation_latency_s": total_latency,
        "tokens_per_s": total_tokens / total_latency,
        "wall_time_s": time.time() - started,
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run(
        [
            sys.executable,
            str(SUITE / "scripts" / "score_v4.py"),
            "--input",
            str(RAW_PATH),
            "--item-csv",
            str(ITEM_PATH),
            "--summary",
            str(SUMMARY_PATH),
        ]
    )

    archive_root = Path("/content/insomnia_non_drug_results_v4")
    if archive_root.exists():
        shutil.rmtree(archive_root)
    shutil.copytree(RESULTS_DIR, archive_root / "results")
    shutil.copytree(SUITE / "benchmark", archive_root / "benchmark")
    shutil.copytree(SUITE / "scripts", archive_root / "scripts")
    shutil.make_archive(str(ARCHIVE_PATH.with_suffix("")), "zip", archive_root)
    print("RUN_COMPLETE", json.dumps(metadata, ensure_ascii=False), flush=True)
    print("RESULT_ARCHIVE", ARCHIVE_PATH, flush=True)


if __name__ == "__main__":
    main()
