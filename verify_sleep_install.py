#!/usr/bin/env python3
"""Dependency-free integrity check for the bundled adapter and benchmark artifacts."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "sleep_lora" / "checkpoint-160"
EXPECTED_HASH = "207588c4401877ce68a1ed8006ba7ec165ecfa8bcc18308b7db375e78d4ae3c5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-runtime", action="store_true", help="同时检查 PyTorch/Transformers/PEFT 是否已安装")
    args = parser.parse_args()

    config = json.loads((ADAPTER / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == "zai-org/chatglm3-6b"
    assert config["r"] == 8 and config["lora_alpha"] == 32 and config["lora_dropout"] == 0.1
    assert config["target_modules"] == ["query_key_value"]

    model_path = ADAPTER / "adapter_model.safetensors"
    actual_hash = sha256(model_path)
    assert actual_hash == EXPECTED_HASH, (actual_hash, EXPECTED_HASH)
    with model_path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    tensor_keys = [key for key in header if key != "__metadata__"]
    assert len(tensor_keys) == 56

    benchmark = read_jsonl(ROOT / "sleep_benchmark" / "benchmark" / "benchmark.jsonl")
    baseline = read_jsonl(ROOT / "sleep_benchmark" / "results" / "baseline_fp16_raw.jsonl")
    sleep = read_jsonl(ROOT / "sleep_benchmark" / "results" / "sleep_lora_raw.jsonl")
    assert len(benchmark) == 26 and sum(len(row["turns"]) for row in benchmark) == 38
    assert len(baseline) == len(sleep) == 14
    assert [row["id"] for row in baseline] == [row["id"] for row in sleep]
    assert sum(len(row["answers"]) for row in baseline) == 18

    if args.check_runtime:
        import accelerate
        import peft
        import safetensors
        import torch
        import transformers
        print("runtime", torch.__version__, transformers.__version__, peft.__version__, accelerate.__version__, safetensors.__version__)

    print("OK: adapter SHA-256", actual_hash)
    print("OK: 56 LoRA tensors, 26 benchmark cases, 14 paired executed cases / 18 turns")


if __name__ == "__main__":
    main()
