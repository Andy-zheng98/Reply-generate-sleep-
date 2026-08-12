#!/usr/bin/env python3
"""Create a current, machine-readable audit of the bundled sleep model."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT / "sleep_lora" / "checkpoint-160"
PACKAGE = ROOT / "packages" / "chatglm3_colab_lora_package.zip"
EXPECTED_HASH = "207588c4401877ce68a1ed8006ba7ec165ecfa8bcc18308b7db375e78d4ae3c5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_records(data: bytes) -> int:
    text = data.decode("utf-8")
    try:
        value = json.loads(text)
        return len(value) if isinstance(value, list) else 1
    except json.JSONDecodeError:
        return sum(1 for line in text.splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "睡眠测试" / "睡眠模型与ChatGLM3对比测试" / "repository" / "repository_audit.json",
    )
    args = parser.parse_args()

    required_adapter = [ADAPTER / "adapter_config.json", ADAPTER / "adapter_model.safetensors"]
    adapter_available = all(path.is_file() for path in required_adapter)
    adapter_hash = sha256(ADAPTER / "adapter_model.safetensors") if adapter_available else None
    if adapter_available:
        assert adapter_hash == EXPECTED_HASH
    config = json.loads((ADAPTER / "adapter_config.json").read_text(encoding="utf-8")) if adapter_available else {}

    with zipfile.ZipFile(PACKAGE) as archive:
        package_files = sorted(name for name in archive.namelist() if not name.endswith("/"))
        train_examples = count_records(archive.read("data/train.json"))
        dev_examples = count_records(archive.read("data/dev.json"))

    report = {
        "audit_scope": "current repository files",
        "base_model": config.get("base_model_name_or_path", "zai-org/chatglm3-6b"),
        "training_method": {
            "peft_type": config.get("peft_type"),
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "target_modules": config.get("target_modules"),
        },
        "train_entry": "packages/chatglm3_colab_lora_package.zip::finetune_hf.py",
        "inference_entries": [
            "sleep_chat.py",
            "睡眠测试/睡眠模型与ChatGLM3对比测试/scripts/run_colab_fp16_pair.py",
            "睡眠测试/失眠非药物专项测试/scripts/run_colab_v4.py",
        ],
        "domain_changes": [
            "睡眠.ipynb",
            "LoRA training package",
            f"{train_examples + dev_examples}-example sleep dialogue dataset ({train_examples} train/{dev_examples} dev)",
            "bundled checkpoint-160 adapter",
            "two reproducible sleep test suites",
        ],
        "package_files": package_files,
        "adapter_available": adapter_available,
        "adapter_sha256": adapter_hash,
        "adapter_files": [str(path.relative_to(ROOT)) for path in required_adapter if path.is_file()],
        "critical_finding": (
            "The trained checkpoint-160 adapter is bundled in the repository and passes its recorded SHA-256 check. "
            "The ChatGLM3 base model is still downloaded separately at runtime."
            if adapter_available
            else "The adapter is missing, so the sleep model cannot be loaded from this repository."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"adapter_available": adapter_available, "adapter_sha256": adapter_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
