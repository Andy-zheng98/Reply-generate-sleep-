#!/usr/bin/env python3
"""Run the repository's bundled ChatGLM3 sleep LoRA without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_ADAPTER_SHA256 = "207588c4401877ce68a1ed8006ba7ec165ecfa8bcc18308b7db375e78d4ae3c5"
DEFAULT_BASE_MODEL = "zai-org/chatglm3-6b"
DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent / "sleep_lora" / "checkpoint-160"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接加载仓库内置的睡眠 LoRA；不进行训练，也不下载 adapter。"
    )
    parser.add_argument("--prompt", help="单轮提问；不提供时进入连续对话模式")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Hugging Face 模型 ID 或本地基座目录")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR, help="本地 LoRA adapter 目录")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--offline", action="store_true", help="只使用已有 Hugging Face 缓存，不访问网络")
    parser.add_argument("--skip-hash-check", action="store_true", help="跳过 adapter SHA-256 校验")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_adapter(adapter_dir: Path, skip_hash_check: bool) -> None:
    required = [adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("仓库中的 adapter 不完整，缺少：" + ", ".join(missing))
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    expected = {"peft_type": "LORA", "r": 8, "lora_alpha": 32, "lora_dropout": 0.1}
    invalid = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if invalid:
        raise ValueError(f"adapter_config.json 与已验证配置不一致：{invalid}")
    if not skip_hash_check:
        actual = sha256(adapter_dir / "adapter_model.safetensors")
        if actual != EXPECTED_ADAPTER_SHA256:
            raise ValueError(f"adapter 权重校验失败：期望 {EXPECTED_ADAPTER_SHA256}，实际 {actual}")


def choose_runtime(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA；请改用 --device mps 或 --device cpu。")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("未检测到 Apple Metal/MPS；请改用 --device cpu。")
    dtype = torch.float16 if requested in {"cuda", "mps"} else torch.float32
    device_map = "auto" if requested == "cuda" else {"": requested}
    return torch, requested, dtype, device_map


def load_model(args: argparse.Namespace):
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    adapter_dir = args.adapter_dir.expanduser().resolve()
    verify_adapter(adapter_dir, args.skip_hash_check)
    torch, device, dtype, device_map = choose_runtime(args.device)
    print(f"正在加载 ChatGLM3 基座（device={device}）……", flush=True)
    if args.offline:
        print("离线模式：只读取已有 Hugging Face 缓存。", flush=True)
    common = {"trust_remote_code": True, "local_files_only": args.offline}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **common)
    base_model = AutoModel.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        **common,
    ).eval()
    model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True).eval()
    print(f"睡眠 LoRA 已加载：{adapter_dir}", flush=True)
    return torch, model, tokenizer


def ask(torch, model, tokenizer, prompt: str, history: list[dict], args: argparse.Namespace, turn: int):
    torch.manual_seed(args.seed + turn)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + turn)
    with torch.inference_mode():
        return model.chat(
            tokenizer,
            prompt,
            history=history,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
        )


def main() -> int:
    args = parse_args()
    if not 0 < args.temperature:
        raise SystemExit("--temperature 必须大于 0")
    if not 0 < args.top_p <= 1:
        raise SystemExit("--top-p 必须在 (0, 1] 范围内")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens 必须大于 0")

    print("提示：该实验模型存在医学与高风险安全缺陷，不能替代医生或紧急服务。")
    try:
        torch, model, tokenizer = load_model(args)
    except Exception as exc:
        print(f"\n模型加载失败：{exc}", file=sys.stderr)
        print("如果是内存不足，请使用带 14GB 以上显存的 NVIDIA GPU；Apple Silicon 建议至少 16GB 统一内存。", file=sys.stderr)
        return 1

    history: list[dict] = []
    if args.prompt:
        response, _ = ask(torch, model, tokenizer, args.prompt, history, args, 0)
        print(response)
        return 0

    print("\n输入问题开始对话；/reset 清空上下文，/quit 退出。")
    turn = 0
    while True:
        try:
            prompt = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("已退出。")
            return 0
        if prompt.lower() == "/reset":
            history = []
            turn = 0
            print("上下文已清空。")
            continue
        response, history = ask(torch, model, tokenizer, prompt, history, args, turn)
        turn += 1
        print(f"睡眠助手：{response}")


if __name__ == "__main__":
    raise SystemExit(main())
