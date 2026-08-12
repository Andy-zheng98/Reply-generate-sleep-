#!/usr/bin/env python3
"""Run every dependency-light integrity, scoring, leakage, and reproduction check."""

from __future__ import annotations

import csv
import json
import py_compile
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
COMPARISON = ROOT / "睡眠测试" / "睡眠模型与ChatGLM3对比测试"
INSOMNIA = ROOT / "睡眠测试" / "失眠非药物专项测试"
PYTHON = sys.executable


def run(*args: str) -> None:
    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def same_file(generated: Path, tracked: Path) -> None:
    if generated.suffix == ".json" and tracked.suffix == ".json":
        generated_value = json.loads(generated.read_text(encoding="utf-8"))
        tracked_value = json.loads(tracked.read_text(encoding="utf-8"))
        if generated_value != tracked_value:
            raise AssertionError(f"Reproduced JSON differs from tracked artifact: {tracked.relative_to(ROOT)}")
        return
    if generated.read_bytes() != tracked.read_bytes():
        raise AssertionError(f"Reproduced file differs from tracked artifact: {tracked.relative_to(ROOT)}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decode_json_or_jsonl(data: bytes) -> list[dict]:
    text = data.decode("utf-8")
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def validate_static_files() -> None:
    for path in [ROOT / "verify_sleep_install.py", ROOT / "sleep_chat.py", *sorted((ROOT / "睡眠测试").rglob("*.py"))]:
        py_compile.compile(str(path), doraise=True)

    for path in sorted((ROOT / "睡眠测试").rglob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
    for path in sorted((ROOT / "睡眠测试").rglob("*.jsonl")):
        read_jsonl(path)

    tracked_text = [
        ROOT / "README.md",
        ROOT / "sleep_lora" / "README.md",
        ROOT / "verify_sleep_install.py",
        ROOT / ".github" / "workflows" / "sleep-bundle-smoke.yml",
        *sorted((ROOT / "睡眠测试").rglob("*.md")),
        *sorted((ROOT / "睡眠测试").rglob("*.py")),
    ]
    stale = [str(path.relative_to(ROOT)) for path in tracked_text if "sleep_benchmark" in path.read_text(encoding="utf-8")]
    if stale:
        raise AssertionError(f"Old sleep_benchmark paths remain: {stale}")

    with (INSOMNIA / "results" / "manual_review_v4.csv").open(encoding="utf-8-sig") as handle:
        manual_rows = list(csv.DictReader(handle))
        assert manual_rows
        benchmark_ids = {row["id"] for row in read_jsonl(INSOMNIA / "benchmark" / "benchmark_v4.jsonl")}
        assert {row["id"] for row in manual_rows} <= benchmark_ids
    print("OK: Python syntax, JSON/JSONL, paths, and manual-review template")


def validate_gpu_entrypoints() -> None:
    run(PYTHON, str(ROOT / "sleep_chat.py"), "--help")
    run(PYTHON, str(COMPARISON / "scripts" / "run_benchmark.py"), "--help")
    run(PYTHON, str(COMPARISON / "scripts" / "run_colab_fp16_pair.py"), "--check-only")
    run(PYTHON, str(INSOMNIA / "scripts" / "run_colab_v4.py"), "--check-only")
    print("OK: local chat and both GPU test entrypoints")


def validate_reproduction(temp: Path) -> None:
    package = ROOT / "packages" / "chatglm3_colab_lora_package.zip"
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        assert {"data/train.json", "data/dev.json"} <= names
        for name in names:
            destination = (temp / "package" / name).resolve()
            if temp.resolve() not in destination.parents:
                raise AssertionError(f"Unsafe ZIP path: {name}")
        archive.extract("data/train.json", temp / "package")
        train = decode_json_or_jsonl(archive.read("data/train.json"))
        dev = decode_json_or_jsonl(archive.read("data/dev.json"))
        assert len(train) == 440 and len(dev) == 60

    train_path = temp / "package" / "data" / "train.json"
    generated = temp / "generated"
    generated.mkdir()

    base_csv = generated / "baseline.csv"
    base_json = generated / "baseline.json"
    sleep_csv = generated / "sleep.csv"
    sleep_json = generated / "sleep.json"
    run(PYTHON, str(COMPARISON / "scripts" / "score_benchmark_v2.py"), "--input", str(COMPARISON / "results" / "baseline_fp16_raw.jsonl"), "--item-csv", str(base_csv), "--summary", str(base_json))
    run(PYTHON, str(COMPARISON / "scripts" / "score_benchmark_v2.py"), "--input", str(COMPARISON / "results" / "sleep_lora_raw.jsonl"), "--item-csv", str(sleep_csv), "--summary", str(sleep_json))
    same_file(base_csv, COMPARISON / "scoring" / "baseline_fp16_item_scores_v2.csv")
    same_file(base_json, COMPARISON / "scoring" / "baseline_fp16_summary_v2.json")
    same_file(sleep_csv, COMPARISON / "scoring" / "sleep_lora_item_scores_v2.csv")
    same_file(sleep_json, COMPARISON / "scoring" / "sleep_lora_summary_v2.json")

    legacy_base_csv = generated / "baseline_v1.csv"
    legacy_base_json = generated / "baseline_v1.json"
    legacy_sleep_csv = generated / "sleep_v1.csv"
    legacy_sleep_json = generated / "sleep_v1.json"
    run(PYTHON, str(COMPARISON / "scripts" / "score_benchmark.py"), "--input", str(COMPARISON / "results" / "baseline_fp16_raw.jsonl"), "--item-csv", str(legacy_base_csv), "--summary", str(legacy_base_json))
    run(PYTHON, str(COMPARISON / "scripts" / "score_benchmark.py"), "--input", str(COMPARISON / "results" / "sleep_lora_raw.jsonl"), "--item-csv", str(legacy_sleep_csv), "--summary", str(legacy_sleep_json))
    same_file(legacy_base_csv, COMPARISON / "scoring" / "baseline_fp16_item_scores.csv")
    same_file(legacy_base_json, COMPARISON / "scoring" / "baseline_fp16_summary.json")
    same_file(legacy_sleep_csv, COMPARISON / "scoring" / "sleep_lora_item_scores.csv")
    same_file(legacy_sleep_json, COMPARISON / "scoring" / "sleep_lora_summary.json")

    comparison_out = generated / "comparison"
    run(PYTHON, str(COMPARISON / "scripts" / "make_comparison.py"), "--baseline", str(base_csv), "--sleep", str(sleep_csv), "--out-dir", str(comparison_out))
    for name in ["comparison_summary.csv", "comparison_summary.json", "category_summary.csv", "comparison_chart.svg"]:
        same_file(comparison_out / name, COMPARISON / "scoring" / name)
    with Image.open(comparison_out / "comparison_chart.png") as image:
        assert image.size == (1500, 940) and image.mode == "RGB"

    v4_csv = generated / "v4.csv"
    v4_json = generated / "v4.json"
    run(PYTHON, str(INSOMNIA / "scripts" / "score_v4.py"), "--input", str(INSOMNIA / "results" / "sleep_lora_raw_v4.jsonl"), "--item-csv", str(v4_csv), "--summary", str(v4_json))
    same_file(v4_csv, INSOMNIA / "results" / "sleep_lora_item_scores_v4.csv")
    same_file(v4_json, INSOMNIA / "results" / "sleep_lora_summary_v4.json")

    leak1 = generated / "leakage.json"
    leak4 = generated / "leakage_v4.json"
    run(PYTHON, str(COMPARISON / "scripts" / "check_leakage.py"), "--benchmark", str(COMPARISON / "benchmark" / "benchmark.jsonl"), "--train", str(train_path), "--report", str(leak1))
    run(PYTHON, str(INSOMNIA / "scripts" / "check_leakage_v4.py"), "--benchmark", str(INSOMNIA / "benchmark" / "benchmark_v4.jsonl"), "--source", str(train_path), "--source", str(COMPARISON / "benchmark" / "benchmark.jsonl"), "--report", str(leak4))
    same_file(leak1, COMPARISON / "benchmark" / "leakage_report.json")
    same_file(leak4, INSOMNIA / "benchmark" / "leakage_report_v4.json")

    audit = generated / "repository_audit.json"
    run(PYTHON, str(COMPARISON / "scripts" / "repository_audit.py"), "--output", str(audit))
    audit_data = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_data["adapter_available"] is True
    assert audit_data["adapter_sha256"] == "207588c4401877ce68a1ed8006ba7ec165ecfa8bcc18308b7db375e78d4ae3c5"
    same_file(audit, COMPARISON / "repository" / "repository_audit.json")
    print("OK: scores, summaries, charts, leakage reports, and repository audit reproduce")


def validate_execution_metadata() -> None:
    comparison_meta = json.loads((COMPARISON / "results" / "execution_metadata.json").read_text(encoding="utf-8"))
    insomnia_meta = json.loads((INSOMNIA / "results" / "execution_metadata_v4.json").read_text(encoding="utf-8"))
    expected_hash = "207588c4401877ce68a1ed8006ba7ec165ecfa8bcc18308b7db375e78d4ae3c5"
    assert comparison_meta["adapter_sha256"] == insomnia_meta["adapter_sha256"] == expected_hash
    assert comparison_meta["case_count"] == 14 and comparison_meta["turn_count"] == 18
    assert insomnia_meta["n_cases"] == 20 and insomnia_meta["n_turns"] == 24
    assert comparison_meta["hardware"] and insomnia_meta["hardware"]
    print("OK: recorded GPU execution metadata")


def main() -> None:
    run(PYTHON, str(ROOT / "verify_sleep_install.py"))
    validate_static_files()
    validate_gpu_entrypoints()
    validate_execution_metadata()
    with tempfile.TemporaryDirectory(prefix="sleep-checks-") as directory:
        validate_reproduction(Path(directory))
    print("ALL_SLEEP_CHECKS_OK")


if __name__ == "__main__":
    main()
