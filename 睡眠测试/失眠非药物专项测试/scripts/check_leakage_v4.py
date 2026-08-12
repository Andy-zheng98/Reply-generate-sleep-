#!/usr/bin/env python3
"""Character n-gram overlap check against one or more JSON/JSONL corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TEXT_KEYS = {"content", "prompt", "query", "instruction", "input", "turns"}


def norm(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def grams(text: str, n: int = 5) -> set[str]:
    value = norm(text)
    return {value[i : i + n] for i in range(max(0, len(value) - n + 1))}


def similarity(left: str, right: str) -> float:
    a, b = grams(left), grams(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def collect(value) -> list[str]:
    output: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TEXT_KEYS and isinstance(item, str):
                output.append(item)
            elif key == "turns" and isinstance(item, list):
                output.extend(text for text in item if isinstance(text, str))
            output.extend(collect(item))
    elif isinstance(value, list):
        for item in value:
            output.extend(collect(item))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    benchmark = load(benchmark_path)
    sources = []
    corpus: list[tuple[str, str]] = []
    for source_text in args.source:
        path = Path(source_text)
        strings = collect(load(path))
        source_label = path.name
        sources.append({"source": source_label, "sha256": sha256(path), "text_strings": len(strings)})
        corpus.extend((source_label, text) for text in strings)

    findings = []
    for case in benchmark:
        for turn_index, turn in enumerate(case["turns"], 1):
            best = max(
                ((similarity(turn, text), source, text) for source, text in corpus),
                default=(0.0, "", ""),
            )
            findings.append(
                {
                    "id": case["id"],
                    "turn": turn_index,
                    "max_5gram_jaccard": round(best[0], 4),
                    "nearest_source": best[1],
                    "nearest_text": best[2],
                }
            )

    report = {
        "method": "normalized Chinese character 5-gram Jaccard",
        "threshold": args.threshold,
        "benchmark_sha256": sha256(benchmark_path),
        "benchmark_turns": len(findings),
        "sources": sources,
        "flagged": [item for item in findings if item["max_5gram_jaccard"] >= args.threshold],
        "maximum_similarity": max(item["max_5gram_jaccard"] for item in findings),
        "all": findings,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("benchmark_turns", "threshold", "maximum_similarity", "flagged")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
