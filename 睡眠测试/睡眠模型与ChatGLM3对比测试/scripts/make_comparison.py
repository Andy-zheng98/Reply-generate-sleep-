#!/usr/bin/env python3
"""Build paired summaries, bootstrap intervals, tables, and a static comparison chart."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


QUALITY = ["relevance", "completeness", "domain_specificity", "empathy", "multi_turn_consistency", "safety"]
ALL_METRICS = QUALITY + ["hallucination_risk", "latency_s", "tokens_per_s"]
LABELS = {
    "relevance": "Relevance",
    "completeness": "Completeness",
    "domain_specificity": "Sleep specificity",
    "empathy": "Empathy",
    "multi_turn_consistency": "Multi-turn consistency",
    "safety": "Safety",
    "hallucination_risk": "Hallucination risk (lower is better)",
    "latency_s": "Latency per case (s)",
    "tokens_per_s": "Generation speed (tokens/s)",
}


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def render_chart(rows, png_path, svg_path):
    lookup = {row["metric"]: row for row in rows}
    width, height = 1500, 940
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = load_font(30, bold=True)
    font_panel = load_font(22, bold=True)
    font_label = load_font(18)
    font_small = load_font(15)
    blue, orange, grid, text = "#4C78A8", "#F58518", "#D9DEE7", "#20242B"
    draw.text((55, 30), "Sleep Dialogue Benchmark — FP16 paired run on Tesla T4", fill=text, font=font_title)
    draw.text((55, 72), "Same base model, prompts, seed, and generation settings", fill="#5B6470", font=font_label)

    # Quality panel: 1–5 rule scores.
    x0, y0, pw, ph = 55, 135, 900, 650
    draw.text((x0, y0), "Quality and safety (higher is better)", fill=text, font=font_panel)
    plot_left, plot_top, plot_right = x0 + 250, y0 + 65, x0 + pw - 30
    metrics = ["relevance", "completeness", "domain_specificity", "empathy", "safety"]
    row_height = 104
    for tick in range(6):
        x = plot_left + (plot_right - plot_left) * tick / 5
        draw.line((x, plot_top - 12, x, plot_top + row_height * len(metrics) - 20), fill=grid, width=1)
        draw.text((x - 5, plot_top + row_height * len(metrics) - 10), str(tick), fill="#5B6470", font=font_small)
    for index, metric in enumerate(metrics):
        y = plot_top + index * row_height
        draw.text((x0, y + 20), LABELS[metric], fill=text, font=font_label)
        base = lookup[metric]["baseline_mean"]
        tuned = lookup[metric]["sleep_lora_mean"]
        base_end = plot_left + (plot_right - plot_left) * base / 5
        tuned_end = plot_left + (plot_right - plot_left) * tuned / 5
        draw.rectangle((plot_left, y + 8, base_end, y + 36), fill=blue)
        draw.rectangle((plot_left, y + 44, tuned_end, y + 72), fill=orange)
        draw.text((base_end + 8, y + 10), f"{base:.2f}", fill=text, font=font_small)
        draw.text((tuned_end + 8, y + 46), f"{tuned:.2f}", fill=text, font=font_small)

    # Right-side small multiples.
    panels = [
        ("hallucination_risk", "Hallucination risk", "lower is better", 5.0),
        ("latency_s", "Latency per case", "seconds; lower is better", None),
        ("tokens_per_s", "Generation speed", "tokens/s; higher is better", None),
    ]
    rx, ry, rw, rh = 1010, 135, 435, 200
    for idx, (metric, title, subtitle, fixed_max) in enumerate(panels):
        py = ry + idx * 245
        draw.text((rx, py), title, fill=text, font=font_panel)
        draw.text((rx, py + 32), subtitle, fill="#5B6470", font=font_small)
        base = lookup[metric]["baseline_mean"]
        tuned = lookup[metric]["sleep_lora_mean"]
        ceiling = fixed_max or max(base, tuned) * 1.25
        bar_left, bar_right = rx, rx + rw - 75
        for j, (name, value, color) in enumerate([("ChatGLM3 FP16", base, blue), ("Sleep LoRA", tuned, orange)]):
            yy = py + 72 + j * 58
            draw.text((bar_left, yy), name, fill=text, font=font_small)
            by = yy + 22
            draw.rectangle((bar_left, by, bar_right, by + 20), fill="#EEF1F5")
            end = bar_left + (bar_right - bar_left) * value / ceiling
            draw.rectangle((bar_left, by, end, by + 20), fill=color)
            draw.text((bar_right + 10, by + 1), f"{value:.2f}", fill=text, font=font_small)

    legend_y = 865
    draw.rectangle((55, legend_y, 78, legend_y + 23), fill=blue)
    draw.text((90, legend_y + 1), "ChatGLM3-6B FP16", fill=text, font=font_label)
    draw.rectangle((320, legend_y, 343, legend_y + 23), fill=orange)
    draw.text((355, legend_y + 1), "Sleep LoRA checkpoint-160", fill=text, font=font_label)
    draw.text((1010, legend_y + 1), "Rule scorer v2 · n=14 cases", fill="#5B6470", font=font_label)
    image.save(png_path)

    # A compact editable SVG counterpart.
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#20242B}.title{font-size:30px;font-weight:700}.panel{font-size:22px;font-weight:700}.label{font-size:17px}.small{font-size:14px;fill:#5B6470}</style>',
        '<text x="55" y="55" class="title">Sleep Dialogue Benchmark — FP16 paired run on Tesla T4</text>',
        '<text x="55" y="88" class="label" fill="#5B6470">Same base model, prompts, seed, and generation settings</text>',
        '<text x="55" y="155" class="panel">Quality and safety (higher is better)</text>',
    ]
    plot_left, plot_top, plot_right = 305, 200, 925
    for tick in range(6):
        x = plot_left + (plot_right - plot_left) * tick / 5
        parts.append(f'<line x1="{x:.1f}" y1="180" x2="{x:.1f}" y2="700" stroke="#D9DEE7"/>')
        parts.append(f'<text x="{x:.1f}" y="730" class="small" text-anchor="middle">{tick}</text>')
    for index, metric in enumerate(metrics):
        y = plot_top + index * row_height
        base = lookup[metric]["baseline_mean"]
        tuned = lookup[metric]["sleep_lora_mean"]
        bw = (plot_right - plot_left) * base / 5
        tw = (plot_right - plot_left) * tuned / 5
        parts += [
            f'<text x="55" y="{y + 26}" class="label">{LABELS[metric]}</text>',
            f'<rect x="{plot_left}" y="{y}" width="{bw:.1f}" height="28" fill="{blue}"/>',
            f'<rect x="{plot_left}" y="{y + 36}" width="{tw:.1f}" height="28" fill="{orange}"/>',
            f'<text x="{plot_left + bw + 8:.1f}" y="{y + 20}" class="small">{base:.2f}</text>',
            f'<text x="{plot_left + tw + 8:.1f}" y="{y + 56}" class="small">{tuned:.2f}</text>',
        ]
    for idx, (metric, title, subtitle, fixed_max) in enumerate(panels):
        py = 155 + idx * 245
        base = lookup[metric]["baseline_mean"]
        tuned = lookup[metric]["sleep_lora_mean"]
        ceiling = fixed_max or max(base, tuned) * 1.25
        parts += [f'<text x="1010" y="{py}" class="panel">{title}</text>', f'<text x="1010" y="{py + 25}" class="small">{subtitle}</text>']
        for j, (name, value, color) in enumerate([("ChatGLM3 FP16", base, blue), ("Sleep LoRA", tuned, orange)]):
            yy = py + 55 + j * 60
            bar_w = 350 * value / ceiling
            parts += [
                f'<text x="1010" y="{yy}" class="small">{name}</text>',
                f'<rect x="1010" y="{yy + 10}" width="350" height="20" fill="#EEF1F5"/>',
                f'<rect x="1010" y="{yy + 10}" width="{bar_w:.1f}" height="20" fill="{color}"/>',
                f'<text x="1370" y="{yy + 26}" class="small">{value:.2f}</text>',
            ]
    parts += [
        f'<rect x="55" y="865" width="23" height="23" fill="{blue}"/><text x="90" y="883" class="label">ChatGLM3-6B FP16</text>',
        f'<rect x="320" y="865" width="23" height="23" fill="{orange}"/><text x="355" y="883" class="label">Sleep LoRA checkpoint-160</text>',
        '<text x="1010" y="883" class="small">Rule scorer v2 · n=14 cases</text>',
        '</svg>',
    ]
    Path(svg_path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def read_scores(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def numeric(row, metric):
    value = row.get(metric, "")
    return None if value in (None, "") else float(value)


def percentile(values, quantile):
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def bootstrap_ci(differences, samples=10000, seed=42):
    rng = random.Random(seed)
    n = len(differences)
    boot = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return percentile(boot, 0.025), percentile(boot, 0.975)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--sleep", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline = read_scores(args.baseline)
    sleep = read_scores(args.sleep)
    ids = sorted(set(baseline) & set(sleep))

    rows = []
    for metric in ALL_METRICS:
        pairs = [(numeric(baseline[case_id], metric), numeric(sleep[case_id], metric)) for case_id in ids]
        pairs = [(base, tuned) for base, tuned in pairs if base is not None and tuned is not None]
        base_mean = sum(base for base, _ in pairs) / len(pairs)
        sleep_mean = sum(tuned for _, tuned in pairs) / len(pairs)
        differences = [tuned - base for base, tuned in pairs]
        low, high = bootstrap_ci(differences)
        rows.append({
            "metric": metric,
            "label": LABELS[metric],
            "n_pairs": len(pairs),
            "baseline_mean": round(base_mean, 4),
            "sleep_lora_mean": round(sleep_mean, 4),
            "delta_sleep_minus_baseline": round(sleep_mean - base_mean, 4),
            "paired_bootstrap_95ci_low": round(low, 4),
            "paired_bootstrap_95ci_high": round(high, 4),
        })

    with (out / "comparison_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    category_rows = []
    grouped = defaultdict(list)
    for case_id in ids:
        grouped[baseline[case_id]["category"]].append(case_id)
    for category, case_ids in sorted(grouped.items()):
        for model_name, data in [("ChatGLM3-6B FP16", baseline), ("Sleep LoRA", sleep)]:
            category_rows.append({
                "category": category,
                "model": model_name,
                "n": len(case_ids),
                "relevance": round(sum(numeric(data[i], "relevance") for i in case_ids) / len(case_ids), 3),
                "completeness": round(sum(numeric(data[i], "completeness") for i in case_ids) / len(case_ids), 3),
                "safety": round(sum(numeric(data[i], "safety") for i in case_ids) / len(case_ids), 3),
                "hallucination_risk": round(sum(numeric(data[i], "hallucination_risk") for i in case_ids) / len(case_ids), 3),
            })
    with (out / "category_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=category_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(category_rows)

    render_chart(rows, out / "comparison_chart.png", out / "comparison_chart.svg")

    machine = {
        "paired_case_ids": ids,
        "metric_rows": rows,
        "bootstrap": {"samples": 10000, "seed": 42, "unit": "case", "method": "paired resampling with replacement"},
        "interpretation": {
            "higher_is_better": QUALITY + ["tokens_per_s"],
            "lower_is_better": ["hallucination_risk", "latency_s"],
        },
    }
    (out / "comparison_summary.json").write_text(json.dumps(machine, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
