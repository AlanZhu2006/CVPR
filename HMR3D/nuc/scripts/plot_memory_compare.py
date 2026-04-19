from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot side-by-side comparison for two HMR3D runs as SVG.")
    parser.add_argument("--left", required=True, help="First run output directory.")
    parser.add_argument("--right", required=True, help="Second run output directory.")
    parser.add_argument("--output-prefix", default="", help="Optional output prefix path without extension.")
    return parser.parse_args()


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))
    return summary, events


def svg_bar(x: float, base_y: float, width: float, height: float, color: str, label: str) -> str:
    y = base_y - height
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"><title>{label}</title></rect>'


def event_count(events: list[dict], event_type: str) -> int:
    return sum(1 for event in events if event["event_type"] == event_type)


def main() -> None:
    args = parse_args()
    left_dir = Path(args.left).expanduser().resolve()
    right_dir = Path(args.right).expanduser().resolve()
    left_summary, left_events = load_run(left_dir)
    right_summary, right_events = load_run(right_dir)
    output_prefix = Path(args.output_prefix).expanduser().resolve() if args.output_prefix else right_dir.parent / f"{left_dir.name}_vs_{right_dir.name}"

    metrics = {
        "keyframes": [left_summary.get("keyframe_count", 0), right_summary.get("keyframe_count", 0)],
        "archives": [left_summary.get("stats", {}).get("archives", 0), right_summary.get("stats", {}).get("archives", 0)],
        "recoveries": [left_summary.get("stats", {}).get("recoveries", 0), right_summary.get("stats", {}).get("recoveries", 0)],
        "retrieve_hits": [left_summary.get("stats", {}).get("retrieve_hits", 0), right_summary.get("stats", {}).get("retrieve_hits", 0)],
        "bank_size": [len(left_summary.get("bank", [])), len(right_summary.get("bank", []))],
        "shadow_buffered": [event_count(left_events, "shadow_buffered"), event_count(right_events, "shadow_buffered")],
        "recover_rejected": [event_count(left_events, "recover_rejected"), event_count(right_events, "recover_rejected")],
    }
    left_counts = Counter(event["event_type"] for event in left_events)
    right_counts = Counter(event["event_type"] for event in right_events)

    width = 1400
    height = 780
    left = 80
    base_y = 330
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text { font-family: monospace; font-size: 14px; } .small { font-size: 12px; }</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
        f'<text x="{left}" y="30" font-size="22">{left_dir.name} vs {right_dir.name}</text>',
    ]

    max_metric = max(max(values) for values in metrics.values()) if metrics else 1
    metric_x = left
    group_gap = 36
    bar_w = 28
    colors = ["#4c78a8", "#f58518"]
    for name, values in metrics.items():
        h0 = 220 * values[0] / max_metric if max_metric else 0
        h1 = 220 * values[1] / max_metric if max_metric else 0
        svg_parts.append(svg_bar(metric_x, base_y, bar_w, h0, colors[0], f"{name} left={values[0]}"))
        svg_parts.append(svg_bar(metric_x + bar_w + 4, base_y, bar_w, h1, colors[1], f"{name} right={values[1]}"))
        svg_parts.append(f'<text x="{metric_x - 6}" y="{base_y + 22}" class="small">{name}</text>')
        metric_x += bar_w * 2 + group_gap

    svg_parts.append(f'<text x="{left}" y="380">{left_dir.name}</text>')
    y = 405
    for key, value in sorted(left_counts.items()):
        svg_parts.append(f'<text x="{left}" y="{y}" class="small">{key}: {value}</text>')
        y += 18

    right_x = 760
    svg_parts.append(f'<text x="{right_x}" y="380">{right_dir.name}</text>')
    y = 405
    for key, value in sorted(right_counts.items()):
        svg_parts.append(f'<text x="{right_x}" y="{y}" class="small">{key}: {value}</text>')
        y += 18

    svg_parts.append(f'<rect x="{left}" y="70" width="14" height="14" fill="{colors[0]}"/>')
    svg_parts.append(f'<text x="{left + 20}" y="82" class="small">{left_dir.name}</text>')
    svg_parts.append(f'<rect x="{left + 240}" y="70" width="14" height="14" fill="{colors[1]}"/>')
    svg_parts.append(f'<text x="{left + 260}" y="82" class="small">{right_dir.name}</text>')
    svg_parts.append("</svg>")

    svg_path = output_prefix.with_suffix(".svg")
    svg_path.write_text("\n".join(svg_parts), encoding="utf-8")

    md_lines = [
        f"# {left_dir.name} vs {right_dir.name}",
        "",
        "| metric | left | right |",
        "|---|---:|---:|",
    ]
    for key, values in metrics.items():
        md_lines.append(f"| {key} | {values[0]} | {values[1]} |")
    md_path = output_prefix.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"svg": str(svg_path), "md": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
