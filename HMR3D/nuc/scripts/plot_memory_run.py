from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one HMR3D replay run as SVG.")
    parser.add_argument("--run-dir", required=True, help="Output directory containing events.json and summary.json.")
    parser.add_argument("--output-prefix", default="", help="Optional output prefix path without extension.")
    return parser.parse_args()


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))
    return summary, events


def svg_circle(x: float, y: float, color: str, label: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>{label}</title></circle>'


def svg_bar(x: float, base_y: float, width: float, height: float, color: str, label: str) -> str:
    y = base_y - height
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"><title>{label}</title></rect>'


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    summary, events = load_run(run_dir)
    output_prefix = Path(args.output_prefix).expanduser().resolve() if args.output_prefix else run_dir / "timeline"

    event_counts = Counter(event["event_type"] for event in events)
    keyframes = summary.get("keyframe_count", 0)
    bank_size = len(summary.get("bank", []))

    width = 1200
    height = 760
    left = 80
    top = 70
    timeline_w = 1050
    metric_base_y = 690
    max_frame = max((event["frame_idx"] for event in events), default=1)
    scale_x = timeline_w / max(1, max_frame)

    timeline_rows = {
        "recover_rejected": (300, "#d62728"),
        "recovered": (250, "#2ca02c"),
        "shadow_buffered": (200, "#ff7f0e"),
        "archived": (150, "#1f77b4"),
        "active_started": (100, "#9467bd"),
    }

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text { font-family: monospace; font-size: 14px; } .small { font-size: 12px; }</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
        f'<text x="{left}" y="32" font-size="22">{run_dir.name}</text>',
        f'<text x="{left}" y="52" class="small">keyframes={keyframes} archives={summary.get("stats", {}).get("archives", 0)} recoveries={summary.get("stats", {}).get("recoveries", 0)} retrieve_hits={summary.get("stats", {}).get("retrieve_hits", 0)} bank={bank_size}</text>',
        f'<line x1="{left}" y1="330" x2="{left + timeline_w}" y2="330" stroke="#cccccc"/>',
    ]

    for name, (y, color) in timeline_rows.items():
        svg_parts.append(f'<text x="10" y="{y + 4}" class="small">{name}</text>')
        svg_parts.append(f'<line x1="{left}" y1="{y}" x2="{left + timeline_w}" y2="{y}" stroke="#e0e0e0"/>')

    for event in events:
        if event["event_type"] not in timeline_rows:
            continue
        y, color = timeline_rows[event["event_type"]]
        x = left + event["frame_idx"] * scale_x
        label = f'{event["event_type"]} frame={event["frame_idx"]}'
        svg_parts.append(svg_circle(x, y, color, label))

    metric_names = ["keyframes", "archives", "recoveries", "retrieve_hits", "bank_size"]
    metric_values = [
        keyframes,
        summary.get("stats", {}).get("archives", 0),
        summary.get("stats", {}).get("recoveries", 0),
        summary.get("stats", {}).get("retrieve_hits", 0),
        bank_size,
    ]
    max_metric = max(metric_values) if metric_values else 1
    bar_w = 80
    gap = 50
    metric_x = left
    for name, value in zip(metric_names, metric_values):
        bar_h = 180 * value / max_metric if max_metric else 0
        svg_parts.append(svg_bar(metric_x, metric_base_y, bar_w, bar_h, "#4c78a8", f"{name}={value}"))
        svg_parts.append(f'<text x="{metric_x}" y="{metric_base_y + 20}" class="small">{name}</text>')
        svg_parts.append(f'<text x="{metric_x}" y="{metric_base_y - bar_h - 8}" class="small">{value}</text>')
        metric_x += bar_w + gap

    legend_x = 760
    legend_y = 420
    svg_parts.append(f'<text x="{legend_x}" y="{legend_y - 20}">Event Counts</text>')
    for idx, (name, count) in enumerate(sorted(event_counts.items())):
        svg_parts.append(f'<text x="{legend_x}" y="{legend_y + idx * 18}" class="small">{name}: {count}</text>')

    svg_parts.append("</svg>")

    svg_path = output_prefix.with_suffix(".svg")
    svg_path.write_text("\n".join(svg_parts), encoding="utf-8")

    md_lines = [
        f"# {run_dir.name}",
        "",
        f"- keyframes: {keyframes}",
        f"- archives: {summary.get('stats', {}).get('archives', 0)}",
        f"- recoveries: {summary.get('stats', {}).get('recoveries', 0)}",
        f"- retrieve_hits: {summary.get('stats', {}).get('retrieve_hits', 0)}",
        f"- bank_size: {bank_size}",
        "",
        "## Event Counts",
        "",
    ]
    for key, value in sorted(event_counts.items()):
        md_lines.append(f"- {key}: {value}")

    md_path = output_prefix.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"svg": str(svg_path), "md": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

