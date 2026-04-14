#!/usr/bin/env python3
"""Print markdown + TSV tables from run_relpose_memory_sweep.py leaderboard.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("leaderboard", type=Path, help="Path to leaderboard.json")
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Write markdown table here (default: <leaderboard_dir>/METRICS_TABLE.md)",
    )
    args = parser.parse_args()
    data = json.loads(args.leaderboard.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    baseline_name = data.get("baseline_trial")

    cols = [
        "trial_name",
        "mode",
        "avg_ate",
        "avg_rpe_trans",
        "avg_rpe_rot",
        "avg_fps",
        "archive_count",
        "retrieval_attempts",
        "retrieval_successes",
        "delta_ate_vs_baseline",
        "delta_rpe_trans_vs_baseline",
        "delta_rpe_rot_vs_baseline",
    ]

    def fmt_row(r: dict) -> list[str]:
        out = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                out.append(f"{v:.6f}" if v != int(v) else str(int(v)))
            else:
                out.append(str(v))
        return out

    lines_md = [
        "# Relpose leaderboard metrics",
        "",
        f"- **dataset**: `{data.get('dataset')}`",
        f"- **weights**: `{data.get('weights')}`",
        f"- **size**: {data.get('size')}",
        f"- **device**: {data.get('device')}",
        f"- **seq_list**: {data.get('seq_list')}",
        f"- **baseline_trial**: `{baseline_name}`",
        "",
        "Lower ATE / RPE is better. Deltas are (trial − baseline).",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines_md.append("| " + " | ".join(fmt_row(r)) + " |")

    lines_md.append("")
    text_md = "\n".join(lines_md)

    tsv_header = "\t".join(cols)
    tsv_lines = [tsv_header]
    for r in rows:
        tsv_lines.append("\t".join(fmt_row(r)))
    text_tsv = "\n".join(tsv_lines) + "\n"

    out_md = args.out_md or (args.leaderboard.parent / "METRICS_TABLE.md")
    out_md.write_text(text_md, encoding="utf-8")
    out_tsv = args.leaderboard.parent / "METRICS_TABLE.tsv"
    out_tsv.write_text(text_tsv, encoding="utf-8")

    print(text_md)
    print(f"\nWrote {out_md}", file=sys.stderr)
    print(f"Wrote {out_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
