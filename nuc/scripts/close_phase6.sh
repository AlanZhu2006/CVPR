#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-$ROOT_DIR/nuc_output/phase6_closure_$TIMESTAMP}"

# Override default comparison groups by passing ON run names after output dir.
# Example:
#   bash nuc/scripts/close_phase6.sh nuc_output/phase6_closure_custom recover_on westlake_recover_on
DEFAULT_ON_RUNS=(
  "recover_on"
  "stereo_taylor_recover_on"
  "westlake_recover_on"
)

ON_RUNS=("${DEFAULT_ON_RUNS[@]}")
if [ "$#" -ge 2 ]; then
  ON_RUNS=("${@:2}")
fi

mkdir -p "$OUT_DIR"

SUMMARY_TABLE="$OUT_DIR/phase6_summary.csv"
REPORT_MD="$OUT_DIR/phase6_report.md"

echo "on_run,off_run,frames_on,frames_off,keyframes_on,keyframes_off,archives_on,archives_off,retrieve_hits_on,retrieve_hits_off,recoveries_on,recoveries_off,recover_enabled_on,recover_enabled_off,status,notes" > "$SUMMARY_TABLE"

for on_run in "${ON_RUNS[@]}"; do
  if [[ "$on_run" != *_on ]]; then
    echo "Skip $on_run: expected name ending with _on" >&2
    continue
  fi

  off_run="${on_run%_on}_off"
  on_summary="$ROOT_DIR/nuc_output/$on_run/summary.json"
  off_summary="$ROOT_DIR/nuc_output/$off_run/summary.json"

  if [ ! -f "$on_summary" ] || [ ! -f "$off_summary" ]; then
    echo "Skip $on_run vs $off_run: missing summary.json" >&2
    continue
  fi

  cmp_csv="$OUT_DIR/${on_run}__vs__${off_run}.csv"
  "$PYTHON_BIN" "$ROOT_DIR/nuc/tools/compare_runs.py" --left "$on_summary" --right "$off_summary" > "$cmp_csv"

  row="$($PYTHON_BIN - "$on_summary" "$off_summary" <<'PY'
import json
import sys

on_path, off_path = sys.argv[1], sys.argv[2]
with open(on_path, "r", encoding="utf-8") as f:
    on = json.load(f)
with open(off_path, "r", encoding="utf-8") as f:
    off = json.load(f)

def get_runtime(d, key, default=0):
    return d.get("runtime", {}).get(key, default)

def get_stats(d, key, default=0):
    return d.get("stats", {}).get(key, default)

frames_on = int(get_runtime(on, "frames", 0) or 0)
frames_off = int(get_runtime(off, "frames", 0) or 0)
keyframes_on = int(get_runtime(on, "keyframes", 0) or 0)
keyframes_off = int(get_runtime(off, "keyframes", 0) or 0)
archives_on = int(get_stats(on, "archives", 0) or 0)
archives_off = int(get_stats(off, "archives", 0) or 0)
retrieve_hits_on = int(get_stats(on, "retrieve_hits", 0) or 0)
retrieve_hits_off = int(get_stats(off, "retrieve_hits", 0) or 0)
recoveries_on = int(get_stats(on, "recoveries", 0) or 0)
recoveries_off = int(get_stats(off, "recoveries", 0) or 0)
recover_enabled_on = get_runtime(on, "recover_enabled", None)
recover_enabled_off = get_runtime(off, "recover_enabled", None)

checks = [
    (frames_on > 0 and frames_off > 0, "frames>0"),
    (keyframes_on > 0 and keyframes_off > 0, "keyframes>0"),
    (archives_on > 0 and archives_off > 0, "archives>0"),
    (retrieve_hits_on > 0 and retrieve_hits_off > 0, "retrieve_hits>0"),
    (recover_enabled_on is True and recover_enabled_off is False, "recover_enabled_on/off"),
    (recoveries_on > 0, "recoveries_on>0"),
    (recoveries_off == 0, "recoveries_off==0"),
]

failed = [name for ok, name in checks if not ok]
status = "PASS" if not failed else "WARN"
notes = "ok" if not failed else ";".join(failed)

values = [
    frames_on,
    frames_off,
    keyframes_on,
    keyframes_off,
    archives_on,
    archives_off,
    retrieve_hits_on,
    retrieve_hits_off,
    recoveries_on,
    recoveries_off,
    str(recover_enabled_on).lower(),
    str(recover_enabled_off).lower(),
    status,
    notes,
]
print(",".join(map(str, values)))
PY
)"

  echo "$on_run,$off_run,$row" >> "$SUMMARY_TABLE"
done

"$PYTHON_BIN" - "$SUMMARY_TABLE" "$REPORT_MD" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
report_md = Path(sys.argv[2])

rows = []
with summary_csv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

pass_count = sum(1 for r in rows if r["status"] == "PASS")
total = len(rows)

lines = []
lines.append("# Phase 6 Closure Report")
lines.append("")
lines.append(f"- Total pairs: {total}")
lines.append(f"- PASS pairs: {pass_count}")
lines.append(f"- WARN pairs: {total - pass_count}")
lines.append("")
lines.append("## Pair Summary")
lines.append("")
lines.append("| on_run | off_run | retrieve_hits_on | retrieve_hits_off | recoveries_on | recoveries_off | status | notes |")
lines.append("|---|---:|---:|---:|---:|---:|---|---|")
for r in rows:
    lines.append(
        "| {on_run} | {off_run} | {retrieve_hits_on} | {retrieve_hits_off} | {recoveries_on} | {recoveries_off} | {status} | {notes} |".format(**r)
    )

lines.append("")
lines.append("## Artifacts")
lines.append("")
lines.append("- phase6_summary.csv")
lines.append("- <on_run>__vs__<off_run>.csv")

report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "Phase 6 closure artifacts written to: $OUT_DIR"
echo "- $SUMMARY_TABLE"
echo "- $REPORT_MD"