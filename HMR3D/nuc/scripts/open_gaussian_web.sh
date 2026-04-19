#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <viewer-html-or-output-dir> [port]"
  exit 1
fi

TARGET="$1"
PORT="${2:-9090}"

if [[ -d "$TARGET" ]]; then
  if [[ -f "$TARGET/kitti06_v4_gaussian_live_800_viewer.html" ]]; then
    HTML_FILE="$TARGET/kitti06_v4_gaussian_live_800_viewer.html"
  else
    HTML_FILE="$(find "$TARGET" -maxdepth 1 -name '*_viewer.html' | head -n 1)"
  fi
else
  HTML_FILE="$TARGET"
fi

if [[ -z "${HTML_FILE:-}" || ! -f "$HTML_FILE" ]]; then
  echo "viewer html not found"
  exit 1
fi

HTML_DIR="$(cd "$(dirname "$HTML_FILE")" && pwd)"
HTML_NAME="$(basename "$HTML_FILE")"

echo "Serving: $HTML_FILE"
echo "Open in browser:"
echo "  http://127.0.0.1:${PORT}/${HTML_NAME}"
echo
cd "$HTML_DIR"
exec python3 -m http.server "$PORT"
