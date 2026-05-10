#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_UI_DIR="$ROOT_DIR/examples/web-ui"

cd "$WEB_UI_DIR"
npm install
