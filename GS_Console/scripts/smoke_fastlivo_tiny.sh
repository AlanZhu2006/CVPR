#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TINY_BAG="/home/chatsign/sgs-slam/tmp_fastlivo_debug/fastlivo_tiny.bag"
CONFIG="$ROOT_DIR/config/fastlivo_tiny_smoke.yaml"
RUN_NAME="${1:-gs-sdf-smoke-$(date +%Y%m%d-%H%M%S)}"

exec "$ROOT_DIR/scripts/train_gssdf.sh" "$TINY_BAG" "$CONFIG" "$RUN_NAME"
