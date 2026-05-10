#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PORT="${1:-${ISAAC_GAUSSIAN_ONLINE_PORT:-8890}}"
RENDERER_URL="${RENDERER_URL:-${ISAAC_GAUSSIAN_RENDERER_URL:-}}"
MAPPER_URL="${MAPPER_URL:-${ISAAC_GAUSSIAN_MAPPER_URL:-}}"
LIVE_RENDER_POSE_MODE="${LIVE_RENDER_POSE_MODE:-${ISAAC_GAUSSIAN_LIVE_RENDER_POSE_MODE:-}}"
MANUAL_RENDER_POSE_HOLD_SEC="${MANUAL_RENDER_POSE_HOLD_SEC:-${ISAAC_GAUSSIAN_MANUAL_RENDER_POSE_HOLD_SEC:-}}"
MAX_RENDER_POSE_DISTANCE_FROM_ORIGIN="${MAX_RENDER_POSE_DISTANCE_FROM_ORIGIN:-${ISAAC_GAUSSIAN_MAX_RENDER_POSE_DISTANCE_FROM_ORIGIN:-}}"

cmd=(python3 "$ROOT_DIR/scripts/isaac_gaussian_online_bridge_server.py" --port "$PORT")
if [[ -n "$RENDERER_URL" ]]; then
  cmd+=(--renderer-url "$RENDERER_URL")
fi
if [[ -n "$MAPPER_URL" ]]; then
  cmd+=(--mapper-url "$MAPPER_URL")
fi
if [[ -n "$LIVE_RENDER_POSE_MODE" ]]; then
  cmd+=(--live-render-pose-mode "$LIVE_RENDER_POSE_MODE")
fi
if [[ -n "$MANUAL_RENDER_POSE_HOLD_SEC" ]]; then
  cmd+=(--manual-render-pose-hold-sec "$MANUAL_RENDER_POSE_HOLD_SEC")
fi
if [[ -n "$MAX_RENDER_POSE_DISTANCE_FROM_ORIGIN" ]]; then
  cmd+=(--max-render-pose-distance-from-origin "$MAX_RENDER_POSE_DISTANCE_FROM_ORIGIN")
fi

echo "Isaac Gaussian online bridge: http://localhost:$PORT"
if [[ -n "$RENDERER_URL" ]]; then
  echo "Forward renderer URL: $RENDERER_URL"
fi
if [[ -n "$MAPPER_URL" ]]; then
  echo "Forward mapper URL: $MAPPER_URL"
fi
if [[ -n "$LIVE_RENDER_POSE_MODE" ]]; then
  echo "Live renderer pose mode: $LIVE_RENDER_POSE_MODE"
fi
if [[ -n "$MAX_RENDER_POSE_DISTANCE_FROM_ORIGIN" ]]; then
  echo "Max renderer pose distance from origin: $MAX_RENDER_POSE_DISTANCE_FROM_ORIGIN"
fi

exec "${cmd[@]}"
