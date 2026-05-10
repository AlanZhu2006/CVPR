#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${1:-gs-sdf-ros-tools:latest}"

docker build \
  -f "$ROOT_DIR/docker/ros-tools/Dockerfile" \
  -t "$IMAGE_TAG" \
  "$ROOT_DIR"
