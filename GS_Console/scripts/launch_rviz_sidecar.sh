#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_CONTAINER="${1:-fast-livo2-compressed-cbd}"
IMAGE_TAG="${IMAGE_TAG:-gs-sdf-ros-tools:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gssdf-rviz}"
RVIZ_CONFIG="${RVIZ_CONFIG:-$ROOT_DIR/rviz/neural_mapping_live.rviz}"
DISPLAY_VALUE="${DISPLAY:-:0}"
XAUTH_FILE="${XAUTHORITY:-$HOME/.Xauthority}"

if [[ ! -S "/tmp/.X11-unix/X${DISPLAY_VALUE#:}" ]]; then
  echo "X11 socket for DISPLAY=$DISPLAY_VALUE is not available." >&2
  echo "If you are on a desktop session, export DISPLAY first, for example: export DISPLAY=:0" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  bash "$ROOT_DIR/scripts/build_ros_tools_image.sh" "$IMAGE_TAG"
fi

ROS_MASTER_HOST="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$SOURCE_CONTAINER")"
SOURCE_HOSTNAME="$(docker inspect -f '{{.Config.Hostname}}' "$SOURCE_CONTAINER")"
if [[ -z "$ROS_MASTER_HOST" ]]; then
  echo "Failed to resolve ROS master container IP from $SOURCE_CONTAINER" >&2
  exit 1
fi

XAUTH_ARGS=()
if [[ -f "$XAUTH_FILE" ]]; then
  XAUTH_ARGS=(-e XAUTHORITY=/tmp/.docker.xauth -v "$XAUTH_FILE:/tmp/.docker.xauth:ro")
fi

xhost +local:root >/dev/null 2>&1 || true

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --rm \
  --name "$CONTAINER_NAME" \
  --network bridge \
  -e DISPLAY="$DISPLAY_VALUE" \
  -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -e QT_X11_NO_MITSHM=1 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e MESA_LOADER_DRIVER_OVERRIDE=swrast \
  -e ROS_MASTER_URI="http://$ROS_MASTER_HOST:11311" \
  --add-host "$SOURCE_HOSTNAME:$ROS_MASTER_HOST" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$RVIZ_CONFIG:/workspace/neural_mapping_live.rviz:ro" \
  "${XAUTH_ARGS[@]}" \
  "$IMAGE_TAG" \
  bash -lc "mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root && export ROS_IP=\$(hostname -I | awk '{print \$1}'); source /opt/ros/noetic/setup.bash && rviz -d /workspace/neural_mapping_live.rviz"
