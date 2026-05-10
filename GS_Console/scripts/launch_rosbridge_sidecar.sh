#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_CONTAINER="${1:-fast-livo2-compressed-cbd}"
HOST_PORT="${2:-9090}"
IMAGE_TAG="${IMAGE_TAG:-gs-sdf-ros-tools:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gssdf-rosbridge}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  bash "$ROOT_DIR/scripts/build_ros_tools_image.sh" "$IMAGE_TAG"
fi

ROS_MASTER_HOST=""
SOURCE_HOSTNAME=""
for ((i=0; i<WAIT_SECONDS; i++)); do
  ROS_MASTER_HOST="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$SOURCE_CONTAINER" 2>/dev/null || true)"
  SOURCE_HOSTNAME="$(docker inspect -f '{{.Config.Hostname}}' "$SOURCE_CONTAINER" 2>/dev/null || true)"
  if [[ -n "$ROS_MASTER_HOST" && "$ROS_MASTER_HOST" != "<no value>" ]]; then
    break
  fi
  sleep 1
done
if [[ -z "$ROS_MASTER_HOST" || "$ROS_MASTER_HOST" == "<no value>" ]]; then
  echo "Failed to resolve ROS master container IP from $SOURCE_CONTAINER" >&2
  exit 1
fi

READY=0
for ((i=0; i<WAIT_SECONDS; i++)); do
  if docker exec "$SOURCE_CONTAINER" bash -lc "source /opt/ros/noetic/setup.bash && rosnode list >/dev/null 2>&1" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "ROS master in $SOURCE_CONTAINER did not become ready within ${WAIT_SECONDS}s" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --network bridge \
  -p "$HOST_PORT:9090" \
  -e ROS_MASTER_URI="http://$ROS_MASTER_HOST:11311" \
  --add-host "$SOURCE_CONTAINER:$ROS_MASTER_HOST" \
  --add-host "$SOURCE_HOSTNAME:$ROS_MASTER_HOST" \
  "$IMAGE_TAG" \
  bash -lc "export ROS_IP=\$(hostname -I | awk '{print \$1}'); source /opt/ros/noetic/setup.bash && roslaunch rosbridge_server rosbridge_websocket.launch address:=0.0.0.0 port:=9090"

echo "rosbridge ready on ws://localhost:$HOST_PORT"
echo "ROS_MASTER_URI=http://$ROS_MASTER_HOST:11311"
