#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE_CONTAINER="${1:-gssdf-view-quality}"
PORT="${2:-8876}"
CONTAINER_NAME="${3:-gssdf-hifi-bridge}"
IMAGE="${GS_SDF_IMAGE:-gs_sdf_img:latest}"
MJPEG_MAX_FPS="${MJPEG_MAX_FPS:-45}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
PREVIEW_JPEG_QUALITY="${PREVIEW_JPEG_QUALITY:-84}"
PREVIEW_SCALE="${PREVIEW_SCALE:-0.5}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

ROS_MASTER_HOST="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$SOURCE_CONTAINER")"
SOURCE_HOSTNAME="$(docker inspect -f '{{.Config.Hostname}}' "$SOURCE_CONTAINER")"
if [[ -z "$ROS_MASTER_HOST" ]]; then
  echo "Failed to resolve ROS master container IP from $SOURCE_CONTAINER" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# The saved-view RViz helper publishes a fixed /rviz/current_camera_pose at 10 Hz.
# Stop it before enabling the interactive HiFi bridge or it will fight the web camera.
docker rm -f gssdf-view-camera-pose >/dev/null 2>&1 || true

echo "Source container: $SOURCE_CONTAINER"
echo "ROS master:       $ROS_MASTER_HOST"
echo "Bridge port:      $PORT"
echo "MJPEG max fps:    $MJPEG_MAX_FPS"
echo "Preview scale:    $PREVIEW_SCALE"

docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --network bridge \
  -p "${PORT}:${PORT}" \
  -e ROS_MASTER_URI="http://$ROS_MASTER_HOST:11311" \
  --add-host "$SOURCE_HOSTNAME:$ROS_MASTER_HOST" \
  -v "$ROOT_DIR:$ROOT_DIR" \
  "$IMAGE" \
  bash -lc "export ROS_IP=\$(hostname -I | awk '{print \$1}'); \
    source /opt/ros/noetic/setup.bash && \
    python3 \"$ROOT_DIR/scripts/gssdf_hifi_bridge_server.py\" \
      --port \"$PORT\" \
      --source-container \"$SOURCE_CONTAINER\" \
      --image-name \"$IMAGE\" \
      --mjpeg-max-fps \"$MJPEG_MAX_FPS\" \
      --jpeg-quality \"$JPEG_QUALITY\" \
      --preview-jpeg-quality \"$PREVIEW_JPEG_QUALITY\" \
      --preview-scale \"$PREVIEW_SCALE\""

READY=0
for ((i=0; i<WAIT_SECONDS; i++)); do
  if curl -fsS "http://127.0.0.1:${PORT}/status" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "HiFi bridge did not become ready on port ${PORT}." >&2
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker logs "$CONTAINER_NAME" || true
  fi
  exit 1
fi

echo "HiFi bridge ready on http://localhost:${PORT}"
