#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_CONTAINER="${1:-gssdf-view}"
IMAGE_TAG="${IMAGE_TAG:-gs-sdf-ros-tools:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gssdf-view-camera-pose}"
RATE="${RATE:-10}"

# Default to the same broad orbit framing used in the RViz config.
FOCAL_X="${FOCAL_X:-0}"
FOCAL_Y="${FOCAL_Y:-0}"
FOCAL_Z="${FOCAL_Z:-0}"
DISTANCE="${DISTANCE:-22}"
YAW="${YAW:-0.65}"
PITCH="${PITCH:-0.9}"

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  bash "$ROOT_DIR/scripts/build_ros_tools_image.sh" "$IMAGE_TAG"
fi

ROS_MASTER_HOST="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$SOURCE_CONTAINER")"
SOURCE_HOSTNAME="$(docker inspect -f '{{.Config.Hostname}}' "$SOURCE_CONTAINER")"
if [[ -z "$ROS_MASTER_HOST" ]]; then
  echo "Failed to resolve ROS master container IP from $SOURCE_CONTAINER" >&2
  exit 1
fi

read -r CAMERA_X CAMERA_Y CAMERA_Z QX QY QZ QW <<<"$(
  python3 - "$FOCAL_X" "$FOCAL_Y" "$FOCAL_Z" "$DISTANCE" "$YAW" "$PITCH" <<'PY'
import math
import sys

fx, fy, fz, dist, yaw, pitch = map(float, sys.argv[1:7])

# RViz Orbit view around the focal point.
cx = fx + dist * math.cos(pitch) * math.cos(yaw)
cy = fy + dist * math.cos(pitch) * math.sin(yaw)
cz = fz + dist * math.sin(pitch)

eye = (cx, cy, cz)
target = (fx, fy, fz)
up_world = (0.0, 0.0, 1.0)

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-8:
        raise SystemExit("degenerate camera vector")
    return (v[0] / n, v[1] / n, v[2] / n)

def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

forward = norm(sub(target, eye))
right = norm(cross(forward, up_world))
up = cross(right, forward)
back = (-forward[0], -forward[1], -forward[2])

# OpenGL camera convention: x=right, y=up, z=back.
m00, m01, m02 = right[0], up[0], back[0]
m10, m11, m12 = right[1], up[1], back[1]
m20, m21, m22 = right[2], up[2], back[2]

trace = m00 + m11 + m22
if trace > 0.0:
    s = math.sqrt(trace + 1.0) * 2.0
    qw = 0.25 * s
    qx = (m21 - m12) / s
    qy = (m02 - m20) / s
    qz = (m10 - m01) / s
elif m00 > m11 and m00 > m22:
    s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
    qw = (m21 - m12) / s
    qx = 0.25 * s
    qy = (m01 + m10) / s
    qz = (m02 + m20) / s
elif m11 > m22:
    s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
    qw = (m02 - m20) / s
    qx = (m01 + m10) / s
    qy = 0.25 * s
    qz = (m12 + m21) / s
else:
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    qw = (m10 - m01) / s
    qx = (m02 + m20) / s
    qy = (m12 + m21) / s
    qz = 0.25 * s

print(f"{cx} {cy} {cz} {qx} {qy} {qz} {qw}")
PY
)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Source container: $SOURCE_CONTAINER"
echo "ROS master:       $ROS_MASTER_HOST"
echo "Camera pose:      [$CAMERA_X, $CAMERA_Y, $CAMERA_Z]"
echo "Quaternion:       [$QX, $QY, $QZ, $QW]"

docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --network bridge \
  -e ROS_MASTER_URI="http://$ROS_MASTER_HOST:11311" \
  --add-host "$SOURCE_HOSTNAME:$ROS_MASTER_HOST" \
  "$IMAGE_TAG" \
  bash -lc "export ROS_IP=\$(hostname -I | awk '{print \$1}'); \
    source /opt/ros/noetic/setup.bash && \
    rostopic pub -r $RATE /rviz/current_camera_pose geometry_msgs/Pose \
    '{position: {x: $CAMERA_X, y: $CAMERA_Y, z: $CAMERA_Z}, orientation: {x: $QX, y: $QY, z: $QZ, w: $QW}}'"

echo "Started $CONTAINER_NAME"
