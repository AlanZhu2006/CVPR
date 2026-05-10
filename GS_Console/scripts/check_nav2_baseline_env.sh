#!/usr/bin/env bash
set -euo pipefail

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
missing=0

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS2 setup file not found: $ROS_SETUP" >&2
  exit 1
fi

# ROS setup files may touch unset AMENT_* variables under `set -u`.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 is not available after sourcing $ROS_SETUP" >&2
  exit 1
fi

echo "ROS2: $(command -v ros2)"

package_exists() {
  python3 - "$1" <<'PY' >/dev/null 2>&1
import sys
from ament_index_python.packages import get_package_share_directory

get_package_share_directory(sys.argv[1])
PY
}

for package in nav2_bringup nav2_bt_navigator nav2_planner nav2_controller nav2_map_server nav2_msgs; do
  if package_exists "$package"; then
    echo "OK: $package"
  else
    echo "MISSING: $package"
    missing=1
  fi
done

for executable in \
  /opt/ros/humble/lib/nav2_map_server/map_server \
  /opt/ros/humble/lib/nav2_planner/planner_server \
  /opt/ros/humble/lib/nav2_controller/controller_server \
  /opt/ros/humble/lib/nav2_bt_navigator/bt_navigator
do
  if [[ -x "$executable" ]]; then
    echo "OK: $executable"
  else
    echo "MISSING: $executable"
    missing=1
  fi
done

if [[ "$missing" == "1" ]]; then
  cat <<'EOF'

Nav2 is not installed for this ROS2 Humble environment.
Install command, requires sudo:

  sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

After install, rerun this check before wiring Isaac /cmd_vel, /odom, /tf, and /map.
EOF
  exit 2
fi

echo "Nav2 baseline dependencies are present."
