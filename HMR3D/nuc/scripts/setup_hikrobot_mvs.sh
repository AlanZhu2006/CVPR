#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  HMR3D/nuc/scripts/setup_hikrobot_mvs.sh /path/to/MVS-*_aarch64_*.deb
  HMR3D/nuc/scripts/setup_hikrobot_mvs.sh /path/to/MVS-*_aarch64_*.tar.gz

This installs HikRobot MVS for USB3/GigE industrial cameras, adds a udev rule
for HikRobot USB cameras, and writes a small environment file for runtime use.

Download the ARM64/aarch64 Linux MVS package from:
  https://www.hikrobotics.com/en/machinevision/service/download/
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -ne 1 ]]; then
  usage
  exit 0
fi

PKG="$1"
if [[ ! -f "$PKG" ]]; then
  echo "MVS package not found: $PKG" >&2
  exit 2
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "Warning: this machine reports arch=$ARCH; Jetson should normally be aarch64." >&2
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script needs sudo because it writes /opt/MVS and /etc/udev/rules.d." >&2
  echo "Run: sudo bash $0 $PKG" >&2
  exit 3
fi

case "$PKG" in
  *.deb)
    dpkg -i "$PKG" || apt-get install -f -y
    ;;
  *.tar.gz|*.tgz)
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    tar -xzf "$PKG" -C "$TMP"
    SETUP="$(find "$TMP" -maxdepth 3 -type f -name setup.sh | head -1)"
    if [[ -z "$SETUP" ]]; then
      echo "Could not find setup.sh inside $PKG" >&2
      exit 4
    fi
    bash "$SETUP"
    ;;
  *)
    echo "Unsupported package type: $PKG" >&2
    usage
    exit 5
    ;;
esac

install -d /etc/udev/rules.d
cat >/etc/udev/rules.d/99-hikrobot-usb3vision.rules <<'EOF'
# HikRobot/Hikvision USB3 Vision industrial cameras.
SUBSYSTEM=="usb", ATTR{idVendor}=="2bdf", MODE="0666", GROUP="video", TAG+="uaccess"
EOF

udevadm control --reload-rules
udevadm trigger || true

ENV_OUT="/home/nvidia/twork/lingbot-map/CVPR/HMR3D/nuc/configs/hikrobot_mvs_env.sh"
install -d "$(dirname "$ENV_OUT")"
cat >"$ENV_OUT" <<'EOF'
# Source this before running HikRobot MVS Python/ROS2 tools.
export MVS_HOME="${MVS_HOME:-/opt/MVS}"
export LD_LIBRARY_PATH="$MVS_HOME/lib/aarch64:$MVS_HOME/lib/64:$MVS_HOME/bin:$LD_LIBRARY_PATH"
export PYTHONPATH="$MVS_HOME/Samples/aarch64/Python/MvImport:$MVS_HOME/Samples/arm64/Python/MvImport:$MVS_HOME/Samples/64/Python/MvImport:$PYTHONPATH"
EOF
chown nvidia:nvidia "$ENV_OUT" 2>/dev/null || true

echo "MVS setup complete."
echo "Environment file: $ENV_OUT"
echo "Unplug/replug the USB camera if permissions do not update immediately."
