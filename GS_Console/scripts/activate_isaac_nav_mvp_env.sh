#!/usr/bin/env bash
# Source this file before working on the Isaac navigation MVP:
#   source /media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh

export ISAAC_WORK_ROOT="${ISAAC_WORK_ROOT:-/media/chatsign/data-002/isaac}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$ISAAC_WORK_ROOT/IsaacLab}"
export ISAAC_NAV_MVP_ENV="${ISAAC_NAV_MVP_ENV:-$ISAAC_WORK_ROOT/env_isaacsim45_lab21}"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PYTHON_EXE:-$ISAAC_NAV_MVP_ENV/bin/python}"

export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ISAAC_WORK_ROOT/cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ISAAC_WORK_ROOT/cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ISAAC_WORK_ROOT/cache/xdg}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$ISAAC_WORK_ROOT/cache/xdg-data}"
export TMPDIR="${TMPDIR:-$ISAAC_WORK_ROOT/tmp}"

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TMPDIR"

if [[ ! -f "$ISAAC_NAV_MVP_ENV/bin/activate" ]]; then
  echo "Isaac MVP environment is missing: $ISAAC_NAV_MVP_ENV" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$ISAAC_NAV_MVP_ENV/bin/activate"

echo "Activated Isaac nav MVP environment:"
echo "  ISAAC_NAV_MVP_ENV=$ISAAC_NAV_MVP_ENV"
echo "  ISAACLAB_PATH=$ISAACLAB_PATH"
echo "  ISAACSIM_PYTHON_EXE=$ISAACSIM_PYTHON_EXE"
