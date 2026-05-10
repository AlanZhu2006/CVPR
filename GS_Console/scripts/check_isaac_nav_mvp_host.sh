#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/media/chatsign/data-002}"
ISAAC_WORK_ROOT="${ISAAC_WORK_ROOT:-$DATA_ROOT/isaac}"
ISAAC_NAV_MVP_ENV="${ISAAC_NAV_MVP_ENV:-$ISAAC_WORK_ROOT/env_isaacsim45_lab21}"

warn_count=0
fail_count=0

section() {
  printf "\n== %s ==\n" "$1"
}

ok() {
  printf "[OK]   %s\n" "$1"
}

warn() {
  warn_count=$((warn_count + 1))
  printf "[WARN] %s\n" "$1"
}

fail() {
  fail_count=$((fail_count + 1))
  printf "[FAIL] %s\n" "$1"
}

exists_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    ok "$label: $path"
  else
    warn "$label missing: $path"
  fi
}

exists_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "$path" ]]; then
    ok "$label: $path"
  else
    warn "$label missing: $path"
  fi
}

section "Host"
printf "Repo: %s\n" "$ROOT_DIR"
printf "Data root: %s\n" "$DATA_ROOT"
printf "Isaac work root: %s\n" "$ISAAC_WORK_ROOT"
uname -a
if command -v lsb_release >/dev/null 2>&1; then
  lsb_release -ds
elif [[ -r /etc/os-release ]]; then
  sed -n 's/^PRETTY_NAME=//p' /etc/os-release
fi

section "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || fail "nvidia-smi query failed"
else
  fail "nvidia-smi not found"
fi

section "Memory"
free -h || warn "free failed"

section "Disk"
df -h / "$DATA_ROOT" 2>/dev/null || warn "df failed"
root_avail_kb="$(df -Pk / 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -n "${root_avail_kb:-}" && "$root_avail_kb" -lt 10485760 ]]; then
  warn "root filesystem has less than 10 GB free; keep Isaac installs and caches on $DATA_ROOT"
fi
data_avail_kb="$(df -Pk "$DATA_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -n "${data_avail_kb:-}" && "$data_avail_kb" -lt 209715200 ]]; then
  warn "$DATA_ROOT has less than 200 GB free"
else
  ok "$DATA_ROOT has enough free space for the first MVP"
fi

section "Docker"
if command -v docker >/dev/null 2>&1; then
  ok "docker found"
  if docker image inspect gs_sdf_img:latest >/dev/null 2>&1; then
    ok "Docker image gs_sdf_img:latest exists"
  else
    warn "Docker image gs_sdf_img:latest not found"
  fi
  if docker image inspect gs-sdf-ros-tools:latest >/dev/null 2>&1; then
    ok "Docker image gs-sdf-ros-tools:latest exists"
  else
    warn "Docker image gs-sdf-ros-tools:latest not found"
  fi
else
  fail "docker not found"
fi

section "Existing GS-SDF MVP Assets"
exists_dir "$ROOT_DIR/runtime/playback-cache/fast_livo2_compressed_rgbdense_kf16" "playback cache"
exists_file "$ROOT_DIR/runtime/playback-cache/fast_livo2_compressed_rgbdense_kf16/meta.json" "playback cache meta"
exists_file "$ROOT_DIR/examples/web-ui/public/scenes/fast-livo2-compressed-live/manifest.json" "scene manifest"

QUALITY_OUTPUT="$ROOT_DIR/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml"
exists_dir "$QUALITY_OUTPUT" "quality GS-SDF output"
exists_file "$QUALITY_OUTPUT/model/gs.ply" "Gaussian PLY"
exists_file "$QUALITY_OUTPUT/mesh_gs_.ply" "mesh/proxy PLY"
exists_file "$QUALITY_OUTPUT/model/as_occ_prior.ply" "occupancy PLY"
exists_file "$ISAAC_WORK_ROOT/nav-mvp/assets/fast_livo2_collision_full.usd" "generated collision USD"
exists_file "$ISAAC_WORK_ROOT/nav-mvp/assets/fast_livo2_collision_full.usd.manifest.json" "generated collision USD manifest"
exists_file "$ISAAC_WORK_ROOT/nav-mvp/maps/fast_livo2_nav_z0p2_2p0_r0p2.pgm" "generated navigation PGM"
exists_file "$ISAAC_WORK_ROOT/nav-mvp/maps/fast_livo2_nav_z0p2_2p0_r0p2.yaml" "generated navigation YAML"
exists_file "$ISAAC_WORK_ROOT/nav-mvp/maps/fast_livo2_nav_z0p2_2p0_r0p2.manifest.json" "generated navigation map manifest"

section "Isaac Sim / Isaac Lab"
isaac_sim_candidates=()
isaac_lab_candidates=()
python_candidates=()

if [[ -n "${ISAACSIM_PATH:-}" ]]; then
  isaac_sim_candidates+=("$ISAACSIM_PATH")
fi
isaac_sim_candidates+=(
  "$ISAAC_NAV_MVP_ENV/bin"
  "$ISAAC_WORK_ROOT/isaac-sim"
  "$ISAAC_WORK_ROOT/isaac-sim-standalone"
  "$DATA_ROOT/isaac-sim"
  "$HOME/Downloads/isaac-sim-standalone@4.5.0"
  "$HOME/Downloads/isaac-sim-standalone@5.0.0"
)

if [[ -n "${ISAACLAB_PATH:-}" ]]; then
  isaac_lab_candidates+=("$ISAACLAB_PATH")
fi
isaac_lab_candidates+=(
  "$ISAAC_WORK_ROOT/IsaacLab"
  "$DATA_ROOT/IsaacLab"
  "$HOME/IsaacLab"
)

if [[ -n "${ISAACSIM_PYTHON_EXE:-}" ]]; then
  python_candidates+=("$ISAACSIM_PYTHON_EXE")
fi
python_candidates+=(
  "$ISAAC_NAV_MVP_ENV/bin/python"
)

found_isaac_sim=0
for path in "${isaac_sim_candidates[@]}"; do
  if [[ -x "$path/isaac-sim.sh" || -x "$path/python.sh" || -x "$path/isaacsim" ]]; then
    ok "Isaac Sim candidate: $path"
    found_isaac_sim=1
  fi
done
if [[ "$found_isaac_sim" -eq 0 ]]; then
  warn "Isaac Sim not found in the checked paths; install it under $ISAAC_WORK_ROOT"
fi

found_isaac_lab=0
for path in "${isaac_lab_candidates[@]}"; do
  if [[ -f "$path/isaaclab.sh" || -d "$path/source/isaaclab" ]]; then
    ok "Isaac Lab candidate: $path"
    found_isaac_lab=1
  fi
done
if [[ "$found_isaac_lab" -eq 0 ]]; then
  warn "Isaac Lab not found in the checked paths; clone it under $ISAAC_WORK_ROOT"
fi

found_isaac_python=0
for python_exe in "${python_candidates[@]}"; do
  if [[ -x "$python_exe" ]]; then
    found_isaac_python=1
    ok "Isaac Python candidate: $python_exe"
    "$python_exe" - <<'PY' || warn "Isaac Python package check failed"
import importlib.metadata as md
for name in ("isaacsim", "isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "torch"):
    try:
        print(f"  {name}=={md.version(name)}")
    except Exception as exc:
        print(f"  {name}: missing ({exc})")
PY
    "$python_exe" - <<'PY' || warn "Torch CUDA check failed"
import torch
print(f"  torch.cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  torch.cuda.device={torch.cuda.get_device_name(0)}")
PY
  fi
done
if [[ "$found_isaac_python" -eq 0 ]]; then
  warn "Isaac Python environment not found at $ISAAC_NAV_MVP_ENV"
fi

section "Summary"
printf "Warnings: %d\n" "$warn_count"
printf "Failures: %d\n" "$fail_count"

if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi

exit 0
