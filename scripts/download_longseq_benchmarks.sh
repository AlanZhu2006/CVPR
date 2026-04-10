#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${ROOT_DIR}/data/raw"
DOWNLOAD_TUM=0
DOWNLOAD_BONN=0
DOWNLOAD_KITTI=0
DOWNLOAD_SCANNET=0
PREPARE_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_longseq_benchmarks.sh [--tum] [--bonn] [--kitti] [--scannet] [--all] [--prepare-only]

Examples:
  bash scripts/download_longseq_benchmarks.sh --tum --bonn
  bash scripts/download_longseq_benchmarks.sh --kitti
  bash scripts/download_longseq_benchmarks.sh --all

Notes:
  - The script downloads raw public datasets into data/raw and then builds the long-sequence subsets under data/.
  - ScanNet download still assumes you already have the right to access the public test scenes listed in the MonST3R script.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tum) DOWNLOAD_TUM=1 ;;
    --bonn) DOWNLOAD_BONN=1 ;;
    --kitti) DOWNLOAD_KITTI=1 ;;
    --scannet) DOWNLOAD_SCANNET=1 ;;
    --all)
      DOWNLOAD_TUM=1
      DOWNLOAD_BONN=1
      DOWNLOAD_KITTI=1
      DOWNLOAD_SCANNET=1
      ;;
    --prepare-only) PREPARE_ONLY=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ ${DOWNLOAD_TUM} -eq 0 && ${DOWNLOAD_BONN} -eq 0 && ${DOWNLOAD_KITTI} -eq 0 && ${DOWNLOAD_SCANNET} -eq 0 ]]; then
  usage
  exit 1
fi

mkdir -p "${RAW_ROOT}"

download_tum() {
  local dst="${RAW_ROOT}/tum"
  mkdir -p "${dst}"
  pushd "${dst}" >/dev/null
  local urls=(
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_rpy.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_rpy.tgz"
  )
  for url in "${urls[@]}"; do
    wget -c "${url}"
  done
  find . -maxdepth 1 -name "*.tgz" -print0 | xargs -0 -n 1 tar -zxf
  find . -maxdepth 1 -name "*.tgz" -delete
  popd >/dev/null
}

download_bonn() {
  local dst="${RAW_ROOT}/bonn"
  mkdir -p "${dst}"
  pushd "${dst}" >/dev/null
  wget -c https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_dataset.zip
  unzip -o rgbd_bonn_dataset.zip
  rm -f rgbd_bonn_dataset.zip
  popd >/dev/null
}

download_kitti() {
  local dst="${RAW_ROOT}/kitti"
  mkdir -p "${dst}"
  pushd "${dst}" >/dev/null
  local urls=(
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0013/2011_09_26_drive_0013_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0020/2011_09_26_drive_0020_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0023/2011_09_26_drive_0023_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0036/2011_09_26_drive_0036_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0079/2011_09_26_drive_0079_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0095/2011_09_26_drive_0095_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0113/2011_09_26_drive_0113_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0037/2011_09_28_drive_0037_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0026/2011_09_29_drive_0026_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip"
  )
  for url in "${urls[@]}"; do
    wget -c "${url}"
  done
  find . -maxdepth 1 -name "*.zip" -print0 | xargs -0 -n 1 unzip -o -q
  find . -maxdepth 1 -name "*.zip" -delete
  popd >/dev/null
}

download_scannet() {
  local dst="${RAW_ROOT}/scannetv2"
  mkdir -p "${dst}"
  pushd "${dst}" >/dev/null
  for i in {707..806}; do
    wget -c "http://kaldir.vc.in.tum.de/scannet/v2/scans/scene0${i}_00/scene0${i}_00.sens"
  done
  popd >/dev/null
}

prepare_downloads() {
  if [[ ${DOWNLOAD_TUM} -eq 1 ]]; then
    python "${ROOT_DIR}/scripts/prepare_longseq_benchmarks.py" --dataset tum --tum-root "${RAW_ROOT}/tum"
  fi
  if [[ ${DOWNLOAD_BONN} -eq 1 ]]; then
    python "${ROOT_DIR}/scripts/prepare_longseq_benchmarks.py" --dataset bonn --bonn-root "${RAW_ROOT}/bonn/rgbd_bonn_dataset"
  fi
  if [[ ${DOWNLOAD_KITTI} -eq 1 ]]; then
    python "${ROOT_DIR}/scripts/prepare_longseq_benchmarks.py" --dataset kitti --kitti-root "${RAW_ROOT}/kitti"
  fi
  if [[ ${DOWNLOAD_SCANNET} -eq 1 ]]; then
    python "${ROOT_DIR}/scripts/prepare_longseq_benchmarks.py" --dataset scannet --scannet-root "${RAW_ROOT}/scannetv2"
  fi
}

if [[ ${PREPARE_ONLY} -eq 0 ]]; then
  [[ ${DOWNLOAD_TUM} -eq 1 ]] && download_tum
  [[ ${DOWNLOAD_BONN} -eq 1 ]] && download_bonn
  [[ ${DOWNLOAD_KITTI} -eq 1 ]] && download_kitti
  [[ ${DOWNLOAD_SCANNET} -eq 1 ]] && download_scannet
fi

prepare_downloads
