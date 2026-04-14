#!/usr/bin/env bash
# Apply local compatibility patches to the pinned TTT3R submodule (PyTorch 2.6+ torch.load,
# CPU RoPE -1 positions, evo/matplotlib lazy plot + degenerate Umeyama fallback).
# Safe to re-run: uses `patch --forward` and will skip already-applied hunks when possible.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="${ROOT}/patches/ttt3r_runtime_compat/0001-ttt3r-runtime-compat.diff"
TTT="${ROOT}/third_party/TTT3R"
if [[ ! -f "${PATCH}" ]]; then
  echo "Missing ${PATCH}" >&2
  exit 1
fi
if [[ ! -d "${TTT}/.git" ]]; then
  echo "Run: git submodule update --init third_party/TTT3R" >&2
  exit 1
fi
cd "${TTT}"
patch -p1 --forward < "${PATCH}" || true
echo "Done. If hunks failed, submodule may already be patched."
