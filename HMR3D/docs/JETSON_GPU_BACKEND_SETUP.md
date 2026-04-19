## Jetson GPU Backend Setup

This document records the working GPU backend environment on the current
Jetson Orin Nano machine for `cuVSLAM/.venv-jetson`.

### Verified system stack

- Device: Jetson Orin Nano
- JetPack: `6.2.1`
- L4T: `R36.4.7`
- System CUDA: `/usr/local/cuda-12.6`
- Python in venv: `3.10.12`

### Installed Python packages in `.venv-jetson`

Virtualenv:

- `/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson`

Installed GPU stack:

- `torch 2.8.0`
- `torchvision 0.23.0`
- `torchaudio 2.8.0`
- `gsplat 1.5.3`
- `numpy 1.26.4`

Package locations:

- `torch`
  - `/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/lib/python3.10/site-packages/torch`
- `torchvision`
  - `/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/lib/python3.10/site-packages/torchvision`
- `torchaudio`
  - `/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/lib/python3.10/site-packages/torchaudio`
- `gsplat`
  - `/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/lib/python3.10/site-packages/gsplat`

### Installed cuDSS runtime

`PyTorch 2.8.0` on this Jetson stack expects `libcudss.so.0`.

Local install path:

- `/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/opt/cudss-0.6.0`

Library directory:

- `/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/opt/cudss-0.6.0/lib`

Important files:

- `libcudss.so`
- `libcudss.so.0`
- `libcudss.so.0.6.0`
- `libcudss_mtlayer_gomp.so`
- `libcudss_mtlayer_gomp.so.0`
- `libcudss_mtlayer_gomp.so.0.6.0`

### Wheels used

PyTorch wheel:

- `https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl`

Torchvision wheel:

- `https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl`

Torchaudio wheel:

- `https://pypi.jetson-ai-lab.io/jp6/cu126/+f/81a/775c8af36ac85/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl`

cuDSS archive:

- `https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-aarch64/libcudss-linux-aarch64-0.6.0.5_cuda12-archive.tar.xz`

### Why `numpy` is pinned below 2

The Jetson `torch 2.8.0` wheel warns on `numpy 2.x` because it was compiled
against the NumPy 1 ABI. The working pin is:

- `numpy==1.26.4`

### Quick activation

Use this helper:

```bash
source /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/use_jetson_gpu_backend.sh
```

It will:

- activate `cuVSLAM/.venv-jetson`
- prepend the local cuDSS runtime to `LD_LIBRARY_PATH`

### Quick verification

```bash
source /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/use_jetson_gpu_backend.sh
python - <<'PY'
import torch, gsplat
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
print(gsplat.__version__)
PY
```

Expected result:

- `torch.cuda.is_available()` returns `True`
- device name is `Orin`
- `gsplat` imports successfully

### Next integration target

With this environment in place, the next clean step is to add a backend adapter
layer so HMR3D can choose between:

- current lightweight CPU renderer
- `gsplat` GPU backend

without changing the memory lifecycle code.
