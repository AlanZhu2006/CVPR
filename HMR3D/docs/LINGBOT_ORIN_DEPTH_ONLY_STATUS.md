# LingBot Orin Depth-Only Status

Date: 2026-04-28

## Summary

The public LingBot checkpoint can run on the Jetson Orin CUDA path only after
using a reduced depth-only checkpoint and a low-peak loading path:

- Drop camera/point heads from the checkpoint.
- Store remaining floating tensors as fp16.
- Load the checkpoint on CPU without `mmap`.
- Cast only the aggregator/backbone to fp16 on CPU.
- Keep the DPT depth head in fp32.
- Move the model to CUDA after the CPU-side cast.

Full-size LingBot is not real-time yet, but it proves Orin-local LingBot is
possible as a slow background dense-depth worker. The most promising Orin path
right now is a tiny distilled depth student: it reaches about 4.7 Hz on the
current KITTI live-window benchmark and about 4.0 Hz inside the standalone depth
worker. The full cuVSLAM + depth + backprojection live demo is lower, about 2.1
Hz in the latest 40-frame smoke, but still much faster than the full LingBot
worker.

## Artifacts

- Depth-only fp16 checkpoint:
  `third_party_research/lingbot_cache/lingbot-map-depth-fp16.pt`
- Reusable benchmark launcher:
  `HMR3D/nuc/scripts/run_lingbot_orin_depth_only_benchmark.sh`
- Component fit probe:
  `HMR3D/nuc/scripts/probe_lingbot_orin_fit.py`
- ONNX/TensorRT feasibility exporter:
  `HMR3D/nuc/scripts/export_lingbot_depth_onnx.py`
- Persistent Orin background depth worker:
  `HMR3D/nuc/scripts/launch_lingbot_orin_depth_worker.sh`
- cuVSLAM + LingBot live reconstruction launcher:
  `HMR3D/nuc/scripts/launch_cuvslam_lingbot_live_reconstruction.sh`
- Student distillation entry point:
  `HMR3D/nuc/scripts/train_lingbot_depth_student.py`
- TensorRT depth-head subgraph exporter:
  `HMR3D/nuc/scripts/export_lingbot_depth_head_onnx.py`
- TensorRT depth-head runtime wrapper:
  `HMR3D/nuc/src/nuc_runtime/lingbot_trt_depth_head.py`
- Student quality evaluator:
  `HMR3D/nuc/scripts/evaluate_lingbot_depth_student.py`

## Reproduce

```bash
HMR3D/nuc/scripts/run_lingbot_orin_depth_only_benchmark.sh
```

Equivalent important environment:

```bash
export LINGBOT_LOAD_CHECKPOINT_ON_CPU=1
export LINGBOT_CHECKPOINT_MMAP=0
export LINGBOT_MODEL_DTYPE=fp16
export LINGBOT_CPU_CAST_BEFORE_CUDA=1
export LINGBOT_CPU_CAST_SCOPE=aggregator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Measured Orin Results

Loaded-model depth-only throughput, using 2-frame windows and skipping the first
load/warmup window:

| Input | Mean latency | Mean rate | Peak allocated |
| --- | ---: | ---: | ---: |
| 518 width | 1.31 s/window | 0.76 Hz | 2.64 GB |
| 336 width | 0.68 s/window | 1.48 Hz | 2.18 GB |
| 224 width | 0.74 s/window | 1.36 Hz | 1.98 GB |

The first window includes checkpoint/model load and takes roughly 28-38 seconds.

## Background Worker

The Orin path now has a persistent depth-only worker. It keeps one LingBot model
instance alive and consumes overlapping frame windows in the background, so the
foreground tracker can stay real-time while dense depth arrives more slowly.

Smoke test:

```bash
PYTHON_BIN=/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python \
MAX_FRAMES=2 \
OUTPUT_DIR=/home/nyu/Codespace/CVPR/nuc_output/lingbot_depth_worker/smoke_2f \
IMAGE_SIZE=336 \
WINDOW_SIZE=2 \
STRIDE=1 \
MAX_QUEUE=1 \
HMR3D/nuc/scripts/launch_lingbot_orin_depth_worker.sh
```

Result:

- Completed 1 two-frame window.
- Wrote `lingbot_predictions.npz`, `lingbot_summary.json`,
  `worker_result.json`, and `worker_status.json`.
- First-window elapsed time was 26.8 seconds, dominated by model load.
- Output directory:
  `nuc_output/lingbot_depth_worker/smoke_2f/window_000000`

Multi-window worker tests:

| Input | Windows | First window | Steady mean | Steady rate |
| --- | ---: | ---: | ---: | ---: |
| 336 | 5 | 30.67 s | 0.945 s/window | 1.06 Hz |
| 224 | 5 | 45.76 s | 0.629 s/window | 1.59 Hz |

The steady numbers include image loading, LingBot inference, postprocess, and
compressed NPZ writes. They are lower than the in-memory benchmark because the
worker persists a bundle for every window.

## Live cuVSLAM + LingBot Reconstruction Demo

The current visible end-to-end path is:

```text
KITTI/real stereo frames
  -> cuVSLAM tracking pose stream
  -> LingBot depth-only background worker
  -> depth + cuVSLAM pose backprojection
  -> rolling local point cloud
  -> auto-refreshing browser viewer
```

Launcher:

```bash
PYTHON_BIN=/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python \
OUTPUT_DIR=/home/nyu/Codespace/CVPR/nuc_output/lingbot_live_reconstruction/kitti0020_live_dense \
IMAGE_SIZE=224 \
FRAME_STEP=2 \
MAX_FRAMES=40 \
WINDOW_SIZE=2 \
SAMPLE_STRIDE=4 \
MAX_POINTS_PER_FRAME=2000 \
MAX_ACTIVE_FRAMES=20 \
PORT=19093 \
HMR3D/nuc/scripts/launch_cuvslam_lingbot_live_reconstruction.sh
```

Result on Orin:

- Submitted 20 cuVSLAM-tracked frames.
- Processed 19 overlapping LingBot windows.
- First LingBot window took 28.78 s because it includes model load.
- Later windows took roughly 0.52-1.17 s each.
- Final rolling local map has 19,040 points.
- Viewer URL:
  `http://127.0.0.1:19093/live_viewer.html`
- Artifacts:
  `nuc_output/lingbot_live_reconstruction/kitti0020_live_dense/live_map.json`,
  `live_map.npz`, `live_map.ply`, and `live_viewer.html`.

This is a real visible reconstruction loop, but not yet true full-rate
reconstruction. cuVSLAM is the real-time tracking thread; full LingBot currently
runs as a delayed mapper. The live viewer shows the latest available local map
as LingBot windows finish.

## TensorRT Status

TensorRT 10.3 is installed system-wide, and `trtexec` exists at:

```bash
/usr/src/tensorrt/bin/trtexec
```

The CUDA PyTorch venv does not include `tensorrt`, `onnx`, `onnxruntime`,
`torchao`, or `flashinfer`.

Fixed-shape depth-only smoke forward for ONNX export works at
`[1, 2, 3, 196, 336]`, but full CUDA ONNX export fails on Jetson with a CUDA
allocator/NvMap error during tracing. This happens before TensorRT sees the
model: PyTorch's ONNX exporter replays the forward pass and holds extra trace
state/initializers on top of the normal inference tensors.

The current exporter has two ONNX-compatibility fixes:

- DINO positional interpolation disables bicubic antialiasing by default because
  legacy ONNX export cannot lower `aten::_upsample_bicubic2d_aa`.
- RoPE position generation avoids `torch.cartesian_prod`, replacing it with
  broadcasted `arange` grids.

CPU ONNX export avoids the Jetson NvMap failure and gets past those operator
issues, but full LingBot ViT-L depth export is still too slow on Orin CPU and
timed out in the current test.

Remote A6000 ONNX export now works after the compatibility fixes:

| Export | Shape | Result |
| --- | --- | --- |
| 224 | `[1, 2, 3, 126, 224]` | `/media/chatsign/data-002/lingbot_trt/depth224_remote_cuda/lingbot_depth_224.onnx`, 1.9 GB |
| 336 | `[1, 2, 3, 196, 336]` | `/media/chatsign/data-002/lingbot_trt/depth336_remote_cuda/lingbot_depth_336.onnx`, about 2.0 GB with external data |

The 224 ONNX was copied back to Orin at:

```bash
nuc_output/lingbot_trt/depth224_remote_cuda/lingbot_depth_224.onnx
```

Orin TensorRT 10.3 can parse that ONNX, but full engine build still fails under
8 GB memory pressure:

- Normal fp16 build fails after parser succeeds with NvMap allocation failures
  while trying to place large constant/weight regions.
- Low-memory build with `--builderOptimizationLevel=0`, `--maxAuxStreams=0`,
  and reduced workspace still exhausts memory.
- `--allowWeightStreaming --stronglyTyped` also reaches the build phase, but
  drives RAM to about 7.2/7.4 GiB and swap to 5.7/5.7 GiB, so it was killed to
  protect the desktop session.

This means the current blocker is no longer ONNX export itself. The blocker is
that the full LingBot ViT-L depth graph is too large for full TensorRT engine
construction on this 8 GB Orin target. The practical path is to either split
only smaller subgraphs into TensorRT, use a smaller/student model, or build and
run this class of full model on a larger GPU target.

### Depth-Head Subgraph

The DPT depth head can be split out as a TensorRT-sized subgraph. The current
prototype runs the PyTorch aggregator/backbone, exports the four selected
feature tensors plus RGB image into a standalone depth-head ONNX, and builds a
TensorRT engine on Orin.

Smoke/export command:

```bash
/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python \
  HMR3D/nuc/scripts/export_lingbot_depth_head_onnx.py \
  --checkpoint third_party_research/lingbot_cache/lingbot-map-depth-fp16.pt \
  --output nuc_output/lingbot_trt/depth_head_224/lingbot_depth_head_224.onnx \
  --input-width 224 \
  --input-height 126 \
  --model-image-size 518 \
  --num-scale-frames 2 \
  --device cuda
```

Build command:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=nuc_output/lingbot_trt/depth_head_224/lingbot_depth_head_224.onnx \
  --saveEngine=nuc_output/lingbot_trt/depth_head_224/lingbot_depth_head_224_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:512M \
  --skipInference
```

Result:

- ONNX export succeeded, size 125 MB.
- TensorRT build succeeded on Orin, engine size 64 MB.
- Engine generation took 242.7 seconds.
- Engine build reported about 66 MB weights and 56 MB activation memory.
- Runtime inference with random feature inputs succeeded:
  - Throughput: 93.9 qps.
  - Mean host latency: 11.37 ms.
  - Mean GPU compute time: 10.63 ms.
- Artifacts:
  `nuc_output/lingbot_trt/depth_head_224/lingbot_depth_head_224.onnx`
  and `nuc_output/lingbot_trt/depth_head_224/lingbot_depth_head_224_fp16.engine`

The engine is now wired into the live worker through:

```bash
DEPTH_HEAD_TRT_ENGINE=/home/nyu/Codespace/CVPR/nuc_output/lingbot_trt/depth_head_224/lingbot_depth_head_224_fp16.engine \
HMR3D/nuc/scripts/launch_lingbot_orin_depth_worker.sh
```

Functional worker smoke result:

- Completed 5 windows at 224 input.
- `lingbot_summary.json` reports `depth_head_backend: tensorrt`.
- No TensorRT runtime failures.

However, this first integration is not faster end-to-end yet:

| Worker | First window | Steady mean | Steady rate |
| --- | ---: | ---: | ---: |
| PyTorch depth head, 224 | 45.76 s | 0.629 s/window | 1.59 Hz |
| TensorRT depth head, 224 | 33.07 s | 0.732 s/window | 1.37 Hz |

The standalone TensorRT head is fast, but the current wrapper still pays for
large fp32 feature tensors, binding/reformat overhead, and a generic execution
path. Treat it as a correct integration milestone, not the current speed path.
The next TRT optimization would be a tighter fp16 feature contract and/or a
larger TensorRT subgraph that avoids PyTorch-to-TRT handoff overhead.

## Student Model Probe

Random-weight smaller student skeletons run, but the speedup is modest because
the DPT/depth path and framework overhead remain significant:

| Student | Mean latency | Mean rate | Peak allocated |
| --- | ---: | ---: | ---: |
| ViT-S, embed 384 | 0.46 s/window | 2.17 Hz | 0.63 GB |
| ViT-B, embed 768 | 0.46 s/window | 2.19 Hz | 1.39 GB |

This suggests that distillation can reduce memory a lot, but real-time likely
also needs a lighter depth head and/or TensorRT-friendly architecture.

The student distillation entry point can train from either an online teacher
checkpoint or cached teacher predictions. `GCTStream` now keeps the default
teacher architecture unchanged, but accepts student-side `depth`, `num_heads`,
and `mlp_ratio` settings so we can build genuinely smaller students. The
Orin-safe smoke path uses cached teacher depth so it does not need to load the
full teacher during student training:

```bash
/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python \
  HMR3D/nuc/scripts/train_lingbot_depth_student.py \
  --image-dir third_party_research/lingbot-map/example/church \
  --glob '*.png' \
  --teacher-predictions-npz nuc_output/lingbot_depth_worker/smoke_2f/window_000000/lingbot_predictions.npz \
  --output-dir nuc_output/lingbot_student_distill/smoke_1step_tiny_depth8_cached_teacher \
  --device cuda \
  --student-patch-embed conv \
  --student-embed-dim 384 \
  --student-depth 8 \
  --student-num-heads 6 \
  --student-mlp-ratio 3.0 \
  --image-size 224 \
  --model-image-size 224 \
  --student-model-image-size 224 \
  --steps 1
```

Result:

- One training step completed on Orin CUDA.
- Tiny student config: conv patch embed, embed dim 384, depth 8, 6 heads,
  MLP ratio 3.0.
- Loss: 0.2020 after one smoke step.
- Checkpoint size: 203 MB.
- Checkpoint:
  `nuc_output/lingbot_student_distill/smoke_1step_tiny_depth8_cached_teacher/lingbot_depth_student.pt`

Tiny student benchmark:

```bash
/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python \
  HMR3D/nuc/scripts/benchmark_lingbot_depth_student.py \
  --image-dir third_party_research/lingbot-map/example/church \
  --glob '*.png' \
  --checkpoint nuc_output/lingbot_student_distill/smoke_1step_tiny_depth8_cached_teacher/lingbot_depth_student.pt \
  --output-json nuc_output/lingbot_student_distill/smoke_1step_tiny_depth8_cached_teacher/benchmark_6f_224.json \
  --image-size 224 \
  --model-image-size 224 \
  --max-frames 6 \
  --window-size 2 \
  --stride 1 \
  --warmup-windows 1 \
  --patch-embed conv \
  --embed-dim 384 \
  --depth 8 \
  --num-heads 6 \
  --mlp-ratio 3.0 \
  --device cuda
```

Result:

- Steady mean latency: 0.233 s/window.
- Steady mean rate: 4.30 Hz.
- Peak allocated CUDA memory: 0.263 GB.
- This is a speed/fit proof only; quality is not meaningful after one smoke
  training step.

### Cached-Teacher 300-Step Tiny Student

The first formal Orin-local tiny student was trained from the cached live dense
teacher windows:

```bash
/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python \
  HMR3D/nuc/scripts/train_lingbot_depth_student.py \
  --teacher-windows-dir nuc_output/lingbot_live_reconstruction/kitti0020_live_dense/worker \
  --output-dir nuc_output/lingbot_student_distill/kitti0020_cached_teacher_depth8_300step \
  --device cuda \
  --student-patch-embed conv \
  --student-embed-dim 384 \
  --student-depth 8 \
  --student-num-heads 6 \
  --student-mlp-ratio 3.0 \
  --image-size 224 \
  --model-image-size 224 \
  --student-model-image-size 224 \
  --steps 300 \
  --lr 1e-4
```

Result:

- Trained 300 steps in 98.1 seconds on Orin CUDA.
- Loss dropped from 0.1188 to 0.0152.
- Checkpoint:
  `nuc_output/lingbot_student_distill/kitti0020_cached_teacher_depth8_300step/lingbot_depth_student.pt`
- Report:
  `nuc_output/lingbot_student_distill/kitti0020_cached_teacher_depth8_300step/distill_report.json`

Benchmark on the same 20 KITTI live frames:

| Model | Mean latency | Mean rate | Peak allocated |
| --- | ---: | ---: | ---: |
| Full LingBot worker, 224 | 0.629 s/window | 1.59 Hz | about 1.98 GB |
| Tiny student depth8, 224 | 0.213 s/window | 4.70 Hz | 0.253 GB |

Benchmark artifact:
`nuc_output/lingbot_student_distill/kitti0020_cached_teacher_depth8_300step/benchmark_kitti_teacher_20f_224.json`

Quality against cached teacher is usable but not finished:

| Metric | Mean |
| --- | ---: |
| log-depth MAE | 0.184 |
| log-depth RMSE | 0.240 |
| AbsRel vs teacher | 0.217 |
| Median scale ratio | 1.124 |

Quality artifact:
`nuc_output/lingbot_student_distill/kitti0020_cached_teacher_depth8_300step/eval_kitti_teacher/eval_report.json`

Preview page:
`nuc_output/lingbot_student_distill/kitti0020_cached_teacher_depth8_300step/eval_kitti_teacher/preview_index.html`

The tiny student can now be loaded directly by the worker/live reconstruction
path with direct `GCTStream` architecture overrides:

```bash
MODEL_PATH=/home/nyu/Codespace/CVPR/nuc_output/lingbot_student_distill/kitti0020_cached_teacher_depth8_300step/lingbot_depth_student.pt \
MODEL_PATCH_EMBED=conv \
MODEL_EMBED_DIM=384 \
MODEL_DEPTH=8 \
MODEL_NUM_HEADS=6 \
MODEL_MLP_RATIO=3.0 \
MODEL_IMAGE_SIZE=224 \
IMAGE_SIZE=224 \
COMPRESS_OUTPUTS=0 \
HMR3D/nuc/scripts/launch_cuvslam_lingbot_live_reconstruction.sh
```

Worker smoke with the student checkpoint:

- 5 windows, no failures.
- First window: 2.00 s.
- Steady mean: 0.249 s/window, about 4.01 Hz.
- Summary reports `model_load_missing_keys: 0` and
  `model_load_unexpected_keys: 0`.
- Output:
  `nuc_output/lingbot_depth_worker/student_depth8_smoke_6f_224`

End-to-end cuVSLAM + student live smoke:

| Run | Windows | Point count | Mean after first | Rate after first | Viewer |
| --- | ---: | ---: | ---: | ---: | --- |
| full LingBot baseline | 19 | 19,040 | 0.842 s/window | 1.19 Hz | `http://127.0.0.1:19093/live_viewer.html` |
| tiny student, compressed | 19 | 19,040 | 0.552 s/window | 1.81 Hz | `http://127.0.0.1:19095/live_viewer.html` |
| tiny student, uncompressed | 19 | 19,040 | 0.465 s/window | 2.15 Hz | `http://127.0.0.1:19096/live_viewer.html` |

Interpretation: the tiny student has crossed the useful speed threshold for the
depth-only worker, but the full live reconstruction still has CPU/GPU
contention, postprocess, backprojection, and bundle-writing overhead. It is a
real speedup, not yet full 4 Hz end-to-end. The next training pass should use
more teacher windows and longer distillation, while the next systems pass should
remove per-window disk bundles from the live path and stream depth arrays in
memory.
