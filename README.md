# CVPR Workspace: cuVSLAM + HMR3D + Streaming Gaussian

这个工作区的主线是：

> 用 `cuVSLAM` 做低延迟 tracking，用 `HMR3D` 管长期记忆生命周期，用 Gaussian submap 做局部可渲染表示。

它现在已经是一个能跑通的系统原型，但还不是 StreamGS / GS-SLAM / GS-SDF 那种高画质 Gaussian 重建器。当前最重要的判断是：

> 现有 Gaussian 画质差，不是 WebUI 问题，而是几何和 Gaussian 参数生成后端还太弱。`cuVSLAM` 能告诉我们相机在哪里，但它的 visual landmarks 不是 dense renderable geometry。

更详细的阶段性备忘在：

- [PROJECT_STATUS_AND_REALTIME_GAUSSIAN_PLAN.md](/home/nyu/Codespace/CVPR/PROJECT_STATUS_AND_REALTIME_GAUSSIAN_PLAN.md)
- [LingBot / cuVSLAM 16GB Orin 复现指南](HMR3D/docs/LINGBOT_ORIN_16GB_REPRO.md)
- [LingBot Orin depth-only 状态记录](HMR3D/docs/LINGBOT_ORIN_DEPTH_ONLY_STATUS.md)

## Quick LingBot 16GB Smoke

在 16GB Orin 或更大 GPU 上，优先验证 LingBot full teacher 的 `church`
demo，而不是先看 8GB depth-only student 融合图：

```bash
MODEL_PATH=/path/to/lingbot-map.pt \
FIRST_K=20 \
IMAGE_SIZE=336 \
KEYFRAME_INTERVAL=2 \
PORT=19115 \
HMR3D/nuc/scripts/run_lingbot_church_16gb_smoke.sh
```

打开：

```text
http://127.0.0.1:19115
```

如果远程看：

```bash
ssh -L 19115:127.0.0.1:19115 nyu@<orin-host>
```

当前 8GB 本机已经跑通 `cuVSLAM mono + LingBot depth8 student + RGB voxel
fusion` 的 church 80 帧 smoke，但它不是官方 full LingBot 效果。主要限制是
student 只有 depth head，pose 依赖 cuVSLAM monocular；在无标定 church 图片上
会有尺度/轨迹漂移。

## Current Status

### 已经跑通

- `cuVSLAM -> HMR3D` adapter
- KITTI 06 offline replay
- HMR3D memory lifecycle:
  - active submap
  - archive
  - retrieve
  - recover
  - merge
  - hierarchical bank
- Incremental Gaussian submap:
  - active Gaussian update
  - archived Gaussian handle
  - recovered Gaussian warm start
  - PLY / NPZ export
- CPU surfel-style renderer
- experimental `gsplat` backend adapter
- LingBot-Map export path:
  - `LingBot predictions -> depth / pointmap / confidence / intrinsic / extrinsic`
  - `LingBot predictions -> Gaussian handle`
- Web timeline demos with draggable playback
- SSH tunnel based browser viewing through port `19090`

### 当前最真实的问题

1. Gaussian render 画质很差。

   `realtime_budget` 版能接近 1 FPS，但图像非常糊。它是低预算预览，不是高质量 Gaussian rendering。

2. 质量版很慢。

   `surfel_denseviewer` 能比 realtime 版好一些，但只有约 `0.2 FPS`。

3. `cuVSLAM` 地图不是 renderable dense map。

   `cuVSLAM` 的 landmark 是为了 tracking / localization，不是为了覆盖每个可见表面。它缺少 dense depth、normal、surface continuity、opacity、SH color 等渲染所需信息。

4. LingBot-Map 在当前小设备上容易 OOM。

   本地实测：
   - 2 frame 可以用于实验
   - 6 frame 已经会接近或触发内存危险
   - 120 frame 会被 guard 停掉

5. LingBot-Map 不是最终 Gaussian renderer。

   它更适合作为 dense geometry front-end / teacher，而不是直接替代 HMR3D 或直接成为实时主循环。

6. Mem3R / LingBot / StreamGS 这类 foundation model 不能假设能在 Orin Nano 上原版实时跑。

   它们论文里的 real-time 通常依赖更强 GPU、更优 CUDA kernel、更低 overhead 的推理环境。机器人本机更现实的角色是跑 tracking；dense geometry / Gaussian 后端应先做异步或远端 GPU。

## Latest Progress: NuRec + GS-SDF

最近一次有效进展不是调 renderer 参数，而是修正 NuRec 几何监督。

当前用于 GS-SDF/NuRec smoke 的 mesh 不是 LingBot 输出，而是 NuRec 自带场景资产：

```text
/media/chatsign/data-002/datasets/nurec/nova_carter-galileo/extracted_assets/stage_volume/mesh.ply
```

之前效果像椭圆球、没有连续墙面/地面，主要原因是 converter 的几何坐标不对：

- 旧流程把 NuRec `training_trajectory_poses.tum` 直接当成 camera pose，用错了坐标含义。
- 旧 `stereo.edex` 路径下 front-left camera center 近似落在 mesh floor 下面，z 约为 `-0.10`，所以 z-buffer depth 主要投到图像上半区，中间墙面和地面覆盖很差。
- 旧流程直接用 mesh vertices 裁 depth PLY，点容易落在边、上下区域，墙面/地面中间没有足够监督。

新的推荐流程改为使用 NuRec `3dgrt/last.usdz` 中的 `rig_trajectories.json` 标定，把 front-left `T_sensor_rig` 按 `camera-to-ego` 使用，并保持 `camera-frame-axis=as-is`。修正后 camera center 高度约为 `z=0.3366`，更符合真实相机安装高度。

当前推荐 smoke 数据集：

```text
/media/chatsign/data-002/gs-sdf/runtime/datasets/nurec_galileo_frontleft_3dgrtcalib_surface20k_z4_60f_gssdf_colmap
```

验证结论：

- depth PLY 由 mesh triangle surface sampling + z-buffer 生成，每帧 `20000` points。
- 旧 stride8 converter 的 bottom-third 覆盖约 `0.06%`，center/floor band 约 `8%`。
- 新 3DGRT calibration 数据集的 top/mid/bottom 覆盖约 `27.8% / 54.1% / 18.1%`，center/floor band 约 `49.7%`。
- 这说明问题核心确实是 geometry supervision，不是 WebUI 或 renderer 本身。

已经完成的一次 GS-SDF 训练：

```text
/media/chatsign/data-002/gs-sdf/runtime/output/2026-04-27-07-06-03_nurec_galileo_frontleft_3dgrtcalib_surface20k_z4_60f_gssdf_colmap_gssdf_colmap.native_3dgrtcalib_gs4000.yaml
```

结果：

- final PSNR: `28.8271`
- Gaussian vertices: `120248`
- `mesh_gs_.ply`: `72026` vertices / `63426` faces
- `model/gs.ply`: about `28.38 MB`
- 视觉上已经比旧的 blob/ellipsoid failure 好，能形成连续 floor/wall，但仍然偏 smooth。

对比旧结果：

- old 114-frame smoke: PSNR `25.7299`
- old stride8 283-frame run: PSNR `21.9054`
- new 3DGRT calibration 60-frame run: PSNR `28.8271`

已经完成 114-frame 正式转换与训练：

```text
/media/chatsign/data-002/gs-sdf/runtime/datasets/nurec_galileo_frontleft_auto3dgrt_surface20k_z4_114f_gssdf_colmap
/media/chatsign/data-002/gs-sdf/runtime/output/2026-04-27-07-34-49_nurec_galileo_frontleft_auto3dgrt_surface20k_z4_114f_gssdf_colmap_gssdf_colmap.native_3dgrt_auto_gs4000.yaml
```

结果：

- frames: `114`
- calibration source: `3dgrt/last.usdz`
- depth points per frame: min/max/mean `20000 / 20000 / 20000`
- projected top/mid/bottom coverage: `27.35% / 55.09% / 17.56%`
- center/floor band coverage: `52.54%`
- final PSNR: `26.9`
- `model/gs.ply`: `121436` Gaussian vertices, about `28 MB`
- `mesh_gs_.ply`: `70057` vertices / `57518` faces

GS-SDF eval/video tail is now fixed for this run.

Root cause:

- Native GS-SDF looked for helper scripts under `/media/chatsign/data-002/gs-sdf/runtime/eval`, but the actual upstream eval directory was under `/media/chatsign/data-002/real2sim/models/GS-SDF/eval`.
- The 114-frame NuRec adapter run exported rendered frames under `gs_log/{train,test}/color/renders`, but `gs_log/{train,test}/color/gt` was empty, so official metrics/video scripts had no paired RGB images.

Fix applied on `gpu-worker`:

```bash
ln -s /media/chatsign/data-002/real2sim/models/GS-SDF/eval \
  /media/chatsign/data-002/gs-sdf/runtime/eval

/media/chatsign/data-002/gs-sdf/scripts/prepare_gssdf_eval_assets.py \
  --output-root /media/chatsign/data-002/gs-sdf/runtime/output/2026-04-27-07-34-49_nurec_galileo_frontleft_auto3dgrt_surface20k_z4_114f_gssdf_colmap_gssdf_colmap.native_3dgrt_auto_gs4000.yaml \
  --source-image-dir /media/chatsign/data-002/datasets/nurec/nova_carter-galileo/raw_images_front_left/rosbag_mapping_data/front_stereo_camera_left \
  --splits train test \
  --mode symlink \
  --force
```

Generated official eval artifacts:

- `gs_log/train/render_eval.json`
- `gs_log/test/render_eval.json`
- `gs_log/evaluation_results.json`
- `gs_log/train/color/video.mp4`: `100` frames, `10.00s`, about `20 MB`
- `gs_log/test/color/video.mp4`: `14` frames, `1.40s`, about `3.1 MB`

Official RGB-vs-Gaussian metrics:

- train: SSIM `0.8175`, PSNR `21.7154`, LPIPS `0.3913`
- test: SSIM `0.7881`, PSNR `18.6798`, LPIPS `0.4124`

Note: these official metrics are lower than the training-log final PSNR because they compare the saved rendered frames against full RGB pairs across the exported train/test splits. The training-log PSNR is the native trainer's internal sampled/logged value.

下一步优先级：

1. 用同样 3DGRT calibration 跑更长序列，例如 frame-stride `8` 或 `4`。
2. 把 corrected depth + NuRec mesh/SDF prior 接回 HMR3D Gaussian builder。
3. 如果长序列仍糊，再查 RGB/pose timestamp sync，而不是继续盲调 splat 半径。

## Repository Layout

- [cuVSLAM](/home/nyu/Codespace/CVPR/cuVSLAM)
  - NVIDIA cuVSLAM experiment tree
  - KITTI example data and trajectory outputs
  - Jetson Python env: [cuVSLAM/.venv-jetson](/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson)

- [HMR3D](/home/nyu/Codespace/CVPR/HMR3D)
  - current main runtime and experiments
  - core code: [HMR3D/nuc/src/nuc_runtime](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime)
  - scripts: [HMR3D/nuc/scripts](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts)
  - configs: [HMR3D/nuc/configs](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs)
  - historical outputs: [HMR3D/nuc_output](/home/nyu/Codespace/CVPR/HMR3D/nuc_output)

- [nuc_output](/home/nyu/Codespace/CVPR/nuc_output)
  - newer local benchmark outputs and generated WebUI pages

- [third_party_research](/home/nyu/Codespace/CVPR/third_party_research)
  - LingBot-Map
  - Gaussian-SLAM
  - GSFusion
  - FAST-LIVO / FAST-LIVO2 references

## Core Code

Tracking / adapter:

- [cuvslam_adapter.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/cuvslam_adapter.py)

Memory lifecycle:

- [memory_router.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/memory_router.py)
- [policies.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/policies.py)
- [models.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/models.py)

Gaussian:

- [gaussian_builder.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_builder.py)
- [gaussian_renderer.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_renderer.py)
- [local_tsdf.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/local_tsdf.py)

Key scripts:

- [run_gaussian_render_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_gaussian_render_benchmark.py)
- [run_cuvslam_kitti_memory.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py)
- [run_cuvslam_lingbot_recon.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_lingbot_recon.py)
- [prepare_lingbot_dense_job.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_lingbot_dense_job.py)
- [run_lingbot_dense_from_manifest.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_dense_from_manifest.py)
- [run_lingbot_dense_remote.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_dense_remote.py)
- [export_lingbot_dense_geometry.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/export_lingbot_dense_geometry.py)
- [run_lingbot_gaussian_init.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_gaussian_init.py)
- [run_lingbot_gaussian_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_gaussian_benchmark.py)
- [generate_gaussian_timeline_demo.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/generate_gaussian_timeline_demo.py)
- [generate_quad_run_compare_viewer.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/generate_quad_run_compare_viewer.py)
- [open_gaussian_web.sh](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/open_gaussian_web.sh)

## Architecture

Current intended split:

```text
camera / KITTI frames
  -> cuVSLAM
      pose, tracking, keyframe rhythm
  -> HMR3D MemoryRouter
      active/archive/retrieve/recover/merge
  -> Gaussian Builder
      active Gaussian submap, archived Gaussian handles
  -> Renderer / WebUI
      current-view visualization and debugging
```

Future intended split:

```text
camera / KITTI frames
  -> local cuVSLAM
      low-latency tracking, pose backbone
  -> async dense geometry front-end
      LingBot / Mem3R / TTT3R / DUSt3R-style pointmaps
  -> HMR3D memory-native submaps
      active/archive/recover lifecycle
  -> Gaussian backend
      standard 3DGS params + gsplat refinement
  -> robot UI / viewer
      delayed high-quality map render
```

The important distinction:

- `cuVSLAM` is the pose/tracking backbone.
- `dense geometry` should come from a reconstruction front-end.
- `Gaussian` should become a learned or optimized renderable representation.
- `HMR3D` should keep the memory lifecycle identity of the project.

## Current Metrics

Representative runs:

| Run | Purpose | PSNR | SSIM | FPS | Max Points | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `realtime_budget` | playable WebUI preview | 13.1584 | 0.45105 | 0.986 | 7200 | looks very blurry; near-1-FPS budget |
| `surfel_denseviewer` | current quality-ish ceiling | 14.6421 | 0.45309 | 0.198 | 150144 | better coverage, far too slow |
| `v15_fast_stable` | fast/stable experimental branch | 11.3769 | 0.38815 | 1.023 | 3057 | stable but visually weak |
| `LingBot balanced` | 2-frame LingBot Gaussian init test | 10.3089 | 0.22699 | 1.768 | 1100 | balanced frame budget, still weak |
| `NuRec GS-SDF 3DGRT calib 60f` | corrected mesh-depth geometry prior | 28.8271 | - | offline train | 120248 | current best NuRec/GS-SDF result; continuous floor/walls, still smooth |
| `NuRec GS-SDF auto3dgrt 114f` | formal 114-frame corrected NuRec run | 26.9 | - | offline train | 121436 | converter now auto-prefers `last.usdz`; stronger coverage, lower PSNR than 60f |

Interpretation:

- The best current image quality is still far below StreamGS-style output.
- Adding points helps but makes rendering too slow.
- Optimizing only point budgets/radius/opacity is not enough.
- The new NuRec/GS-SDF result confirms that good geometry prior matters much more than tuning the old lightweight renderer.
- The next quality jump needs better dense geometry and real Gaussian optimization.

## Demos

The current web server usually serves [nuc_output](/home/nyu/Codespace/CVPR/nuc_output) on port `19090`.

Tunnel from your local machine:

```bash
ssh -L 19090:127.0.0.1:19090 nyu@nuc-6913-frp
```

Open:

- realtime timeline:
  `http://127.0.0.1:19090/gaussian_timeline_demo_realtime_budget.html?v=2`
- quality-ish timeline:
  `http://127.0.0.1:19090/gaussian_timeline_demo_surfel_denseviewer.html?v=1`
- LingBot old vs balanced compare:
  `http://127.0.0.1:19090/gaussian_quad_compare_lingbot_balanced_v9_v12_nocache.html`

If the server is not running:

```bash
/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/open_gaussian_web.sh \
  /home/nyu/Codespace/CVPR/nuc_output/gaussian_timeline_demo_realtime_budget.html \
  19090
```

## LingBot-Map Status

LingBot-Map is useful, but the role should be clear.

What works:

- `LingBot predictions -> npz/json export`
- `LingBot depth + intrinsic + cuVSLAM poses -> Gaussian handle`
- 2-frame benchmark and compare viewer
- balanced Gaussian anchor selection across LingBot frames

Important outputs:

- [kitti06_cuvslam_lingbot_cmp](/home/nyu/Codespace/CVPR/nuc_output/kitti06_cuvslam_lingbot_cmp)
- [kitti06_cuvslam_lingbot_gaussian_cmp_balanced](/home/nyu/Codespace/CVPR/nuc_output/kitti06_cuvslam_lingbot_gaussian_cmp_balanced)
- [kitti06_render_benchmark_lingbot_gaussian_cmp_balanced](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_lingbot_gaussian_cmp_balanced)

Known problems:

- On-device LingBot inference is memory-heavy.
- 6-frame local run already approaches OOM.
- 120-frame local run had to be stopped by memory guard.
- LingBot world coordinates need careful alignment to cuVSLAM coordinates.
- The first aligned version looked good on frame 0 but failed on frame 25 because most Gaussian budget was consumed by frame 0.
- Balanced anchor selection fixed the 830/270 frame imbalance to roughly 550/550, improving frame 25 but lowering frame 0.

Current judgment:

> LingBot-Map is a promising dense geometry teacher/front-end, not a drop-in realtime loop on this hardware.

## Mem3R / TTT3R / StreamGS Judgment

These methods are relevant, but not all in the same role.

- `LingBot-Map`
  - best short-term candidate because we already have an adapter
  - useful as dense geometry front-end
  - currently too heavy locally

- `Mem3R`
  - conceptually very close to our desired memory-native geometry front-end
  - hybrid memory, streaming reconstruction, fixed-size state
  - likely still heavy for Orin Nano unless optimized/quantized

- `TTT3R`
  - useful as test-time adaptation / streaming geometry improvement
  - likely more of an algorithmic component than a full replacement for HMR3D

- `StreamGS`
  - closest to the render quality target
  - learned Gaussian generation and merging, not just geometry
  - should be treated as high-quality teacher/baseline first
  - original-style real-time does not automatically mean real-time on our small local device

Current engineering assumption:

> Local robot should run `cuVSLAM` and memory bookkeeping. Dense geometry / learned Gaussian generation should initially run asynchronously on a remote GPU or a larger onboard GPU.

## Why cuVSLAM Alone Is Not Enough

`cuVSLAM` is very useful, but for a different objective.

It provides:

- low-latency pose
- tracking stability
- keyframe rhythm
- visual landmarks for localization

Gaussian rendering needs:

- dense depth or pointmaps
- surface normals
- visibility
- scale/rotation/opacity
- color or SH coefficients
- multi-view photometric consistency

Visual landmarks are not enough because they are sparse and selected for tracking, not for covering visible surfaces. They are good anchors for localization, but poor raw material for high-quality rendering.

## Next Plan

### Phase 1: Stop Over-optimizing the Current Heuristic Renderer

Current CPU/surfel renderer is useful for debugging memory lifecycle, but not enough for high-quality output.

Do:

- keep it as a diagnostic viewer
- keep timeline demos for presentations and debugging
- avoid spending too much time tuning radius/opacity/budget only

Success condition:

- viewer remains usable
- metrics remain reproducible
- no claim that this is StreamGS-quality rendering

### Phase 2: Remote GPU Dense Geometry Front-End

Build an async dense geometry service.

Input:

- selected keyframe window
- image paths or compressed frames
- optional cuVSLAM poses

Output:

- depth map
- point map
- confidence
- intrinsic/extrinsic
- optional normals
- metadata for alignment

Initial candidate:

- LingBot-Map, because adapter exists.

Current first-pass implementation:

- [prepare_lingbot_dense_job.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_lingbot_dense_job.py) selects cuVSLAM keyframes and writes a self-contained job manifest.
- [run_lingbot_dense_remote.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_dense_remote.py) uploads the job to a remote GPU host, runs LingBot, and downloads predictions.
- [run_lingbot_dense_from_manifest.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_dense_from_manifest.py) is the remote-side runner.
- [export_lingbot_dense_geometry.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/export_lingbot_dense_geometry.py) converts LingBot `depth/depth_conf/intrinsic` plus cuVSLAM poses into a normalized dense geometry `.npz`.
- [run_lingbot_gaussian_init.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_gaussian_init.py) can now initialize Gaussian handles directly from the dense geometry `.npz`.

Next candidate:

- Mem3R, because it is a better conceptual match for long streaming memory.

Success condition:

- can process more than 2 frames without local OOM
- outputs align to cuVSLAM coordinates
- can inject dense geometry into active HMR3D submap

### Phase 3: Standard Gaussian Representation

Move from current surfel-like state to more standard 3DGS state:

- `xyz`
- `rotation`
- `scale`
- `opacity`
- `SH/color`
- source frame / confidence / memory provenance

Use dense geometry to initialize `xyz`, normals, and scale.

Success condition:

- Gaussian data can be consumed by `gsplat`
- output quality is meaningfully above current surfel renderer
- data can still be archived/recovered by HMR3D

### Phase 4: Differentiable Local Refinement

Use `gsplat` or a similar backend for local refinement.

Optimize:

- color / SH
- opacity
- scale
- rotation
- limited xyz correction

Use losses:

- RGB
- SSIM or perceptual proxy
- depth consistency
- opacity sparsity
- temporal consistency

Success condition:

- active submap improves after background optimization
- frontend tracking is not blocked
- map quality improves without losing HMR3D lifecycle behavior

### Phase 5: Memory-Native Gaussian System

Once the Gaussian backend is stronger, make HMR3D matter again at representation level.

Add:

- active Gaussian lifecycle policy
- archive compression
- coarse/full Gaussian bank
- recover alignment
- cross-submap merge
- confidence-aware memory routing

Success condition:

- revisiting an old region recovers useful Gaussian state
- recovered Gaussian improves render or update speed
- memory lifecycle gives measurable benefit beyond a single active model

## Immediate TODO

Most useful next tasks:

1. Use the new remote LingBot dense job path on a real GPU host.
2. Run LingBot on a longer KITTI 06 window without local OOM.
3. Validate the dense geometry schema on more than 2 keyframes.
4. Improve robust alignment and scale between LingBot depth and cuVSLAM coordinates.
5. Replace current heuristic Gaussian init with dense-geometry-derived Gaussian init.
6. Add a real `gsplat` local optimization loop.
7. Benchmark against StreamGS-style output as a teacher/baseline.
8. Keep HMR3D memory lifecycle as the research identity, not the renderer itself.

## Useful Commands

Run current realtime-ish benchmark:

```bash
source HMR3D/nuc/scripts/use_jetson_gpu_backend.sh
cuVSLAM/.venv-jetson/bin/python HMR3D/nuc/scripts/run_gaussian_render_benchmark.py \
  --sequence-path cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --config HMR3D/nuc/configs/kitti06_v4_realtime_budget.yaml \
  --save-images
```

Generate a timeline demo:

```bash
python HMR3D/nuc/scripts/generate_gaussian_timeline_demo.py \
  --run-dir nuc_output/hmr3d_nuc_output/kitti06_render_benchmark_realtime_budget \
  --trajectory-path cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --output-html nuc_output/gaussian_timeline_demo_realtime_budget.html \
  --title "KITTI 06 Gaussian Walkthrough" \
  --label "realtime budget" \
  --root-relative
```

Open web viewer:

```bash
HMR3D/nuc/scripts/open_gaussian_web.sh \
  nuc_output/gaussian_timeline_demo_realtime_budget.html \
  19090
```

Launch remote Isaac Gaussian Online WebUI:

This is the preferred path when the Gaussian/Isaac side runs on `gpu-worker` and the browser is local.

One-click launch from this workspace:

```bash
HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh
```

This starts the remote stack on `gpu-worker`, then opens a local SSH tunnel and prints:

```text
http://127.0.0.1:55173/?scene=/scenes/isaac-gaussian-online/manifest.json&mode=hifi
```

Default Isaac scene for this one-click path is now our converted NuRec/GS-SDF collision USD, not the Isaac built-in warehouse:

```text
/media/chatsign/data-002/isaac/nav-mvp/assets/nurec_galileo_gssdf_mesh_collision.usd
```

That USD was converted from:

```text
/media/chatsign/data-002/gs-sdf/runtime/output/2026-04-27-07-34-49_nurec_galileo_frontleft_auto3dgrt_surface20k_z4_114f_gssdf_colmap_gssdf_colmap.native_3dgrt_auto_gs4000.yaml/mesh_gs_.ply
```

Conversion summary:

- input mesh: `70057` vertices / `57518` faces
- output USD: about `1.1 MB`
- collision approximation: `meshSimplification`
- bounds min/max: `[-8.277, -2.362, -1.748]` to `[6.823, 8.538, 1.952]`

Useful one-click variants:

```bash
# Start WebUI stack only, without launching Isaac.
HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh --no-isaac

# Start the stack but do not keep a local SSH tunnel open.
HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh --no-tunnel

# Force-restart managed remote ports.
HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh --restart

# Stop the remote stack.
HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh --stop

# Use an Isaac built-in scene instead of the converted NuRec/GS-SDF USD.
HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh \
  --use-built-in-scene \
  --isaac-scene full-warehouse

# Use another converted USD scene.
HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh \
  --scene-usd /path/to/scene.usd
```

If the WebUI opens on an empty grid, check the active tab first:

- `Hifi` shows the Isaac Gaussian Bridge MJPEG/RGB stream and is the best default smoke-view.
- `Gs` shows the browser Gaussian layer.
- `Live` is the RViz-style working view; if it says `ROS connecting` it may still show only the grid even while the HTTP bridge is healthy.

Smoke verification on `2026-04-27`:

- remote ports online: WebUI `55173`, bridge `8890`, mapper `8891`, world-nav `8892`
- Isaac publisher launched with `--scene-usd /media/chatsign/data-002/isaac/nav-mvp/assets/nurec_galileo_gssdf_mesh_collision.usd`
- bridge after startup: `ready=true`, RGB/depth/semantic ready, trajectory `105` frames
- mapper after startup: `gaussianReady=true`, live point count `3007`

Service split:

- renderer service: `8876`
- mapper service: `8891`
- Isaac Gaussian online bridge: `8890`
- world navigation module: `8892`
- WebUI dev server: `55173`

On `gpu-worker`, start the online bridge:

```bash
RENDERER_URL=http://127.0.0.1:8876 \
MAPPER_URL=http://127.0.0.1:8891 \
bash /home/chatsign/gs-sdf/scripts/launch_isaac_gaussian_online_bridge.sh 8890
```

On `gpu-worker`, start the world navigation module:

```bash
export DISPLAY=:0
export QT_X11_NO_MITSHM=1

WORLD_NAV_VIEWER=1 \
WORLD_NAV_HOLD_OPEN_SEC=60 \
WORLD_NAV_EPISODE_HOLD_SEC=2 \
WORLD_NAV_STEP_SLEEP=0.02 \
WORLD_NAV_BRIDGE_URL=http://127.0.0.1:8890 \
WORLD_NAV_DEVICE=cuda:0 \
bash /home/chatsign/gs-sdf/scripts/launch_world_nav_module.sh 8892
```

On `gpu-worker`, start the WebUI:

```bash
WEB_PORT=55173 \
WEB_SCENE=/scenes/isaac-gaussian-online/manifest.json \
WEB_MODE=hifi \
WEB_ISAAC_GAUSSIAN_ONLINE_PORT=8890 \
WEB_ISAAC_GAUSSIAN_MAPPER_PORT=8891 \
WEB_WORLD_NAV_PORT=8892 \
bash /home/chatsign/gs-sdf/scripts/launch_web_ui_dev.sh
```

If the browser runs on the same remote desktop, open:

```text
http://localhost:55173/?scene=/scenes/isaac-gaussian-online/manifest.json&mode=hifi
```

If the browser runs on the local machine, keep this tunnel open locally:

```bash
ssh -N \
  -L 55173:127.0.0.1:55173 \
  -L 8876:127.0.0.1:8876 \
  -L 8890:127.0.0.1:8890 \
  -L 8891:127.0.0.1:8891 \
  -L 8892:127.0.0.1:8892 \
  gpu-worker
```

Then open locally:

```text
http://127.0.0.1:55173/?scene=/scenes/isaac-gaussian-online/manifest.json&mode=hifi
```

Important:

- This does connect the remote Isaac demo interface into the WebUI Gaussian page.
- The WebUI page can load as long as the dev server is up.
- The actual high-quality Gaussian stream also needs the renderer on `8876` and mapper on `8891` to be alive; otherwise the bridge page may load but have no useful Gaussian content.

Prepare a LingBot dense job locally:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/prepare_lingbot_dense_job.py \
  --sequence-path cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --config HMR3D/nuc/configs/kitti06_v9_structured_init.yaml \
  --output-dir nuc_output/lingbot_dense_jobs/kitti06_win8 \
  --window-keyframes 8
```

Run that job on a remote GPU:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/run_lingbot_dense_remote.py \
  --remote USER@GPU_HOST \
  --job-dir nuc_output/lingbot_dense_jobs/kitti06_win8 \
  --remote-root /tmp/lingbot_dense_jobs \
  --remote-repo-root /path/to/CVPR \
  --model-path /path/to/lingbot-map.pt \
  --remote-python /path/to/python \
  --lingbot-map-root /path/to/lingbot-map
```

Export normalized dense geometry:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/export_lingbot_dense_geometry.py \
  --predictions-npz nuc_output/lingbot_dense_jobs/kitti06_win8/lingbot_output/lingbot_predictions.npz \
  --summary-json nuc_output/lingbot_dense_jobs/kitti06_win8/lingbot_output/lingbot_summary.json \
  --image-root nuc_output/lingbot_dense_jobs/kitti06_win8 \
  --output-dir nuc_output/lingbot_dense_jobs/kitti06_win8/dense_geometry \
  --stride 4 \
  --min-conf 1.0
```

Run LingBot Gaussian init:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/run_lingbot_gaussian_init.py \
  --predictions-npz nuc_output/kitti06_cuvslam_lingbot_cmp/lingbot_predictions.npz \
  --summary-json nuc_output/kitti06_cuvslam_lingbot_cmp/lingbot_summary.json \
  --output-dir nuc_output/kitti06_cuvslam_lingbot_gaussian_cmp_balanced \
  --submap-id 9025 \
  --config HMR3D/nuc/configs/kitti06_v9_structured_init.yaml
```

Run Gaussian init from normalized dense geometry:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/run_lingbot_gaussian_init.py \
  --dense-geometry-npz nuc_output/lingbot_dense_jobs/kitti06_win8/dense_geometry/lingbot_dense_geometry.npz \
  --output-dir nuc_output/kitti06_lingbot_dense_gaussian \
  --submap-id 9100 \
  --config HMR3D/nuc/configs/kitti06_v9_structured_init.yaml
```

## Working Conclusion

The project is not blocked because the WebUI looks bad. The WebUI is showing the truth.

Current system has proven:

- cuVSLAM can feed HMR3D.
- HMR3D can manage Gaussian submap lifecycle.
- Gaussian handles can be archived, recovered, and visualized.
- LingBot-style dense geometry can be exported and injected.

Current system has not yet solved:

- high-quality dense geometry
- standard 3DGS parameter generation
- real differentiable Gaussian refinement
- embedded real-time foundation-model inference

The next real step is therefore:

> Keep cuVSLAM + HMR3D as the system backbone, and replace the weak heuristic Gaussian front-end with an async dense-geometry / learned-Gaussian backend.
