# cuVSLAM Adapter

这个最小 adapter 不是直接把 `cuVSLAM` 改造成 HMR3D 后端，而是先把已经跑出来的：

- `trajectory_tum.txt`
- `image_0/*.png`

转换成 `nuc_runtime.models.TrackingOutput`，再送入 `MemoryRouter` 验证：

- `observe`
- `promote`
- `archive`
- `retrieve`
- `recover`

## 运行

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --output-dir /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_cuvslam
```

## 当前边界

- 这是离线 replay adapter，不是 live `cuVSLAM` callback 集成。
- 位姿来自 `trajectory_tum.txt`，局部特征与全局描述子从图像重算。
- `match_count / inlier_count / pixel_motion / is_keyframe` 目前是为 lifecycle 验证服务的近似量。
- 下一步若要变成正式 Jetson 方案，应把这个离线 adapter 替换成 live frontend adapter，并直接对接 `cuVSLAM` 的逐帧输出。

## v1 / v2 / v3 对比

基线版：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --frame-step 5 \
  --max-frames 400 \
  --output-dir /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_baseline_v1
```

v2 最小机制版：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py \
  --enable-v2 \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --frame-step 5 \
  --max-frames 400 \
  --output-dir /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v2_smoke
```

v3 生命周期增强版：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py \
  --config /home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v3_full.yaml \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt
```

并排比较：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/compare_memory_runs.py \
  --left /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_baseline_v1 \
  --right /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v2_smoke
```

`--enable-v2` 当前会打开三类最小机制：

- `write policy`
- `pose-anchor gate`
- `shadow recover`

`kitti06_v3_full.yaml` 在 v2 的基础上再打开：

- `scene-level hierarchical bank`
- `query/readout routing`
- `multi-candidate merge`
- `local adaptation`
- `support merge` 软候选融合

## 配置文件

- baseline: [kitti06_baseline_v1.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_baseline_v1.yaml:1)
- v2 tuned: [kitti06_v2_tuned.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v2_tuned.yaml:1)
- v3 full: [kitti06_v3_full.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v3_full.yaml:1)
- v4 gaussian: [kitti06_v4_gaussian.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v4_gaussian.yaml:1)

使用方式：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py \
  --config /home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v2_tuned.yaml \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt
```

## 可视化

单次运行时间线：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/plot_memory_run.py \
  --run-dir /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v2_tuned
```

两次运行对比：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/plot_memory_compare.py \
  --left /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_baseline_v1 \
  --right /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v2_tuned
```

## 当前建议看的结果

同条件、同入口、修正配置读取后的三版输出：

- baseline: [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_baseline_v1_cfgfix/summary.json:1)
- v2 tuned: [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v2_tuned_cfgfix/summary.json:1)
- v3 full: [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v3_full_cfgfix/summary.json:1)
- v4 gaussian smoke: [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_cfgfix/summary.json:1)
- v4 gaussian long replay: [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_800/summary.json:1)

关键变化：

- `baseline -> v2 tuned`：archive 从 `76` 降到 `24`，submap 不再碎片化
- `v2 tuned -> v3 full`：recover 从 `7` 升到 `8`
- `v2 tuned -> v3 full`：pose-anchor reject 从 `37` 降到 `29`
- `v2 tuned -> v3 full`：scene summaries 从 `24` 压缩到 `7`
- `v2 tuned -> v3 full`：`merge_events = 4`，`local_adapt_applied = 8`
- `v3 full -> v4 gaussian smoke`：核心 lifecycle 指标保持不退化，同时新增 `gaussian_archives = 7`、`gaussian_archived_points_total = 420`
- `v4 gaussian long replay`：`gaussian_archives = 33`、`gaussian_warmstart_requests = 8`、`gaussian_warmstart_points = 766`

对应图表：

- v3 时间线：[timeline.svg](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v3_full_cfgfix/timeline.svg:1)
- v2 vs v3：[kitti06_v2_tuned_cfgfix_vs_kitti06_v3_full_cfgfix.svg](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v2_tuned_cfgfix_vs_kitti06_v3_full_cfgfix.svg:1)
- baseline vs v3：[kitti06_baseline_v1_cfgfix_vs_kitti06_v3_full_cfgfix.svg](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_baseline_v1_cfgfix_vs_kitti06_v3_full_cfgfix.svg:1)

## Incremental Gaussian

这版不是完整 3DGS 训练器，而是最小 incremental Gaussian submap builder：

- 相邻关键帧用 ORB 匹配
- 利用已知 pose 与 `calib.txt` 三角化世界点
- 逐帧把新点加进 active Gaussian state
- archive 时导出 `gaussians.ply` 与 `gaussians.npz`
- recover 命中历史子图时，把历史 Gaussian handle 作为当前 active 的 warm start

400 帧 smoke：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py \
  --config /home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v4_gaussian.yaml \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --output-dir /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_cfgfix
```

更长 replay，用来观察 Gaussian warm start：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py \
  --config /home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v4_gaussian.yaml \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --max-frames 800 \
  --output-dir /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_800
```

示例导出文件：

- [gaussians.ply](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_cfgfix/gaussian_bank/submap_0020/gaussians.ply:1)
- [gaussians.npz](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_cfgfix/gaussian_bank/submap_0020/gaussians.npz:1)

## Live Gaussian Rerun

新增了一个 Rerun replay 脚本：

- [live_gaussian_rerun.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/live_gaussian_rerun.py:1)

它会在 replay 时分别显示：

- `active Gaussian`：绿色
- `archived Gaussian`：蓝色
- `recovered warm-start Gaussian`：黄色

录制 400 帧：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/live_gaussian_rerun.py \
  --config /home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v4_gaussian.yaml \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --max-frames 400 \
  --rerun-mode save \
  --rerun-file /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_400.rrd \
  --output-dir /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_400
```

录制 800 帧，能看到 warm-start：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/live_gaussian_rerun.py \
  --config /home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v4_gaussian.yaml \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --max-frames 800 \
  --rerun-mode save \
  --rerun-file /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_800.rrd \
  --output-dir /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_800
```

已经录好的文件：

- 400 帧：[kitti06_v4_gaussian_live_400.rrd](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_400.rrd:1)
- 800 帧：[kitti06_v4_gaussian_live_800.rrd](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_800.rrd:1)

如果 `rerun --web-viewer` 手动打开后只看到 welcome screen，不要直接开 `127.0.0.1:9090` 根页面。  
更稳的办法是直接生成一个绑定到具体 `.rrd` 的本地 HTML viewer：

- 生成器：[generate_rerun_web_viewer.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/generate_rerun_web_viewer.py:1)
- 启动脚本：[open_gaussian_web.sh](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/open_gaussian_web.sh:1)
- 已生成页面：[kitti06_v4_gaussian_live_800_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_800_viewer.html:1)

使用方式：

```bash
/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/open_gaussian_web.sh \
  /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_800_viewer.html \
  9090
```

然后手动打开：

```text
http://127.0.0.1:9090/kitti06_v4_gaussian_live_800_viewer.html
```

## Minimal Splat Renderer

新增了一个最小 `gsplat` 风格渲染后端：

- [gaussian_renderer.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_renderer.py:1)
- [run_gaussian_render_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_gaussian_render_benchmark.py:1)
- [kitti06_v4_render_benchmark.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v4_render_benchmark.yaml:1)

它做的是：

- 每帧 replay 后按当前 pose 做 Gaussian 投影
- 用简化 2D Gaussian splat 方式渲染当前视角
- 输出 render 图、GT 图、diff 图
- 统计 `PSNR / SSIM / update_ms / render_ms / approx_fps`

运行：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_gaussian_render_benchmark.py \
  --config /home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v4_render_benchmark.yaml \
  --sequence-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --output-dir /home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_cfgfix
```

当前 400 帧 replay 的结果：

- summary: [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_cfgfix/render_benchmark_summary.json:1)
- per-frame: [render_benchmark_frames.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_cfgfix/render_benchmark_frames.json:1)

关键统计：

- `mean_psnr = 7.14`
- `mean_ssim = 0.00321`
- `mean_update_ms = 38.602`
- `mean_render_ms = 7.605`
- `approx_fps = 21.642`
- `max_point_count = 456`
- `max_projected_points = 141`

示例图：

- [000300_triplet.png](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_cfgfix/renders/000300_triplet.png:1)
- [000350_triplet.png](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_cfgfix/renders/000350_triplet.png:1)

注意：这还是最小 splat renderer，不是标准 3DGS/gsplat 实现。当前质量主要受限于稀疏 ORB 三角化点，而不是渲染回路本身。

### Stereo-Dense Gaussian Seed

为了提高 render 质量，当前 builder 已增加 KITTI 双目 seed：

- 每个 keyframe 额外读取 `image_0` / `image_1`
- 用 `StereoSGBM` 生成稠密 disparity
- 规则采样有效 disparity 点并反投影到 3D
- 把这些 stereo 点直接写进 active Gaussian state

相关参数在：

- [config.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/config.py:1)
- [gaussian_builder.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_builder.py:1)

当前对比结果：

- baseline render: [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_cfgfix/render_benchmark_summary.json:1)
- stereo-dense render: [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_stereo/render_benchmark_summary.json:1)

核心变化：

- `mean_psnr: 7.14 -> 7.5731`
- `mean_ssim: 0.00321 -> 0.04024`
- `max_point_count: 456 -> 21120`
- `max_projected_points: 141 -> 3093`

代价：

- `mean_update_ms: 38.602 -> 137.103`
- `mean_render_ms: 7.605 -> 105.608`
- `approx_fps: 21.642 -> 4.12`

也就是说，这版已经验证了：

- 双目稠密 seed 能明显提升 Gaussian 输入密度
- render 质量确实比纯稀疏 ORB 三角化更好
- 但当前最小 renderer 还没有做足够的加速，实时性明显下降

示例图：

- [000000_triplet.png](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_stereo/renders/000000_triplet.png:1)
- [000200_triplet.png](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_stereo/renders/000200_triplet.png:1)
- [000350_triplet.png](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_stereo/renders/000350_triplet.png:1)
