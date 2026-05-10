# HMR3D / cuVSLAM / Streaming Gaussian 项目进度与实机落地计划

> Note: this document was merged from the older realtime Gaussian branch
> (`origin/codex/lingbot-dense-remote-gpu`). Some paths inside the document
> reflect that historical workspace. In the cleaned Mono2Sim-GS repository, use
> this as design/experiment context; use `README.md` and the `docs/RUNBOOK_*`
> files as the current entry points.

说明：

- 顶层总入口 README 现在在 [README.md](/home/nyu/Codespace/CVPR/README.md:1)
- 本文档保留为更细的阶段性状态说明与实机落地备忘

这份文档是当前本地实验线的正式状态说明，目标是把下面几件事讲清楚：

- 现在已经做到了什么
- 现在为什么还没有达到 `GS-SDF / GS Console` 那种效果
- 下一步最值得做什么
- 以后有实机之后，该怎么把这条线真正搬到机器人上
- 最终要怎样实现“实时跑高斯”的目标

本文面向当前本地工作区：

- `cuVSLAM`: [/home/nyu/Codespace/CVPR/cuVSLAM](/home/nyu/Codespace/CVPR/cuVSLAM)
- `HMR3D`: [/home/nyu/Codespace/CVPR/HMR3D](/home/nyu/Codespace/CVPR/HMR3D)

## 1. 项目目标

当前项目的核心目标，不是单独做一个高质量离线 Gaussian 重建器，而是要走一条更偏系统的路线：

1. `cuVSLAM` 负责 tracking / mapping 解耦后的位姿与地图前端。
2. `HMR3D` 负责长期记忆生命周期：
   `observe -> archive -> retrieve -> verify -> recover -> merge`
3. `active Gaussian submap` 负责当前局部区域的高斯表示。
4. 最终希望达到：
   “机器人边走边追踪、边维护 active Gaussian、边把历史区域 archive，并在回到旧区域时 recover 历史高斯子图。”

一句话说：

> 最终目标不是“只把图渲染好”，而是做一个能在线运行的、带长期记忆能力的实时 Gaussian 系统。

## 2. 当前系统结构

当前本地这条线已经形成了一个可运行闭环，但它还是实验版，不是最终产品。

### 2.1 当前数据流

当前主链是：

`KITTI offline replay`
-> `cuVSLAM trajectory_tum.txt`
-> `CUVSLAMOfflineKITTIAdapter`
-> `TrackingOutput`
-> `MemoryRouter`
-> `Active / Archived Submaps`
-> `Incremental Gaussian Builder`
-> `Minimal / Surfel-style Renderer`

对应代码主要在：

- [cuvslam_adapter.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/cuvslam_adapter.py:1)
- [memory_router.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/memory_router.py:1)
- [policies.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/policies.py:1)
- [gaussian_builder.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_builder.py:1)
- [gaussian_renderer.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_renderer.py:1)

### 2.2 当前系统里每层的职责

`cuVSLAM`

- 提供位姿主链
- 提供离线 replay 用的轨迹
- 提供和真实系统接轨时最重要的 tracking 基础

`HMR3D MemoryRouter`

- 决定什么时候开新 submap
- 决定什么时候 archive
- 决定如何 retrieve 历史子图
- 决定 recover 是否允许注入
- 决定如何 merge 多候选历史块

`Incremental Gaussian Builder`

- 把 active submap 的关键帧增量变成当前局部高斯表示
- archive 时把当前高斯子图冻结到磁盘
- recover 时把历史高斯 handle 作为 warm start 取回

`Renderer`

- 当前只是验证器，不是最终正式渲染器
- 目标是验证：
  当前视角下，高斯结果能不能形成连续视图

## 3. 已经完成的工作

### 3.1 cuVSLAM -> HMR3D adapter 已完成

当前已经把 `KITTI 06` 的 `trajectory_tum.txt + image_0` 接到了 HMR3D runtime：

- 入口脚本：
  [run_cuvslam_kitti_memory.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py:1)
- 说明文档：
  [CUVSLAM_ADAPTER.md](/home/nyu/Codespace/CVPR/HMR3D/docs/CUVSLAM_ADAPTER.md:1)

这一步已经完成了从“有 cuVSLAM 结果”到“能跑 memory lifecycle”的桥接。

### 3.2 HMR3D lifecycle v1 / v2 / v3 已完成

已经落地并验证的机制包括：

- `write-aware archive policy`
- `anchor retention`
- `pose-anchor gate`
- `shadow recover`
- `scene-level hierarchical bank`
- `query/readout routing`
- `multi-candidate merge`
- `local adaptation`

推荐查看的对比结果：

- baseline:
  [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_baseline_v1_cfgfix/summary.json:1)
- v2 tuned:
  [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v2_tuned_cfgfix/summary.json:1)
- v3 full:
  [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v3_full_cfgfix/summary.json:1)

关键趋势：

- `baseline -> v2 tuned`:
  archive 从 `76` 降到 `24`，submap 不再碎片化
- `v2 tuned -> v3 full`:
  recover 从 `7` 升到 `8`
- `v2 tuned -> v3 full`:
  pose-anchor reject 从 `37` 降到 `29`
- `v2 tuned -> v3 full`:
  `scene summaries` 从 `24` 压到 `7`
- `v2 tuned -> v3 full`:
  已出现 `merge_events` 与 `local_adapt_applied`

这说明：

> HMR3D 作为“长期记忆生命周期层”已经不是空概念，而是有可复现收益的。

### 3.3 Incremental Gaussian submap 已完成第一版

当前已经能做到：

- active submap 逐帧增量加入 Gaussian-like points
- archive 时导出 `gaussians.ply` 与 `gaussians.npz`
- recover 时把历史 Gaussian handle 作为 warm start 带回

相关配置：

- [kitti06_v4_gaussian.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v4_gaussian.yaml:1)

相关输出：

- smoke:
  [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_cfgfix/summary.json:1)
- long replay:
  [summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_800/summary.json:1)

示例导出：

- [gaussians.ply](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_cfgfix/gaussian_bank/submap_0020/gaussians.ply:1)
- [gaussians.npz](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_cfgfix/gaussian_bank/submap_0020/gaussians.npz:1)

### 3.4 可视化与网页播放页已完成

当前已经有三类查看方式：

#### 方式 A：Rerun 3D live viewer

- 录制文件：
  [kitti06_v4_gaussian_live_800.rrd](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_800.rrd:1)
- 绑定页面：
  [kitti06_v4_gaussian_live_800_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_800_viewer.html:1)

这适合看：

- 相机轨迹
- active / archived / warmstart Gaussian 点在 3D 空间里的位置

不适合看：

- 最终连续视角渲染质量

#### 方式 B：Triplet viewer

- 页面：
  [render_triplets_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_surfel_denseviewer/render_triplets_viewer.html:1)

适合看：

- `GT / Render / Diff` 三联图连续播放

#### 方式 C：GS_Console 风格 compare viewer

- 页面：
  [gsconsole_compare_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_surfel_denseviewer/gsconsole_compare_viewer.html:1)

适合看：

- 左边 `Playback RGB`
- 右边 `HMR3D Gaussian`
- 下方误差与当前帧统计

这类页面的价值是：

> 展示形式已经可以接近 `GS_Console` 的 compare demo，但它并不代表渲染质量已经接近 `GS-SDF`。

## 4. 当前渲染质量到底到了什么程度

这部分必须讲实话。

### 4.1 当前“连续视角感”已经有了

当前新一版 surfel-style 渲染已经不再只是“几个亮点球”，而是开始出现：

- 路面区域
- 建筑/树木的大块连续结构
- 当前视角下的整体形状感

推荐查看：

- [000000_triplet.png](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_surfel_denseviewer/renders/000000_triplet.png)
- [000100_triplet.png](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_surfel_denseviewer/renders/000100_triplet.png)
- [000150_triplet.png](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_surfel_denseviewer/renders/000150_triplet.png)

### 4.2 但它还远没有达到 GS-SDF / 正式 gsplat 级别

当前 summary：

- [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_surfel_denseviewer/render_benchmark_summary.json:1)

关键指标：

- `mean_psnr = 14.6421`
- `mean_ssim = 0.45309`
- `approx_fps = 0.198`

和更早的 stereo point-style renderer 比较：

- `PSNR: 7.5731 -> 14.6421`
- `SSIM: 0.04024 -> 0.45309`

这说明：

- 质量有明显提升
- 但仍是“实验性连续视角”
- 还不是“高保真、实时、可上机”的最终版

### 4.3 当前为什么还做不到 GS-SDF 那种效果

核心原因不是展示问题，而是渲染本体问题。

当前还缺：

1. 更高质量的几何初始化  
   现在主要是 stereo seed + 简化 surfel patch，不是成熟的 dense geometry pipeline。

2. 真正的 Gaussian 参数优化  
   当前只有近似的：
   `xyz / rgb / scale / opacity / axis_u / axis_v`
   还不是完整的高质量 Gaussian 状态。

3. active window 的局部 photometric optimization  
   现在主要还是“生成 + 渲染”，不是“持续优化高斯参数”。

4. 更高效的正式 splat renderer  
   当前 renderer 是验证器，目标是证明系统链路成立，不是最终最优实现。

一句话：

> 现在这版已经证明“这条系统路线可行”，但还没有走到“正式实时高质量 Gaussian 系统”。

## 5. 为什么不能直接拿 GS_Console / GS-SDF 顶上

这件事也要讲清楚。

### 5.1 GS_Console 是整套工作流，不是通用 viewer

它依赖的是固定后端：

## 5.5 `方案 B`: cuVSLAM + LingBot-style Reconstruction Baseline

为了避免继续只在当前轻量 Gaussian 原型上做微调，这一轮已经开始建立一条新的 3D 重建 baseline：

- `cuVSLAM` 继续负责前端 tracking / pose metadata
- `LingBot-Map` 先作为 streaming 3D reconstruction baseline
- 后续再考虑怎样把这条更强的局部几何底座接回 `HMR3D + Gaussian`

当前已完成：

- 参考仓准备：
  - [third_party_research/lingbot-map](/home/nyu/Codespace/CVPR/third_party_research/lingbot-map)
  - [third_party_research/Gaussian-SLAM](/home/nyu/Codespace/CVPR/third_party_research/Gaussian-SLAM)
  - [third_party_research/GSFusion](/home/nyu/Codespace/CVPR/third_party_research/GSFusion)
- `LingBot-Map` checkpoint 下载完成：
  - [lingbot-map.pt](/home/nyu/Codespace/CVPR/third_party_research/lingbot_cache/lingbot-map.pt)
- 针对 Jetson/Orin Nano 修复了 checkpoint 加载方式：
  - 在 [third_party_research/lingbot-map/demo.py](/home/nyu/Codespace/CVPR/third_party_research/lingbot-map/demo.py) 里改成 `CPU + mmap` 低峰值加载
- 补了 LingBot 导出与 `cuVSLAM + LingBot` 最小适配层：
  - [lingbot_adapter.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/lingbot_adapter.py:1)
  - [run_lingbot_smoke.sh](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_smoke.sh:1)
  - [run_lingbot_export.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_export.py:1)
  - [run_cuvslam_lingbot_recon.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_lingbot_recon.py:1)

当前已验证的结论：

- `LingBot-Map` 在 Orin Nano 上
  - 原始 GPU 路径会在 checkpoint load 时 OOM
  - 原始 CPU 路径会被系统以 `137` 杀掉
  - `CPU + mmap` 路径已经成功跑通最小 streaming smoke
- 最小 `cuVSLAM + LingBot` 导出已跑通：
  - 输出目录：[kitti06_cuvslam_lingbot_smoke](/home/nyu/Codespace/CVPR/nuc_output/kitti06_cuvslam_lingbot_smoke)
  - 预测：[lingbot_predictions.npz](/home/nyu/Codespace/CVPR/nuc_output/kitti06_cuvslam_lingbot_smoke/lingbot_predictions.npz)
  - 汇总：[lingbot_summary.json](/home/nyu/Codespace/CVPR/nuc_output/kitti06_cuvslam_lingbot_smoke/lingbot_summary.json)

这个最小 `方案 B` 目前做到了：

- 用 `cuVSLAM` 选取 keyframe window
- 在这个 window 上跑 `LingBot` reconstruction
- 把 `depth / depth_conf / world_points / world_points_conf / extrinsic / intrinsic` 落盘
- 同时保留 `cuVSLAM` 的：
  - `frame_indices`
  - `timestamps`
  - `poses`
  - `descriptors`

这说明：

> 现在已经有一条真正可运行的 `cuVSLAM pose metadata + LingBot reconstruction export` 基线，不再只是口头方案。

### 5.6 `LingBot recon -> Gaussian init` 正式对比

在 `方案 B` 跑通之后，这一轮继续把 `LingBot` 导出真正接到了现有 Gaussian 初始化链里，并且和当前代表线做了正式对比：

- `LingBot` 适配与导出：
  [lingbot_adapter.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/lingbot_adapter.py:1)
- `LingBot -> Gaussian handle`：
  [run_lingbot_gaussian_init.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_gaussian_init.py:1)
- `LingBot Gaussian benchmark`：
  [run_lingbot_gaussian_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_gaussian_benchmark.py:1)
- 四路对比页：
  [gaussian_quad_compare_lingbot_vs_v9_v11_v12.html](/home/nyu/Codespace/CVPR/nuc_output/gaussian_quad_compare_lingbot_vs_v9_v11_v12.html:1)

为了保证对比公平，这次对比统一取了共同帧 `frame_idx = [0, 25]`。最早第一版会出现黑屏，原因不是没有点，而是直接使用了 `LingBot world_points` 作为全局世界坐标，和当前 renderer 的世界坐标定义并不一致。后续已经修成：

- 优先使用 `LingBot depth + intrinsic`
- 再配合 `cuVSLAM pose`
- 重新反投影到我们自己的世界坐标

对齐后的输出：
- [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_lingbot_gaussian_cmp_aligned/render_benchmark_summary.json:1)
- [gaussian_quad_compare_lingbot_aligned_vs_v9_v11_v12.html](/home/nyu/Codespace/CVPR/nuc_output/gaussian_quad_compare_lingbot_aligned_vs_v9_v11_v12.html:1)
- [lingbot_aligned_vs_v9_v11_v12_common_frames_summary.json](/home/nyu/Codespace/CVPR/nuc_output/lingbot_aligned_vs_v9_v11_v12_common_frames_summary.json:1)

当前结果：

- `LingBot Recon -> Gaussian Init (Aligned)`
  - `PSNR 10.1836`
  - `SSIM 0.22636`
  - `render 541.297 ms`
- `v9 Structured Init`
  - `PSNR 11.0951`
  - `SSIM 0.27016`
  - `render 1002.211 ms`
- `v11 Structured Fast Refine`
  - `PSNR 10.7651`
  - `SSIM 0.23940`
  - `render 734.909 ms`
- `v12 Recover-aware Structured`
  - `PSNR 10.7600`
  - `SSIM 0.24019`
  - `render 735.170 ms`

这次结果说明：

- `LingBot` 的 streaming 3D reconstruction 输出已经可以直接进入 Gaussian 初始化链
- 黑屏问题来自最初坐标系未对齐，而不是 `LingBot` 没有输出 3D 数据
- 对齐后，这条线已经进入“可和现有 structured 线正常比较”的状态
- 它还没有自然带来更高画质
- 但它比 `v9 / v11 / v12` 更快，也说明 `LingBot` 分支更适合继续往“更强重建底座 + 轻量 refinement”方向深化

### 5.7 `v13`: LingBot Structured Init + Fast Refine + Recover-aware Refine

为了验证“`LingBot` 这条更强重建底座，完整吸收 `v11 / v12` 优势后会不会成为主线”，这轮又补了：

- [kitti06_v13_lingbot_structured_recover_refine.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v13_lingbot_structured_recover_refine.yaml:1)
- [run_lingbot_structured_refine_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_structured_refine_benchmark.py:1)

这条线做的事不是静态初始化，而是：

- 以 `LingBot aligned` 的 Gaussian 作为 structured init
- 把点源标成 structured source
- 打开 `fast refine`
- 同时打开 `recover-aware refine` 的权重调度

输出：

- [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v13_lingbot_structured_recover_refine/render_benchmark_summary.json:1)
- [gaussian_quad_compare_v13_lingbot_vs_aligned_v11_v12.html](/home/nyu/Codespace/CVPR/nuc_output/gaussian_quad_compare_v13_lingbot_vs_aligned_v11_v12.html:1)
- [v13_lingbot_vs_aligned_v11_v12_common_frames_summary.json](/home/nyu/Codespace/CVPR/nuc_output/v13_lingbot_vs_aligned_v11_v12_common_frames_summary.json:1)

共同帧结果：

- `v13 LingBot + Structured Fast/Recover Refine`
  - `PSNR 10.1562`
  - `SSIM 0.22120`
  - `update 179.740 ms`
  - `render 425.218 ms`
- `LingBot Recon -> Gaussian Init (Aligned)`
  - `PSNR 10.1836`
  - `SSIM 0.22636`
  - `render 541.297 ms`
- `v11 Structured Fast Refine`
  - `PSNR 10.7651`
  - `SSIM 0.23940`
  - `update 759.370 ms`
  - `render 734.909 ms`
- `v12 Recover-aware Structured`
  - `PSNR 10.7600`
  - `SSIM 0.24019`
  - `update 750.673 ms`
  - `render 735.170 ms`

这次结果说明：

- `v13` 确实吸收了 `v11 / v12` 的在线优化优势
- 但当前收益主要体现为**更轻、更快**
- 画质还没有超过现有 `v11 / v12`
- 这说明 `LingBot` 分支还需要继续做“更强重建底座 + 更轻局部 refinement”的组合优化，而不是简单初始化后直接替代现有 structured 主线

- `FAST-LIVO2`
- `GS-SDF`
- `rosbridge`
- `scene pack`
- `Vite UI`

仓库位置：

- [GS_Console](/home/nyu/Codespace/CVPR/GS_Console/README.md:1)

它不是“拿一个 `.npz` 就能直接渲染”的小工具。

### 5.2 GS-SDF 也不是 drop-in renderer

它要求的是它自己那套运行栈和数据组织方式，而不是我们当前这条：

- `cuVSLAM pose`
- `HMR3D memory`
- `incremental Gaussian submap`

所以：

- `GS_Console` 适合借展示思路
- `GS-SDF` 适合借最终后端目标
- 但都不能直接无缝替换当前链路

## 6. 当前最值得做的下一步

如果现在继续研发，我建议按下面优先级推进。

### 6.1 第一优先级：active window 局部优化

目标：

- 每个 keyframe 进来后
- 对当前 active Gaussian 跑 2 到 5 步小规模局部优化

优化变量优先级：

- `position`
- `scale`
- `opacity`
- `color`

目的：

- 让当前 render 更贴近真实图像
- 让 recover 后的 Gaussian submap 更容易重新对齐当前区域

这是最接近“真正 streaming GS” 的下一步。

本地落地状态：

- 已落地第一版 active-window 局部优化
- 每个 keyframe 后，会对当前 active Gaussian 做小步优化
- 当前实现优先更新：
  - `position`
  - `scale`
  - `opacity`
  - `color`
- 代码位置：
  [gaussian_builder.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_builder.py:214)
  和
  [memory_router.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/memory_router.py:438)

### 6.2 第二优先级：更稳定的几何输入

建议：

- 继续保留 stereo seed
- 但把规则网格采样进一步升级成更稳定的 depth/surfel 生成
- 降低噪声 patch，提升结构连续性

本地落地状态：

- 已从“纯规则网格 disparity 采样”升级成：
  - bilateral 预滤波
  - disparity 稳定性筛选
  - texture/consistency 过滤
  - `goodFeaturesToTrack` 稳定角点补充
  - 局部深度导数生成 surfel axes
- 代码位置：
  [gaussian_builder.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_builder.py:543)

### 6.3 第三优先级：更高效的 renderer

当前最大问题之一是速度太慢。

应该做：

- tile-based splatting
- active window 限域
- 更少的全图 patch 重复计算
- 更强的 depth-aware compositing

如果不解决效率问题，后面上实机会很吃力。

本地落地状态：

- 已加入：
  - internal render scaling
  - tile-based splatting
  - archived / warmstart 点数上限
  - depth window 剪枝
  - 更强的 active-window 限域
- 代码位置：
  [gaussian_renderer.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_renderer.py:105)

### 6.4 这三步落地后的当前结果

新一轮完整 benchmark 输出：

- [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_vnext/render_benchmark_summary.json:1)

主要指标：

- `mean_psnr = 13.1523`
- `mean_ssim = 0.45487`
- `mean_update_ms = 397.006`
- `mean_render_ms = 4735.579`
- `approx_fps = 0.195`

和上一版 surfel benchmark 相比：

- 画质：
  - `SSIM: 0.45309 -> 0.45487`
  - `PSNR: 14.6421 -> 13.1523`
- 渲染速度：
  - `render_ms: 4851.62 -> 4735.579`

这说明：

- 局部优化和更稳的几何输入让结构连续性保持住了
- renderer 加速已开始生效
- 但 `position/scale/opacity/color` 的当前启发式优化还不够成熟，PSNR 还有回落

推荐查看的新结果页面：

- triplet viewer:
  [render_triplets_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_vnext/render_triplets_viewer.html:1)
- GS_Console 风格 compare viewer:
  [gsconsole_compare_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_vnext/gsconsole_compare_viewer.html:1)

### 6.5 更激进的 realtime budget 结果

为了验证“只更新高误差区域 + coarse archived bank + view-aware scheduler”到底能把延迟压到什么程度，又补了一条更激进的实时预算配置：

- 配置：
  [kitti06_v4_realtime_budget.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v4_realtime_budget.yaml:1)
- 结果：
  [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_realtime_budget/render_benchmark_summary.json:1)

关键指标：

- `mean_psnr = 13.1584`
- `mean_ssim = 0.45105`
- `mean_update_ms = 265.91`
- `mean_render_ms = 748.4`
- `approx_fps = 0.986`

相对上一版 `vnext`：

- `mean_update_ms: 397.006 -> 265.91`
- `mean_render_ms: 4735.579 -> 748.4`
- `approx_fps: 0.195 -> 0.986`
- `mean_ssim: 0.45487 -> 0.45105`

这说明：

- 这三类机制确实能把延迟大幅压下去
- 质量没有彻底崩掉
- 但仍然没有达到严格意义上的实时

对应查看页面：

- triplet viewer:
  [render_triplets_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_realtime_budget/render_triplets_viewer.html:1)
- GS_Console 风格 compare viewer:
  [gsconsole_compare_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_realtime_budget/gsconsole_compare_viewer.html:1)

## 7. 有了实机之后，应该怎么上实机

这一部分是最重要的长期方案。

### 7.1 实机目标不是直接“开箱即用高斯”

真正稳妥的上机路径应该是分阶段的。

不要一上来就：

- 实时 tracking
- 实时 memory
- 实时 Gaussian update
- 实时渲染
- 实时 recover

全开。

这样非常容易不知道是哪里先坏。

### 7.2 建议的实机阶段划分

#### Phase A：只上 tracking

目标：

- 把 `cuVSLAM` 先在实机上稳定跑通
- 拿到实时 pose / keyframe / tracking status

输入建议：

- 优先双目 RGB
- 如果有 IMU 更好

产出：

- 实时相机位姿
- 实时关键帧事件
- 简单轨迹可视化

验收标准：

- 机器人连续移动时位姿不乱跳
- 回到旧区域能保持可接受定位稳定性

#### Phase B：上实时 HMR3D memory

目标：

- 把当前离线 adapter 换成 live frontend adapter
- 实时把 `TrackingOutput` 喂给 `MemoryRouter`

要做的事：

- 在 ROS2 或自定义 runtime 里加一个 `cuvslam_to_hmr3d_adapter` 节点
- 每帧把：
  - pose
  - image
  - descriptors
  - keyframe flag
  推给 HMR3D

验收标准：

- 机器人跑起来时，submap 能正常创建/归档
- retrieve / recover 能在真实回访时触发

#### Phase C：只上 active Gaussian，不做重渲染展示

目标：

- 在线维护 active Gaussian state
- archive 时冻结到磁盘
- recover 时把历史高斯取回

但此时先不要求：

- 高质量当前视角 render

验收标准：

- 不拖垮 tracking
- active submap 生命周期稳定
- Gaussian archive / warmstart 正常发生

#### Phase D：上轻量在线渲染

目标：

- 在机器人端或上位机端做简化实时当前视角 render

这一步优先看：

- latency
- 稳定性
- 是否能形成连续视角感

不是优先看：

- 是否已经达到 GS-SDF 级别画质

#### Phase E：再追求高质量 GS back-end

这一步才适合考虑：

- 更重的优化
- 更正式的 splat renderer
- 更高质量 compare demo

### 7.3 实机软件架构建议

推荐 ROS2 架构如下：

`camera driver`
-> `image rectification`
-> `cuVSLAM frontend node`
-> `hmr3d_memory node`
-> `active_gaussian node`
-> `viewer / logger / map saver`

各节点职责：

`cuVSLAM frontend node`

- 输入相机图像
- 输出 pose、tracking quality、keyframe event

`hmr3d_memory node`

- 输入 tracking outputs
- 管理 archive / retrieve / recover / merge

`active_gaussian node`

- 输入当前 active submap 关键帧
- 增量更新 active Gaussian
- archive 时存盘
- recover 时 warm start

`viewer / logger`

- 可选在机器人本机或远端上位机
- 用于监控当前运行状态

### 7.4 实机硬件建议

建议最小配置：

- 双目 RGB 相机
- 同步时间戳
- 尽量有 IMU
- 一台能稳定跑 `cuVSLAM` 的边缘 GPU 设备

更稳妥的配置：

- 双目 + IMU
- 单独的数据记录链
- 远端上位机做可视化，不把显示全部放在机载端

### 7.5 上实机时最重要的工程原则

1. tracking 优先级永远高于高斯渲染  
   先保证位姿稳定，再谈漂亮画面。

2. memory 优先级高于高质量 render  
   先保证 archive / recover 真能在真实世界里工作。

3. active Gaussian 只维护局部窗口  
   不要一开始就全局高斯地图。

4. 可视化最好尽量远端化  
   机载端做核心计算，上位机做播放和查看。

## 8. 最终“实时跑高斯”的目标应该怎么定义

最终目标不能只写成“看起来像 GS-SDF”。

更合理的最终目标定义应该是：

### 8.1 系统目标

- 机器人运动时，tracking 稳定
- 当前 active submap 持续更新
- 历史 submap 可 archive
- 回到旧区域时可 retrieve + recover
- 当前视角能持续输出可接受的 Gaussian render

### 8.2 性能目标

可以分阶段：

第一阶段目标：

- `tracking`: 实时
- `memory`: 实时
- `active Gaussian update`: 接近实时
- `render`: 低帧率但连续

第二阶段目标：

- `tracking + memory + gaussian update`: 全实时
- `render`: 接近实时

第三阶段目标：

- 在不破坏 tracking 的前提下，追求更高保真当前视角 render

### 8.3 研究目标

最终真正有论文价值的地方不只是“画面像不像”，而是：

- `cuVSLAM + HMR3D + active Gaussian` 这条组合路线是否可行
- 长期记忆是否真的帮助了 streaming Gaussian
- recover 是否让在线高斯系统更稳定、更可持续

## 9. 当前最推荐的执行路线

如果继续推进，我建议严格按这个顺序：

1. 保持 `cuVSLAM` 作为 tracking 主线
2. 保持 `HMR3D` 作为 memory lifecycle 主线
3. 继续把 active Gaussian 限定在局部 submap
4. 先做 active window 的局部优化
5. 再做更高效 renderer
6. 再考虑接近正式 gsplat / GS-SDF 的高质量后端
7. 有实机后按 `tracking -> memory -> gaussian -> render` 分阶段上机

## 10. 当前结论

### 10.1 新一轮参考式改进结果

为减少“只靠 heuristic 堆点”的问题，当前代码已经补进了一轮更接近现有 streaming GS 方法的机制：

- `quadtree-like stereo seed`
- `stereo depth fusion`
- `depth + color active-window optimization`
- 更强的 `surface-aware depth compositing`

相关配置和结果：

- 质量向配置：
  [kitti06_v5_quality_fused.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v5_quality_fused.yaml)
- 微调后的质量向配置：
  [kitti06_v5_quality_tuned.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v5_quality_tuned.yaml)
- 结果目录：
  [kitti06_render_benchmark_v5_quality_tuned](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_v5_quality_tuned/render_benchmark_summary.json)

目前这轮改进说明两件事：

- 方向上已经开始借鉴 `GSFusion / Gaussian-SLAM` 的“稳几何 + active submap 优化”思路
- 但仅靠 `KITTI stereo + 轻量 renderer`，还不足以直接达到 `GS-SDF` 级别画质

因此，当前最好的结论不是“问题已完全解决”，而是：

- 我们已经从“纯 heuristic streaming Gaussian”推进到了“开始具备 reference-style geometry/optimization 机制”
- 但如果想真正接近高质量 streaming GS，仍然需要更强的深度输入或更成熟的 GPU splat backend

### 10.2 Local Surface Volume 原型

当前已经新增一个 `local surface volume` 原型层：

- 代码：
  [local_tsdf.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/local_tsdf.py)
- builder 接入：
  [gaussian_builder.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_builder.py)
- 配置：
  [kitti06_v6_local_volume.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v6_local_volume.yaml)

它做的事情是：

- 每个 active submap 维护一个局部多帧融合表面体
- 用 stereo depth + confidence 融合进局部体素
- 再从局部体素表面中提取 surfel/Gaussian

这说明“更强局部几何底座”这条方向已经正式接进当前系统，而不是停留在说明层。

但当前第一版结果也说明：

- 系统结构更完整了
- 但画质还没有明显优于旧的 best surfel 版本
- 延迟也进一步上升

因此，下一步更应该做的是：

- 更 aggressive 的 surfel decimation / confidence pruning
- 只从局部体积里提取 near-visible surface
- 减少低质量深度块直接进入 Gaussian 渲染

### 10.3 Semantic-like Region Filter

为了避免系统继续朝“更重的后端重建”方向偏移，当前又补了一条更贴近实时高斯目标的轻量路线：

- 用规则化 region mask 先剔除明显低价值区域
- 不引入重语义模型
- 主要针对：
  - 天空
  - 超远低价值区域
  - 高纹理但低置信的区域
  - 不稳定前景块

相关配置和结果：

- 配置：
  [kitti06_v7_semantic_mask.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v7_semantic_mask.yaml)
- 结果：
  [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_v7_semantic_mask/render_benchmark_summary.json)

这条路线的意义在于：

- 更符合“前端实时高斯”的方向
- 先减少不该进入高斯预算的区域
- 不把系统继续推向更重的 TSDF / mesh backend

当前结果说明：

- 这条路线确实把预算和区域控制进一步系统化了
- 但画质提升仍然有限
- 它更像是在为以后真正的实时版本打下更合理的区域调度基础

### 10.4 LingBot-Map 风格轻量借鉴

为了验证“能不能借现有 streaming reconstruction 的轻量流式机制，而不把系统继续推向重后端”，当前又补了一条 `LingBot-Map` 风格借鉴线：

- 模块化 `sky mask`
- `keyframe interval` 风格的 Gaussian append cadence
- `optimize interval` 风格的流式优化节奏

相关文件和结果：

- 配置：
  [kitti06_v8_lingbot_borrowed.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v8_lingbot_borrowed.yaml)
- 结果：
  [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v8_lingbot_borrowed/render_benchmark_summary.json)
- triplet 页面：
  [render_triplets_viewer.html](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v8_lingbot_borrowed/render_triplets_viewer.html:1)
- compare 页面：
  [gsconsole_compare_viewer.html](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v8_lingbot_borrowed/gsconsole_compare_viewer.html:1)
- 三路对比：
  [gaussian_triple_compare_lingbot.html](/home/nyu/Codespace/CVPR/nuc_output/gaussian_triple_compare_lingbot.html:1)

这条线的结果很清楚：

- `mean_psnr = 11.4331`
- `mean_ssim = 0.29443`
- `mean_update_ms = 856.691`
- `mean_render_ms = 1148.231`
- `approx_fps = 0.499`

这说明：

- `LingBot-Map` 风格的轻量流式借鉴更像“前端预算和节奏控制器”
- 它对降低更新/渲染负担是有帮助的
- 但它本身不是高质量 Gaussian renderer，也不会直接替代 `HMR3D` 的长期记忆层

因此这条结果反而更支持当前项目的主判断：

> `LingBot-Map` 这类方法适合被借来增强前端流式节奏，但真正有研究辨识度的主线，仍然是 `cuVSLAM + HMR3D memory-native Gaussian lifecycle`。

当前项目已经完成了最难的前半段：

- `cuVSLAM` 跑通
- `HMR3D lifecycle` 跑通
- `incremental Gaussian submap` 跑通
- `surfel-style continuous render` 初步跑通
- `GS_Console` 风格 compare 展示页已具备

但还没有完成后半段：

- 高质量 GS-SDF 级别画质
- 实时性能
- 实机在线闭环

因此现在最准确的表述是：

> 我们已经从“只有 tracking 和零散点云”走到了“带长期记忆的 active Gaussian 连续视角原型系统”，但距离“真正实时高质量 Gaussian 机器人系统”仍有明确的工程和算法工作要做。

这不是失败，而是非常正常的中期状态。

如果接下来继续研发，最该做的是：

- 优化 active window Gaussian
- 提升局部几何质量
- 提升 render 效率
- 再逐步上实机

### 10.5 `gsplat` GPU Backend Adapter

为了验证“当前系统是否已经具备挂成熟 GPU backend 的结构条件”，这轮又给 renderer 增加了一条可切换的 `gsplat` backend 路径。

这次做的不是替换 `HMR3D` 主线，而是：

- 保留现有 `cuVSLAM + HMR3D + Gaussian lifecycle`
- 保留原有 CPU 近似 renderer
- 在此基础上新增 `render_backend=gsplat`
- 在 Jetson Orin Nano 的 GPU Python 环境里真实跑通一版 benchmark

相关文件和结果：

- 配置：
  [kitti06_v10_gsplat_backend.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v10_gsplat_backend.yaml)
- 结果：
  [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v10_gsplat_backend/render_benchmark_summary.json)
- triplet 页面：
  [render_triplets_viewer.html](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v10_gsplat_backend/render_triplets_viewer.html:1)
- compare 页面：
  [gsconsole_compare_viewer.html](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v10_gsplat_backend/gsconsole_compare_viewer.html:1)
- 三路对比：
  [gaussian_triple_compare_v10_gsplat.html](/home/nyu/Codespace/CVPR/nuc_output/gaussian_triple_compare_v10_gsplat.html:1)

当前结果：

- `mean_psnr = 9.7931`
- `mean_ssim = 0.17969`
- `mean_update_ms = 820.801`
- `mean_render_ms = 1598.007`
- `approx_fps = 0.413`

这条线说明了三件事：

- `gsplat` backend adapter 已经真正接通
- Jetson Orin Nano 当前环境已经具备 `torch + gsplat` 运行条件
- 但第一版 `gsplat` 路径画质还不如现有 CPU 代表线，说明当前瓶颈已经转向输入几何质量和参数映射，而不只是“有没有成熟 backend”

因此，这一步的价值主要是：

- 证明当前系统已经不再被锁死在自写 CPU renderer
- 后续可以在不改 `HMR3D` lifecycle 的前提下，继续打磨 GPU backend 路径
- 进一步把研究重点收束到真正有辨识度的主线：
  `cuVSLAM + HMR3D memory-native Gaussian lifecycle + switchable backend`

### 10.6 `v11`: Structured Fast Refine

为了回答“`v9` 这条线是不是成立，只是还缺实时约束下的轻量 refinement”，这轮又补了一条 `v11 structured fast refine`。

这次不再继续加重几何前半段，而是：

- 保留 `v9` 的 structured Gaussian initialization
- 增加 `fast refine mode`
- 只保留当前可见、最近、高误差、recover 影响到的点作为候选
- 只对少量高价值 Gaussian 做小步 refinement

相关文件和结果：

- 配置：
  [kitti06_v11_structured_fast_refine.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v11_structured_fast_refine.yaml)
- 结果：
  [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v11_structured_fast_refine/render_benchmark_summary.json)
- triplet 页面：
  [render_triplets_viewer.html](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v11_structured_fast_refine/render_triplets_viewer.html:1)
- compare 页面：
  [gsconsole_compare_viewer.html](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v11_structured_fast_refine/gsconsole_compare_viewer.html:1)
- 三路对比：
  [gaussian_triple_compare_v11.html](/home/nyu/Codespace/CVPR/nuc_output/gaussian_triple_compare_v11.html:1)

当前结果：

- `mean_psnr = 10.6207`
- `mean_ssim = 0.24647`
- `mean_update_ms = 809.511`
- `mean_render_ms = 891.105`
- `approx_fps = 0.588`

和 `v9 structured init` 对比，这条线说明：

- `v9` 的结构化初始化方向是成立的
- 继续往重几何走不是最优下一步
- 更合适的是在 `v9` 后面加 realtime-friendly 的轻量 refinement

虽然 `v11` 目前还没有追上 `Realtime Budget` 的画质，但它已经把 `v9` 往更接近实时前端的方向推了一步：

- 比 `v9` 更快
- 不再继续扩大前端几何负担
- 更符合“`cuVSLAM + HMR3D + structured Gaussian + fast refine`”这条新主线

### 10.7 `v12`: Recover-aware Structured Fast Refine

为了把 `HMR3D` 的长期记忆真正接进 Gaussian 优化本身，这轮在 `v11` 基础上又补了一条 `v12 recover-aware structured fast refine`。

这次不是再改前半段几何，而是：

- 保留 `v11` 的 structured init + fast refine
- 利用 warm-start / recover 事件
- 在 recover 发生后的若干帧里，把 refinement 预算更多给：
  - recovered 来源的 Gaussian
  - structured 来源的 Gaussian
  - 最近被记忆事件影响到的点

相关文件和结果：

- 配置：
  [kitti06_v12_recover_aware_structured.yaml](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs/kitti06_v12_recover_aware_structured.yaml)
- 结果：
  [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v12_recover_aware_structured/render_benchmark_summary.json)
- triplet 页面：
  [render_triplets_viewer.html](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v12_recover_aware_structured/render_triplets_viewer.html:1)
- compare 页面：
  [gsconsole_compare_viewer.html](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_v12_recover_aware_structured/gsconsole_compare_viewer.html:1)
- 三路对比：
  [gaussian_triple_compare_v12.html](/home/nyu/Codespace/CVPR/nuc_output/gaussian_triple_compare_v12.html:1)

当前结果：

- `mean_psnr = 10.6844`
- `mean_ssim = 0.2515`
- `mean_update_ms = 810.641`
- `mean_render_ms = 911.179`
- `approx_fps = 0.581`

这条结果说明：

- 把 recover 事件接进 refinement 预算这件事是有作用的
- 相比 `v11`，指标有小幅前进
- 但收益目前仍然是渐进式的，而不是立刻带来大幅画质跃升

因此，当前最准确的判断是：

> `v12` 已经开始体现“memory-native Gaussian system”的真正特色，但还需要继续把 recover-aware refinement 做得更细，才能把 `HMR3D` 的优势真正转化成画质和稳定性收益。
