# CVPR Workspace: cuVSLAM + HMR3D + Streaming Gaussian

这个工作区当前的主线，不是单独做一个离线高质量 Gaussian 重建器，而是构建一条更偏系统的路线：

- `cuVSLAM` 负责低延迟 tracking / mapping 前端
- `HMR3D` 负责长期记忆生命周期
- `Gaussian` 负责 active region 的局部表示与渲染

目标是把它做成一个 **memory-native Gaussian system**：

> 机器人边走边 tracking，边维护当前 active Gaussian submap，边把历史区域 archive 成长期记忆，并在回到旧区域时 retrieve / recover 历史高斯子图。

相关工作目录：

- [cuVSLAM](/home/nyu/Codespace/CVPR/cuVSLAM)
- [HMR3D](/home/nyu/Codespace/CVPR/HMR3D)
- 参考前端壳子：[GS_Console](/home/nyu/Codespace/CVPR/GS_Console/README.md)
- 详细状态文档：[PROJECT_STATUS_AND_REALTIME_GAUSSIAN_PLAN.md](/home/nyu/Codespace/CVPR/PROJECT_STATUS_AND_REALTIME_GAUSSIAN_PLAN.md:1)
- Jetson GPU backend 环境说明：[JETSON_GPU_BACKEND_SETUP.md](/home/nyu/Codespace/CVPR/HMR3D/docs/JETSON_GPU_BACKEND_SETUP.md:1)

## 项目结构

顶层目录当前可以这样理解：

- [cuVSLAM](/home/nyu/Codespace/CVPR/cuVSLAM)
  - 原始 SLAM 前端与示例数据入口
  - 本地 Jetson Python 环境在 [`.venv-jetson`](/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson)
  - 已实际跑通 KITTI 06，并生成轨迹与地图
- [HMR3D](/home/nyu/Codespace/CVPR/HMR3D)
  - 记忆生命周期与 Gaussian 实验主仓
  - `nuc/src/nuc_runtime/` 里是当前核心 runtime
  - `nuc/scripts/` 里是 replay、benchmark、viewer 生成脚本
  - `nuc/configs/` 里是各阶段配置
  - `nuc_output/` 里是实验结果、viewer 页面、render 输出
- [GS_Console](/home/nyu/Codespace/CVPR/GS_Console)
  - 只作为展示与 playback 参考，不是当前系统主后端

当前最重要的代码文件：

- Adapter / tracking bridge
  - [cuvslam_adapter.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/cuvslam_adapter.py:1)
- Memory lifecycle
  - [memory_router.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/memory_router.py:1)
  - [policies.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/policies.py:1)
  - [models.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/models.py:1)
  - [config.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/config.py:1)
- Gaussian system
  - [gaussian_builder.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_builder.py:1)
  - [gaussian_renderer.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_renderer.py:1)
  - [local_tsdf.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/local_tsdf.py:1)
- Entry scripts
  - [run_cuvslam_kitti_memory.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py:1)
  - [run_gaussian_render_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_gaussian_render_benchmark.py:1)
  - [open_gaussian_web.sh](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/open_gaussian_web.sh:1)
  - [use_jetson_gpu_backend.sh](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/use_jetson_gpu_backend.sh:1)

## 整体架构

当前系统不是单层高斯建图，而是三层结构。

### 1. `cuVSLAM`: Tracking / Pose Layer

职责：

- 提供位姿主链
- 提供关键帧节奏
- 提供可复用的 tracking / mapping 前端
- 对接 Jetson / NVIDIA 生态，适合作为实机底座

这里我们选择 `cuVSLAM`，不是因为它直接让高斯画得更清楚，而是因为：

- 它更适合 Jetson / Orin Nano
- tracking 更像实机系统底座
- 它本身已有 tracking / mapping 解耦与地图复用能力
- 它让 Gaussian 不用自己硬扛 tracking

### 2. `HMR3D`: Memory Lifecycle Layer

职责：

- `observe`
- `archive`
- `retrieve`
- `verify`
- `recover`
- `merge`
- `hierarchy`

这层是本项目最有研究辨识度的部分。  
很多 streaming GS 工作会做 active window、submap、loop closure，但通常不会把高斯表示组织成这么明确的长期记忆单元。

这里的核心不是“当前怎么画”，而是：

- 哪块 active Gaussian 什么时候该冻结
- 历史 Gaussian submap 怎样被检索和恢复
- 旧区域怎样 coarse/full 分层
- budget 怎样由 memory 状态来驱动

### 3. `Gaussian`: Active Representation Layer

职责：

- 把当前 active submap 表示成局部 Gaussian / surfel-like 结构
- archive 时导出 Gaussian handle
- recover 时把历史 Gaussian handle warm-start 回当前 active submap
- 为当前视角提供可渲染结果

重要的是：

- `Gaussian` 在这里是 **表示层**
- `HMR3D` 是 **组织与生命周期层**
- `cuVSLAM` 是 **位姿与 tracking 层**

所以项目主线不是：

- `cuVSLAM + 一个更清楚的渲染器`

而是：

- `cuVSLAM + HMR3D memory + active Gaussian representation`

## 当前数据流

当前离线实验主链：

`KITTI replay`
-> `trajectory_tum.txt`
-> `CUVSLAMOfflineKITTIAdapter`
-> `TrackingOutput`
-> `MemoryRouter`
-> `Active / Archived Submaps`
-> `Incremental Gaussian Builder`
-> `Renderer / Viewer`

其中：

- `Adapter` 把 `cuVSLAM` 轨迹结果变成 HMR3D 可消费的 `TrackingOutput`
- `MemoryRouter` 决定 submap 生命周期
- `GaussianBuilder` 把 active submap 变成 Gaussian handle
- `Renderer` 负责当前视角的验证性渲染

## 已完成内容

### cuVSLAM -> HMR3D 桥接

已完成：

- KITTI 06 轨迹接入 HMR3D runtime
- 可重复执行的 replay 入口
- `TrackingOutput` 适配

主要入口：

- [run_cuvslam_kitti_memory.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py:1)
- [CUVSLAM_ADAPTER.md](/home/nyu/Codespace/CVPR/HMR3D/docs/CUVSLAM_ADAPTER.md:1)

### HMR3D 生命周期机制

已落地并验证的机制包括：

- write-aware archive policy
- anchor retention
- pose-anchor gate
- shadow recover
- hierarchical bank
- scene-level summaries
- query routing
- multi-candidate merge
- local adapt

### Incremental Gaussian

已完成：

- active Gaussian submap 增量更新
- archived Gaussian handle 导出
- recover 时 warm-start 历史 Gaussian
- PLY / NPZ 导出
- live replay 与 web viewer

### Viewer / Demo

当前已有三类查看方式：

- `Rerun` 3D viewer
- `GT / Render / Diff` triplet viewer
- `GS_Console` 风格 compare viewer

当前代表性页面：

- Rerun 录制：[kitti06_v4_gaussian_live_800.rrd](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_v4_gaussian_live_800.rrd:1)
- triplet 页面：[render_triplets_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_v7_semantic_mask/render_triplets_viewer.html:1)
- compare 页面：[gsconsole_compare_viewer.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_v7_semantic_mask/gsconsole_compare_viewer.html:1)
- 三路对比：[gaussian_triple_compare_semantic.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/gaussian_triple_compare_semantic.html:1)

### Jetson GPU backend 环境

当前 Orin Nano 已经打通：

- `torch 2.8.0`
- `torchvision 0.23.0`
- `torchaudio 2.8.0`
- `gsplat 1.5.3`
- `torch.cuda.is_available() == True`

验证脚本：

- [use_jetson_gpu_backend.sh](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/use_jetson_gpu_backend.sh:1)

说明文档：

- [JETSON_GPU_BACKEND_SETUP.md](/home/nyu/Codespace/CVPR/HMR3D/docs/JETSON_GPU_BACKEND_SETUP.md:1)

## `v1` 到 `v7` 分阶段演进

下面这 7 个阶段不是“七篇不同论文式方法”，而是当前这条实验线的 7 次关键演进。

### `v1`: Baseline Memory + Minimal Render

做了什么：

- 打通 `cuVSLAM -> HMR3D`
- 建立 baseline lifecycle
- 提供最小 render benchmark 回路

意义：

- 从“只有轨迹”推进到“能跑 memory system”

### `v2`: Write-aware / Recover-aware Memory

做了什么：

- write-aware archive policy
- anchor retention
- pose-anchor gate
- shadow recover

意义：

- 解决 archive 过碎
- 让 recover 更克制、更可信

### `v3`: Full HMR3D Memory Lifecycle

做了什么：

- hierarchical bank
- scene summary
- query routing
- multi-candidate merge
- local adapt

意义：

- HMR3D 从简单 bank 变成显式长期记忆层

### `v4`: Incremental Gaussian + Realtime Budget 线

做了什么：

- active Gaussian submap
- archive Gaussian handle
- recover Gaussian warm-start
- live Rerun / web viewer
- realtime budget scheduler
- unstable mask / coarse archived bank / view-aware budget

代表结果：

- [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_realtime_budget/render_benchmark_summary.json:1)

关键指标：

- `PSNR 13.1584`
- `SSIM 0.45105`
- `update 265.91 ms`
- `render 748.4 ms`
- `approx_fps 0.986`

### `v5`: 更强几何 + 质量向尝试

做了什么：

- 更稳定的 stereo / surfel 输入
- 更偏质量向的配置
- 尝试靠更强几何与优化改善画面

代表结果：

- [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_v5_quality_tuned/render_benchmark_summary.json:1)

作用：

- 验证“继续堆质量向模块”并没有自然把系统推到 GS-SDF 水平

### `v6`: Local Surface Volume / Thin Surface

做了什么：

- local surface volume / local TSDF-like 几何底座
- near-visible surface
- confidence pruning
- thin surface extraction
- bad depth block suppression

代表结果：

- [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_v6_thin_surface/render_benchmark_summary.json:1)

关键指标：

- `PSNR 13.0304`
- `SSIM 0.36866`
- `update 1142.968 ms`
- `render 2196.215 ms`
- `approx_fps 0.299`

意义：

- 方向更接近 `GSFusion / Gaussian-SLAM` 的局部几何思路
- 但也明确暴露出：再往重几何走，会离“实时高斯前端”越来越远

### `v7`: Semantic-like Region Filter

做了什么：

- 轻量规则版 semantic-like mask
- 剔除天空、超远低价值区域、低置信和不稳定块
- 让系统更像“实时前端 budget 控制器”，而不是更重的后端

代表结果：

- [render_benchmark_summary.json](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/kitti06_render_benchmark_v7_semantic_mask/render_benchmark_summary.json:1)

关键指标：

- `PSNR 12.6755`
- `SSIM 0.35835`
- `update 1095.143 ms`
- `render 1946.152 ms`
- `approx_fps 0.329`

意义：

- 更符合“实时高斯前端”方向
- 但质量仍然没有接近 GS-SDF

## 目前已经测试过什么

当前已经做过的验证，可以分成 5 类：

### 1. Lifecycle 验证

验证：

- active/archive 切换
- retrieve / recover 是否触发
- hierarchical bank 是否形成
- merge / local adapt 是否实际发生

### 2. Gaussian 生成与导出

验证：

- active Gaussian 是否逐帧增长
- archived Gaussian handle 是否落盘
- warm-start recover 是否能够重新挂回当前 active submap

### 3. 当前视角渲染

验证：

- 能否输出 `GT / Render / Diff`
- 渲染是否至少有连续视角感
- 画质和延迟的权衡

### 4. 不同策略对比

已对比：

- quality-oriented surfel
- realtime budget
- semantic-like mask

三路对比入口：

- [gaussian_triple_compare_semantic.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/gaussian_triple_compare_semantic.html:1)

### 5. Jetson GPU backend

验证：

- `.venv-jetson` 下 GPU `torch` 可用
- `gsplat` 可导入
- Orin Nano 当前可作为成熟 GPU backend 目标环境

## 当前结论

现在最重要的判断，不是“我们还差几个 heuristic”，而是：

### 1. 轻量原型线已经试得比较全了

当前已经综合过：

- stereo seed
- surfel patch
- local optimization
- unstable mask
- hierarchical gaussian bank
- view-aware budget
- local surface volume
- thin surface extraction
- semantic-like region filter

所以现在不是“还没综合”，而是：

> 该综合的轻量策略已经做过一轮了。

### 2. 当前画质还明显不如成熟 streaming GS

这不是页面问题，而是底层问题：

- depth 质量还不够强
- Gaussian 参数还是简化版
- renderer 还是近似版

### 3. 真正的研究辨识度在 `HMR3D memory-native Gaussian`

当前最值得强调的不是：

- “我们能不能把图再磨清楚一点”

而是：

- “我们能不能把 Gaussian 真正纳入一个显式长期记忆系统”

也就是：

- Gaussian archive
- Gaussian retrieve
- Gaussian recover
- coarse/full hierarchy
- memory-aware budget scheduling

这部分相比许多只做 local rendering / local mapping 的 streaming GS，更像本项目真正的创新点。

## 下一步计划

当前最合理的下一步，不建议再继续无休止地往同一条轻量 CPU renderer 线上堆模块，而是分成两条更清楚的路线。

### 路线 A：Realtime Frontend Mode

目标：

- 继续坚持“实时高斯前端”
- 不追求 GS-SDF 级别视觉上限
- 优先保证：
  - tracking 稳
  - active Gaussian 能持续长
  - budget 能收住
  - recover / archive / hierarchy 可用

重点应该做：

- 把 `semantic-like mask + unstable mask + view-aware budget` 真正联动
- 只保留静态、近处、结构性强的区域
- 显式限制 active Gaussian 数量
- 明确 `realtime_frontend` 配置

### 路线 B：Quality Upper Bound Mode

目标：

- 不再强求实时
- 看在更强 backend 下，这条 `cuVSLAM + HMR3D + Gaussian` 能达到怎样的质量上限

重点应该做：

- 增加 GPU backend adapter
- 接入 `gsplat` 作为可选 backend
- 不改 HMR3D lifecycle 主逻辑，只替换底层 renderer/backend

### 当前推荐主线

就研究辨识度和系统路线来说，当前更推荐把主线收束成：

1. `cuVSLAM` 继续做前端 tracking
2. `HMR3D` 明确做长期记忆层
3. `Gaussian` 做 active / recovered region 的表示层
4. `gsplat` 作为后续可切换 backend，而不是替代 HMR3D 主线

一句话说：

> 后续最该做的不是“再做一个更像 GS-SDF 的独立 renderer”，而是把当前系统推进成一个真正的 `memory-native Gaussian system`，再让成熟 GPU backend 为它服务。

## 实机落地建议

后续如果上实机，建议分阶段推进：

### Phase A: 只上 tracking

- 先在 Jetson 上稳定跑 `cuVSLAM`
- 拿到实时 pose / keyframe / tracking status

### Phase B: 上 HMR3D memory

- 把 live `TrackingOutput` 喂给 `MemoryRouter`
- 验证 archive / retrieve / recover 是否在真实数据上触发

### Phase C: 上 active Gaussian

- 只在当前 active submap 维护高斯
- 暂时不追求全局高质量重建

### Phase D: 接成熟 GPU backend

- 用当前已经打通的 Jetson GPU Python 环境
- 给 HMR3D 增加 backend adapter
- 在不改变 memory lifecycle 的前提下切换到 `gsplat`

## 当前最重要的文档与入口

- 顶层状态说明：
  - [PROJECT_STATUS_AND_REALTIME_GAUSSIAN_PLAN.md](/home/nyu/Codespace/CVPR/PROJECT_STATUS_AND_REALTIME_GAUSSIAN_PLAN.md:1)
- Jetson GPU backend：
  - [JETSON_GPU_BACKEND_SETUP.md](/home/nyu/Codespace/CVPR/HMR3D/docs/JETSON_GPU_BACKEND_SETUP.md:1)
  - [use_jetson_gpu_backend.sh](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/use_jetson_gpu_backend.sh:1)
- HMR3D / cuVSLAM bridge：
  - [CUVSLAM_ADAPTER.md](/home/nyu/Codespace/CVPR/HMR3D/docs/CUVSLAM_ADAPTER.md:1)
- 当前三路渲染对比：
  - [gaussian_triple_compare_semantic.html](/home/nyu/Codespace/CVPR/HMR3D/nuc_output/gaussian_triple_compare_semantic.html:1)
