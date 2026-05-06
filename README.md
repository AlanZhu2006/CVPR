# CVPR Workspace

这个仓库现在保留两条还在维护的主线：

- `third_party_research/lingbot-map`
  - 作为独立 Git 子仓库管理的 LingBot-Map
- `HMR3D/nuc`
  - 本地 Jetson / cuVSLAM / LingBot 集成层

## 初始化

首次拉取后先同步子仓库：

```bash
cd /home/nvidia/twork/lingbot-map/CVPR
git submodule update --init --recursive
```

如果你刚创建好自己的 `lingbot-map` fork，再执行一次：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

当前 `.gitmodules` 已经预留为你的 fork URL。创建好仓库后，把子仓库里的
`cvpr/local-base` 推上去即可。

## 目录职责

- [third_party_research/lingbot-map](/home/nvidia/twork/lingbot-map/CVPR/third_party_research/lingbot-map)
  - 本地 LingBot 子仓库
  - `origin` 预期指向你的 fork
  - `upstream` 指向 `robbyant/lingbot-map`
- [third_party_research/lingbot_cache](/home/nvidia/twork/lingbot-map/CVPR/third_party_research/lingbot_cache)
  - 本地模型缓存，不纳入版本控制
- [HMR3D/nuc/src/nuc_runtime](/home/nvidia/twork/lingbot-map/CVPR/HMR3D/nuc/src/nuc_runtime)
  - 本地运行时封装，负责把 cuVSLAM、LingBot、重建和发布脚本串起来
- [HMR3D/nuc/scripts](/home/nvidia/twork/lingbot-map/CVPR/HMR3D/nuc/scripts)
  - Jetson 运行入口、导出脚本、viewer/publisher 启动脚本
- [HMR3D/docs/LINGBOT_ORIN_16GB_REPRO.md](/home/nvidia/twork/lingbot-map/CVPR/HMR3D/docs/LINGBOT_ORIN_16GB_REPRO.md)
  - 当前保留的 LingBot Orin 复现说明
- [nuc_output](/home/nvidia/twork/lingbot-map/CVPR/nuc_output)
  - 本地产物目录，不纳入版本控制

## 环境组织

当前主要使用两个本地虚拟环境：

- `cuVSLAM/.venv-jetson`
  - Jetson 主运行环境
  - `HMR3D/nuc/scripts/use_jetson_gpu_backend.sh` 会激活它，并设置本地 `cuDSS`
- `.venv-lingbot`
  - 仅用于独立 LingBot 实验时保留

推荐优先走 Jetson 主环境：

```bash
cd /home/nvidia/twork/lingbot-map/CVPR
source HMR3D/nuc/scripts/use_jetson_gpu_backend.sh
```

## LingBot 版本管理

在子仓库里直接看本地 patch 相对 upstream 的差异：

```bash
cd /home/nvidia/twork/lingbot-map/CVPR
git -C third_party_research/lingbot-map fetch upstream
git -C third_party_research/lingbot-map log --oneline --decorate --graph upstream/main..HEAD
git -C third_party_research/lingbot-map diff upstream/main..HEAD
```

## 当前 LingBot 入口

16GB Orin 上优先使用：

```bash
cd /home/nvidia/twork/lingbot-map/CVPR

MODEL_PATH=/path/to/lingbot-map.pt \
FIRST_K=20 \
IMAGE_SIZE=336 \
KEYFRAME_INTERVAL=2 \
PORT=19115 \
HMR3D/nuc/scripts/run_lingbot_church_16gb_smoke.sh
```

如果 `IMAGE_SIZE == MODEL_IMAGE_SIZE`，脚本会优先走上游 `demo.py`。
如果为了本地兼容改成较小输入尺寸，则会走本地 export/viewer 兼容路径。

## 当前整理原则

- 保留当前仍在服务 Jetson + LingBot 主线的文档和脚本
- 删除明显过时、依赖旧机器绝对路径、只用于历史实验记录的文档
- `third_party_research/lingbot-map` 作为 submodule 追踪，避免整块 vendor 目录污染父仓库状态

## 当前项目主线：RGB Video to Real-to-Sim

当前建议把项目主线从“强行实时高斯建图”调整为更稳定、也更完整的 offline / semi-offline real-to-sim pipeline：

```text
monocular RGB video / image sequence
  -> LingBot / Mem3R geometry front-end
  -> camera poses + depth + dense colored point cloud
  -> TSDF fusion
  -> mesh / occupancy / navigation geometry
  -> Gaussian seed / GS-SDF optimization
  -> photorealistic Gaussian scene + physics/navigation mesh
  -> Isaac Sim / GS Console / robot simulation assets
```

这个方向的核心目标不是只把已有 Gaussian 或 mesh 资产放进 viewer，而是从普通单目 RGB 视频或图像序列自动构建一个 real-to-sim scene bundle。输出应该同时服务两类用途：

```text
mesh / TSDF:
  负责真实几何、碰撞检测、可通行区域、导航规划、Isaac Sim 物理侧资产

Gaussian / GS-SDF:
  负责真实感渲染、视觉观测、机器人相机模拟、GS Console 展示侧资产
```

### 目标产物

一个完整 scene bundle 建议组织为：

```text
scene_bundle/
  manifest.json
  rgb/
  poses/
  depth/
  confidence/
  pointcloud/
  tsdf/
  mesh/
  gaussian_seed/
  gaussian_optimized/
  nav/
  isaac/
  reports/
```

其中 `manifest.json` 记录输入视频、相机内参、尺度来源、坐标系约定、frame transforms、资产路径和质量指标。

### 技术分工

- LingBot / Mem3R
  - 从单目 RGB 恢复相机位姿、深度、局部稠密几何和置信度
  - 作为 neural geometry front-end
- TSDF / mesh
  - 融合多帧深度和 pose，生成更稳定的显式几何
  - 用于 collision、navigation、occupancy 和 Isaac Sim 几何侧
- Gaussian / GS-SDF
  - 从点云或 mesh 初始化 Gaussian seed
  - 继续优化为高质量视觉场景
  - 用于 photo-realistic rendering 和视觉传感器模拟
- RealSense / 标定板 / 已知尺度
  - 不一定作为主输入
  - 更适合作为 scale anchor、质量验证和局部 depth sanity check

### 与 GS Playground 的区别

这个方向和 GS Playground 接近，但重点不同：

```text
GS Playground:
  偏向已有 Gaussian / mesh / scene asset 的交互、编辑和展示

本项目:
  偏向从 monocular RGB 自动恢复 geometry，并生成可进入仿真和导航系统的 hybrid mesh-Gaussian scene
```

所以研究叙事可以概括为：

```text
RGB-only / RGB-first Real-to-Sim Scene Reconstruction with Neural Geometry
and Hybrid Mesh-Gaussian Representation
```

### 下一步离线 MVP

优先做一个稳定 offline MVP：

1. 选择 20-60 秒室内 RGB 视频或图像序列。
2. 用 LingBot `demo.py` 或 Mem3R 跑出 depth、pose、colored point cloud。
3. 写统一 exporter，把模型输出转成 `scene_bundle/`。
4. 用 TSDF 融合 depth + pose，导出 mesh / occupancy / nav geometry。
5. 从 point cloud / mesh 生成 Gaussian seed。
6. 接 GS-SDF / SplaTAM / 3DGS 优化，导出 visual Gaussian scene。
7. 在 GS Console 中对比展示 RGB trajectory、TSDF mesh、Gaussian render。
8. 预留 Isaac Sim 导出接口，至少包含 mesh collision asset、visual asset 和坐标系 manifest。

### 当前 WebUI 视频回放入口

已经加了一个第一版 video playback sidecar，用来把 `videos/` 里的 RGB 视频接到原来的 live WebUI：

```text
videos/*.mp4
  -> 自动抽帧
  -> MJPEG RGB fast path
  -> binary WebSocket current cloud
  -> binary WebSocket global / Gaussian seed cloud
  -> real2sim/latest/gaussian_seed
  -> 原 GS Console live monitor layout
```

启动：

```bash
cd /home/nvidia/twork/lingbot-map
VIDEO_DIR=/home/nvidia/twork/lingbot-map/videos \
EXTRACT_FPS=2 \
PLAYBACK_FPS=2 \
MAX_FRAMES=80 \
POINTS_PER_FRAME=8000 \
MAX_GLOBAL_POINTS=180000 \
bash GS_Console/scripts/launch_video_real2sim_playback_stack.sh
```

### A6000 迁移策略

Jetson 继续适合做 camera / ROS / robot-side preview；LingBot / Mem3R / TSDF / Gaussian / GS-SDF 优化建议迁到 A6000。不要把 30GB+ 的 `CVPR/nuc_output` 和 4GB+ checkpoint 直接塞进 `GS_Console` git 仓库，推荐用 rsync 迁移运行包：

```bash
cd /home/nvidia/twork/lingbot-map

# 本地打包到 /home/nvidia/twork/lingbot-map/a6000_migration_bundle
bash GS_Console/scripts/prepare_a6000_migration_bundle.sh

# 或者直接同步到 A6000
A6000_SSH=user@a6000-host \
A6000_ROOT=/data/lingbot-map \
bash GS_Console/scripts/prepare_a6000_migration_bundle.sh
```

默认迁移：

```text
GS_Console/
CVPR/HMR3D/
CVPR/third_party_research/lingbot-map/
CVPR/third_party_research/SplaTAM/
CVPR/third_party_research/lingbot_cache/lingbot-map.pt
videos/
CVPR/nuc_output/video_real2sim_playback/
CVPR/nuc_output/real2sim_hikrobot_lingbot_live_baseline/
CVPR/nuc_output/hikrobot_lingbot_ros2_current_cloud_live/
```

如果确实要同步整个 `nuc_output/`：

```bash
INCLUDE_ALL_NUC_OUTPUT=1 \
A6000_SSH=user@a6000-host \
A6000_ROOT=/data/lingbot-map \
bash GS_Console/scripts/prepare_a6000_migration_bundle.sh
```

推荐 git 策略：

```text
GS_Console repo:
  只放 WebUI / launcher / contracts / migration scripts

CVPR repo:
  放 reconstruction backend / scripts / docs

Large artifacts:
  checkpoint、videos、nuc_output 用 rsync / shared storage / release artifact，不进 git
```

打开：

```text
http://10.209.93.176:5173/?scene=/scenes/lingbot-live/manifest.json&mode=live&liveContract=/contracts/lingbot-map-video-playback.live-contract.json
```

当前 WebUI sidecar 已经支持两种输入：

- `LINGBOT_PREDICTIONS_NPZ + LINGBOT_SUMMARY_JSON`：优先使用 LingBot 真实 `world_points / world_points_conf`，左上角显示累积 global point cloud / Gaussian seed，左下角显示当前帧 cloud。
- 没有 LingBot 输出时：fallback 到 RGB-derived synthetic depth scaffold，只用于测试 display / bundle / WebSocket / Gaussian seed 链路。这种模式会像一张方框状 RGB 点云，不代表真实重建质量。

当前已跑通的真实 LingBot video smoke：

```text
nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_predictions.npz
nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_summary.json
```

对应 playback 启动时设置：

```bash
LINGBOT_PREDICTIONS_NPZ=/home/nvidia/twork/lingbot-map/CVPR/nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_predictions.npz \
LINGBOT_SUMMARY_JSON=/home/nvidia/twork/lingbot-map/CVPR/nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_summary.json \
NORMALIZE_LINGBOT_WORLD=1 \
PLAYBACK_FPS=2 \
MAX_FRAMES=20 \
POINTS_PER_FRAME=12000 \
MAX_GLOBAL_POINTS=180000 \
bash GS_Console/scripts/launch_video_real2sim_playback_stack.sh
```

### 当前实时链路定位

实时 HikRobot / RealSense + cuVSLAM + LingBot + GS Console 链路继续保留，但定位调整为 backup baseline 和系统扩展方向：

```text
primary path:
  offline RGB video -> real-to-sim scene bundle

backup / extension path:
  live RGB/RGB-D -> low-latency RGB/pose preview
  + async LingBot dense updates
  + rolling/global colored point cloud
```

实时链路的价值是：

- 验证同一套 reconstruction backend 能否在线运行
- 提供 live demo backup
- 作为后续机器人在线扫描 / 在线更新场景的扩展路线
- 用 RealSense depth 给 offline 单目重建提供尺度和质量校验

## LingBot 实时链路优化进度

当前实时系统的目标是把 HikRobot RGB、cuVSLAM tracking、LingBot dense geometry、colored point cloud / Gaussian splat 可视化和 GS Console WebUI 串成一条低延迟链路。

### 核心判断

已经确认主要瓶颈不是 HikRobot RGB 采集，也不是 WebUI 渲染，而是 dense geometry worker 的调度和 GPU 竞争。

实测过的典型状态：

```text
HikRobot RGB:              4-5fps
ROS RGB topic:             4-5fps
LingBot 新几何推理:        avg ~3.23s / window
队列等待 queue_wait_sec:   avg ~6.29s
worker end-to-end:         avg ~9.51s
```

注意区分：

```text
WebUI 画面刷新流畅 != 新几何实时更新
鼠标转动视角 != LingBot 生成新几何
/lingbot/current_cloud_rgb 高频重发 != 每次都是新点云 / 新 GS
```

目前 GS Console 的 live 布局中，左上角 Gaussian splat 和左下角 colored point cloud 都主要消费：

```text
/lingbot/current_cloud_rgb
```

所以可视化可以很顺，但新几何内容仍可能是几秒前生成的。

### 已完成改动

1. Dense worker 改为实时优先调度

   - `MAX_QUEUE=1`
   - `DROP_WHEN_BUSY=1`
   - latest-only pending window
   - worker 忙时丢弃旧 pending window，只保留最新 window

2. Dense 投喂频率从每帧改为低频关键窗

   当前推荐默认：

   ```bash
   DENSE_FRAME_INTERVAL=15
   DENSE_SUBMIT_WHEN_WORKER_IDLE=1
   ```

   对 5fps RGB 来说，这更接近当前 LingBot dense worker 的消费能力。

3. 增加 profiling / debug 指标

   日志和 WebUI debug 里重点区分：

   ```text
   rgb_fps
   webui_render_fps
   current_cloud_publish_fps
   new_geometry_fps
   geometry_age_sec
   queue_wait_sec
   lingbot_elapsed_sec
   worker_end_to_end_sec
   dense_queue_size
   dropped_window_count
   processed_window_count
   preprocess_sec
   model_forward_sec
   postprocess_sec
   pointcloud_build_sec
   ros_publish_sec
   ```

4. 增加 LingBot fast-path benchmark

   脱离 ROS/WebUI 的最小 benchmark 说明模型本体可以明显更快：

   ```text
   image_size=224, model_image_size=518, window_size=2
   second iteration model_forward_sec ~= 0.39-0.43s / window
   ```

   这说明实时链路里的 3s+ 并不全是模型能力上限，工程集成、GPU 竞争和调度占了很大比例。

5. 验证 FlashInfer / Jetson 环境

   当前 Jetson 主环境使用：

   ```text
   Python: CVPR/cuVSLAM/.venv-jetson/bin/python
   torch CUDA: cu126 stack
   FlashInfer: enabled
   FLASHINFER_CUDA_ARCH_LIST=8.7
   ```

   已避免原先 `flashinfer-cubin` / `flashinfer-jit-cache` 预编译 kernel 不适配 Orin `sm_87` 导致的：

   ```text
   no kernel image is available for execution on the device
   ```

6. 尝试 torch.compile

   结论：暂不作为 live 默认路径。

   `torch.compile` 在 NX/Orin 当前环境下首次 warmup 太重，曾出现数分钟级编译/图捕获等待，不适合 live demo 默认启动。

7. 尝试 LingBot persistent streaming cache

   结论：暂不作为默认优化。

   persistent streaming 可以跑通，后续窗口能只输出最新帧，但在 cuVSLAM 同时运行时没有解决 12-15s 级别慢窗。说明当时主要问题不是 KV/cache reset，而是 cuVSLAM 与 LingBot dense 的 GPU/系统资源竞争。

8. 增加 cuVSLAM 与 LingBot dense 解耦开关

   当前最有效改动：

   ```bash
   PAUSE_TRACKING_WHILE_DENSE=1
   ```

   这会在 dense worker 活跃时暂停主 tracking/cuVSLAM loop，降低 GPU 竞争。

### 已做实验结果

#### A. 原始过载链路

旧参数接近：

```bash
FRAME_STEP=1
DENSE_FRAME_INTERVAL=1
WINDOW_SIZE=2
```

问题：

```text
producer: 4-5 windows/s
worker:   ~0.31 windows/s
```

结果是 FIFO/队列堆积，新几何端到端延迟到 8-10s。

#### B. latest-only queue + interval=15

结论：队列等待基本消除。

预期和实测方向一致：

```text
queue_wait_sec:      ~6s -> 接近 0
dense_queue_size:    不持续大于 1
dropped_window_count 增加是预期行为
```

#### C. cuVSLAM dense 时暂停

这是目前最有效 baseline。

稳定测试中过的典型窗口：

```text
queue_wait_sec:        ~0.001s
worker_end_to_end_sec: ~3-5s
geometry_age_sec:      ~4-6s
model_forward_sec:     ~2.4-3.8s，冷启动窗口除外
```

最近一次恢复后的稳定测试摘要：

```text
events: 5 windows
queue_wait_sec mean:        0.0076s
worker_end_to_end_sec mean: 5.73s
latest worker_end_to_end:   4.20s
latest geometry_age_sec:    5.57s
latest model_forward_sec:   3.60s
dense_queue_size:           0
drops:                      0
```

注意第一窗常包含模型加载 / warmup，不应拿第一窗代表 steady-state。

#### D. LingBot 自己 pose / camera head

新增了可切换实验开关：

```bash
LINGBOT_ENABLE_CAMERA=1
PREFER_LINGBOT_POSE=1
HIKROBOT_DISABLE_CUVSLAM=1
```

结论：Jetson NX/Orin 16GB 上暂不推荐默认使用。

实测现象：

```text
第一窗还未出结果
RAM ~= 14.7 / 15.6GB
swap ~= 2GB
GPU 基本空转
```

这说明 LingBot camera/pose head 在当前配置下会把系统推入内存/交换区瓶颈。当前更稳妥方案仍是：

```text
cuVSLAM 做低延迟 tracking
LingBot 做异步 dense geometry
dense 时暂停或限速 cuVSLAM，避免 GPU 竞争
```

#### E. dense-busy throttle 调度

已加入 throttle 实验开关：

```bash
DENSE_BUSY_TRACKING_POLICY=throttle
DENSE_BUSY_TRACKING_MIN_INTERVAL_SEC=1.0
```

第一轮测试暴露了一个调度问题：throttle 在模型 preload 阶段继续推进 tracking，而 `DENSE_SUBMIT_WHEN_WORKER_IDLE=1` 会跳过后续 dense candidate，导致有限帧测试里 80 帧只提交 1 帧，无法形成第一个 window。

已经修复为：

```text
模型 preload 阶段强制 pause
模型加载完成后，dense active 阶段才允许 throttle tracking
初始 dense window 未凑齐 WINDOW_SIZE 前，允许继续提交 seed frames
```

修复后 `MAX_FRAMES=40` smoke 结果：

```text
processed_windows:              2
dense_busy_tracking_steps:      16
dense_busy_tracking_throttle_waits: 363
queue_wait_sec:                 ~0.001s
latest worker_end_to_end_sec:   ~5.0s
latest geometry_age_sec:        ~6.6s
latest model_forward_sec:       ~5.0s
```

结论：throttle 已经能工作，但还不应直接替代 pause baseline。它改善了 pose/path 连续性，但会重新引入一定 cuVSLAM/LingBot 竞争，尤其冷窗和长时间运行阶段仍需继续调参。

#### F. cuVSLAM async tracking worker

已把 HikRobot live 路径里的 cuVSLAM tracking 做成可选后台 worker：

```bash
HIKROBOT_ASYNC_TRACKING=1
HIKROBOT_TRACKING_IDLE_FPS=5.0
HIKROBOT_TRACKING_DENSE_FPS=1.0
PAUSE_TRACKING_WHILE_DENSE=0
DENSE_BUSY_TRACKING_POLICY=none
```

结构变为：

```text
HikRobot capture thread:
  持续采集 RGB，并发布 RGB preview

cuVSLAM async tracking thread:
  latest-only 输入/输出队列
  dense 空闲时按 HIKROBOT_TRACKING_IDLE_FPS 跑
  dense 忙时按 HIKROBOT_TRACKING_DENSE_FPS 降频

LingBot dense worker:
  latest-only dense queue
  preload 阶段不接收 dense seed
  preload 完成后从最新 pose 重新开始凑 WINDOW_SIZE
```

调度修复点：

```text
preload 阶段 pose 可以继续跑，但 dense 不提交
preload 完成后 dense interval 从最新帧重新计数
初始 WINDOW_SIZE seed frames 绕过 DENSE_FRAME_INTERVAL
```

`MAX_FRAMES=80` smoke 结果：

```text
tracked_frames:            80
submitted_frames:          2
processed_windows:         1
queue_wait_sec:            ~0.004s
worker_end_to_end_sec:     ~13.6s
geometry_age_sec:          ~13.9s
model_forward_sec:         ~12.8s
```

结论：架构解耦已跑通，`queue_wait_sec` 不再被 preload 污染；但 dense 期间 cuVSLAM 即使 1fps 仍会和 LingBot 抢资源，把单窗 forward 拉慢。下一轮应把 dense-period tracking 调到更低：

```bash
HIKROBOT_TRACKING_DENSE_FPS=0.2
HIKROBOT_TRACKING_DENSE_FPS=0.0
```

其中 `0.0` 等价于 dense 期间 tracking worker 暂停，但 RGB capture 和主 loop 不再被 cuVSLAM 阻塞，后续可以继续做更细粒度恢复。

#### G. rolling local map backend

已加入 rolling local colored point cloud map 后端：

```bash
ROLLING_MAP=1
ROLLING_MAP_VOXEL_SIZE=0.06
ROLLING_MAP_RADIUS_M=0.12
ROLLING_MAP_MIN_NEIGHBORS=2
ROLLING_MAP_MAX_WINDOWS=8
ROLLING_MAP_MAX_AGE_SEC=30.0
ROLLING_MAP_MAX_POINTS=180000
```

新增组件：

```text
KeyframeManager:
  根据平移、旋转、时间阈值选择 keyframe
  保存 timestamp、RGB、intrinsics、T_world_camera
  输出 keyframes/keyframes.json 和 keyframes/*.png

RollingPointCloudMap:
  接收 LingBot dense window 生成的 world-frame colored points
  使用 cuVSLAM T_world_camera 完成 camera/local -> world 转换
  voxel downsample
  简单 radius/min-neighbor outlier filtering
  只保留最近 N 个 dense windows 或最近 T 秒
  输出 rolling_map.ply，并通过现有 current cloud 通道给 WebUI

Dense state:
  IDLE
  PREPROCESSING
  MODEL_FORWARD_ACTIVE
  POSTPROCESSING
  PUBLISHING
```

cuVSLAM async tracking worker 现在只在 `MODEL_FORWARD_ACTIVE` 时认为 LingBot 真正占用推理资源；preprocess/postprocess/publish 阶段可作为后续插空 tracking 的机会。

新增 run artifacts：

```text
metrics.json
full_stack_metrics.json
trajectory.txt
keyframes/
rolling_map.ply
run.log
```

WebUI live monitor 继续显示：

```text
live RGB
cuVSLAM trajectory / pose count
rolling colored point cloud
geometry age
dense update count
```

Smoke 测试结果：

```text
Python py_compile: pass
launch bash -n: pass
WebUI npm run build: pass
artifacts generated: metrics.json, trajectory.txt, keyframes/, rolling_map.ply, run.log
```

短 smoke 的 `MAX_FRAMES=45` 仍不足以等到 LingBot 完整 dense window，所以 rolling point count 为 0；长时间 live 运行时 rolling map 会在 dense window 完成后更新。

### 当前推荐启动方式

推荐从 GS Console 入口启动完整 live stack：

```bash
cd /home/nvidia/twork/lingbot-map

COMPILE_LINGBOT_MODEL=0 \
LINGBOT_PERSISTENT_STREAMING=0 \
LINGBOT_ENABLE_CAMERA=0 \
PREFER_LINGBOT_POSE=0 \
HIKROBOT_DISABLE_CUVSLAM=0 \
PAUSE_TRACKING_WHILE_DENSE=1 \
bash GS_Console/scripts/launch_lingbot_realtime_stack.sh
```

也可以实验更柔和的 dense-busy tracking throttle。它不会在 dense worker 活跃时完全停 cuVSLAM，而是按最低间隔低频跑 tracking：

```bash
cd /home/nvidia/twork/lingbot-map

DENSE_BUSY_TRACKING_POLICY=throttle \
DENSE_BUSY_TRACKING_MIN_INTERVAL_SEC=1.0 \
PAUSE_TRACKING_WHILE_DENSE=1 \
bash GS_Console/scripts/launch_lingbot_realtime_stack.sh
```

或者实验新的 cuVSLAM async tracking worker：

```bash
cd /home/nvidia/twork/lingbot-map

HIKROBOT_ASYNC_TRACKING=1 \
HIKROBOT_TRACKING_IDLE_FPS=5.0 \
HIKROBOT_TRACKING_DENSE_FPS=0.2 \
PAUSE_TRACKING_WHILE_DENSE=0 \
DENSE_BUSY_TRACKING_POLICY=none \
bash GS_Console/scripts/launch_lingbot_realtime_stack.sh
```

如果 `DENSE_BUSY_TRACKING_POLICY=throttle`，即使 `PAUSE_TRACKING_WHILE_DENSE=1` 仍然保留，也会优先使用 throttle。A/B 时重点对比：

```text
geometry_age_sec
worker_end_to_end_sec
model_forward_sec
track_next_sec
dense_busy_tracking_steps
dense_busy_tracking_throttle_waits
```

WebUI：

```text
http://localhost:5173/?scene=/scenes/lingbot-live/manifest.json&mode=live&liveContract=/contracts/lingbot-map-ros2-live.live-contract.json
```

关键默认参数在 [GS_Console/scripts/launch_lingbot_realtime_stack.sh](/home/nvidia/twork/lingbot-map/GS_Console/scripts/launch_lingbot_realtime_stack.sh) 中设置：

```bash
DENSE_FRAME_INTERVAL=15
MAX_QUEUE=1
DROP_WHEN_BUSY=1
DENSE_SUBMIT_WHEN_WORKER_IDLE=1
PAUSE_TRACKING_WHILE_DENSE=1
DENSE_BUSY_TRACKING_POLICY=none
DENSE_BUSY_TRACKING_MIN_INTERVAL_SEC=1.0
OFFLOAD_TO_CPU=1
PRELOAD_LINGBOT_MODEL=1
COMPILE_LINGBOT_MODEL=0
LINGBOT_ENABLE_CAMERA=0
```

### 可选实验命令

谨慎测试 LingBot pose-only，不建议长时间跑：

```bash
cd /home/nvidia/twork/lingbot-map

COMPILE_LINGBOT_MODEL=0 \
LINGBOT_PERSISTENT_STREAMING=0 \
PAUSE_TRACKING_WHILE_DENSE=0 \
LINGBOT_ENABLE_CAMERA=1 \
PREFER_LINGBOT_POSE=1 \
HIKROBOT_DISABLE_CUVSLAM=1 \
bash GS_Console/scripts/launch_lingbot_realtime_stack.sh
```

如果出现 RAM/swap 快速上涨，立刻停止并恢复推荐 baseline。

### 当前架构结论

短期不要追求：

```text
RGB fps == new geometry fps
```

当前合理架构是：

```text
RGB fast path:
  HikRobot RGB -> ROS RGB topic -> WebUI
  目标：4-5fps，低显示延迟

Tracking path:
  cuVSLAM -> pose/path
  目标：低延迟位姿

Geometry slow path:
  latest keyframe/window -> LingBot worker -> point cloud / GS display
  目标：异步 0.1-0.5Hz 起步，几何 age 稳定不增长
```

cuVSLAM 和 LingBot 可以互相帮助，但目前最现实的方式不是让 LingBot pose 替代 cuVSLAM，而是：

```text
cuVSLAM 提供稳定实时 pose
LingBot 生成高质量 dense geometry
调度层避免二者同时抢 GPU
后续再让 cuVSLAM submap / tracking confidence 反哺 LingBot keyframe 选择
```

### 2026-05-05 Rolling Map Live Validation

本轮目标是验证 LingBot dense output 是否真的进入 rolling backend，而不是只看 WebUI 的重发帧。使用 safe 配置：

```bash
ROLLING_MAP=1 \
HIKROBOT_ASYNC_TRACKING=1 \
HIKROBOT_TRACKING_IDLE_FPS=5.0 \
HIKROBOT_TRACKING_DENSE_FPS=0.0 \
PAUSE_TRACKING_WHILE_DENSE=0 \
DENSE_BUSY_TRACKING_POLICY=none \
MAX_FRAMES=180 \
bash GS_Console/scripts/launch_lingbot_realtime_stack.sh
```

已修复两个验证阶段暴露的调度问题：

```text
1. MAX_FRAMES live validation 会等待 LingBot preload 完成后再开始计帧，避免 preload 期间把所有 dense candidate 都记成 worker busy skip。
2. HikRobot async tracking iterator 在 tracking worker 结束或停止产出时会正常退出，避免主进程空转等待满 180 个输出。
```

本次有效结果：

```text
rolling_map.ply:             2.3 MB，非空
live_map.ply:                2.3 MB，非空
rolling_map_point_count:     49,753
dense_update_count:          7
processed_windows:           7
trajectory.txt:              103 poses
keyframes/:                  73 RGB keyframes
dense_windows/:              7 local PLY + 7 world PLY
worker failures:             0
queue drops:                 0
swap used at finish:         ~2.18 GB
RAM used at finish:          ~12.58 / 15.29 GB
```

关键 latency：

```text
queue_wait_sec mean:         0.011s
queue_wait_sec max:          0.038s
worker_end_to_end mean:      5.14s
worker_end_to_end median:    4.09s
geometry_age mean:           5.59s
geometry_age median:         4.47s
model_forward mean:          4.11s
model_forward median:        3.11s
geometry_update_interval:    ~10.5s mean
```

注意：首个 dense window 仍有明显冷启动成本：

```text
window_000 model_forward_sec: ~11.54s
后续 window model_forward_sec: 多数约 1.9-3.3s
```

这说明 rolling backend 已经成功，但实时性下一步应优化两件事：

```text
1. 首窗 warmup / compile / dtype setup，让首个有效 dense 不再拉高 age。
2. async pose 输出队列，目前 latest-only 队列较小，dense/write 阶段会丢中间 pose；如果要更连续 trajectory，可测试 HIKROBOT_TRACKING_QUEUE_SIZE=16/32。
```

新增 artifact：

```text
metrics.json
full_stack_metrics.json
trajectory.txt
run.log
rolling_map.ply
keyframes/keyframes.json
dense_windows/window_000000_local.ply
dense_windows/window_000000_world.ply
...
```

`local.ply` 用于检查 LingBot 本地几何是否合理，`world.ply` 用于检查 cuVSLAM `T_world_camera` 变换是否正确。如果 rolling map 飞掉，优先对比这两组文件。

### Geometry Quality Gate

下一步优化顺序明确为：

```text
P0: geometry quality validation
P1: first window warmup
P2: tracking queue ablation
P3: geometry update interval 10.5s -> 5-7s
P4: background artifact writer
P5: v0.1 rolling backend demo baseline
```

不要在 P0 通过前只调速度。当前新增质量检查脚本：

```bash
python3 CVPR/HMR3D/nuc/scripts/validate_rolling_geometry_quality.py \
  --output-dir CVPR/nuc_output/hikrobot_lingbot_ros2_current_cloud_live
```

输出：

```text
geometry_quality_report.json
geometry_quality_report.md
geometry_quality_viewer.html
```

当前 P0 数值结论：

```text
rolling_map points:          49,753
trajectory poses:            103
trajectory path length:      1.632m
local z median:              ~12.8-13.6m
rolling distance-to-traj:    13.47m median
world window overlap:        high
consecutive NN median:       0.18-0.66m
```

解释：

```text
local PLY 非空，且 z 为正，LingBot 局部几何不是空输出。
world PLY 之间没有随机飞散，相邻窗口 bbox overlap 约 0.94-1.00。
rolling map 不是只有“有点”，而是多个 world window 能重叠融合。
但 rolling/world 点云整体离 trajectory 约 13m，且 local depth median 也约 13m。
```

所以当前最可疑的不是 `T_world_camera` 完全用反，而是：

```text
LingBot depth scale 和 cuVSLAM mono pose scale 不一致
或者 DEPTH_SCALE=20 对当前室内 live 场景偏大
或者 cuVSLAM Mono 轨迹尺度未被真实尺度约束
```

在继续调 warmup/fps 之前，应该人工打开：

```text
dense_windows/window_000000_local.ply
dense_windows/window_000000_world.ply
rolling_map.ply
trajectory.txt
geometry_quality_viewer.html
```

要确认：

```text
local 几何是否像真实场景
world 几何是否只是整体尺度偏大，还是轴向/旋转错
多个 world window 是否有重影
trajectory 是否落在合理相机位置附近
```

### 下一步计划

1. First window warmup

   preload 后立刻做一次 dummy forward，不发布 geometry，只用于 CUDA kernel / memory warmup。目标：

   ```text
   first real window model_forward_sec: 11.5s -> 3-4s
   WebUI 状态: PRELOADING / WARMING_UP / LIVE
   ```

2. Tracking queue ablation

   不直接跳到 16/32，按顺序测：

   ```text
   HIKROBOT_TRACKING_QUEUE_SIZE=1,2,4,8,16,32
   ```

   每组记录：

   ```text
   pose count
   pose fps
   pose_gap_sec median / max
   queue_wait_sec median / max
   geometry_update_interval
   model_forward_sec
   rolling map quality
   ```

3. 做 motion/keyframe scheduler

   从固定 `DENSE_FRAME_INTERVAL=15` 升级到：

   ```text
   worker idle
   + 最小帧间隔
   + 平移/旋转阈值
   + pixel motion / scene change
   + 新区域比例
   ```

4. 降低 single-window 耗时

   继续测试：

   ```bash
   MAX_POINTS_PER_FRAME=8000
   MAX_POINTS_PER_FRAME=4000
   ```

   重点观察：

   ```text
   model_forward_sec
   pointcloud_build_sec
   ros_publish_sec
   geometry_age_sec
   点云/GS 可视质量
   ```

5. Background artifact writer + binary cloud WebSocket

   已接入第一版低延迟 WebUI 传输优化：

   ```text
   RGB:          MJPEG fast path，不走 rosbridge image JSON
   pose/path:    仍走 rosbridge，保持 ROS/WebUI 兼容
   local cloud:  优先走 binary WebSocket ws://<host>:19093/cloud
   global cloud: persistent map binary WebSocket ws://<host>:19094/cloud
   fallback:     binary WS 未连接时自动回退 /lingbot/current_cloud_rgb
   artifacts:    dense window PLY / live snapshot 走后台 writer
   ```

   `binary_cloud_smoke2` 这轮 180s live smoke 的结果：

   ```text
   tracks:                    402
   pose fps approx:            2.40 Hz
   pose gap median / max:      0.35s / 2.97s
   dense updates:              17
   queue_wait_sec median:      0.0016s
   worker_end_to_end median:   3.27s
   model_forward median:       2.18s
   geometry_age median:        3.57s
   new_geometry_fps median:    0.154 Hz
   rolling_map_point_count:    max 84365
   binary cloud publish median:0.028s
   artifact submit median:     0.00009s
   live_write_sec median:      0.0s
   dropped windows:            0
   ```

   解释：

   ```text
   WebUI 点云已经不再必须走 rosbridge JSON。
   live_map / PLY 写盘基本不再占 worker 主路径。
   现在主瓶颈回到 LingBot forward + preprocess，而不是 WebUI transport。
   ```

   2026-05-05 之后的 live monitor 布局约定：

   ```text
   主屏幕:     live RGB
   左上角:     Persistent Global Map，长期累计室内点云，来自 :19094
   左下角:     Rolling Local Colored Point Cloud，低延迟局部预览，来自 :19093
   右下角:     Nav2-style 2D projection/debug map
   ```

   注意：当前 Nav2-style 2D map 不是带语义/地面的真实 costmap。cuVSLAM mono 没有 IMU/depth 时并不知道重力方向和地面高度，2D map 只是从 3D 点云按当前坐标系投影/切片得到。摄像头正对墙时，墙面点会被投影成占据区域，这是预期限制。要做真正可导航地图，需要加入：

   ```text
   camera-to-base 外参
   gravity / IMU / 手动 floor normal
   floor height band
   obstacle height threshold
   wall/ground filtering
   ```

6. 保留 LingBot pose-only 为实验路径

   如果后续要继续尝试，需要先降低内存压力，例如更小点数、更小 image size、关闭多余输出、确认 camera head 是否能单独轻量化。

7. HikRobot 相机内参标定

   当前全局点云如果“旋转后所有 dense window 叠在一坨”，首先要排除 cuVSLAM mono 输入不准的问题。之前 live stack 默认用 `fx=width, fy=width, cx=width/2, cy=height/2`，且按无畸变 Pinhole 处理，这对 cuVSLAM mono 不够可靠。

   已加入棋盘格 / ChArUco 双模式标定脚本。对于 calib.io 上这张 200x150mm、11 列 x 8 行、方格 15mm、marker 11mm、DICT_4X4 的 ChArUco 板，使用：

   ```bash
   cd /home/nvidia/twork/lingbot-map/CVPR
   source HMR3D/nuc/configs/hikrobot_mvs_env.sh 2>/dev/null || true

   python3 HMR3D/nuc/scripts/calibrate_hikrobot_intrinsics.py \
     --target charuco \
     --capture-mode manual \
     --charuco-cols 11 \
     --charuco-rows 8 \
     --square-size-m 0.015 \
     --marker-size-m 0.011 \
     --aruco-dict DICT_4X4_50 \
     --min-charuco-corners 12 \
     --samples 30 \
     --no-camera-roi \
     --detect-scale 0.5 \
     --preview-max-width 1280 \
     --fps 5 \
     --exposure-us 8000 \
     --gain 6 \
     --output-dir nuc_output/hikrobot_calibration
   ```

   `manual` 是默认采集模式：预览窗口里检测到绿色角点后，按 `Space` 或 `s` 保存当前视角；按 `q` 或 `Esc` 提前结束。这样可以避免脚本连续保存几张几乎一样的姿态。需要自动采集时改成 `--capture-mode auto`。

   `--no-camera-roi` 会让 HikRobot 采集完整传感器画面，避免把 640x512 当作从左上角裁剪的 ROI。`--detect-scale 0.5` 只用于检测加速，角点会被缩放回原始完整图像坐标，标定输出仍对应完整分辨率。

   打印时选择 `100% / Actual Size / 不缩放`。打印后用尺子量方格边长；如果实际不是 15mm，需要把 `--square-size-m` 和 `--marker-size-m` 按实际尺寸改掉。标定时让板覆盖画面中心、四角、边缘，做近/远、左右倾斜、上下倾斜，至少采 20-30 张不同姿态。

   如果换成普通棋盘格，则用：

   ```bash
   --target checkerboard \
   --checkerboard-cols 9 \
   --checkerboard-rows 6 \
   --square-size-m 0.025
   ```

   注意：普通棋盘格的 `checkerboard-cols/rows` 是内角点数量，不是格子数量。例如 10x7 个黑白格通常对应 9x6 内角点。

   结果会写出：

   ```text
   nuc_output/hikrobot_calibration/hikrobot_camera_calibration.json
   nuc_output/hikrobot_calibration/hikrobot_camera_calibration.yaml
   nuc_output/hikrobot_calibration/hikrobot_calibration.env
   nuc_output/hikrobot_calibration/accepted/
   nuc_output/hikrobot_calibration/corners/
   ```

   live 运行前加载标定：

   ```bash
   source /home/nvidia/twork/lingbot-map/CVPR/nuc_output/hikrobot_calibration/hikrobot_calibration.env

   CAMERA_FX="$HIKROBOT_CAMERA_FX" \
   CAMERA_FY="$HIKROBOT_CAMERA_FY" \
   CAMERA_CX="$HIKROBOT_CAMERA_CX" \
   CAMERA_CY="$HIKROBOT_CAMERA_CY" \
   CAMERA_DISTORTION_COEFFS="$HIKROBOT_DISTORTION_COEFFS" \
   CAMERA_UNDISTORT=1 \
   bash /home/nvidia/twork/lingbot-map/GS_Console/scripts/launch_lingbot_realtime_stack.sh
   ```

   `CAMERA_UNDISTORT=1` 会在 HikRobot 帧进入 cuVSLAM / LingBot 前先去畸变，并使用新的 pinhole K；如果想让 cuVSLAM 自己吃 Brown 畸变模型，可以设 `CAMERA_UNDISTORT=0` 做对比。

## A6000 部署说明

这一节给迁移到 A6000 的机器使用。当前项目已经拆成两个路径：

```text
主线:
  offline / semi-offline RGB video
  -> LingBot or Mem3R
  -> world_points / pose / depth / confidence
  -> TSDF / mesh / occupancy
  -> Gaussian seed / GS-SDF / 3DGS optimization
  -> GS Console / Isaac Sim / robot simulation assets

备份实时链路:
  Jetson camera / ROS / cuVSLAM / async LingBot / WebUI
```

A6000 应该承担主线中的重计算部分：

```text
LingBot / Mem3R inference
TSDF / mesh fusion
Gaussian seed generation
GS-SDF / SplaTAM / 3DGS optimization
high-quality render / evaluation
WebUI playback of reconstructed scene
```

Jetson 只保留为 camera / robot-side runtime：

```text
HikRobot / RealSense capture
ROS2 camera topics
low-latency preview
Nav2 / robot control
```

### 迁移方式

不要把 checkpoint、视频和 `nuc_output/` 全部提交进 git。推荐从 Jetson 用 rsync 同步运行包：

```bash
cd /home/nvidia/twork/lingbot-map

A6000_SSH=user@a6000-host \
A6000_ROOT=/data/lingbot-map \
bash GS_Console/scripts/prepare_a6000_migration_bundle.sh
```

如果需要把整个 `CVPR/nuc_output/` 也搬过去：

```bash
cd /home/nvidia/twork/lingbot-map

INCLUDE_ALL_NUC_OUTPUT=1 \
A6000_SSH=user@a6000-host \
A6000_ROOT=/data/lingbot-map \
bash GS_Console/scripts/prepare_a6000_migration_bundle.sh
```

默认会迁移：

```text
GS_Console/
CVPR/HMR3D/
CVPR/README.md
CVPR/command.txt
CVPR/third_party_research/lingbot-map/
CVPR/third_party_research/SplaTAM/
CVPR/third_party_research/lingbot_cache/lingbot-map.pt
videos/
CVPR/nuc_output/video_real2sim_playback/
CVPR/nuc_output/real2sim_hikrobot_lingbot_live_baseline/
CVPR/nuc_output/hikrobot_lingbot_ros2_current_cloud_live/
```

### A6000 环境准备

A6000 不需要 Jetson 的 HikRobot / RealSense SDK 才能跑 offline 主线。它需要：

```text
CUDA / PyTorch GPU 环境
Python packages for LingBot
Node.js / npm for GS Console WebUI
OpenCV / numpy / pillow / scikit-image
optional: gsplat / SplaTAM / GS-SDF dependencies
```

最小 smoke test 先跑 LingBot export：

```bash
cd /data/lingbot-map/CVPR

python HMR3D/nuc/scripts/run_lingbot_export.py \
  --model-path third_party_research/lingbot_cache/lingbot-map.pt \
  --lingbot-map-root third_party_research/lingbot-map \
  --image-folder ../videos/vid_frames \
  --output-dir nuc_output/video_real2sim_playback/lingbot_vid20 \
  --first-k 20 \
  --image-size 518 \
  --model-image-size 518 \
  --mode streaming \
  --keyframe-interval 2 \
  --camera-num-iterations 1
```

如果 A6000 上 FlashInfer / compile 还没配好，可以先加：

```bash
--use-sdpa
```

输出应包含：

```text
nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_predictions.npz
nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_summary.json
```

并且 `prediction_keys` 应该包含：

```text
world_points
world_points_conf
depth
depth_conf
extrinsic
intrinsic
```

### A6000 WebUI 回放

用 LingBot 输出启动 GS Console live-style playback：

```bash
cd /data/lingbot-map

LINGBOT_PREDICTIONS_NPZ=/data/lingbot-map/CVPR/nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_predictions.npz \
LINGBOT_SUMMARY_JSON=/data/lingbot-map/CVPR/nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_summary.json \
NORMALIZE_LINGBOT_WORLD=1 \
PLAYBACK_FPS=2 \
MAX_FRAMES=20 \
POINTS_PER_FRAME=12000 \
MAX_GLOBAL_POINTS=180000 \
bash GS_Console/scripts/launch_video_real2sim_playback_stack.sh
```

打开：

```text
http://A6000_IP:5173/?scene=/scenes/lingbot-live/manifest.json&mode=live&liveContract=/contracts/lingbot-map-video-playback.live-contract.json
```

WebUI 布局：

```text
主屏幕: live RGB playback
左上角: LingBot world_points 累积 global map / Gaussian seed preview
左下角: 当前帧 colored point cloud
右下角: Nav2-style 2D projection placeholder
```

注意：如果不传 `LINGBOT_PREDICTIONS_NPZ`，系统会 fallback 到 synthetic RGB-depth scaffold。那个模式只用于测试 WebUI/WS 链路，左上角会像一张方框状 RGB 点云，不代表真实重建。

### A6000 后续执行顺序

1. 跑 20 帧 smoke test，确认 LingBot 输出 `world_points`。
2. 扩到完整视频或 20-60 秒片段。
3. 用 `export_lingbot_worker_to_real2sim.py` 或后续 video bundle exporter 生成：

   ```text
   scene_points.ply
   scene_mesh.ply
   gaussians_seed.npz
   gaussians_seed.ply
   manifest.json
   ```

4. 在 A6000 上跑 Gaussian / GS-SDF / SplaTAM 优化。
5. 导出 Nav2 / Isaac Sim 所需 mesh、occupancy、collision assets。

### 当前已推送的代码位置

```text
GS_Console:
  commit 6eae7c1 Add LingBot video playback and A6000 migration tools

CVPR:
  commit 02293fd Add LingBot real-to-sim backend and playback tools
```

A6000 可以直接 clone 这两个仓库，但大文件仍需要 rsync / shared storage：

```text
lingbot-map.pt
videos/
nuc_output/
```
