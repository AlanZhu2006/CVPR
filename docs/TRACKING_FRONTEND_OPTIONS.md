# Tracking Frontend Options

这份文档把当前 `nuc` 原型真正需要的 tracking 接口收敛清楚，并给出适合本项目的前端替换建议。

## 1. 当前代码真正依赖什么

`MemoryRouter` 并不关心前端是不是 ORB、VIO 还是 cuVSLAM。

它真正依赖的是 `TrackingOutput` 这组字段：

- `frame_idx`
- `timestamp_sec`
- `pose`
- `is_keyframe`
- `descriptor`
- `orb_descriptors`
- `keypoint_count`
- `match_count`
- `inlier_count`
- `pixel_motion`
- `track_ok`
- `frame_shape`

对应代码：

- `TrackingOutput` 定义在 `nuc/src/nuc_runtime/models.py`
- 当前 ORB 前端在 `nuc/src/nuc_runtime/tracking.py`
- lifecycle 路由在 `nuc/src/nuc_runtime/memory_router.py`

这意味着后续替换前端时，最稳的做法不是改 `MemoryRouter`，而是写一个新的 adapter，把外部前端输出映射成同样的 `TrackingOutput`。

## 2. 一个可直接替换的前端至少要做到什么

对本项目来说，一个“可直接用”的 tracking frontend 最少应满足：

- 能稳定输出逐帧位姿 `pose`
- 能给出关键帧事件，或至少能支持我们自己做 `is_keyframe` 判定
- 能提供一个全局描述子 `descriptor`，用于 archive / retrieve
- 最好能提供局部特征或 anchor 匹配信息，便于 recover 前的几何验证
- 最好有 relocalization / loop signal，但这不是第一版接入的硬前提

其中最关键的一点是：

如果新前端不能输出 ORB 描述子，也仍然可以接入，只是需要把当前 `recover` 里的 ORB 几何验证替换成新的局部匹配或几何校验模块。

## 3. 推荐顺序

### A. ORB-SLAM3

最适合：

- 现在就在 NUC 上尽快替换掉原型 ORB tracking
- 传感器是 stereo / RGB-D / mono+IMU / stereo+IMU
- 希望直接获得 relocalization 和 loop closure 能力

为什么适合你们：

- 它本质上仍然可以被当作“tracking frontend + keyframe source”
- 有 place recognition / relocalization / multi-map，和你们的长期记忆叙事是兼容的
- 很容易把输出压成 `pose + keyframe + track_ok + loop event`

接入建议：

- 不要把 ORB-SLAM3 当最终地图系统
- 只把它当姿态、关键帧和可选回环信号源
- `descriptor` 可以先复用你们现有的 `compute_global_descriptor()`
- `orb_descriptors` 可以直接从其前端特征中导出，继续服务 `recover` 的几何验证

风险：

- 官方仓库和示例环境偏旧，接到当前 Ubuntu/ROS 栈时通常要做编译适配

### B. OpenVINS

最适合：

- 你们手上有 IMU
- 更在意“稳的前端状态估计”而不是自带回环
- 希望更像库一样嵌到自己的系统里

为什么适合你们：

- 它更像真正的前端/VIO 模块，而不是整套地图系统
- 支持 ROS-free build，适合直接塞进自己的 runtime
- 对你们这种“tracking 主路径”和“memory 旁路”分离的设计很自然

接入建议：

- 如果是 stereo+IMU，这是我在 NUC 上最推荐的工程化方案之一
- `pose` 直接来自 OpenVINS
- `is_keyframe` 第一版可以继续由你们自己根据位姿变化和时间间隔判定
- `descriptor` 仍可由当前图像侧模块单独计算

风险：

- 它不提供你们最终想要的完整 relocalization / loop closure 体系
- 如果要把 recover 做强，后面还得单独补 place recognition 或 loop signal

### C. RTAB-Map

最适合：

- 你们想尽快在 ROS2 下把 RGB-D / stereo / IMU 传感器链跑通
- 更看重开箱即用的 bring-up，而不是最干净的模块边界

为什么只排第三：

- RTAB-Map 本身已经是“带记忆和回环的完整系统”
- 它和你们自己的 `HMR3D` 生命周期有概念重叠
- 如果直接全量接入，容易出现“谁来定义长期地图”的职责冲突

更合理的用法：

- 只拿它的 odometry / keyframe / loop hints
- 不把它当你们最终的 archive / retrieve / recover 主体

### D. Isaac ROS Visual SLAM / cuVSLAM

最适合：

- 后续正式迁到 Jetson 或 x86 + NVIDIA GPU
- 你们确认会走 NVIDIA 生态和 ROS2

为什么它不是当前 NUC 首选：

- 它的强项在 NVIDIA GPU 平台
- 你们当前这台 Intel NUC 阶段，更重要的是先把前端接口和 lifecycle 适配打通

更合理的项目路径：

1. 先在 NUC 上用 `ORB-SLAM3` 或 `OpenVINS` 把 adapter 写稳
2. 保持 `MemoryRouter` 与输出协议不变
3. 到 Jetson 时把 adapter 后端切换为 `cuVSLAM`

## 4. 我对这个项目的实际建议

如果你们现在要尽快进入“可持续开发”状态，我的建议是：

### 路线 1：没有 IMU，或者先不想碰 IMU

用 `ORB-SLAM3`

理由：

- 最接近当前原型思路
- 关键帧、重定位、回环都比较顺手
- 你们只需要写一个 adapter，而不需要重写 lifecycle

### 路线 2：有 stereo+IMU，而且更在意 tracking 主路径稳定

用 `OpenVINS`

理由：

- 它更像纯前端
- 更适合你们“tracking 主线程 + memory 旁路线程”的系统结构
- 后面再补检索/回环，不会和 `HMR3D` 的长期记忆逻辑打架

### 路线 3：正式 Jetson 演示阶段

切到 `Isaac ROS Visual SLAM / cuVSLAM`

理由：

- 这更符合报告里的最终系统叙事
- 也更符合 Jetson 上 tracking 和高斯 mapping 解耦的目标

## 5. 最小落地步骤

无论你选哪个前端，最小接法都建议保持一致：

1. 新建一个 frontend adapter，而不是改 `MemoryRouter`
2. 让 adapter 负责把外部前端输出压成 `TrackingOutput`
3. 第一版先复用现有 `descriptor` 计算逻辑
4. 如果没有 `orb_descriptors`，先把 `recover` 的几何验证改成新前端可提供的局部验证
5. 等 adapter 稳定后，再决定是否接入 loop signal / relocalization signal

这样做的好处是：

- `nuc` 原型可以继续作为生命周期验证层
- 后续切换 `ORB-SLAM3 -> OpenVINS -> cuVSLAM` 时，不会把 memory 层一起拖乱
