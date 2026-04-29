# KITTI Raw Benchmark Plan

这份文档说明如何把 `cuVSLAM` 和 `FAST-LIVO2` 放到 **同一条 KITTI raw 多传感器序列** 上做更公平的 benchmark。

当前结论先说：

- 现在本地已经有的 `KITTI 06` 只是一份 **odometry stereo 子集**
- 它适合 `cuVSLAM`
- 但 **不够** 直接跑 `FAST-LIVO2`
- 如果想做公平对比，应该换成 **KITTI raw**

## 1. 为什么要用 KITTI raw

`cuVSLAM` 和 `FAST-LIVO2` 的输入模态不同：

- `cuVSLAM`：视觉前端，至少要有左/右相机
- `FAST-LIVO2`：LiDAR + IMU + camera

所以只有在同一条 **多传感器原始序列** 上，才能尽量公平地比较：

- 轨迹效果
- 运行速度
- 资源占用

KITTI 官方 raw 数据页明确提供：

- synced+rectified grayscale stereo
- Velodyne point clouds
- GPS/IMU (OXTS)
- calibration

因此 `KITTI raw` 是更合适的公共比较底座。

## 2. Sequence 06 对应的 raw drive

我们当前本地常用的是 odometry `sequence 06`。

公开映射资料表明，odometry `06` 对应：

- raw date: `2011_09_30`
- raw drive: `2011_09_30_drive_0020_sync`
- frame range: `000000` to `001100`

这条映射不是来自 KITTI 官方首页，而是来自常用 odometry/raw 对照资料，例如：

- `yfcube/kitti-devkit-odom`
- 若干 KITTI odometry/raw 映射整理文档

因此我们后续默认按这条映射准备 KITTI raw 数据。

## 3. 需要下载的最小数据

为了只做 `sequence 06` 的 benchmark，最少需要下面这些 raw 资源：

### 3.1 图像

从 KITTI raw 下载页面获取这条 drive 的 **synced+rectified grayscale stereo**：

- `2011_09_30_drive_0020_sync`
- 左右灰度相机

如果下载的是按日期打包的数据，通常需要保留这条 drive 下的：

- `image_00/data`
- `image_01/data`
- `image_00/timestamps.txt`
- `image_01/timestamps.txt`

### 3.2 LiDAR

必须要有这条 drive 的 Velodyne：

- `velodyne_points/data`
- `velodyne_points/timestamps.txt`

### 3.3 IMU / OXTS

必须要有 OXTS：

- `oxts/data`
- `oxts/timestamps.txt`

### 3.4 Calibration

必须要有这条日期对应的 calibration：

- 相机内参/外参
- Velodyne 到相机
- OXTS 到相机/车体

通常来自 KITTI raw development kit 和 raw 数据目录里的 calibration 文件。

## 4. 本地当前已有和缺失

### 已有

当前本地已有：

- `/home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti/dataset/sequences/06`

其中有：

- `image_0`
- `image_1`
- `calib.txt`
- `times.txt`
- `trajectory_tum.txt`

这足够继续跑 `cuVSLAM`，但不够跑 `FAST-LIVO2`。

### 缺失

当前本地还没有看到：

- `2011_09_30_drive_0020_sync/velodyne_points`
- `2011_09_30_drive_0020_sync/oxts`
- 这条 raw drive 的完整多传感器原始目录

所以现在还不能开始 `KITTI raw` 公平 benchmark。

## 5. 我们建议的 benchmark 结构

### 方法

- `cuVSLAM`
- `FAST-LIVO2`

### 数据

同一条 raw 序列：

- `2011_09_30_drive_0020_sync`

统一 frame range：

- `000000:001100`

### 指标

主指标：

- `ATE RMSE`
- `ATE mean / median / max`
- `RPE translation RMSE`
- `RPE rotation RMSE`

速度和资源：

- `wall time`
- `throughput`
- `realtime factor`
- `max RSS`
- `avg / max CPU`
- `avg / max GPU`

### 可视化

最终生成一个 web compare：

- top-down trajectories
- metrics cards
- runtime / memory cards
- 说明输入模态与配置

## 6. 推荐执行顺序

### Step 1

先把 KITTI raw 的这条 drive 下载完整：

- `2011_09_30_drive_0020_sync`
- stereo
- velodyne
- oxts

### Step 2

做一个 `KITTI raw -> cuVSLAM` 提取脚本：

- 生成 `image_0`
- `image_1`
- `times.txt`
- 如有需要，生成 absolute timestamp CSV

### Step 3

做一个 `KITTI raw -> FAST-LIVO2` 适配：

- 点云
- IMU/OXTS
- 相机时间轴
- topic 或离线输入组织

### Step 4

统一轨迹输出成 `TUM` 或 KITTI pose 格式，再做 benchmark。

## 7. 为什么这会比 NTU VIRAL 更稳

相比 `NTU VIRAL`：

- `KITTI` 没有 prism GT 这种特殊参考点问题
- `cuVSLAM` 对 KITTI 更原生友好
- `cuVSLAM` 的官方例子已经是 KITTI stereo
- benchmark 更不容易被 “时间轴 / GT 定义 / 参考点定义” 干扰

所以如果目标是：

> 更可信地比较 `cuVSLAM` 和 `FAST-LIVO2`

那么 `KITTI raw` 比当前 `NTU VIRAL` 更适合作为下一轮主 benchmark。

## 8. 当前最现实的下一步

1. 下载 `2011_09_30_drive_0020_sync` 的 raw 多传感器数据
2. 本地整理成：
   - stereo image
   - velodyne
   - oxts
3. 再开始正式做 `cuVSLAM vs FAST-LIVO2 on KITTI raw`

在没有这批 raw 多传感器数据之前，我们还不能给出一份真正公平、完整的 KITTI 对比结果。

## 9. 当前已补好的准备脚本

现在工作区里已经有两条直接可用的准备脚本：

- [prepare_kitti_raw_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_kitti_raw_benchmark.py:1)
  - 检查 `KITTI raw` drive 是否完整
  - 生成 `cuVSLAM` 输入目录
  - 生成 `FAST-LIVO2` 输入目录骨架
  - 从 `OXTS` 导出 `TUM GT`
- [validate_kitti_raw_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/validate_kitti_raw_benchmark.py:1)
  - 检查工作目录里的 frame 数、时间戳和 GT 是否一致
  - 在真正跑 benchmark 前先卡掉明显错误
- [benchmark_trajectories_generic.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/benchmark_trajectories_generic.py:1)
  - 用标准 `TUM` 轨迹直接做 `GT / cuVSLAM / FAST-LIVO2` 对比
  - 适合 `KITTI raw` 这种不需要 prism 补偿的 6DoF 数据集
- [run_cuvslam_kitti_raw.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_raw.py:1)
  - 直接在准备好的 `KITTI raw` workspace 上跑 stereo `cuVSLAM`
  - 读取 `prepare_kitti_raw_benchmark.py` 生成的 `calib.txt`
- [write_kitti_raw_download_instructions.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/write_kitti_raw_download_instructions.py:1)
  - 生成指定 drive 的下载清单

当前已经生成好的 0020 下载清单：

- [KITTI_RAW_2011_09_30_0020_DOWNLOAD_CHECKLIST.md](/home/nyu/Codespace/CVPR/HMR3D/docs/KITTI_RAW_2011_09_30_0020_DOWNLOAD_CHECKLIST.md:1)

拿到数据后可直接运行：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_kitti_raw_benchmark.py \
  --raw-root /path/to/KITTI/raw \
  --date 2011_09_30 \
  --drive 0020 \
  --frame-start 0 \
  --frame-end 1100 \
  --output-dir /home/nyu/Codespace/CVPR/nuc_output/kitti_raw_2011_09_30_0020_benchmark
```

然后先做一次 workspace 校验：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/validate_kitti_raw_benchmark.py \
  --workspace /home/nyu/Codespace/CVPR/nuc_output/kitti_raw_2011_09_30_0020_benchmark
```

`cuVSLAM` 这一半已经可以直接开跑：

```bash
source /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/use_jetson_gpu_backend.sh
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_raw.py \
  --sequence-dir /home/nyu/Codespace/CVPR/nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_input \
  --output-trajectory /home/nyu/Codespace/CVPR/nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_tum.txt \
  --absolute-time
```

等 `cuVSLAM` 和 `FAST-LIVO2` 两边都导出成 `TUM` 轨迹后，再做最终对比：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/benchmark_trajectories_generic.py \
  --gt /home/nyu/Codespace/CVPR/nuc_output/kitti_raw_2011_09_30_0020_benchmark/gt/gt_tum_absolute.txt \
  --cuvslam /path/to/cuvslam_tum.txt \
  --fastlivo /path/to/fastlivo2_tum.txt \
  --output-dir /home/nyu/Codespace/CVPR/nuc_output/kitti_raw_2011_09_30_0020_compare
```
