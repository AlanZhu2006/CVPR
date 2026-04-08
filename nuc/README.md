# NUC Prototype

这个目录把仓库整理成一套可以直接在 Intel NUC 上运行的第一阶段原型。目标不是复现最终 Jetson + Gaussian 系统，而是先把下面这条主链跑通：

`image/video -> tracking -> pose/keyframe -> archive/retrieve/recover`

## 1. 这套原型解决什么问题

它对应你当前项目里最关键、也最适合 NUC 的部分：

- 稳定输出 `pose` 和 `keyframe`
- 把关键帧组织成 `M_short / M_active / B_bank`
- 在 `B_bank` 里做历史子图检索
- 在 revisit 时触发 recover

它不做的事：

- 不依赖 CUDA
- 不跑 cuVSLAM
- 不做完整 Gaussian mapping
- 不要求 ROS 在线硬件闭环

## 2. 目录结构

```text
nuc/
  configs/default_replay.yaml
  scripts/setup_nuc.sh
  scripts/check_nuc_env.sh
  scripts/run_nuc_replay.sh
  src/nuc_runtime/
  tools/run_nuc_replay.py
  tools/compare_runs.py
```

## 3. 最短启动路径

在 Ubuntu NUC 上：

```bash
git clone <your-repo-url>
cd CVPR

bash nuc/scripts/setup_nuc.sh
source .venv-nuc/bin/activate

bash nuc/scripts/check_nuc_env.sh
bash nuc/scripts/run_nuc_replay.sh \
  nuc/configs/default_replay.yaml \
  /path/to/video.mp4 \
  nuc_output/run_a
```

如果输入是图片序列目录，也可以直接把 `/path/to/video.mp4` 换成目录路径。

## 4. 输出文件

一次运行后，`output_dir` 里会生成：

- `poses.csv`: 每帧位姿、关键帧标记、tracking 质量和当前 active/bank 状态
- `events.jsonl`: `active_started / promoted / archived / retrieved / recovered`
- `summary.json`: 本次运行的配置和统计
- `submaps.json`: 当前 active 和 bank 摘要
- `keyframes/`: 保存下来的关键帧图像
- `debug.mp4`: 带状态叠字的视频，默认开启

## 5. 最小对照实验

先跑一遍启用 recover 的版本：

```bash
bash nuc/scripts/run_nuc_replay.sh \
  nuc/configs/default_replay.yaml \
  /path/to/video.mp4 \
  nuc_output/recover_on
```

再跑一遍关闭 recover 的版本：

```bash
python3 nuc/tools/run_nuc_replay.py \
  --config nuc/configs/default_replay.yaml \
  --input /path/to/video.mp4 \
  --output-dir nuc_output/recover_off \
  --disable-recover
```

然后比较两次的 `summary.json`：

```bash
python3 nuc/tools/compare_runs.py \
  --left nuc_output/recover_on/summary.json \
  --right nuc_output/recover_off/summary.json
```

## 6. 当前实现的模块映射

- `nuc/src/nuc_runtime/tracking.py`
  轻量 ORB tracking 前端，输出 `pose / keyframe`
- `nuc/src/nuc_runtime/memory_router.py`
  实现 `observe -> promote -> archive -> retrieve -> recover`
- `nuc/src/nuc_runtime/io.py`
  统一处理视频和图片序列输入
- `nuc/src/nuc_runtime/output.py`
  统一落 `csv/jsonl/json/mp4`

## 7. 当前版本的边界

这是一个 CPU 友好的系统原型，不是最终算法结果。你需要明确它的用途：

- 它的 tracking 是替代前端，不是最终前端
- pose 是可用接口，不是高精度 SLAM 基准结果
- recover 是“历史子图重新注入 active”的系统动作，不是高斯恢复
- 它的价值在于验证 memory lifecycle，而不是重建画质

## 8. 下一步怎么接到你的正式系统

你后续只需要替换两层：

1. 把 `tracking.py` 换成 ORB-SLAM3 / VINS / cuVSLAM 的统一接口输出
2. 把 `M_active` 从关键帧块升级成 active Gaussian submap

`MemoryRouter` 这层可以保持不变，继续作为整个系统的生命周期调度层。
