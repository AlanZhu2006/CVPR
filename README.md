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
