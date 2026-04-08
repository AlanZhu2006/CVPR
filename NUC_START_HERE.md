# NUC Start Here

如果你的目标是先在 Intel NUC 上验证 `tracking -> archive / retrieve / recover` 这条主链，而不是直接做 CUDA / Gaussian 主系统，请从这里开始：

- 快速开始: [nuc/README.md](./nuc/README.md)
- 架构说明: [docs/NUC_ARCHITECTURE.md](./docs/NUC_ARCHITECTURE.md)
- 任务清单与验收: [docs/NUC_TASKS.md](./docs/NUC_TASKS.md)

这个入口对应的是第一阶段 CPU 友好原型：

- 输入: 视频或图片序列
- tracking: 轻量 ORB 前端
- memory: `M_short / M_active / B_bank`
- lifecycle: `observe -> promote -> archive -> retrieve -> recover`
- 输出: `poses.csv`、`events.jsonl`、`summary.json`、`submaps.json`、可选 `debug.mp4`

它不是最终 Jetson 版，也不依赖 CUDA。它的作用是先把 memory lifecycle 系统逻辑做对，再迁移到你报告里的 Jetson / cuVSLAM / active Gaussian 路线。
