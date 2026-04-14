# GPU 机器：拉仓后的操作清单 & 训练下一步

本文说明两件事：（1）在有 NVIDIA GPU 的机器上 **clone 本仓库后要做什么**；（2）当前仓库里 **“训练”在路线图上的含义** 和 **建议的下一步**。

---

## Part A — 有 GPU 的机器上 fetch / clone 之后要做什么

### A1. 克隆与子模块

```bash
git clone --recurse-submodules https://github.com/AlanZhu2006/CVPR.git
cd CVPR
```

若已克隆但未带子模块：

```bash
git submodule update --init --recursive third_party/TTT3R
```

### A2. 对 TTT3R 打运行时兼容补丁（必须）

上游子模块 **不会** 包含本仓库依赖的少量兼容性修改（PyTorch 2.6+ `torch.load`、`CPU` 上 RoPE 对 `-1` pose 占位、evo/matplotlib 与退化轨迹对齐等）。拉仓后请执行：

```bash
bash scripts/apply_ttt3r_runtime_compat_patches.sh
```

若子模块被 `git checkout` 回干净状态，**需要重新执行** 该脚本后再跑 eval / smoke。

### A3. Python 环境与依赖

**推荐**（与 TTT3R 编译扩展一致）：按 `scripts/bootstrap_env.sh` 用 conda 建环境、装 PyTorch（**选择与当前 NVIDIA 驱动匹配的 CUDA 版本**，见 [pytorch.org](https://pytorch.org/get-started/locally/)）、再装 TTT3R 依赖并编 `curope`（脚本内已有步骤）。

**最小**（仅跑推理与 pytest，不编 CUDA 扩展）：

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pip install -r third_party/TTT3R/requirements.txt
bash scripts/setup_dust3r.sh
```

### A4. 权重

```bash
bash scripts/download_weights.sh
```

得到 `checkpoints/cut3r_512_dpt_4_64.pth`（约 3GB）。若机器无外网，请从有网的机器拷贝该文件到同一路径。

### A5. 确认 CUDA 可用

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.version.cuda); print('cuda:', torch.cuda.is_available())"
```

`torch.cuda.is_available()` 应为 `True`。否则先解决 **驱动与 PyTorch wheel 的匹配**（升级驱动或换成较低 CUDA 的 PyTorch），再跑 sweep。

### A6. 可选：编译 `curope`（RoPE CUDA 加速）

见 `scripts/bootstrap_env.sh` 中 `croco/models/curope` 的 `python setup.py build_ext --inplace`。未编译时会回退到 PyTorch RoPE（较慢，但可用）。

### A7. 冒烟与长评测

- **单测（避免 ROS pytest 插件）**：`bash scripts/run_pytest.sh -q`
- **一键冒烟（含 fixture relpose + v2 smoke）**：`bash scripts/run_validation_suite.sh`（默认 `device` 在 fixture JSON 里为 `cpu`；要上 GPU 请改对应 config 的 `"device": "cuda"` 或复制一份 manifest）
- **实长 TUM 相对位姿**：准备 `data/long_tum_s1/...` 后，例如  
  `python3 scripts/run_relpose_memory_sweep.py --config configs/tum_relpose_sweep_224_v7_geometry_multi_tum.json`  
  并在 JSON 中设 `"device": "cuda"`、`"size": 224`（显存紧时勿盲目上 512）

### A8. 数据集

长序列 TUM 等不在仓库内。准备流程见 `python3 scripts/prepare_longseq_benchmarks.py --help` 与 README。`data/` 已在 `.gitignore`，需在每台机器上自行下载或同步。

---

## Part B — “下一步怎么训练”在本仓库里的含义

### B1. HMR3D / HMR3D v2 主线（当前）

- **默认是推理期记忆与策略**：`archive / retrieve / verify / recover` 以及 v2 的 write / recover gate、hierarchy、merge、`local_adapt` 等，在 **`hmr3d_memory/`** 中实现，**不自带端到端 backbone 训练脚本**。
- **“训练”前的务实步骤**通常是：
  1. 在 GPU 上跑 **固定权重** 的 **sweep / longseq eval**，扫 `configs/*.json` 与 `MemoryConfig` 超参；
  2. 用 `leaderboard.json` 与 `summary.json` 做 **ATE / RPE / 内存统计** 对比（见 `scripts/print_relpose_leaderboard.py`）；
  3. 将稳定 preset 写回默认 manifest 或论文附录表。

实现细节与设计对照见 [HMR3D_v2_code_implementation_plan.md](HMR3D_v2_code_implementation_plan.md) 与 [HMR3D_v2_roadmap.md](../HMR3D_v2_roadmap.md)。

### B2. Mem3R 脚手架线（与“训练”最接近的一条）

本仓库包含 **Mem3R 式 fast-weight / gate 脚手架**（`mem3r_pose_probe`、`mem3r_like_runtime`），用于在 **不改动上游 TTT3R 主体** 的前提下做 **test-time 侧** 实验；**官方 Mem3R 训练代码与完整复现权重** 仍依赖外部仓库与 HPC。

更具体的 **HPC 交接、建议训练顺序、缺失项** 见：

- [mem3r_hpc_handoff.md](mem3r_hpc_handoff.md)

建议顺序（摘自该文档思路并与本仓库对齐）：

1. 固定 **base 权重** `cut3r_512_dpt_4_64.pth` 与（若已有）**脚手架初始化** `mem3r_scaffold_init.pt` / manifest；
2. 在 HPC 上实现 **官方 Mem3R 或自研** 的 **训练循环与 loss**（本仓库 **未实现** 官方训练 loop）；
3. 训练后把新 checkpoint 接回 `export_mem3r_scaffold.py` / `run_relpose_memory_sweep.py` 做 **回归**；
4. 再扩展到更多序列与 longseq 配置。

### B3. 若你指的是 “把 v2 也训出来”

v2 当前是可调 **策略与阈值**；若未来要对某子模块做 **可学习** 扩展（例如 gate 网络、merge 权重），需要 **新增数据集与 loss、在 `hmr3d_memory` 外或内单独建训练工程**，并与 TTT3R 前向对齐。本仓库暂无该训练模板；建议先在 GPU 上完成 **全量 eval 与消融表**，再立项训练。

---

## 快速对照表

| 目标 | 命令 / 文档 |
|------|-------------|
| 打 TTT3R 补丁 | `bash scripts/apply_ttt3r_runtime_compat_patches.sh` |
| 装 dust3r 路径 + pip 依赖 | `bash scripts/setup_dust3r.sh` |
| 下权重 | `bash scripts/download_weights.sh` |
| 单测 | `bash scripts/run_pytest.sh -q` |
| 本地冒烟 + 短 relpose | `bash scripts/run_validation_suite.sh` |
| v2 与 v1 短对比表 | `configs/tum_relpose_v2_smoke_metrics.json` + `scripts/print_relpose_leaderboard.py` |
| Mem3R / HPC 训练下一步 | [mem3r_hpc_handoff.md](mem3r_hpc_handoff.md) |
| v2 实现与测试说明 | [HMR3D_v2_code_implementation_plan.md](HMR3D_v2_code_implementation_plan.md) |

---

*文档版本：与当前 `master` 上脚本路径一致；若移动脚本或子模块 URL，请同步更新本节命令。*
