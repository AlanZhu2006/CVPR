# HMR3D：分层记忆长程在线三维重建

**更新时间：2026-04-13**

本仓库以 **HMR3D**（*Hierarchical Memory Reconstruction for 3D*）为主线：在固定算力与显存预算下，把在线重建从**单一 recurrent / 全局地图状态**，升级为**短程 — 活动 — 归档 — 路由**的显式 **memory lifecycle**（`archive / retrieve / recover`），而不是再堆一个语义头或换一套巨型 backbone。

---

## 目录

1. [核心命题与一句话定位](#1-核心命题与一句话定位)  
2. [文档与报告索引（建议阅读顺序）](#2-文档与报告索引建议阅读顺序)  
3. [仓库结构](#3-仓库结构)  
4. [Git：克隆方式与 main / master 分支说明](#4-git克隆方式与-main--master-分支说明)  
5. [当前进度（与报告对齐）](#5-当前进度与报告对齐)  
6. [双轨路线：神经主干 vs Jetson 边缘版](#6-双轨路线神经主干-vs-jetson-边缘版)  
7. [系统架构：四层与数据流](#7-系统架构四层与数据流)  
8. [生命周期、调度与 Jetson 预算](#8-生命周期调度与-jetson-预算)  
9. [研究问题、基线、Stress Tests 与验证矩阵](#9-研究问题基线stress-tests-与验证矩阵)  
10. [阶段验收 P0–P4 与交付物](#10-阶段验收-p0p4-与交付物)  
11. [技术细节：状态、Token 与各层机制](#11-技术细节状态token-与各层机制)  
12. [实现路线：NUC 原型与 TTT3R 改造](#12-实现路线nuc-原型与-ttt3r-改造)  
13. [借鉴 Attention、梯度与 LoGeR 的边界](#13-借鉴-attention梯度与-loger-的边界)  
14. [2026 年 3 月前后相关工作速览](#14-2026-年-3-月前后相关工作速览)  
15. [参考资料](#15-参考资料)

---

## 1. 核心命题与一句话定位

### 1.1 问题意识

长程在线重建的主要失败模式往往**不是**「backbone 不够大」，而是**记忆如何组织**：

- **局部高保真**需要近期高分辨率信息；  
- **长期稳定**需要慢变锚点与可检索历史；  
- **固定预算**要求主动裁剪与归档；  
- **重访与回环**要求历史**可恢复**，而不是被均匀遗忘。

单一 online state 或无限膨胀的全局地图难以同时满足以上四点，因此需要**显式 lifecycle**：**活动区维护、历史区归档、重访时检索与恢复**。

### 1.2 神经视角（TTT3R 扩展）

> 以 **TTT3R** 的置信驱动在线状态更新为底座，以 **OVGGT** 的常数预算缓存与 anchor 保护为内存控制，以 **MERG3R / 经典 SLAM 子图思想** 为归档与恢复结构，构建面向长序列的**分层记忆**重建系统。

答辩式表述：

> 我们不是在 TTT3R 上再加一个语义头，而是把单一在线状态升级成**可写入、可裁剪、可归档、可恢复**的分层记忆系统。

### 1.3 边缘视角（HMR3D-GS Lite，见 `edge_gaussian_report`）

> **轻前端（如 cuVSLAM）**保持实时位姿主路径；**HMR3D** 管理 `M_short / M_active / B_bank` 与 **archive / retrieve / recover**；**Gaussian 仅表示当前 active 子图**；历史在 **CPU/NVMe bank**。  
> 核心命题：**在 Jetson 级预算下，让地图具备可持续的 memory lifecycle，Gaussian 只是活动局部的高质量表达层。**

### 1.4 刻意不强调什么

- 不把「首次 Gaussian SLAM」当主创新；  
- 不把语义双状态当 headline（语义可降为辅助：动静态过滤、检索增强、评测）；  
- 不把门控写成不可解释的大黑盒（优先可解释的重要性与规则组合）。

---

## 2. 文档与报告索引（建议阅读顺序）

| 文档 | 路径 | 内容侧重 |
|------|------|----------|
| **边缘系统报告（PDF）** | `latex/edge_gaussian_report.pdf`（源 `latex/edge_gaussian_report.tex`） | Jetson 动机、与手机扫描差异、**HMR3D-GS Lite** 模块图、生命周期公式、**预算表 / 实时性表 / 三线程**、RQ 与基线、stress tests、验证矩阵、P0–P4、风险与 fallback |
| **统一方法长文（LaTeX）** | `latex/main.tex` | **HMR3D** 命名、TTT3R/VGGT/OVGGT/MERG3R/LoGoPlanner 职责划分、分层记忆数学叙述、评测与实现清单 |
| **课程 Proposal** | `latex/project_proposal.tex` | 英文叙事与假设陈述 |
| **可行性长文** | `latex/项目可行性分析与验证.tex` / `latex/feasibility_validation_from_pdf.tex` | 答辩级「像不像拼装」辨析、与 LoGeR 对照叙事 |
| **NUC 入口** | `NUC_START_HERE.md` → `nuc/README.md` → `docs/NUC_TASKS.md` | CPU 原型阶段任务与验收 |
| **前端选项** | `docs/TRACKING_FRONTEND_OPTIONS.md` | tracking 与 HMR3D lifecycle 的衔接讨论 |
| **NUC 架构** | `docs/NUC_ARCHITECTURE.md` | 原型架构说明 |

**README 与报告的关系：** 本文件是**总览与进度源**；细节证明、公式与表格以 **`edge_gaussian_report`** 与 **`main.tex`** 为准。

---

## 3. 仓库结构

```text
CVPR/
  README.md                 # 本文件
  NUC_START_HERE.md         # NUC 阶段入口
  latex/                    # 报告与论文草稿（含 edge_gaussian_report）
  docs/                     # NUC 任务、架构、前端选项等
  nuc/                      # 无 CUDA 的 memory lifecycle 原型
  TTT3R/                    # 神经在线重建底座（计划沿此扩展分层记忆）
  CUT3R/                    # 相关基线代码（按需）
```

---

## 4. Git：克隆方式与 main / master 分支说明

### 4.1 正确克隆

远端：`https://github.com/AlanZhu2006/CVPR`

- **错误（会 `repository not found`）：**  
  `git clone https://github.com/AlanZhu2006/CVPR/tree/master`  
  （浏览器路径，不是 Git URL）
- **正确：**  
  `git clone https://github.com/AlanZhu2006/CVPR.git`

同步默认开发分支：

```bash
git pull origin main
```

### 4.2 `main` 与 `origin/master` 的区别（重要）

- **`main`（本工作区默认）：** 课程与系统向**大仓库**：`nuc/`、`latex/`、内嵌 `TTT3R/`、`docs/` 等，适合报告 + NUC + 后续在 `dust3r` 上改代码。  
- **`origin/master`：** 与 `main` **并非同一 Git 根历史**（无共同 merge-base），是另一套较小的 **HMR3D 编排树**：围绕 `third_party/TTT3R` 子模块、`hmr3d_memory/`（`MemoryRouter`、`adapter`、几何验证、pose-anchor、Mem3R 实验脚手架等）与评测脚本。  

若需阅读或对比 `master` 上的实现，建议：

```bash
git fetch origin master
git worktree add ../CVPR-master origin/master
```

**合并两条线**需要显式策略（拣选文件、子树或手动移植），不能假设一次 `merge` 无冲突完成。

---

## 5. 当前进度（与报告对齐）

| 轨道 | 内容 | 状态（截至 2026-04-13） |
|------|------|-------------------------|
| **文档** | `edge_gaussian_report`、`main.tex`、可行性文档 | **完整**：命题、模块、评测与风险已写清 |
| **NUC 原型** | `nuc/`：`observe → promote → archive → retrieve → recover` | **代码骨架就绪**：ORB 类轻前端 + `memory_router`；验收按 `docs/NUC_TASKS.md` Phase 0–6 |
| **TTT3R 内线改造** | `TTT3R/src/dust3r/` 结构化 state、submap bank、router | **以 Phase 1 为起点**（见第 12 节）；与报告中的模块名一致 |
| **Jetson + cuVSLAM + Active GS** | HMR3D-GS Lite 全文 | **工程目标**：在 lifecycle 验证后再接，避免先绑死重量级表示 |
| **远端 `master` 分支** | 状态级 archive/retrieve/recover + 验证管线 | **另一仓库树**：已实现 TTT3R 外接记忆与实验配置；与 `main` 需手动对齐 |

**推荐下一步顺序：** NUC 上跑到 **retrieve / recover 可度量** → **TTT3R 最小 bank + 结构化 state** → 再 **cuVSLAM + active Gaussian**。

---

## 6. 双轨路线：神经主干 vs Jetson 边缘版

| 维度 | **HMR3D（神经 / TTT3R）** | **HMR3D-GS Lite（边缘报告）** |
|------|---------------------------|--------------------------------|
| 前端 | 序列帧 + 模型内 pose/几何 | **cuVSLAM** 实时位姿与事件信号 |
| 活动表示 | `state_feat` / `state_pos` / `mem` 等 | **Active Gaussian submap**（仅当前区域） |
| 归档 | 子图摘要、descriptor、anchors（设计） | **CPU/NVMe bank**，冻结子图包 |
| 创新叙事 | 分层 token + TTT3R/OVGGT/MERG3R | **Edge-oriented map lifecycle** + 预算内生存 |

两轨共享同一内核：**显式 archive / retrieve / recover**，不把长程责任压给单一状态或无限全局地图。

---

## 7. 系统架构：四层与数据流

系统命名为 **HMR3D**，由四层组成（对应四类 **token 维护职责**，而非泛泛模块分工）：

1. **短程层 `M_short`** — 「看清楚」：最近窗口高保真、query-based 聚合（LoGoPlanner 思想进短程，不进长程地图主体）。  
2. **活动层 `M_active`** — 「记得住」：当前区域 persistent 状态；对齐 **TTT3R** 的 `state_feat/state_pos/mem` 并扩展锚点/上下文双速率等。  
3. **归档层 `B_bank`** — 「存得下、找得回」：子图摘要、descriptor、anchors；对齐 **MERG3R/SLAM 子图** 思想。  
4. **调度层 `Router`** — 写入 / 晋升 / 裁剪 / 归档 / 恢复；组合 **TTT3R 置信写回** 与 **OVGGT 式重要性 + anchor 保护**，避免单一大 gate。

```text
输入帧流 x_t
    → M_short（局部工作区）
    → 候选 Delta_t、对齐置信、重要性、重访描述子 d_t
    → Router
         ├→ M_active（活动地图）
         └→ B_bank（归档库） ⟵ retrieve / recover
    → 输出：深度/点云/位姿/地图状态
```

**边缘版补充（报告图）：** `Stereo/RGB-D/IMU → cuVSLAM → HMR3D Memory Router → (M_short, M_active GS, B_bank) → 可选 mesh shadow**。

---

## 8. 生命周期、调度与 Jetson 预算

### 8.1 生命周期（与报告一致）

典型事件链：**observe → promote → archive → retrieve → recover**（必要时轻量 merge / refine）。

- **归档触发（概念）：** 预算压力、与当前区域 overlap 下降、活动片段年龄、场景切换或回环边界事件等。  
- **检索：** 描述子 top-k **不等于**直接闭环；需 **几何验证** 降低假阳性。  
- **恢复：** 通常**部分注入**（anchors + summary + 必要局部表示），再经**短窗 refine**，而非整库灌回 GPU。

### 8.2 调度原则（报告原意）

> **Tracking 是实时主线；memory 与 active map 更新只能消费 tracking 剩余预算，不得反向阻塞 tracking。**

因此：archive/retrieve/recover 为 **事件驱动**；active 高斯或神经更新放在 **keyframe 级 / 低频线程**。

### 8.3 分层实时性（摘自报告思想）

| 层级 | 目标频率量级 | 说明 |
|------|----------------|------|
| tracking | ~20–30 Hz | 位姿主路径 |
| keyframe | ~5–10 Hz | 建图不必每帧 |
| active map 更新 | ~1–5 Hz | 关键帧级即可 |
| archive / recover | 事件触发 | 场景切换、重访、歧义 |

### 8.4 Jetson 预算分层（报告表意）

- **cuVSLAM：** 最高优先级，避免被高斯优化拖死。  
- **active Gaussian：** GPU 常驻但**容量有上限**，局部更新与渲染。  
- **bank：** CPU/NVMe，可增长但条目需压缩与索引。  
- **可选全局 mesh/TSDF：** 稳定支撑，不必与高斯同频。

### 8.5 三线程模型（报告建议）

1. **实时主线程：** tracking、关键帧、loop signal。  
2. **记忆路由线程：** promote、预算、archive/retrieve 触发。  
3. **低频建图线程：** active 更新、recover 注入、局部 merge。

成本直觉（报告中的分层）：只有 tracking 是每帧硬约束；其余项带事件指示函数，可按 keyframe / 事件开关。

---

## 9. 研究问题、基线、Stress Tests 与验证矩阵

### 9.1 三个核心研究问题（RQ，来自 `edge_gaussian_report`）

1. **RQ1：** 固定显存下，**active/archive 解耦**是否优于单一膨胀活动地图？  
2. **RQ2：** **retrieve/recover** 是否降低 long-gap revisit 与 branch confusion 失败率？  
3. **RQ3：** 在 Jetson 上，**仅 active Gaussian submap** 是否在可接受实时性下优于 mesh-only 的局部质量或可查询性？

### 9.2 建议基线（报告）

1. Tracking + mesh（如 cuVSLAM + nvblox），无长期 archive/recover。  
2. 无 archive 的 **active GS 滑窗**。  
3. 有 archive 但 **无 retrieve**（只压缩）。  
4. **完整系统**：active GS + bank + retrieve/recover。

### 9.3 Stress Tests（报告 + `main.tex` 一致方向）

1. **Horizon Scaling** — 随序列长度误差与延迟退化。  
2. **Branch Disambiguation** — 重复结构身份混淆。  
3. **Forced Backtracking** — 远离再返回是否仍能利用历史。  
4. **Revisit Recovery** — retrieve/recover 后误差是否真下降。

### 9.4 验证矩阵与成功档次（报告）

验证需分层：**记忆假设 / 表示假设 / 部署假设**。报告给出「正向证据 vs 若不成立的信号」表（memory lifecycle、retrieve/recover、active GS、Jetson 在线触发）。

**结果分档简述：**

- **完整成功：** Jetson（或 Orin 级）全链路 + 固定预算下相对基线显著改善。  
- **部分成功：** 桌面侧证实 lifecycle 收益 + Jetson 最小展示证明在线触发。  
- **负结果仍有价值：** 若 GS 不划算但 lifecycle 有效，则叙事收缩为「边缘长程关键在记忆组织，不必绑定 Gaussian」。

---

## 10. 阶段验收 P0–P4 与交付物

### 10.1 分层验收（报告）

- **P0/P1：** 数据流、关键帧、`M_short/M_active/B_bank` 与 **archive/retrieve/recover 路由**跑通（active 可为 mesh-only 或占位）。  
- **P2：** active Gaussian patch 可局部更新与渲染，**不拖垮** tracking。  
- **P3/P4：** 固定预算下，在 revisit、branch、backtracking 等 stress 上相对 **no-archive / no-retrieve** 基线有可测收益。

### 10.2 阶段划分（报告）

- **P0：** 稳定基线与数据流。  
- **P1：** HMR3D 生命周期。  
- **P2：** Active Gaussian（仅活动区）。  
- **P3：** Jetson 裁剪与固定容量。  
- **P4：** 重访恢复与系统对比实验。

### 10.3 每阶段证据（报告）

每阶段应有可复核产出：tracking 频率与资源表、bank 样例与日志、局部渲染或更新样例、budget 曲线、stress test 图表等。

### 10.4 建议最终交付物

桌面可运行原型、Jetson 最小展示、以 horizon/branch/revisit 为核心的实验、以及**明确区分**「多少增益来自 lifecycle、多少来自 Gaussian 表示」的报告叙述。

---

## 11. 技术细节：状态、Token 与各层机制

### 11.1 总状态

```text
S_t = { M_short^t, M_active^t, B_bank^t }
```

当前帧输出包括但不限于：`Delta_t`、`c_align^t`、`s_import^t`、重访描述子 `d_t`、各类 query token（`q_pose`、`q_geo`、`q_desc`、`q_sum`）。

### 11.2 Token 类型（五类）

| 类型 | 符号 | 角色 |
|------|------|------|
| 观测 | `P_t` | 噪声大、寿命最短，主要停短程 |
| 局部锚点 | `A_t` | 稳定、可晋升至活动层 |
| 活动状态 | `H_t` | 对应 TTT3R 状态扩展 |
| 归档摘要 | `Z_j` | 子图级压缩 |
| 查询 | `Q` | 从记忆中读出任务相关摘要 |

**要点：** 短程与长程不「全员混更新」，而是由 **query 读出**再决定 promote/archive/recover。

### 11.3 短程层

```text
M_short^t = { L_recent^t, A_short^t, Q_short }
```

聚合形式（示意）：

```text
P_t = Enc(x_t, r_t, pi_t)
z_pose^t = Attn(q_pose, L_recent ∪ A_short)
z_geo^t  = Attn(q_geo,  L_recent ∪ A_short)
d_t      = Proj(Attn(q_desc, L_recent ∪ A_short))
```

晋升：`m_promote(i) = 1[s_import(i) > τ_s ∧ support(i) > τ_k]`。

### 11.4 活动层（双速率）

```text
M_active^t = { A_active^t, C_active^t }
```

- 锚点慢更新、上下文快更新；写回强度乘 **TTT3R 式** `c_align`。  
- 摘要：`S_active^t = Attn(q_sum, A_active ∪ C_active)`，服务后续归档。

### 11.5 内存控制（OVGGT 思想）

重要性可组合 residual、attention、几何锚、置信、**revisit** 等；动作为 `keep / promote / archive / evict`；**anchor 保护**后再裁剪。

### 11.6 归档包（MERG3R/SLAM 思想）

```text
B_j = { desc_j, latent_j, anchors_j, pose_j, conf_j, meta_j }
U_j = { S_active^j, A_active^j, T_j } → latent_j = Pool_Q(q_sum, U_j)
```

恢复后流程：**retrieve → 注入 M_active → 短程 refine**。

### 11.7 语义（若保留）

仅作辅助：动静态过滤、检索增强、长期一致性评测；**不作 headline**。

---

## 12. 实现路线：NUC 原型与 TTT3R 改造

### 12.1 NUC 阶段（CPU，无 CUDA）

- **入口：** `NUC_START_HERE.md`、`nuc/README.md`  
- **任务与验收：** `docs/NUC_TASKS.md`（Phase 0–6：环境 → tracking → Router → archive → retrieve → recover → 对照实验）  
- **一键收口：** `bash nuc/scripts/close_phase6.sh`  
- **模块：** `nuc/src/nuc_runtime/tracking.py`、`memory_router.py`、`descriptors.py`、`io.py`、`output.py`

**边界：** NUC tracking 为**占位前端**；价值在 **lifecycle 逻辑与日志可验收**，不是最终 SLAM 精度。

### 12.2 TTT3R 改造（`CVPR/TTT3R/`）

建议顺序（与旧版路线图一致，浓缩）：

1. **`demo.py`**：增加 `--memory_budget`、`--short_window`、`--enable_submap_bank` 等参数。  
2. **`inference.py`**：将 `state_args` 提升为结构化对象（dict/dataclass）。  
3. **`model.py`**：`forward_recurrent_lighter` 等路径上输出 importance、anchor mask；写回前 keep/evict；段末导出子图摘要。  
4. **新文件（建议）：** `hier_state.py`、`submap_bank.py`、`revisit.py`、`memory_router.py`（与 NUC 概念对齐，便于以后统一接口）。

### 12.3 评测扩展

在 ATE/RPE/深度外，增加：重访命中率、恢复前后误差变化、bank 压缩比、峰值显存、延迟、每百帧漂移等。

---

## 13. 借鉴 Attention、梯度与 LoGeR 的边界

**原则：借机制，不整套换架构。**

- **Attention 适合：** archive **保留决策**、retrieve **读 bank**、recover **部分注入**（小规模 Q/K/V，避免全历史 global attention）。  
- **梯度 / TTT 式适应适合：** **active 局部**少量步 refine、recover 后对齐、descriptor/summary 的**极轻量**修正；配合**置信度调学习率**。  
- **不建议：** 全序列 Transformer 式历史 attention；把整个 bank 压成单一 fast-weight 隐状态（损害可归档、可检索、可控预算）。  
- **LoGeR：** 作**长序列混合记忆**的对标与思想参考（仓库外或本地 `LoGeR/` 可参考），**不作为**主线工程依赖；主线保持 **显式 bank + lifecycle**。

论文式表述：

> 在显式 memory lifecycle 下引入轻量 **attention 式 readout/筛选** 与 **置信度感知局部 refine**，提升 archive/retrieve/recover；**主干贡献仍为分层记忆与生命周期**，而非通用 Transformer 或端到端 TTT 记忆替代 bank。

---

## 14. 2026 年 3 月前后相关工作速览

| 工作 | 要点 | 在本项目中的用法 |
|------|------|------------------|
| [OVGGT](https://arxiv.org/abs/2603.05959) | 常数成本缓存、anchor 保护 | 预算与重要性筛选 |
| [MERG3R](https://arxiv.org/abs/2603.02351) | 分治、子图、对齐 | 归档/恢复/合并**结构**参考，非一上来全量 BA |
| [TTT3R](https://arxiv.org/abs/2509.26645) | 对齐置信 → 更新率 | **活动层**主写入规则 |
| [StreamVGGT / 4D VGGT](https://arxiv.org/abs/2507.11539) | 因果局部高质量 | **短程层**候选，不承担无限长程记忆 |
| [LoGeR](https://arxiv.org/abs/2603.03269) | 长上下文混合记忆 | **对照组与思想**，非模块拼装核心 |
| [CUT3R](https://arxiv.org/abs/2501.12387) | 持续状态基线 | 对比与实现参考 |
| [VGGT](https://arxiv.org/abs/2503.11651) | 强几何 backbone | 局部/离线增强可选 |

---

## 15. 参考资料

### 15.1 论文与项目链接

- [OVGGT](https://arxiv.org/abs/2603.05959)  
- [MERG3R](https://arxiv.org/abs/2603.02351)  
- [TTT3R](https://arxiv.org/abs/2509.26645)  
- [LoGeR](https://arxiv.org/abs/2603.03269)  
- [Streaming 4D Visual Geometry Transformer](https://arxiv.org/abs/2507.11539)  
- [CUT3R](https://arxiv.org/abs/2501.12387)  
- [VGGT](https://arxiv.org/abs/2503.11651)  
- [TTT3R 代码](https://github.com/Inception3D/TTT3R)  
- [CUT3R 代码](https://github.com/CUT3R/CUT3R)  
- [cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM)  
- [Isaac ROS Visual SLAM](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/index.html)  
- [Isaac ROS nvblox](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/index.html)  
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)  
- [gsplat](https://github.com/nerfstudio-project/gsplat)  

### 15.2 最终对外叙事建议

优先使用：

- **分层记忆长程在线三维重建**  
- **SLAM 启发的子图归档与重访恢复**  
- **预算受限的神经/混合几何记忆系统**

避免把项目讲成「又一个语义 SLAM」或「仅换 backbone」。

---

*本 README 将 `latex/edge_gaussian_report` 与神经路线 `latex/main.tex` 对齐为单一进度叙事；细节公式、证明与完整表格以对应 TeX/PDF 为准。*
