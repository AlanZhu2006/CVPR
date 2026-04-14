# HMR3D v2 代码落地方案（对照 `HMR3D_v2_roadmap.md`）

> 本文档把路线图中的目标拆成**可合并进当前 `master` 线 `hmr3d_memory/` 的具体文件、类、接口与开源借鉴点**。  
> 基准代码结构以当前仓库为准：`MemoryRouter`（`router.py`）、`adapter.py` 中 `run_sequence_with_mode` / `_run_step`、`MemoryConfig`（`config.py`）、`eval_relpose.py`、`summarize.py`。

---

## 0. 总原则（与路线图一致）

1. **不改 `third_party/TTT3R` 子模块源码**；所有扩展放在 `hmr3d_memory/` 与 `scripts/`。  
2. **先规则、后学习**：v2 第一版全部用标量特征 + 阈值 + 可记录决策原因；第二版再在 gate 上接 MLP。  
3. **可观测性优先**：每个 gate 必须写入 `events` 或独立 `decision_log.jsonl`，便于算「recover pollution」「write acceptance ratio」。  
4. **评测契约不变**：`cut3r` / `ttt3r` / `hmr_archive_only` / `hmr_full` 行为在默认 config 下与 v1 数值对齐（回归测试）。

---

## 1. 开源仓库与方法：借什么、从哪里看

| 来源 | 链接 / 仓库 | 建议借鉴的内容 | 不建议照搬 |
|------|-------------|----------------|------------|
| **TTT3R** | https://github.com/Inception3D/TTT3R | `dust3r` 中 **cross-attn → sigmoid → update_mask** 的思想；**confidence 调制更新强度**（对照本仓库 `adapter._compute_update_masks`） | 整套模型 |
| **CUT3R** | https://github.com/CUT3R/CUT3R | 与 TTT3R 对比的 **固定更新掩码** 基线 | 作为无记忆对照即可 |
| **LoGeR** | https://github.com/junyi42/LoGeR（或论文实现） | **chunk 边界**、跨 chunk **状态 handoff** 的调度直觉；可参考其 eval 脚本组织长序列 | 整套 hybrid memory 替换 bank |
| **Mem3R** | arXiv:2604.07279；实现若开源则对照 **pose vs geometry 分支** 与 **gate** | **职责拆分**（pose / geometry / gate）、**写入门控** 的接口形态 | 隐式 fast-weight 作为唯一记忆 |
| **MERG3R** | arXiv:2603.02351；实现若公布可看 **partition / align** | **多子块 + 置信加权融合** 的流程设计；工程上可参考 **COLMAP** `model_merge`、**ORB-SLAM3** MapPoint fusion 的「加权平均」模式 | 离线大规模 BA 作为在线默认路径 |
| **ZipMap** | arXiv:2603.04385 | **query 读 scene state** 的接口设计（query tensor → readout） | 非流式主干 |
| **Scal3R** | arXiv:2604.08542 | **global context** 压缩为向量、在测试时轻量适配的叙事 | 完整子网络第一版就上 |
| **本仓库已有** | `hmr3d_memory/mem3r_probe.py`、`adapter._compute_update_masks` | Mem3R 脚手架与 **state_gate** 相关字段已在 `MemoryConfig` | 已满足第一版 gate 实验时可复用变量名与 device 约定 |

**工程习惯**：每个 v2 子模块顶部用 5 行注释写清「对应论文/仓库的哪一节 + 我们只实现了哪一条机制」。

---

## 2. 目标目录结构（v2 增量）

在 `hmr3d_memory/` 下**新增包级子模块**（避免单文件过大）：

```text
hmr3d_memory/
  __init__.py              # 导出 v2 公共 API（可选）
  config.py                # 扩展 MemoryConfig / 或 MemoryConfigV2 继承
  router.py                # MemoryRouter：变薄，委托 policies
  adapter.py               # 增加 local_adaptation_hook 调用点
  policies/
    __init__.py
    write_policy.py        # WritePolicy 规则版
    retrieve_policy.py     # RetrievePolicy：coarse+fine（第一版 coarse 可为规则）
    verify_policy.py       # 可选：从 router 抽离几何/pose 阈值逻辑
    recover_policy.py      # RecoverPolicy：注入比例、recover gate
    merge_policy.py        # MergePolicy：多候选加权融合
  bank/
    __init__.py
    hierarchical_bank.py   # HierarchicalMemoryBank：L1 entries + L2 scene nodes
    scene_summary.py       # 在线更新 scene descriptor / k-means 质心等
  adaptation/
    __init__.py
    local_adapt.py         # recover 后 1~3 步小更新（仅动 summary/辅助张量）
  logging/
    __init__.py
    decision_log.py        # 结构化决策日志 + 与 eval 对齐的 schema
```

**第一迭代**可只加 `policies/write_policy.py`、`policies/recover_policy.py` 两个文件 + `logging/decision_log.py`，其余先放 `router.py` 内联类，第二迭代再拆。

---

## 3. Phase A（路线图 Week 1）：Write Gate + Recover Gate

### 3.1 现状锚点

- **写入**：`MemoryRouter.should_archive()` 仅按 `archive_interval`；`archive()` 收集 `ArchiveEntry`。  
- **恢复**：`propose_recovery()` → adapter 几何验证 → `accept_recovery()` / `reject_recovery()`；已有 `recovery_alpha`。

### 3.2 新增配置字段（`config.py`）

在 `MemoryConfig` 中增加一组 **v2 开关与阈值**（默认值使行为与旧版一致，`enable_v2_write_gate=False` 时零差异）：

```python
# Write gate
enable_v2_write_gate: bool = False
write_min_segment_novelty: float = 0.0       # 与上一段 segment descriptor 最小 cos 距离
write_min_state_confidence: float = 0.0    # 若 adapter 传入 conf，低于则延迟 archive
write_delay_frames_on_low_conf: int = 0    # 延迟 archive 的帧数
archive_quality_score_thresh: float = 0.0  # 综合分低于则标记低质 archive（仍写入但打标）

# Recover gate
enable_v2_recover_gate: bool = False
recover_max_injection_alpha: float = 1.0   # 上限夹住 recovery_alpha
recover_min_pose_agreement: float = 0.0    # 与 archive pose 的最小一致性（已有 anchor 可复用）
recover_blend_with_identity: bool = True   # 低置信时更接近「不注入」
```

### 3.3 新类 `WritePolicy`（`policies/write_policy.py`）

**职责**：在 `should_archive` 为真时，再返回「是否允许本帧归档」「是否延迟」「archive 元数据标签」。

建议接口：

```python
@dataclass
class WriteDecision:
    accept: bool
    delay_frames: int = 0
    reason: str = ""                    # e.g. "interval", "novelty_low", "conf_low"
    quality_score: float = 1.0        # 0~1，供 bank 与日志使用

class WritePolicy:
    def __init__(self, config: MemoryConfig): ...
    def decide(
        self,
        frame_idx: int,
        segment_descriptors: list[torch.Tensor],
        last_archive_descriptor: torch.Tensor | None,
        optional_state_conf: float | None,
    ) -> WriteDecision: ...
```

**规则版逻辑（第一版）**：

1. **Novelty**：`segment_desc` 与 `last_archive` 的 segment descriptor 余弦相似度若 **高于** `1 - write_min_segment_novelty`，认为「场景未变」，可 `delay` 或 `reject`（与路线图「延迟 archive」一致）。  
2. **Confidence**：若 `adapter` 在每帧传入 `last_geo_conf`（从 `StepResult` 或现有 metric 取），低于阈值则 `delay_frames = write_delay_frames_on_low_conf`。  
3. **Quality score**：`quality = w1 * novelty + w2 * conf + w3 * (tail sequence consistency)`，权重写死在 policy 或 config。

**接入点**：在 `MemoryRouter.archive()` **开头**调用 `WritePolicy.decide`；若 `not accept`，不创建 `ArchiveEntry`，更新 `stats["write_gate_rejects"]` + `events`。

### 3.4 新类 `RecoverPolicy`（`policies/recover_policy.py`）

**职责**：在 `accept_recovery` **之前**（几何验证已通过时），调制 `recovery_alpha` 与是否允许注入。

```python
@dataclass
class RecoverDecision:
    allow: bool
    effective_alpha: float
    reason: str

class RecoverPolicy:
    def decide_after_verify(
        self,
        proposal: RecoveryProposal,
        verify_geo_gain: float,
        verify_conf_delta: float,
        current_pose_quality: float,
        archive_pose_quality: float,
    ) -> RecoverDecision: ...
```

**规则版**：

- 若 `verify_geo_gain` 接近 0，`effective_alpha *= 0.5`（借鉴 TTT3R「小步更新」）。  
- 若 anchor pose 已拒绝过类似候选，`allow=False`（与现有 `anchor_pose` 逻辑组合，不重复则委托）。  
- `effective_alpha = min(proposal.recovery_alpha, recover_max_injection_alpha)`。

**接入点**：`adapter.py` 中在调用 `router.accept_recovery` 前插入 `RecoverPolicy.decide_after_verify`；或在 `router.accept_recovery` 内部首行调用。

### 3.5 统计与日志

- `MemoryRouter.stats` 增加：`write_gate_accepts`, `write_gate_rejects`, `write_gate_delays`, `recover_gate_alpha_reduced`, `recover_gate_blocks`。  
- `hmr3d_memory/logging/decision_log.py`：  
  - `DecisionRecord` dataclass（frame_idx, policy, decision, reason, scalars）  
  - `append_jsonl(path, record)`  
- `eval_relpose.py`：增加 CLI `--decision-log-dir`；序列结束 flush。

### 3.6 测试（`tests/`）

| 文件 | 内容 |
|------|------|
| `test_write_gate.py` | 高相似连续段 → 应 delay/reject；`enable_v2_write_gate=False` 与旧行为一致 |
| `test_recover_gate.py` | 低 geo_gain → alpha 降低；边界阈值快照 |

### 3.7 配置命名（`configs/`）

新增：

- `configs/hmr_v2_writegate_tum_224_baseline.json`（从 `tum_relpose_sweep_224_v11_pose_anchor_full7.json` 复制并只开 write gate）  
- `configs/hmr_v2_recovergate_tum_224_baseline.json`  

**回归**：对同一序列跑 `hmr_full` 旧 config 与新 config（gate 全关）ATE 差应为 0（deterministic）。

---

## 4. Phase B（路线图 Week 2）：Hierarchical Bank + Coarse-to-Fine Retrieve

### 4.1 数据结构（`bank/hierarchical_bank.py`）

```python
@dataclass
class SceneNode:
    scene_id: int
    centroid_descriptor: torch.Tensor   # 归一化，CPU
    member_archive_ids: list[int]
    last_update_frame: int

@dataclass
class HierarchicalMemoryBank:
    entries: dict[int, ArchiveEntry]    # archive_id -> entry（或保持 list + id 索引）
    scenes: list[SceneNode]
    entry_to_scene: dict[int, int]      # archive_id -> scene_id
```

**L1**：现有 `ArchiveEntry` 不变。  
**L2**：`SceneNode` 维护 **若干 entry 的 descriptor 质心**（Scal3R 思想的极简版）。

### 4.2 在线聚类（第一版：`bank/scene_summary.py`）

不训练网络，可选两种：

1. **固定 K 场景**：维护 `K` 个质心，新 entry descriptor 与质心 cosine 最近则并入；否则新开 scene（上限 `max_scenes` 时合并最近两簇）。  
2. **时间滑窗**：仅对最近 `M` 个 entry 做 batch k-means（每 `archive` 触发一次，k 小）。

借鉴：**sklearn.cluster.MiniBatchKMeans**（仅 CPU、离线批处理脚本也可）；在线路径用纯 torch CPU 避免依赖。

### 4.3 检索两阶段（`policies/retrieve_policy.py`）

替换/包装 `MemoryRouter.propose_recovery` 中「全 bank 线性扫」的前半段：

```python
class RetrievePolicy:
    def coarse_candidates(
        self,
        query_desc: torch.Tensor,
        bank: HierarchicalMemoryBank,
        top_scenes: int = 2,
    ) -> set[int]:  # archive_id 集合
        ...

    def fine_rank(
        self,
        query_desc: torch.Tensor,
        entries: list[ArchiveEntry],
    ) -> list[tuple[int, float]]:  # (archive_id, score)
        ...
```

**流程**：`coarse` 只在 `SceneNode.centroid` 上 top-k → 展开 member entries → 原有 `fine` 相似度 + gap + sequence 逻辑。

### 4.4 `MemoryRouter` 改动

- 将 `archive_bank: List[ArchiveEntry]` 改为持有 `HierarchicalMemoryBank`（内部仍可 `list` 兼容）。  
- `archive()` 末尾调用 `bank.on_new_entry(archive_id, descriptor)` 更新 scene。  
- `stats` 增加：`scene_routing_hits`, `coarse_miss_fallback_full_scan`（第一版 coarse 失败则回退全扫描，保证鲁棒）。

### 4.5 配置

```python
enable_v2_hierarchy: bool = False
hierarchy_top_scenes: int = 2
hierarchy_max_scenes: int = 16
```

`configs/hmr_v2_hierarchy_tum_224.json`。

### 4.6 测试

`test_scene_summary_routing.py`：构造 3 个 scene、人工 descriptor，断言 coarse 只返回相关 scene 内 id。

---

## 5. Phase C（路线图 Week 3）：Multi-Candidate Merge + Local Adaptation

### 5.1 Multi-candidate（`policies/merge_policy.py`）

现状：`retrieval_topk` 可能 >1，但最终往往选一个。v2：

1. 对 **通过几何验证** 的多个 `RecoveryProposal` 计算 `confidence_i`（例如 `geo_gain * sigmoid(conf_delta)`）。  
2. **Pose-anchor 融合**：对 SE(3) 用 **加权平均李代数** 或对平移/旋转分别加权（借鉴 MERG3R「confidence-weighted」思想；实现参考 **Eigen::Quaterniond slerp** 或 `scipy.spatial.transform` 仅用于 CPU 后处理）。  
3. **状态张量融合**：`state_feat_merge = sum w_i * state_feat_i`（`w_i` softmax(confidence)），再归一化；**仅在 adapter 内**做一次，不写回 bank。

接口：

```python
@dataclass
class MergedRecovery:
    blended_state_args: StateTuple
    blend_weights: list[float]
    source_archive_ids: list[int]

class MergePolicy:
    def merge(self, accepted: list[RecoveryProposal], weights: list[float]) -> MergedRecovery: ...
```

**接入点**：`adapter._run_step` 中在多个候选都 accept 时调用；若仅 1 个，路径与现网一致。

### 5.2 Local adaptation（`adaptation/local_adapt.py`）

**目标**：recover 后 **1~3 步**，只更新「外挂」量，**不改 TTT3R 权重**。

第一版可选：

1. **Exponential moving average**：`running_summary = beta * running_summary + (1-beta) * new_desc`（`running_summary` 存 `MemoryRouter` 或 adapter 上下文）。  
2. **小步梯度**：若 `mem3r_probe` 已有 fast weight 路径，仅对 **probe 头**用 `loss = 1 - cos(pred_desc, current_obs_desc)`，`steps=1~3`，`lr` 来自 `verify_conf`（TTT3R 风格）。

接口：

```python
def local_adapt_after_recover(
    adapter_state: Any,
    merged: MergedRecovery,
    current_view: dict,
    num_steps: int,
    lr: float,
) -> Any: ...
```

**接入点**：`adapter.run_sequence_with_mode` 在 `accept_recovery` 成功后、`next` 帧前调用。

配置：

```python
enable_v2_local_adapt: bool = False
local_adapt_steps: int = 1
local_adapt_lr_max: float = 1e-3
```

### 5.3 测试

`test_multicandidate_merge.py`：两个假 proposal，检查权重和为 1、融合后 tensor 形状。  
`test_local_adaptation.py`：mock loss 下降一步。

---

## 6. `eval_relpose.py` / `summarize.py` / `scripts` 改动清单

### 6.1 新模式字符串（与路线图 §13.1 对齐）

在 `MemoryConfig.for_mode` 或 `run_sequence_with_mode` 的 mode 分支增加：

- `hmr_v2_writegate`  
- `hmr_v2_hierarchy`  
- `hmr_v2_merge`  
- `hmr_v2_local_adapt`  
- `hmr_v2_full`（组合开关，内部 `enable_v2_*` 全 true）

实现方式：**优先**用单一 `hmr_full` + JSON 覆盖 `enable_v2_*`，避免 mode 组合爆炸；mode 仅作 sweep 标签。

### 6.2 `summarize.py`

增加聚合：

- `write_acceptance_ratio = writes / (writes + rejects + delays)`  
- `recover_pollution_proxy`：recover 后 `N` 帧内 ATE 急剧变差的次数 / 总 recover（需位姿缓冲，可从 eval 输出解析）  
- `scene_routing_hit_rate`：从 decision log  
- `merge_used_count`：多候选 merge 触发次数  

### 6.3 `scripts/run_relpose_memory_sweep.py`

增加 preset：`hmr_v2_ablation_grid.json` 指向 5 个 config 文件批量跑。

---

## 7. 与 `MemoryRouter` 方法映射表（重构指南）

| 现有方法 | v2 动作 |
|----------|---------|
| `should_archive` | 先 interval；再 `WritePolicy` |
| `archive` | 写入 `HierarchicalMemoryBank`；记录 `quality_score` |
| `propose_recovery` | 前插 `RetrievePolicy.coarse` + `fine` |
| `accept_recovery` | 前插 `RecoverPolicy`；可选后接 `MergePolicy` |
| `observe` | 可向 `WritePolicy` 提供 tail 统计 |

**重构技巧**：第一轮 **不删**原逻辑，用 `if config.enable_v2_*` 分支；全部稳定后再抽类。

---

## 8. 风险对策在代码中的体现（对照路线图 §14）

| 风险 | 代码对策 |
|------|----------|
| 层级过复杂 | `hierarchy_top_scenes=1` 等价于关闭 coarse；单测强制回退路径 |
| learned gate 不稳 | `policies/learned_gate.py` 单独文件，默认不 import |
| merge 误差累积 | `MergePolicy` 要求 `max_candidates=2` 默认；`confidence` 低于阈值的候选丢弃 |
| local adapt 失控 | `num_steps` 上限 3；`lr` 上限；仅在 `recover` 事件触发 |
| 项目发散 | `docs/HMR3D_v2_code_implementation_plan.md` 与 roadmap 列为唯一架构源；PR 必须改此表勾选完成项 |

---

## 9. 建议提交顺序（Git）

1. `feat(v2): decision log + stats schema`（无行为变化）  
2. `feat(v2): WritePolicy + config + tests`  
3. `feat(v2): RecoverPolicy + adapter hook + tests`  
4. `feat(v2): HierarchicalMemoryBank + RetrievePolicy`  
5. `feat(v2): MergePolicy + optional multi-candidate path`  
6. `feat(v2): local_adapt_after_recover`  
7. `chore: configs/hmr_v2_*.json + sweep preset`  

每步前后跑：**同一 TUM 窗口 + deterministic + hmr_full 旧 config** 做数值回归。

---

## 10. 验收检查表（Definition of Done）

- [ ] `enable_v2_*` 全 false 时，与当前 `v11` manifest 的指标 bitwise/数值一致（允许浮点容差文档化）。  
- [ ] 每个新 policy 有单元测试 + 至少一条 JSON fixture。  
- [ ] `summarize.py` 能读 decision log 输出 RQ1 所需比率。  
- [ ] README 或 `HMR3D_v2_roadmap.md` 增加「指向本文件」的一节。

---

*文档版本：与 `HMR3D_v2_roadmap.md` 第 10–12、15 节对齐；实现中若 `router.py` 行号变化，以 `MemoryRouter` 公开方法名为准。*

---

## 11. 落地状态（代码已合入仓库）

以下模块已在 `hmr3d_memory/` 中实现并通过 `python3 -m py_compile` 语法检查；**运行 `pytest` 需要已安装 `torch` 的环境**（与 TTT3R 一致）。

| 组件 | 路径 | 说明 |
|------|------|------|
| Write / Recover Policy | `hmr3d_memory/policies/write_policy.py`, `recover_policy.py` | 默认关闭，与 v1 行为一致 |
| Merge | `hmr3d_memory/policies/merge_policy.py` | `enable_v2_merge` 且多候选通过时加权融合 state/mem |
| Coarse retrieve | `hmr3d_memory/policies/retrieve_policy.py` + `bank/hierarchical_bank.py` | `enable_v2_hierarchy`；空子集回退全库扫描并计 `coarse_miss_fallback_full_scan` |
| Local adapt | `hmr3d_memory/adaptation/local_adapt.py` | recover 成功后对 `next_state` 做保守混合 |
| Router | `hmr3d_memory/router.py` | `can_archive`、`rebuild_proposal_with_alpha`、`_proposals_for_entries`（silent 二次扫描不计重复统计） |
| Adapter | `hmr3d_memory/adapter.py` | `can_archive` + 多候选 + recover gate + `local_adapt`；`MemoryConfig.for_mode` 增加 `hmr_v2_writegate` / `hmr_v2_full` |
| Config | `hmr3d_memory/config.py` | 全部 `enable_v2_*` 与阈值字段 |
| 日志 | `hmr3d_memory/logging/decision_log.py` | 可选 JSONL |
| 单测 | `tests/test_write_gate.py`, `test_recover_gate.py`, `test_hierarchical_bank.py` | 依赖 `torch` |

示例配置：`configs/hmr_v2_smoke_defaults.json`（全关，用于 JSON merge 模板）。

**运行测试：**

1. 安装测试依赖（含 `einops`，跑完整 `adapter` / TTT3R 时需要）：  
   `cd CVPR && pip install -r requirements-test.txt`
2. 若系统 **source 了 ROS 2 Humble**，pytest 会被 `launch_testing*` 插件劫持，请使用：  
   `bash scripts/run_pytest.sh -q`  
   或手动：  
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
3. **anchor 单测**已改为从 `hmr3d_memory.anchor_gate` 导入，不再在收集阶段加载 `einops`。
4. **一键本地验证（pytest + 最小 TUM fixture relpose + smoke 含 `hmr_v2_full`）**：`bash scripts/run_validation_suite.sh`（需权重 `checkpoints/cut3r_512_dpt_4_64.pth`；fixture 写入 `data/long_tum_s1/smoke_seq/`，目录已在 `.gitignore` 的 `data/` 下）。
5. **v2 与 v1 的定量对比（fixture，短序列）**：`python3 scripts/prepare_tum_relpose_smoke_fixture.py` 后执行  
   `python3 scripts/run_relpose_memory_sweep.py --config configs/tum_relpose_v2_smoke_metrics.json`，再  
   `python3 scripts/print_relpose_leaderboard.py reports/generated/tum_relpose_v2_smoke_metrics/leaderboard.json`  
   会生成 `METRICS_TABLE.md` / `METRICS_TABLE.tsv`。短序列上 **ATE 可能与 v1 相同**；请看各 trial 目录下 `smoke_seq/memory_stats.json` 里的 **`write_gate_*`、`recover_gate_*`、`merge_events`、`local_adapt_applied`** 等 v2 计数是否按预期变化。真实 ~8% ATE 类结论需 **`configs/tum_relpose_sweep_224_v7_geometry_multi_tum.json`** 与完整 `data/long_tum_s1`。

**GPU 拉仓、补丁、训练/下一步总览**：见 [gpu_setup_and_training_next_steps.md](gpu_setup_and_training_next_steps.md)。

**测 v2 在 TUM 相对位姿上的效果：** 在现有 sweep 的 JSON 里 merge 进 `configs/hmr_v2_tum_relpose_overlay.json`（或把其中字段拷进你的 manifest），并把 `mode` 设为 `hmr_full` 或继续用 `hmr_full` + `memory` 字段覆盖；若 sweep 脚本支持 `hmr_v2_full` mode，可直接切到该 mode。对比同一 manifest 下 **overlay 全关 vs 全开** 的 ATE 曲线即可。真实 TUM 长序列仍用 `configs/tum_relpose_sweep_224_v2_ablation.json`（需自备 `data/long_tum_s1/...`）；无数据时可用 `configs/tum_relpose_smoke_fixture.json` 做端到端冒烟。
