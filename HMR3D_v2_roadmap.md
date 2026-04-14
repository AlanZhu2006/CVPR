# HMR3D v2 技术路线图（基于当前 master 线）

> 面向当前 `master` 分支的下一阶段设计文档  
> 目标：把现有的 **TTT3R 状态外挂记忆 + 检索 + 几何验证 + pose-anchor**，推进成一个更稳、更可解释、更具论文叙事完整性的 **HMR3D v2**。

---

## 0. 这份文档解决什么问题

你当前的 `master` 分支已经不再是“想法阶段”，而是一条可以运行、可比较、可评测的实验主线：

- 不改 `third_party/TTT3R` 上游，在外部实现 `archive → retrieve → verify → recover`
- 已有长期窗口 / 相对位姿评测与大量 sweep 配置
- 几何验证和 pose-anchor 已经拿到了可讲的提升
- `shadow recover`、`Mem3R scaffold`、更强恢复线还处于实验向阶段
- Gaussian / active submap 尚未进入 clean tree 的正式主线

这意味着：

**下一步最该做的，不是继续堆新的 memory 名词或急着把 Gaussian 拉回主线，而是把 HMR3D 这条 lifecycle 路线做厚、做稳、做成系统。**

换句话说，HMR3D 现在最缺的不是“能不能 recover”，而是：

1. **记忆怎么写入（write policy）**
2. **记忆怎么读取（query/readout policy）**
3. **恢复以后怎样稳定注入（recover stability）**
4. **多个历史块怎样合并成更强的长期地图（merge / hierarchy）**

这份文档的主结论是：

> **HMR3D v2 应该从“外挂检索记忆”升级成“有写入策略、有查询状态、有子图合并能力的生命周期记忆系统”。**

---

## 1. 当前 master 线的准确定位

按你现在的描述，当前分支最强的东西不是 Gaussian，而是：

### 1.1 已经成型的部分

- **状态级记忆生命周期**：`archive → retrieve → verify → recover`
- **统一长序列评测入口**：TUM 等长窗口上做 `cut3r / ttt3r / hmr_archive_only / hmr_full` 对比
- **几何验证 + pose-anchor**：已纳入主线叙事，并有相对成熟配置
- **文档**：已有 summary / review / handoff 文档
- **Mem3R scaffold**：有脚手架和导出，但不是主评测线
- **Shadow recover**：有 manifest 和实验线，但非默认 preset

### 1.2 还没有形成正式主线的部分

- **Gaussian / active submap**：未进入 clean tree
- **显式子图合并 / 多候选 merge**：尚未成为主叙事
- **scene-level summary / hierarchy**：bank 更多仍是局部 archive entry，而不是层级记忆系统
- **learned write policy**：看起来主要还是规则驱动和阈值驱动

### 1.3 因此，当前最合理的定位

HMR3D 当前最适合被表述为：

> **一个外挂在 TTT3R / CUT3R 状态流之上的 lifecycle memory system，能够对历史状态进行归档、检索、几何验证和恢复。**

这条线已经足以形成一篇“系统性 memory design”方向的工作雏形。

---

## 2. 为什么下一步不建议先拉回 Gaussian 主线

这件事要讲清楚，不然开发会再次发散。

### 2.1 因为当前 strongest validated line 不在 Gaussian

你现在最强的验证证据是：

- relpose / long-window evaluation
- geometry verification
- pose-anchor
- recover 相关叙事

而不是：

- active Gaussian mapping
- Gaussian-based retrieve / recover
- Jetson 上的局部高斯演示

### 2.2 因为 Gaussian 会把问题重新缠在一起

一旦你现在把高斯拉回来，立刻会同时引入：

- 表示层设计
- CUDA / GPU 资源问题
- 渲染 / densify / prune
- recover 与 Gaussian 参数注入
- 子图几何与高斯几何的双重一致性问题

这会导致你很难分清到底：

- HMR3D 本身是否有效
- 还是高斯实现拖慢了验证
- 还是 recover 本身不稳

### 2.3 所以正确顺序应该是

1. **先把 lifecycle memory 做强**
2. **再把 lifecycle memory 挂到 active Gaussian 上**

也就是说：

> **HMR3D v2 应该先成为“好的长期记忆系统”，然后再成为“高斯长期记忆系统”。**

---

## 3. HMR3D v2 的核心目标

我建议你把下一阶段明确写成下面这三个目标。

### Goal A：从“能检索”升级成“会写记忆”

当前 HMR3D 更像一个：

- 有 archive
- 有 retrieve
- 有 verify
- 有 recover

的系统。

下一步要让它变成：

- 知道什么该写进 archive
- 知道写成什么形式最利于以后恢复
- 知道什么时候该保留旧状态、什么时候该吸收新观测

也就是从 **read-heavy system** 升级成 **write-aware system**。

### Goal B：从“recover 成功”升级成“recover 稳定”

不是只证明：

- recover 能触发
- 命中历史块
- ATE 偶尔下降

而是证明：

- recover 后不会污染当前状态
- recover 的提升是稳定的，不依赖某几个幸运序列
- recover 可以与多候选、不同置信度、不同 bank 质量共存

### Goal C：从“历史块 bank”升级成“层级记忆系统”

最终你需要的不只是：

- 一个 archive entry 列表

而是：

- 局部 archive entries
- 场景级 summary / context
- 多候选合并与 consolidation

也就是：

> **HMR3D v2 = lifecycle memory + hierarchy + merge**

---

## 4. 各外部方法该怎么借：总原则

### 总原则

> **借机制，不借整套 backbone。**

你不应该把 HMR3D 改写成另一个 Mem3R、ZipMap 或 Scal3R。

因为你当前最有价值的主线是：

- 显式 archive bank
- 显式 retrieve / verify / recover
- 不改上游 TTT3R 主体

这条线最能形成“可解释的 lifecycle memory”叙事。

因此，所有外部工作都应该被看成：

- 某种 **增强写入的机制**
- 某种 **增强检索的机制**
- 某种 **增强恢复稳定性的机制**
- 某种 **增强层级记忆与合并的机制**

而不是新的主干替代品。

---

## 5. TTT3R：最值得直接借的方向

### 5.1 TTT3R 真正值得借什么

TTT3R 的核心不是“它有 memory”，而是：

- 把状态更新看成 online learning
- 根据 memory state 与当前观测的 alignment confidence
- 动态调节记忆更新的学习率
- 在 retaining history 与 adapting to new observations 之间做平衡

这非常适合 HMR3D，因为你已经有外部 memory router，但还缺少：

- 写入强度控制
- 恢复后的小步再适配
- 状态注入时的保守更新

### 5.2 我建议你在 HMR3D 里这样借

#### 方向 A：recover 后的局部微调

在当前流程中加入：

- 只对成功 recover 的候选触发
- 只做 1~3 步小更新
- 只更新少量可恢复参数（例如 summary latent / pose-anchor / adapter weights）
- 学习率由一致性分数决定

这一步的目的不是“重训练”，而是：

> **让 recover 成为 “检索 + confidence-aware adaptation”**

#### 方向 B：archive / promote 的写入强度控制

给每次写入定义一个 confidence / importance score，用来控制：

- 进入 archive 的比例
- 是否生成更高质量 summary
- 是否保留更多 anchors
- 是否延迟 archive

这会让你的 bank 质量显著上升。

### 5.3 不建议做什么

- 不要把整个 HMR3D 重新写成 TTT3R 风格 backbone
- 不要让所有 archive entry 都变成隐式 fast weights

你最需要的是 **TTT-style local adaptation**，不是 **TTT-only memory**。

---

## 6. Mem3R：最该借“职责解耦”和“写入门控”

### 6.1 Mem3R 的关键启发

Mem3R 不是简单改 update rule，而是把 memory architecture 重新设计成：

- **implicit fast-weight memory**：负责 tracking / pose
- **explicit token state**：负责 geometry
- **channel-wise state update gate**：控制写多少新信息、留多少旧信息

最重要的不是它的 RNN 形式，而是：

> **tracking memory 和 geometry memory 不应该是同一种东西。**

### 6.2 对 HMR3D 的具体借法

#### 借法 A：显式拆职责

把 HMR3D 内部显式拆成四类 memory role：

1. **Pose memory**
   - pose-anchor
   - recover 后位姿先验
   - relpose 约束

2. **Geometry memory**
   - archived structural tokens / summary
   - 局部几何摘要

3. **Retrieval memory**
   - descriptor
   - routing score
   - query/readout summary

4. **Recover adapter memory**
   - recover 后局部微调所需的临时状态

这样你的系统就不再只是一个“统一 bank + 几个阈值”。

#### 借法 B：写入门控（最推荐）

你现在最值得加入的是一个 **write gate / recover gate**。

##### write gate 用在：

- M_short → archive entry
- active state → summary latent
- keyframe → anchor retention

##### recover gate 用在：

- 多候选恢复时决定注入比例
- pose-anchor 是否覆盖当前值
- 旧信息和新观测的融合比例

第一版不用神经网络都可以：

- 规则 gate
- 几何一致性 gate
- descriptor-confidence gate

第二版再考虑做轻量 learned gate。

### 6.3 不建议借什么

不要借 Mem3R 的整套连续 recurrent state 作为主线。

因为你的优势在于：

- archive bank 是显式的
- retrieve / recover 是显式的
- 生命周期是显式的

如果你把系统整体改回 continuous streaming state，就会削弱你最强的叙事。

---

## 7. Scal3R：最该借“global context summary”

### 7.1 Scal3R 真正有价值的地方

Scal3R 的核心不是一般意义上的 recurrent state，而是：

- 用 neural global context representation
- 把长程场景信息压缩成轻量神经子网络
- 在测试时快速自监督适配
- 让模型能利用更长距离的全局上下文

### 7.2 HMR3D 当前最缺什么

HMR3D 当前的 bank 很可能更像：

- 一堆离散 archive entries

而缺少：

- 一个 scene-level / segment-level / corridor-level 的全局 summary

### 7.3 我建议你这样借

给 bank 加一层 **global context memory**。

#### bank 分两层：

##### Level 1：local entries
- 每个 archive submap 的 descriptor
- pose-anchor
- summary latent
- structural anchors

##### Level 2：global summaries
- 对一组 archive entries 的场景级摘要
- 提供 coarse retrieval prior
- 帮助 long-gap revisit 时先缩小搜索范围
- 帮助 branch disambiguation

### 7.4 为什么这一步很重要

你现在的 retrieve 很可能仍然偏 local matching：

- descriptor 命中 top-k
- 几何验证过滤假阳性

这在长距离 / 相似走廊 / 多分支场景中会越来越吃力。加入 global context summary 后，你就能做：

> **coarse scene recall → local entry retrieve → geometry verify → recover**

这会比单层 descriptor bank 强很多。

### 7.5 不建议借什么

不要把 Scal3R 的整套神经子网络式 global context 当作你第一优先级实现目标。

先做：

- 轻量 summary latent
- scene-level descriptor aggregation
- simple routing prior

足够了。

---

## 8. ZipMap：最该借“scene-state querying”

### 8.1 ZipMap 的关键思想

ZipMap 的核心不是 streaming，而是：

- 用 TTT layers 把一大组图像压缩进一个紧凑 hidden scene state
- 支持 scene-state querying
- 用 query 从 state 中读出结构信息

### 8.2 HMR3D 现在的 retrieve 更像什么

当前 retrieve 更像：

- 根据 descriptor 找 entry
- 验证 entry
- 恢复 entry

这是“命中某个块”。

### 8.3 下一步最值得做什么

把 retrieve 从“命中 entry”升级成“**查询 entry**”。

也就是：

#### coarse query
- 这个 archive entry 与当前观测是否相关？
- 如果相关，大致属于哪个场景段？

#### fine query
- 这个 archive entry 里哪些 anchors 最值得恢复？
- 哪些姿态约束最值得注入？
- 哪些 token / summary 对当前观测解释力最大？

### 8.4 这意味着什么

你的 archive entry 不应只是：

- `id`
- `descriptor`
- `pose`

而应该是：

- `descriptor`
- `summary latent`
- `queryable anchors`
- `recover candidates`
- `routing metadata`

这会让 HMR3D 真正从“检索列表”变成“可查询 memory bank”。

### 8.5 不建议借什么

不要把 ZipMap 的整体非 streaming / bidirectional feed-forward 主干直接搬进 HMR3D。

你更需要它的：

> **scene-state querying 观念**

而不是它的完整 backbone。

---

## 9. MERG3R：最适合借来做“子图合并能力”

### 9.1 为什么 MERG3R 特别适合你

MERG3R 做的是：

- 重排图像
- 分成 overlapping、几何多样的 subsets
- 分别重建
- 再通过 global alignment + confidence-weighted BA 合并

这个思想和你现在的 archive bank 天然兼容。

你已经有：

- archive entries
- retrieve
- geometry verify
- pose-anchor

下一步最自然的增强就是：

> **从单候选 recover 升级到多候选 merge。**

### 9.2 具体怎么借

#### 借法 A：overlap-aware archive partition

你现在 archive 可能更多按时间窗口 / 触发条件切。  
下一步可以改成：

- archive boundary 带空间 overlap
- archive 之间保留可 merge 的共享锚点
- 避免完全硬切分

#### 借法 B：confidence-weighted multi-candidate recover

不是只恢复 top-1，而是：

- retrieve top-k
- geometry verify
- 给每个候选一个置信度
- 按置信度做 merge / pose fusion

这会比单个候选 recover 稳得多。

#### 借法 C：offline consolidation stage

在 online HMR3D 外，再加一个 offline consolidation：

- 把长期 archive 的子图做统一整理
- 合并重复 entry
- 更新 scene-level summaries
- 形成更干净的长期地图资产

### 9.3 为什么这一步重要

这一步会把 HMR3D 从：

- “临时恢复系统”

推进成：

- “长期记忆地图系统”

这是质变。

---

## 10. HMR3D v2 的推荐架构

下面给出一个建议中的完整逻辑结构。

```text
Input Frames / Poses / Base State
        │
        ▼
┌────────────────────────────┐
│ Tracking Adapter / TTT3R IO │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────────────┐
│ HMR3D Memory Router v2             │
│------------------------------------│
│ 1. Write Policy                    │
│    - promote gate                  │
│    - archive gate                  │
│    - anchor retention              │
│                                    │
│ 2. Retrieval Policy                │
│    - coarse scene summary routing  │
│    - local archive entry query     │
│    - cooldown / anti-repeat logic  │
│                                    │
│ 3. Verification Policy             │
│    - counterfactual rollout        │
│    - geometry verify               │
│    - pose-anchor consistency       │
│                                    │
│ 4. Recovery / Merge Policy         │
│    - top-k recover                 │
│    - confidence-weighted fusion    │
│    - local adaptation update       │
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│ Hierarchical Memory Bank           │
│------------------------------------│
│ Level 1: local archive entries     │
│   - descriptor                     │
│   - summary latent                 │
│   - pose-anchor                    │
│   - structural anchors             │
│                                    │
│ Level 2: scene/global summaries    │
│   - segment descriptor             │
│   - context prior                  │
│   - branch disambiguation hints    │
└────────────────────────────────────┘
```

这个结构有几个重要特点：

1. **不破坏当前主线**
2. **每个增强都有明确来源和职责**
3. **可以逐步实现，不需要一次性大改**

---

## 11. 推荐的实现优先级（非常关键）

### 第一优先级：write gate + recover gate

这是最该先做的。

#### 目标
让系统具备：

- 更可控的 archive 写入
- 更可控的 recover 注入
- 更少的错误恢复污染

#### 第一版实现方式
先做规则版：

- descriptor confidence gate
- geometry consistency gate
- pose-anchor agreement gate
- overlap / novelty gate

#### 第二版实现方式
再做轻量 learned gate：

- MLP over current state features
- 预测 write ratio / recover ratio

### 第二优先级：hierarchical bank / global summary

#### 目标
让 retrieve 不再是平面的 top-k descriptor search，而是：

- coarse scene routing
- fine local retrieve

#### 第一版实现方式
- archive entry 聚合成 segment-level summary
- 每个 summary 维护 scene prior

### 第三优先级：multi-candidate recover + confidence-weighted merge

#### 目标
从单候选 recover 升级到稳健 merge。

#### 第一版实现方式
- retrieve top-k
- verify each candidate
- confidence-weighted pose-anchor fusion
- 不做复杂 BA，先做加权融合

### 第四优先级：local adaptation after recover

#### 目标
让 recover 成为“检索 + 适配”，而不是“检索 + 硬写回”。

#### 第一版实现方式
- 小步数更新
- 只调少量 summary / anchor 参数
- learning rate 由一致性预测

---

## 12. 代码层面的建议落点

下面按你已有目录来给建议。

### 12.1 `hmr3d_memory/MemoryRouter`

这里应该成为 v2 的中心。

建议加：

- `WritePolicy`
- `RetrievePolicy`
- `VerifyPolicy`
- `RecoverPolicy`
- `MergePolicy`

即使先只是规则类，也要把接口独立出来。

### 12.2 `adapter.py`

这里很适合承接：

- counterfactual rollout geometry verification
- pose-anchor consistency score
- recover 后的 local adaptation hook

### 12.3 `config/` 与 `configs/*.json`

建议新增明确的 v2 family：

- `hmr_v2_writegate_*.json`
- `hmr_v2_hierarchy_*.json`
- `hmr_v2_merge_*.json`
- `hmr_v2_local_adapt_*.json`

避免实验线和旧 sweep 混在一起。

### 12.4 `summarize.py`

建议增加新的统计项：

- write acceptance ratio
- average archive quality score
- retrieve precision / recall
- recover success rate
- recover pollution rate
- merge confidence histogram
- scene-summary routing hit rate

### 12.5 `tests/`

至少增加：

- `test_write_gate.py`
- `test_recover_gate.py`
- `test_scene_summary_routing.py`
- `test_multicandidate_merge.py`
- `test_local_adaptation.py`

---

## 13. 实验设计：怎么证明这些改进真的有用

### 13.1 主对比线

保持你已有的四条线：

- `cut3r`
- `ttt3r`
- `hmr_archive_only`
- `hmr_full`

在此基础上新增：

- `hmr_v2_writegate`
- `hmr_v2_hierarchy`
- `hmr_v2_merge`
- `hmr_v2_local_adapt`
- `hmr_v2_full`

### 13.2 核心评测问题

#### RQ1：写入策略是否提高了 bank 质量？
指标：
- retrieve precision
- retrieve recall
- recover 成功率
- average ATE after revisit

#### RQ2：层级 summary 是否减少误检索和分支混淆？
指标：
- branch ambiguity 下的错误恢复率
- top-k routing accuracy

#### RQ3：multi-candidate merge 是否比 top-1 recover 更稳？
指标：
- recover pollution rate
- ATE variance across sequences
- top-1 vs top-k 差异

#### RQ4：recover 后 local adaptation 是否真的带来稳定增益？
指标：
- recover 前后 ATE 改变量
- 几何一致性变化
- 误恢复情况下的灾难性退化频率

### 13.3 新增 stress tests

非常建议你把以下 stress tests 加到正式评测里：

- long-gap revisit
- repeated corridor / repeated door
- branch disambiguation
- forced backtracking
- partial overlap retrieval
- false-positive recover robustness

这些比简单的平均误差更能体现 HMR3D 的价值。

---

## 14. 风险与对策

### 风险 1：层级 bank 太复杂，反而拉低可控性

**对策：**
先做两层，不要一上来做多层神经记忆树。

### 风险 2：learned gate 很难训练，收益不稳定

**对策：**
先做规则版 gate，把效果和可解释性立住，再考虑学习版。

### 风险 3：multi-candidate merge 引入错误累积

**对策：**
先做 confidence-weighted pose fusion，不急着上全量 BA。

### 风险 4：local adaptation 变成另一套复杂训练系统

**对策：**
严格限制：
- 小步数
- 低频
- 事件触发
- 只更新少量参数

### 风险 5：项目再次发散

**对策：**
坚持主线：

> HMR3D v2 不是新 backbone，不是 Gaussian 论文，不是新的 CUT3R 变体。  
> 它是一套 **lifecycle memory system**。

---

## 15. 最推荐的 3 周推进顺序

### Week 1：Write / Recover Gate

交付：
- 规则版 write gate
- 规则版 recover gate
- 新统计项
- 初步 sweep

### Week 2：Hierarchical Bank

交付：
- scene-level summary
- coarse-to-fine retrieval
- branch disambiguation 实验

### Week 3：Multi-Candidate Merge + Local Adaptation

交付：
- top-k recover
- confidence-weighted merge
- 小步 local adaptation
- 完整 ablation

---

## 16. 最后的技术判断

如果只用一句话总结下一步，我的建议是：

> **不要再把 HMR3D 当作“外挂检索器”；要把它升级成“有写入策略、有查询状态、有多候选恢复与合并能力的层级记忆系统”。**

更具体一点：

- **TTT3R** 借“confidence-aware local update”
- **Mem3R** 借“职责解耦 + 写入门控”
- **Scal3R** 借“global context summary”
- **ZipMap** 借“scene-state querying”
- **MERG3R** 借“overlap-aware partition + confidence-weighted merge”

而 HMR3D 自己的主线必须始终保持：

> **archive → retrieve → verify → recover → merge**

这条线一旦做厚，你之后再接 Gaussian active map，才会是自然扩展，而不是重新开一个半成品支线。

---

## 17. 参考资料（建议后续继续查阅）

### 你已在用 / 最直接相关
- TTT3R: 3D Reconstruction as Test-Time Training  
  https://arxiv.org/abs/2509.26645
- Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training  
  https://arxiv.org/abs/2604.07279
- LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory  
  https://arxiv.org/abs/2603.03269

### 下一阶段最值得借的外部方法
- Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction  
  https://arxiv.org/abs/2604.08542
- ZipMap: Linear-Time Stateful 3D Reconstruction with Test-Time Training  
  https://arxiv.org/abs/2603.04385
- MERG3R: A Divide-and-Conquer Approach to Large-Scale Neural Visual Geometry  
  https://arxiv.org/abs/2603.02351

### 建议保留视野但不急于直接整合
- CUT3R
- TTSA3R
- LONG3R
- Point3R
- StreamVGGT / InfiniteVGGT

---

## 18. 一句话最终版（可直接放进汇报）

> 当前 HMR3D 已经完成了显式 archive-retrieve-recover 的主干验证。下一阶段不应急于回到 Gaussian 主线，而应先把 HMR3D 升级为一个具有写入策略、层级查询和多候选恢复合并能力的生命周期记忆系统；在此基础上，再将 active Gaussian 作为表示层接入，形成更完整的长期在线地图框架。

