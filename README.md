# SLAM 启发的分层记忆长程在线 3D 重建

更新时间：2026-04-04

## 1. 执行摘要

本项目最新版不再把核心创新定义为“语义-几何双状态融合”，而是重新聚焦到一个更强、更贴近 2026 年 3 月最新工作的主线：

- 短程高质量局部重建
- 长程预算受限记忆管理
- 子图归档与重访恢复

一句话定位：

> 以 `TTT3R` 的在线状态更新为底座，以 `OVGGT` 的常数预算缓存为内存控制器，以 `MERG3R` 的子图划分、合并与恢复为长程结构管理，构建一个面向长序列在线 3D 重建的分层记忆系统。

这意味着项目主叙事从：

- “给 TTT3R 再加一个语义状态”

调整为：

- “把在线 3D 重建从单一 recurrent state，升级成分层记忆系统”

这是更像论文、也更容易讲清楚创新边界的方向。

---

## 2. 为什么要改题

旧版方案的问题不是“做不出来”，而是创新叙事偏弱：

- 单纯增加语义分支，容易被理解成已有语义 SLAM 或语义重建工作的延伸。
- 如果核心仍然只有一个 recurrent state，再加多少附加头，长程问题依旧没有从根上解决。
- 你们现在真正稀缺的不是类别信息，而是“如何在固定预算下把在线重建做成长程”。

因此，最新版报告做两个明确收缩：

1. 语义不再作为主创新点。
2. 门控不再写成纯黑盒 learned gate。

对应地，方案改成三个更稳的机制：

1. `TTT3R` 风格的 confidence-aware active-state update。
2. `OVGGT` 风格的 importance selection + anchor protection。
3. `MERG3R` 风格的 submap archive / retrieve / merge。

语义仍然可以保留，但它的角色降级为辅助信号，例如：

- 动静态区域过滤
- 重访检索辅助
- 语义一致性评测

而不是系统的 headline。

---

## 3. 2026 年 3 月最相关的最新工作

### 3.1 结论先行

截至 **2026 年 4 月 4 日**，对本项目最有用的 2026 年 3 月相关 arXiv 非常集中，核心不是“语义”，而是：

- `bounded memory`：如何把在线重建真正做成长程
- `submap merge/recovery`：如何把长序列组织成可归档、可恢复、可重访的结构

### 3.2 建议重点参考的四篇

| 工作 | 时间 | 核心机制 | 对本项目最有用的部分 | 不建议直接照搬的部分 |
| --- | --- | --- | --- | --- |
| [OVGGT: O(1) Constant-Cost Streaming Visual Geometry Transformer](https://arxiv.org/abs/2603.05959) | 提交于 2026-03-06，修订于 2026-03-09 | Self-Selective Caching + Dynamic Anchor Protection | 重要性筛选、anchor 保护、固定预算缓存 | 不必直接整套换 backbone |
| [MERG3R: A Divide-and-Conquer Approach to Large-Scale Neural Visual Geometry](https://arxiv.org/abs/2603.02351) | 提交于 2026-03-02 | reorder -> partition -> overlapping subsets -> global alignment + confidence-weighted BA | 子图划分、归档、重访恢复、全局合并 | 不必一开始就做离线大规模全量求解 |
| [TTT3R: 3D Reconstruction as Test-Time Training](https://arxiv.org/abs/2509.26645) | 最新 arXiv 版本修订于 2026-03-03 | alignment confidence -> closed-form update rate | 在线 active state 写入规则 | 单靠它还不够解决无限长程 |
| [Streaming 4D Visual Geometry Transformer](https://arxiv.org/abs/2507.11539) | 最新 arXiv 版本修订于 2026-03-31 | causal transformer + historical KV cache | 短程高质量局部模块 | 它自身 memory 仍然会持续增长 |

### 3.3 直接的技术判断

- 如果只选一篇 2026 年 3 月论文来直接改你们项目，我会选 `OVGGT`。
- 如果只选一个“长程结构管理”方向，我会选 `MERG3R`。
- `TTT3R` 仍然是最合适的工程底座，但不再是唯一创新来源。
- `StreamVGGT` 更适合放在短程层，而不是作为整套系统的最终长程解。

---

## 4. 新项目定位

最新版项目建议题目：

**中文：**

- 基于分层记忆与子图恢复的长程在线 3D 重建
- SLAM 启发的分层记忆长程在线 3D 重建
- 预算受限的长程在线 3D 重建与重访恢复

**英文可选：**

- Memory-Bounded Long-Horizon 3D Reconstruction
- Hierarchical Memory for Streaming 3D Reconstruction
- SLAM-Inspired Submap Memory for Long-Range Neural Reconstruction

### 4.1 三个可以对外讲的 contribution

1. 提出一种面向长序列在线 3D 重建的分层记忆架构，将系统拆成短程局部工作区、在线活动地图和归档子图库，而不是继续依赖单一 recurrent state。
2. 用 `TTT3R` 的置信写回、`OVGGT` 的重要性缓存和 anchor 保护，构建预算受限的在线状态管理机制，避免纯黑盒 gate。
3. 借鉴 `MERG3R` 和经典 SLAM 的子图思想，实现重访触发的子图恢复和轻量校正，使系统具备更强的长程一致性。

### 4.2 一句话答辩表述

> 我们不是在 TTT3R 上再加一个语义头，而是把它从单一在线状态，升级成一个可写入、可裁剪、可归档、可恢复的分层记忆重建系统。

---

## 5. 总体架构

本文将拟议系统暂命名为 `HMR3D`（Hierarchical Memory Reconstruction for 3D）。

系统由四层组成：

1. `短程层 M_short`
2. `活动层 M_active`
3. `归档层 B_bank`
4. `调度层 Router`

和旧版最大的区别是：  
这四层不再只是“功能分工”的抽象框架，而是明确对应四类不同的 token 维护职责。

### 5.1 每层分别负责什么

- `M_short`
  - 保存最近 `W` 帧的高保真局部 token
  - 目标是“看清楚”
  - 负责短程几何细节、局部位姿修正、局部查询聚合
  - 吸收 `StreamVGGT/VGGT` 的局部高保真建模思想
  - 吸收 `LoGoPlanner` 的 query-based aggregation 思想

- `M_active`
  - 保存当前活动区域的 persistent state
  - 目标是“记得住”
  - 负责长程在线更新、状态稳定传播、低频重写
  - 直接复用并扩展当前 `TTT3R` 的 `state_feat / state_pos / mem`

- `B_bank`
  - 保存历史子图的压缩摘要、锚点和检索索引
  - 目标是“存得下、找得回”
  - 负责长期归档、重访恢复、子图级组织
  - 参考 `MERG3R`

- `Router`
  - 负责“写入 / 晋升 / 保留 / 裁剪 / 归档 / 恢复”的决策
  - 不是单一 learned gate
  - 优先采用 `TTT3R` 的置信写回、`OVGGT` 的重要性选择和规则化约束

### 5.2 引入 LoGoPlanner 的方式

这里可以明确回答你的问题：

**可以借 `LoGoPlanner` 的思路做 token 维护，但不能把它当成长程地图方案。**

因为 `LoGoPlanner` 真正值得借的是：

- metric-aware geometry memory
- task-specific query aggregation
- 用 query 从隐式几何状态里“读出”局部环境和自状态，而不是把显式中间量一路硬传

但它不适合直接充当长程层主干，原因也很清楚：

- 它的重点是规划，不是持久地图
- 它的几何记忆更偏局部决策服务
- 论文和项目页也明确提到 real-world reconstruction 还不够强

所以新版方案里，`LoGoPlanner` 只进入：

- 短程层的 query-based token 聚合
- token 的类型划分
- token 的“读”和“汇总”机制

不进入：

- 长程 persistent map 的最终组织方式
- 子图归档与恢复的主体逻辑

### 5.3 系统图

```text
输入帧流 x_t
    |
    v
短程局部工作区 M_short
    |
    v
候选增量 Delta_t + 查询摘要 + 重访描述子
    |
    v
记忆调度器 Router
    |------------------------------|
    |                              |
    v                              v
活动地图 M_active            归档子图库 B_bank
    |                              ^
    |                              |
    |------ 重访检索 / 恢复 / 校正 ------|
    |
    v
当前帧深度 / 点云 / 位姿 / 地图状态
```

---

## 6. 技术细节

## 6.1 状态定义

系统在时刻 `t` 维护：

```text
S_t = {M_short^t, M_active^t, B_bank^t}
```

其中：

- `M_short^t`：短程局部工作区
- `M_active^t`：活动地图
- `B_bank^t`：归档子图库

当前帧 `x_t` 进入后，系统输出：

- `Delta_t`：当前观测对几何状态的候选增量
- `c_align^t`：与历史 active state 的对齐置信
- `s_import^t`：token 重要性分数
- `d_t`：用于重访检索的描述子
- `q_pose^t / q_geo^t / q_desc^t`：不同任务的查询 token

## 6.2 token 类型与维护机制

为了把短程层和长程层说清楚，先定义 token 类型。

系统内部不再把所有 token 当成同一种东西，而是区分为五类：

### A. 观测 token `P_t`

- 由当前帧 patch / ray / 几何编码得到
- 信息量最大，但噪声也最大
- 生命周期最短
- 默认只在短程层停留

### B. 局部锚点 token `A_t`

- 从 `P_t` 中筛出来的高稳定、高几何贡献 token
- 用于局部位姿和结构支撑
- 可以从短程层晋升到活动层

### C. 活动状态 token `H_t`

- 当前活动地图中的 persistent token
- 对应 `TTT3R` 的 `state_feat/state_pos/mem` 扩展版本
- 生命周期中等到较长

### D. 归档摘要 token `Z_j`

- 某个历史子图的压缩表示
- 只保留摘要、锚点和检索描述子
- 生命周期最长

### E. 查询 token `Q`

这是从 `LoGoPlanner` 借来的最关键思路。

- `q_pose`：读取局部 ego-state / relative pose 相关信息
- `q_geo`：读取局部几何结构摘要
- `q_desc`：生成用于重访检索的描述子
- `q_sum`：把一个活动片段压成子图摘要

**核心变化是：**  
短程层和长程层不再“直接把全部 token 混在一起更新”，而是让 query token 去读 memory，再决定哪些 token 需要被保留、晋升、归档或恢复。

这就是 `LoGoPlanner` 思路真正可以迁移到你们项目里的地方。

## 6.3 短程层：LoGoPlanner 风格的 query-based 局部工作区

短程层的目标不是无限记忆，而是高质量局部估计。  
但现在要把它说得更具体：

### 6.3.1 短程层内部结构

短程层不只是“最近 `W` 帧队列”，而是三个子缓存：

```text
M_short^t = {L_recent^t, A_short^t, Q_short}
```

其中：

- `L_recent^t`
  - 最近 `W` 帧的 live observation tokens
  - 保留最新但未稳定的观测

- `A_short^t`
  - 最近窗口内已经被证明稳定的局部 anchor tokens
  - 它们不会像 live tokens 一样快速淘汰

- `Q_short`
  - 一组固定职责的查询 token
  - 包括 `q_pose`, `q_geo`, `q_desc`

### 6.3.2 每帧在短程层发生什么

给定当前帧 `x_t`，短程层执行以下步骤：

1. `Encode`
   - 将图像、ray map、scale prior、时间索引编码成观测 token `P_t`

2. `Append`
   - 把 `P_t` 追加进 `L_recent^t`

3. `Query-based aggregation`
   - `q_pose` 读取 `L_recent^t + A_short^t`，输出局部位姿摘要 `z_pose^t`
   - `q_geo` 读取 `L_recent^t + A_short^t`，输出局部几何摘要 `z_geo^t`
   - `q_desc` 读取 `L_recent^t + A_short^t`，输出检索描述子 `d_t`

4. `Anchor promotion`
   - 根据重要性和多帧支持度，把一部分 `P_t` 晋升为 `A_t`

5. `Evict`
   - 淘汰低重要性、低支持度、与局部结构无关的 live tokens

### 6.3.3 短程层的数学形式

可以写成：

```text
P_t = Enc(x_t, r_t, pi_t)
z_pose^t = Attn(q_pose, L_recent^t ∪ A_short^t)
z_geo^t  = Attn(q_geo,  L_recent^t ∪ A_short^t)
d_t      = Proj(Attn(q_desc, L_recent^t ∪ A_short^t))
```

anchor 晋升规则：

```text
m_promote(i) = 1[s_import(i) > tau_s and support(i) > tau_k]
```

这里：

- `s_import(i)`：token 重要性
- `support(i)`：被连续观测到且几何一致的次数

### 6.3.4 为什么这里要借 LoGoPlanner

因为 `LoGoPlanner` 给出的一个重要启发是：

> 局部环境和自状态不一定要通过显式大中间量传递，而可以通过任务查询从统一几何记忆中读出来。

借到你们项目里，就是：

- 不让短程层无脑存所有 patch tokens
- 让 query token 先做任务聚合
- 再根据 query 读出的信息决定 token 的后续去向

这样短程层就从“滑动窗口缓存”升级成了“查询驱动的局部工作区”。

### 6.3.5 初始实现建议

第一阶段不换 backbone，先做轻量版本：

- 保持当前 `TTT3R` 编码器
- 只额外维护 `L_recent / A_short / Q_short`
- 先用现有特征做 query aggregation

第二阶段再考虑：

- 把局部模块增强成 `StreamVGGT/VGGT` 风格的 causal local block
- 但只服务短程层，不承担长程记忆

### 6.3.6 短程层输出

短程层每帧至少应产出：

- 当前深度/点云预测
- 局部相对位姿修正 `z_pose^t`
- 局部几何摘要 `z_geo^t`
- 当前窗口描述子 `d_t`
- 局部置信度 `c_local^t`
- 与 active map 的 overlap 分数 `o_t`
- 可晋升的 anchor mask `m_promote`

## 6.4 活动层：双速率的长程在线状态更新

旧版这里写得过于简单，现在明确细化。

活动层不是“把短程输出直接写到一个大状态里”，而是拆成两部分：

```text
M_active^t = {A_active^t, C_active^t}
```

其中：

- `A_active^t`
  - 稳定锚点状态
  - 更新慢
  - 用于长期支撑和重访匹配

- `C_active^t`
  - 上下文状态
  - 更新快
  - 用于吸收当前区域新观测

### 6.4.1 为什么要分双速率

如果所有 active token 都用同一个更新率：

- 太快：稳定锚点会被污染
- 太慢：新区域吸收不了新信息

所以活动层采用双速率写回：

```text
alpha_i =
    alpha_anchor * c_align^t,   if i in A_active
    alpha_ctx    * c_align^t,   if i in C_active
```

且满足：

```text
alpha_anchor < alpha_ctx
```

也就是说：

- 锚点更新慢，强调稳定
- 上下文更新快，强调适应

### 6.4.2 写回来源

活动层不接受全部 `P_t`，只接受两类输入：

1. 从短程层晋升上来的 `A_t`
2. 由 `q_geo` 聚合后的局部摘要 `z_geo^t`

这点很关键，因为它避免了：

- 把噪声 patch token 直接污染长程状态
- 让长程层退化成另一个短程缓存

### 6.4.3 写回规则

保留 `TTT3R` 的置信写回作为 base rule：

```text
m_write^t = m_base^t * clip(c_align^t, 0, 1)
```

然后区分 anchor/context 更新：

```text
A_active^{t+1} = (1 - alpha_anchor m_write^t) * A_active^t
               + alpha_anchor m_write^t * Delta_anchor^t

C_active^{t+1} = (1 - alpha_ctx m_write^t) * C_active^t
               + alpha_ctx m_write^t * Delta_ctx^t
```

### 6.4.4 活动层内部还需要一个 summary token

为了后续归档，活动层里还需要维护一类低频更新的 summary token：

```text
S_active^t = Attn(q_sum, A_active^t ∪ C_active^t)
```

它的作用是：

- 形成当前活动片段的紧凑摘要
- 为 `B_bank` 归档准备子图级表示

也就是说，活动层不是只负责“更新”，还负责“产出可归档摘要”。

## 6.5 内存控制层：OVGGT 风格的重要性筛选与 anchor 保护

用户前面质疑“门控是不是拍脑袋”是对的，所以这里继续坚持：

- 不用大而全黑盒 gate
- 优先用训练无关的重要性选择

### 6.5.1 重要性分数

对每个 token 或 state slot 计算：

```text
score_i = lambda_res * residual_i
        + lambda_att * attn_i
        + lambda_geo * geo_anchor_i
        + lambda_conf * conf_i
        + lambda_rev * revisit_i
```

新加的：

- `revisit_i`
  - 表示该 token 对重访匹配的贡献
  - 这是把长程需求显式写进 token 维护

### 6.5.2 三种动作

内存控制层不再只有 keep / evict，而是三种动作：

1. `keep`
   - 保留在当前层级

2. `promote`
   - 从短程层晋升到活动层

3. `archive`
   - 从活动层压缩到子图库

可以写成：

```text
action_i in {keep, promote, archive, evict}
```

### 6.5.3 anchor 保护

以下 token 默认不参与普通淘汰：

- 几何边界稳定且可重复观测的 token
- 对相机位姿估计贡献大的 token
- 多次重访仍然稳定的 token
- 在子图摘要中反复被 query 读取的 token

因此，内存控制层不是“统一裁剪”，而是：

1. 先打分
2. 再做 anchor 保护
3. 最后在剩余 token 上决定 keep / promote / archive / evict

## 6.6 归档层：MERG3R 风格的子图归档与恢复

旧版这里已经有基本轮廓，但还不够详细。现在补成完整流程。

长程问题不能只靠 active state 硬撑，因此每个子图条目保存：

```text
B_j = {desc_j, latent_j, anchors_j, pose_j, conf_j, meta_j}
```

### 6.6.1 归档的输入不是原始 token，而是“子图包”

一个待归档子图 `U_j` 由三部分组成：

```text
U_j = {S_active^j, A_active^j, T_j}
```

其中：

- `S_active^j`：活动片段摘要
- `A_active^j`：该片段的锚点集合
- `T_j`：参考位姿/参考坐标框架

### 6.6.2 子图压缩

对子图 `U_j` 用 `q_sum` 做摘要压缩：

```text
latent_j = Pool_Q(q_sum, U_j)
desc_j   = Proj(latent_j)
```

这里的设计仍然借了 `LoGoPlanner` 的 query 思路：

- 不是直接平均所有 token
- 而是由摘要 query 去读取子图中最有价值的部分

### 6.6.3 何时归档

当满足以下任一条件时触发子图归档：

- active memory 超过预算上限
- 当前区域与历史 active 区域 overlap 降低
- 当前片段已经稳定，可以从活动区移出

可写成：

```text
T_archive(t) = 1[mem_budget > tau_mem
                 or overlap_t < tau_overlap
                 or age_t > tau_age]
```

### 6.6.4 何时恢复

当检测到潜在重访时：

1. 用 `d_t` 在 `B_bank` 里做 top-k 检索
2. 用几何一致性做验证
3. 把匹配子图的 `anchors_j + latent_j` 恢复到 `M_active`
4. 用短程层对当前片段做一次局部 refinement

### 6.6.5 为什么恢复后还要回到短程层

因为恢复出来的是“历史先验”，不是当前局部细节。  
真正的当前细节仍然应该由短程层重估。

所以恢复后的正确流程是：

```text
retrieve -> inject prior into M_active -> local short-window refine
```

这一步让系统真正具备：

- submap retrieval
- relocalization
- loop closure
- local correction

## 6.7 为什么这套方案比“再加语义”更强

因为它回答的是一个更根本的问题：

> 在固定预算下，如何把在线 3D 重建做成长程，而不是在一个会爆炸的状态上不断叠补丁。

语义只能帮助解释和过滤，不能替代记忆系统本身。

---

## 7. 语义在新版方案中的角色

新版报告不是完全放弃语义，而是把语义降为辅助模块。

语义更适合承担三种职责：

1. 动静态区域过滤
2. 重访检索增强
3. 语义长期一致性评测

不建议让语义成为：

- 第一优先级 memory controller
- 论文 headline
- 唯一的创新主线

---

## 8. 与当前仓库代码的映射关系

当前最重要的事实是：

- 你们现有可运行底座是 `CVPR/TTT3R`
- 因此下一步实现必须优先“沿 TTT3R 改造”
- 而不是一开始就试图彻底换到全新的外部 backbone

### 8.1 `CVPR/TTT3R/demo.py`

作用：

- 输入序列准备
- 控制 `model_update_type`
- 执行 `inference_recurrent_lighter`

建议新增参数：

- `--memory_budget`
- `--short_window`
- `--anchor_ratio`
- `--archive_overlap`
- `--archive_age`
- `--retrieve_topk`
- `--enable_submap_bank`
- `--enable_anchor_protect`

### 8.2 `CVPR/TTT3R/src/dust3r/inference.py`

当前 `state_args` 仍然是 tuple 风格。

建议改为结构化状态对象，例如：

```python
state = {
    "state_feat": ...,
    "state_pos": ...,
    "init_state_feat": ...,
    "mem": ...,
    "init_mem": ...,
    "anchor_mask": ...,
    "active_submap_id": ...,
    "submap_bank": ...,
    "meta": ...,
}
```

这样后面加入：

- anchor 保护
- 子图归档
- 重访恢复

才不会继续把状态接口写得越来越难维护。

### 8.3 `CVPR/TTT3R/src/dust3r/model.py`

这是最核心的改造点。

建议优先修改这几个位置：

1. `forward_recurrent_lighter`
2. `_forward_impl`
3. `update_mask` / `update_mask1` 分支

具体实现顺序建议是：

1. 保留现有 `TTT3R` 的 soft update 作为 base write rule
2. 在 `cross_attn_state` 后额外输出：
   - `alignment_score`
   - `importance_score`
   - `anchor_mask`
3. 在写回之前先做 keep / evict
4. 在每段 active chunk 结束时导出子图摘要

### 8.4 建议新增的模块文件

为了不把所有逻辑继续堆进 `model.py`，建议新增：

- `CVPR/TTT3R/src/dust3r/hier_state.py`
  - 定义结构化状态对象
- `CVPR/TTT3R/src/dust3r/submap_bank.py`
  - 子图存取、归档、检索
- `CVPR/TTT3R/src/dust3r/revisit.py`
  - 检索与几何验证
- `CVPR/TTT3R/src/dust3r/memory_router.py`
  - 负责 write / keep / archive / retrieve 的调度

### 8.5 `CVPR/TTT3R/eval/*`

除了现有：

- ATE
- RPE
- 深度相关指标

还应该补：

- 重访命中率
- 检索后误差下降幅度
- 子图归档压缩比
- 峰值显存
- 平均时延
- 每 100 帧漂移率

---

## 9. 具体实现计划

这里给出一版以“能在当前仓库逐步落地”为目标的实现路线。

## 9.1 Phase 1：先做最小可运行版本

目标：

- 不换 backbone
- 不做语义分支
- 只在 `TTT3R` 上实现分层记忆雏形

具体内容：

1. 把 `state_args` 从 tuple 改成结构化字典或 dataclass
2. 在 `model.py` 中输出 token importance 和 anchor mask
3. 在 active state 上加 keep / evict
4. 加一个最简单的 `submap_bank`，先用 CPU 侧 list 存摘要

交付标准：

- 系统能跑
- 内存预算可控
- 可以看到 active / archive 的切换

## 9.2 Phase 2：加入重访检索与子图恢复

目标：

- 把“能存”升级为“能找回来”

具体内容：

1. 从短程窗口提取全局描述子 `d_t`
2. 用余弦相似度做 coarse retrieval
3. 用几何一致性做二次验证
4. 恢复 top-k 子图 anchors 到 active map
5. 做轻量局部 refinement

交付标准：

- 重访片段能够触发恢复
- 恢复后 ATE / RPE / depth drift 出现可测改善

## 9.3 Phase 3：增强短程层

目标：

- 把短程局部精度补上来

可选两条路线：

1. 轻量路线
   - 仍用 `TTT3R` 特征
   - 对最近窗口重跑局部 refinement

2. 强化路线
   - 接入 `StreamVGGT/VGGT` 风格局部模块
   - 只让它负责最近窗口，不让它承担长程记忆

如果时间有限，优先做轻量路线。

## 9.4 Phase 4：再决定是否引入语义

语义只在以下条件都满足时再加：

- 几何长程主线已经跑通
- 子图恢复已有收益
- 需要动静态过滤或重访辅助

这时语义只作为：

- mask
- descriptor enhancement
- long-term consistency metric

而不是重新把项目拖回“语义双状态”的叙事里。

---

## 10. 实验设计

### 10.1 数据集建议

优先复用当前仓库已有评测链路，避免一开始数据侧失控。

建议主测：

- KITTI
- Bonn
- Sintel
- TUM / ScanNet / Neural RGB-D 中可复用的长序列部分

补充：

- `examples/westlake.mp4`
- `examples/taylor.mp4`

可以作为可视化和定性验证。

### 10.2 指标

几何指标：

- ATE
- RPE
- 深度误差

长程与系统指标：

- 峰值显存
- 平均时延
- 每帧吞吐
- 每 100 帧漂移率
- 重访检索命中率
- 检索后误差下降幅度
- 子图压缩比

### 10.3 必做消融

1. 单一 active state vs 分层记忆
2. 无裁剪 vs importance pruning
3. 无 anchor protection vs anchor protection
4. 无归档 vs 归档不恢复 vs 归档+恢复
5. 无短程增强 vs 短程窗口增强

---

## 11. 风险与 fallback

### 11.1 风险：短程模块太重

应对：

- 第一阶段先不接入 `VGGT`
- 仅用 `TTT3R` 现有特征做窗口重解码

### 11.2 风险：检索误触发

应对：

- 粗检索只负责召回
- 几何验证决定是否真正恢复

### 11.3 风险：子图库越来越大

应对：

- 加上 bank budget
- 对 bank 内条目再做二级压缩

### 11.4 风险：论文叙事不够聚焦

应对：

- headline 始终围绕“分层记忆”
- 不把语义写成主创新
- 不把所有机制都包装成 learned gate

---

## 12. 最终建议

最新版项目不建议再讲成：

- 语义-几何双状态线性 SLAM

更建议讲成：

- 分层记忆长程在线 3D 重建
- SLAM 启发的子图归档与重访恢复
- 预算受限的在线神经几何系统

如果后续实现顺利，最有说服力的论文式表述应该是：

> 我们提出一种面向长程在线 3D 重建的分层记忆系统，将短程局部高保真建模、活动状态在线更新、预算有界缓存和子图归档恢复统一起来，在固定预算下改善重建稳定性与重访一致性。

---

## 13. 参考资料

- [OVGGT: O(1) Constant-Cost Streaming Visual Geometry Transformer](https://arxiv.org/abs/2603.05959)
- [MERG3R: A Divide-and-Conquer Approach to Large-Scale Neural Visual Geometry](https://arxiv.org/abs/2603.02351)
- [TTT3R: 3D Reconstruction as Test-Time Training](https://arxiv.org/abs/2509.26645)
- [Streaming 4D Visual Geometry Transformer](https://arxiv.org/abs/2507.11539)
- [CUT3R: Continuous 3D Perception Model with Persistent State](https://arxiv.org/abs/2501.12387)
- [VGGT](https://arxiv.org/abs/2503.11651)
