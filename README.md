# 基于 TTT3R+ 的语义-几何双状态线性 SLAM

本项目围绕一个核心问题展开：
在长时序、预算受限、场景变化明显的在线三维重建任务中，如何同时维持几何一致性与语义稳定性。

我们将该问题落地为一个可执行框架：
以 TTT3R 的在线状态更新为主干，吸收 MERG3R 的长期压缩/恢复思想，以及 ZipMap 的线性状态读写思想，构建语义-几何双状态线性 SLAM。

## 1. 项目定位

- 目标不是“再加一个语义头”。
- 目标是“改在线状态更新机制本身”。
- 重点是预算约束下的长期稳定性，而不是只优化单帧精度。

一句话定位：
在 TTT3R 可执行链路上，加入语义一等状态、门控写回、事件触发压缩与重访恢复，使系统在长时运行中获得更好的一致性-效率折中。

## 2. 仓库结构

当前工作区核心目录：

- CVPR/TTT3R: 可运行的在线重建主干代码。
- CVPR/CUT3R: 上游基座代码。
- CVPR/latex: 研究报告与论文化描述。

本 README 聚焦工程实现主线，即 CVPR/TTT3R。

## 3. 三篇锚点工作的“化学反应”

### 3.1 TTT3R 提供什么

TTT3R 提供的是在线递推状态更新链路：

- 输入视图序列被逐帧编码。
- 维护全局状态 state_feat 与局部记忆 mem。
- 每帧通过 recurrent rollout 产生新状态候选，再用 update_mask 执行写回。
- 支持 reset 周期重置，避免超长序列漂移失控。

这部分是本项目的数据流主骨架。

### 3.2 MERG3R 提供什么

MERG3R 的关键价值不在“网络结构替换”，而在长期状态管理策略：

- Divide: 长序列按局部事件拆解。
- Compress: 将高冗余状态压缩为紧凑表示。
- Recover: 在重访阶段恢复局部细节，避免重复学习。

在本项目中，MERG3R 的作用是把“长期运行”从时间维度变成可管理的状态预算问题。

### 3.3 ZipMap 提供什么

ZipMap 的可迁移思想是线性状态接口：

- 线性可查询状态，支持快速 read。
- 预算友好的增量 write，避免全量重算。

在本项目中，ZipMap 的角色是语义状态的读写范式来源，尤其适合构建 token bank 形式的语义记忆。

### 3.4 三者如何串联

三者不是并列拼接，而是功能分层耦合：

1. TTT3R 管在线更新主环。
2. ZipMap 管语义状态读写接口。
3. MERG3R 管长期压缩恢复调度。

因此计算图可以理解为：

输入编码 -> 几何/语义状态候选 -> 门控写回 -> 事件触发压缩 -> 重访恢复 -> 输出与评估。

## 4. 当前代码中的真实信息流

以下内容是基于现有代码路径抽象出的已实现主流程。

### 4.1 输入封装

入口：TTT3R/demo.py 的 prepare_input。

每个 view 包含：

- img: 图像张量。
- ray_map: 几何射线输入或占位。
- img_mask / ray_mask: 输入模态有效性掩码。
- update: 当前帧是否允许写回状态。
- reset: 是否触发状态重置。

其中 reset 由 reset_interval 周期触发。

### 4.2 推理主调用

入口：TTT3R/demo.py 的 run_inference。

- model = ARCroco3DStereo.from_pretrained(...)
- model.config.model_update_type = {cut3r 或 ttt3r}
- outputs, state_args = inference_recurrent_lighter(...)

注意：演示脚本默认走 inference_recurrent_lighter 路径。

### 4.3 编码与状态初始化

核心路径：TTT3R/src/dust3r/model.py。

- _encode_views: 将 img 与 ray_map 编码为 token，并按 mask 对齐。
- _init_state: 用第一帧特征初始化 state_feat 与 state_pos。
- LocalMemory.mem: 初始化局部记忆 mem（姿态检索相关）。

### 4.4 recurrent rollout

每一帧做三件事：

1. 解码交互
- _recurrent_rollout 调用 _decoder。
- 状态 token 与当前帧 token 交叉注意力交互。

2. 头部预测
- _downstream_head 产生几何相关输出（深度/点云/姿态相关）。

3. 状态更新
- 计算 update_mask。
- 对 state_feat 与 mem 做加权写回。
- 若 reset=True，回退到 init_state_feat 与 init_mem。

## 5. 门控逻辑的数学解释（当前 TTT3R 已实现）

### 5.1 基础写回掩码

令 m_base 为基础更新掩码：

- 若存在 update 标志，则 m_base = img_mask AND update。
- 否则 m_base = img_mask。

写回采用逐元素线性插值：

state_feat <- m_state ⊙ new_state_feat + (1 - m_state) ⊙ state_feat

mem <- m_mem ⊙ new_mem + (1 - m_mem) ⊙ mem

### 5.2 cut3r 模式

m_state = m_base，m_mem = m_base。

即几何状态与局部记忆都按同一掩码硬更新。

### 5.3 ttt3r 模式

在 forward_recurrent_lighter 与 _forward_impl 路径中，state 更新会使用注意力强度调制：

1. 从 cross_attn_state 收集多层多头注意力。
2. 沿图像维和头维做平均，得到 state_query_img_key。
3. 通过 sigmoid 得到每个 state token 的软门控系数。

可写为：

m_state = m_base * sigmoid(q_state)

其中 q_state 来自跨注意力统计。

这意味着：

- 相关性高的状态 token 更新更强。
- 相关性低的状态 token 更偏向保留历史值。

### 5.4 reset 逻辑

若 reset=True：

- state_feat 直接回到 init_state_feat。
- mem 直接回到 init_mem。

这是长序列稳定性的第一道保险。

## 6. 为什么“创新不只是融合语义和几何”

常见误解是：
“加语义分支 = 创新”。

本项目的创新主轴其实是：
在线状态机制升级。

具体包括四个层面：

1. 语义状态一等化
- 语义不是后处理标签，而是参与状态演化。

2. 联合门控
- 门控不仅看输入有效性，还看可信度、冲突与新颖度。

3. 事件驱动压缩恢复
- 状态管理从固定周期变为按事件触发。

4. 长时序联合评测
- 同时考察几何、语义和预算指标。

## 7. TTT3R+ 的目标计算图（扩展设计）

### 7.1 双状态定义

系统维护：

- S_geo: 几何状态。
- S_sem: 语义状态。

每帧输入 x_t 后，产生候选增量：

- Delta S_geo^t
- Delta S_sem^t

### 7.2 语义读写（ZipMap 风格）

语义状态采用 token bank：

S_sem = {(k_i, v_i, c_i)}

- k_i: 语义键。
- v_i: 语义值。
- c_i: 置信与稳定度统计。

线性读出：

y_t = sum_i alpha_i(q_t, k_i, c_i) * v_i

核心意图是让语义检索成本接近线性、支持预算下在线查询。

### 7.3 联合门控（项目新增）

定义三个信号：

- u_t: 不确定性。
- n_t: 新颖度。
- c_t: 冲突度。

语义门控：

g_sem = sigmoid(w_u * u_t + w_n * n_t + w_c * c_t + b)

语义写回：

S_sem^{t+1} = g_sem ⊙ Delta S_sem^t + (1 - g_sem) ⊙ S_sem^t

几何写回沿用 TTT3R 主链，最终形成双状态耦合更新。

### 7.4 事件触发压缩与恢复（MERG3R 风格）

触发函数：

T_compress(t) = I[r_t > tau_r OR u_t > tau_u OR b_t > tau_b]

- r_t: 重访密度或重访信号。
- u_t: 语义状态不确定性。
- b_t: 当前预算压力（显存/延迟/容量）。

触发后执行：

1. 聚类与原型蒸馏（压缩）。
2. 建立索引与恢复键。
3. 在重访时按键恢复局部状态并轻量校正。

## 8. 工程改造落点

建议严格从以下入口改造，确保可测、可回滚：

- TTT3R/src/dust3r/model.py
  - 扩展 state 结构为几何+语义。
  - 在 update_mask 分支中增加语义门控分支。
  - 在 recurrent 路径加入语义状态的 reset 与恢复。

- TTT3R/src/dust3r/inference.py
  - 扩展 state_args 为双状态。
  - 让 inference_step / inference_recurrent_lighter 支持语义状态传递。

- TTT3R/demo.py
  - 新增语义门控与压缩开关参数。
  - 保持 model_update_type 与 reset_interval 控制兼容。

- TTT3R/eval/*
  - 在 relpose/depth 指标之外增加长期语义一致性与预算曲线统计。

## 9. 训练与推理信息流（建议）

### 9.1 训练阶段

- 预训练几何主干，保证几何稳定。
- 加入语义状态与联合损失。
- 通过消融验证每个机制模块的边际收益。

联合损失示意：

L = lambda_pose * L_pose + lambda_depth * L_depth + lambda_sem * L_sem + lambda_cross * L_cross

其中 L_cross 约束语义边界与几何边界的一致性。

### 9.2 在线推理阶段

每帧循环：

1. 编码当前观测。
2. 读取几何状态与语义状态。
3. 预测并计算候选增量。
4. 执行门控写回。
5. 判断是否触发压缩/恢复。
6. 产出当前帧结果与状态快照。

## 10. 评测协议（必须覆盖）

几何指标：

- ATE
- RPE
- 深度误差

语义长期指标：

- 重访语义一致率
- 语义漂移率
- ID 保持率

系统指标：

- 峰值显存
- 平均时延
- 吞吐
- 状态容量利用率

必要消融：

- 单状态 vs 双状态。
- 固定更新 vs 门控更新。
- 无压缩 vs 周期压缩 vs 事件触发压缩。
- 无恢复 vs 检索恢复 vs 检索恢复+轻量校正。

## 11. 快速运行（当前可用）

在 TTT3R 目录执行：

1. 安装依赖与编译（见 TTT3R/README.md）。
2. 准备 checkpoint。
3. 运行示例：

python demo.py --model_path src/cut3r_512_dpt_4_64.pth --size 512 --seq_path examples/westlake.mp4 --output_dir tmp/run1 --port 8080 --model_update_type ttt3r --frame_interval 1 --reset_interval 100 --downsample_factor 100 --vis_threshold 6.0

## 12. 常见问题

### Q1: 这个项目的创新点只是“几何+语义融合”吗？

不是。
融合只是表层，核心创新在在线状态更新机制：
门控、压缩、恢复和预算约束是方法主体。

### Q2: 和传统语义 SLAM 的关键区别是什么？

传统方法常把语义作为前后端约束或后处理标签。
本项目让语义进入状态主回路，直接影响后续状态演化。

### Q3: 当前代码已经完成双状态了吗？

当前公开代码主干仍以几何状态链路为主。
双状态与事件压缩恢复是本项目的下一阶段实现内容。

## 13. 版本与里程碑

- M1: 打通双状态接口与基础门控。
- M2: 加入事件触发压缩与重访恢复。
- M3: 完成长期评测与消融矩阵。
- M4: 论文化整理与可复现发布。

---

如果你要继续推进工程实现，建议先从 model.py 的 update_mask 分支入手，把语义门控接入 forward_recurrent_lighter 路径，再扩展 inference.py 的 state_args 结构，最后补 eval 脚本的长期指标统计。