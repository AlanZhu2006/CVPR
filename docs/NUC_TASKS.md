# NUC Tasks And Acceptance

这份清单把 NUC 阶段需要做的事情按阶段拆开，并给出对应的验收标准。它的目标不是“做到最终系统”，而是确保第一阶段主链成立。

## Phase 0: 环境与入口

任务：

- 用 `bash nuc/scripts/setup_nuc.sh` 建立环境
- 用 `bash nuc/scripts/check_nuc_env.sh` 确认依赖
- 用 `bash nuc/scripts/run_nuc_replay.sh` 跑通一段样例视频

验收：

- 能从空机器 `clone -> setup -> run`
- 输出目录里出现 `summary.json` 和 `poses.csv`

## Phase 1: Tracking 接口稳定

任务：

- 输入视频或图片序列
- 稳定输出 `pose` 和 `is_keyframe`
- 把关键帧图像存下来，方便后续排查

验收：

- `poses.csv` 中每帧都有位姿
- 至少有一批关键帧进入 `keyframes/`
- debug 视频里能看到位姿和关键帧状态在持续更新

## Phase 2: Memory Router 跑通

任务：

- 每个关键帧都进入 `M_short`
- 当前活动块写入 `M_active`
- bank 为空时也能持续运行，不崩

验收：

- `events.jsonl` 中持续出现 `promoted`
- `summary.json` 中有关键帧统计
- `submaps.json` 中 active 区域字段存在

## Phase 3: Archive 成立

任务：

- 把 active 按规则关闭并归档
- bank 能持续增长，但不会无限把所有内容都留在 active

验收：

- `events.jsonl` 中出现 `archived`
- `submaps.json` 中 bank 有条目
- `summary.json` 中 `archives > 0`

## Phase 4: Retrieve 成立

任务：

- 当前关键帧生成 query descriptor
- 在 bank 中做 top-k 检索
- 记录命中和失败的统计

验收：

- `events.jsonl` 中出现 `retrieved`
- `summary.json` 中 `retrieve_hits > 0`
- bank 中被检索到的子图 id 可追踪

## Phase 5: Recover 成立

任务：

- 对 retrieve 的候选做几何验证
- 验证通过后，把历史子图信息注入当前 active
- 记录 recover 次数

验收：

- `events.jsonl` 中出现 `recovered`
- `summary.json` 中 `recoveries > 0`
- `submaps.json` 中当前 active 的 `recovered_from` 非空

## Phase 6: 对照实验

任务：

- 跑一版 `recover on`
- 跑一版 `recover off`
- 用 `python3 nuc/tools/compare_runs.py` 比较两次统计

验收：

- 两次运行都能稳定完成
- 输出统计可直接对比
- 至少能明确看到 retrieve / recover 是否触发、触发频率如何

建议收口命令（自动批量比较并输出报告）：

- `bash nuc/scripts/close_phase6.sh`

默认会比较这三组：

- `recover_on` vs `recover_off`
- `stereo_taylor_recover_on` vs `stereo_taylor_recover_off`
- `westlake_recover_on` vs `westlake_recover_off`

脚本会在 `nuc_output/phase6_closure_<timestamp>/` 生成：

- `phase6_summary.csv`（总表，含 PASS/WARN）
- `<on_run>__vs__<off_run>.csv`（逐组 compare_runs 输出）
- `phase6_report.md`（可读报告）

## 第一周建议

如果你只做一周，建议只锁这三件事：

1. 跑通 tracking 接口
2. 写稳 `MemoryRouter`
3. 让 archive 真能生成 bank entry

做到这里，你的项目就已经从“概念”进入“系统原型”了。

## 暂时不要做的事

- 不要先接 cuVSLAM
- 不要先接 Gaussian
- 不要先做 ROS 整机联调
- 不要先追求全局地图视觉效果

NUC 阶段的目标很单纯：

先证明 `tracking -> memory lifecycle` 这条链是可运行、可观察、可对照验证的。
