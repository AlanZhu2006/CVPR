# NUC Architecture

这份文档把报告里的系统术语，映射到当前仓库里已经可运行的 NUC 原型实现。

## 1. 系统目标

NUC 原型不是最终版 Jetson 系统。它只负责验证下面这条主链：

`tracking -> keyframe -> active/archive split -> retrieve -> recover`

也就是说，它先证明“地图像记忆一样运转”是成立的，再把 active 部分替换成你后续的 Gaussian submap。

## 2. 模块映射

报告术语和代码文件的对应关系如下：

| 报告里的概念 | 当前实现 | 文件 |
| --- | --- | --- |
| Tracking frontend | ORB-based lightweight frontend | `nuc/src/nuc_runtime/tracking.py` |
| `M_short` | 最近关键帧窗口 | `nuc/src/nuc_runtime/memory_router.py` |
| `M_active` | 当前活动关键帧子图 | `nuc/src/nuc_runtime/memory_router.py` |
| `B_bank` | 已归档历史子图库 | `nuc/src/nuc_runtime/memory_router.py` |
| Descriptor query | 全局图像描述子 + ORB anchor verify | `nuc/src/nuc_runtime/descriptors.py` |
| Archive / retrieve / recover | lifecycle router | `nuc/src/nuc_runtime/memory_router.py` |
| Logging / visualization | csv, jsonl, json, debug video | `nuc/src/nuc_runtime/output.py` |

## 3. Tracking 输出接口

NUC 原型里，tracking 层的统一输出是 `TrackingOutput`。它提供：

- `timestamp_sec`
- `pose`
- `is_keyframe`
- `descriptor`
- `orb_descriptors`
- `match_count / inlier_count / pixel_motion`

这个接口是故意固定下来的。以后你把 ORB 前端换成更强的 tracking，只需要继续输出同一结构，后面的 memory 层不用重写。

## 4. Memory Router 的职责

当前原型里，`MemoryRouter` 做四件事：

1. 把关键帧送进 `M_short`
2. 把关键帧提升进 `M_active`
3. 在满足规则时把 `M_active` 归档到 `B_bank`
4. 在 query 命中时从 `B_bank` 触发 retrieve / recover

当前触发规则是规则化的，不是 learned gate：

- `max_keyframes`
- `max_age`
- `pose_distance`
- `active_similarity_drop`

这和你报告里“先把逻辑写对，再考虑 learned controller”的路线一致。

## 5. Retrieve 与 Recover 是怎么落地的

当前第一版实现分两步：

1. `retrieve`
   用当前关键帧 descriptor 和 bank 中 archived submap 的 descriptor 做 top-k 检索。
2. `recover`
   对 top-k 候选做 ORB anchor 匹配验证，验证通过后，把命中的 archived descriptor 注入当前 active。

所以第一版 recover 不是“恢复高斯”，而是“把历史子图先重新挂回 active 的记忆状态”。

这正是 NUC 阶段应该验证的东西。

## 6. 输出文件怎么看

最重要的三个输出是：

- `events.jsonl`
  看整个 lifecycle 是否真的发生了
- `submaps.json`
  看 active 和 archive 是否真的分层了
- `summary.json`
  看 archive / retrieve / recover 的统计是否成立

如果你只看 `debug.mp4`，很容易把这个系统误当成可视化 demo。真正的验证证据应该优先看日志和摘要。

## 7. 和正式系统的关系

正式系统要替换的部分：

- tracking 前端
- active map 表示
- recover 后的几何校正

可以直接复用的部分：

- 数据流边界
- lifecycle 状态机
- 归档/检索接口
- 输出协议

所以这个原型不是临时脚本，而是正式系统的第一层骨架。
