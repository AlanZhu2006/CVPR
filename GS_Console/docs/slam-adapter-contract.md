# SLAM Adapter Contract

## 目标

这份 contract 的目标不是把所有 SLAM 算法内部统一，而是把它们对外暴露给 UI、scene pack 和 bridge 的接口统一。

结论很直接：

- 每个算法或数据源仍然要有一个 adapter
- 但 adapter 只需要对接一次统一 contract
- 前端、导入器、viewer、benchmark 都只消费 contract，不再认识具体算法

这和 `GSLAM` / `SLAMBench` 的 plugin 思路一致，只是这里把目标收敛到当前仓库的 `scene pack + web UI + live ROS bridge`。

## Contract 分层

### 1. Scene Manifest

文件：`schemas/scene-manifest.schema.json`

职责：

- 描述静态产物
- 告诉 viewer 如何加载 gaussian / mesh / occupancy / raw point cloud
- 提供相机、初始视角、地图原点、机器人尺寸等基础元数据
- 可选地通过 `source.liveContractUrl` 指向在线通道 contract
- 可选地通过 `source.capabilityMatrixUrl` 指向 adapter capability registry

当前版本号：

```text
slam-adapter.scene-manifest/v1alpha1
```

适用场景：

- `GS-SDF` 导出 scene pack
- `CityGaussian` 导入 scene
- `SGS-SLAM` 导入 scene
- 任何能产出离线资产的 SLAM / reconstruction 方法

### 2. Live Contract

文件：`schemas/live-contract.schema.json`

职责：

- 描述在线 topic / endpoint
- 声明通道语义，而不是把 topic 名硬编码进前端
- 允许同一个 UI 在 `neural`、`playback`、`navigation` 等 profile 间切换

当前版本号：

```text
slam-adapter.live-contract/v1alpha1
```

核心思想：

- `key` 是 UI 认的逻辑通道名
- `semanticType` 是数据语义
- `topic` / `messageType` 只是某个 transport 的绑定细节

### 3. Capability Matrix

文件：`schemas/adapter-capability-matrix.schema.json`

职责：

- 注册每个 adapter 会什么
- 区分 `native / derived / fallback / passthrough / none`
- 让系统知道某个 source 有没有 gaussian、mesh、occupancy、live feed、导航发布能力

当前版本号：

```text
slam-adapter.adapter-capability-matrix/v1alpha1
```

这层的作用不是传 runtime 数据，而是做“能力声明”和后续集成治理。

## 规范化词汇

### 静态 artifact

| 字段 | 语义 |
| --- | --- |
| `gaussian` | 主视觉 3DGS 层 |
| `mesh` | 碰撞层或几何可视化层 |
| `occupancy` | 2D 俯视地图输入 |
| `rawPointCloud` | 3D overlay / fallback 点云 |
| `trajectory` | 轨迹产物或离线轨迹 |
| `rgb` | RGB 帧或图像缓存 |
| `depth` | 深度图或深度缓存 |
| `evaluation` | benchmark / 指标产物 |

### Live channel key

当前仓库内建议保留这些 canonical key：

| key | 语义 | 当前默认 topic |
| --- | --- | --- |
| `robot_pose` | 在线机器人位姿 | `/neural_mapping/pose` |
| `trajectory` | 在线轨迹 | `/neural_mapping/path` |
| `live_point_cloud` | 在线建图点云 | `/neural_mapping/pointcloud` |
| `rgb_frame` | 在线 RGB | `/neural_mapping/rgb` |
| `depth_frame` | 在线 depth | `/neural_mapping/depth` |
| `playback_odometry` | 回放位姿 | `/aft_mapped_to_init` |
| `playback_scan_body` | 回放机体系点云 | `/cloud_registered_body` |
| `playback_scan_body_web` | 回放网页降采样点云 | `/cloud_registered_body_web` |
| `playback_rgb` | 回放 RGB | `/origin_img` |
| `playback_global_map` | 回放全局地图 | `/fastlivo/global_map_web` |
| `playback_current_scan_world` | 回放世界系当前帧点云 | `/fastlivo/current_scan_world_web` |
| `nav_goal` | 发布导航目标 | `/move_base_simple/goal` |
| `initial_pose` | 发布初始位姿 | `/initialpose` |
| `camera_pose` | 发布当前观察相机位姿 | `/rviz/current_camera_pose` |

这些 key 是 adapter 层的稳定入口。topic 可以换，key 不应该乱变。

## 最小接入流程

接一个新算法时，优先按下面的顺序做：

1. 选 integration mode

可选值：

- `scene-pack`
- `ros-live`
- `playback-cache`
- `hybrid`
- `importer-only`

2. 如果有离线产物，就输出 `scene manifest`

要求：

- 产物字段尽量走 canonical vocabulary
- 没有的字段不要伪造
- fallback 明确写到 `assets`

3. 如果有在线流，就写 `live contract`

要求：

- 先映射语义，再映射 topic
- 不要把算法私有 topic 直接散落在前端代码里
- 新算法优先复用既有 key 和 semantic type

4. 在 capability matrix 里加一行

要求：

- 明确哪些能力是 `native`
- 哪些只是 `derived` 或 `fallback`
- 把 `sceneManifest.producerScript` 和 `canonicalContract` 指到真实文件

5. 只有在引入新语义时才改前端

例如：

- 如果只是把新算法位姿映射到 `pose3d`
- 或把新点云映射到 `pointcloud3d`

那么前端不应该再加算法特判。

## 当前仓库落点

这套 contract 已经和当前实现对齐到以下位置：

- `examples/web-ui/src/lib/gs/gsSdfSceneAdapter.ts`
- `examples/web-ui/src/lib/ros/useNeuralMappingRos.ts`
- `examples/web-ui/src/lib/ros/navGoalPublisher.ts`
- `scripts/sync_scene_pack.py`
- `scripts/import_citygaussian_scene.py`
- `scripts/import_sgsslam_scene.py`

其中：

- `sync_scene_pack.py` 负责把 `GS-SDF` 输出对齐到 `scene manifest`
- `import_citygaussian_scene.py` 负责把 `CityGaussian` 输出对齐到同一份 contract
- `import_sgsslam_scene.py` 负责把 `SGS-SLAM` 输出对齐到同一份 contract

## 示例文件

参考文件：

- `contracts/scene-manifest.example.json`
- `contracts/live-contract.default.json`
- `contracts/adapter-capability-matrix.example.json`

用途分别是：

- scene pack 样例
- 默认 ROS bridge 通道定义
- 当前仓库的 adapter registry 样例

## 设计原则

- adapter 的成本不可避免，但应该“一次适配，处处复用”
- UI 认语义，不认算法
- schema 先保守，先统一已有字段，再慢慢扩
- `source.kind` 和 capability matrix 用来承载算法差异，不把差异硬编码进 viewer

如果后面要继续扩，这三层已经足够支撑：

- `ORB-SLAM3`
- `RTAB-Map`
- `LIO-SAM`
- `VINS-Fusion`
- 任意能导出 gaussian / mesh / point cloud / pose 的方法
