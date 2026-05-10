# 新数据接入 UI 指南

这份文档专门回答一个问题：

如果你手上有一份新的数据，应该怎么接入当前这套 `GS-SDF Live / GS Console` UI。

这里的“新数据”可能是 4 种情况：

1. 一份新的 `FAST-LIVO2` 处理后 rosbag
2. 一份新的 `GS-SDF` 输出目录
3. 一份新的高斯 / mesh / occupancy 资产，但还没有 scene pack
4. 一份话题名、外参、配置都和当前默认不完全一致的数据

这份文档不讲论文原理，只讲当前工作区里**实际可执行**的接入流程。

## 1. 先判断你属于哪种接入路径

最简单的判断方式：

- 你只想先看 `rosbag` 回放、RGB、点云逐渐长出来：
  走 `Playback 接入`
- 你想让这份数据进入 `GS / Live` 主界面，看到 `gaussian / mesh / occupancy`：
  走 `GS-SDF 训练 + scene pack 接入`
- 你已经有现成的 `GS-SDF output`：
  走 `输出目录 -> scene pack 接入`
- 你只有散落的资产文件，不是标准输出目录：
  走 `手工 scene pack 接入`

## 2. 当前 UI 在吃什么数据

先明确 UI 三个模式各自依赖什么。

### 2.1 Playback 模式

`Playback` 模式优先消费这三类数据：

- playback 控制 API：
  `http://localhost:8765`
- rosbridge：
  `ws://localhost:9090`
- FAST-LIVO2 回放话题：
  - `/aft_mapped_to_init`
  - `/cloud_registered_body`
  - `/origin_img`

如果走 sidecar 路线，浏览器会优先订阅：

- `/fastlivo/current_scan_world`
- `/fastlivo/global_map`

所以 `Playback` 的本质是：

- bag / cache 驱动
- world-frame 点云
- RGB 回放
- timeline / seek / preview

### 2.2 Live 模式

`Live` 模式优先吃：

- ROS bridge 进来的 pose / path / pointcloud / rgb / depth
- 如果 live pointcloud 不稳定，则退回 scene pack 里的静态几何：
  `rawPointCloud` 或 `occupancy`

所以 `Live` 的本质是：

- ROS 运行态优先
- 静态 scene asset 兜底

### 2.3 GS 模式

`GS` 模式完全依赖 scene pack：

- `manifest.json`
- `gaussian`
- `mesh`
- `rawPointCloud` / `occupancy`

所以如果你想让新数据进入 `GS` 模式，核心不是 rosbridge，而是：

- 先得到合格的 scene pack

## 3. 路径 A：新的 FAST-LIVO2 bag，只想先接 Playback

这是最快的一条路。

适合：

- 先验证数据是否可用
- 先看 RGB + 点云 + timeline
- 不急着训练 GS-SDF

### 3.1 数据前提

当前 playback 这条链路默认假设 bag 内有这些 topic：

- `/aft_mapped_to_init`
- `/cloud_registered_body`
- `/origin_img`

你可以先检查：

```bash
rosbag info /path/to/your_new_data.bag
```

如果 topic 名完全不同，这条链路不能直接无改动复用。

### 3.2 直接跑 sidecar playback

推荐命令：

```bash
RATE=5 START_PAUSED=1 bash /home/chatsign/gs-sdf/scripts/launch_fastlivo_sidecar_playback_stack.sh \
  /path/to/your_new_data.bag \
  fastlivo-direct-playback \
  9090 \
  5173 \
  8765
```

打开：

```text
http://localhost:5173/?mode=playback
```

你应该看到：

- 主视图点云逐渐长出来
- 右侧 `Playback RGB`
- 底部 timeline 可拖动
- 2D mini-map 可编辑、可下 goal、可做 local path preview

### 3.3 建议先 build 一次 playback cache

如果你会反复看同一个新 bag，不要每次都重新扫 rosbag。

先 build cache：

```bash
bash /home/chatsign/gs-sdf/scripts/build_fastlivo_playback_cache.sh \
  /path/to/your_new_data.bag \
  /home/chatsign/gs-sdf/runtime/playback-cache/your_new_data
```

后面统一从 cache 启动：

```bash
MJPEG_MAX_FPS=90 START_PAUSED=1 bash /home/chatsign/gs-sdf/scripts/launch_fastlivo_cached_playback_stack.sh \
  /home/chatsign/gs-sdf/runtime/playback-cache/your_new_data \
  fastlivo-cached-playback \
  9090 \
  5173 \
  8765
```

### 3.4 什么时候需要重建 cache

同一个 bag 不需要每次重建。

只有这些情况才建议重建：

- 你换了新的 bag
- 你改了缓存精度参数，例如：
  `SCAN_MAX_POINTS`、`MAP_MAX_POINTS`、`KEYFRAME_EVERY`
- 你改了相机/外参配置，导致投色逻辑需要重算

## 4. 路径 B：新的 FAST-LIVO2 bag，要完整接入 Live / GS

这条路适合：

- 你不仅要看 playback
- 还想让这份数据进入 `GS` 模式
- 想要 `gaussian / mesh / occupancy` 三层都进浏览器

完整链路是：

```text
new FAST-LIVO2 bag
-> GS-SDF train
-> runtime/output/<run>
-> gaussian preprocess
-> sync_scene_pack.py
-> Web UI manifest
```

### 4.1 准备配置

默认可从当前配置起步：

```text
/home/chatsign/gs-sdf/config/fastlivo_cbd_host.yaml
```

如果你的新数据：

- 相机外参不同
- `map_origin` / `map_size` 不同
- 相机内参不同

建议复制一份新配置，例如：

```text
/home/chatsign/gs-sdf/config/fastlivo_my_scene.yaml
```

再按你的数据改：

- 相机内参
- 外参
- `leaf_sizes`
- `map_origin`
- `map_size`

### 4.2 启动训练

```bash
bash /home/chatsign/gs-sdf/scripts/train_gssdf.sh \
  /path/to/your_new_data.bag \
  /home/chatsign/gs-sdf/config/fastlivo_my_scene.yaml \
  your-new-run
```

查看状态：

```bash
bash /home/chatsign/gs-sdf/scripts/gssdf_status.sh
```

### 4.3 得到输出目录

训练产物通常会落到：

```text
/home/chatsign/gs-sdf/runtime/output/<timestamp>_<bag>_<config>
```

一个可用的输出目录，至少应该关注这些文件：

- `model/gs.ply`
- `model/as_occ_prior.ply`
- `mesh*.ply` 或 `model/mesh*.ply`

### 4.4 预处理高斯

先做 chunked stream，优先保证浏览器能吃：

```bash
node /home/chatsign/gs-sdf/scripts/preprocess_gaussian_stream.mjs \
  --output-dir /home/chatsign/gs-sdf/runtime/output/<run-dir> \
  --grid 6,6 \
  --max-sh 1 \
  --opacity-threshold 0.02
```

如果你只想先看规划，不想真的写文件，可以先 `--dry-run`。

### 4.5 同步成 scene pack

```bash
python3 /home/chatsign/gs-sdf/scripts/sync_scene_pack.py \
  --output-dir /home/chatsign/gs-sdf/runtime/output/<run-dir> \
  --scene-config /home/chatsign/gs-sdf/config/fastlivo_my_scene.yaml \
  --status completed \
  --scene-id your-new-scene
```

输出会到：

```text
/home/chatsign/gs-sdf/examples/web-ui/public/scenes/your-new-scene
```

最关键的文件是：

```text
/home/chatsign/gs-sdf/examples/web-ui/public/scenes/your-new-scene/manifest.json
```

### 4.6 启动 saved view + rosbridge + Web UI

```bash
bash /home/chatsign/gs-sdf/scripts/launch_gssdf_view.sh \
  /home/chatsign/gs-sdf/runtime/output/<run-dir> \
  gssdf-view

bash /home/chatsign/gs-sdf/scripts/launch_rosbridge_sidecar.sh gssdf-view 9090

bash /home/chatsign/gs-sdf/scripts/launch_web_ui_dev.sh 0.0.0.0 5173
```

打开：

```text
http://localhost:5173/?scene=/scenes/your-new-scene/manifest.json
```

如果你想直接进 `GS`：

```text
http://localhost:5173/?scene=/scenes/your-new-scene/manifest.json&mode=gs
```

## 5. 路径 C：你已经有现成的 GS-SDF output

这时不需要重新训练。

只需要做两步：

1. 预处理高斯
2. 同步 scene pack

### 5.1 预处理

```bash
node /home/chatsign/gs-sdf/scripts/preprocess_gaussian_stream.mjs \
  --output-dir /path/to/existing_output_dir \
  --grid 6,6 \
  --max-sh 1 \
  --opacity-threshold 0.02
```

### 5.2 同步 scene pack

```bash
python3 /home/chatsign/gs-sdf/scripts/sync_scene_pack.py \
  --output-dir /path/to/existing_output_dir \
  --scene-config /home/chatsign/gs-sdf/config/fastlivo_cbd_host.yaml \
  --status completed \
  --scene-id existing-scene
```

然后直接打开：

```text
http://localhost:5173/?scene=/scenes/existing-scene/manifest.json
```

## 6. 路径 D：只有散落资产，不是标准输出目录

这是最不推荐但最灵活的一条路。

比如你只有：

- 一个 `gaussian` 文件
- 一个 `mesh.ply`
- 一个 `occupancy` 点云

但没有完整的 GS-SDF output 目录。

### 6.1 最低要求

你至少要准备一个 scene pack 目录，例如：

```text
/home/chatsign/gs-sdf/examples/web-ui/public/scenes/my-manual-scene
```

以及一个 `manifest.json`。

最关键的字段通常是：

- `sceneId`
- `frameId`
- `gaussian`
- `mesh`
- `rawPointCloud` 或 `occupancy`
- `robot`
- `meta`

### 6.2 推荐做法

不要手搓所有结构，优先先拿一个已有 scene pack 复制一份，再替换资源路径。

推荐起点：

```text
/home/chatsign/gs-sdf/examples/web-ui/public/scenes/fast-livo2-compressed-live
```

复制后改：

- `sceneId`
- `gaussian.url` 或 `gaussian.chunks`
- `mesh.url`
- `rawPointCloud.url`
- `meta.leafSize`
- `meta.mapOrigin`
- `meta.mapSize`

### 6.3 什么时候必须手工改 manifest

这些情况通常绕不开手改：

- 你的高斯不是 `GS-SDF` 默认输出结构
- 你的 `PLY` 文件不在标准 `model/` 路径下
- 你的 `frameId` 不是默认 `world`
- 你的地图原点 / 分辨率不是当前 scene config 能自动抽出来的

## 7. 如果新数据的话题名不一样，怎么办

当前 playback / sidecar 这条链默认依赖：

- `/aft_mapped_to_init`
- `/cloud_registered_body`
- `/origin_img`

如果你的新数据不是这些名字，有 3 种选择。

### 7.1 最推荐：先做 topic 对齐

把 bag 做一层重命名/预处理，让它长得像当前默认 FAST-LIVO2 输出。

这是最稳的，因为：

- 训练脚本不用改
- playback sidecar 不用改
- UI 不用改

### 7.2 中等成本：改 playback server

如果你只想让它进 `Playback`，可以改：

- `/home/chatsign/gs-sdf/scripts/fastlivo_direct_playback_server.py`
- `/home/chatsign/gs-sdf/scripts/fastlivo_cached_playback_server.py`

把默认 topic 名改成你的数据实际名字。

### 7.3 成本最高：同时改 train / playback / view

如果 topic 语义本身就不一样，比如不是 FAST-LIVO2 输出，而是更底层的：

- `/livox/lidar`
- `/imu`
- `/camera/image_color/compressed`

那就不是简单改 topic 名的问题，而是整条 parser / preprocessing 路线都要重新定义。

## 8. Scene ID 和命名建议

为了避免后面 scene 越堆越乱，建议统一命名。

### 8.1 scene id

建议：

```text
<dataset>-<site>-<variant>
```

例如：

- `fastlivo-campus-a`
- `fastlivo-cbd-night`
- `replica-room2-quality`

### 8.2 run name

训练时建议：

```text
<dataset>-<slice-or-scene>-<config>
```

例如：

- `fast-livo2-campus-cbd`
- `fast-livo2-mid30s-cbd`

### 8.3 cache 目录

建议和 bag 对齐：

```text
runtime/playback-cache/<bag-stem>
```

例如：

```text
runtime/playback-cache/fast_livo2_compressed
runtime/playback-cache/campus_loop_a
```

## 9. UI 接入后的验证清单

无论你走哪条路径，最后都建议按这份 checklist 过一遍。

### 9.1 Playback 验证

- 页面能打开 `?mode=playback`
- 右侧有 `Playback RGB`
- timeline 可以拖动
- `Current Scan / Global Map` 会随 seek 恢复
- 右侧 2D map 能显示 build-up

### 9.2 Live 验证

- 轨迹和机器人位姿能显示
- pointcloud / fallback cloud 能显示
- top-down map 可点击 goal
- local path preview 能生成

### 9.3 GS 验证

- `Gaussian` 图层能打开
- `SDF Mesh` 能显示
- `Orbit / FPS` 两种控制都可用
- 切换到新 scene manifest 不报错

### 9.4 Map 编辑验证

- `Obstacle` 工具能加障碍
- `Erase` 工具能清除噪声
- `Save PGM` 能下载 `.pgm`
- 编辑后的地图会影响 local A* path

### 9.5 Nav Preview 验证

- `Run Preview` 能让机器人沿 `plannedPath` 运动
- 2D / 3D 轨迹同步更新
- 停止后不会污染原始 ROS 数据

## 10. 常见错误与对应处理

### 10.1 新 bag 可以 playback，但不能训练

先查：

- 它是不是 FAST-LIVO2 处理后的 bag
- 它有没有：
  `/aft_mapped_to_init`
  `/cloud_registered_body`
  `/origin_img`

如果没有，这条训练链不能直接复用。

### 10.2 新 scene pack 能打开，但 GS 没有画面

先查：

- `manifest.json` 里的 `gaussian.url` 或 `gaussian.chunks` 是否真实存在
- `sync_scene_pack.py` 是否成功把链接或文件放到了 `public/scenes/<scene-id>`
- 浏览器控制台有没有 WebGL / resource 404

### 10.3 新 playback 启动时端口冲突

先清：

```bash
docker rm -f fastlivo-direct-playback fastlivo-cached-playback gssdf-rosbridge >/dev/null 2>&1 || true
```

### 10.4 `Save PGM` 后找不到文件

当前 `Save PGM` 是浏览器下载行为：

- 文件在浏览器默认下载目录
- 不是直接写入工作区
- 也不会自动生成 `.yaml`

## 11. 推荐的最小接入策略

如果你以后拿到一份新数据，最稳妥的顺序建议是：

1. 先确认 bag 里有没有当前 playback 需要的话题
2. 先走 `Playback`
3. 确认 RGB / 点云 / 轨迹 / timeline 都正常
4. 再决定要不要训练 `GS-SDF`
5. 训练完成后再做 `preprocess + sync_scene_pack`
6. 最后才切到 `GS` 看高斯

一句话总结：

- `Playback` 是最低成本验数路径
- `scene pack` 是进入 `GS / Live` 的关键
- `GS-SDF output` 是高保真层的来源
- `cache` 是长期复用同一个 bag 的关键
