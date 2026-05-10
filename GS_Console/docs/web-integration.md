# GS-SDF Output To Web UI

## 目标

把 GS-SDF 的输出目录同步到前端可直接消费的 `scene pack`：

- `manifest.json`
- `model/gs.ply`
- `model/as_occ_prior.ply`
- `model/mesh*.ply`

默认落点：

`/home/chatsign/gs-sdf/examples/web-ui/public/scenes`

## 脚本

- 状态查看：
  `/home/chatsign/gs-sdf/scripts/gssdf_status.sh`
- 单次同步：
  `/home/chatsign/gs-sdf/scripts/sync_scene_pack.py`
- 持续同步：
  `/home/chatsign/gs-sdf/scripts/watch_scene_pack.sh`

## 当前正式训练

- 容器：
  `fast-livo2-compressed-cbd`
- 输出目录：
  `/home/chatsign/gs-sdf/runtime/output/2026-04-02-21-33-37_fast_livo2_compressed.bag_fastlivo_cbd_host.yaml`
- 配置：
  `/home/chatsign/gs-sdf/config/fastlivo_cbd_host.yaml`

## 启动持续同步

```bash
/home/chatsign/gs-sdf/scripts/watch_scene_pack.sh
```

生成的 scene id 默认是：

`fast-livo2-compressed-live`

对应 manifest：

`/home/chatsign/gs-sdf/examples/web-ui/public/scenes/fast-livo2-compressed-live/manifest.json`

## 和前端对接

前端直接加载：

```ts
await adapter.bootstrap(
  canvas,
  "/scenes/fast-livo2-compressed-live/manifest.json"
);
```

如果你要边训练边看，把 manifest 轮询刷新加上：

```ts
await adapter.bootstrap(
  canvas,
  "/scenes/fast-livo2-compressed-live/manifest.json"
);

setInterval(async () => {
  await adapter.refreshManifest(
    "/scenes/fast-livo2-compressed-live/manifest.json"
  );
}, 5000);
```

## 当前已知情况

- 训练正在跑时，manifest 会先处于 partial 状态
- `as_occ_prior.ply` 通常会先出现
- `gs.ply` 要到更后面的阶段才会出现
- 你可以先用 occupancy 和 mesh 驱动 2D/3D UI，同步等待 gaussian 资产出现
- `GsSdfSceneAdapter` 现在支持先无 `gaussian` 字段启动，后续通过 `refreshManifest()` 补加载
