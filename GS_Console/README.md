# GS-SDF Web UI

This repository wraps the existing `GS-SDF` and `FAST-LIVO2` runtime into a browser-first console for:

- cached playback driven by a single master timeline
- 2D map editing and replay inspection
- side-by-side RGB vs GS-SDF compare video
- scene-pack based `GS` / `HiFi` / `Live` views
- contract-driven adapter integration for multiple SLAM / mapping backends
- remote access through `rosbridge` and a Vite web UI

This README is the current operational guide for this workspace. It is not a generic upstream GS-SDF manual.

## Current status

As of `2026-04-21`, the mainline that is working best in this repo is:

- playback-first workflow
- cached FAST-LIVO2 playback
- 1x compare video as the visual master timeline
- `kf16` playback cache for denser restore and lower seek lag
- contract-first scene/runtime integration in the web UI

Active defaults in this workspace:

- active cache:
  `/home/chatsign/gs-sdf/runtime/playback-cache/fast_livo2_compressed_rgbdense_kf16`
- active 1x compare video:
  `/home/chatsign/gs-sdf/runtime/videos/playback_rgb_vs_gssdf_quality_v2_1x.mp4`
- active stack command:
  see `/home/chatsign/gs-sdf/command.txt`

World-navigation MVP:

- control API:
  `scripts/world_nav_module_server.py` on `http://127.0.0.1:8892`
- live Isaac bridge:
  `scripts/isaac_gaussian_online_bridge_server.py` on `http://127.0.0.1:8890`
- WebUI scene:
  `/scenes/isaac-gaussian-online/manifest.json`
- runbook:
  [`docs/world-nav-module-mvp.md`](docs/world-nav-module-mvp.md)
- NuRec / real-data adapter:
  [`scripts/nurec_to_gssdf_colmap.py`](scripts/nurec_to_gssdf_colmap.py) and
  [`docs/nurec-gssdf-adapter.md`](docs/nurec-gssdf-adapter.md)

What is implemented and verified:

- `Playback` mode with timeline, pause, seek, restart, speed control
- playback cache build from rosbag
- split overlay fetches for scan and accumulated map
- 2D map editing:
  `Goal`, `Initial Pose`, `Obstacle`, `Erase`, `Save PGM`
- `Workspace` and `Showcase` playback presentations
- compare video side-by-side:
  `Playback RGB` vs `GS-SDF Gaussian`
- 1x compare video export with sidecar metadata
- denser `kf16` playback cache
- browser-side `Live`, `Gs`, `HiFi` shell integration
- runtime `scene manifest + live contract + capability matrix` integration
- `GS-SDF`, `CityGaussian`, and `SGS-SLAM` scene import/export aligned to one browser contract

Current boundaries:

- `Playback` is the best-supported path
- `Live` and browser `Gs` require WebGL in the browser session
- native `HiFi` depends on the GS-SDF native renderer and free GPU capacity
- high playback rates can still show some lag in accumulated map generation because the backend still reconstructs map snapshots on demand

## Why Gaussian quality can be worse than expected

Yes, the source SLAM / odometry frontend can affect the final Gaussian result.

For the current FAST-LIVO-derived runs, likely quality bottlenecks are:

- camera-LiDAR extrinsic or timestamp error, which makes RGB supervision project onto the wrong geometry
- trajectory drift or local pose jitter, which creates duplicated edges and blurry splats
- sparse or uneven RGB coverage, especially under roofs, reflective surfaces, and motion blur
- exposure changes and rolling-shutter artifacts in the image stream
- weak geometric regularization in gaussian-only methods compared with GS-SDF's SDF / mesh constraints

This is why SGS-SLAM output can look worse than GS-SDF even when both are rendered as Gaussian splats.
It does not necessarily mean the web renderer is wrong.
It usually means the upstream reconstruction signal, optimization objective, or exported Gaussian representation is weaker.

The practical comparison rule in this workspace is:

| Baseline | What it measures | Main limitation |
| --- | --- | --- |
| FAST-LIVO playback | odometry, RGB, point-cloud replay quality | not a photoreal renderer |
| GS-SDF | Gaussian + SDF-constrained reconstruction | tied to GS-SDF training/export pipeline |
| SGS-SLAM | gaussian SLAM import quality | current imported asset is SH0 / blob-heavy |
| CityGaussian | large-scene gaussian rendering baseline | offline scene-pack, not live SLAM |
| LingBot-Map | non-gaussian streaming 3D reconstruction baseline | runs its own viser viewer; not a Spark/GS renderer |

## LingBot-Map baseline

`LingBot-Map` is now registered as an external adapter baseline:

- adapter source kind: `lingbot_map`
- capability contract: `/contracts/adapter-capability-matrix.default.json`
- live contract: `/contracts/lingbot-map.live-contract.json`
- frontend scene manifest: `/scenes/lingbot-map-viewer/manifest.json`
- external viewer URL: `http://localhost:8080`

This integration deliberately does **not** treat LingBot-Map as a Gaussian Splatting method.
The upstream project describes it as a streaming 3D reconstruction model and its demo serves a browser-based `viser` viewer.
The GS-SDF web UI now links to that viewer and keeps the same adapter/manifest/capability structure, so it can be compared as another baseline without pretending it is a Gaussian renderer.

### Start LingBot-Map viewer

First run LingBot-Map from its own Python environment:

```bash
LINGBOT_MODEL_PATH=/media/chatsign/data-002/models/lingbot-map/lingbot-map-long.pt \
LINGBOT_IMAGE_FOLDER=/home/chatsign/work/lingbot-map/example/church \
LINGBOT_PORT=8080 \
bash /home/chatsign/gs-sdf/scripts/launch_lingbot_map_viewer.sh
```

Then start the web UI with the LingBot-Map adapter preset:

```bash
WEB_ADAPTER=lingbot-map WEB_PORT=5173 \
bash /home/chatsign/gs-sdf/scripts/launch_web_ui_dev.sh
```

Open:

```text
http://localhost:5173/?scene=/scenes/lingbot-map-viewer/manifest.json&mode=gs
```

The header shows `Open LingBot-Map`, which opens the viser viewer on `http://localhost:8080`.

For video input instead of an image folder:

```bash
LINGBOT_MODEL_PATH=/media/chatsign/data-002/models/lingbot-map/lingbot-map-long.pt \
LINGBOT_VIDEO_PATH=/path/to/video.mp4 \
LINGBOT_FPS=10 \
LINGBOT_PORT=8080 \
bash /home/chatsign/gs-sdf/scripts/launch_lingbot_map_viewer.sh
```

### Use LingBot-Map inside Playback

The `Playback` UI cannot consume the raw LingBot-Map viser process directly.
It can consume a LingBot-generated playback cache with the same layout as the current FAST-LIVO cache:

- `images/frame_*.jpg`
- `scans/scan_*.npz`
- `keyframes/map_*.npz`
- `meta.json` with pose, timing, and frame metadata

Build a LingBot playback cache:

```bash
python3 /home/chatsign/gs-sdf/scripts/build_lingbot_playback_cache.py \
  --image-folder /home/chatsign/work/lingbot-map/example/church \
  --output-dir /home/chatsign/gs-sdf/runtime/playback-cache/lingbot_church \
  --fps 10 \
  --scan-max-points 12000 \
  --map-max-points 240000 \
  --overwrite
```

Launch the existing playback stack against that cache:

```bash
START_PAUSED=1 \
bash /home/chatsign/gs-sdf/scripts/launch_lingbot_playback_stack.sh \
  /home/chatsign/gs-sdf/runtime/playback-cache/lingbot_church \
  lingbot-map-playback \
  9090 \
  5173 \
  8765
```

Open:

```text
http://localhost:5173/?mode=playback
```

This replaces FAST-LIVO as the playback data source for the browser timeline.
It does not make LingBot-Map a ROS odometry frontend, and it does not provide LiDAR-native FAST-LIVO topics.

## What changed from the previous method

Before the adapter-contract refactor, the web UI mostly assumed:

- specific ROS topic names such as `/neural_mapping/*`, `/aft_mapped_to_init`, `/cloud_registered_body`, `/origin_img`
- a mostly fixed feature set:
  Gaussian, mesh, occupancy cloud, playback scan, nav publishing
- frontend logic that was still biased toward `FAST-LIVO2 + GS-SDF`

Now the runtime is split into three layers:

1. `scene manifest`
   static assets and scene metadata
2. `live contract`
   semantic live channels mapped to ROS topics or other transports
3. `capability matrix`
   what a given adapter can actually provide to the UI

In practical terms, the difference is:

| Old method | Current method | Why it matters |
| --- | --- | --- |
| topics effectively hardcoded in the frontend | topics resolved through `live contract` | new adapters do not require frontend topic rewrites |
| UI modes and toggles assumed the same backend features | mode/tool/layer availability driven by `capability matrix + manifest + live contract` | unsupported functions are disabled instead of failing implicitly |
| scene packs mainly described assets | scene packs can also point to runtime contracts via `source.liveContractUrl` and `source.capabilityMatrixUrl` | the same UI can load different backend families cleanly |
| adding a new SLAM backend usually meant frontend edits | adding a new backend is now primarily `manifest + live contract + capability matrix` work | integration cost moves into adapters, not UI forks |

This does **not** mean the core playback / Docker / rosbridge / Vite stack was replaced.
It means the UI no longer assumes one fixed backend contract.

## Do you need to change the launch command

Short answer:

- for the current `FAST-LIVO2` playback workflow: **no**
- for normal `GS-SDF` scene viewing: **no**
- for `CityGaussian` / `SGS-SLAM` scenes generated by the updated importers: **usually no**
- for `LingBot-Map`: **yes**, because it runs a separate Python `viser` server and the web UI only links to it as an external baseline

The current change is mostly a **runtime contract interpretation change**, not a stack-launch change.
`LingBot-Map` is the exception because it is not a ROS/GS-SDF renderer process.

### Commands that stay the same

These commands are still the normal path:

- `build_fastlivo_playback_cache.sh`
- `launch_fastlivo_cached_playback_stack.sh`
- `launch_gssdf_hifi_stack.sh`
- `sync_scene_pack.py`
- `import_citygaussian_scene.py`
- `import_sgsslam_scene.py`

The web UI still starts through Vite the same way, and rosbridge is still launched the same way.

### What changed instead

What changed is what the web UI reads **after it starts**:

- it now loads a `live contract`
- it now loads a `capability matrix`
- it now uses `manifest.source.kind` to resolve the active adapter row

### Command migration examples

For the common paths, think of the migration like this:

| Task | Previous way | Current way |
| --- | --- | --- |
| start cached FAST-LIVO2 playback | same shell command | same shell command |
| start GS-SDF native HiFi | same shell command | same shell command |
| open playback UI | `http://localhost:5173/?mode=playback` | same URL |
| open a normal scene pack | `?scene=...&mode=gs` | same URL |
| open LingBot-Map baseline | separate viewer only | run `launch_lingbot_map_viewer.sh`, then open `/scenes/lingbot-map-viewer/manifest.json` |
| test a custom adapter contract | usually required frontend edits | keep the same UI, add `?liveContract=...` and/or `?capabilityMatrix=...` only if needed |
| import a non-GS-SDF scene | usually custom one-off work | run the importer, then open the generated manifest in the same UI |

So the operational difference is mostly:

- launch commands are stable
- scene metadata is richer
- browser URL overrides are optional, not mandatory

### When you would change something

You only need to change browser startup parameters or scene metadata in these cases:

1. You want to force a non-default runtime contract:

```text
http://localhost:5173/?scene=/scenes/your-scene/manifest.json&mode=gs&liveContract=/contracts/live-contract.default.json
```

2. You want to force a non-default capability registry:

```text
http://localhost:5173/?scene=/scenes/your-scene/manifest.json&mode=gs&capabilityMatrix=/contracts/adapter-capability-matrix.default.json
```

3. You have an older scene pack that was generated before `source.liveContractUrl` and `source.capabilityMatrixUrl` existed, and you want a custom adapter contract without re-exporting the scene.

4. You want the playback stack to auto-resolve HiFi from a different manifest than the default one, in which case you still change:

```bash
HIFI_SCENE_MANIFEST=/abs/path/to/manifest.json
```

### Compatibility rule

For existing scenes and the current playback stack, the UI falls back to built-in defaults if contract files are missing.

So the migration rule is:

- old launch command still works
- updated scene manifests improve correctness
- custom adapters only need URL overrides when you do not want to regenerate the scene pack

## Recommended workflow

If your goal is to inspect data or demo results, use this path:

1. Build or reuse a cached playback pack.
2. Optionally export a 1x compare video.
3. Launch the cached playback stack.
4. Open `http://localhost:5173/?mode=playback`.

If your goal is to inspect Gaussian assets or native rendering, only then move to `Gs` or `HiFi`.

## Prerequisites

The current scripts assume the following are already available on the machine:

- Docker
- a working GPU Docker runtime
- image `gs_sdf_img:latest`
- ROS Noetic tooling inside the image
- `node` and `npm` on the host for the web UI
- `python3` on the host

For playback from a new bag, the bag should contain these topics:

- `/aft_mapped_to_init`
- `/cloud_registered_body`
- `/origin_img`

You can verify that with:

```bash
rosbag info /path/to/your_data.bag
```

For browser `Live` / `Gs` modes:

- the browser must support WebGL
- hardware acceleration should be enabled

For native `HiFi`:

- the GS-SDF view container must be able to publish `/neural_mapping/rgb`
- GPU memory pressure must be low enough for the renderer to stay alive

## Quick start for the current dataset

The current verified command is:

```bash
VIEW_WIDTH=1920 VIEW_HEIGHT=1536 MJPEG_MAX_FPS=90 START_PAUSED=1 \
bash /home/chatsign/gs-sdf/scripts/launch_fastlivo_cached_playback_stack.sh \
  /home/chatsign/gs-sdf/runtime/playback-cache/fast_livo2_compressed_rgbdense_kf16 \
  fastlivo-cached-playback \
  9090 \
  5173 \
  8765
```

Open:

```text
http://localhost:5173/?mode=playback
```

This starts:

- cached playback control API on `8765`
- rosbridge sidecar on `9090`
- web UI dev server on `5173`
- native HiFi stack if it can be resolved from the current scene manifest and `DISABLE_HIFI` is not set

## Browser URLs and runtime overrides

### Normal playback

```text
http://localhost:5173/?mode=playback
```

### Normal scene-pack review

```text
http://localhost:5173/?scene=/scenes/your-scene/manifest.json&mode=gs
```

### Scene-pack review with explicit contract override

Use this when testing a new adapter contract without regenerating the scene:

```text
http://localhost:5173/?scene=/scenes/your-scene/manifest.json&mode=gs&liveContract=/contracts/live-contract.default.json&capabilityMatrix=/contracts/adapter-capability-matrix.default.json
```

### What the UI resolves at runtime

When a scene is opened, the UI now resolves in this order:

1. `?liveContract=` and `?capabilityMatrix=` query overrides
2. `manifest.source.liveContractUrl` and `manifest.source.capabilityMatrixUrl`
3. built-in defaults under:
   - `/contracts/live-contract.default.json`
   - `/contracts/adapter-capability-matrix.default.json`

That is why most launch commands do not need to change.

## What the main scripts do

Core playback scripts:

- `/home/chatsign/gs-sdf/scripts/build_fastlivo_playback_cache.sh`
  builds a playback cache from a rosbag
- `/home/chatsign/gs-sdf/scripts/launch_fastlivo_cached_playback.sh`
  launches only the cached playback control server
- `/home/chatsign/gs-sdf/scripts/launch_fastlivo_cached_playback_stack.sh`
  launches cached playback, rosbridge, optional HiFi, and the web UI
- `/home/chatsign/gs-sdf/scripts/fastlivo_cached_playback_server.py`
  serves `/status`, `/overlay.json`, `/video_sync`, `/video.mp4`, `/frame.jpg`
- `/home/chatsign/gs-sdf/scripts/export_playback_rgb_vs_gssdf_video.py`
  exports the side-by-side compare video and writes `playback_video.json`

Scene / GS-SDF scripts:

- `/home/chatsign/gs-sdf/scripts/train_gssdf.sh`
  runs GS-SDF training or processing
- `/home/chatsign/gs-sdf/scripts/preprocess_gaussian_stream.mjs`
  prepares streamed Gaussian chunks
- `/home/chatsign/gs-sdf/scripts/sync_scene_pack.py`
  syncs a GS-SDF output directory into the web scene pack format and writes contract URLs into the manifest
- `/home/chatsign/gs-sdf/scripts/import_citygaussian_scene.py`
  imports a `CityGaussian` output directory into the same scene-pack contract
- `/home/chatsign/gs-sdf/scripts/import_sgsslam_scene.py`
  imports an `SGS-SLAM` output directory into the same scene-pack contract
- `/home/chatsign/gs-sdf/scripts/launch_gssdf_hifi_stack.sh`
  starts native GS-SDF view plus MJPEG bridge

## Minimal onboarding for new data

This is the shortest path if you have new data and want it in the UI quickly.

### Path A: new rosbag, playback only

Use this if you want:

- timeline replay
- 2D map and robot trajectory
- scan and accumulated map
- optional compare video later

You do not need a full GS-SDF scene pack for this path.

#### Step 1: verify the bag

Make sure your bag contains the expected FAST-LIVO2 topics:

```bash
rosbag info /path/to/your_data.bag
```

Expected topics:

- `/aft_mapped_to_init`
- `/cloud_registered_body`
- `/origin_img`

If your topics are different, you should not force this pipeline unchanged. Adjust the playback extraction path first.

#### Step 2: build a playback cache

Recommended current build:

```bash
KEYFRAME_EVERY=16 \
bash /home/chatsign/gs-sdf/scripts/build_fastlivo_playback_cache.sh \
  /path/to/your_data.bag \
  /home/chatsign/gs-sdf/runtime/playback-cache/your_data_rgbdense_kf16
```

Why this is the recommended baseline:

- `KEYFRAME_EVERY=16` improves restore density compared with the older `64`
- the current recommended naming in this repo is `_rgbdense_kf16`
- older names such as `_rgbclean_kf16` still work if the cache contents are valid

Useful cache build environment variables:

- `KEYFRAME_EVERY`
- `SCAN_MAX_POINTS`
- `SCAN_VOXEL_SIZE`
- `MAP_MAX_POINTS`
- `MAP_VOXEL_SIZE`
- `MAX_FRAMES`
- `CONFIG_PATH`

Example with explicit overrides:

```bash
CONFIG_PATH=/home/chatsign/gs-sdf/config/fastlivo_cbd_host.yaml \
KEYFRAME_EVERY=16 \
SCAN_MAX_POINTS=3600 \
MAP_MAX_POINTS=90000 \
bash /home/chatsign/gs-sdf/scripts/build_fastlivo_playback_cache.sh \
  /path/to/your_data.bag \
  /home/chatsign/gs-sdf/runtime/playback-cache/your_data_rgbdense_kf16
```

#### Step 3: launch the cached playback stack

```bash
VIEW_WIDTH=1920 VIEW_HEIGHT=1536 MJPEG_MAX_FPS=90 START_PAUSED=1 \
bash /home/chatsign/gs-sdf/scripts/launch_fastlivo_cached_playback_stack.sh \
  /home/chatsign/gs-sdf/runtime/playback-cache/your_data_rgbdense_kf16 \
  fastlivo-cached-playback \
  9090 \
  5173 \
  8765
```

Open:

```text
http://localhost:5173/?mode=playback
```

At this point you already have the minimum useful integration.

### Path B: add a 1x compare video for that playback cache

Use this if you want the side-by-side `Playback RGB` vs `GS-SDF Gaussian` compare video to be the master timeline.

This path needs:

- a playback cache
- a GS-SDF output directory for the same scene
- a working HiFi bridge

#### Step 1: start the HiFi stack

```bash
VIEW_WIDTH=1920 VIEW_HEIGHT=1536 MJPEG_MAX_FPS=45 KEEP_CONTAINER_ON_EXIT=1 \
bash /home/chatsign/gs-sdf/scripts/launch_gssdf_hifi_stack.sh \
  /path/to/your_gssdf_output_dir \
  gssdf-view-quality \
  8876 \
  gssdf-hifi-bridge
```

Health check:

```bash
curl -fsS http://127.0.0.1:8876/status
```

You want:

- `"ready": true`
- `"framesReceived"` increasing

#### Step 2: export the 1x compare video

```bash
python3 /home/chatsign/gs-sdf/scripts/export_playback_rgb_vs_gssdf_video.py \
  --cache-dir /home/chatsign/gs-sdf/runtime/playback-cache/your_data_rgbdense_kf16 \
  --scene-config /path/to/your_gssdf_output_dir/model/config/scene/config.yaml \
  --bridge-url http://127.0.0.1:8876 \
  --output-video /home/chatsign/gs-sdf/runtime/videos/your_data_rgb_vs_gssdf_1x.mp4 \
  --fps 12 \
  --speed 1.0 \
  --start-offset 0 \
  --end-offset -1
```

What this script also does:

- writes `/home/chatsign/gs-sdf/runtime/playback-cache/your_data_rgbdense_kf16/playback_video.json`
- writes a registry fallback under `/home/chatsign/gs-sdf/runtime/playback-video-index`

That means the cached playback server can pick up the video automatically on the next restart.

#### Step 3: restart the cached playback stack

```bash
VIEW_WIDTH=1920 VIEW_HEIGHT=1536 MJPEG_MAX_FPS=90 START_PAUSED=1 \
bash /home/chatsign/gs-sdf/scripts/launch_fastlivo_cached_playback_stack.sh \
  /home/chatsign/gs-sdf/runtime/playback-cache/your_data_rgbdense_kf16 \
  fastlivo-cached-playback \
  9090 \
  5173 \
  8765
```

Verify:

```bash
curl -fsS http://127.0.0.1:8765/status
```

You want:

- `"video"` not `null`
- `"video.speed": 1.0`
- `"video.url": "video.mp4"`

### Path C: full GS / HiFi scene integration

Use this if you want the new data to appear in `Gs`, `HiFi`, or scene-pack-backed `Live` workflows.

#### Step 1: run GS-SDF

```bash
bash /home/chatsign/gs-sdf/scripts/train_gssdf.sh \
  /path/to/your_data.bag \
  /home/chatsign/gs-sdf/config/fastlivo_cbd_host.yaml \
  your-run-name
```

Check status:

```bash
bash /home/chatsign/gs-sdf/scripts/gssdf_status.sh
```

Expected output directory pattern:

```text
/home/chatsign/gs-sdf/runtime/output/<timestamp>_<bag>_<config>
```

#### Step 2: preprocess the Gaussian asset

```bash
node /home/chatsign/gs-sdf/scripts/preprocess_gaussian_stream.mjs \
  --output-dir /path/to/your_gssdf_output_dir \
  --variant quality \
  --grid 6,6 \
  --max-sh 2 \
  --opacity-threshold 0.005 \
  --force
```

This produces streamed chunk metadata and `spz` chunks for the browser.

#### Step 3: sync the scene pack

```bash
python3 /home/chatsign/gs-sdf/scripts/sync_scene_pack.py \
  --output-dir /path/to/your_gssdf_output_dir \
  --scene-config /home/chatsign/gs-sdf/config/fastlivo_cbd_host.yaml \
  --scene-root /home/chatsign/gs-sdf/examples/web-ui/public/scenes \
  --scene-id your-scene-id \
  --status completed
```

After that, the web UI can load the new scene pack in `Gs` / `HiFi`.

The synced manifest now also carries:

- `source.kind`
- `source.liveContractUrl`
- `source.capabilityMatrixUrl`

That means the web UI can automatically resolve:

- which adapter family the scene belongs to
- which live topic contract to use
- which modes / layers / navigation actions should be enabled

### Path D: non-GS-SDF scene import with the same UI

Use this if the source scene was not produced by the in-repo GS-SDF training path.

#### CityGaussian

```bash
python3 /home/chatsign/gs-sdf/scripts/import_citygaussian_scene.py \
  --citygaussian-dir /path/to/CityGaussian/output \
  --scene-id your-city-scene \
  --scene-root /home/chatsign/gs-sdf/examples/web-ui/public/scenes \
  --preprocess-gaussian
```

Open:

```text
http://localhost:5173/?scene=/scenes/your-city-scene/manifest.json&mode=gs
```

#### SGS-SLAM

```bash
python3 /home/chatsign/gs-sdf/scripts/import_sgsslam_scene.py \
  --experiment-dir /path/to/sgsslam/output \
  --scene-id your-sgs-scene \
  --scene-root /home/chatsign/gs-sdf/examples/web-ui/public/scenes \
  --preprocess-gaussian
```

Open:

```text
http://localhost:5173/?scene=/scenes/your-sgs-scene/manifest.json&mode=gs
```

These importers now write the same runtime contract pointers as `sync_scene_pack.py`.
So from the web UI point of view, they are just different adapters behind one contract.

## What files matter in a playback cache

A playback cache directory should contain at least:

- `meta.json`
- `scans/`
- `images/`
- `keyframes/`

Optional but recommended:

- `playback_video.json`

Key `meta.json` fields you will care about:

- `durationSec`
- `frameCount`
- `keyframeEvery`
- `scanMaxPoints`
- `mapMaxPoints`
- `configPath`
- `frames`

## Current recommended defaults

For new playback data in this repo, the current recommended defaults are:

- cache naming:
  `your_data_rgbdense_kf16`
- `KEYFRAME_EVERY=16`
- compare video:
  `1x`
- compare video fps:
  `12`
- stack start:
  `START_PAUSED=1`

Why:

- `kf16` makes random restore and dense playback more stable than `64`
- `1x` avoids timeline mismatch between video and bag
- `START_PAUSED=1` makes initial verification easier

## Verification checklist

After launching a new cache, verify these in order:

1. `curl -fsS http://127.0.0.1:8765/status`
2. `curl -fsS http://127.0.0.1:8765/topics`
3. open `http://localhost:5173/?mode=playback`
4. confirm timeline moves, 2D map appears, robot/trajectory update

If compare video is expected:

1. `curl -fsS http://127.0.0.1:8765/status`
2. confirm `"video"` is present
3. confirm UI no longer shows offline-video-missing state

## Troubleshooting

### `localhost:5173` does not open

The web UI is a Vite dev server. Common causes:

- the stack was not started
- port `5173` is already occupied locally
- when using SSH forwarding on Windows, `5173` may be in an excluded port range

If local forwarding is blocked, use a different local port such as `55173`.

### `Live` or `Gs` is black

Browser `Live` / `Gs` needs WebGL.

If the page says WebGL is unavailable:

- enable browser hardware acceleration
- use a browser/session with WebGL support
- if on remote desktop, use a mode that preserves GPU/WebGL

### `HiFi` is black

Native HiFi depends on the backend renderer, not just the web UI.

Check:

```bash
curl -fsS http://127.0.0.1:8876/status
```

If it is not ready, the native GS-SDF renderer is not producing frames.

### Playback is globally slow at high speed

This is usually backend throughput, not a pure front-end bug.

The current repo already uses:

- split scan/map overlay requests
- denser keyframes
- exact per-frame overlay map restore

If that is still not enough, the next step is precomputing denser playback map snapshots rather than tuning the browser further.

## Related docs

Additional docs in this repo:

- `/home/chatsign/gs-sdf/docs/isaac-gaussian-navigation-mvp.md`
- `/home/chatsign/gs-sdf/docs/new-data-onboarding.md`
- `/home/chatsign/gs-sdf/docs/playback_1x_reexport_status.md`
- `/home/chatsign/gs-sdf/docs/gs-sdf-ui-architecture.md`
- `/home/chatsign/gs-sdf/docs/web-integration.md`

Those are still useful, but this README is the current entry point for actually running the system.

For the contractized runtime model specifically, also read:

- `/home/chatsign/gs-sdf/docs/slam-adapter-contract.md`
