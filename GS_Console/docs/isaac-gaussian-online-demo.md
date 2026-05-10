# Isaac Gaussian Online Demo

This is the first runnable split-process demo for:

```text
Isaac Lab
-> live pose + RGB + depth + semantic preview + accumulated point cloud
-> standalone HTTP bridge
-> online gaussian mapper
-> optional external Gaussian renderer
-> web UI Live / HiFi review
```

## Components

### 1. HTTP bridge

Script:

`scripts/isaac_gaussian_online_bridge_server.py`

Responsibilities:

- receives live state from Isaac
- stores robot pose and trajectory
- serves RGB / depth / semantic preview frames over HTTP
- optionally forwards camera poses to an external Gaussian renderer
- serves a unified MJPEG endpoint to the web UI

### 2. Isaac publisher

Script:

`scripts/run_isaac_gaussian_online_demo.py`

Responsibilities:

- launches an Isaac Lab scene
- loads a quadruped with an onboard camera
- runs a scripted motion path
- publishes robot pose, render pose, RGB, depth, semantic segmentation preview, and an accumulated world-frame point cloud to the bridge

### 3. Online Gaussian mapper

Script:

`scripts/isaac_online_gaussian_mapper_server.py`

Responsibilities:

- receives the same live Isaac state as the HTTP bridge
- converts the accumulated live point cloud into a lightweight SH0 `gs-ply`
- serves a dynamic gaussian manifest patch for the web UI `Gs` tab
- writes the current live gaussian artifact under `runtime/online-gaussian/isaac-online/gaussian/live.gs.ply`

### 4. Web UI scene entry

Scene manifest:

`examples/web-ui/public/scenes/isaac-gaussian-online/manifest.json`

Live contract:

`examples/web-ui/public/contracts/isaac-gaussian-online.live-contract.json`

## Run

### A. Start the online gaussian mapper

```bash
bash /home/chatsign/gs-sdf/scripts/launch_isaac_online_gaussian_mapper.sh 8891
```

### B. Start the bridge

Without an external Gaussian renderer:

```bash
MAPPER_URL=http://127.0.0.1:8891 \
bash /home/chatsign/gs-sdf/scripts/launch_isaac_gaussian_online_bridge.sh 8890
```

With an existing renderer bridge, for example the GS-SDF HiFi bridge on `8876`:

```bash
RENDERER_URL=http://127.0.0.1:8876 \
MAPPER_URL=http://127.0.0.1:8891 \
bash /home/chatsign/gs-sdf/scripts/launch_isaac_gaussian_online_bridge.sh 8890
```

### C. Start Isaac

```bash
source /media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh
export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY=:0
export XAUTHORITY=/var/run/lightdm/root/:0
export QT_X11_NO_MITSHM=1

"$ISAACSIM_PYTHON_EXE" /home/chatsign/gs-sdf/scripts/run_isaac_gaussian_online_demo.py \
  --device cuda:0 \
  --scene full-warehouse \
  --bridge-url http://127.0.0.1:8890 \
  --path-radius 4.5 \
  --path-speed 1.0
```

Other scenes:

- `--scene warehouse`
- `--scene office`
- `--scene hospital`
- `--scene empty`

If the chosen Isaac environment asset is unavailable, the script falls back to a ground plane.

### D. Start the web UI

```bash
WEB_PORT=5173 \
WEB_SCENE=/scenes/isaac-gaussian-online/manifest.json \
WEB_MODE=hifi \
WEB_ISAAC_GAUSSIAN_ONLINE_PORT=8890 \
WEB_ISAAC_GAUSSIAN_MAPPER_PORT=8891 \
bash /home/chatsign/gs-sdf/scripts/launch_web_ui_dev.sh
```

Open:

```text
http://localhost:5173/?scene=/scenes/isaac-gaussian-online/manifest.json&mode=hifi
```

Use:

- `Hifi` tab for the external Gaussian renderer stream, or Isaac RGB fallback if no renderer is configured
- `Live` tab for the accumulated live point cloud in the main 3D viewer
- `Gs` tab for the live `gs-ply` artifact synthesized by the online mapper

## Endpoints

The bridge exposes:

- `/status`
- `/robot_pose`
- `/trajectory.json`
- `/live_point_cloud.json`
- `/frame.jpg`
- `/frame.preview.jpg`
- `/frame.mjpeg`
- `/frame/rgb.jpg`
- `/frame/depth.jpg`
- `/frame/semantic.jpg`
- `/camera_pose`
- `/ingest/state`

The mapper exposes:

- `/status`
- `/gaussian_manifest.json`
- `/gaussian/live.gs.ply`
- `/frame/rgb.jpg`
- `/frame/depth.jpg`
- `/frame/semantic.jpg`

## Notes

- This first version uses HTTP, not ROS 2, for the Isaac-to-bridge loop.
- The web UI keeps its existing ROS path intact; only the Isaac online path uses the new HTTP bridge.
- If `RENDERER_URL` is configured, the bridge forwards the latest camera pose to that renderer and surfaces its frame as the main HiFi feed.
- If no external renderer is configured, the bridge falls back to Isaac RGB as the visible feed.
- The `Live` workspace now renders the bridge point cloud directly, so the left 3D panel should no longer stay on the empty grid once frames arrive.
- The current mapper is intentionally lightweight: it bootstraps a live Gaussian layer from the streamed point cloud. It is not yet a learned incremental Gaussian SLAM backend.
