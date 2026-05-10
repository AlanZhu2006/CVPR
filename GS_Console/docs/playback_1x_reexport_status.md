# Playback 1x Re-export Status

Date: 2026-04-15

## Current status

- A new 1x playback compare video has been exported successfully for the `rgbclean` cache.
- Active cache:
  - [fast_livo2_compressed_rgbclean](/home/chatsign/gs-sdf/runtime/playback-cache/fast_livo2_compressed_rgbclean)
- Active sidecars:
  - [playback_video.json](/home/chatsign/gs-sdf/runtime/playback-cache/fast_livo2_compressed_rgbclean/playback_video.json)
  - [fast_livo2_compressed_rgbclean-4908c2faac1b.json](/home/chatsign/gs-sdf/runtime/playback-video-index/fast_livo2_compressed_rgbclean-4908c2faac1b.json)
- Active video:
  - [playback_rgb_vs_gssdf_quality_v2_1x.mp4](/home/chatsign/gs-sdf/runtime/videos/playback_rgb_vs_gssdf_quality_v2_1x.mp4)

## Final sidecar values

```json
{
  "path": "/media/chatsign/data-002/gs-sdf/runtime/videos/playback_rgb_vs_gssdf_quality_v2_1x.mp4",
  "fps": 12.0,
  "speed": 1.0,
  "startOffsetSec": 0.0,
  "endOffsetSec": -1.0,
  "label": "Playback RGB · GS-SDF"
}
```

## What was fixed

- [export_playback_rgb_vs_gssdf_video.py](/home/chatsign/gs-sdf/scripts/export_playback_rgb_vs_gssdf_video.py)
  - default `--speed` is `1.0`
  - duplicate source frames are preserved during 1x export
- [fastlivo_cached_playback_server.py](/home/chatsign/gs-sdf/scripts/fastlivo_cached_playback_server.py)
  - default `--video-speed` is now `1.0`
  - fallback video descriptor loading also defaults to `1.0`
- [launch_fastlivo_cached_playback.sh](/home/chatsign/gs-sdf/scripts/launch_fastlivo_cached_playback.sh)
  - `PLAYBACK_VIDEO_SPEED` default is now `1`

## Verification

- Exported video probe:

```text
width=2560
height=1024
r_frame_rate=12/1
avg_frame_rate=12/1
nb_frames=3190
duration=265.834000
```

- Cached playback status after restart:

```text
cacheDir=/media/chatsign/data-002/gs-sdf/runtime/playback-cache/fast_livo2_compressed_rgbclean
video.speed=1.0
rate=1.0
video.url=video.mp4
paused=true
```

## Re-run command

### 1. Start or verify the HiFi bridge

```bash
curl -fsS http://127.0.0.1:8876/status
```

Healthy state should show:

- `"ready": true`
- `"framesReceived"` increasing

If needed:

```bash
VIEW_WIDTH=1920 VIEW_HEIGHT=1536 MJPEG_MAX_FPS=45 KEEP_CONTAINER_ON_EXIT=1 \
bash /home/chatsign/gs-sdf/scripts/launch_gssdf_hifi_stack.sh \
  /home/chatsign/gs-sdf/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml \
  gssdf-view-quality \
  8876 \
  gssdf-hifi-bridge
```

### 2. Re-export the 1x compare video

```bash
python3 /home/chatsign/gs-sdf/scripts/export_playback_rgb_vs_gssdf_video.py \
  --cache-dir /home/chatsign/gs-sdf/runtime/playback-cache/fast_livo2_compressed_rgbclean \
  --scene-config /home/chatsign/gs-sdf/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml/model/config/scene/config.yaml \
  --bridge-url http://127.0.0.1:8876 \
  --output-video /home/chatsign/gs-sdf/runtime/videos/playback_rgb_vs_gssdf_quality_v2_1x.mp4 \
  --fps 12 \
  --speed 1.0 \
  --start-offset 0 \
  --end-offset -1
```

### 3. Restart cached playback so it reloads the sidecar

```bash
VIEW_WIDTH=1920 VIEW_HEIGHT=1536 MJPEG_MAX_FPS=90 START_PAUSED=1 \
bash /home/chatsign/gs-sdf/scripts/launch_fastlivo_cached_playback_stack.sh \
  /home/chatsign/gs-sdf/runtime/playback-cache/fast_livo2_compressed_rgbclean \
  fastlivo-cached-playback \
  9090 \
  5173 \
  8765
```

### 4. Verify playback status

```bash
curl -fsS http://127.0.0.1:8765/status
```

Expected:

- `"video"` is not `null`
- `"video.speed": 1.0`
- `"video.url": "video.mp4"`

## Summary

- The active `rgbclean` cache now uses a valid 1x compare video.
- Cached playback has been restarted and is serving `video.speed=1.0`.
- The old 8x sidecar is no longer used by the current `rgbclean` playback flow.
