from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nuc_runtime.lingbot_adapter import LingBotReconstructor


@dataclass
class LingBotDepthWorkerConfig:
    model_path: str
    output_dir: str
    image_size: int = 336
    model_image_size: int = 518
    patch_size: int = 14
    window_size: int = 2
    stride: int = 1
    num_scale_frames: int = 2
    keyframe_interval: int = 1
    camera_num_iterations: int = 1
    max_queue: int = 4
    force_cpu: bool = False
    offload_to_cpu: bool = True
    use_sdpa: bool = False
    enable_camera: bool = False
    enable_depth: bool = True
    enable_point: bool = False
    enable_3d_rope: bool = False
    depth_head_trt_engine: str = ""
    model_patch_embed: str = ""
    model_embed_dim: int = 0
    model_depth: int = 0
    model_num_heads: int = 0
    model_mlp_ratio: float = 0.0
    compile_model: bool = False
    compile_warmup_passes: int = 3
    compile_warmup_stream_frames: int = 10
    persistent_streaming: bool = False
    compress_outputs: bool = True
    preload_model: bool = False
    warmup_first_window: bool = False


@dataclass
class LingBotFrameItem:
    image_path: str
    frame_idx: int | None = None
    timestamp_sec: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class LingBotWindowResult:
    index: int
    output_dir: str
    image_paths: list[str]
    elapsed_sec: float
    summary_json: str
    predictions_npz: str
    queued_monotonic_sec: float = 0.0
    started_monotonic_sec: float = 0.0
    finished_monotonic_sec: float = 0.0
    queue_wait_sec: float = 0.0
    end_to_end_sec: float = 0.0


class LingBotDepthWorker:
    """Persistent LingBot worker for slow background mapping.

    The model is loaded once in a worker thread. The foreground tracking loop can
    submit frames quickly; the worker consumes overlapping windows and writes one
    LingBot prediction bundle per window.
    """

    def __init__(self, config: LingBotDepthWorkerConfig):
        if config.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if config.stride < 1:
            raise ValueError("stride must be >= 1")
        if config.max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=config.max_queue)
        self._frames: deque[LingBotFrameItem] = deque(maxlen=config.window_size)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_requested = False
        self._submitted_frames = 0
        self._scheduled_windows = 0
        self._completed_windows = 0
        self._failed_windows = 0
        self._queue_full_drops = 0
        self._pending_window_drops = 0
        self._active_window_index: int | None = None
        self._preloading_model = bool(config.preload_model)
        self._dense_state = "IDLE"
        self._last_result: LingBotWindowResult | None = None
        self._last_error: str | None = None
        self._error_events: deque[dict[str, Any]] = deque(maxlen=16)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested = False
        self._thread = threading.Thread(target=self._run, name="LingBotDepthWorker", daemon=True)
        self._thread.start()
        self._write_status()

    def submit(
        self,
        image_path: str | Path,
        *,
        frame_idx: int | None = None,
        timestamp_sec: float | None = None,
        metadata: dict[str, Any] | None = None,
        block: bool = False,
    ) -> bool:
        if self._stop_requested:
            raise RuntimeError("Cannot submit after stop()")
        item = LingBotFrameItem(
            image_path=str(Path(image_path).expanduser().resolve()),
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            metadata=metadata,
        )
        self._frames.append(item)
        self._submitted_frames += 1
        if len(self._frames) < self.config.window_size:
            self._write_status()
            return False
        if (self._submitted_frames - self.config.window_size) % self.config.stride != 0:
            self._write_status()
            return False

        window_index = self._scheduled_windows
        payload = {
            "index": window_index,
            "frames": [asdict(frame) for frame in self._frames],
            "queued_monotonic_sec": time.perf_counter(),
        }
        if block:
            self._queue.put(payload, block=True)
        else:
            while True:
                try:
                    self._queue.put_nowait(payload)
                    break
                except queue.Full:
                    if not self._drop_oldest_pending_window():
                        with self._lock:
                            self._queue_full_drops += 1
                            self._last_error = "queue_full"
                        self._write_status()
                        return False
        self._scheduled_windows += 1
        self._write_status()
        return True

    def wait_until_idle(self, poll_sec: float = 0.25) -> None:
        self._queue.join()
        while True:
            with self._lock:
                done = self._completed_windows + self._failed_windows + self._pending_window_drops
                scheduled = self._scheduled_windows
            if done >= scheduled:
                return
            time.sleep(poll_sec)

    def stop(self, *, drain: bool = True) -> None:
        if drain:
            self.wait_until_idle()
        self._stop_requested = True
        if not drain:
            while self._drop_oldest_pending_window():
                pass
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=30)
        self._write_status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "config": asdict(self.config),
                "submitted_frames": self._submitted_frames,
                "scheduled_windows": self._scheduled_windows,
                "completed_windows": self._completed_windows,
                "failed_windows": self._failed_windows,
                "queue_full_drops": self._queue_full_drops,
                "pending_window_drops": self._pending_window_drops,
                "active_window_index": self._active_window_index,
                "preloading_model": self._preloading_model,
                "dense_state": self._dense_state,
                "worker_busy": self._preloading_model or self._active_window_index is not None,
                "queue_size": self._queue.qsize(),
                "stop_requested": self._stop_requested,
                "last_error": self._last_error,
                "error_events": list(self._error_events),
                "last_result": asdict(self._last_result) if self._last_result else None,
            }

    def _drop_oldest_pending_window(self) -> bool:
        try:
            dropped = self._queue.get_nowait()
        except queue.Empty:
            return False
        if dropped is None:
            self._queue.task_done()
            return False
        self._queue.task_done()
        with self._lock:
            self._queue_full_drops += 1
            self._pending_window_drops += 1
            self._last_error = "dropped_old_pending_window_for_latest"
        return True

    def _run(self) -> None:
        reconstructor = LingBotReconstructor(
            model_path=self.config.model_path,
            image_size=self.config.image_size,
            model_image_size=self.config.model_image_size,
            patch_size=self.config.patch_size,
            mode="streaming",
            num_scale_frames=self.config.num_scale_frames,
            keyframe_interval=self.config.keyframe_interval,
            camera_num_iterations=self.config.camera_num_iterations,
            offload_to_cpu=self.config.offload_to_cpu,
            use_sdpa=self.config.use_sdpa,
            force_cpu=self.config.force_cpu,
            enable_camera=self.config.enable_camera,
            enable_depth=self.config.enable_depth,
            enable_point=self.config.enable_point,
            enable_3d_rope=self.config.enable_3d_rope,
            depth_head_trt_engine=self.config.depth_head_trt_engine or None,
            model_patch_embed=self.config.model_patch_embed,
            model_embed_dim=self.config.model_embed_dim,
            model_depth=self.config.model_depth,
            model_num_heads=self.config.model_num_heads,
            model_mlp_ratio=self.config.model_mlp_ratio,
            compile_model=self.config.compile_model,
            compile_warmup_passes=self.config.compile_warmup_passes,
            compile_warmup_stream_frames=self.config.compile_warmup_stream_frames,
            persistent_streaming=self.config.persistent_streaming,
            dense_state_callback=self._set_dense_state,
        )
        if self.config.preload_model:
            with self._lock:
                self._preloading_model = True
                self._dense_state = "PREPROCESSING"
            self._write_status()
            try:
                reconstructor.preload()
            finally:
                with self._lock:
                    self._preloading_model = False
                    self._dense_state = "IDLE"
                self._write_status()
        while True:
            payload = self._queue.get()
            if payload is None:
                self._queue.task_done()
                break
            try:
                with self._lock:
                    self._active_window_index = int(payload.get("index", -1))
                    self._dense_state = "PREPROCESSING"
                self._write_status()
                self._process_window(reconstructor, payload)
            except BaseException as exc:
                window_index = int(payload.get("index", -1))
                message = f"{type(exc).__name__}: {exc}"
                trace = traceback.format_exc()
                print(
                    f"LingBot dense worker failed for window {window_index}: {message}\n{trace}",
                    flush=True,
                )
                with self._lock:
                    self._failed_windows += 1
                    self._last_error = message
                    self._error_events.append(
                        {
                            "time_sec": time.time(),
                            "window_index": window_index,
                            "error": message,
                            "traceback": trace,
                        }
                    )
                self._write_status()
            finally:
                with self._lock:
                    self._active_window_index = None
                    self._dense_state = "IDLE"
                self._queue.task_done()
                self._write_status()

    def _set_dense_state(self, state: str) -> None:
        with self._lock:
            self._dense_state = str(state)
        self._write_status()

    def _process_window(self, reconstructor: LingBotReconstructor, payload: dict[str, Any]) -> None:
        index = int(payload["index"])
        frames = payload["frames"]
        image_paths = [str(frame["image_path"]) for frame in frames]
        window_dir = self.output_dir / f"window_{index:06d}"
        frame_metadata = [frame.get("metadata") or {} for frame in frames]
        queued_monotonic = float(payload.get("queued_monotonic_sec") or 0.0)
        metadata = {
            "source": "lingbot_worker",
            "window_index": index,
            "frames": frames,
            "frame_indices": [frame.get("frame_idx") for frame in frames],
            "timestamps_sec": [frame.get("timestamp_sec") for frame in frames],
            "original_image_paths": image_paths,
        }
        if all("pose" in item for item in frame_metadata):
            metadata["cuvslam_poses"] = [item["pose"] for item in frame_metadata]
        if all("frame_shape" in item for item in frame_metadata):
            metadata["frame_shapes"] = [item["frame_shape"] for item in frame_metadata]
        for key in ("track_ok", "is_keyframe", "keypoint_count", "match_count", "inlier_count", "pixel_motion"):
            values = [item.get(key) for item in frame_metadata]
            if any(value is not None for value in values):
                metadata[key + "s"] = values
        if index == 0 and self.config.warmup_first_window:
            reconstructor.run_on_image_paths(image_paths)
        start = time.perf_counter()
        bundle = reconstructor.export_bundle(
            image_paths,
            window_dir,
            metadata=metadata,
            compress_outputs=self.config.compress_outputs,
        )
        finished = time.perf_counter()
        with self._lock:
            self._dense_state = "PUBLISHING"
        self._write_status()
        elapsed = finished - start
        result = LingBotWindowResult(
            index=index,
            output_dir=str(window_dir),
            image_paths=image_paths,
            elapsed_sec=elapsed,
            summary_json=str(bundle.summary_json),
            predictions_npz=str(bundle.predictions_npz),
            queued_monotonic_sec=queued_monotonic,
            started_monotonic_sec=start,
            finished_monotonic_sec=finished,
            queue_wait_sec=max(0.0, start - queued_monotonic) if queued_monotonic > 0 else 0.0,
            end_to_end_sec=max(0.0, finished - queued_monotonic) if queued_monotonic > 0 else elapsed,
        )
        result_path = window_dir / "worker_result.json"
        tmp_result_path = result_path.with_suffix(".json.tmp")
        tmp_result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        tmp_result_path.replace(result_path)
        with self._lock:
            self._completed_windows += 1
            self._last_result = result
            self._last_error = None
        self._write_status()

    def _write_status(self) -> None:
        status_path = self.output_dir / "worker_status.json"
        status_path.write_text(json.dumps(self.status(), indent=2), encoding="utf-8")
