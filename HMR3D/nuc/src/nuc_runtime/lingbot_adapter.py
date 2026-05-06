from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from nuc_runtime.cuvslam_adapter import CUVSLAMOfflineKITTIAdapter
from nuc_runtime.descriptors import compute_global_descriptor
from nuc_runtime.lingbot_trt_depth_head import TensorRTDepthHead
from nuc_runtime.models import TrackingOutput


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def _import_lingbot_demo():
    import importlib.util

    candidates = []
    env_root = os.environ.get("LINGBOT_MAP_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path(__file__).resolve().parents[4] / "third_party_research" / "lingbot-map",
            Path("/home/nyu/Codespace/CVPR/third_party_research/lingbot-map"),
        ]
    )
    lingbot_root = next((path.resolve() for path in candidates if (path / "demo.py").exists()), None)
    if lingbot_root is None:
        searched = ", ".join(str(path) for path in candidates)
        raise ImportError(
            "Failed to locate LingBot demo.py. Set LINGBOT_MAP_ROOT to the "
            f"lingbot-map checkout. Searched: {searched}"
        )
    if str(lingbot_root) not in sys.path:
        sys.path.insert(0, str(lingbot_root))
    demo_path = lingbot_root / "demo.py"
    spec = importlib.util.spec_from_file_location("lingbot_demo_local", demo_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load LingBot demo module from {demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_postprocess_without_camera(
    demo_module,
    predictions: dict[str, Any],
    images: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Move depth-only/point-only LingBot outputs to CPU without pose decoding."""
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)
    squeeze_single_batch = getattr(demo_module, "_squeeze_single_batch", lambda _key, value: value)
    for key in list(predictions.keys()):
        value = predictions[key]
        if isinstance(value, torch.Tensor):
            predictions[key] = squeeze_single_batch(key, value.to("cpu", non_blocking=True))
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return predictions, images_cpu


def _prediction_frame_count(predictions: dict[str, np.ndarray], fallback: int) -> int:
    for key in ("depth", "depth_conf", "world_points", "world_points_conf", "pose_enc"):
        value = predictions.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            return int(value.shape[0])
    return int(fallback)


def _trim_metadata_to_frame_count(metadata: dict[str, Any], frame_count: int) -> dict[str, Any]:
    frame_count = max(1, int(frame_count))
    trimmed = dict(metadata)
    for key in (
        "frames",
        "frame_indices",
        "timestamps_sec",
        "original_image_paths",
        "cuvslam_poses",
        "frame_shapes",
        "track_oks",
        "is_keyframes",
        "keypoint_counts",
        "match_counts",
        "inlier_counts",
        "pixel_motions",
    ):
        value = trimmed.get(key)
        if isinstance(value, list) and len(value) > frame_count:
            trimmed[key] = value[-frame_count:]
    return trimmed


@dataclass
class LingBotReconBundle:
    image_paths: list[str]
    predictions_npz: Path
    summary_json: Path
    summary: dict[str, Any]


class LingBotReconstructor:
    def __init__(
        self,
        model_path: str | Path,
        image_size: int = 518,
        model_image_size: int | None = None,
        patch_size: int = 14,
        mode: str = "streaming",
        num_scale_frames: int = 8,
        keyframe_interval: int = 1,
        camera_num_iterations: int = 1,
        use_sdpa: bool = False,
        offload_to_cpu: bool = True,
        force_cpu: bool = True,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_point: bool = True,
        enable_3d_rope: bool = True,
        depth_head_trt_engine: str | Path | None = None,
        model_patch_embed: str = "",
        model_embed_dim: int = 0,
        model_depth: int = 0,
        model_num_heads: int = 0,
        model_mlp_ratio: float = 0.0,
        compile_model: bool = False,
        compile_warmup_passes: int = 3,
        compile_warmup_stream_frames: int = 10,
        persistent_streaming: bool = False,
        dense_state_callback: Any | None = None,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.image_size = image_size
        self.model_image_size = model_image_size or image_size
        self.patch_size = patch_size
        self.mode = mode
        self.num_scale_frames = num_scale_frames
        self.keyframe_interval = keyframe_interval
        self.camera_num_iterations = camera_num_iterations
        self.use_sdpa = use_sdpa
        self.offload_to_cpu = offload_to_cpu
        self.force_cpu = force_cpu
        self.enable_camera = enable_camera
        self.enable_depth = enable_depth
        self.enable_point = enable_point
        self.enable_3d_rope = enable_3d_rope
        self.depth_head_trt_engine = (
            Path(depth_head_trt_engine).expanduser().resolve()
            if depth_head_trt_engine
            else None
        )
        self.model_patch_embed = model_patch_embed
        self.model_embed_dim = model_embed_dim
        self.model_depth = model_depth
        self.model_num_heads = model_num_heads
        self.model_mlp_ratio = model_mlp_ratio
        self.compile_model_enabled = bool(compile_model)
        self.compile_warmup_passes = max(1, int(compile_warmup_passes))
        self.compile_warmup_stream_frames = max(1, int(compile_warmup_stream_frames))
        self.persistent_streaming = bool(persistent_streaming)
        self.dense_state_callback = dense_state_callback
        self._depth_head_backend = "torch"
        self._model_load_missing = 0
        self._model_load_unexpected = 0
        self._aggregator_dtype: str | None = None
        self._compiled = False
        self._stream_initialized = False
        self._stream_seen_frames = 0

        self._demo = _import_lingbot_demo()
        self._device = torch.device(
            "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = None

    def _set_dense_state(self, state: str) -> None:
        if self.dense_state_callback is None:
            return
        try:
            self.dense_state_callback(str(state))
        except Exception:
            pass

    @property
    def device(self) -> torch.device:
        return self._device

    def preload(self) -> None:
        self._load_model()

    def _build_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            mode=self.mode,
            image_size=self.model_image_size,
            patch_size=self.patch_size,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=1024,
            kv_cache_sliding_window=64,
            num_scale_frames=self.num_scale_frames,
            camera_num_iterations=self.camera_num_iterations,
            use_sdpa=self.use_sdpa,
            model_path=str(self.model_path),
            enable_camera=self.enable_camera,
            enable_depth=self.enable_depth,
            enable_point=self.enable_point,
        )

    def _use_direct_model(self) -> bool:
        return any(
            [
                bool(self.model_patch_embed),
                self.model_embed_dim > 0,
                self.model_depth > 0,
                self.model_num_heads > 0,
                self.model_mlp_ratio > 0,
            ]
        )

    def _load_direct_model(self):
        from lingbot_map.models.gct_stream import GCTStream

        patch_embed = self.model_patch_embed or "dinov2_vitl14_reg"
        embed_dim = self.model_embed_dim or 1024
        depth = self.model_depth or 24
        num_heads = self.model_num_heads or 16
        mlp_ratio = self.model_mlp_ratio or 4.0
        model = GCTStream(
            img_size=self.model_image_size,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_embed=patch_embed,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=1024,
            kv_cache_sliding_window=64,
            kv_cache_scale_frames=self.num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self.use_sdpa,
            camera_num_iterations=self.camera_num_iterations,
            enable_camera=self.enable_camera,
            enable_depth=self.enable_depth,
            enable_point=self.enable_point,
        )
        state_dict = _load_state(self.model_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self._model_load_missing = len(missing)
        self._model_load_unexpected = len(unexpected)
        return model.to(self.device).eval()

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._use_direct_model():
            self._model = self._load_direct_model()
        else:
            args = self._build_args()
            self._model = self._demo.load_model(args, self.device)
        if self.depth_head_trt_engine is not None:
            if self.device.type != "cuda":
                raise RuntimeError("TensorRT depth head requires CUDA LingBotReconstructor")
            self._model.depth_head = TensorRTDepthHead(self.depth_head_trt_engine)
            self._depth_head_backend = "tensorrt"
        return self._model

    def _maybe_compile_model(
        self,
        model: Any,
        images: torch.Tensor,
        dtype: torch.dtype,
        profile: dict[str, float],
    ) -> None:
        if self._compiled or not self.compile_model_enabled:
            return
        if self.device.type != "cuda" or self.mode != "streaming":
            profile["compile_skipped_sec"] = 0.0
            return
        if self.use_sdpa:
            raise RuntimeError("LingBot compile fast path expects FlashInfer; disable USE_SDPA.")
        if not hasattr(self._demo, "compile_model") or not hasattr(self._demo, "_warm_streaming"):
            raise RuntimeError("LingBot demo module does not expose compile_model/_warm_streaming.")

        num_frames = int(images.shape[0])
        scale_for_warm = max(1, min(int(self.num_scale_frames), num_frames))
        warm_stream_n = max(1, int(self.compile_warmup_stream_frames))
        warm_images = images
        required_frames = scale_for_warm + warm_stream_n
        if num_frames < required_frames:
            repeat_count = required_frames - num_frames
            pad = images[-1:].repeat(repeat_count, 1, 1, 1)
            warm_images = torch.cat([images, pad], dim=0)
        else:
            warm_stream_n = min(warm_stream_n, max(1, num_frames - scale_for_warm))

        print(
            "LingBot torch.compile warmup: "
            f"window_frames={num_frames} warm_frames={int(warm_images.shape[0])} "
            f"scale_frames={scale_for_warm} stream_frames={warm_stream_n} "
            f"passes={self.compile_warmup_passes}",
            flush=True,
        )

        eager_start = time.perf_counter()
        self._demo._warm_streaming(
            model,
            warm_images,
            scale_for_warm,
            warm_stream_n,
            dtype,
            passes=1,
            keyframe_interval=self.keyframe_interval,
        )
        profile["compile_eager_warmup_sec"] = time.perf_counter() - eager_start
        print(
            f"LingBot torch.compile eager warmup done: {profile['compile_eager_warmup_sec']:.3f}s",
            flush=True,
        )

        compile_start = time.perf_counter()
        self._demo.compile_model(model)
        _sync_if_cuda(self.device)
        profile["compile_model_sec"] = time.perf_counter() - compile_start
        print(
            f"LingBot torch.compile graph wrapping done: {profile['compile_model_sec']:.3f}s",
            flush=True,
        )

        compiled_start = time.perf_counter()
        self._demo._warm_streaming(
            model,
            warm_images,
            scale_for_warm,
            warm_stream_n,
            dtype,
            passes=self.compile_warmup_passes,
            keyframe_interval=self.keyframe_interval,
        )
        profile["compile_warmup_sec"] = time.perf_counter() - compiled_start
        self._compiled = True
        print(
            f"LingBot torch.compile compiled warmup done: {profile['compile_warmup_sec']:.3f}s",
            flush=True,
        )

    def _run_persistent_streaming(
        self,
        model: Any,
        images: torch.Tensor,
        dtype: torch.dtype,
        output_device: torch.device | None,
        profile: dict[str, float],
    ) -> dict[str, Any]:
        if self.mode != "streaming":
            raise RuntimeError("persistent_streaming only supports streaming mode")
        if images.ndim != 4:
            raise ValueError(f"Expected preprocessed images [S,C,H,W], got shape={tuple(images.shape)}")
        num_frames = int(images.shape[0])
        scale_frames = max(1, int(self.num_scale_frames))
        if not self._stream_initialized and num_frames < scale_frames:
            raise ValueError(
                f"persistent_streaming needs at least {scale_frames} scale frames for first call, "
                f"got {num_frames}"
            )

        def _to_out(t: torch.Tensor) -> torch.Tensor:
            if output_device is not None:
                return t.to(output_device)
            return t

        predictions: dict[str, Any] = {}
        model_device = next(model.parameters()).device
        with torch.no_grad():
            if self.device.type == "cuda":
                autocast = torch.amp.autocast("cuda", dtype=dtype)
            else:
                autocast = torch.amp.autocast("cpu", enabled=False)
            with autocast:
                if not self._stream_initialized:
                    clean_start = time.perf_counter()
                    model.clean_kv_cache()
                    _sync_if_cuda(self.device)
                    profile["stream_clean_cache_sec"] = time.perf_counter() - clean_start

                    scale_images = images[:scale_frames].unsqueeze(0).to(model_device, non_blocking=True)
                    torch.compiler.cudagraph_mark_step_begin()
                    forward_start = time.perf_counter()
                    output = model.forward(
                        scale_images,
                        num_frame_for_scale=scale_frames,
                        num_frame_per_block=scale_frames,
                        causal_inference=True,
                    )
                    _sync_if_cuda(self.device)
                    profile["stream_scale_forward_sec"] = time.perf_counter() - forward_start
                    self._stream_initialized = True
                    self._stream_seen_frames = scale_frames
                    profile["persistent_stream_stage"] = "scale"
                else:
                    frame_image = images[-1:].unsqueeze(0).to(model_device, non_blocking=True)
                    is_keyframe = (
                        self.keyframe_interval <= 1
                        or ((self._stream_seen_frames - scale_frames) % self.keyframe_interval == 0)
                    )
                    profile["persistent_stream_stage"] = "incremental"
                    profile["persistent_stream_keyframe"] = 1.0 if is_keyframe else 0.0
                    if not is_keyframe:
                        model._set_skip_append(True)
                    torch.compiler.cudagraph_mark_step_begin()
                    forward_start = time.perf_counter()
                    output = model.forward(
                        frame_image,
                        num_frame_for_scale=scale_frames,
                        num_frame_per_block=1,
                        causal_inference=True,
                    )
                    _sync_if_cuda(self.device)
                    profile["stream_incremental_forward_sec"] = time.perf_counter() - forward_start
                    if not is_keyframe:
                        model._set_skip_append(False)
                    self._stream_seen_frames += 1

        for key in ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"):
            if key in output:
                predictions[key] = _to_out(output[key])
        del output
        profile["persistent_stream_seen_frames"] = float(self._stream_seen_frames)
        if hasattr(model, "get_kv_cache_info"):
            try:
                info = model.get_kv_cache_info()
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        profile[f"kv_cache_{key}"] = float(value)
            except Exception:
                pass
        return predictions

    def run_on_image_paths(self, image_paths: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if not image_paths:
            raise ValueError("LingBotReconstructor requires at least one image path")

        profile: dict[str, float] = {}
        total_start = time.perf_counter()

        self._set_dense_state("PREPROCESSING")
        load_start = time.perf_counter()
        model = self._load_model()
        _sync_if_cuda(self.device)
        profile["model_load_sec"] = time.perf_counter() - load_start

        preprocess_start = time.perf_counter()
        images = self._demo.load_and_preprocess_images(
            image_paths,
            mode="crop",
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        profile["preprocess_sec"] = time.perf_counter() - preprocess_start

        device_transfer_start = time.perf_counter()
        images = images.to(self.device)
        _sync_if_cuda(self.device)
        profile["image_to_device_sec"] = time.perf_counter() - device_transfer_start

        dtype_start = time.perf_counter()
        if self.device.type == "cuda":
            env_dtype = os.environ.get("LINGBOT_MODEL_DTYPE", "").strip().lower()
            if env_dtype in {"fp16", "float16", "half"}:
                dtype = torch.float16
            elif env_dtype in {"bf16", "bfloat16"}:
                dtype = torch.bfloat16
            elif env_dtype in {"fp32", "float32"}:
                dtype = torch.float32
            else:
                dtype = (
                    torch.bfloat16
                    if torch.cuda.get_device_capability()[0] >= 8
                    else torch.float16
                )
        else:
            dtype = torch.float32

        dtype_key = str(dtype)
        if (
            dtype != torch.float32
            and getattr(model, "aggregator", None) is not None
            and self._aggregator_dtype != dtype_key
        ):
            model.aggregator = model.aggregator.to(dtype=dtype)
            self._aggregator_dtype = dtype_key
        _sync_if_cuda(self.device)
        profile["dtype_setup_sec"] = time.perf_counter() - dtype_start

        self._maybe_compile_model(model, images, dtype, profile)

        output_device = torch.device("cpu") if self.offload_to_cpu else None
        _sync_if_cuda(self.device)
        self._set_dense_state("MODEL_FORWARD_ACTIVE")
        forward_start = time.perf_counter()
        if self.persistent_streaming:
            predictions = self._run_persistent_streaming(model, images, dtype, output_device, profile)
        else:
            with torch.no_grad():
                if self.device.type == "cuda":
                    autocast = torch.amp.autocast("cuda", dtype=dtype)
                else:
                    autocast = torch.amp.autocast("cpu", enabled=False)
                with autocast:
                    if self.mode == "streaming":
                        predictions = model.inference_streaming(
                            images,
                            num_scale_frames=self.num_scale_frames,
                            keyframe_interval=self.keyframe_interval,
                            output_device=output_device,
                        )
                    else:
                        predictions = model.inference_windowed(
                            images,
                            window_size=max(2, len(image_paths)),
                            overlap_size=0,
                            output_device=output_device,
                        )
        _sync_if_cuda(self.device)
        profile["model_forward_sec"] = time.perf_counter() - forward_start

        self._set_dense_state("POSTPROCESSING")
        postprocess_start = time.perf_counter()
        if "pose_enc" in predictions:
            predictions, _ = self._demo.postprocess(predictions, images)
        else:
            predictions, _ = _cpu_postprocess_without_camera(self._demo, predictions, images)
        _sync_if_cuda(self.device)
        profile["postprocess_sec"] = time.perf_counter() - postprocess_start

        numpy_start = time.perf_counter()
        predictions_np = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions_np[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                predictions_np[key] = value
        _sync_if_cuda(self.device)
        profile["tensor_to_numpy_sec"] = time.perf_counter() - numpy_start
        profile["total_sec"] = time.perf_counter() - total_start

        summary = {
            "image_paths": image_paths,
            "frame_count": len(image_paths),
            "device": self.device.type,
            "mode": self.mode,
            "image_size": self.image_size,
            "model_image_size": self.model_image_size,
            "num_scale_frames": self.num_scale_frames,
            "keyframe_interval": self.keyframe_interval,
            "camera_num_iterations": self.camera_num_iterations,
            "enable_camera": self.enable_camera,
            "enable_depth": self.enable_depth,
            "enable_point": self.enable_point,
            "enable_3d_rope": self.enable_3d_rope,
            "use_sdpa": self.use_sdpa,
            "depth_head_backend": self._depth_head_backend,
            "depth_head_trt_engine": str(self.depth_head_trt_engine) if self.depth_head_trt_engine else "",
            "model_patch_embed": self.model_patch_embed,
            "model_embed_dim": self.model_embed_dim,
            "model_depth": self.model_depth,
            "model_num_heads": self.model_num_heads,
            "model_mlp_ratio": self.model_mlp_ratio,
            "compile_model": self.compile_model_enabled,
            "compile_warmup_passes": self.compile_warmup_passes,
            "compile_warmup_stream_frames": self.compile_warmup_stream_frames,
            "compiled": self._compiled,
            "persistent_streaming": self.persistent_streaming,
            "persistent_stream_initialized": self._stream_initialized,
            "persistent_stream_seen_frames": self._stream_seen_frames,
            "model_load_missing_keys": self._model_load_missing,
            "model_load_unexpected_keys": self._model_load_unexpected,
            "prediction_keys": sorted(predictions_np.keys()),
            "profile_sec": profile,
        }
        if self.device.type == "cuda":
            summary["cuda_memory"] = {
                "allocated_gb": float(torch.cuda.memory_allocated(self.device) / (1024**3)),
                "reserved_gb": float(torch.cuda.memory_reserved(self.device) / (1024**3)),
                "max_allocated_gb": float(torch.cuda.max_memory_allocated(self.device) / (1024**3)),
                "max_reserved_gb": float(torch.cuda.max_memory_reserved(self.device) / (1024**3)),
            }
        if "depth" in predictions_np:
            summary["depth_shape"] = list(predictions_np["depth"].shape)
        if "world_points" in predictions_np:
            summary["world_points_shape"] = list(predictions_np["world_points"].shape)
        if "extrinsic" in predictions_np:
            summary["extrinsic_shape"] = list(predictions_np["extrinsic"].shape)
        return predictions_np, summary

    def export_bundle(
        self,
        image_paths: list[str],
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
        compress_outputs: bool = True,
    ) -> LingBotReconBundle:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_np, summary = self.run_on_image_paths(image_paths)
        if metadata:
            frame_count = _prediction_frame_count(predictions_np, fallback=len(image_paths))
            if frame_count != len(image_paths):
                metadata = _trim_metadata_to_frame_count(metadata, frame_count)
                image_paths = list(image_paths[-frame_count:])
                summary["image_paths"] = image_paths
                summary["frame_count"] = frame_count
            summary["metadata"] = metadata

        predictions_npz = output_dir / "lingbot_predictions.npz"
        save_npz_start = time.perf_counter()
        if compress_outputs:
            np.savez_compressed(predictions_npz, **predictions_np)
        else:
            np.savez(predictions_npz, **predictions_np)
        summary.setdefault("profile_sec", {})["save_npz_sec"] = time.perf_counter() - save_npz_start
        summary_json = output_dir / "lingbot_summary.json"
        save_summary_start = time.perf_counter()
        summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["profile_sec"]["save_summary_json_sec"] = time.perf_counter() - save_summary_start
        summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return LingBotReconBundle(
            image_paths=image_paths,
            predictions_npz=predictions_npz,
            summary_json=summary_json,
            summary=summary,
        )


class CUVSLAMLingBotReconAdapter:
    def __init__(
        self,
        sequence_path: str | Path,
        trajectory_path: str | Path,
        tracking_config,
        model_path: str | Path,
        frame_step: int = 1,
        max_frames: int = 0,
        lingbot_window_keyframes: int = 2,
        lingbot_force_cpu: bool = True,
    ):
        self.source = CUVSLAMOfflineKITTIAdapter(
            sequence_path=sequence_path,
            trajectory_path=trajectory_path,
            config=tracking_config,
            frame_step=frame_step,
            max_frames=max_frames,
        )
        self.reconstructor = LingBotReconstructor(
            model_path=model_path,
            force_cpu=lingbot_force_cpu,
            camera_num_iterations=1,
            keyframe_interval=1,
        )
        self.window_keyframes = max(1, lingbot_window_keyframes)

    def collect_first_keyframe_window(self) -> list[TrackingOutput]:
        keyframes: list[TrackingOutput] = []
        for output in self.source:
            if output.is_keyframe:
                keyframes.append(output)
            if len(keyframes) >= self.window_keyframes:
                break
        return keyframes

    def export_first_window(self, output_dir: str | Path) -> LingBotReconBundle:
        keyframes = self.collect_first_keyframe_window()
        if not keyframes:
            raise RuntimeError("No keyframes collected from cuVSLAM adapter")
        image_paths = [item.image_path for item in keyframes if item.image_path]
        if len(image_paths) != len(keyframes):
            raise RuntimeError("Some keyframes are missing image paths")

        metadata = {
            "source": "cuvslam_plus_lingbot_window",
            "frame_indices": [item.frame_idx for item in keyframes],
            "timestamps_sec": [item.timestamp_sec for item in keyframes],
            "cuvslam_poses": [item.pose.tolist() for item in keyframes],
            "cuvslam_descriptors": [item.descriptor.tolist() for item in keyframes],
            "descriptors_mean": np.mean(
                np.vstack([item.descriptor for item in keyframes]), axis=0
            ).tolist(),
        }
        return self.reconstructor.export_bundle(
            image_paths=image_paths,
            output_dir=output_dir,
            metadata=metadata,
        )


def build_lingbot_window_descriptor(bundle: LingBotReconBundle) -> np.ndarray:
    predictions = np.load(bundle.predictions_npz)
    depth = predictions.get("depth")
    depth_conf = predictions.get("depth_conf")
    world_points_conf = predictions.get("world_points_conf")
    stats = []
    if depth is not None:
        stats.append(np.asarray([float(depth.mean()), float(depth.std())], dtype=np.float32))
    if depth_conf is not None:
        stats.append(
            np.asarray([float(depth_conf.mean()), float(depth_conf.std())], dtype=np.float32)
        )
    if world_points_conf is not None:
        stats.append(
            np.asarray(
                [float(world_points_conf.mean()), float(world_points_conf.std())],
                dtype=np.float32,
            )
        )

    frame_descs = []
    for path_str in bundle.image_paths:
        frame_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        frame_descs.append(compute_global_descriptor(frame_bgr, None))

    if frame_descs:
        stats.append(np.mean(np.vstack(frame_descs), axis=0).astype(np.float32))
    if not stats:
        return np.zeros(8, dtype=np.float32)

    descriptor = np.concatenate(stats, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    if norm > 0:
        descriptor /= norm
    return descriptor
