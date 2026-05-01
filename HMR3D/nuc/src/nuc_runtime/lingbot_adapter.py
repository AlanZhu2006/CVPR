from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
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
        use_sdpa: bool = True,
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
        self._depth_head_backend = "torch"
        self._model_load_missing = 0
        self._model_load_unexpected = 0

        self._demo = _import_lingbot_demo()
        self._device = torch.device(
            "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = None

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

    def run_on_image_paths(self, image_paths: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if not image_paths:
            raise ValueError("LingBotReconstructor requires at least one image path")

        model = self._load_model()
        images = self._demo.load_and_preprocess_images(
            image_paths,
            mode="crop",
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        images = images.to(self.device)

        if self.device.type == "cuda":
            env_dtype = os.environ.get("LINGBOT_MODEL_DTYPE", "").strip().lower()
            if env_dtype in {"fp16", "float16", "half"}:
                dtype = torch.float16
            elif env_dtype in {"bf16", "bfloat16"}:
                dtype = torch.bfloat16
            else:
                dtype = (
                    torch.bfloat16
                    if torch.cuda.get_device_capability()[0] >= 8
                    else torch.float16
                )
        else:
            dtype = torch.float32

        if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
            model.aggregator = model.aggregator.to(dtype=dtype)

        output_device = torch.device("cpu") if self.offload_to_cpu else None
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

        predictions, _ = self._demo.postprocess(predictions, images)
        predictions_np = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions_np[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                predictions_np[key] = value

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
            "depth_head_backend": self._depth_head_backend,
            "depth_head_trt_engine": str(self.depth_head_trt_engine) if self.depth_head_trt_engine else "",
            "model_patch_embed": self.model_patch_embed,
            "model_embed_dim": self.model_embed_dim,
            "model_depth": self.model_depth,
            "model_num_heads": self.model_num_heads,
            "model_mlp_ratio": self.model_mlp_ratio,
            "model_load_missing_keys": self._model_load_missing,
            "model_load_unexpected_keys": self._model_load_unexpected,
            "prediction_keys": sorted(predictions_np.keys()),
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
            summary["metadata"] = metadata

        predictions_npz = output_dir / "lingbot_predictions.npz"
        if compress_outputs:
            np.savez_compressed(predictions_npz, **predictions_np)
        else:
            np.savez(predictions_npz, **predictions_np)
        summary_json = output_dir / "lingbot_summary.json"
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
