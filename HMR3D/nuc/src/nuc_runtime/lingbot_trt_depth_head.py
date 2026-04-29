from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn


def _import_tensorrt():
    for path in (
        "/usr/lib/python3.10/dist-packages",
        "/usr/local/lib/python3.10/dist-packages",
        "/usr/lib/python3/dist-packages",
    ):
        if path not in sys.path and Path(path).exists():
            sys.path.append(path)
    import tensorrt as trt

    return trt


class TensorRTDepthHead(nn.Module):
    """LingBot DPT depth-head replacement backed by a fixed-shape TensorRT engine."""

    def __init__(self, engine_path: str | Path):
        super().__init__()
        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Missing TensorRT depth-head engine: {self.engine_path}")
        self.trt = _import_tensorrt()
        logger = self.trt.Logger(self.trt.Logger.WARNING)
        runtime = self.trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, torch.dtype] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            dtype = torch.float16 if trt_dtype == self.trt.float16 else torch.float32
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        expected_inputs = ["feat0", "feat1", "feat2", "feat3"]
        if self.input_names != expected_inputs:
            raise RuntimeError(
                f"Unexpected TensorRT depth-head inputs {self.input_names}; expected {expected_inputs}"
            )
        if "depth" not in self.output_names or "depth_conf" not in self.output_names:
            raise RuntimeError(f"Unexpected TensorRT depth-head outputs {self.output_names}")

    @property
    def fixed_image_shape(self) -> tuple[int, int] | None:
        shape = self.shapes.get("depth")
        if shape is None or len(shape) < 4:
            return None
        return int(shape[2]), int(shape[3])

    def _prepare_feature(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        expected = self.shapes[name]
        if tuple(tensor.shape) != expected:
            raise ValueError(f"TensorRT depth-head input {name} expected {expected}, got {tuple(tensor.shape)}")
        if not tensor.is_cuda:
            raise ValueError("TensorRT depth-head inputs must be CUDA tensors")
        return tensor.to(dtype=self.dtypes[name]).contiguous()

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del images, patch_start_idx
        if len(aggregated_tokens_list) != 4:
            raise ValueError(f"TensorRT depth-head expects 4 feature maps, got {len(aggregated_tokens_list)}")
        bindings: dict[str, torch.Tensor] = {}
        for name, tensor in zip(self.input_names, aggregated_tokens_list):
            bindings[name] = self._prepare_feature(name, tensor)
        for name in self.output_names:
            bindings[name] = torch.empty(
                self.shapes[name],
                device="cuda",
                dtype=self.dtypes[name],
            ).contiguous()
        for name, tensor in bindings.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        stream = torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("TensorRT depth-head execute_async_v3 failed")
        return bindings["depth"], bindings["depth_conf"]
