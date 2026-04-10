from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryConfig


def _rms_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)
    return value * denom


def _softplus_inverse(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return torch.log(torch.expm1(tensor)).item()


@dataclass
class FastWeightState:
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor


class FastWeightPoseMemory(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        cfg: MemoryConfig,
        base_query_proj: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_dim = output_dim // num_heads
        if self.head_dim * num_heads != output_dim:
            raise ValueError("Output dimension must be divisible by num_heads.")

        self.query_proj = nn.Linear(input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.lr_head = nn.Linear(input_dim, num_heads * 3)
        self.decay_head = nn.Linear(input_dim, num_heads)
        if base_query_proj is not None:
            self.query_proj.load_state_dict(base_query_proj.state_dict())
        nn.init.eye_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.lr_head.weight)
        nn.init.constant_(self.lr_head.bias, _softplus_inverse(cfg.fast_weight_lr))
        nn.init.zeros_(self.decay_head.weight)
        nn.init.zeros_(self.decay_head.bias)

        self.runtime_state: FastWeightState | None = None

    @classmethod
    def from_model(cls, model, cfg: MemoryConfig) -> "FastWeightPoseMemory":
        return cls(
            input_dim=model.enc_embed_dim,
            output_dim=model.dec_embed_dim,
            num_heads=model.dec_num_heads,
            cfg=cfg,
            base_query_proj=model.pose_retriever.proj_q,
        )

    def reset_runtime_state(self, device: torch.device) -> None:
        eye = torch.eye(self.head_dim, device=device, dtype=torch.float32)
        self.runtime_state = FastWeightState(
            w1=eye.unsqueeze(0).repeat(self.num_heads, 1, 1).clone(),
            w2=eye.unsqueeze(0).repeat(self.num_heads, 1, 1).clone(),
            w3=eye.unsqueeze(0).repeat(self.num_heads, 1, 1).clone(),
        )

    def _ensure_runtime_state(self, device: torch.device) -> FastWeightState:
        if self.runtime_state is None:
            self.reset_runtime_state(device)
        assert self.runtime_state is not None
        return self.runtime_state

    def _project_query(self, query: torch.Tensor) -> torch.Tensor:
        projected = self.query_proj(query)
        return projected.reshape(projected.shape[0], projected.shape[1], self.num_heads, self.head_dim)

    def _read_heads(self, projected_query: torch.Tensor, state: FastWeightState) -> torch.Tensor:
        hidden_a = torch.einsum("bthd,hdf->bthf", projected_query, state.w1)
        hidden_b = torch.einsum("bthd,hdf->bthf", projected_query, state.w2)
        out = torch.einsum("bthd,hdf->bthf", F.silu(hidden_a) * hidden_b, state.w3)
        return out.reshape(projected_query.shape[0], projected_query.shape[1], -1)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        state = self._ensure_runtime_state(query.device)
        projected_query = self._project_query(query)
        out = self._read_heads(projected_query, state)
        return self.output_proj(_rms_norm(out))

    def update(self, query: torch.Tensor, posterior_pose: torch.Tensor) -> None:
        state = self._ensure_runtime_state(query.device)
        context = query.mean(dim=1)
        with torch.enable_grad():
            projected_query = self._project_query(query).detach()
            tracked_state = FastWeightState(
                w1=state.w1.detach().clone().requires_grad_(True),
                w2=state.w2.detach().clone().requires_grad_(True),
                w3=state.w3.detach().clone().requires_grad_(True),
            )
            prior_pose = self.output_proj(_rms_norm(self._read_heads(projected_query, tracked_state)))
            target = F.normalize(posterior_pose.detach(), dim=-1)
            loss = (F.normalize(prior_pose, dim=-1) * target).sum()
            grads = torch.autograd.grad(loss, [tracked_state.w1, tracked_state.w2, tracked_state.w3])

        lr_values = F.softplus(self.lr_head(context)).mean(dim=0).reshape(self.num_heads, 3, 1, 1)
        alpha = 1.0 - self.cfg.fast_weight_decay_scale * torch.sigmoid(self.decay_head(context)).mean(dim=0)
        alpha = alpha.reshape(self.num_heads, 1, 1)
        self.runtime_state = FastWeightState(
            w1=(alpha * state.w1 + lr_values[:, 0] * grads[0]).detach(),
            w2=(alpha * state.w2 + lr_values[:, 1] * grads[1]).detach(),
            w3=(alpha * state.w3 + lr_values[:, 2] * grads[2]).detach(),
        )


class ChannelwiseStateGate(nn.Module):
    def __init__(self, state_dim: int, img_dim: int, cfg: MemoryConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim + img_dim, cfg.state_gate_hidden_dim)
        self.fc2 = nn.Linear(cfg.state_gate_hidden_dim, state_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, cfg.state_gate_init_bias)

    def forward(self, global_img_feat: torch.Tensor, state_feat: torch.Tensor) -> torch.Tensor:
        img_tokens = global_img_feat.expand(-1, state_feat.shape[1], -1)
        fused = torch.cat([state_feat, img_tokens], dim=-1)
        return torch.sigmoid(self.fc2(F.gelu(self.fc1(fused))))


class Mem3RLikeRuntime:
    def __init__(
        self,
        pose_memory: FastWeightPoseMemory,
        initial_pose_token: torch.Tensor,
        state_gate: ChannelwiseStateGate | None,
    ) -> None:
        self.pose_memory = pose_memory
        self.initial_pose_token = initial_pose_token.detach().clone()
        self.state_gate = state_gate

    @classmethod
    def from_model(cls, model, cfg: MemoryConfig, *, enable_state_gate: bool) -> "Mem3RLikeRuntime":
        state_gate = (
            ChannelwiseStateGate(state_dim=model.dec_embed_dim, img_dim=model.enc_embed_dim, cfg=cfg)
            if enable_state_gate
            else None
        )
        return cls(
            pose_memory=FastWeightPoseMemory.from_model(model, cfg),
            initial_pose_token=model.pose_token,
            state_gate=state_gate,
        )

    def to(self, device: torch.device | str) -> "Mem3RLikeRuntime":
        self.pose_memory.to(device)
        if self.state_gate is not None:
            self.state_gate.to(device)
        self.initial_pose_token = self.initial_pose_token.to(device)
        return self

    def reset(self, device: torch.device, batch_size: int) -> torch.Tensor:
        self.pose_memory.reset_runtime_state(device)
        return self.initial_pose_token.to(device).expand(batch_size, -1, -1)

    def read_pose(self, global_img_feat: torch.Tensor) -> torch.Tensor:
        return self.pose_memory.read(global_img_feat)

    def update_pose(self, global_img_feat: torch.Tensor, posterior_pose: torch.Tensor) -> None:
        self.pose_memory.update(global_img_feat, posterior_pose)

    def blend_state(
        self,
        global_img_feat: torch.Tensor,
        previous_state: torch.Tensor,
        candidate_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.state_gate is None:
            return candidate_state
        gate = self.state_gate(global_img_feat, previous_state)
        return gate * candidate_state + (1.0 - gate) * previous_state
