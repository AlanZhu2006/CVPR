from __future__ import annotations

from typing import Tuple

import torch

from ..config import MemoryConfig


def apply_local_adaptation(
    cfg: MemoryConfig,
    state_feat: torch.Tensor,
    mem: torch.Tensor,
    target_state_feat: torch.Tensor,
    target_mem: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-step conservative blend toward a recovery target (TTT-style local adaptation)."""
    if not cfg.enable_v2_local_adapt:
        return state_feat, mem
    lr = float(cfg.local_adapt_lr)
    for _ in range(max(1, int(cfg.local_adapt_steps))):
        state_feat = (1.0 - lr) * state_feat + lr * target_state_feat
        mem = (1.0 - lr) * mem + lr * target_mem
    return state_feat, mem
