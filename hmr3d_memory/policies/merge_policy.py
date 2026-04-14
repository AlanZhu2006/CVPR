from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch

from ..config import MemoryConfig


@dataclass
class MergedRecoveryBundle:
    """Blended recurrent tensors after multi-candidate merge."""

    state_feat: torch.Tensor
    mem: torch.Tensor
    blend_weights: List[float]
    source_archive_ids: List[int]


class MergePolicy:
    """Confidence-weighted merge of multiple accepted recovery candidates (HMR3D v2)."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config

    def merge(
        self,
        proposals: List[Any],
        geo_rmses: List[float],
    ) -> MergedRecoveryBundle:
        if len(proposals) == 1:
            sf, _, _, mem, _ = proposals[0].state_args
            return MergedRecoveryBundle(
                state_feat=sf,
                mem=mem,
                blend_weights=[1.0],
                source_archive_ids=[proposals[0].archive_id],
            )
        tau = max(float(self.config.merge_softmax_temperature), 1e-6)
        scores = torch.tensor([-g / tau for g in geo_rmses], dtype=torch.float32)
        weights = torch.softmax(scores, dim=0).tolist()
        device = proposals[0].state_args[0].device
        dtype = proposals[0].state_args[0].dtype
        acc_sf = torch.zeros_like(proposals[0].state_args[0])
        acc_mem = torch.zeros_like(proposals[0].state_args[3])
        ids: List[int] = []
        for w, p in zip(weights, proposals):
            sf, _, _, mem, _ = p.state_args
            acc_sf = acc_sf + float(w) * sf
            acc_mem = acc_mem + float(w) * mem
            ids.append(p.archive_id)
        return MergedRecoveryBundle(
            state_feat=acc_sf.to(device=device, dtype=dtype),
            mem=acc_mem.to(device=device, dtype=dtype),
            blend_weights=[float(x) for x in weights],
            source_archive_ids=ids,
        )
