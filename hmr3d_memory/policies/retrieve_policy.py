from __future__ import annotations

from typing import Any, List

import torch

from ..bank.hierarchical_bank import HierarchicalMemoryBank
from ..config import MemoryConfig


class RetrievePolicy:
    """Coarse-to-fine entry filtering for retrieval (HMR3D v2)."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config

    def filter_entries(
        self,
        query_descriptor: torch.Tensor,
        archive_bank: List[Any],
        hierarchical_bank: HierarchicalMemoryBank | None,
    ) -> tuple[List[Any], bool]:
        if not self.config.enable_v2_hierarchy or hierarchical_bank is None or not hierarchical_bank.scenes:
            return list(archive_bank), False
        allowed = hierarchical_bank.coarse_archive_ids(
            query_descriptor,
            top_scenes=self.config.hierarchy_top_scenes,
        )
        filtered = [e for e in archive_bank if e.archive_id in allowed]
        if not filtered:
            return list(archive_bank), True
        return filtered, False
