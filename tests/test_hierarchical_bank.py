from __future__ import annotations

import torch

from hmr3d_memory.bank.hierarchical_bank import HierarchicalMemoryBank


def test_hierarchical_register_and_coarse():
    bank = HierarchicalMemoryBank(max_scenes=8)
    d0 = torch.nn.functional.normalize(torch.randn(32), dim=-1)
    d1 = torch.nn.functional.normalize(d0 + 0.01 * torch.randn_like(d0), dim=-1)
    bank.register_entry(0, d0)
    bank.register_entry(1, d1)
    ids = bank.coarse_archive_ids(d0, top_scenes=1)
    assert 0 in ids or 1 in ids
