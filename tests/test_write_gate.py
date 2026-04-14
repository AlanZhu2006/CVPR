from __future__ import annotations

import torch

from hmr3d_memory.config import MemoryConfig
from hmr3d_memory.policies.write_policy import WritePolicy


def test_write_gate_disabled_always_accepts():
    cfg = MemoryConfig(enable_v2_write_gate=False)
    pol = WritePolicy(cfg)
    d = torch.randn(1, 8)
    dec = pol.decide(
        frame_idx=10,
        segment_descriptors=[d],
        last_archived_segment_desc=d,
        optional_state_conf=-100.0,
    )
    assert dec.accept is True


def test_write_gate_novelty_rejects():
    cfg = MemoryConfig(enable_v2_write_gate=True, write_min_segment_novelty=0.01)
    pol = WritePolicy(cfg)
    d = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    dec = pol.decide(
        frame_idx=10,
        segment_descriptors=[d],
        last_archived_segment_desc=d.clone(),
        optional_state_conf=None,
    )
    assert dec.accept is False
    assert "novelty" in dec.reason


def test_write_gate_conf_delay():
    d = torch.randn(1, 8)
    cfg2 = MemoryConfig(
        enable_v2_write_gate=True,
        write_min_state_confidence=10.0,
        write_delay_frames_on_low_conf=3,
    )
    pol2 = WritePolicy(cfg2)
    dec = pol2.decide(
        frame_idx=10,
        segment_descriptors=[d],
        last_archived_segment_desc=None,
        optional_state_conf=0.0,
    )
    assert dec.delay_frames == 3
