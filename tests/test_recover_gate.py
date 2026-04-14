from __future__ import annotations

from types import SimpleNamespace

from hmr3d_memory.config import MemoryConfig
from hmr3d_memory.policies.recover_policy import RecoverPolicy


def _fake_proposal(alpha: float = 0.5):
    return SimpleNamespace(recovery_alpha=alpha)


def test_recover_gate_disabled():
    cfg = MemoryConfig(enable_v2_recover_gate=False)
    pol = RecoverPolicy(cfg)
    p = _fake_proposal(0.5)
    rd = pol.decide_after_verify(p, geo_gain=1.0, conf_delta=1.0, baseline_anchor_score=None, candidate_anchor_score=None)
    assert rd.allow and rd.effective_alpha == 0.5


def test_recover_gate_blocks_low_alpha():
    cfg = MemoryConfig(
        enable_v2_recover_gate=True,
        recover_max_injection_alpha=1.0,
        recover_min_effective_alpha=0.9,
        recover_geo_gain_soft_thresh=0.0,
        recover_alpha_scale_on_low_geo_gain=0.01,
    )
    pol = RecoverPolicy(cfg)
    p = _fake_proposal(0.5)
    rd = pol.decide_after_verify(p, geo_gain=-10.0, conf_delta=0.0, baseline_anchor_score=None, candidate_anchor_score=None)
    assert rd.allow is False
