"""Light-weight MemoryRouter checks for HMR3D v2 (no einops / no dust3r)."""

from __future__ import annotations

import torch

from hmr3d_memory.config import MemoryConfig
from hmr3d_memory.router import ArchiveEntry, MemoryRouter, RecoveryProposal


def _fake_state_tuple(device="cpu"):
    b, n, c = 1, 4, 8
    z = torch.zeros(b, n, c, device=device)
    return (z, z, z, z, z)


def test_can_archive_respects_write_gate_novelty():
    cfg = MemoryConfig(
        enable_archive=True,
        enable_retrieval=False,
        archive_interval=2,
        enable_v2_write_gate=True,
        write_min_segment_novelty=0.02,
    )
    r = MemoryRouter(cfg)
    d1 = torch.nn.functional.normalize(torch.randn(1, 1, 8), dim=-1)
    d2 = torch.nn.functional.normalize(torch.randn(1, 1, 8), dim=-1)
    r.observe(d1.cpu())
    r.observe(d2.cpu())
    assert r.can_archive(1, _fake_state_tuple(), optional_state_conf=0.0) is True
    r.archive(1, _fake_state_tuple(), camera_pose=None)
    r.observe(d1.cpu())
    r.observe(d2.cpu())
    assert r.can_archive(3, _fake_state_tuple(), optional_state_conf=0.0) is False
    assert int(r.stats["write_gate_rejects"]) >= 1


def test_hierarchical_coarse_then_fallback_smoke():
    cfg = MemoryConfig(
        enable_archive=True,
        enable_retrieval=True,
        archive_interval=5,
        min_frames_before_retrieve=0,
        retrieval_cooldown=0,
        retrieval_attempt_cooldown=0,
        retrieval_similarity_thresh=0.0,
        verification_similarity_thresh=0.0,
        max_state_similarity_for_recover=2.0,
        sequence_similarity_thresh=0.0,
        query_state_gap_thresh=-1.0,
        enable_v2_hierarchy=True,
        hierarchy_top_scenes=1,
        hierarchy_max_scenes=4,
    )
    r = MemoryRouter(cfg)
    for t in range(5):
        r.observe(torch.nn.functional.normalize(torch.randn(1, 1, 8), dim=-1))
    r.archive(4, _fake_state_tuple(), camera_pose=None)
    for t in range(5, 10):
        r.observe(torch.nn.functional.normalize(torch.randn(1, 1, 8), dim=-1))
    r.archive(9, _fake_state_tuple(), camera_pose=None)
    st = _fake_state_tuple()
    q = torch.nn.functional.normalize(torch.randn(1, 1, 8), dim=-1)
    props = r.propose_recovery(20, st, q)
    assert isinstance(props, list)


def test_rebuild_proposal_with_alpha():
    cfg = MemoryConfig(enable_archive=True, enable_retrieval=False, recovery_alpha=0.5)
    r = MemoryRouter(cfg)
    for _ in range(3):
        r.observe(torch.nn.functional.normalize(torch.randn(1, 1, 8), dim=-1))
    r.archive(2, _fake_state_tuple(), camera_pose=None)
    entry = r.archive_bank[0]
    sf = torch.randn_like(entry.state_feat)
    mem = torch.randn_like(entry.mem)
    prop = RecoveryProposal(
        archive_id=entry.archive_id,
        archive_frame_idx=entry.frame_idx,
        candidate_rank=1,
        query_similarity=1.0,
        state_similarity=0.5,
        sequence_similarity=0.0,
        query_state_gap=0.1,
        recovery_alpha=0.5,
        is_latest_archive=True,
        archive_camera_pose=None,
        state_args=(0.5 * entry.state_feat + 0.5 * sf, entry.state_pos, entry.init_state_feat, 0.5 * entry.mem + 0.5 * mem, entry.init_mem),
    )
    rebuilt = r.rebuild_proposal_with_alpha(prop, sf, mem, 0.2)
    assert abs(rebuilt.recovery_alpha - 0.2) < 1e-5
