from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from einops import rearrange

from .config import MemoryConfig
from .mem3r_probe import Mem3RLikeRuntime
from .router import MemoryRouter, RecoveryProposal
from .ttt3r_env import bootstrap_ttt3r_imports
from .ttt3r_io import compute_anchor_pose_quality, compute_prediction_quality, extract_camera_pose_matrix


def _prepare_features(model, view: Dict[str, torch.Tensor], device: str):
    bootstrap_ttt3r_imports()
    from dust3r.utils.device import to_gpu

    gpu_view = to_gpu(view, device)
    batch_size = gpu_view["img"].shape[0]
    img_mask = gpu_view["img_mask"].reshape(-1, batch_size)
    ray_mask = gpu_view["ray_mask"].reshape(-1, batch_size)
    imgs = gpu_view["img"].unsqueeze(0).view(-1, *gpu_view["img"].shape[1:])
    ray_maps = gpu_view["ray_map"].unsqueeze(0).view(-1, *gpu_view["ray_map"].shape[1:])
    shapes = view["true_shape"].unsqueeze(0).view(-1, 2).to(imgs.device)

    selected_imgs = imgs[img_mask.view(-1)]
    selected_shapes = shapes[img_mask.view(-1)]
    if selected_imgs.size(0) > 0:
        img_out, img_pos, _ = model._encode_image(selected_imgs, selected_shapes)
    else:
        img_out, img_pos = None, None

    ray_maps = ray_maps.permute(0, 3, 1, 2)
    selected_ray_maps = ray_maps[ray_mask.view(-1)]
    selected_shapes_ray = shapes[ray_mask.view(-1)]
    if selected_ray_maps.size(0) > 0:
        ray_out, ray_pos, _ = model._encode_ray_map(selected_ray_maps, selected_shapes_ray)
    else:
        ray_out, ray_pos = None, None

    if img_out is not None and ray_out is None:
        feat_i = img_out[-1]
        pos_i = img_pos
    elif img_out is None and ray_out is not None:
        feat_i = ray_out[-1]
        pos_i = ray_pos
    elif img_out is not None and ray_out is not None:
        feat_i = img_out[-1] + ray_out[-1]
        pos_i = img_pos
    else:
        raise RuntimeError("Unable to build features for the current view.")

    return gpu_view, shapes, feat_i, pos_i


def _compute_update_masks(
    model,
    gpu_view: Dict[str, torch.Tensor],
    cross_attn_state: List[torch.Tensor],
    frame_idx: int,
    reset_mask: torch.Tensor | bool,
    base_update_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    img_mask = gpu_view["img_mask"]
    update = gpu_view.get("update", None)
    if update is not None:
        update_mask = img_mask & update
    else:
        update_mask = img_mask
    update_mask = update_mask[:, None, None].float()

    if frame_idx == 0 or bool(torch.as_tensor(reset_mask).any().item()):
        update_mask1 = update_mask
    elif base_update_type == "cut3r":
        update_mask1 = update_mask
    elif base_update_type == "ttt3r":
        cross_attn_state = rearrange(
            torch.cat(cross_attn_state, dim=0),
            "l h nstate nimg -> 1 nstate nimg (l h)",
        )
        state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
        update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None]
    else:
        raise ValueError(f"Invalid base update type: {base_update_type}")

    update_mask2 = update_mask
    return update_mask1, update_mask2


@dataclass
class StepResult:
    result: Dict[str, torch.Tensor]
    next_state_feat: torch.Tensor
    next_mem: torch.Tensor


@dataclass
class PendingShadowRecovery:
    proposal: RecoveryProposal
    shadow_state_feat: torch.Tensor
    shadow_mem: torch.Tensor
    baseline_geo_rmse_sum: float
    shadow_geo_rmse_sum: float
    baseline_conf_sum: float
    shadow_conf_sum: float
    comparison_frames: int
    future_frames_remaining: int

    @property
    def cumulative_geo_gain(self) -> float:
        return self.baseline_geo_rmse_sum - self.shadow_geo_rmse_sum

    @property
    def cumulative_conf_delta(self) -> float:
        return self.shadow_conf_sum - self.baseline_conf_sum


def _should_accept_verified_candidate(
    *,
    baseline_quality: Dict[str, float],
    candidate_quality: Dict[str, float],
    cfg: MemoryConfig,
    required_geo_gain: float | None = None,
) -> bool:
    geo_gain = baseline_quality["geo_rmse"] - candidate_quality["geo_rmse"]
    geo_ok = candidate_quality["geo_rmse"] <= baseline_quality["geo_rmse"] * cfg.verification_geo_ratio_thresh
    conf_ok = candidate_quality["mean_log_conf"] + cfg.verification_conf_tolerance >= baseline_quality["mean_log_conf"]
    if required_geo_gain is None:
        required_geo_gain = cfg.verification_min_geo_gain
    gain_ok = geo_gain >= required_geo_gain
    return geo_ok and conf_ok and gain_ok


def _required_geo_gain_for_proposal(
    proposal,
    cfg: MemoryConfig,
) -> float:
    required_geo_gain = cfg.verification_min_geo_gain
    is_ambiguous = (
        proposal.candidate_rank >= cfg.verification_ambiguity_rank_threshold
        or abs(proposal.query_state_gap) < cfg.verification_ambiguity_gap_thresh
    )
    if (not proposal.is_latest_archive) and is_ambiguous:
        required_geo_gain = max(required_geo_gain, cfg.verification_stale_min_geo_gain)
    return required_geo_gain


def _should_commit_shadow_recovery(
    pending_shadow: PendingShadowRecovery,
    cfg: MemoryConfig,
) -> bool:
    geo_ok = pending_shadow.cumulative_geo_gain >= cfg.shadow_recovery_min_cumulative_geo_gain
    conf_ok = (
        pending_shadow.cumulative_conf_delta + cfg.verification_conf_tolerance * pending_shadow.comparison_frames
        >= 0.0
    )
    return geo_ok and conf_ok


def _should_accept_anchor_pose_candidate(
    *,
    proposal: RecoveryProposal,
    baseline_anchor_quality: Dict[str, float] | None,
    candidate_anchor_quality: Dict[str, float] | None,
    cfg: MemoryConfig,
) -> bool:
    if not cfg.enable_anchor_pose_verification:
        return True
    if cfg.anchor_pose_only_for_ambiguous:
        is_ambiguous = (
            proposal.candidate_rank >= cfg.verification_ambiguity_rank_threshold
            or abs(proposal.query_state_gap) < cfg.verification_ambiguity_gap_thresh
            or not proposal.is_latest_archive
        )
        if not is_ambiguous:
            return True
    if baseline_anchor_quality is None or candidate_anchor_quality is None:
        return False
    baseline_score = baseline_anchor_quality["anchor_score"]
    candidate_score = candidate_anchor_quality["anchor_score"]
    ratio_ok = candidate_score <= baseline_score * cfg.anchor_pose_score_ratio_thresh
    gain_ok = (baseline_score - candidate_score) >= cfg.anchor_pose_min_score_gain
    return ratio_ok and gain_ok


def _run_step(
    *,
    model,
    gpu_view: Dict[str, torch.Tensor],
    shape: torch.Tensor,
    feat_i: torch.Tensor,
    pos_i: torch.Tensor,
    state_feat: torch.Tensor,
    state_pos: torch.Tensor,
    init_state_feat: torch.Tensor,
    mem: torch.Tensor,
    init_mem: torch.Tensor,
    reset_mask: torch.Tensor | bool,
    cfg: MemoryConfig,
    mem3r_runtime: Mem3RLikeRuntime | None,
    frame_idx: int,
) -> StepResult:
    if model.pose_head_flag:
        global_img_feat_i = model._get_img_level_feat(feat_i)
    else:
        global_img_feat_i = feat_i.mean(dim=1, keepdim=True)

    if model.pose_head_flag:
        if frame_idx == 0 or bool(torch.as_tensor(reset_mask).any().item()):
            if mem3r_runtime is not None:
                pose_feat_i = mem3r_runtime.reset(feat_i.device, feat_i.shape[0])
            else:
                pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1)
        elif mem3r_runtime is not None:
            pose_feat_i = mem3r_runtime.read_pose(global_img_feat_i)
        else:
            pose_feat_i = model.pose_retriever.inquire(global_img_feat_i, mem)
        pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
    else:
        pose_feat_i = None
        pose_pos_i = None

    new_state_feat, dec, _, cross_attn_state, _, _ = model._recurrent_rollout(
        state_feat,
        state_pos,
        feat_i,
        pos_i,
        pose_feat_i,
        pose_pos_i,
        init_state_feat,
        img_mask=gpu_view["img_mask"],
        reset_mask=gpu_view["reset"],
        update=gpu_view.get("update", None),
        return_attn=True,
    )
    out_pose_feat_i = dec[-1][:, 0:1]
    if mem3r_runtime is not None:
        mem3r_runtime.update_pose(global_img_feat_i, out_pose_feat_i)
        new_mem = mem
    else:
        new_mem = model.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)

    head_input = [
        dec[0].float(),
        dec[model.dec_depth * 2 // 4][:, 1:].float(),
        dec[model.dec_depth * 3 // 4][:, 1:].float(),
        dec[model.dec_depth].float(),
    ]
    result = model._downstream_head(head_input, shape, pos=pos_i)

    update_mask1, update_mask2 = _compute_update_masks(
        model=model,
        gpu_view=gpu_view,
        cross_attn_state=cross_attn_state,
        frame_idx=frame_idx,
        reset_mask=reset_mask,
        base_update_type=cfg.base_update_type,
    )
    if mem3r_runtime is not None:
        new_state_feat = mem3r_runtime.blend_state(global_img_feat_i, state_feat, new_state_feat)
    next_state_feat = new_state_feat * update_mask1 + state_feat * (1.0 - update_mask1)
    next_mem = new_mem * update_mask2 + mem * (1.0 - update_mask2)

    step_reset_mask = gpu_view["reset"]
    if step_reset_mask is not None:
        step_reset_mask = step_reset_mask[:, None, None].float()
        next_state_feat = init_state_feat * step_reset_mask + next_state_feat * (1.0 - step_reset_mask)
        next_mem = init_mem * step_reset_mask + next_mem * (1.0 - step_reset_mask)
    return StepResult(result=result, next_state_feat=next_state_feat, next_mem=next_mem)


@torch.no_grad()
def run_sequence_with_mode(
    views: List[Dict[str, torch.Tensor]],
    model,
    device: str,
    mode: str,
    memory_config: MemoryConfig,
):
    bootstrap_ttt3r_imports()
    from dust3r.utils.device import to_cpu

    cfg = memory_config.for_mode(mode)
    router = MemoryRouter(cfg) if (cfg.enable_archive or cfg.enable_retrieval) else None
    use_mem3r_pose_probe = mode == "mem3r_pose_probe"
    use_mem3r_like_runtime = mode == "mem3r_like_runtime"
    mem3r_runtime = (
        Mem3RLikeRuntime.from_model(
            model,
            cfg,
            enable_state_gate=use_mem3r_like_runtime,
        )
        if (use_mem3r_pose_probe or use_mem3r_like_runtime)
        else None
    )
    if mem3r_runtime is not None:
        mem3r_runtime.to(device)
    outputs: List[Dict[str, torch.Tensor]] = []
    reset_mask = False
    state_feat = None
    state_pos = None
    init_state_feat = None
    mem = None
    init_mem = None
    pending_shadow: PendingShadowRecovery | None = None

    for frame_idx, raw_view in enumerate(views):
        gpu_view, shape, feat_i, pos_i = _prepare_features(model, raw_view, device)

        if state_feat is None:
            state_feat, state_pos = model._init_state(feat_i, pos_i)
            mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            init_state_feat = state_feat.clone()
            init_mem = mem.clone()

        if model.pose_head_flag:
            global_img_feat_i = model._get_img_level_feat(feat_i)
        else:
            global_img_feat_i = feat_i.mean(dim=1, keepdim=True)

        if pending_shadow is not None:
            if router is not None:
                query_descriptor = router.describe_global_feat(global_img_feat_i)
                router.observe(query_descriptor)
            baseline_step = _run_step(
                model=model,
                gpu_view=gpu_view,
                shape=shape,
                feat_i=feat_i,
                pos_i=pos_i,
                state_feat=state_feat,
                state_pos=state_pos,
                init_state_feat=init_state_feat,
                mem=mem,
                init_mem=init_mem,
                reset_mask=reset_mask,
                cfg=cfg,
                mem3r_runtime=mem3r_runtime,
                frame_idx=frame_idx,
            )
            shadow_step = _run_step(
                model=model,
                gpu_view=gpu_view,
                shape=shape,
                feat_i=feat_i,
                pos_i=pos_i,
                state_feat=pending_shadow.shadow_state_feat,
                state_pos=state_pos,
                init_state_feat=init_state_feat,
                mem=pending_shadow.shadow_mem,
                init_mem=init_mem,
                reset_mask=reset_mask,
                cfg=cfg,
                mem3r_runtime=None,
                frame_idx=frame_idx,
            )
            baseline_quality = compute_prediction_quality(baseline_step.result)
            shadow_quality = compute_prediction_quality(shadow_step.result)
            shadow_frame_ok = _should_accept_verified_candidate(
                baseline_quality=baseline_quality,
                candidate_quality=shadow_quality,
                cfg=cfg,
                required_geo_gain=0.0,
            )
            pending_shadow.baseline_geo_rmse_sum += baseline_quality["geo_rmse"]
            pending_shadow.shadow_geo_rmse_sum += shadow_quality["geo_rmse"]
            pending_shadow.baseline_conf_sum += baseline_quality["mean_log_conf"]
            pending_shadow.shadow_conf_sum += shadow_quality["mean_log_conf"]
            pending_shadow.comparison_frames += 1
            pending_shadow.future_frames_remaining -= 1

            if router is not None:
                router.update_shadow_recovery(
                    frame_idx=frame_idx,
                    proposal=pending_shadow.proposal,
                    comparison_frames=pending_shadow.comparison_frames,
                    baseline_geo_rmse=baseline_quality["geo_rmse"],
                    baseline_conf=baseline_quality["mean_log_conf"],
                    candidate_geo_rmse=shadow_quality["geo_rmse"],
                    candidate_conf=shadow_quality["mean_log_conf"],
                    cumulative_geo_gain=pending_shadow.cumulative_geo_gain,
                    cumulative_conf_delta=pending_shadow.cumulative_conf_delta,
                )

            chosen_step = baseline_step
            state_feat = baseline_step.next_state_feat
            mem = baseline_step.next_mem
            if cfg.shadow_recovery_require_consistent_frames and not shadow_frame_ok:
                if router is not None:
                    router.reject_shadow_recovery(
                        frame_idx,
                        pending_shadow.proposal,
                        comparison_frames=pending_shadow.comparison_frames,
                        cumulative_geo_gain=pending_shadow.cumulative_geo_gain,
                        cumulative_conf_delta=pending_shadow.cumulative_conf_delta,
                        baseline_geo_rmse=baseline_quality["geo_rmse"],
                        baseline_conf=baseline_quality["mean_log_conf"],
                        candidate_geo_rmse=shadow_quality["geo_rmse"],
                        candidate_conf=shadow_quality["mean_log_conf"],
                        reason="shadow_recovery_inconsistent_frame",
                    )
                pending_shadow = None
            else:
                pending_shadow.shadow_state_feat = shadow_step.next_state_feat
                pending_shadow.shadow_mem = shadow_step.next_mem

            if pending_shadow is not None and pending_shadow.future_frames_remaining <= 0:
                if router is not None and _should_commit_shadow_recovery(pending_shadow, cfg):
                    chosen_step = shadow_step
                    state_feat = shadow_step.next_state_feat
                    mem = shadow_step.next_mem
                    router.commit_shadow_recovery(
                        frame_idx,
                        pending_shadow.proposal,
                        comparison_frames=pending_shadow.comparison_frames,
                        cumulative_geo_gain=pending_shadow.cumulative_geo_gain,
                        cumulative_conf_delta=pending_shadow.cumulative_conf_delta,
                        baseline_geo_rmse=baseline_quality["geo_rmse"],
                        baseline_conf=baseline_quality["mean_log_conf"],
                        candidate_geo_rmse=shadow_quality["geo_rmse"],
                        candidate_conf=shadow_quality["mean_log_conf"],
                    )
                elif router is not None:
                    router.reject_shadow_recovery(
                        frame_idx,
                        pending_shadow.proposal,
                        comparison_frames=pending_shadow.comparison_frames,
                        cumulative_geo_gain=pending_shadow.cumulative_geo_gain,
                        cumulative_conf_delta=pending_shadow.cumulative_conf_delta,
                        baseline_geo_rmse=baseline_quality["geo_rmse"],
                        baseline_conf=baseline_quality["mean_log_conf"],
                        candidate_geo_rmse=shadow_quality["geo_rmse"],
                        candidate_conf=shadow_quality["mean_log_conf"],
                    )
                pending_shadow = None

            outputs.append(to_cpu(chosen_step.result))
            reset_mask = gpu_view["reset"]
            if router is not None and pending_shadow is None and router.should_archive(frame_idx):
                router.archive(
                    frame_idx,
                    (state_feat, state_pos, init_state_feat, mem, init_mem),
                    camera_pose=extract_camera_pose_matrix(chosen_step.result),
                )
                if cfg.reset_on_archive:
                    state_feat = init_state_feat.clone()
                    mem = init_mem.clone()
                    reset_mask = True
            continue

        if router is not None:
            query_descriptor = router.describe_global_feat(global_img_feat_i)
            router.observe(query_descriptor)
            state_args = (state_feat, state_pos, init_state_feat, mem, init_mem)
            proposals = router.propose_recovery(frame_idx, state_args, query_descriptor)
        else:
            proposals = []

        baseline_step = _run_step(
            model=model,
            gpu_view=gpu_view,
            shape=shape,
            feat_i=feat_i,
            pos_i=pos_i,
            state_feat=state_feat,
            state_pos=state_pos,
            init_state_feat=init_state_feat,
            mem=mem,
            init_mem=init_mem,
            reset_mask=reset_mask,
            cfg=cfg,
            mem3r_runtime=mem3r_runtime,
            frame_idx=frame_idx,
        )

        chosen_step = baseline_step
        if proposals:
            if not cfg.enable_geometric_verification:
                chosen_proposal = proposals[0]
                proposal_state_feat, proposal_state_pos, proposal_init_state_feat, proposal_mem, proposal_init_mem = (
                    chosen_proposal.state_args
                )
                chosen_step = _run_step(
                    model=model,
                    gpu_view=gpu_view,
                    shape=shape,
                    feat_i=feat_i,
                    pos_i=pos_i,
                    state_feat=proposal_state_feat,
                    state_pos=proposal_state_pos,
                    init_state_feat=proposal_init_state_feat,
                    mem=proposal_mem,
                    init_mem=proposal_init_mem,
                    reset_mask=reset_mask,
                    cfg=cfg,
                    mem3r_runtime=None,
                    frame_idx=frame_idx,
                )
                router.accept_recovery_without_geometry(frame_idx, chosen_proposal)
            else:
                baseline_quality = compute_prediction_quality(baseline_step.result)
                accepted_proposal = None
                accepted_step = None
                accepted_quality = None
                accepted_baseline_anchor_quality = None
                accepted_anchor_quality = None
                for proposal in proposals:
                    required_geo_gain = _required_geo_gain_for_proposal(proposal, cfg)
                    proposal_state_feat, proposal_state_pos, proposal_init_state_feat, proposal_mem, proposal_init_mem = (
                        proposal.state_args
                    )
                    candidate_step = _run_step(
                        model=model,
                        gpu_view=gpu_view,
                        shape=shape,
                        feat_i=feat_i,
                        pos_i=pos_i,
                        state_feat=proposal_state_feat,
                        state_pos=proposal_state_pos,
                        init_state_feat=proposal_init_state_feat,
                        mem=proposal_mem,
                        init_mem=proposal_init_mem,
                        reset_mask=reset_mask,
                        cfg=cfg,
                        mem3r_runtime=None,
                        frame_idx=frame_idx,
                    )
                    candidate_quality = compute_prediction_quality(candidate_step.result)
                    baseline_anchor_quality = compute_anchor_pose_quality(
                        baseline_step.result,
                        proposal.archive_camera_pose,
                        rotation_weight=cfg.anchor_pose_rotation_weight,
                    )
                    candidate_anchor_quality = compute_anchor_pose_quality(
                        candidate_step.result,
                        proposal.archive_camera_pose,
                        rotation_weight=cfg.anchor_pose_rotation_weight,
                    )
                    if _should_accept_verified_candidate(
                        baseline_quality=baseline_quality,
                        candidate_quality=candidate_quality,
                        cfg=cfg,
                        required_geo_gain=required_geo_gain,
                    ) and _should_accept_anchor_pose_candidate(
                        proposal=proposal,
                        baseline_anchor_quality=baseline_anchor_quality,
                        candidate_anchor_quality=candidate_anchor_quality,
                        cfg=cfg,
                    ):
                        if accepted_quality is None or candidate_quality["geo_rmse"] < accepted_quality["geo_rmse"]:
                            accepted_proposal = proposal
                            accepted_step = candidate_step
                            accepted_quality = candidate_quality
                            accepted_baseline_anchor_quality = baseline_anchor_quality
                            accepted_anchor_quality = candidate_anchor_quality
                    else:
                        router.reject_recovery(
                            frame_idx,
                            proposal,
                            baseline_geo_rmse=baseline_quality["geo_rmse"],
                            baseline_conf=baseline_quality["mean_log_conf"],
                            candidate_geo_rmse=candidate_quality["geo_rmse"],
                            candidate_conf=candidate_quality["mean_log_conf"],
                            baseline_anchor_score=(
                                baseline_anchor_quality["anchor_score"] if baseline_anchor_quality is not None else None
                            ),
                            candidate_anchor_score=(
                                candidate_anchor_quality["anchor_score"]
                                if candidate_anchor_quality is not None
                                else None
                            ),
                            reason="rejected_by_geometry_or_anchor_verification",
                        )

                if accepted_proposal is not None and accepted_step is not None and accepted_quality is not None:
                    if cfg.enable_shadow_recovery and cfg.shadow_recovery_window > 0:
                        router.record_geometry_verification(
                            baseline_geo_rmse=baseline_quality["geo_rmse"],
                            baseline_conf=baseline_quality["mean_log_conf"],
                            candidate_geo_rmse=accepted_quality["geo_rmse"],
                            candidate_conf=accepted_quality["mean_log_conf"],
                            accepted=True,
                        )
                        if accepted_anchor_quality is not None:
                            router.record_anchor_pose_verification(
                                baseline_anchor_score=accepted_baseline_anchor_quality["anchor_score"],
                                candidate_anchor_score=accepted_anchor_quality["anchor_score"],
                                accepted=True,
                            )
                        pending_shadow = PendingShadowRecovery(
                            proposal=accepted_proposal,
                            shadow_state_feat=accepted_step.next_state_feat,
                            shadow_mem=accepted_step.next_mem,
                            baseline_geo_rmse_sum=baseline_quality["geo_rmse"],
                            shadow_geo_rmse_sum=accepted_quality["geo_rmse"],
                            baseline_conf_sum=baseline_quality["mean_log_conf"],
                            shadow_conf_sum=accepted_quality["mean_log_conf"],
                            comparison_frames=1,
                            future_frames_remaining=cfg.shadow_recovery_window,
                        )
                        router.start_shadow_recovery(
                            frame_idx,
                            accepted_proposal,
                            baseline_geo_rmse=baseline_quality["geo_rmse"],
                            baseline_conf=baseline_quality["mean_log_conf"],
                            candidate_geo_rmse=accepted_quality["geo_rmse"],
                            candidate_conf=accepted_quality["mean_log_conf"],
                            shadow_window=cfg.shadow_recovery_window,
                        )
                    else:
                        chosen_step = accepted_step
                        router.accept_recovery(
                            frame_idx,
                            accepted_proposal,
                            baseline_geo_rmse=baseline_quality["geo_rmse"],
                            baseline_conf=baseline_quality["mean_log_conf"],
                            candidate_geo_rmse=accepted_quality["geo_rmse"],
                            candidate_conf=accepted_quality["mean_log_conf"],
                            baseline_anchor_score=(
                                accepted_baseline_anchor_quality["anchor_score"]
                                if accepted_baseline_anchor_quality is not None
                                else None
                            ),
                            candidate_anchor_score=(
                                accepted_anchor_quality["anchor_score"]
                                if accepted_anchor_quality is not None
                                else None
                            ),
                        )
                else:
                    best_proposal = proposals[0] if proposals else None
                    if best_proposal is not None:
                        baseline_anchor_quality = compute_anchor_pose_quality(
                            baseline_step.result,
                            best_proposal.archive_camera_pose,
                            rotation_weight=cfg.anchor_pose_rotation_weight,
                        )
                    else:
                        baseline_anchor_quality = None
                    router.reject_recovery(
                        frame_idx,
                        best_proposal,
                        baseline_geo_rmse=baseline_quality["geo_rmse"],
                        baseline_conf=baseline_quality["mean_log_conf"],
                        baseline_anchor_score=(
                            baseline_anchor_quality["anchor_score"] if baseline_anchor_quality is not None else None
                        ),
                        reason="rejected_by_geometry_or_anchor_verification",
                    )

        outputs.append(to_cpu(chosen_step.result))
        state_feat = chosen_step.next_state_feat
        mem = chosen_step.next_mem

        reset_mask = gpu_view["reset"]

        if router is not None and pending_shadow is None and router.should_archive(frame_idx):
            router.archive(
                frame_idx,
                (state_feat, state_pos, init_state_feat, mem, init_mem),
                camera_pose=extract_camera_pose_matrix(chosen_step.result),
            )
            if cfg.reset_on_archive:
                state_feat = init_state_feat.clone()
                mem = init_mem.clone()
                reset_mask = True

    memory_stats = router.finalize() if router is not None else {
        "archive_count": 0,
        "retrieval_attempts": 0,
        "retrieval_successes": 0,
        "retrieval_redundant_skips": 0,
        "retrieval_threshold_rejects": 0,
        "retrieval_gap_rejects": 0,
        "retrieval_sequence_rejects": 0,
        "archive_bank_size": 0,
        "avg_best_similarity": 0.0,
        "avg_best_gap": 0.0,
        "avg_best_sequence_similarity": 0.0,
        "avg_verified_similarity": 0.0,
        "avg_verified_gap": 0.0,
        "avg_verified_sequence_similarity": 0.0,
        "geometry_verification_rollouts": 0,
        "geometry_verification_accepts": 0,
        "geometry_verification_rejects": 0,
        "avg_geometry_verification_geo_gain": 0.0,
        "avg_geometry_verification_conf_delta": 0.0,
        "anchor_pose_verification_accepts": 0,
        "anchor_pose_verification_rejects": 0,
        "avg_anchor_pose_score_gain": 0.0,
        "shadow_recovery_starts": 0,
        "shadow_recovery_commits": 0,
        "shadow_recovery_rejects": 0,
        "shadow_recovery_frames": 0,
        "avg_shadow_recovery_geo_gain": 0.0,
        "avg_shadow_recovery_conf_delta": 0.0,
        "avg_shadow_recovery_frames": 0.0,
        "evicted_archives": 0,
    }
    if router is not None:
        memory_stats["events"] = router.events
    return {"views": views, "pred": outputs}, memory_stats
