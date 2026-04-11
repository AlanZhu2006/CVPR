# HMR3D Retrieval Review

Current `HMR3D` is an inference-time memory router, so the optimization target is not new training first. The more realistic near-term gain is better recover verification.

## Relevant References

- ORB-SLAM3: https://arxiv.org/abs/2007.11898
  - Retrieval or loop hypotheses should not be trusted directly. They need geometric validation before graph integration.
- HF-Net: https://openaccess.thecvf.com/content_CVPR_2019/papers/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.pdf
  - Strong pattern: global retrieval for shortlist, local verification for final pose.
- NetVLAD: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf
  - Good reminder that coarse global descriptors are useful for proposal generation, not necessarily for final acceptance.
- Patch-NetVLAD: https://arxiv.org/abs/2103.01486
  - Re-ranking with finer local evidence improves place-recognition decisions after global shortlist.

## HMR3D Mapping

- `query_descriptor` retrieval already plays the coarse retrieval role.
- Sequence consistency is a light re-ranking signal.
- The missing piece was fine verification.

## Implemented Direction

The current optimization adds counterfactual geometric verification:

1. Retrieve archive candidates with descriptor thresholds.
2. Run a counterfactual rollout on the current frame using each candidate recovered state.
3. Score each candidate using self-consistency from:
   - `pts3d_in_self_view`
   - `pts3d_in_other_view`
   - confidence maps
4. Accept recovery only if the candidate improves weighted geometric consistency without a large confidence drop.

This keeps the system training-free while making the recover path closer to the coarse-to-fine verification pattern used in localization literature.

## Current Extension: Shadow Recover

The next step is a short shadow branch after geometric verification:

1. Candidate passes the current-frame counterfactual geometric check.
2. Candidate state is not committed immediately.
3. Run the candidate state in parallel for a small future window.
4. Commit only if the short-window evidence remains favorable.

This follows the same spirit as robust loop closure: hypothesis generation, local verification, then cautious integration. Current experiments show that shadow recovery can repair some failure windows, but the verification signal is still not strong enough to make it the default path across every tested TUM window.

## Current Extension: Pose-Anchor Verification

Another follow-up direction is to keep the current-frame geometric verification, but add an explicit anchor-pose check against the archived hypothesis:

1. Archive a lightweight camera-pose summary together with the recovered state.
2. For ambiguous or stale recover candidates, compare the current predicted pose against that archived anchor.
3. Accept only when the candidate improves both current-frame geometry and anchor-pose consistency.

This is intended to be stricter than descriptor-only recovery, but less invasive than shadow recovery.
