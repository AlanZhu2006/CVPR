from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch

from .ttt3r_env import bootstrap_ttt3r_imports


def prepare_input(
    img_paths: List[str],
    size: int,
    revisit: int = 1,
    update: bool = True,
    crop: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    bootstrap_ttt3r_imports()
    from dust3r.utils.image import load_images_for_eval as load_images

    images = load_images(img_paths, size=size, crop=crop, verbose=False)
    views: List[Dict[str, torch.Tensor]] = []
    for idx, image in enumerate(images):
        view = {
            "img": image["img"],
            "ray_map": torch.full(
                (
                    image["img"].shape[0],
                    6,
                    image["img"].shape[-2],
                    image["img"].shape[-1],
                ),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(image["true_shape"]),
            "idx": idx,
            "instance": str(idx),
            "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)

    if revisit <= 1:
        return views

    repeated: List[Dict[str, torch.Tensor]] = []
    for revisit_idx in range(revisit):
        for view_idx, view in enumerate(views):
            new_view = deepcopy(view)
            new_view["idx"] = revisit_idx * len(views) + view_idx
            new_view["instance"] = str(revisit_idx * len(views) + view_idx)
            if revisit_idx > 0 and not update:
                new_view["update"] = torch.tensor(False).unsqueeze(0)
            repeated.append(new_view)
    return repeated


def recover_cam_params(
    pts3ds_self: torch.Tensor,
    pts3ds_other: torch.Tensor,
    conf_self: torch.Tensor,
    conf_other: torch.Tensor,
):
    bootstrap_ttt3r_imports()
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.utils.geometry import weighted_procrustes

    batch, height, width, _ = pts3ds_self.shape
    principal_point = (
        torch.tensor([width // 2, height // 2], device=pts3ds_self.device)
        .float()
        .repeat(batch, 1)
        .reshape(batch, 1, 2)
    )
    focal = estimate_focal_knowing_depth(pts3ds_self, principal_point, focal_mode="weiszfeld")

    pts3ds_self = pts3ds_self.reshape(batch, -1, 3)
    pts3ds_other = pts3ds_other.reshape(batch, -1, 3)
    conf_self = conf_self.reshape(batch, -1)
    conf_other = conf_other.reshape(batch, -1)
    c2w = weighted_procrustes(
        pts3ds_self,
        pts3ds_other,
        torch.log(conf_self) * torch.log(conf_other),
        use_weights=True,
        return_T=True,
    )
    return c2w, focal, principal_point.reshape(batch, 2)


def compute_prediction_quality(
    prediction: Dict[str, torch.Tensor],
    *,
    geometry_weight_floor: float = 1e-6,
) -> Dict[str, float]:
    bootstrap_ttt3r_imports()
    from dust3r.utils.geometry import geotrf, weighted_procrustes

    pts3ds_self = prediction["pts3d_in_self_view"]
    pts3ds_other = prediction["pts3d_in_other_view"]
    conf_self = prediction["conf_self"]
    conf_other = prediction["conf"]

    batch = pts3ds_self.shape[0]
    pts_self_flat = pts3ds_self.reshape(batch, -1, 3)
    pts_other_flat = pts3ds_other.reshape(batch, -1, 3)
    conf_self_flat = conf_self.reshape(batch, -1).clamp_min(geometry_weight_floor)
    conf_other_flat = conf_other.reshape(batch, -1).clamp_min(geometry_weight_floor)
    weights = (torch.log(conf_self_flat) * torch.log(conf_other_flat)).clamp_min(geometry_weight_floor)

    transform = weighted_procrustes(
        pts_self_flat,
        pts_other_flat,
        weights,
        use_weights=True,
        return_T=True,
    )
    aligned_self = geotrf(transform, pts_self_flat)
    sq_error = (aligned_self - pts_other_flat).pow(2).sum(dim=-1)
    weighted_sq_error = (weights * sq_error).sum(dim=-1) / weights.sum(dim=-1).clamp_min(geometry_weight_floor)
    geo_rmse = torch.sqrt(weighted_sq_error).mean().item()
    mean_log_conf = (
        torch.log(conf_self_flat).mean().item() + torch.log(conf_other_flat).mean().item()
    ) / 2.0
    return {
        "geo_rmse": geo_rmse,
        "mean_log_conf": mean_log_conf,
    }


def extract_camera_pose_matrix(prediction: Dict[str, torch.Tensor]) -> torch.Tensor | None:
    pose_encoding = prediction.get("camera_pose")
    if pose_encoding is None:
        return None
    bootstrap_ttt3r_imports()
    from dust3r.utils.camera import pose_encoding_to_camera

    return pose_encoding_to_camera(pose_encoding.detach().clone())


def compute_anchor_pose_score_from_matrices(
    pred_camera_pose: torch.Tensor,
    archive_camera_pose: torch.Tensor,
    *,
    rotation_weight: float = 0.25,
) -> Dict[str, float]:
    archive_camera_pose = archive_camera_pose.to(pred_camera_pose.device, dtype=pred_camera_pose.dtype)
    relative = torch.linalg.inv(archive_camera_pose) @ pred_camera_pose
    translation_error = relative[:, :3, 3].norm(dim=-1).mean().item()

    rotation = relative[:, :3, :3]
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    rotation_error = torch.arccos(cosine).mean().item()
    anchor_score = translation_error + rotation_weight * rotation_error
    return {
        "anchor_translation_error": translation_error,
        "anchor_rotation_error": rotation_error,
        "anchor_score": anchor_score,
    }


def compute_anchor_pose_quality(
    prediction: Dict[str, torch.Tensor],
    archive_camera_pose: torch.Tensor | None,
    *,
    rotation_weight: float = 0.25,
) -> Dict[str, float] | None:
    if archive_camera_pose is None:
        return None
    pred_camera_pose = extract_camera_pose_matrix(prediction)
    if pred_camera_pose is None:
        return None
    return compute_anchor_pose_score_from_matrices(
        pred_camera_pose,
        archive_camera_pose,
        rotation_weight=rotation_weight,
    )


def prepare_relpose_output(outputs: Dict[str, List[Dict[str, torch.Tensor]]], revisit: int = 1, solve_pose: bool = False):
    bootstrap_ttt3r_imports()
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.utils.camera import pose_encoding_to_camera

    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    pts3ds_self = [pred["pts3d_in_self_view"].cpu() for pred in outputs["pred"]]
    pts3ds_other = [pred["pts3d_in_other_view"].cpu() for pred in outputs["pred"]]
    conf_self = [pred["conf_self"].cpu() for pred in outputs["pred"]]
    conf_other = [pred["conf"].cpu() for pred in outputs["pred"]]
    pts3ds_self_tensor = torch.cat(pts3ds_self, dim=0)

    if solve_pose:
        pr_poses, focal, principal_point = recover_cam_params(
            pts3ds_self_tensor,
            torch.cat(pts3ds_other, dim=0),
            torch.cat(conf_self, dim=0),
            torch.cat(conf_other, dim=0),
        )
    else:
        pr_poses = [pose_encoding_to_camera(pred["camera_pose"].clone()).cpu() for pred in outputs["pred"]]
        pr_poses = torch.cat(pr_poses, dim=0)
        batch, height, width, _ = pts3ds_self_tensor.shape
        principal_point = (
            torch.tensor([width // 2, height // 2], device=pts3ds_self_tensor.device)
            .float()
            .repeat(batch, 1)
            .reshape(batch, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self_tensor, principal_point, focal_mode="weiszfeld")

    colors = [0.5 * (pred["rgb"][0] + 1.0) for pred in outputs["pred"]]
    cam_dict = {"focal": focal.cpu().numpy(), "pp": principal_point.cpu().numpy()}
    return (
        colors,
        pts3ds_self_tensor,
        pts3ds_other,
        conf_self,
        conf_other,
        cam_dict,
        pr_poses,
    )


def prepare_video_depth_output(outputs: Dict[str, List[Dict[str, torch.Tensor]]], revisit: int = 1):
    return prepare_relpose_output(outputs, revisit=revisit, solve_pose=False)
