from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from nuc_runtime.config import TrackingConfig
from nuc_runtime.descriptors import compute_global_descriptor
from nuc_runtime.models import TrackingOutput


EARTH_RADIUS_M = 6378137.0


@dataclass
class _MonoFrameState:
    frame_idx: int
    pose: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None


def _rotx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _roty(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _load_oxts_packet(path: Path) -> dict[str, float]:
    values = [float(v) for v in path.read_text(encoding="utf-8").strip().split()]
    if len(values) < 6:
        raise ValueError(f"Unexpected OXTS packet format in {path}")
    return {
        "lat": values[0],
        "lon": values[1],
        "alt": values[2],
        "roll": values[3],
        "pitch": values[4],
        "yaw": values[5],
    }


def _oxts_to_pose(packet: dict[str, float], scale: float) -> np.ndarray:
    tx = scale * packet["lon"] * math.pi * EARTH_RADIUS_M / 180.0
    ty = scale * EARTH_RADIUS_M * math.log(math.tan((90.0 + packet["lat"]) * math.pi / 360.0))
    tz = packet["alt"]
    rotation = _rotz(packet["yaw"]) @ _roty(packet["pitch"]) @ _rotx(packet["roll"])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return pose


def _load_oxts_step_scales(oxts_dir: Path, frame_count: int) -> list[float]:
    paths = sorted(oxts_dir.expanduser().resolve().glob("*.txt"))[:frame_count]
    if len(paths) < 2:
        return []
    packets = [_load_oxts_packet(path) for path in paths]
    scale = math.cos(packets[0]["lat"] * math.pi / 180.0)
    first_inv = np.linalg.inv(_oxts_to_pose(packets[0], scale))
    poses = [first_inv @ _oxts_to_pose(packet, scale) for packet in packets]
    steps = [0.0]
    for prev, curr in zip(poses[:-1], poses[1:]):
        steps.append(float(np.linalg.norm(curr[:3, 3] - prev[:3, 3])))
    return steps


class RGBMonocularVOAdapter:
    """Small monocular RGB VO baseline for RGB-only reconstruction smoke tests.

    This is intentionally a baseline, not a production SLAM system. It estimates
    relative pose from ORB matches and an essential matrix. Monocular translation
    scale is either a fixed step or an optional external KITTI OXTS step scale.
    """

    def __init__(
        self,
        image_dir: str | Path,
        timestamps_path: str | Path,
        intrinsic: np.ndarray,
        config: TrackingConfig,
        frame_step: int = 1,
        max_frames: int = 0,
        fixed_step_scale: float = 0.5,
        scale_source: str = "fixed",
        oxts_dir: str | Path = "",
    ):
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.timestamps_path = Path(timestamps_path).expanduser().resolve()
        self.K = intrinsic.astype(np.float64)
        self.config = config
        self.frame_step = max(1, int(frame_step))
        self.max_frames = max_frames
        self.fixed_step_scale = float(fixed_step_scale)
        self.scale_source = scale_source
        self.orb = cv2.ORB_create(nfeatures=config.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev_state: _MonoFrameState | None = None
        self._last_keyframe_idx = -10**9
        self._pose = np.eye(4, dtype=np.float32)

        self.image_paths = sorted(
            [
                *self.image_dir.glob("*.png"),
                *self.image_dir.glob("*.jpg"),
                *self.image_dir.glob("*.jpeg"),
            ]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No RGB images found under {self.image_dir}")
        self.timestamps = np.array(
            [
                float(line.strip())
                for line in self.timestamps_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
            dtype=np.float64,
        )
        total = min(len(self.image_paths), len(self.timestamps))
        self.image_paths = self.image_paths[:total]
        self.timestamps = self.timestamps[:total]
        self.oxts_steps = (
            _load_oxts_step_scales(Path(oxts_dir), total)
            if scale_source == "oxts" and oxts_dir
            else []
        )

    def __iter__(self):
        total = len(self.image_paths)
        if self.max_frames > 0:
            total = min(total, self.max_frames)
        for frame_idx in range(0, total, self.frame_step):
            yield self._build_output(frame_idx)

    def _build_output(self, frame_idx: int) -> TrackingOutput:
        image_path = self.image_paths[frame_idx]
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"Failed to load RGB image: {image_path}")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        keypoints = keypoints or []
        descriptor = compute_global_descriptor(frame_bgr, descriptors)

        match_count = 0
        inlier_count = 0
        pixel_motion = 0.0
        track_ok = self._prev_state is None
        is_keyframe = self._prev_state is None
        scale = 0.0
        notes = {
            "source": "opencv_mono_rgb",
            "scale_source": self.scale_source,
        }

        if self._prev_state is not None:
            (
                relative_pose,
                match_count,
                inlier_count,
                pixel_motion,
                scale,
                track_ok,
            ) = self._estimate_relative_pose(frame_idx, self._prev_state, keypoints, descriptors)
            if track_ok:
                self._pose = (self._pose @ relative_pose).astype(np.float32)
            is_keyframe = self._is_keyframe(frame_idx, self._pose, match_count, pixel_motion)
            notes["mono_step_scale"] = float(scale)
            notes["pose_translation_step"] = float(
                np.linalg.norm(self._pose[:3, 3] - self._prev_state.pose[:3, 3])
            )

        if is_keyframe:
            self._last_keyframe_idx = frame_idx

        pose = self._pose.copy()
        self._prev_state = _MonoFrameState(
            frame_idx=frame_idx,
            pose=pose,
            keypoints=keypoints,
            descriptors=descriptors,
        )
        return TrackingOutput(
            frame_idx=frame_idx,
            timestamp_sec=float(self.timestamps[frame_idx]),
            pose=pose,
            is_keyframe=is_keyframe,
            descriptor=descriptor,
            orb_descriptors=descriptors,
            keypoints_xy=self._keypoints_to_array(keypoints),
            keypoint_count=len(keypoints),
            match_count=match_count,
            inlier_count=inlier_count,
            pixel_motion=pixel_motion,
            track_ok=track_ok,
            frame_shape=gray.shape[:2],
            image_path=str(image_path),
            right_image_path=None,
            notes=notes,
        )

    def _step_scale(self, frame_idx: int) -> float:
        if self.scale_source == "oxts" and 0 <= frame_idx < len(self.oxts_steps):
            value = self.oxts_steps[frame_idx]
            if math.isfinite(value) and value > 1e-6:
                return float(value)
        return self.fixed_step_scale

    def _estimate_relative_pose(
        self,
        frame_idx: int,
        prev_state: _MonoFrameState,
        curr_keypoints: list,
        curr_descriptors: np.ndarray | None,
    ) -> tuple[np.ndarray, int, int, float, float, bool]:
        relative = np.eye(4, dtype=np.float32)
        if prev_state.descriptors is None or curr_descriptors is None:
            return relative, 0, 0, 0.0, 0.0, False
        raw_matches = self.matcher.knnMatch(prev_state.descriptors, curr_descriptors, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.config.ratio_test * second.distance:
                good_matches.append(first)
        match_count = len(good_matches)
        if match_count < max(12, self.config.min_matches):
            return relative, match_count, 0, 0.0, 0.0, False

        pts_prev = np.float32([prev_state.keypoints[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches])
        displacements = np.linalg.norm(pts_curr - pts_prev, axis=1)
        pixel_motion = float(np.median(displacements)) if displacements.size else 0.0

        essential, mask = cv2.findEssentialMat(
            pts_prev,
            pts_curr,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5,
        )
        if essential is None or mask is None:
            return relative, match_count, 0, pixel_motion, 0.0, False
        if essential.ndim == 2 and essential.shape[0] > 3:
            essential = essential[:3, :3]
        _, rotation, translation, pose_mask = cv2.recoverPose(
            essential,
            pts_prev,
            pts_curr,
            self.K,
            mask=mask,
        )
        if pose_mask is not None:
            inlier_count = int(np.count_nonzero(pose_mask))
        else:
            inlier_count = int(np.count_nonzero(mask))
        if inlier_count < max(10, self.config.min_matches // 2):
            return relative, match_count, inlier_count, pixel_motion, 0.0, False

        scale = self._step_scale(frame_idx)
        rotation = rotation.astype(np.float32)
        translation = (translation.reshape(3).astype(np.float32) * float(scale))

        # recoverPose returns current_from_previous. We store camera-to-world, so
        # compose with previous_from_current.
        relative[:3, :3] = rotation.T
        relative[:3, 3] = -rotation.T @ translation
        return relative, match_count, inlier_count, pixel_motion, scale, True

    def _is_keyframe(
        self,
        frame_idx: int,
        pose: np.ndarray,
        match_count: int,
        pixel_motion: float,
    ) -> bool:
        frames_since_kf = frame_idx - self._last_keyframe_idx
        if frames_since_kf >= self.config.max_keyframe_gap:
            return True
        if frames_since_kf < self.config.min_keyframe_gap:
            return False
        if pixel_motion >= self.config.keyframe_motion_threshold:
            return True
        if match_count < self.config.low_match_keyframe_threshold:
            return True
        if self._prev_state is None:
            return True
        translation_step = float(np.linalg.norm(pose[:3, 3] - self._prev_state.pose[:3, 3]))
        return translation_step >= self.config.min_translation_step

    def _keypoints_to_array(self, keypoints: list) -> np.ndarray:
        if not keypoints:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)
