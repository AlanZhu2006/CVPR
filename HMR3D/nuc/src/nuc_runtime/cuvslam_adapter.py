from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from nuc_runtime.config import TrackingConfig
from nuc_runtime.descriptors import compute_global_descriptor
from nuc_runtime.models import TrackingOutput


@dataclass
class _FrameState:
    frame_idx: int
    pose: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None


def _quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw.astype(np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def tum_pose_to_matrix(tum_row: np.ndarray) -> np.ndarray:
    # TUM pose rows are usually: timestamp tx ty tz qx qy qz qw.
    # Some internal helpers may pass only: tx ty tz qx qy qz qw.
    if tum_row.shape[0] >= 8:
        translation = tum_row[1:4]
        quaternion = tum_row[4:8]
    else:
        translation = tum_row[:3]
        quaternion = tum_row[3:7]
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quaternion_to_matrix(quaternion)
    matrix[:3, 3] = translation.astype(np.float32)
    return matrix


class CUVSLAMOfflineKITTIAdapter:
    def __init__(
        self,
        sequence_path: str | Path,
        trajectory_path: str | Path,
        config: TrackingConfig,
        frame_step: int = 1,
        max_frames: int = 0,
    ):
        self.sequence_path = Path(sequence_path).expanduser().resolve()
        self.trajectory_path = Path(trajectory_path).expanduser().resolve()
        self.config = config
        self.frame_step = max(1, frame_step)
        self.max_frames = max_frames
        self.orb = cv2.ORB_create(nfeatures=config.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.timestamps = np.array(
            [float(line.strip()) for line in (self.sequence_path / "times.txt").read_text(encoding="utf-8").splitlines()],
            dtype=np.float64,
        )
        self.trajectory = np.loadtxt(self.trajectory_path, dtype=np.float32)
        if self.trajectory.ndim == 1:
            self.trajectory = self.trajectory[None, :]
        self._prev_state: _FrameState | None = None
        self._last_keyframe_idx = -10**9

    def __iter__(self):
        total = min(len(self.timestamps), len(self.trajectory))
        if self.max_frames > 0:
            total = min(total, self.max_frames)

        for frame_idx in range(0, total, self.frame_step):
            yield self._build_output(frame_idx)

    def _build_output(self, frame_idx: int) -> TrackingOutput:
        image_path = self.sequence_path / "image_0" / f"{frame_idx:06d}.png"
        right_image_path = self.sequence_path / "image_1" / f"{frame_idx:06d}.png"
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        keypoints = keypoints or []
        pose = tum_pose_to_matrix(self.trajectory[frame_idx])
        descriptor = compute_global_descriptor(frame_bgr, descriptors)

        match_count = 0
        inlier_count = 0
        pixel_motion = 0.0
        track_ok = self._prev_state is None
        is_keyframe = self._prev_state is None
        notes = {
            "source": "cuvslam_offline_kitti",
        }

        if self._prev_state is not None:
            match_count, inlier_count, pixel_motion = self._match_stats(
                self._prev_state, keypoints, descriptors
            )
            track_ok = match_count >= self.config.min_matches
            is_keyframe = self._is_keyframe(frame_idx, pose, match_count, pixel_motion)
            notes["pose_translation_step"] = float(
                np.linalg.norm(pose[:3, 3] - self._prev_state.pose[:3, 3])
            )

        if is_keyframe:
            self._last_keyframe_idx = frame_idx

        self._prev_state = _FrameState(
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
            right_image_path=str(right_image_path),
            notes=notes,
        )

    def _match_stats(
        self,
        prev_state: _FrameState,
        curr_keypoints: list,
        curr_descriptors: np.ndarray | None,
    ) -> tuple[int, int, float]:
        if prev_state.descriptors is None or curr_descriptors is None:
            return 0, 0, 0.0

        raw_matches = self.matcher.knnMatch(prev_state.descriptors, curr_descriptors, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.config.ratio_test * second.distance:
                good_matches.append(first)

        match_count = len(good_matches)
        if match_count == 0:
            return 0, 0, 0.0

        prev_points = np.float32([prev_state.keypoints[m.queryIdx].pt for m in good_matches])
        curr_points = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches])
        displacements = np.linalg.norm(curr_points - prev_points, axis=1)
        pixel_motion = float(np.median(displacements))

        inlier_count = match_count
        if match_count >= 8:
            homography, mask = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, 3.0)
            if homography is not None and mask is not None:
                inlier_count = int(mask.ravel().sum())

        return match_count, inlier_count, pixel_motion

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


class CUVSLAMMonocularRGBAdapter:
    """Run PyCuVSLAM monocular odometry on one RGB image stream.

    cuVSLAM consumes a single grayscale image in Mono mode, but the source stream
    remains the same RGB camera used by LingBot/colorization. ORB stats are kept
    only for our telemetry/keyframe policy; poses come from cuVSLAM.
    """

    def __init__(
        self,
        image_dir: str | Path,
        timestamps_path: str | Path,
        intrinsic: np.ndarray,
        config: TrackingConfig,
        frame_step: int = 1,
        max_frames: int = 0,
        image_size_hw: tuple[int, int] | None = None,
    ):
        import cuvslam

        self.image_dir = Path(image_dir).expanduser().resolve()
        self.timestamps_path = Path(timestamps_path).expanduser().resolve()
        self.K = intrinsic.astype(np.float64)
        self.config = config
        self.frame_step = max(1, int(frame_step))
        self.max_frames = max_frames
        self.orb = cv2.ORB_create(nfeatures=config.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev_state: _FrameState | None = None
        self._last_keyframe_idx = -10**9
        self._last_pose = np.eye(4, dtype=np.float32)

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
        if total == 0:
            raise ValueError("No overlapping RGB images/timestamps for cuVSLAM Mono")

        if image_size_hw is None:
            probe = cv2.imread(str(self.image_paths[0]), cv2.IMREAD_COLOR)
            if probe is None:
                raise FileNotFoundError(f"Failed to load RGB image: {self.image_paths[0]}")
            image_size_hw = probe.shape[:2]
        height, width = int(image_size_hw[0]), int(image_size_hw[1])

        camera = cuvslam.Camera()
        camera.size = [width, height]
        camera.focal = [float(self.K[0, 0]), float(self.K[1, 1])]
        camera.principal = [float(self.K[0, 2]), float(self.K[1, 2])]
        camera.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole, [])
        camera.border_top = 10
        camera.border_bottom = 10
        camera.border_left = 10
        camera.border_right = 10
        rig = cuvslam.Rig([camera])
        cfg = cuvslam.Tracker.OdometryConfig()
        if not hasattr(cuvslam.Tracker.OdometryMode, "Mono"):
            raise RuntimeError("Installed cuVSLAM does not expose Tracker.OdometryMode.Mono")
        cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Mono
        self.tracker = cuvslam.Tracker(rig, cfg)

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
        gray = np.ascontiguousarray(gray)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        keypoints = keypoints or []
        descriptor = compute_global_descriptor(frame_bgr, descriptors)

        match_count = 0
        inlier_count = 0
        pixel_motion = 0.0
        if self._prev_state is not None:
            match_count, inlier_count, pixel_motion = self._match_stats(
                self._prev_state,
                keypoints,
                descriptors,
            )

        timestamp_ns = int(float(self.timestamps[frame_idx]) * 1e9)
        pose_estimate, _ = self.tracker.track(timestamp_ns, [gray])
        track_ok = bool(getattr(pose_estimate, "world_from_rig", None) is not None)
        notes = {
            "source": "cuvslam_mono_rgb",
            "odometry_mode": "Mono",
            "pose_source": "cuvslam",
        }
        if track_ok:
            pose = self._pose_estimate_to_matrix(pose_estimate)
            self._last_pose = pose
        else:
            pose = self._last_pose.copy()
            notes["warning"] = "cuvslam_world_from_rig_missing_reusing_last_pose"

        if self._prev_state is None:
            is_keyframe = True
        elif not track_ok:
            is_keyframe = False
        else:
            is_keyframe = self._is_keyframe(frame_idx, pose, match_count, pixel_motion)
            notes["pose_translation_step"] = float(
                np.linalg.norm(pose[:3, 3] - self._prev_state.pose[:3, 3])
            )
        if is_keyframe:
            self._last_keyframe_idx = frame_idx

        self._prev_state = _FrameState(
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

    def _pose_estimate_to_matrix(self, pose_estimate) -> np.ndarray:
        pose_raw = pose_estimate.world_from_rig.pose
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = _quaternion_to_matrix(np.asarray(pose_raw.rotation, dtype=np.float32))
        matrix[:3, 3] = np.asarray(pose_raw.translation, dtype=np.float32)
        return matrix

    def _match_stats(
        self,
        prev_state: _FrameState,
        curr_keypoints: list,
        curr_descriptors: np.ndarray | None,
    ) -> tuple[int, int, float]:
        if prev_state.descriptors is None or curr_descriptors is None:
            return 0, 0, 0.0
        raw_matches = self.matcher.knnMatch(prev_state.descriptors, curr_descriptors, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.config.ratio_test * second.distance:
                good_matches.append(first)
        match_count = len(good_matches)
        if match_count == 0:
            return 0, 0, 0.0
        prev_points = np.float32([prev_state.keypoints[m.queryIdx].pt for m in good_matches])
        curr_points = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches])
        displacements = np.linalg.norm(curr_points - prev_points, axis=1)
        pixel_motion = float(np.median(displacements))
        inlier_count = match_count
        if match_count >= 8:
            homography, mask = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, 3.0)
            if homography is not None and mask is not None:
                inlier_count = int(mask.ravel().sum())
        return match_count, inlier_count, pixel_motion

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
