from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nuc_runtime.config import TrackingConfig
from nuc_runtime.descriptors import compute_global_descriptor, normalize_vector
from nuc_runtime.models import FramePacket, TrackingOutput, identity_pose


@dataclass
class _FeatureState:
    gray: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None
    pose: np.ndarray
    frame_idx: int
    stereo_points: np.ndarray | None = None


class ORBTrackingFrontend:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=config.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev_state: _FeatureState | None = None
        self.last_keyframe_idx = -10**9

    def process(self, packet: FramePacket) -> TrackingOutput:
        frame_bgr = packet.frame_bgr
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        keypoints = keypoints or []
        keypoint_count = len(keypoints)

        stereo_points = None
        stereo_point_count = 0
        if packet.frame_bgr_right is not None:
            right_gray = cv2.cvtColor(packet.frame_bgr_right, cv2.COLOR_BGR2GRAY)
            right_keypoints, right_descriptors = self.orb.detectAndCompute(right_gray, None)
            stereo_points, stereo_point_count = self._estimate_stereo_points(
                frame_shape=gray.shape[:2],
                left_keypoints=keypoints,
                left_descriptors=descriptors,
                right_keypoints=right_keypoints or [],
                right_descriptors=right_descriptors,
            )
        descriptor = self._build_descriptor(frame_bgr, descriptors, packet.frame_bgr_right)

        if self.prev_state is None:
            pose = identity_pose()
            self.last_keyframe_idx = packet.frame_idx
            self.prev_state = _FeatureState(
                gray,
                keypoints,
                descriptors,
                pose,
                packet.frame_idx,
                stereo_points=stereo_points,
            )
            return TrackingOutput(
                frame_idx=packet.frame_idx,
                timestamp_sec=packet.timestamp_sec,
                pose=pose,
                is_keyframe=True,
                descriptor=descriptor,
                orb_descriptors=descriptors,
                keypoint_count=keypoint_count,
                match_count=0,
                inlier_count=0,
                pixel_motion=0.0,
                track_ok=True,
                frame_shape=gray.shape[:2],
                notes={
                    "bootstrap": True,
                    "tracking_mode": "bootstrap_stereo" if packet.frame_bgr_right is not None else "bootstrap_mono",
                    "stereo_points": stereo_point_count,
                },
            )

        pose, match_count, inlier_count, pixel_motion, track_ok, tracking_mode = self._estimate_pose(
            prev_state=self.prev_state,
            curr_keypoints=keypoints,
            curr_descriptors=descriptors,
            frame_shape=gray.shape[:2],
        )

        frames_since_kf = packet.frame_idx - self.last_keyframe_idx
        is_keyframe = False
        if frames_since_kf >= self.config.max_keyframe_gap:
            is_keyframe = True
        elif frames_since_kf >= self.config.min_keyframe_gap:
            if pixel_motion >= self.config.keyframe_motion_threshold:
                is_keyframe = True
            elif match_count < self.config.low_match_keyframe_threshold:
                is_keyframe = True

        if is_keyframe:
            self.last_keyframe_idx = packet.frame_idx

        self.prev_state = _FeatureState(
            gray,
            keypoints,
            descriptors,
            pose,
            packet.frame_idx,
            stereo_points=stereo_points,
        )
        return TrackingOutput(
            frame_idx=packet.frame_idx,
            timestamp_sec=packet.timestamp_sec,
            pose=pose,
            is_keyframe=is_keyframe,
            descriptor=descriptor,
            orb_descriptors=descriptors,
            keypoint_count=keypoint_count,
            match_count=match_count,
            inlier_count=inlier_count,
            pixel_motion=pixel_motion,
            track_ok=track_ok,
            frame_shape=gray.shape[:2],
            notes={
                "tracking_mode": tracking_mode,
                "stereo_points": stereo_point_count,
                "has_right_frame": packet.frame_bgr_right is not None,
            },
        )

    def _build_descriptor(
        self,
        frame_bgr: np.ndarray,
        left_descriptors: np.ndarray | None,
        right_frame_bgr: np.ndarray | None,
    ) -> np.ndarray:
        descriptor = compute_global_descriptor(frame_bgr, left_descriptors)
        if right_frame_bgr is None:
            return descriptor

        right_descriptor = compute_global_descriptor(right_frame_bgr, None)
        return normalize_vector((descriptor + right_descriptor) * 0.5)

    def _estimate_pose(
        self,
        prev_state: _FeatureState,
        curr_keypoints: list,
        curr_descriptors: np.ndarray | None,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, int, int, float, bool, str]:
        if prev_state.descriptors is None or curr_descriptors is None:
            return prev_state.pose.copy(), 0, 0, 0.0, False, "no_descriptors"

        raw_matches = self.matcher.knnMatch(prev_state.descriptors, curr_descriptors, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.config.ratio_test * second.distance:
                good_matches.append(first)

        match_count = len(good_matches)
        if match_count < self.config.min_matches:
            return prev_state.pose.copy(), match_count, 0, 0.0, False, "low_match_count"

        prev_points = np.float32([prev_state.keypoints[m.queryIdx].pt for m in good_matches])
        curr_points = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches])
        displacements = np.linalg.norm(curr_points - prev_points, axis=1)
        pixel_motion = float(np.median(displacements))

        stereo_result = self._estimate_stereo_pose(
            prev_state=prev_state,
            good_matches=good_matches,
            curr_keypoints=curr_keypoints,
            frame_shape=frame_shape,
            pixel_motion=pixel_motion,
        )
        if stereo_result is not None:
            pose, inlier_count, track_ok = stereo_result
            return pose, match_count, inlier_count, pixel_motion, track_ok, "stereo_pnp"

        pose, inlier_count, track_ok = self._estimate_monocular_pose(
            prev_state=prev_state,
            prev_points=prev_points,
            curr_points=curr_points,
            frame_shape=frame_shape,
            pixel_motion=pixel_motion,
        )
        return pose, match_count, inlier_count, pixel_motion, track_ok, "mono_essential"

    def _estimate_monocular_pose(
        self,
        prev_state: _FeatureState,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        frame_shape: tuple[int, int],
        pixel_motion: float,
    ) -> tuple[np.ndarray, int, bool]:
        camera_matrix = self._build_camera_matrix(frame_shape)
        height, width = frame_shape

        essential, _ = cv2.findEssentialMat(
            prev_points,
            curr_points,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if essential is None:
            return prev_state.pose.copy(), 0, False

        _, rotation, translation, pose_mask = cv2.recoverPose(
            essential,
            prev_points,
            curr_points,
            camera_matrix,
        )
        inlier_count = int(np.count_nonzero(pose_mask))
        if inlier_count < self.config.min_pose_inliers:
            return prev_state.pose.copy(), inlier_count, False

        step_scale = pixel_motion / max(float(height), float(width))
        step_scale = float(
            np.clip(
                step_scale,
                self.config.min_translation_step,
                self.config.max_translation_step,
            )
        )
        translation = translation.reshape(3) * step_scale

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation.T.astype(np.float32)
        transform[:3, 3] = (-rotation.T @ translation).astype(np.float32)
        current_pose = prev_state.pose @ transform
        return current_pose.astype(np.float32), inlier_count, True

    def _estimate_stereo_pose(
        self,
        prev_state: _FeatureState,
        good_matches: list,
        curr_keypoints: list,
        frame_shape: tuple[int, int],
        pixel_motion: float,
    ) -> tuple[np.ndarray, int, bool] | None:
        if prev_state.stereo_points is None:
            return None

        object_points = []
        image_points = []
        for match in good_matches:
            point_3d = prev_state.stereo_points[match.queryIdx]
            if not np.isfinite(point_3d).all():
                continue
            object_points.append(point_3d)
            image_points.append(curr_keypoints[match.trainIdx].pt)

        if len(object_points) < self.config.min_stereo_points:
            return None

        camera_matrix = self._build_camera_matrix(frame_shape)
        object_points_np = np.asarray(object_points, dtype=np.float32)
        image_points_np = np.asarray(image_points, dtype=np.float32)
        ok, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
            object_points_np,
            image_points_np,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.99,
            iterationsCount=100,
        )
        if not ok or translation_vec is None:
            return None

        inlier_count = 0 if inliers is None else int(len(inliers))
        if inlier_count < self.config.min_pose_inliers:
            return None

        rotation, _ = cv2.Rodrigues(rotation_vec)
        translation = translation_vec.reshape(3).astype(np.float32)
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm > self.config.max_translation_step > 0:
            translation *= self.config.max_translation_step / translation_norm
        elif translation_norm == 0.0 and pixel_motion > 0.0:
            return None

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation.T.astype(np.float32)
        transform[:3, 3] = (-rotation.T @ translation).astype(np.float32)
        current_pose = prev_state.pose @ transform
        return current_pose.astype(np.float32), inlier_count, True

    def _estimate_stereo_points(
        self,
        frame_shape: tuple[int, int],
        left_keypoints: list,
        left_descriptors: np.ndarray | None,
        right_keypoints: list,
        right_descriptors: np.ndarray | None,
    ) -> tuple[np.ndarray | None, int]:
        if left_descriptors is None or right_descriptors is None or not left_keypoints or not right_keypoints:
            return None, 0

        points = np.full((len(left_keypoints), 3), np.nan, dtype=np.float32)
        raw_matches = self.matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        camera_matrix = self._build_camera_matrix(frame_shape)
        focal = float(camera_matrix[0, 0])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])

        valid_count = 0
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance >= self.config.stereo_ratio_test * second.distance:
                continue

            left_pt = left_keypoints[first.queryIdx].pt
            right_pt = right_keypoints[first.trainIdx].pt
            disparity = float(left_pt[0] - right_pt[0])
            vertical_diff = abs(float(left_pt[1] - right_pt[1]))
            if disparity < self.config.min_stereo_disparity:
                continue
            if vertical_diff > self.config.max_stereo_vertical_diff:
                continue

            depth = focal * self.config.stereo_baseline_m / disparity
            if depth <= 0.0 or depth > self.config.max_stereo_depth_m:
                continue

            x_coord = (left_pt[0] - cx) * depth / focal
            y_coord = (left_pt[1] - cy) * depth / focal
            points[first.queryIdx] = np.array([x_coord, y_coord, depth], dtype=np.float32)
            valid_count += 1

        if valid_count == 0:
            return None, 0
        return points, valid_count

    def _build_camera_matrix(self, frame_shape: tuple[int, int]) -> np.ndarray:
        height, width = frame_shape
        focal = self.config.focal_length_scale * max(height, width)
        principal = (width / 2.0, height / 2.0)
        return np.array(
            [[focal, 0.0, principal[0]], [0.0, focal, principal[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
