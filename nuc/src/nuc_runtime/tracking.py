from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nuc_runtime.config import TrackingConfig
from nuc_runtime.descriptors import compute_global_descriptor
from nuc_runtime.models import FramePacket, TrackingOutput, identity_pose


@dataclass
class _FeatureState:
    gray: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None
    pose: np.ndarray
    frame_idx: int


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
        keypoint_count = 0 if keypoints is None else len(keypoints)
        descriptor = compute_global_descriptor(frame_bgr, descriptors)

        if self.prev_state is None:
            pose = identity_pose()
            self.last_keyframe_idx = packet.frame_idx
            self.prev_state = _FeatureState(gray, keypoints or [], descriptors, pose, packet.frame_idx)
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
                notes={"bootstrap": True},
            )

        pose, match_count, inlier_count, pixel_motion, track_ok = self._estimate_pose(
            prev_state=self.prev_state,
            curr_keypoints=keypoints or [],
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

        self.prev_state = _FeatureState(gray, keypoints or [], descriptors, pose, packet.frame_idx)
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
            notes={},
        )

    def _estimate_pose(
        self,
        prev_state: _FeatureState,
        curr_keypoints: list,
        curr_descriptors: np.ndarray | None,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, int, int, float, bool]:
        if prev_state.descriptors is None or curr_descriptors is None:
            return prev_state.pose.copy(), 0, 0, 0.0, False

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
            return prev_state.pose.copy(), match_count, 0, 0.0, False

        prev_points = np.float32([prev_state.keypoints[m.queryIdx].pt for m in good_matches])
        curr_points = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches])
        displacements = np.linalg.norm(curr_points - prev_points, axis=1)
        pixel_motion = float(np.median(displacements))

        height, width = frame_shape
        focal = self.config.focal_length_scale * max(height, width)
        principal = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [[focal, 0.0, principal[0]], [0.0, focal, principal[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        essential, _ = cv2.findEssentialMat(
            prev_points,
            curr_points,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if essential is None:
            return prev_state.pose.copy(), match_count, 0, pixel_motion, False

        _, rotation, translation, pose_mask = cv2.recoverPose(
            essential,
            prev_points,
            curr_points,
            camera_matrix,
        )
        inlier_count = int(np.count_nonzero(pose_mask))
        if inlier_count < self.config.min_pose_inliers:
            return prev_state.pose.copy(), match_count, inlier_count, pixel_motion, False

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
        return current_pose.astype(np.float32), match_count, inlier_count, pixel_motion, True
