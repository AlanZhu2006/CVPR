from __future__ import annotations

import cv2
import numpy as np

from nuc_runtime.config import MemoryConfig


class StreamingSkyMask:
    """Lightweight sky-like region suppression inspired by LingBot-Map's sky masking step."""

    def __init__(self, config: MemoryConfig):
        self.config = config

    def build(
        self,
        bgr: np.ndarray,
        gray: np.ndarray,
        depth: np.ndarray,
        texture: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        h, _ = depth.shape
        yy = np.arange(h, dtype=np.float32)[:, None] / max(1.0, float(h - 1))
        brightness = gray.astype(np.float32)
        far_depth = min(
            max(self.config.gaussian_sky_far_depth_m, 0.45 * self.config.gaussian_stereo_max_depth_m),
            self.config.gaussian_stereo_max_depth_m,
        )
        invalid_or_far = (depth <= 0.05) | (depth >= far_depth)
        low_texture = texture <= self.config.gaussian_sky_gradient_threshold
        low_confidence = confidence <= max(0.08, self.config.gaussian_region_low_confidence_threshold)

        # Prefer top-of-image, bright, low-texture regions with no reliable depth.
        sky_mask = (
            (yy <= self.config.gaussian_region_sky_top_ratio)
            & invalid_or_far
            & low_texture
            & low_confidence
            & (brightness >= self.config.gaussian_sky_brightness_threshold)
        )

        # A slightly looser band above the horizon catches large blank sky patches.
        upper_band = (
            (yy <= min(0.62, self.config.gaussian_region_sky_top_ratio + 0.14))
            & (depth <= 0.05)
            & (brightness >= self.config.gaussian_sky_brightness_threshold + 8.0)
            & (texture <= 0.75 * self.config.gaussian_sky_gradient_threshold)
        )

        drop = (sky_mask | upper_band).astype(np.uint8) * 255
        drop = cv2.morphologyEx(drop, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        drop = cv2.morphologyEx(drop, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        return (drop == 0).astype(bool)
