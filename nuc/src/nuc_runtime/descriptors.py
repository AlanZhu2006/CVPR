from __future__ import annotations

import cv2
import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def compute_global_descriptor(
    frame_bgr: np.ndarray,
    orb_descriptors: np.ndarray | None,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256]).flatten()
    hsv_hist = normalize_vector(hsv_hist.astype(np.float32))

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    gray_hist = normalize_vector(gray_hist.astype(np.float32))

    if orb_descriptors is not None and len(orb_descriptors) > 0:
        orb_stats = orb_descriptors.astype(np.float32).mean(axis=0)
        orb_stats = normalize_vector(orb_stats)
    else:
        orb_stats = np.zeros(32, dtype=np.float32)

    return normalize_vector(np.concatenate([hsv_hist, gray_hist, orb_stats], axis=0))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(normalize_vector(left), normalize_vector(right)))
