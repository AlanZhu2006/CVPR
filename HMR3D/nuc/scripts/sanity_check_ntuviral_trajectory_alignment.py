from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep global time shifts for NTU VIRAL trajectories to sanity-check whether large "
            "trajectory errors are dominated by timestamp mismatch or true geometric drift."
        )
    )
    parser.add_argument("--gt", required=True, help="Ground-truth TUM trajectory.")
    parser.add_argument("--estimate", required=True, help="Estimated TUM trajectory.")
    parser.add_argument("--label", required=True, help="Method label stored in the output.")
    parser.add_argument("--output-json", required=True, help="Output JSON summary path.")
    parser.add_argument("--time-tolerance", type=float, default=0.05, help="GT interpolation tolerance in seconds.")
    parser.add_argument("--shift-min", type=float, default=-120.0, help="Minimum global time shift to test.")
    parser.add_argument("--shift-max", type=float, default=120.0, help="Maximum global time shift to test.")
    parser.add_argument("--shift-step", type=float, default=2.0, help="Global shift step size in seconds.")
    parser.add_argument(
        "--local-window-sec",
        type=float,
        default=20.0,
        help="Short local window used to separate systematic offset from long-term drift.",
    )
    return parser.parse_args()


def load_tum_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) != 8:
            raise ValueError(f"Expected 8 columns in TUM trajectory, got: {line}")
        timestamps.append(vals[0])
        positions.append(np.array(vals[1:4], dtype=np.float64))
    if not timestamps:
        raise ValueError(f"No poses loaded from {path}")
    return np.array(timestamps, dtype=np.float64), np.stack(positions, axis=0)


def interpolate_gt_positions(
    gt_ts: np.ndarray,
    gt_pos: np.ndarray,
    query_ts: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    matched_ts: list[float] = []
    matched_pos: list[np.ndarray] = []
    matched_indices: list[int] = []
    for idx, t in enumerate(query_ts):
        right = np.searchsorted(gt_ts, t)
        if right == 0 or right >= len(gt_ts):
            continue
        t0, t1 = gt_ts[right - 1], gt_ts[right]
        if min(abs(t - t0), abs(t - t1)) > tolerance and not (t0 <= t <= t1):
            continue
        alpha = 0.0 if abs(t1 - t0) < 1e-12 else (t - t0) / (t1 - t0)
        p = (1.0 - alpha) * gt_pos[right - 1] + alpha * gt_pos[right]
        matched_ts.append(float(t))
        matched_pos.append(p.astype(np.float64))
        matched_indices.append(idx)
    if len(matched_ts) < 10:
        return None, None, None
    return (
        np.array(matched_ts, dtype=np.float64),
        np.stack(matched_pos, axis=0),
        np.array(matched_indices, dtype=np.int64),
    )


def similarity_align_umeyama(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src_xyz.mean(axis=0)
    dst_mean = dst_xyz.mean(axis=0)
    src_centered = src_xyz - src_mean
    dst_centered = dst_xyz - dst_mean
    cov = src_centered.T @ dst_centered / len(src_xyz)
    U, singular_values, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.sum(singular_values) / src_var) if src_var > 1e-12 else 1.0
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def apply_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (R @ points.T)).T + t


def compute_rmse(gt_pos: np.ndarray, est_pos: np.ndarray) -> float:
    error = np.linalg.norm(gt_pos - est_pos, axis=1)
    return float(np.sqrt(np.mean(np.square(error))))


def main() -> None:
    args = parse_args()
    gt_ts, gt_pos = load_tum_positions(Path(args.gt).expanduser().resolve())
    est_ts, est_pos = load_tum_positions(Path(args.estimate).expanduser().resolve())

    trials: list[dict] = []
    best_trial: dict | None = None
    for shift in np.arange(args.shift_min, args.shift_max + 1e-9, args.shift_step):
        shifted_ts = est_ts + shift
        matched_ts, matched_gt, matched_idx = interpolate_gt_positions(gt_ts, gt_pos, shifted_ts, args.time_tolerance)
        if matched_ts is None or matched_gt is None or matched_idx is None:
            continue
        matched_est = est_pos[matched_idx]
        gt_norm = matched_gt - matched_gt[0]
        est_norm = matched_est - matched_est[0]
        scale, R, t = similarity_align_umeyama(est_norm, gt_norm)
        aligned = apply_similarity(est_norm, scale, R, t)
        ate_rmse = compute_rmse(gt_norm, aligned)
        local_mask = (matched_ts - matched_ts[0]) <= args.local_window_sec
        local_rmse = compute_rmse(gt_norm[local_mask], aligned[local_mask]) if int(local_mask.sum()) >= 10 else None
        trial = {
            "shift_sec": float(shift),
            "matched_frames": int(len(matched_ts)),
            "ate_rmse_m": float(ate_rmse),
            "local_window_sec": float(args.local_window_sec),
            "local_rmse_m": None if local_rmse is None else float(local_rmse),
        }
        trials.append(trial)
        if best_trial is None or ate_rmse < best_trial["ate_rmse_m"]:
            best_trial = trial

    trials.sort(key=lambda item: item["ate_rmse_m"])
    output = {
        "label": args.label,
        "estimate_path": str(Path(args.estimate).expanduser().resolve()),
        "ground_truth_path": str(Path(args.gt).expanduser().resolve()),
        "time_tolerance_sec": float(args.time_tolerance),
        "search_range_sec": [float(args.shift_min), float(args.shift_max)],
        "search_step_sec": float(args.shift_step),
        "top_trials": trials[:20],
        "best_trial": best_trial,
        "zero_shift_trial": next((trial for trial in trials if abs(trial["shift_sec"]) < 1e-12), None),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
