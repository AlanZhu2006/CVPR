from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


NTU_BODY_TO_PRISM_M = np.array([-0.293656, -0.012288, -0.273095], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cuVSLAM and FAST-LIVO/FAST-LIVO2 against NTU VIRAL prism-position ground truth. "
            "The benchmark compensates estimates to the prism position and evaluates position-only errors."
        )
    )
    parser.add_argument("--gt", required=True, help="Ground-truth trajectory in TUM format.")
    parser.add_argument("--cuvslam", required=True, help="cuVSLAM trajectory in TUM format.")
    parser.add_argument(
        "--fastlivo",
        default="",
        help="FAST-LIVO/FAST-LIVO2 trajectory in TUM format. Leave empty to benchmark cuVSLAM only.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for benchmark artifacts.")
    parser.add_argument(
        "--camera-body-yaml",
        default="",
        help=(
            "Optional NTU-style camera YAML that contains T_Body2Cam. "
            "Needed to convert cuVSLAM's left-camera poses into body/prism positions."
        ),
    )
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=0.05,
        help="Max timestamp difference in seconds when associating estimate timestamps to GT samples.",
    )
    parser.add_argument(
        "--rpe-delta",
        type=int,
        default=5,
        help="Frame delta for translational relative position error.",
    )
    parser.add_argument(
        "--no-umeyama",
        action="store_true",
        help="Disable similarity Umeyama alignment and only use first-pose normalization.",
    )
    return parser.parse_args()


def load_tum(path: Path) -> list[dict]:
    poses: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) != 8:
            raise ValueError(f"TUM trajectory line must have 8 values: {line}")
        timestamp, tx, ty, tz, qx, qy, qz, qw = vals
        poses.append(
            {
                "timestamp": timestamp,
                "t": np.array([tx, ty, tz], dtype=np.float64),
                "q": np.array([qx, qy, qz, qw], dtype=np.float64),
            }
        )
    if not poses:
        raise ValueError(f"No poses loaded from {path}")
    return poses


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = q / max(np.linalg.norm(q), 1e-12)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    m = R
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def pose_to_matrix(entry: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(entry["q"])
    T[:3, 3] = entry["t"]
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def normalize_first_pose_positions(points: np.ndarray) -> np.ndarray:
    return points - points[0]


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


def apply_similarity_to_points(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (R @ points.T)).T + t


def parse_opencv_matrix(path: Path, key: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    pattern = rf"{re.escape(key)}:\s*!!opencv-matrix\s*rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*\w+\s*data:\s*\[([^\]]+)\]"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find {key} in {path}")
    rows = int(match.group(1))
    cols = int(match.group(2))
    values = [float(v.strip()) for v in match.group(3).replace("\n", " ").split(",") if v.strip()]
    return np.array(values, dtype=np.float64).reshape(rows, cols)


def positions_to_tum_rows(timestamps: np.ndarray, positions: np.ndarray) -> list[str]:
    rows: list[str] = []
    for ts, p in zip(timestamps, positions):
        rows.append(f"{ts:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} 0.000000000 0.000000000 0.000000000 1.000000000")
    return rows


def extract_gt_positions(gt_raw: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.array([p["timestamp"] for p in gt_raw], dtype=np.float64)
    positions = np.stack([p["t"] for p in gt_raw], axis=0)
    return timestamps, positions


def interpolate_gt_positions(gt_ts: np.ndarray, gt_pos: np.ndarray, query_ts: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    matched_ts: list[float] = []
    matched_pos: list[np.ndarray] = []
    for t in query_ts:
        idx = np.searchsorted(gt_ts, t)
        if idx == 0 or idx >= len(gt_ts):
            continue
        t0, t1 = gt_ts[idx - 1], gt_ts[idx]
        if min(abs(t - t0), abs(t - t1)) > tolerance and not (t0 <= t <= t1):
            continue
        alpha = 0.0 if abs(t1 - t0) < 1e-12 else (t - t0) / (t1 - t0)
        p = (1.0 - alpha) * gt_pos[idx - 1] + alpha * gt_pos[idx]
        matched_ts.append(float(t))
        matched_pos.append(p.astype(np.float64))
    if len(matched_ts) < 2:
        raise ValueError("Too few matched/interpolated GT positions.")
    return np.array(matched_ts, dtype=np.float64), np.stack(matched_pos, axis=0)


def convert_estimate_to_prism_positions(
    est_raw: list[dict],
    *,
    mode: str,
    T_body_to_cam: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.array([p["timestamp"] for p in est_raw], dtype=np.float64)
    prism_positions: list[np.ndarray] = []
    for entry in est_raw:
        T_world_sensor = pose_to_matrix(entry)
        if mode == "cuvslam":
            if T_body_to_cam is None:
                raise ValueError("cuVSLAM mode requires T_body_to_cam")
            T_sensor_to_body = invert_pose(T_body_to_cam)
            T_world_body = T_world_sensor @ T_sensor_to_body
        elif mode == "fastlivo":
            T_world_body = T_world_sensor
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        prism_world = T_world_body[:3, :3] @ NTU_BODY_TO_PRISM_M + T_world_body[:3, 3]
        prism_positions.append(prism_world.astype(np.float64))
    return timestamps, np.stack(prism_positions, axis=0)


def compute_position_metrics(gt_pos: np.ndarray, est_pos: np.ndarray, rpe_delta: int) -> dict:
    trans_errors = np.linalg.norm(gt_pos - est_pos, axis=1)
    rel_gt = gt_pos[rpe_delta:] - gt_pos[:-rpe_delta]
    rel_est = est_pos[rpe_delta:] - est_pos[:-rpe_delta]
    rel_err = np.linalg.norm(rel_gt - rel_est, axis=1) if len(gt_pos) > rpe_delta else np.array([], dtype=np.float64)
    return {
        "matched_frames": int(len(gt_pos)),
        "ate_rmse": float(np.sqrt(np.mean(np.square(trans_errors)))),
        "ate_mean": float(np.mean(trans_errors)),
        "ate_median": float(np.median(trans_errors)),
        "ate_max": float(np.max(trans_errors)),
        "rpe_trans_rmse": float(np.sqrt(np.mean(np.square(rel_err)))) if rel_err.size else None,
        "rpe_rot_rmse_rad": None,
        "trajectory_length_m": float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))),
        "trans_errors": [float(x) for x in trans_errors.tolist()],
    }


def write_csv(path: Path, timestamps: np.ndarray, cu_errors: list[float], fl_errors: list[float] | None) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if fl_errors is None:
            writer.writerow(["timestamp", "cuvslam_trans_error_m"])
            for ts, cu_err in zip(timestamps, cu_errors):
                writer.writerow([f"{ts:.6f}", f"{cu_err:.6f}"])
        else:
            writer.writerow(["timestamp", "cuvslam_trans_error_m", "fastlivo_trans_error_m"])
            for ts, cu_err, fl_err in zip(timestamps, cu_errors, fl_errors):
                writer.writerow([f"{ts:.6f}", f"{cu_err:.6f}", f"{fl_err:.6f}"])


def maybe_plot(path: Path, gt_xyz: np.ndarray, cu_xyz: np.ndarray, fl_xyz: np.ndarray | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], label="GT prism", linewidth=2)
    ax.plot(cu_xyz[:, 0], cu_xyz[:, 1], label="cuVSLAM prism", linewidth=1.5)
    if fl_xyz is not None:
        ax.plot(fl_xyz[:, 0], fl_xyz[:, 1], label="FAST-LIVO2 prism", linewidth=1.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top-down aligned prism-position comparison")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=160)
    except Exception:
        pass
    finally:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_raw = load_tum(Path(args.gt).expanduser().resolve())
    cu_raw = load_tum(Path(args.cuvslam).expanduser().resolve())
    fl_raw = load_tum(Path(args.fastlivo).expanduser().resolve()) if args.fastlivo else None

    T_body_to_cam = None
    if args.camera_body_yaml:
        T_body_to_cam = parse_opencv_matrix(Path(args.camera_body_yaml).expanduser().resolve(), "T_Body2Cam")

    gt_ts, gt_pos = extract_gt_positions(gt_raw)

    cu_ts_raw, cu_prism_raw = convert_estimate_to_prism_positions(cu_raw, mode="cuvslam", T_body_to_cam=T_body_to_cam)
    cu_ts, gt_cu = interpolate_gt_positions(gt_ts, gt_pos, cu_ts_raw, args.time_tolerance)
    keep_mask_cu = np.isin(np.round(cu_ts_raw, 9), np.round(cu_ts, 9))
    cu_est = cu_prism_raw[keep_mask_cu]

    # First-pose normalize before optional similarity alignment.
    gt_cu_norm = normalize_first_pose_positions(gt_cu)
    cu_est_norm = normalize_first_pose_positions(cu_est)
    if args.no_umeyama:
        cu_aligned = cu_est_norm
        cu_align = {"scale": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}
    else:
        scale, R, t = similarity_align_umeyama(cu_est_norm, gt_cu_norm)
        cu_aligned = apply_similarity_to_points(cu_est_norm, scale, R, t)
        cu_align = {"scale": float(scale), "R": R.tolist(), "t": t.tolist()}

    fl_metrics = None
    fl_aligned = None
    fl_plot = None
    common_frames = len(cu_ts)
    if fl_raw is not None:
        fl_ts_raw, fl_prism_raw = convert_estimate_to_prism_positions(fl_raw, mode="fastlivo", T_body_to_cam=None)
        fl_ts, gt_fl = interpolate_gt_positions(gt_ts, gt_pos, fl_ts_raw, args.time_tolerance)
        keep_mask_fl = np.isin(np.round(fl_ts_raw, 9), np.round(fl_ts, 9))
        fl_est = fl_prism_raw[keep_mask_fl]
        gt_fl_norm = normalize_first_pose_positions(gt_fl)
        fl_est_norm = normalize_first_pose_positions(fl_est)
        if args.no_umeyama:
            fl_aligned = fl_est_norm
            fl_align = {"scale": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}
        else:
            scale_fl, R_fl, t_fl = similarity_align_umeyama(fl_est_norm, gt_fl_norm)
            fl_aligned = apply_similarity_to_points(fl_est_norm, scale_fl, R_fl, t_fl)
            fl_align = {"scale": float(scale_fl), "R": R_fl.tolist(), "t": t_fl.tolist()}
        fl_metrics = compute_position_metrics(gt_fl_norm, fl_aligned, args.rpe_delta)
        fl_metrics["alignment_transform"] = fl_align
        fl_plot = (fl_ts, gt_fl_norm, fl_aligned)
        common_frames = min(common_frames, len(fl_ts))

    cu_metrics = compute_position_metrics(gt_cu_norm, cu_aligned, args.rpe_delta)
    cu_metrics["alignment_transform"] = cu_align

    summary = {
        "dataset": "NTU VIRAL",
        "sequence": "spms_02",
        "alignment": "prism_compensated + first_position + similarity_umeyama_pos_only"
        if not args.no_umeyama
        else "prism_compensated + first_position",
        "ground_truth_mode": "position_only_prism",
        "prism_offset_m_body_frame": NTU_BODY_TO_PRISM_M.tolist(),
        "camera_body_yaml": str(Path(args.camera_body_yaml).expanduser().resolve()) if args.camera_body_yaml else None,
        "time_tolerance_sec": float(args.time_tolerance),
        "rpe_delta": int(args.rpe_delta),
        "common_matched_frames": int(common_frames),
        "cuvslam": cu_metrics,
    }
    if fl_metrics is not None:
        summary["fastlivo"] = fl_metrics
        summary["winner_by_ate_rmse"] = "fastlivo" if fl_metrics["ate_rmse"] < cu_metrics["ate_rmse"] else "cuvslam"

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        output_dir / "per_frame_translation_error.csv",
        cu_ts,
        cu_metrics["trans_errors"],
        fl_metrics["trans_errors"] if fl_metrics is not None else None,
    )
    maybe_plot(output_dir / "trajectory_topdown.png", gt_cu_norm, cu_aligned, fl_aligned)

    (output_dir / "gt_aligned_cuvslam.tum").write_text(
        "\n".join(positions_to_tum_rows(cu_ts, gt_cu_norm)) + "\n", encoding="utf-8"
    )
    (output_dir / "cuvslam_aligned.tum").write_text(
        "\n".join(positions_to_tum_rows(cu_ts, cu_aligned)) + "\n", encoding="utf-8"
    )
    if fl_plot is not None and fl_aligned is not None:
        fl_ts, gt_fl_norm, _ = fl_plot
        (output_dir / "gt_aligned_fastlivo.tum").write_text(
            "\n".join(positions_to_tum_rows(fl_ts, gt_fl_norm)) + "\n", encoding="utf-8"
        )
        (output_dir / "fastlivo_aligned.tum").write_text(
            "\n".join(positions_to_tum_rows(fl_ts, fl_aligned)) + "\n", encoding="utf-8"
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
