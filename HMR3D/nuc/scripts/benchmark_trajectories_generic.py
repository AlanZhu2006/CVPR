from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic TUM trajectory benchmark for GT / cuVSLAM / FAST-LIVO2 style comparisons. "
            "Designed for KITTI raw or any standard 6DoF trajectory pair without NTU-specific prism logic."
        )
    )
    parser.add_argument("--gt", required=True, help="Ground-truth trajectory in TUM format.")
    parser.add_argument("--cuvslam", required=True, help="cuVSLAM trajectory in TUM format.")
    parser.add_argument("--fastlivo", default="", help="Optional FAST-LIVO/FAST-LIVO2 trajectory in TUM format.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--time-tolerance", type=float, default=0.02, help="Association tolerance in seconds.")
    parser.add_argument("--rpe-delta", type=int, default=5, help="Frame delta for RPE.")
    parser.add_argument(
        "--alignment",
        choices=["first_pose", "rigid_umeyama", "similarity_umeyama"],
        default="rigid_umeyama",
        help="Alignment mode before evaluation.",
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
            raise ValueError(f"Expected 8 TUM values per line in {path}: {line}")
        ts, tx, ty, tz, qx, qy, qz, qw = vals
        poses.append(
            {
                "timestamp": ts,
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
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return q


def pose_to_matrix(entry: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(entry["q"])
    T[:3, 3] = entry["t"]
    return T


def matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return T[:3, 3].copy(), rot_to_quat(T[:3, :3])


def invert_pose(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def associate(gt: list[dict], est: list[dict], tolerance: float) -> tuple[np.ndarray, list[dict], list[dict]]:
    gt_ts = np.array([p["timestamp"] for p in gt], dtype=np.float64)
    matched_ts: list[float] = []
    gt_matched: list[dict] = []
    est_matched: list[dict] = []
    for est_pose in est:
        t = est_pose["timestamp"]
        idx = int(np.argmin(np.abs(gt_ts - t)))
        if abs(gt_ts[idx] - t) <= tolerance:
            matched_ts.append(float(t))
            gt_matched.append(gt[idx])
            est_matched.append(est_pose)
    if len(matched_ts) < 2:
        raise ValueError("Too few matched timestamps for evaluation.")
    return np.array(matched_ts, dtype=np.float64), gt_matched, est_matched


def umeyama(src_xyz: np.ndarray, dst_xyz: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
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
    scale = float(np.sum(singular_values) / src_var) if with_scale and src_var > 1e-12 else 1.0
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def align_estimates(gt_matched: list[dict], est_matched: list[dict], mode: str) -> tuple[np.ndarray, list[np.ndarray]]:
    gt_mats = [pose_to_matrix(p) for p in gt_matched]
    est_mats = [pose_to_matrix(p) for p in est_matched]

    if mode == "first_pose":
        T_align = gt_mats[0] @ invert_pose(est_mats[0])
        aligned = [T_align @ T for T in est_mats]
        return T_align, aligned

    gt_xyz = np.stack([T[:3, 3] for T in gt_mats], axis=0)
    est_xyz = np.stack([T[:3, 3] for T in est_mats], axis=0)
    scale, R, t = umeyama(est_xyz, gt_xyz, with_scale=(mode == "similarity_umeyama"))
    T_align = np.eye(4, dtype=np.float64)
    T_align[:3, :3] = R
    T_align[:3, 3] = t
    aligned: list[np.ndarray] = []
    for T in est_mats:
        Tout = np.eye(4, dtype=np.float64)
        Tout[:3, :3] = R @ T[:3, :3]
        Tout[:3, 3] = scale * (R @ T[:3, 3]) + t
        aligned.append(Tout)
    return T_align, aligned


def relative_transform(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    return invert_pose(T1) @ T2


def rotation_angle(R: np.ndarray) -> float:
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(tr))


def compute_metrics(gt_mats: list[np.ndarray], est_mats: list[np.ndarray], rpe_delta: int) -> dict:
    gt_xyz = np.stack([T[:3, 3] for T in gt_mats], axis=0)
    est_xyz = np.stack([T[:3, 3] for T in est_mats], axis=0)
    trans_errors = np.linalg.norm(gt_xyz - est_xyz, axis=1)

    rpe_trans: list[float] = []
    rpe_rot: list[float] = []
    if len(gt_mats) > rpe_delta:
        for i in range(len(gt_mats) - rpe_delta):
            d_gt = relative_transform(gt_mats[i], gt_mats[i + rpe_delta])
            d_est = relative_transform(est_mats[i], est_mats[i + rpe_delta])
            d_err = invert_pose(d_gt) @ d_est
            rpe_trans.append(float(np.linalg.norm(d_err[:3, 3])))
            rpe_rot.append(rotation_angle(d_err[:3, :3]))
    rel_len = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)
    return {
        "matched_frames": int(len(gt_mats)),
        "ate_rmse": float(np.sqrt(np.mean(np.square(trans_errors)))),
        "ate_mean": float(np.mean(trans_errors)),
        "ate_median": float(np.median(trans_errors)),
        "ate_max": float(np.max(trans_errors)),
        "rpe_trans_rmse": float(np.sqrt(np.mean(np.square(rpe_trans)))) if rpe_trans else None,
        "rpe_rot_rmse_rad": float(np.sqrt(np.mean(np.square(rpe_rot)))) if rpe_rot else None,
        "trajectory_length_m": float(np.sum(rel_len)),
        "trans_errors": [float(x) for x in trans_errors.tolist()],
    }


def write_tum(path: Path, timestamps: np.ndarray, mats: list[np.ndarray]) -> None:
    rows: list[str] = []
    for ts, T in zip(timestamps, mats):
        t, q = matrix_to_pose(T)
        rows.append(
            f"{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def maybe_plot(path: Path, gt_mats: list[np.ndarray], cu_mats: list[np.ndarray], fl_mats: list[np.ndarray] | None) -> None:
    try:
        # Some Jetson Python environments have a NumPy/matplotlib ABI mismatch.
        # Keep the benchmark metrics usable even when optional plotting is broken.
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] skipping trajectory plot because matplotlib is unavailable: {exc}")
        return
    gt_xyz = np.stack([T[:3, 3] for T in gt_mats], axis=0)
    cu_xyz = np.stack([T[:3, 3] for T in cu_mats], axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="GT", linewidth=2)
    ax.plot(cu_xyz[:, 0], cu_xyz[:, 2], label="cuVSLAM", linewidth=1.5)
    if fl_mats is not None:
        fl_xyz = np.stack([T[:3, 3] for T in fl_mats], axis=0)
        ax.plot(fl_xyz[:, 0], fl_xyz[:, 2], label="FAST-LIVO2", linewidth=1.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Top-down trajectory comparison")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, timestamps: np.ndarray, cu_errors: list[float], fl_errors: list[float] | None) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if fl_errors is None:
            writer.writerow(["timestamp", "cuvslam_trans_error_m"])
            for ts, cu in zip(timestamps, cu_errors):
                writer.writerow([f"{ts:.6f}", f"{cu:.6f}"])
        else:
            writer.writerow(["timestamp", "cuvslam_trans_error_m", "fastlivo_trans_error_m"])
            for ts, cu, fl in zip(timestamps, cu_errors, fl_errors):
                writer.writerow([f"{ts:.6f}", f"{cu:.6f}", f"{fl:.6f}"])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_tum(Path(args.gt).expanduser().resolve())
    cu = load_tum(Path(args.cuvslam).expanduser().resolve())
    fl = load_tum(Path(args.fastlivo).expanduser().resolve()) if args.fastlivo else None

    timestamps_cu, gt_cu_raw, cu_raw = associate(gt, cu, args.time_tolerance)
    T_align_cu, cu_aligned = align_estimates(gt_cu_raw, cu_raw, args.alignment)
    gt_cu_mats = [pose_to_matrix(p) for p in gt_cu_raw]
    cu_metrics = compute_metrics(gt_cu_mats, cu_aligned, args.rpe_delta)
    write_tum(output_dir / "gt_matched_for_cuvslam.tum", timestamps_cu, gt_cu_mats)
    write_tum(output_dir / "cuvslam_aligned.tum", timestamps_cu, cu_aligned)

    fl_metrics = None
    fl_aligned = None
    gt_fl_mats = None
    timestamps_fl = None
    if fl is not None:
        timestamps_fl, gt_fl_raw, fl_raw = associate(gt, fl, args.time_tolerance)
        _, fl_aligned = align_estimates(gt_fl_raw, fl_raw, args.alignment)
        gt_fl_mats = [pose_to_matrix(p) for p in gt_fl_raw]
        fl_metrics = compute_metrics(gt_fl_mats, fl_aligned, args.rpe_delta)
        write_tum(output_dir / "gt_matched_for_fastlivo.tum", timestamps_fl, gt_fl_mats)
        write_tum(output_dir / "fastlivo_aligned.tum", timestamps_fl, fl_aligned)

    summary = {
        "alignment": args.alignment,
        "time_tolerance_sec": args.time_tolerance,
        "rpe_delta": args.rpe_delta,
        "gt_path": str(Path(args.gt).expanduser().resolve()),
        "cuvslam_path": str(Path(args.cuvslam).expanduser().resolve()),
        "fastlivo_path": str(Path(args.fastlivo).expanduser().resolve()) if args.fastlivo else None,
        "cuvslam": cu_metrics,
        "fastlivo": fl_metrics,
        "winner_by_ate_rmse": (
            "cuvslam"
            if fl_metrics is not None and cu_metrics["ate_rmse"] < fl_metrics["ate_rmse"]
            else ("fastlivo" if fl_metrics is not None else "cuvslam")
        ),
        "common_matched_frames": int(min(cu_metrics["matched_frames"], fl_metrics["matched_frames"])) if fl_metrics else int(cu_metrics["matched_frames"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(
        output_dir / "per_frame_translation_error.csv",
        timestamps_cu if fl_metrics is None else timestamps_cu[: min(len(timestamps_cu), len(timestamps_fl))],
        cu_metrics["trans_errors"][: min(len(cu_metrics["trans_errors"]), len(fl_metrics["trans_errors"]) if fl_metrics else len(cu_metrics["trans_errors"]))],
        fl_metrics["trans_errors"][: min(len(cu_metrics["trans_errors"]), len(fl_metrics["trans_errors"]))] if fl_metrics else None,
    )
    maybe_plot(output_dir / "trajectory_topdown.png", gt_cu_mats, cu_aligned, fl_aligned)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
