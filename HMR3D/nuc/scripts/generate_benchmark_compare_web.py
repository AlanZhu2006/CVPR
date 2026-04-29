from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a lightweight benchmark HTML compare page from TUM trajectories and benchmark summary."
    )
    parser.add_argument("--summary-json", required=True, help="Benchmark summary.json path.")
    parser.add_argument("--gt", required=True, help="Ground-truth TUM trajectory.")
    parser.add_argument("--cuvslam", required=True, help="cuVSLAM TUM trajectory.")
    parser.add_argument("--fastlivo", default="", help="Optional FAST-LIVO/FAST-LIVO2 TUM trajectory.")
    parser.add_argument("--output-html", required=True, help="Output HTML path.")
    parser.add_argument("--title", default="Trajectory Benchmark")
    parser.add_argument("--subtitle", default="")
    parser.add_argument(
        "--plot-axes",
        default="xz",
        choices=["xy", "xz", "yz"],
        help="TUM translation axes for the 2D plot. KITTI/OXTS top-down usually needs xy; camera-forward datasets often use xz.",
    )
    return parser.parse_args()


def load_tum(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows, dtype=np.float64)


def polyline_points(xz: np.ndarray, mins: np.ndarray, maxs: np.ndarray, width: int, height: int, pad: int = 24) -> str:
    span = np.maximum(maxs - mins, 1e-6)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    pts = []
    for x, z in xz:
        px = pad + (x - mins[0]) * scale
        py = height - pad - (z - mins[1]) * scale
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gt_path = Path(args.gt).expanduser().resolve()
    cu_path = Path(args.cuvslam).expanduser().resolve()
    fl_path = Path(args.fastlivo).expanduser().resolve() if args.fastlivo else None
    base_dir = summary_path.parent

    gt_aligned_cu = base_dir / "gt_matched_for_cuvslam.tum"
    cu_aligned = base_dir / "cuvslam_aligned.tum"
    gt = load_tum(gt_aligned_cu if gt_aligned_cu.exists() else gt_path)
    cu = load_tum(cu_aligned if cu_aligned.exists() else cu_path)

    fl = None
    if fl_path is not None:
        fl_aligned = base_dir / "fastlivo_aligned.tum"
        fl = load_tum(fl_aligned if fl_aligned.exists() else fl_path)

    width, height = 980, 380
    axis_cols = {
        "xy": (1, 2),
        "xz": (1, 3),
        "yz": (2, 3),
    }[args.plot_axes]
    xz_arrays = [gt[:, axis_cols], cu[:, axis_cols]]
    if fl is not None:
        xz_arrays.append(fl[:, axis_cols])
    all_xz = np.concatenate(xz_arrays, axis=0)
    mins = all_xz.min(axis=0)
    maxs = all_xz.max(axis=0)
    gt_pts = polyline_points(gt[:, axis_cols], mins, maxs, width, height)
    cu_pts = polyline_points(cu[:, axis_cols], mins, maxs, width, height)
    fl_pts = polyline_points(fl[:, axis_cols], mins, maxs, width, height) if fl is not None else ""

    cu_metrics = summary["cuvslam"]
    fl_metrics = summary.get("fastlivo")

    fastlivo_block = ""
    if fl_metrics is not None:
        fastlivo_block = f"""
        <div class="card">
          <h3>FAST-LIVO2</h3>
          <div class="metric">ATE RMSE: {fl_metrics['ate_rmse']:.4f} m</div>
          <div class="metric">ATE Mean: {fl_metrics['ate_mean']:.4f} m</div>
          <div class="metric">ATE Median: {fl_metrics['ate_median']:.4f} m</div>
          <div class="metric">RPE Trans RMSE: {fl_metrics['rpe_trans_rmse']:.4f} m</div>
          <div class="metric">RPE Rot RMSE: {"n/a" if fl_metrics['rpe_rot_rmse_rad'] is None else f"{fl_metrics['rpe_rot_rmse_rad']:.4f} rad"}</div>
          <div class="metric">Matched Frames: {fl_metrics['matched_frames']}</div>
        </div>
        """

    fastlivo_svg = ""
    if fl is not None:
        fastlivo_svg = f'<polyline points="{fl_pts}" fill="none" stroke="#eab308" stroke-width="2.2" />'

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Compare</title>
  <style>
    body {{
      margin: 0;
      background: #0b1220;
      color: #e5eefc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p.sub {{ margin: 0 0 24px; color: #93a3b8; }}
    .grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; }}
    .panel {{
      background: #111a2b;
      border: 1px solid #22314d;
      border-radius: 18px;
      padding: 18px;
    }}
    .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .card {{
      background: #0d1524;
      border: 1px solid #1f2b42;
      border-radius: 14px;
      padding: 16px;
    }}
    .metric {{ margin: 8px 0; font-size: 15px; color: #dbe7ff; }}
    .legend {{ display: flex; gap: 14px; margin-top: 10px; color: #aebcd6; font-size: 14px; }}
    .dot {{ width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px; }}
    code {{ background: #09101c; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{args.title}</h1>
    <p class="sub">{args.subtitle or "GT, cuVSLAM, and optional FAST-LIVO2 trajectories are aligned and compared in one shared top-down coordinate frame."}</p>
    <div class="grid">
      <div class="panel">
        <h2 style="margin-top:0">Top-down Trajectory Compare</h2>
        <div style="margin:-6px 0 10px;color:#93a3b8;font-size:14px">Plot axes: <code>{args.plot_axes}</code></div>
        <svg viewBox="0 0 {width} {height}" width="100%" height="auto" style="background:#08101b;border-radius:12px">
          <polyline points="{gt_pts}" fill="none" stroke="#f8fafc" stroke-width="2.8" />
          <polyline points="{cu_pts}" fill="none" stroke="#38bdf8" stroke-width="2.2" />
          {fastlivo_svg}
        </svg>
        <div class="legend">
          <div><span class="dot" style="background:#f8fafc"></span>GT</div>
          <div><span class="dot" style="background:#38bdf8"></span>cuVSLAM</div>
          {"<div><span class='dot' style='background:#eab308'></span>FAST-LIVO2</div>" if fl is not None else "<div><span class='dot' style='background:#64748b'></span>FAST-LIVO2 pending</div>"}
        </div>
      </div>
      <div class="panel">
        <h2 style="margin-top:0">Metrics</h2>
        <div class="cards">
          <div class="card">
            <h3>cuVSLAM</h3>
            <div class="metric">ATE RMSE: {cu_metrics['ate_rmse']:.4f} m</div>
            <div class="metric">ATE Mean: {cu_metrics['ate_mean']:.4f} m</div>
            <div class="metric">ATE Median: {cu_metrics['ate_median']:.4f} m</div>
            <div class="metric">RPE Trans RMSE: {cu_metrics['rpe_trans_rmse']:.4f} m</div>
            <div class="metric">RPE Rot RMSE: {"n/a" if cu_metrics['rpe_rot_rmse_rad'] is None else f"{cu_metrics['rpe_rot_rmse_rad']:.4f} rad"}</div>
            <div class="metric">Matched Frames: {cu_metrics['matched_frames']}</div>
          </div>
          {fastlivo_block if fastlivo_block else '<div class="card"><h3>FAST-LIVO2</h3><div class="metric">Pending ROS1-capable runtime environment.</div><div class="metric">Once trajectory is available, this page will become a full two-method compare.</div></div>'}
        </div>
        <div class="card" style="margin-top:14px">
          <h3 style="margin-top:0">Run Context</h3>
          <div class="metric">Sequence: <code>{Path(summary['gt_path']).parent.parent.name}</code></div>
          <div class="metric">Alignment: <code>{summary['alignment']}</code></div>
          <div class="metric">GT mode: <code>{summary.get('ground_truth_mode', 'full_pose')}</code></div>
          <div class="metric">Matched Frames: <code>{summary['common_matched_frames']}</code></div>
          <div class="metric">Time tolerance: <code>{summary['time_tolerance_sec']}</code> s</div>
          <div class="metric">RPE delta: <code>{summary['rpe_delta']}</code></div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    print(output_html)


if __name__ == "__main__":
    main()
