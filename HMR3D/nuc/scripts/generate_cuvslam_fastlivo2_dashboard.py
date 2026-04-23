from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a combined NTU VIRAL dashboard for trajectory and runtime comparison.")
    parser.add_argument("--traj-summary", required=True, help="Trajectory benchmark summary.json path.")
    parser.add_argument("--runtime-summary", required=True, help="Runtime benchmark summary.json path.")
    parser.add_argument("--output-html", required=True, help="Output HTML path.")
    return parser.parse_args()


def fmt(val: float | None, digits: int = 3, suffix: str = "") -> str:
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}{suffix}"


def metric_block(title: str, rows: list[tuple[str, str]]) -> str:
    inner = "\n".join(f'<div class="metric"><span>{k}</span><strong>{v}</strong></div>' for k, v in rows)
    return f'<div class="card"><h3>{title}</h3>{inner}</div>'


def main() -> None:
    args = parse_args()
    traj = json.loads(Path(args.traj_summary).expanduser().resolve().read_text(encoding="utf-8"))
    runtime = json.loads(Path(args.runtime_summary).expanduser().resolve().read_text(encoding="utf-8"))

    cu_traj = traj["cuvslam"]
    fl_traj = traj["fastlivo"]
    cu_run = runtime["cuvslam"]
    fl_run = runtime["fastlivo2"]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>cuVSLAM vs FAST-LIVO2 Dashboard</title>
  <style>
    body {{
      margin: 0;
      background: #0b1220;
      color: #e5eefc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p.sub {{ margin: 0 0 20px; color: #93a3b8; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }}
    .panel {{
      background: #111a2b;
      border: 1px solid #22314d;
      border-radius: 18px;
      padding: 18px;
    }}
    .cards {{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; }}
    .card {{
      background:#0d1524;
      border:1px solid #1f2b42;
      border-radius:14px;
      padding:16px;
    }}
    .metric {{
      display:flex;
      justify-content:space-between;
      gap:12px;
      margin:8px 0;
      color:#dbe7ff;
    }}
    code {{
      background:#09101c;
      padding:2px 6px;
      border-radius:6px;
    }}
    .winner {{
      font-size: 18px;
      color:#a7f3d0;
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>cuVSLAM vs FAST-LIVO2 on NTU VIRAL</h1>
    <p class="sub">Sequence: <code>{traj['sequence']}</code>. Trajectory benchmark uses <code>{traj['ground_truth_mode']}</code> GT with <code>{traj['alignment']}</code>. Runtime benchmark is measured on the same Jetson machine with <code>tegrastats</code> sampling.</p>

    <div class="panel" style="margin-bottom:18px">
      <h2 style="margin-top:0">Headline</h2>
      <div class="winner">Trajectory winner by ATE RMSE: <strong>{traj.get('winner_by_ate_rmse', 'n/a')}</strong></div>
      <div class="winner">Faster runtime path: <strong>{"fastlivo2" if (fl_run.get('wall_time_sec', 1e18) < cu_run.get('wall_time_sec', 1e18)) else "cuvslam"}</strong></div>
    </div>

    <div class="row">
      <div class="panel">
        <h2 style="margin-top:0">Trajectory Accuracy</h2>
        <div class="cards">
          {metric_block("cuVSLAM", [
              ("ATE RMSE", fmt(cu_traj.get("ate_rmse"), 3, " m")),
              ("ATE Mean", fmt(cu_traj.get("ate_mean"), 3, " m")),
              ("ATE Median", fmt(cu_traj.get("ate_median"), 3, " m")),
              ("RPE Trans RMSE", fmt(cu_traj.get("rpe_trans_rmse"), 3, " m")),
              ("Matched Frames", str(cu_traj.get("matched_frames"))),
              ("Trajectory Length", fmt(cu_traj.get("trajectory_length_m"), 2, " m")),
          ])}
          {metric_block("FAST-LIVO2", [
              ("ATE RMSE", fmt(fl_traj.get("ate_rmse"), 3, " m")),
              ("ATE Mean", fmt(fl_traj.get("ate_mean"), 3, " m")),
              ("ATE Median", fmt(fl_traj.get("ate_median"), 3, " m")),
              ("RPE Trans RMSE", fmt(fl_traj.get("rpe_trans_rmse"), 3, " m")),
              ("Matched Frames", str(fl_traj.get("matched_frames"))),
              ("Trajectory Length", fmt(fl_traj.get("trajectory_length_m"), 2, " m")),
          ])}
        </div>
      </div>
      <div class="panel">
        <h2 style="margin-top:0">Runtime & Resources</h2>
        <div class="cards">
          {metric_block("cuVSLAM", [
              ("Wall Time", fmt(cu_run.get("wall_time_sec"), 2, " s")),
              ("Realtime Factor", fmt(cu_run.get("realtime_factor"), 3, "x")),
              ("Throughput", fmt(cu_run.get("throughput_hz"), 2, " Hz")),
              ("Completion Ratio", fmt(cu_run.get("completion_ratio"), 3)),
              ("Max RSS", fmt(cu_run.get("max_rss_mb"), 1, " MB")),
              ("Avg CPU", fmt(cu_run.get("avg_cpu_percent"), 1, " %")),
              ("Avg GR3D", fmt(cu_run.get("avg_gr3d_percent"), 1, " %")),
              ("Avg Power", fmt(cu_run.get("avg_vdd_in_mw"), 0, " mW")),
          ])}
          {metric_block("FAST-LIVO2", [
              ("Wall Time", fmt(fl_run.get("wall_time_sec"), 2, " s")),
              ("Realtime Factor", fmt(fl_run.get("realtime_factor"), 3, "x")),
              ("Throughput", fmt(fl_run.get("throughput_hz"), 2, " Hz")),
              ("Output Poses", str(fl_run.get("output_poses"))),
              ("Max RSS", fmt(fl_run.get("max_rss_mb"), 1, " MB")),
              ("Avg CPU", fmt(fl_run.get("avg_cpu_percent"), 1, " %")),
              ("Avg GR3D", fmt(fl_run.get("avg_gr3d_percent"), 1, " %")),
              ("Avg Power", fmt(fl_run.get("avg_vdd_in_mw"), 0, " mW")),
          ])}
        </div>
      </div>
    </div>

    <div class="panel">
      <h2 style="margin-top:0">Run Context</h2>
      <div class="metric"><span>Trajectory GT mode</span><strong>{traj.get("ground_truth_mode", "n/a")}</strong></div>
      <div class="metric"><span>Prism offset</span><strong>{traj.get("prism_offset_m_body_frame")}</strong></div>
      <div class="metric"><span>Common matched frames</span><strong>{traj.get("common_matched_frames")}</strong></div>
      <div class="metric"><span>Time tolerance</span><strong>{traj.get("time_tolerance_sec")} s</strong></div>
      <div class="metric"><span>RPE delta</span><strong>{traj.get("rpe_delta")}</strong></div>
    </div>
  </div>
</body>
</html>
"""
    output = Path(args.output_html).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
