from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a draggable timeline demo for Gaussian render benchmark output.")
    parser.add_argument("--run-dir", required=True, help="Benchmark directory containing render_benchmark_*.json and renders/.")
    parser.add_argument("--output-html", required=True, help="Output HTML path.")
    parser.add_argument("--trajectory-path", default="", help="Optional cuVSLAM trajectory_tum.txt for the mini-map.")
    parser.add_argument("--title", default="Gaussian Timeline Demo")
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--root-relative",
        action="store_true",
        help="Emit image URLs from the web server root instead of relative to the HTML file.",
    )
    return parser.parse_args()


def _rel(path: Path, root: Path, *, root_relative: bool) -> str:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace(os.sep, "/")
    return f"/{rel}" if root_relative else rel


def _load_trajectory(path: str | Path | None) -> list[dict[str, float]]:
    if not path:
        return []
    traj_path = Path(path).expanduser().resolve()
    if not traj_path.exists():
        return []
    poses: list[dict[str, float]] = []
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        items = line.strip().split()
        if len(items) < 7:
            continue
        values = [float(item) for item in items]
        if len(values) >= 8:
            tx, ty, tz = values[1:4]
        else:
            tx, ty, tz = values[0:3]
        poses.append({"x": tx, "y": ty, "z": tz})
    return poses


def _frame_image_paths(item: dict[str, Any], run_dir: Path) -> tuple[Path, Path, Path]:
    frame_idx = int(item["frame_idx"])
    renders_dir = run_dir / "renders"
    local_gt = renders_dir / f"{frame_idx:06d}_gt.png"
    local_render = renders_dir / f"{frame_idx:06d}_render.png"
    local_triplet = renders_dir / f"{frame_idx:06d}_triplet.png"
    gt = local_gt if local_gt.exists() else Path(item.get("gt_path", local_gt))
    render = local_render if local_render.exists() else Path(item.get("render_path", local_render))
    triplet = local_triplet if local_triplet.exists() else Path(item.get("triplet_path", local_triplet))
    return gt, render, triplet


def _load_payload(
    run_dir: Path,
    output_root: Path,
    *,
    root_relative: bool,
    trajectory_path: str,
    title: str,
    label: str,
) -> dict[str, Any]:
    frames_path = run_dir / "render_benchmark_frames.json"
    summary_path = run_dir / "render_benchmark_summary.json"
    frames_raw = json.loads(frames_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    poses = _load_trajectory(trajectory_path or summary.get("trajectory_path", ""))

    frames: list[dict[str, Any]] = []
    for item in frames_raw:
        gt, render, triplet = _frame_image_paths(item, run_dir)
        if not gt.exists() or not render.exists():
            continue
        frame_idx = int(item["frame_idx"])
        pose = poses[frame_idx] if 0 <= frame_idx < len(poses) else None
        snapshot = item.get("snapshot") if isinstance(item.get("snapshot"), dict) else {}
        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp_sec": item.get("timestamp_sec", 0.0),
                "is_keyframe": bool(item.get("is_keyframe", False)),
                "update_ms": item.get("update_ms", 0.0),
                "render_ms": item.get("render_ms", 0.0),
                "psnr": item.get("psnr", 0.0),
                "ssim": item.get("ssim", 0.0),
                "point_count": item.get("point_count", 0),
                "projected_points": item.get("projected_points", 0),
                "active_points": item.get("active_points", 0),
                "archived_points": item.get("archived_points", 0),
                "warmstart_points": item.get("warmstart_points", 0),
                "events": item.get("events", []),
                "active_id": snapshot.get("active_id"),
                "bank_size": snapshot.get("bank_size", 0),
                "recoveries": snapshot.get("recoveries", 0),
                "pose": pose,
                "gt": _rel(gt, output_root, root_relative=root_relative),
                "render": _rel(render, output_root, root_relative=root_relative),
                "triplet": _rel(triplet, output_root, root_relative=root_relative) if triplet.exists() else "",
            }
        )

    return {
        "title": title,
        "label": label or run_dir.name,
        "run_dir": str(run_dir),
        "metrics": summary.get("render_metrics", {}),
        "measured_frames": summary.get("measured_frames", len(frames)),
        "keyframe_count": summary.get("keyframe_count", 0),
        "frames": frames,
        "version": str(int(time.time())),
    }


def _build_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{payload["title"]}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg:#0c0d0b;
      --panel:#171713;
      --panel2:#202017;
      --line:#34342a;
      --text:#f4f2e8;
      --muted:#aaa68f;
      --green:#7bcf8e;
      --gold:#efb75d;
      --red:#e77662;
    }}
    * {{ box-sizing:border-box; }}
    html, body {{ margin:0; min-height:100%; background:var(--bg); color:var(--text); font-family:Inter, "Segoe UI", system-ui, sans-serif; }}
    body {{ overflow:hidden; }}
    .app {{ height:100vh; display:grid; grid-template-rows:auto minmax(0,1fr) auto; }}
    header {{ display:grid; grid-template-columns:minmax(260px,1fr) auto; gap:18px; align-items:center; padding:14px 18px; border-bottom:1px solid var(--line); background:#11120e; }}
    h1 {{ margin:0; font-size:20px; font-weight:650; letter-spacing:0; }}
    .run {{ color:var(--muted); font-size:13px; margin-top:4px; }}
    .metrics {{ display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }}
    .metric {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:8px 10px; min-width:96px; }}
    .metric .k {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .metric .v {{ margin-top:2px; font-variant-numeric:tabular-nums; font-size:16px; }}
    main {{ min-height:0; display:grid; grid-template-columns:minmax(0,1.6fr) minmax(320px,0.7fr); gap:14px; padding:14px; }}
    .stage {{ min-height:0; display:grid; grid-template-rows:minmax(0,1fr) minmax(180px,0.42fr); gap:14px; }}
    .viewer {{ min-height:0; display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; }}
    .pane, .side, .timeline {{ border:1px solid var(--line); border-radius:8px; background:var(--panel); overflow:hidden; }}
    .pane {{ display:grid; grid-template-rows:auto minmax(0,1fr); }}
    .paneHead {{ display:flex; justify-content:space-between; gap:12px; align-items:center; padding:10px 12px; border-bottom:1px solid var(--line); color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .paneHead strong {{ color:var(--text); font-size:13px; text-transform:none; }}
    .imageWrap {{ min-height:0; display:grid; place-items:center; background:#050604; }}
    .imageWrap img {{ display:block; width:100%; height:100%; object-fit:contain; }}
    .triplet {{ border:1px solid var(--line); border-radius:8px; background:#050604; overflow:hidden; }}
    .triplet img {{ width:100%; height:100%; display:block; object-fit:contain; }}
    .side {{ min-height:0; display:grid; grid-template-rows:minmax(220px,0.8fr) minmax(190px,0.7fr) auto; }}
    .map, .chartBox {{ min-height:0; border-bottom:1px solid var(--line); padding:12px; }}
    .sectionTitle {{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px; text-transform:uppercase; margin-bottom:10px; }}
    canvas {{ width:100%; height:calc(100% - 24px); display:block; }}
    .readout {{ padding:12px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    .read {{ border:1px solid var(--line); border-radius:8px; padding:8px; background:var(--panel2); min-width:0; }}
    .read .k {{ color:var(--muted); font-size:11px; text-transform:uppercase; white-space:nowrap; }}
    .read .v {{ margin-top:3px; font-variant-numeric:tabular-nums; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .timeline {{ display:grid; grid-template-columns:auto 1fr auto auto; gap:12px; align-items:center; padding:12px 14px; margin:0 14px 14px; background:#11120e; }}
    button {{ border:1px solid var(--line); border-radius:8px; background:var(--panel2); color:var(--text); height:38px; padding:0 14px; cursor:pointer; }}
    button:hover {{ border-color:var(--green); }}
    input[type=range] {{ width:100%; accent-color:var(--green); }}
    .timeLabel {{ color:var(--muted); min-width:176px; text-align:right; font-variant-numeric:tabular-nums; }}
    select {{ height:38px; border-radius:8px; border:1px solid var(--line); background:var(--panel2); color:var(--text); padding:0 10px; }}
    .events {{ display:flex; gap:6px; flex-wrap:wrap; min-height:24px; }}
    .tag {{ border:1px solid #4a4a3c; border-radius:999px; padding:3px 8px; color:var(--muted); font-size:12px; }}
    .tag.key {{ color:#10130f; background:var(--gold); border-color:var(--gold); }}
    @media (max-width: 1100px) {{
      body {{ overflow:auto; }}
      .app {{ height:auto; min-height:100vh; }}
      main {{ grid-template-columns:1fr; }}
      .viewer {{ grid-template-columns:1fr; }}
      .side {{ min-height:720px; }}
      .timeline {{ grid-template-columns:auto 1fr; }}
      .timeLabel, select {{ grid-column:1 / -1; width:100%; text-align:left; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1 id="title"></h1>
        <div class="run" id="runLabel"></div>
      </div>
      <div class="metrics" id="metrics"></div>
    </header>
    <main>
      <section class="stage">
        <div class="viewer">
          <div class="pane">
            <div class="paneHead"><strong>Gaussian Render</strong><span id="renderMeta"></span></div>
            <div class="imageWrap"><img id="renderImg" /></div>
          </div>
          <div class="pane">
            <div class="paneHead"><strong>Ground Truth</strong><span id="gtMeta"></span></div>
            <div class="imageWrap"><img id="gtImg" /></div>
          </div>
        </div>
        <div class="triplet"><img id="tripletImg" /></div>
      </section>
      <aside class="side">
        <div class="map">
          <div class="sectionTitle"><span>Trajectory</span><span id="poseMeta"></span></div>
          <canvas id="mapCanvas"></canvas>
        </div>
        <div class="chartBox">
          <div class="sectionTitle"><span>Render Stream</span><span id="chartMeta"></span></div>
          <canvas id="chartCanvas"></canvas>
        </div>
        <div class="readout">
          <div class="read"><div class="k">Frame</div><div class="v" id="frameRead"></div></div>
          <div class="read"><div class="k">PSNR / SSIM</div><div class="v" id="qualityRead"></div></div>
          <div class="read"><div class="k">Points</div><div class="v" id="pointsRead"></div></div>
          <div class="read"><div class="k">Projected</div><div class="v" id="projectedRead"></div></div>
          <div class="read"><div class="k">Update</div><div class="v" id="updateRead"></div></div>
          <div class="read"><div class="k">Submap</div><div class="v" id="submapRead"></div></div>
          <div class="read" style="grid-column:1 / -1;"><div class="k">Events</div><div class="events" id="eventsRead"></div></div>
        </div>
      </aside>
    </main>
    <div class="timeline">
      <button id="playBtn">Play</button>
      <input id="slider" type="range" min="0" max="0" value="0" />
      <select id="speed">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
      </select>
      <div class="timeLabel" id="timeLabel"></div>
    </div>
  </div>
  <script>
    const payload = {data};
    const frames = payload.frames;
    let cursor = 0;
    let playing = false;
    let timer = null;
    const $ = (id) => document.getElementById(id);
    const fmt = (n, d = 2) => Number(n || 0).toFixed(d);
    function asset(path) {{
      if (!path) return "";
      return `${{path}}${{path.includes("?") ? "&" : "?"}}v=${{payload.version}}`;
    }}
    function metric(label, value) {{
      const div = document.createElement("div");
      div.className = "metric";
      div.innerHTML = `<div class="k">${{label}}</div><div class="v">${{value}}</div>`;
      return div;
    }}
    function setCanvasSize(canvas) {{
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return [ctx, rect.width, rect.height];
    }}
    function drawMap() {{
      const [ctx, w, h] = setCanvasSize($("mapCanvas"));
      ctx.clearRect(0, 0, w, h);
      const pts = frames.map((f) => f.pose).filter(Boolean);
      if (pts.length < 2) {{
        ctx.fillStyle = "#aaa68f";
        ctx.fillText("no trajectory", 12, 24);
        return;
      }}
      const xs = pts.map((p) => p.x), zs = pts.map((p) => p.z);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minZ = Math.min(...zs), maxZ = Math.max(...zs);
      const pad = 18;
      const sx = (x) => pad + (x - minX) / Math.max(1e-6, maxX - minX) * (w - pad * 2);
      const sz = (z) => h - pad - (z - minZ) / Math.max(1e-6, maxZ - minZ) * (h - pad * 2);
      function pathUntil(n, stroke, width) {{
        ctx.beginPath();
        let started = false;
        for (let i = 0; i <= n; i++) {{
          const p = frames[i]?.pose;
          if (!p) continue;
          if (!started) {{ ctx.moveTo(sx(p.x), sz(p.z)); started = true; }}
          else ctx.lineTo(sx(p.x), sz(p.z));
        }}
        ctx.strokeStyle = stroke;
        ctx.lineWidth = width;
        ctx.stroke();
      }}
      pathUntil(frames.length - 1, "#3d3d32", 2);
      pathUntil(cursor, "#7bcf8e", 3);
      for (let i = 0; i < frames.length; i++) {{
        if (!frames[i].is_keyframe || !frames[i].pose) continue;
        ctx.beginPath();
        ctx.arc(sx(frames[i].pose.x), sz(frames[i].pose.z), i === cursor ? 5 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = i === cursor ? "#efb75d" : "#6b6a58";
        ctx.fill();
      }}
      const current = frames[cursor]?.pose;
      if (current) {{
        ctx.beginPath();
        ctx.arc(sx(current.x), sz(current.z), 6, 0, Math.PI * 2);
        ctx.fillStyle = "#7bcf8e";
        ctx.fill();
        ctx.strokeStyle = "#10130f";
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
    }}
    function drawChart() {{
      const [ctx, w, h] = setCanvasSize($("chartCanvas"));
      ctx.clearRect(0, 0, w, h);
      const pad = {{l:32, r:10, t:12, b:24}};
      const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
      const maxPoints = Math.max(...frames.map((f) => f.point_count || 0), 1);
      const maxProjected = Math.max(...frames.map((f) => f.projected_points || 0), 1);
      function y(v, max) {{ return pad.t + plotH - (v / max) * plotH; }}
      function x(i) {{ return pad.l + (i / Math.max(1, frames.length - 1)) * plotW; }}
      ctx.strokeStyle = "#34342a";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 3; g++) {{
        const yy = pad.t + (plotH * g) / 3;
        ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
      }}
      function line(key, max, color) {{
        ctx.beginPath();
        frames.forEach((f, i) => {{
          const xx = x(i), yy = y(f[key] || 0, max);
          if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
        }});
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
      line("point_count", maxPoints, "#efb75d");
      line("projected_points", maxProjected, "#7bcf8e");
      const cx = x(cursor);
      ctx.strokeStyle = "#f4f2e8";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, h - pad.b); ctx.stroke();
      ctx.fillStyle = "#aaa68f";
      ctx.font = "11px system-ui";
      ctx.fillText("points", pad.l, h - 8);
      ctx.fillStyle = "#7bcf8e";
      ctx.fillText("projected", pad.l + 56, h - 8);
    }}
    function preload(index) {{
      for (let i = index; i < Math.min(frames.length, index + 5); i++) {{
        [frames[i].render, frames[i].gt, frames[i].triplet].forEach((src) => {{
          if (!src) return;
          const img = new Image();
          img.src = asset(src);
        }});
      }}
    }}
    function render(index) {{
      cursor = Math.max(0, Math.min(frames.length - 1, index));
      const f = frames[cursor];
      $("slider").value = String(cursor);
      $("renderImg").src = asset(f.render);
      $("gtImg").src = asset(f.gt);
      $("tripletImg").src = asset(f.triplet || f.render);
      $("renderMeta").textContent = `render ${{fmt(f.render_ms, 1)}} ms`;
      $("gtMeta").textContent = f.is_keyframe ? "keyframe" : "tracking";
      $("poseMeta").textContent = f.pose ? `x ${{fmt(f.pose.x)}} z ${{fmt(f.pose.z)}}` : "";
      $("chartMeta").textContent = `${{f.projected_points}} / ${{f.point_count}}`;
      $("frameRead").textContent = `${{f.frame_idx}} / ${{fmt(f.timestamp_sec, 2)}} s`;
      $("qualityRead").textContent = `${{fmt(f.psnr, 2)}} / ${{fmt(f.ssim, 3)}}`;
      $("pointsRead").textContent = `${{f.point_count}}`;
      $("projectedRead").textContent = `${{f.projected_points}}`;
      $("updateRead").textContent = `${{fmt(f.update_ms, 1)}} ms`;
      $("submapRead").textContent = `active ${{f.active_id ?? "-"}}, bank ${{f.bank_size}}`;
      $("timeLabel").textContent = `${{cursor + 1}} / ${{frames.length}}`;
      const events = $("eventsRead");
      events.innerHTML = "";
      if (f.is_keyframe) {{
        const tag = document.createElement("span");
        tag.className = "tag key";
        tag.textContent = "keyframe";
        events.appendChild(tag);
      }}
      (f.events || []).forEach((event) => {{
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = event;
        events.appendChild(tag);
      }});
      if (!events.children.length) {{
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = "tracking";
        events.appendChild(tag);
      }}
      drawMap();
      drawChart();
      preload(cursor + 1);
    }}
    function start() {{
      if (playing || frames.length <= 1) return;
      playing = true;
      $("playBtn").textContent = "Pause";
      timer = setInterval(() => {{
        const next = cursor + 1 >= frames.length ? 0 : cursor + 1;
        render(next);
      }}, 520 / Number($("speed").value || 1));
    }}
    function stop() {{
      playing = false;
      $("playBtn").textContent = "Play";
      if (timer) clearInterval(timer);
      timer = null;
    }}
    $("title").textContent = payload.title;
    $("runLabel").textContent = `${{payload.label}} / ${{payload.measured_frames}} frames / ${{payload.keyframe_count}} keyframes`;
    const metrics = $("metrics");
    metrics.appendChild(metric("PSNR", fmt(payload.metrics.mean_psnr, 2)));
    metrics.appendChild(metric("SSIM", fmt(payload.metrics.mean_ssim, 3)));
    metrics.appendChild(metric("FPS", fmt(payload.metrics.approx_fps, 2)));
    metrics.appendChild(metric("Max Pts", payload.metrics.max_point_count || 0));
    $("slider").max = String(Math.max(0, frames.length - 1));
    $("slider").addEventListener("input", (e) => {{ stop(); render(Number(e.target.value)); }});
    $("playBtn").addEventListener("click", () => playing ? stop() : start());
    $("speed").addEventListener("change", () => {{ if (playing) {{ stop(); start(); }} }});
    window.addEventListener("resize", () => {{ drawMap(); drawChart(); }});
    render(0);
  </script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_payload(
        run_dir=Path(os.path.abspath(Path(args.run_dir).expanduser())),
        output_root=output_html.parent,
        root_relative=args.root_relative,
        trajectory_path=args.trajectory_path,
        title=args.title,
        label=args.label,
    )
    output_html.write_text(_build_html(payload), encoding="utf-8")
    print(json.dumps({"output_html": str(output_html), "frames": len(payload["frames"])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
