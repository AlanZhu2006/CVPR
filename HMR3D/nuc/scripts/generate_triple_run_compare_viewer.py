from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3-way compare viewer for three Gaussian render benchmark runs."
    )
    parser.add_argument("--run-a", required=True, help="First run directory.")
    parser.add_argument("--label-a", default="Run A", help="Label for first run.")
    parser.add_argument("--run-b", required=True, help="Second run directory.")
    parser.add_argument("--label-b", default="Run B", help="Label for second run.")
    parser.add_argument("--run-c", required=True, help="Third run directory.")
    parser.add_argument("--label-c", default="Run C", help="Label for third run.")
    parser.add_argument(
        "--output-html",
        required=True,
        help="Output HTML path. Put it under a common parent so relative image paths resolve.",
    )
    return parser.parse_args()


def _load_run(run_dir: Path, output_root: Path, label: str) -> dict:
    frames_path = run_dir / "render_benchmark_frames.json"
    summary_path = run_dir / "render_benchmark_summary.json"
    renders_dir = run_dir / "renders"
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["render_metrics"]
    by_idx: dict[int, dict] = {}
    for item in frames:
        frame_idx = int(item["frame_idx"])
        gt = Path(item.get("gt_path", ""))
        render = Path(item.get("render_path", ""))
        triplet = Path(item.get("triplet_path", ""))
        if not render.is_file():
            render = renders_dir / f"{frame_idx:06d}_render.png"
        if not gt.is_file():
            gt = renders_dir / f"{frame_idx:06d}_gt.png"
        if not triplet.is_file():
            triplet = renders_dir / f"{frame_idx:06d}_triplet.png"
        if not render.exists() or not gt.exists():
            continue
        by_idx[frame_idx] = {
            "frame_idx": frame_idx,
            "gt": _rel(gt, output_root),
            "render": _rel(render, output_root),
            "triplet": _rel(triplet, output_root) if triplet.exists() else "",
            "psnr": item.get("psnr"),
            "ssim": item.get("ssim"),
            "update_ms": item.get("update_ms"),
            "render_ms": item.get("render_ms"),
            "projected_points": item.get("projected_points"),
            "point_count": item.get("point_count"),
        }
    return {"label": label, "summary": summary, "frames": by_idx}


def _rel(path: Path, root: Path) -> str:
    import os

    return os.path.relpath(path.resolve(), root.resolve())


def main() -> None:
    args = parse_args()
    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    root = output_html.parent

    run_a = _load_run(Path(args.run_a).expanduser().resolve(), root, args.label_a)
    run_b = _load_run(Path(args.run_b).expanduser().resolve(), root, args.label_b)
    run_c = _load_run(Path(args.run_c).expanduser().resolve(), root, args.label_c)

    common = sorted(set(run_a["frames"]) & set(run_b["frames"]) & set(run_c["frames"]))
    if not common:
        raise SystemExit("No common frame indices across the three runs.")

    entries = []
    for frame_idx in common:
        a = run_a["frames"][frame_idx]
        b = run_b["frames"][frame_idx]
        c = run_c["frames"][frame_idx]
        entries.append(
            {
                "frame_idx": frame_idx,
                "gt": a["gt"],
                "a": a,
                "b": b,
                "c": c,
            }
        )

    payload = {
        "runs": {
            "a": {"label": run_a["label"], "summary": run_a["summary"]},
            "b": {"label": run_b["label"], "summary": run_b["summary"]},
            "c": {"label": run_c["label"], "summary": run_c["summary"]},
        },
        "frames": entries,
    }
    output_html.write_text(_build_html(payload), encoding="utf-8")
    print(json.dumps({"output_html": str(output_html), "frames": len(entries)}, ensure_ascii=False, indent=2))


def _build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gaussian Triple Compare</title>
  <style>
    :root {{
      --bg: #091018;
      --panel: #111a23;
      --panel2: #15212c;
      --text: #edf3f8;
      --muted: #9db0c1;
      --line: #263444;
      --accent: #88d7ff;
      --accent2: #8ee3b1;
      --accent3: #ffc56e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background: radial-gradient(circle at top, rgba(75,112,148,0.18), transparent 32%), linear-gradient(180deg, #0a1017, var(--bg));
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .wrap {{ max-width: 1680px; margin: 0 auto; padding: 26px 24px 42px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: -0.03em; }}
    .sub {{ color: var(--muted); margin-bottom: 18px; }}
    .panel {{
      background: linear-gradient(180deg, rgba(18,27,36,0.98), rgba(10,15,21,0.98));
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 24px 60px rgba(0,0,0,0.35);
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: auto auto auto 1fr auto;
      gap: 12px;
      align-items: center;
      margin-bottom: 16px;
    }}
    button {{
      appearance: none;
      border: 1px solid #324454;
      color: var(--text);
      background: linear-gradient(180deg, #1c2834, #121a23);
      border-radius: 999px;
      padding: 10px 16px;
      font-size: 14px;
      cursor: pointer;
    }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent2); }}
    .frame-label {{ color: var(--muted); text-align: right; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0,1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    .summary {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel2);
      padding: 14px;
    }}
    .summary h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .summary .m {{ color: var(--muted); font-size: 13px; line-height: 1.7; }}
    .stage {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: #0a0f14;
    }}
    .card .head {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      text-transform: uppercase;
      font-size: 13px;
      letter-spacing: 0.08em;
    }}
    .card img {{ display: block; width: 100%; height: auto; background: #000; }}
    .triple {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }}
    .feed {{
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: #0a0f14;
    }}
    .feed .head {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
    }}
    .feed .meta {{ color: var(--muted); }}
    .thumbs {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .thumb {{
      border: 1px solid #24303a;
      border-radius: 12px;
      overflow: hidden;
      cursor: pointer;
      background: #0c1115;
    }}
    .thumb.active {{ border-color: var(--accent2); box-shadow: 0 0 0 1px rgba(142,227,177,0.35); }}
    .thumb img {{ width: 100%; display: block; }}
    .thumb .cap {{ padding: 10px 12px; font-size: 13px; color: var(--muted); }}
    @media (max-width: 1180px) {{
      .summary-grid, .triple, .stage {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Gaussian Triple Compare</h1>
    <div class="sub">Ground truth plus three runs: quality-first surfel, realtime budget, and thin-surface local volume.</div>
    <div class="panel">
      <div class="toolbar">
        <button id="prevBtn">Prev</button>
        <button id="playBtn">Play</button>
        <button id="nextBtn">Next</button>
        <input id="frameSlider" type="range" min="0" max="0" value="0" />
        <div class="frame-label" id="frameLabel">Frame</div>
      </div>
      <div class="summary-grid" id="summaryGrid"></div>
      <div class="stage">
        <div class="card">
          <div class="head">Ground Truth</div>
          <img id="gtImg" alt="Ground truth" />
        </div>
        <div class="card">
          <div class="head">Frame Overview</div>
          <img id="tripletImg" alt="Triplet overview" />
        </div>
      </div>
      <div class="triple">
        <div class="feed">
          <div class="head"><span id="labelA"></span><span class="meta" id="metaA"></span></div>
          <img id="imgA" alt="Run A" />
        </div>
        <div class="feed">
          <div class="head"><span id="labelB"></span><span class="meta" id="metaB"></span></div>
          <img id="imgB" alt="Run B" />
        </div>
        <div class="feed">
          <div class="head"><span id="labelC"></span><span class="meta" id="metaC"></span></div>
          <img id="imgC" alt="Run C" />
        </div>
      </div>
      <div class="thumbs" id="thumbs"></div>
    </div>
  </div>
  <script>
    const payload = {data};
    const frames = payload.frames;
    let index = 0;
    let timer = null;

    const slider = document.getElementById('frameSlider');
    slider.max = Math.max(0, frames.length - 1);
    const labelA = document.getElementById('labelA');
    const labelB = document.getElementById('labelB');
    const labelC = document.getElementById('labelC');
    labelA.textContent = payload.runs.a.label;
    labelB.textContent = payload.runs.b.label;
    labelC.textContent = payload.runs.c.label;

    function metricText(item) {{
      return `PSNR ${{fmt(item.psnr)}} | SSIM ${{fmt(item.ssim)}} | render ${{fmt(item.render_ms)}} ms`;
    }}
    function fmt(v) {{
      if (v === null || v === undefined) return 'n/a';
      return Number(v).toFixed(3).replace(/\\.000$/, '');
    }}
    function renderSummary() {{
      const grid = document.getElementById('summaryGrid');
      grid.innerHTML = '';
      for (const key of ['a','b','c']) {{
        const run = payload.runs[key];
        const div = document.createElement('div');
        div.className = 'summary';
        div.innerHTML = `<h3>${{run.label}}</h3>
          <div class="m">PSNR: <strong>${{fmt(run.summary.mean_psnr)}}</strong><br/>
          SSIM: <strong>${{fmt(run.summary.mean_ssim)}}</strong><br/>
          Update: <strong>${{fmt(run.summary.mean_update_ms)}} ms</strong><br/>
          Render: <strong>${{fmt(run.summary.mean_render_ms)}} ms</strong><br/>
          FPS: <strong>${{fmt(run.summary.approx_fps)}}</strong></div>`;
        grid.appendChild(div);
      }}
    }}
    function renderThumbs() {{
      const thumbs = document.getElementById('thumbs');
      thumbs.innerHTML = '';
      frames.forEach((item, idx) => {{
        const div = document.createElement('div');
        div.className = 'thumb' + (idx === index ? ' active' : '');
        div.innerHTML = `<img src="${{item.a.render}}" alt="Frame ${{item.frame_idx}}" /><div class="cap">Frame ${{item.frame_idx}}</div>`;
        div.onclick = () => {{ index = idx; update(); }};
        thumbs.appendChild(div);
      }});
    }}
    function update() {{
      const frame = frames[index];
      document.getElementById('gtImg').src = frame.gt;
      document.getElementById('tripletImg').src = frame.a.triplet || frame.a.render;
      document.getElementById('imgA').src = frame.a.render;
      document.getElementById('imgB').src = frame.b.render;
      document.getElementById('imgC').src = frame.c.render;
      document.getElementById('metaA').textContent = metricText(frame.a);
      document.getElementById('metaB').textContent = metricText(frame.b);
      document.getElementById('metaC').textContent = metricText(frame.c);
      document.getElementById('frameLabel').textContent = `Frame ${{frame.frame_idx}} (${{index + 1}}/${{frames.length}})`;
      slider.value = String(index);
      renderThumbs();
    }}
    function togglePlay() {{
      const btn = document.getElementById('playBtn');
      if (timer) {{
        clearInterval(timer);
        timer = null;
        btn.textContent = 'Play';
        return;
      }}
      timer = setInterval(() => {{
        index = (index + 1) % frames.length;
        update();
      }}, 900);
      btn.textContent = 'Pause';
    }}
    document.getElementById('prevBtn').onclick = () => {{ index = (index - 1 + frames.length) % frames.length; update(); }};
    document.getElementById('nextBtn').onclick = () => {{ index = (index + 1) % frames.length; update(); }};
    document.getElementById('playBtn').onclick = togglePlay;
    slider.oninput = (e) => {{ index = Number(e.target.value); update(); }};
    renderSummary();
    update();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
