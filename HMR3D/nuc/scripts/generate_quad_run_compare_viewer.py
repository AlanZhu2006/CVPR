from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 4-way compare viewer for Gaussian benchmark runs.")
    for name in ("a", "b", "c", "d"):
        parser.add_argument(f"--run-{name}", required=True, help=f"Run directory {name}.")
        parser.add_argument(f"--label-{name}", required=True, help=f"Label for run {name}.")
    parser.add_argument("--output-html", required=True, help="Output HTML path.")
    parser.add_argument(
        "--root-relative",
        action="store_true",
        help="Emit image URLs from the web server root instead of relative to the HTML file.",
    )
    return parser.parse_args()


def _rel(path: Path, root: Path, *, root_relative: bool) -> str:
    import os

    rel = os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")
    return f"/{rel}" if root_relative else rel


def _load_run(run_dir: Path, output_root: Path, label: str, *, root_relative: bool) -> dict:
    frames = json.loads((run_dir / "render_benchmark_frames.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "render_benchmark_summary.json").read_text(encoding="utf-8"))["render_metrics"]
    by_idx = {}
    renders_dir = run_dir / "renders"
    for item in frames:
        idx = int(item["frame_idx"])
        gt = Path(item.get("gt_path", renders_dir / f"{idx:06d}_gt.png"))
        render = Path(item.get("render_path", renders_dir / f"{idx:06d}_render.png"))
        triplet = Path(item.get("triplet_path", renders_dir / f"{idx:06d}_triplet.png"))
        if not gt.exists() or not render.exists():
            continue
        by_idx[idx] = {
            "frame_idx": idx,
            "gt": _rel(gt, output_root, root_relative=root_relative),
            "render": _rel(render, output_root, root_relative=root_relative),
            "triplet": _rel(triplet, output_root, root_relative=root_relative) if triplet.exists() else "",
            "psnr": item.get("psnr"),
            "ssim": item.get("ssim"),
            "render_ms": item.get("render_ms"),
            "point_count": item.get("point_count"),
            "projected_points": item.get("projected_points"),
        }
    return {"label": label, "summary": summary, "frames": by_idx}


def _build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gaussian Quad Compare</title>
  <style>
    :root {{
      --bg:#091018; --panel:#111a23; --panel2:#15212c; --text:#edf3f8; --muted:#9db0c1; --line:#263444;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--text); background:linear-gradient(180deg,#0a1017,var(--bg)); font-family:"IBM Plex Sans","Segoe UI",sans-serif; }}
    .wrap {{ max-width:1760px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 8px; font-size:34px; }}
    .sub {{ color:var(--muted); margin-bottom:16px; }}
    .panel {{ background:linear-gradient(180deg,rgba(18,27,36,0.98),rgba(10,15,21,0.98)); border:1px solid var(--line); border-radius:22px; padding:18px; }}
    .toolbar {{ display:grid; grid-template-columns:auto auto auto 1fr auto; gap:12px; align-items:center; margin-bottom:16px; }}
    button {{ border:1px solid #324454; color:var(--text); background:linear-gradient(180deg,#1c2834,#121a23); border-radius:999px; padding:10px 16px; cursor:pointer; }}
    input[type="range"] {{ width:100%; }}
    .summary-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:16px; }}
    .summary, .card {{ border:1px solid var(--line); border-radius:16px; background:var(--panel2); padding:14px; }}
    .summary h3, .card .head {{ margin:0 0 10px; font-size:15px; }}
    .summary .m {{ color:var(--muted); font-size:13px; line-height:1.7; }}
    .stage {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }}
    .card {{ padding:0; overflow:hidden; background:#0a0f14; }}
    .card .head {{ padding:12px 14px; border-bottom:1px solid var(--line); color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; font-size:13px; }}
    .card img {{ display:block; width:100%; background:#000; }}
    .feeds {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    .feed {{ border:1px solid var(--line); border-radius:18px; overflow:hidden; background:#0a0f14; }}
    .feed .head {{ display:flex; justify-content:space-between; gap:8px; padding:12px 14px; border-bottom:1px solid var(--line); font-size:13px; }}
    .feed .meta {{ color:var(--muted); }}
    .feed img {{ display:block; width:100%; }}
    .thumbs {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px; margin-top:18px; }}
    .thumb {{ border:1px solid #24303a; border-radius:12px; overflow:hidden; cursor:pointer; background:#0c1115; }}
    .thumb.active {{ border-color:#8ee3b1; box-shadow:0 0 0 1px rgba(142,227,177,0.35); }}
    .thumb img {{ width:100%; display:block; }}
    .thumb .cap {{ padding:10px 12px; font-size:13px; color:var(--muted); }}
    @media (max-width: 1280px) {{ .summary-grid,.feeds,.stage {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Gaussian Quad Compare</h1>
    <div class="sub">Ground truth plus LingBot Gaussian init against v9 / v11 / v12 baselines.</div>
    <div class="panel">
      <div class="toolbar">
        <button id="prevBtn">Prev</button>
        <button id="playBtn">Play</button>
        <button id="nextBtn">Next</button>
        <input id="frameSlider" type="range" min="0" max="0" value="0" />
        <div id="frameLabel">Frame</div>
      </div>
      <div class="summary-grid" id="summaryGrid"></div>
      <div class="stage">
        <div class="card"><div class="head">Ground Truth</div><img id="gtImg" /></div>
        <div class="card"><div class="head">Frame Overview</div><img id="tripletImg" /></div>
      </div>
      <div class="feeds" id="feedGrid"></div>
      <div class="thumbs" id="thumbs"></div>
    </div>
  </div>
  <script>
    const payload = {data};
    const entries = Object.entries(payload.runs);
    const summaryGrid = document.getElementById("summaryGrid");
    const feedGrid = document.getElementById("feedGrid");
    const gtImg = document.getElementById("gtImg");
    const tripletImg = document.getElementById("tripletImg");
    const frameLabel = document.getElementById("frameLabel");
    const slider = document.getElementById("frameSlider");
    const thumbs = document.getElementById("thumbs");
    const playBtn = document.getElementById("playBtn");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    let cursor = 0;
    let timer = null;
    let feedImages = {{}};
    function asset(path) {{
      if (!path) return "";
      const sep = path.includes("?") ? "&" : "?";
      return `${{path}}${{sep}}v=${{payload.version}}`;
    }}
    entries.forEach(([key, run]) => {{
      const summary = document.createElement("div");
      summary.className = "summary";
      summary.innerHTML = `<h3>${{run.label}}</h3><div class="m">PSNR ${{run.summary.mean_psnr}}<br/>SSIM ${{run.summary.mean_ssim}}<br/>Render ${{run.summary.mean_render_ms}} ms<br/>FPS ${{run.summary.approx_fps}}</div>`;
      summaryGrid.appendChild(summary);
      const feed = document.createElement("div");
      feed.className = "feed";
      feed.innerHTML = `<div class="head"><span>${{run.label}}</span><span class="meta" id="meta-${{key}}"></span></div><img id="img-${{key}}" />`;
      feedGrid.appendChild(feed);
      feedImages[key] = document.getElementById(`img-${{key}}`);
    }});
    function renderFrame(index) {{
      cursor = index;
      const frame = payload.frames[cursor];
      frameLabel.textContent = `Frame ${{frame.frame_idx}} (${{cursor + 1}} / ${{payload.frames.length}})`;
      slider.value = cursor;
      gtImg.src = asset(frame.gt);
      tripletImg.src = asset(frame.a.triplet || frame.gt);
      for (const [key, run] of entries) {{
        const item = frame[key];
        feedImages[key].src = asset(item.render);
        document.getElementById(`meta-${{key}}`).textContent = `PSNR ${{item.psnr}} | SSIM ${{item.ssim}} | render ${{item.render_ms}} ms`;
      }}
      [...thumbs.children].forEach((thumb, idx) => thumb.classList.toggle("active", idx === cursor));
    }}
    payload.frames.forEach((frame, idx) => {{
      const thumb = document.createElement("div");
      thumb.className = "thumb";
      thumb.innerHTML = `<img src="${{asset(frame.a.triplet || frame.gt)}}" /><div class="cap">Frame ${{frame.frame_idx}}</div>`;
      thumb.addEventListener("click", () => renderFrame(idx));
      thumbs.appendChild(thumb);
    }});
    slider.max = Math.max(0, payload.frames.length - 1);
    slider.addEventListener("input", () => renderFrame(Number(slider.value)));
    prevBtn.addEventListener("click", () => renderFrame((cursor - 1 + payload.frames.length) % payload.frames.length));
    nextBtn.addEventListener("click", () => renderFrame((cursor + 1) % payload.frames.length));
    playBtn.addEventListener("click", () => {{
      if (timer) {{
        clearInterval(timer); timer = null; playBtn.textContent = "Play"; return;
      }}
      timer = setInterval(() => renderFrame((cursor + 1) % payload.frames.length), 1200);
      playBtn.textContent = "Pause";
    }});
    renderFrame(0);
  </script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    root = output_html.parent

    runs = {}
    for name in ("a", "b", "c", "d"):
        runs[name] = _load_run(
            Path(getattr(args, f"run_{name}")).expanduser().resolve(),
            root,
            getattr(args, f"label_{name}"),
            root_relative=args.root_relative,
        )

    common = sorted(set.intersection(*(set(run["frames"]) for run in runs.values())))
    if not common:
        raise SystemExit("No common frames across the four runs.")

    frames = []
    for idx in common:
        entry = {"frame_idx": idx, "gt": runs["a"]["frames"][idx]["gt"]}
        for name in ("a", "b", "c", "d"):
            entry[name] = runs[name]["frames"][idx]
        frames.append(entry)

    payload = {
        "runs": {name: {"label": run["label"], "summary": run["summary"]} for name, run in runs.items()},
        "frames": frames,
        "version": str(int(time.time())),
    }
    output_html.write_text(_build_html(payload), encoding="utf-8")
    print(json.dumps({"output_html": str(output_html), "frames": len(frames)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
