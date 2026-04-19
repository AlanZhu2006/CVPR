from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a GS_Console-style side-by-side compare viewer for local render outputs."
    )
    parser.add_argument("--run-dir", required=True, help="Render benchmark output directory.")
    parser.add_argument(
        "--output-html",
        default="",
        help="Optional output html path. Defaults to <run-dir>/gsconsole_compare_viewer.html",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    renders_dir = run_dir / "renders"
    frames_json = run_dir / "render_benchmark_frames.json"
    if not renders_dir.exists():
        raise SystemExit(f"renders directory not found: {renders_dir}")

    frames = json.loads(frames_json.read_text(encoding="utf-8")) if frames_json.exists() else []
    frame_by_idx = {int(item["frame_idx"]): item for item in frames if "frame_idx" in item}

    entries: list[dict] = []
    for gt_path in sorted(renders_dir.glob("*_gt.png")):
        stem = gt_path.stem.replace("_gt", "")
        try:
            frame_idx = int(stem)
        except ValueError:
            continue
        render_path = renders_dir / f"{stem}_render.png"
        triplet_path = renders_dir / f"{stem}_triplet.png"
        if not render_path.exists():
            continue
        meta = frame_by_idx.get(frame_idx, {})
        entries.append(
            {
                "frame_idx": frame_idx,
                "gt": gt_path.name,
                "render": render_path.name,
                "triplet": triplet_path.name if triplet_path.exists() else "",
                "psnr": meta.get("psnr"),
                "ssim": meta.get("ssim"),
                "point_count": meta.get("point_count"),
                "projected_points": meta.get("projected_points"),
                "update_ms": meta.get("update_ms"),
                "render_ms": meta.get("render_ms"),
            }
        )

    if not entries:
        raise SystemExit(f"no GT/render frame pairs found in {renders_dir}")

    output_html = (
        Path(args.output_html).expanduser().resolve()
        if args.output_html
        else run_dir / "gsconsole_compare_viewer.html"
    )
    output_html.write_text(_build_html(entries), encoding="utf-8")
    print(json.dumps({"output_html": str(output_html), "frames": len(entries)}, ensure_ascii=False, indent=2))


def _build_html(entries: list[dict]) -> str:
    payload = json.dumps(entries, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Playback RGB vs HMR3D Gaussian</title>
  <style>
    :root {{
      --bg: #070b10;
      --panel: #0f151d;
      --panel-soft: #141d26;
      --text: #edf3f8;
      --muted: #9eb0c0;
      --accent: #86d6ff;
      --accent-2: #8ee3b1;
      --line: #273443;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top, rgba(64, 96, 128, 0.22), transparent 38%),
        linear-gradient(180deg, #0a1017 0%, var(--bg) 100%);
      color: var(--text);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1520px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}
    .hero {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 760px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: rgba(15, 21, 29, 0.72);
      backdrop-filter: blur(6px);
      font-size: 14px;
    }}
    .panel {{
      background: linear-gradient(180deg, rgba(17, 25, 34, 0.96), rgba(9, 14, 19, 0.98));
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 22px 60px rgba(0, 0, 0, 0.35);
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
    button:hover {{ border-color: var(--accent); }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent-2);
    }}
    .frame-label {{
      text-align: right;
      color: var(--muted);
      font-size: 14px;
    }}
    .stage {{
      background: #06090d;
      border: 1px solid #1e2a35;
      border-radius: 18px;
      overflow: hidden;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
      min-height: 420px;
    }}
    .feed {{
      position: relative;
      background: #000;
      border-right: 1px solid #172029;
    }}
    .feed:last-child {{ border-right: 0; }}
    .feed img {{
      display: block;
      width: 100%;
      height: auto;
      background: #000;
    }}
    .feed-label {{
      position: absolute;
      top: 14px;
      left: 14px;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: white;
      background: rgba(8, 12, 16, 0.68);
      border: 1px solid rgba(255,255,255,0.12);
      backdrop-filter: blur(8px);
    }}
    .rail {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 16px;
      margin-top: 16px;
    }}
    .card {{
      background: var(--panel-soft);
      border: 1px solid #202d39;
      border-radius: 16px;
      overflow: hidden;
      min-width: 0;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid #202d39;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .card img {{
      width: 100%;
      display: block;
      background: #000;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      padding: 14px;
    }}
    .metric {{
      background: #0b1117;
      border: 1px solid #1d2833;
      border-radius: 14px;
      padding: 12px;
    }}
    .metric .k {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .metric .v {{
      font-size: 21px;
      font-weight: 600;
      letter-spacing: -0.02em;
    }}
    .thumbs {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
    }}
    .thumb {{
      cursor: pointer;
      border: 1px solid #243342;
      border-radius: 14px;
      overflow: hidden;
      background: #0a1016;
    }}
    .thumb.active {{
      border-color: var(--accent-2);
      box-shadow: 0 0 0 1px rgba(142, 227, 177, 0.45);
    }}
    .thumb img {{
      display: block;
      width: 100%;
      background: #000;
    }}
    .thumb .cap {{
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 1100px) {{
      .compare-grid, .rail {{ grid-template-columns: 1fr; }}
      .toolbar {{ grid-template-columns: auto auto auto 1fr; }}
      .frame-label {{ grid-column: 1 / -1; text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>Playback RGB vs HMR3D Gaussian</h1>
        <p>GS_Console-inspired compare playback for the local surfel-style Gaussian pipeline. This improves the presentation layer to match the style you want, even though the underlying render quality still has headroom.</p>
      </div>
      <div class="pill">Cinema Compare Mode</div>
    </div>

    <section class="panel">
      <div class="toolbar">
        <button id="prevBtn">Prev</button>
        <button id="playBtn">Play</button>
        <button id="nextBtn">Next</button>
        <input id="frameSlider" type="range" min="0" max="{max(len(entries) - 1, 0)}" value="0" />
        <div class="frame-label" id="frameLabel"></div>
      </div>

      <div class="stage">
        <div class="compare-grid">
          <div class="feed">
            <div class="feed-label">Playback RGB</div>
            <img id="gtImg" alt="Playback RGB" />
          </div>
          <div class="feed">
            <div class="feed-label">HMR3D Gaussian</div>
            <img id="renderImg" alt="HMR3D Gaussian" />
          </div>
        </div>
      </div>

      <div class="rail">
        <section class="card">
          <div class="card-header">
            <span>Error / Diff</span>
            <span id="diffLabel">Diff</span>
          </div>
          <img id="tripletImg" alt="Diff and overview" />
        </section>
        <section class="card">
          <div class="card-header">
            <span>Frame Metrics</span>
            <span>Current Snapshot</span>
          </div>
          <div class="metrics">
            <div class="metric"><div class="k">PSNR</div><div class="v" id="psnrVal">-</div></div>
            <div class="metric"><div class="k">SSIM</div><div class="v" id="ssimVal">-</div></div>
            <div class="metric"><div class="k">Projected</div><div class="v" id="projVal">-</div></div>
            <div class="metric"><div class="k">Total Points</div><div class="v" id="pointsVal">-</div></div>
            <div class="metric"><div class="k">Update ms</div><div class="v" id="updateVal">-</div></div>
            <div class="metric"><div class="k">Render ms</div><div class="v" id="renderVal">-</div></div>
          </div>
        </section>
      </div>

      <div class="thumbs" id="thumbs"></div>
    </section>
  </div>

  <script>
    const frames = {payload};
    let index = 0;
    let timer = null;
    const fpsMs = 850;

    const gtImg = document.getElementById("gtImg");
    const renderImg = document.getElementById("renderImg");
    const tripletImg = document.getElementById("tripletImg");
    const frameLabel = document.getElementById("frameLabel");
    const slider = document.getElementById("frameSlider");
    const thumbs = document.getElementById("thumbs");
    const playBtn = document.getElementById("playBtn");

    const psnrVal = document.getElementById("psnrVal");
    const ssimVal = document.getElementById("ssimVal");
    const projVal = document.getElementById("projVal");
    const pointsVal = document.getElementById("pointsVal");
    const updateVal = document.getElementById("updateVal");
    const renderVal = document.getElementById("renderVal");
    const diffLabel = document.getElementById("diffLabel");

    function setIndex(next) {{
      index = (next + frames.length) % frames.length;
      render();
    }}

    function togglePlay() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
        playBtn.textContent = "Play";
        return;
      }}
      timer = setInterval(() => setIndex(index + 1), fpsMs);
      playBtn.textContent = "Pause";
    }}

    function render() {{
      const item = frames[index];
      slider.value = index;
      frameLabel.textContent = `Frame ${{item.frame_idx}} (${{index + 1}} / ${{frames.length}})`;
      gtImg.src = `renders/${{item.gt}}`;
      renderImg.src = `renders/${{item.render}}`;
      tripletImg.src = item.triplet ? `renders/${{item.triplet}}` : `renders/${{item.render}}`;
      psnrVal.textContent = item.psnr ?? "-";
      ssimVal.textContent = item.ssim ?? "-";
      projVal.textContent = item.projected_points ?? "-";
      pointsVal.textContent = item.point_count ?? "-";
      updateVal.textContent = item.update_ms ?? "-";
      renderVal.textContent = item.render_ms ?? "-";
      diffLabel.textContent = item.triplet ? "GT / Render / Diff" : "Render";
      [...thumbs.children].forEach((node, i) => node.classList.toggle("active", i === index));
    }}

    frames.forEach((item, i) => {{
      const node = document.createElement("div");
      node.className = "thumb";
      node.innerHTML = `<img src="renders/${{item.triplet || item.render}}" alt="frame ${{item.frame_idx}}" /><div class="cap">Frame ${{item.frame_idx}}</div>`;
      node.addEventListener("click", () => setIndex(i));
      thumbs.appendChild(node);
    }});

    document.getElementById("prevBtn").addEventListener("click", () => setIndex(index - 1));
    document.getElementById("nextBtn").addEventListener("click", () => setIndex(index + 1));
    document.getElementById("playBtn").addEventListener("click", togglePlay);
    slider.addEventListener("input", (event) => setIndex(Number(event.target.value)));
    window.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft") setIndex(index - 1);
      if (event.key === "ArrowRight") setIndex(index + 1);
      if (event.key === " ") {{
        event.preventDefault();
        togglePlay();
      }}
    }});

    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
