from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a lightweight HTML viewer for saved GT/Render/Diff triplets."
    )
    parser.add_argument("--run-dir", required=True, help="Render benchmark output directory.")
    parser.add_argument(
        "--output-html",
        default="",
        help="Optional output html path. Defaults to <run-dir>/render_triplets_viewer.html",
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

    entries = []
    for triplet_path in sorted(renders_dir.glob("*_triplet.png")):
        stem = triplet_path.stem.replace("_triplet", "")
        try:
            frame_idx = int(stem)
        except ValueError:
            continue
        meta = frame_by_idx.get(frame_idx, {})
        entries.append(
            {
                "frame_idx": frame_idx,
                "triplet": triplet_path.name,
                "gt": f"{stem}_gt.png",
                "render": f"{stem}_render.png",
                "psnr": meta.get("psnr"),
                "ssim": meta.get("ssim"),
                "point_count": meta.get("point_count"),
                "projected_points": meta.get("projected_points"),
            }
        )

    if not entries:
        raise SystemExit(f"no *_triplet.png files found in {renders_dir}")

    output_html = Path(args.output_html).expanduser().resolve() if args.output_html else run_dir / "render_triplets_viewer.html"
    output_html.write_text(_build_html(entries), encoding="utf-8")
    print(json.dumps({"output_html": str(output_html), "frames": len(entries)}, ensure_ascii=False, indent=2))


def _build_html(entries: list[dict]) -> str:
    payload = json.dumps(entries, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Surfel Render Viewer</title>
  <style>
    :root {{
      --bg: #101417;
      --panel: #151b20;
      --text: #edf2f7;
      --muted: #9eb0be;
      --accent: #8ee3b1;
      --accent-2: #7dcfff;
      --border: #2a333c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1c252d 0%, var(--bg) 55%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      letter-spacing: -0.02em;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 22px;
    }}
    .panel {{
      background: color-mix(in srgb, var(--panel) 92%, black);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: auto auto auto 1fr auto;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
    }}
    button {{
      background: linear-gradient(180deg, #24303a, #182028);
      color: var(--text);
      border: 1px solid #33414d;
      border-radius: 999px;
      padding: 10px 16px;
      cursor: pointer;
      font-size: 14px;
    }}
    button:hover {{ border-color: var(--accent-2); }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .metrics {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 14px;
    }}
    .metrics strong {{ color: var(--text); }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }}
    .img-card {{
      background: #0d1216;
      border: 1px solid #202930;
      border-radius: 14px;
      overflow: hidden;
    }}
    .img-title {{
      padding: 10px 14px;
      font-size: 13px;
      color: var(--muted);
      border-bottom: 1px solid #202930;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      image-rendering: auto;
      background: #000;
    }}
    .thumbs {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
    }}
    .thumb {{
      border: 1px solid #24303a;
      border-radius: 12px;
      overflow: hidden;
      cursor: pointer;
      background: #0c1115;
    }}
    .thumb.active {{ border-color: var(--accent); box-shadow: 0 0 0 1px rgba(142, 227, 177, 0.35); }}
    .thumb .cap {{
      padding: 10px 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    @media (min-width: 980px) {{
      .grid {{ grid-template-columns: 1fr 1fr; }}
      .hero {{ grid-column: 1 / -1; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Surfel Render Viewer</h1>
    <div class="sub">GT / Render / Diff triplets from the new continuous surfel-style Gaussian pipeline.</div>
    <div class="panel">
      <div class="toolbar">
        <button id="prevBtn">Prev</button>
        <button id="playBtn">Play</button>
        <button id="nextBtn">Next</button>
        <input id="frameSlider" type="range" min="0" max="{max(len(entries) - 1, 0)}" value="0" />
        <div id="frameLabel">Frame</div>
      </div>
      <div class="metrics" id="metrics"></div>
      <div class="grid" style="margin-top:16px;">
        <div class="img-card hero">
          <div class="img-title">Triplet Overview</div>
          <img id="tripletImg" alt="Triplet view" />
        </div>
        <div class="img-card">
          <div class="img-title">Ground Truth</div>
          <img id="gtImg" alt="Ground truth" />
        </div>
        <div class="img-card">
          <div class="img-title">Rendered View</div>
          <img id="renderImg" alt="Rendered view" />
        </div>
      </div>
      <div class="thumbs" id="thumbs"></div>
    </div>
  </div>
  <script>
    const frames = {payload};
    let index = 0;
    let timer = null;

    const slider = document.getElementById('frameSlider');
    const frameLabel = document.getElementById('frameLabel');
    const metrics = document.getElementById('metrics');
    const tripletImg = document.getElementById('tripletImg');
    const gtImg = document.getElementById('gtImg');
    const renderImg = document.getElementById('renderImg');
    const thumbs = document.getElementById('thumbs');
    const playBtn = document.getElementById('playBtn');

    function metric(label, value) {{
      return `<span><strong>${{label}}</strong> ${{value}}</span>`;
    }}

    function render() {{
      const item = frames[index];
      slider.value = index;
      frameLabel.textContent = `Frame ${{item.frame_idx}} (${{index + 1}} / ${{frames.length}})`;
      metrics.innerHTML = [
        metric('PSNR', item.psnr ?? '-'),
        metric('SSIM', item.ssim ?? '-'),
        metric('Points', item.point_count ?? '-'),
        metric('Projected', item.projected_points ?? '-'),
      ].join('');
      tripletImg.src = `renders/${{item.triplet}}`;
      gtImg.src = `renders/${{item.gt}}`;
      renderImg.src = `renders/${{item.render}}`;
      [...thumbs.children].forEach((node, i) => node.classList.toggle('active', i === index));
    }}

    function setIndex(next) {{
      index = (next + frames.length) % frames.length;
      render();
    }}

    function togglePlay() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
        playBtn.textContent = 'Play';
        return;
      }}
      timer = setInterval(() => setIndex(index + 1), 900);
      playBtn.textContent = 'Pause';
    }}

    frames.forEach((item, i) => {{
      const node = document.createElement('div');
      node.className = 'thumb';
      node.innerHTML = `<img src="renders/${{item.triplet}}" alt="Frame ${{item.frame_idx}} preview" /><div class="cap">Frame ${{item.frame_idx}}</div>`;
      node.addEventListener('click', () => setIndex(i));
      thumbs.appendChild(node);
    }});

    document.getElementById('prevBtn').addEventListener('click', () => setIndex(index - 1));
    document.getElementById('nextBtn').addEventListener('click', () => setIndex(index + 1));
    document.getElementById('playBtn').addEventListener('click', togglePlay);
    slider.addEventListener('input', (event) => setIndex(Number(event.target.value)));
    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowLeft') setIndex(index - 1);
      if (event.key === 'ArrowRight') setIndex(index + 1);
      if (event.key === ' ') {{
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
