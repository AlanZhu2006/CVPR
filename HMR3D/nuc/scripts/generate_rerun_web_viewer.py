from __future__ import annotations

import argparse
from pathlib import Path


RERUN_WEB_VIEWER_VERSION = "0.22.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a local HTML page that opens a specific Rerun .rrd recording."
    )
    parser.add_argument("--rrd", required=True, help="Path to the .rrd file to load.")
    parser.add_argument(
        "--output-html",
        default="",
        help="Optional output HTML path. Defaults to <rrd_dir>/<rrd_stem>_viewer.html.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional page title. Defaults to the .rrd filename.",
    )
    return parser.parse_args()


def build_html(rrd_name: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      html, body {{
        margin: 0;
        width: 100%;
        height: 100%;
        background: #0d0f12;
        color: #f5f7fa;
        font-family: system-ui, sans-serif;
      }}
      #viewer {{
        width: 100vw;
        height: 100vh;
      }}
      #status {{
        position: fixed;
        top: 14px;
        left: 14px;
        z-index: 10;
        padding: 8px 10px;
        border-radius: 10px;
        background: rgba(12, 14, 18, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.12);
        font-size: 13px;
        backdrop-filter: blur(8px);
      }}
      #status code {{
        color: #9ad1ff;
      }}
    </style>
  </head>
  <body>
    <div id="status">Loading <code>{rrd_name}</code>...</div>
    <div id="viewer"></div>
    <script type="module">
      import {{ WebViewer }} from "https://esm.sh/@rerun-io/web-viewer@{RERUN_WEB_VIEWER_VERSION}";

      const viewer = new WebViewer();
      const container = document.getElementById("viewer");
      const status = document.getElementById("status");
      const params = new URLSearchParams(window.location.search);
      const requested = params.get("rrd");
      const rrdUrl = requested || "./{rrd_name}";

      try {{
        await viewer.start(rrdUrl, container, {{
          width: "100vw",
          height: "100vh",
          hide_welcome_screen: true,
          allow_fullscreen: true,
        }});
        status.textContent = `Loaded ${{
          rrdUrl
        }}`;
      }} catch (error) {{
        console.error(error);
        status.textContent = `Failed to load ${{rrdUrl}}: ${{error}}`;
      }}
    </script>
  </body>
</html>
"""


def main() -> None:
    args = parse_args()
    rrd_path = Path(args.rrd).expanduser().resolve()
    if not rrd_path.exists():
        raise FileNotFoundError(f"RRD file not found: {rrd_path}")

    output_html = (
        Path(args.output_html).expanduser().resolve()
        if args.output_html
        else rrd_path.with_name(f"{rrd_path.stem}_viewer.html")
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or rrd_path.name
    output_html.write_text(build_html(rrd_path.name, title), encoding="utf-8")
    print(output_html)


if __name__ == "__main__":
    main()
