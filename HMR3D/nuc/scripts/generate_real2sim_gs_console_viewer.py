#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local WebGL viewer for live real-to-sim outputs.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-html", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve() if args.output_html else baseline_dir / "real2sim_gs_console_viewer.html"
    manifest_path = baseline_dir / "latest_manifest.json"
    latest_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    html = _build_html(latest_manifest)
    output_html.write_text(html, encoding="utf-8")
    print(json.dumps({"output_html": str(output_html)}, indent=2, ensure_ascii=False))
    return 0


def _build_html(latest_manifest: dict) -> str:
    payload = json.dumps(latest_manifest, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Real-to-Sim GS Console</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #090b0f;
      --panel: #111821;
      --panel-2: #17202a;
      --line: #293644;
      --text: #eef4f8;
      --muted: #9fb0bd;
      --accent: #6fd0ff;
      --ok: #7ee0a8;
      --warn: #ffd36f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      height: 100vh;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #app {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      height: 100vh;
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, #121a23 0%, #0c1118 100%);
      padding: 18px;
      overflow: auto;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .sub {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 18px;
    }}
    .row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 0;
      border-top: 1px solid rgba(255,255,255,0.07);
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .value {{
      font-variant-numeric: tabular-nums;
      font-size: 13px;
      text-align: right;
      overflow-wrap: anywhere;
    }}
    .controls {{
      display: grid;
      gap: 10px;
      margin: 16px 0;
    }}
    button {{
      width: 100%;
      border: 1px solid #344556;
      border-radius: 7px;
      background: #182331;
      color: var(--text);
      padding: 10px 12px;
      font-size: 14px;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); }}
    .button-link {{
      display: block;
      width: 100%;
      border: 1px solid #344556;
      border-radius: 7px;
      background: #182331;
      color: var(--text);
      padding: 10px 12px;
      font-size: 14px;
      text-decoration: none;
      text-align: center;
    }}
    .button-link:hover {{ border-color: var(--accent); }}
    label {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }}
    input[type="checkbox"] {{ accent-color: var(--accent); }}
    input[type="range"] {{ width: 100%; accent-color: var(--ok); grid-column: 1 / -1; }}
    main {{
      position: relative;
      min-width: 0;
      background: #05070a;
    }}
    #viewer {{
      position: absolute;
      inset: 0;
    }}
    #status {{
      position: absolute;
      left: 16px;
      right: 16px;
      bottom: 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      pointer-events: none;
    }}
    .chip {{
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 7px;
      padding: 8px 10px;
      background: rgba(11, 16, 22, 0.78);
      backdrop-filter: blur(10px);
      color: var(--muted);
      font-size: 12px;
    }}
    code {{
      color: var(--accent);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    @media (max-width: 880px) {{
      #app {{ grid-template-columns: 1fr; grid-template-rows: auto 1fr; }}
      aside {{ max-height: 280px; border-right: 0; border-bottom: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <div id="app">
    <aside>
      <h1>Live Real-to-Sim</h1>
      <div class="sub">TSDF mesh + Gaussian seed preview from the latest LingBot/HikRobot baseline export.</div>
      <div class="controls">
        <button id="reload">Reload Latest</button>
        <button id="reset">Reset Camera</button>
        <a class="button-link" href="latest_gaussian_seed_renders_gsplat/gsconsole_compare_viewer.html" target="_blank">Open gsplat render</a>
        <label>Gaussian seed <input id="showGaussians" type="checkbox" checked /></label>
        <label>TSDF mesh <input id="showMesh" type="checkbox" checked /></label>
        <label>Point size <span id="pointSizeValue">0.055</span><input id="pointSize" type="range" min="0.01" max="0.18" value="0.055" step="0.005" /></label>
        <label>Mesh opacity <span id="meshOpacityValue">0.30</span><input id="meshOpacity" type="range" min="0.05" max="1" value="0.30" step="0.05" /></label>
      </div>
      <div class="row"><span class="label">Sequence</span><span id="sequence" class="value">loading</span></div>
      <div class="row"><span class="label">Frames</span><span id="frames" class="value">-</span></div>
      <div class="row"><span class="label">Mesh</span><span id="meshStats" class="value">-</span></div>
      <div class="row"><span class="label">Gaussian points</span><span id="gaussianStats" class="value">-</span></div>
      <div class="row"><span class="label">Source</span><span class="value"><code>latest</code></span></div>
    </aside>
    <main>
      <div id="viewer"></div>
      <div id="status">
        <div id="loadStatus" class="chip">Starting viewer</div>
        <div class="chip">Mouse drag rotate</div>
        <div class="chip">Wheel zoom</div>
      </div>
    </main>
  </div>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.161.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.161.0/examples/jsm/"
      }}
    }}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
    import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';

    const initialManifest = {payload};
    const viewer = document.getElementById('viewer');
    const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: 'high-performance' }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
    renderer.setClearColor(0x05070a, 1);
    viewer.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x05070a, 40, 160);
    const camera = new THREE.PerspectiveCamera(55, viewer.clientWidth / Math.max(viewer.clientHeight, 1), 0.01, 1000);
    camera.position.set(0, -18, 8);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;

    scene.add(new THREE.HemisphereLight(0xf4fbff, 0x18222d, 2.2));
    const light = new THREE.DirectionalLight(0xffffff, 1.2);
    light.position.set(5, -8, 12);
    scene.add(light);
    const grid = new THREE.GridHelper(20, 20, 0x2b3a45, 0x17212a);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    let gaussianObject = null;
    let meshObject = null;
    let currentCenter = new THREE.Vector3(0, 0, 0);
    const loader = new PLYLoader();

    const showGaussians = document.getElementById('showGaussians');
    const showMesh = document.getElementById('showMesh');
    const pointSize = document.getElementById('pointSize');
    const pointSizeValue = document.getElementById('pointSizeValue');
    const meshOpacity = document.getElementById('meshOpacity');
    const meshOpacityValue = document.getElementById('meshOpacityValue');

    function setStatus(text) {{
      document.getElementById('loadStatus').textContent = text;
    }}

    function disposeObject(obj) {{
      if (!obj) return;
      scene.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }}

    async function readJson(url) {{
      const response = await fetch(url + '?t=' + Date.now(), {{ cache: 'no-store' }});
      if (!response.ok) throw new Error(url + ' ' + response.status);
      return response.json();
    }}

    function setMeta(manifest) {{
      const exp = manifest.export || manifest;
      const mesh = exp.mesh || {{}};
      const gs = exp.gaussian_seed || {{}};
      document.getElementById('sequence').textContent = manifest.sequence || exp.sequence || 'latest';
      document.getElementById('frames').textContent = exp.frame_count ?? '-';
      document.getElementById('meshStats').textContent = mesh.vertex_count ? `${{mesh.vertex_count}} v / ${{mesh.face_count}} f` : '-';
      document.getElementById('gaussianStats').textContent = gs.point_count ?? '-';
    }}

    function fitCamera() {{
      const box = new THREE.Box3();
      if (gaussianObject) box.expandByObject(gaussianObject);
      if (meshObject) box.expandByObject(meshObject);
      if (box.isEmpty()) return;
      const size = box.getSize(new THREE.Vector3());
      currentCenter = box.getCenter(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 1);
      controls.target.copy(currentCenter);
      camera.position.copy(currentCenter).add(new THREE.Vector3(0, -radius * 1.5, radius * 0.55));
      camera.near = Math.max(radius / 1000, 0.01);
      camera.far = radius * 20;
      camera.updateProjectionMatrix();
      controls.update();
    }}

    function loadPly(url) {{
      return new Promise((resolve, reject) => {{
        loader.load(url + '?t=' + Date.now(), resolve, undefined, reject);
      }});
    }}

    async function reloadLatest() {{
      try {{
        setStatus('Loading latest manifest');
        const manifest = await readJson('latest_manifest.json').catch(() => initialManifest);
        setMeta(manifest);
        const gaussianUrl = 'latest/gaussian_seed/gaussians_seed.ply';
        const meshUrl = 'latest/geometry/scene_mesh.ply';

        setStatus('Loading Gaussian seed');
        const gaussianGeometry = await loadPly(gaussianUrl);
        disposeObject(gaussianObject);
        const pointMaterial = new THREE.PointsMaterial({{
          size: Number(pointSize.value),
          vertexColors: gaussianGeometry.hasAttribute('color'),
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.94
        }});
        gaussianObject = new THREE.Points(gaussianGeometry, pointMaterial);
        gaussianObject.visible = showGaussians.checked;
        scene.add(gaussianObject);

        setStatus('Loading TSDF mesh');
        const meshGeometry = await loadPly(meshUrl);
        meshGeometry.computeVertexNormals();
        disposeObject(meshObject);
        const meshMaterial = new THREE.MeshStandardMaterial({{
          vertexColors: meshGeometry.hasAttribute('color'),
          roughness: 0.8,
          metalness: 0.0,
          transparent: true,
          opacity: Number(meshOpacity.value),
          side: THREE.DoubleSide
        }});
        meshObject = new THREE.Mesh(meshGeometry, meshMaterial);
        meshObject.visible = showMesh.checked;
        scene.add(meshObject);
        fitCamera();
        setStatus('Loaded latest baseline');
      }} catch (err) {{
        console.error(err);
        setStatus('Load failed: ' + err.message);
      }}
    }}

    document.getElementById('reload').addEventListener('click', reloadLatest);
    document.getElementById('reset').addEventListener('click', fitCamera);
    showGaussians.addEventListener('change', () => {{ if (gaussianObject) gaussianObject.visible = showGaussians.checked; }});
    showMesh.addEventListener('change', () => {{ if (meshObject) meshObject.visible = showMesh.checked; }});
    pointSize.addEventListener('input', () => {{
      pointSizeValue.textContent = Number(pointSize.value).toFixed(3);
      if (gaussianObject) gaussianObject.material.size = Number(pointSize.value);
    }});
    meshOpacity.addEventListener('input', () => {{
      meshOpacityValue.textContent = Number(meshOpacity.value).toFixed(2);
      if (meshObject) meshObject.material.opacity = Number(meshOpacity.value);
    }});

    window.addEventListener('resize', () => {{
      const w = viewer.clientWidth;
      const h = Math.max(viewer.clientHeight, 1);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
    reloadLatest();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
