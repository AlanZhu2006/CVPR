#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate monitor-style GS Console viewer.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-html", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve() if args.output_html else baseline_dir / "real2sim_gs_console_viewer.html"
    output_html.write_text(_html(), encoding="utf-8")
    print(output_html)
    return 0


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GS Console Live Monitor</title>
  <style>
    :root { color-scheme: dark; --text:#edf7ff; --muted:#a8bbc8; --line:rgba(255,255,255,.15); --panel:rgba(3,7,10,.78); }
    * { box-sizing:border-box; }
    html, body { margin:0; height:100%; overflow:hidden; background:#030608; color:var(--text); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
    #stage { position:fixed; inset:0; background:#05080b; }
    #rgb { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; background:#071018; }
    #shade { position:absolute; inset:0; pointer-events:none; background:linear-gradient(180deg,rgba(0,0,0,.35),rgba(0,0,0,.05) 30%,rgba(0,0,0,.24)); }
    .hud { position:absolute; left:18px; top:12px; right:18px; height:42px; display:flex; align-items:center; justify-content:space-between; text-shadow:0 2px 8px #000; pointer-events:none; }
    .brand { font-size:18px; font-weight:780; letter-spacing:.04em; }
    .metrics { display:flex; gap:14px; font-size:13px; color:var(--muted); }
    .metrics b { color:var(--text); }
    .tile { position:absolute; overflow:hidden; background:var(--panel); border:1px solid var(--line); box-shadow:0 18px 50px rgba(0,0,0,.4); }
    .tile h2 { position:absolute; z-index:3; left:10px; top:7px; margin:0; font-size:12px; letter-spacing:.05em; text-transform:uppercase; text-shadow:0 2px 7px #000; }
    #gaussianTile { left:18px; top:66px; width:25vw; height:23vh; min-width:280px; min-height:170px; }
    #cloudTile { left:18px; bottom:24px; width:27vw; height:27vh; min-width:310px; min-height:220px; }
    #mapTile { right:24px; bottom:24px; width:28vw; height:30vh; min-width:330px; min-height:250px; }
    #gaussianView, #cloudView, #mapCanvas { width:100%; height:100%; display:block; }
    #gaussianImage { width:100%; height:100%; object-fit:cover; display:block; background:#05080b; }
    #gaussianFallback { position:absolute; inset:0; opacity:0; pointer-events:none; }
    #mapCanvas { background:#dfe7ee; }
    #log { position:absolute; left:50%; top:64px; transform:translateX(-50%); max-width:560px; color:#def7ff; background:rgba(0,0,0,.45); border:1px solid rgba(255,255,255,.13); padding:8px 11px; border-radius:7px; font-size:12px; }
    #commandBar { position:absolute; left:50%; bottom:28px; transform:translateX(-50%); display:flex; gap:8px; padding:8px; background:rgba(2,5,8,.68); border:1px solid var(--line); backdrop-filter:blur(10px); }
    input, button { border:1px solid rgba(255,255,255,.22); background:rgba(14,24,32,.86); color:var(--text); border-radius:7px; padding:9px 11px; font-size:13px; }
    #goalText { width:250px; }
    button { cursor:pointer; }
    .mapLegend { position:absolute; right:9px; top:8px; z-index:3; display:grid; gap:4px; color:#10202a; font-size:11px; font-weight:760; text-shadow:0 1px 0 rgba(255,255,255,.55); }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:-1px; }
    @media (max-width:980px){ #gaussianTile{width:36vw;height:22vh} #cloudTile{width:42vw;height:25vh} #mapTile{width:42vw;height:27vh} #commandBar{left:18px;right:18px;transform:none;justify-content:center} #goalText{width:min(42vw,220px)} }
  </style>
</head>
<body>
  <div id="stage">
    <img id="rgb" src="monitor/latest_rgb.png" alt="live rgb" />
    <div id="shade"></div>
    <div class="hud">
      <div class="brand">GS Console Live RGB Navigation</div>
      <div class="metrics">
        <span>RGB <b id="rgbMetric">live</b></span>
        <span>GS <b id="gsMetric">loading</b></span>
        <span>Cloud <b id="cloudMetric">-</b></span>
        <span>Pose <b id="poseMetric">-</b></span>
      </div>
    </div>
    <div id="log">loading GS console monitor</div>

    <section id="gaussianTile" class="tile"><h2>Gaussian render</h2><img id="gaussianImage" src="monitor/latest_gaussian.png" alt="gsplat render" /><div id="gaussianFallback"></div></section>
    <section id="cloudTile" class="tile"><h2>Colored point cloud</h2><div id="cloudView"></div></section>
    <section id="mapTile" class="tile"><h2>Nav2 style map</h2><div class="mapLegend"><span><i class="dot" style="background:#00c8ff"></i>free/seen</span><span><i class="dot" style="background:#f03752"></i>occupied</span><span><i class="dot" style="background:#1a62ff"></i>trajectory</span></div><canvas id="mapCanvas"></canvas></section>

    <div id="commandBar"><input id="goalText" placeholder="target label / image-goal baseline later" /><button id="goText">Navigate</button></div>
  </div>

  <script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.161.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.161.0/examples/jsm/"}}</script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

    const el = id => document.getElementById(id);
    const state = { map:null, path:null, cloud:{}, gs:{} };
    initCloud();
    initGaussian();
    await reloadAll();
    setInterval(reloadMonitor, 1200);
    setInterval(() => { el('rgb').src = 'monitor/latest_rgb.png?t=' + Date.now(); }, 500);
    setInterval(updateGaussianImage, 1500);
    addEventListener('resize', () => { resizeThree(state.cloud); resizeThree(state.gs); drawMap(); });

    function initScene(rootId, bg) {
      const root = el(rootId);
      const scene = new THREE.Scene(); scene.background = new THREE.Color(bg);
      const camera = new THREE.PerspectiveCamera(55, root.clientWidth / Math.max(root.clientHeight,1), .01, 1000);
      camera.position.set(0,-3,2.5);
      const renderer = new THREE.WebGLRenderer({antialias:true, powerPreference:'high-performance'});
      renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(root.clientWidth, root.clientHeight); root.appendChild(renderer.domElement);
      const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
      scene.add(new THREE.HemisphereLight(0xf4fbff, 0x18222d, 2.2));
      const light = new THREE.DirectionalLight(0xffffff, 1.1); light.position.set(3,-5,8); scene.add(light);
      const pack = {root,scene,camera,renderer,controls,objects:[]};
      function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene,camera); }
      animate(); return pack;
    }
    function resizeThree(pack){ if(!pack.renderer) return; pack.camera.aspect = pack.root.clientWidth / Math.max(pack.root.clientHeight,1); pack.camera.updateProjectionMatrix(); pack.renderer.setSize(pack.root.clientWidth, pack.root.clientHeight); }
    function clear(pack){ for(const o of pack.objects||[]) { pack.scene.remove(o); o.geometry?.dispose(); o.material?.dispose(); } pack.objects=[]; }
    function initCloud(){ state.cloud = initScene('cloudView', 0x05080b); }
    function initGaussian(){ state.gs = initScene('gaussianFallback', 0x05080b); }

    async function reloadAll(){ await Promise.allSettled([reloadGaussian(), reloadMonitor()]); }
    async function reloadMonitor(){
      try {
        const res = await fetch('monitor/live_monitor.json?t=' + Date.now(), {cache:'no-store'});
        state.map = await res.json();
        el('cloudMetric').textContent = `${state.map.shown_point_count}/${state.map.raw_point_count}`;
        el('poseMetric').textContent = `${state.map.trajectory.length}`;
        drawCloud(); drawMap();
      } catch(e) { el('log').textContent = 'waiting for monitor/live_monitor.json'; }
    }
    async function reloadGaussian(){
      try {
        await updateGaussianImage();
        const loader = new PLYLoader();
        const [g, m] = await Promise.all([
          new Promise((ok, bad)=>loader.load('latest/gaussian_seed/gaussians_seed.ply?t='+Date.now(), ok, undefined, bad)),
          new Promise((ok, bad)=>loader.load('latest/geometry/scene_mesh.ply?t='+Date.now(), ok, undefined, bad))
        ]);
        clear(state.gs);
        const pts = new THREE.Points(g, new THREE.PointsMaterial({size:.055, vertexColors:g.hasAttribute('color'), sizeAttenuation:true, transparent:true, opacity:.95}));
        m.computeVertexNormals();
        const mesh = new THREE.Mesh(m, new THREE.MeshStandardMaterial({vertexColors:m.hasAttribute('color'), transparent:true, opacity:.26, roughness:.85, side:THREE.DoubleSide}));
        state.gs.scene.add(mesh, pts); state.gs.objects.push(mesh, pts); fitObjects(state.gs);
        if (el('gsMetric').textContent === 'loading') el('gsMetric').textContent = 'seed fallback';
      } catch(e) { el('gsMetric').textContent = 'waiting'; }
    }
    async function updateGaussianImage(){
      try {
        const meta = await fetch('monitor/latest_gaussian.json?t=' + Date.now(), {cache:'no-store'}).then(r => r.ok ? r.json() : null);
        el('gaussianImage').src = 'monitor/latest_gaussian.png?t=' + Date.now();
        if (meta) el('gsMetric').textContent = `${meta.backend} ${Math.round(meta.render_ms)}ms`;
      } catch(e) {
        if (el('gsMetric').textContent === 'loading') el('gsMetric').textContent = 'seed fallback';
      }
    }
    function drawCloud(){
      const m = state.map; if(!m) return; clear(state.cloud);
      const pts = m.points || [];
      const pos = new Float32Array(pts.length*3), col = new Float32Array(pts.length*3);
      for(let i=0;i<pts.length;i++){ const p=pts[i]; pos[i*3]=p[0]; pos[i*3+1]=p[1]; pos[i*3+2]=p[2]; col[i*3]=p[3]/255; col[i*3+1]=p[4]/255; col[i*3+2]=p[5]/255; }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos,3)); geo.setAttribute('color', new THREE.BufferAttribute(col,3));
      const cloud = new THREE.Points(geo, new THREE.PointsMaterial({size:.035, vertexColors:true, sizeAttenuation:true}));
      state.cloud.scene.add(cloud); state.cloud.objects.push(cloud);
      const traj = makeLine((m.trajectory||[]).map(t=>t.position), 0x1e8bff); if(traj){ state.cloud.scene.add(traj); state.cloud.objects.push(traj); }
      if(state.path){ const path = makeLine(state.path, 0x55e89b); if(path){ state.cloud.scene.add(path); state.cloud.objects.push(path); } }
      fitObjects(state.cloud);
    }
    function makeLine(points, color){ if(!points || points.length<2) return null; const geo=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(p[0],p[1],p[2]))); return new THREE.Line(geo, new THREE.LineBasicMaterial({color})); }
    function fitObjects(pack){ const box = new THREE.Box3(); for(const o of pack.objects||[]) box.expandByObject(o); if(box.isEmpty()) return; const center=box.getCenter(new THREE.Vector3()); const size=box.getSize(new THREE.Vector3()); const span=Math.max(size.x,size.y,size.z,1); pack.controls.target.copy(center); pack.camera.position.copy(center).add(new THREE.Vector3(0,-span*1.35,span*.75)); pack.camera.near=Math.max(.01,span/1000); pack.camera.far=span*20; pack.camera.updateProjectionMatrix(); }
    function drawMap(){
      const m=state.map, canvas=el('mapCanvas'), rect=canvas.getBoundingClientRect(); canvas.width=Math.max(1,rect.width*devicePixelRatio); canvas.height=Math.max(1,rect.height*devicePixelRatio);
      const ctx=canvas.getContext('2d'); ctx.fillStyle='#dfe7ee'; ctx.fillRect(0,0,canvas.width,canvas.height); if(!m) return;
      const pts=m.points||[], traj=m.trajectory||[], path=state.path||[];
      const xs=pts.map(p=>p[0]).concat(traj.map(t=>t.position[0]),path.map(p=>p[0])); const zs=pts.map(p=>p[2]).concat(traj.map(t=>t.position[2]),path.map(p=>p[2])); if(!xs.length) return;
      const minX=percentile(xs,.02)-.5, maxX=percentile(xs,.98)+.5, minZ=percentile(zs,.02)-.5, maxZ=percentile(zs,.98)+.5;
      const scale=Math.min(canvas.width/Math.max(maxX-minX,.1), canvas.height/Math.max(maxZ-minZ,.1)); const ox=(canvas.width-(maxX-minX)*scale)/2, oy=(canvas.height-(maxZ-minZ)*scale)/2;
      const ys=pts.map(p=>p[1]).sort((a,b)=>a-b); const floorY=ys.length?ys[Math.floor(ys.length*.08)]:0; const to2=p=>[ox+(p[0]-minX)*scale, canvas.height-(oy+(p[2]-minZ)*scale)];
      ctx.globalAlpha=.72; for(const p of pts){ const q=to2(p); ctx.fillStyle=p[1]>floorY+.35?'#f03752':'#18c9f4'; ctx.fillRect(q[0],q[1],1.4*devicePixelRatio,1.4*devicePixelRatio); } ctx.globalAlpha=1;
      draw2d(ctx, traj.map(t=>t.position).map(to2), '#125cff', 2.5*devicePixelRatio); draw2d(ctx, path.map(to2), '#20c86b', 4*devicePixelRatio);
      if(traj.length){ const cur=to2(traj[traj.length-1].position); ctx.save(); ctx.translate(cur[0],cur[1]); ctx.rotate(-Math.PI/4); ctx.fillStyle='#111827'; ctx.strokeStyle='#fff'; ctx.lineWidth=1.5*devicePixelRatio; ctx.fillRect(-7*devicePixelRatio,-5*devicePixelRatio,14*devicePixelRatio,10*devicePixelRatio); ctx.strokeRect(-7*devicePixelRatio,-5*devicePixelRatio,14*devicePixelRatio,10*devicePixelRatio); ctx.restore(); }
    }
    function draw2d(ctx, arr, color, width){ if(!arr || arr.length<2) return; ctx.strokeStyle=color; ctx.lineWidth=width; ctx.lineJoin='round'; ctx.lineCap='round'; ctx.beginPath(); arr.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])); ctx.stroke(); }
    function percentile(values,q){ const a=values.filter(Number.isFinite).sort((x,y)=>x-y); return a.length?a[Math.max(0,Math.min(a.length-1,Math.floor(q*(a.length-1))))]:0; }
    el('goText').onclick = () => { el('log').textContent = 'text/image goal API will be wired to GS Console control bridge next'; };
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
