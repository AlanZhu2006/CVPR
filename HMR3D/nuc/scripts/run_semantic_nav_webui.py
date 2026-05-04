#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import heapq
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass
class FrameRecord:
    frame_idx: int
    timestamp_sec: float
    image_path: str
    pose: list[list[float]]
    position: list[float]
    track_ok: bool


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(raw)


def _html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    raw = html.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    return handler.rfile.read(length) if length > 0 else b""


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("annotations", [])
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        pos = item.get("position")
        label = str(item.get("label", "")).strip()
        if not label or not isinstance(pos, list) or len(pos) != 3:
            continue
        out.append(
            {
                "id": str(item.get("id", f"ann_{len(out):04d}")),
                "label": label,
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "kind": str(item.get("kind", "manual")),
                "confidence": float(item.get("confidence", 1.0)),
                "updated_at": str(item.get("updated_at", "")),
                "note": str(item.get("note", "")),
            }
        )
    return out


def _save_annotations(path: Path, annotations: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"annotations": annotations}, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_worker_frames(worker_dir: Path, limit_windows: int = 900) -> list[FrameRecord]:
    records: dict[int, FrameRecord] = {}
    paths = sorted(worker_dir.glob("window_*/worker_result.json"))[-limit_windows:]
    for result_path in paths:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        except Exception:
            continue
        for frame in summary.get("metadata", {}).get("frames", []):
            meta = frame.get("metadata") or {}
            pose = meta.get("pose")
            if pose is None:
                continue
            arr = np.asarray(pose, dtype=np.float32)
            if arr.shape == (3, 4):
                pose4 = np.eye(4, dtype=np.float32)
                pose4[:3, :4] = arr
                arr = pose4
            if arr.shape != (4, 4):
                continue
            frame_idx = int(frame.get("frame_idx", len(records)))
            records[frame_idx] = FrameRecord(
                frame_idx=frame_idx,
                timestamp_sec=float(frame.get("timestamp_sec", 0.0)),
                image_path=str(frame.get("image_path", "")),
                pose=arr.astype(float).tolist(),
                position=arr[:3, 3].astype(float).tolist(),
                track_ok=bool(meta.get("track_ok", True)),
            )
    return [records[idx] for idx in sorted(records)]


def _voxel_clean_sample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    semantic_label: np.ndarray,
    semantic_conf: np.ndarray,
    *,
    voxel_size: float,
    min_voxel_points: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if xyz.shape[0] == 0:
        return xyz, rgb, semantic_label, semantic_conf
    finite = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[finite], rgb[finite]
    semantic_label, semantic_conf = semantic_label[finite], semantic_conf[finite]
    if xyz.shape[0] == 0:
        return xyz, rgb, semantic_label, semantic_conf
    voxel_size = max(float(voxel_size), 1e-4)
    keys = np.floor(xyz / voxel_size).astype(np.int32)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= max(1, int(min_voxel_points))
    if keep.any():
        xyz, rgb, semantic_label, semantic_conf = xyz[keep], rgb[keep], semantic_label[keep], semantic_conf[keep]
    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(7)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz, rgb, semantic_label, semantic_conf = xyz[idx], rgb[idx], semantic_label[idx], semantic_conf[idx]
    return xyz, rgb, semantic_label, semantic_conf


def _load_points(live_dir: Path, max_points: int, voxel_size: float, min_voxel_points: int) -> dict[str, Any]:
    npz_path = live_dir / "live_map.npz"
    json_path = live_dir / "live_map.json"
    updated_at = ""
    active_frames: list[int] = []
    if json_path.exists():
        try:
            live_json = json.loads(json_path.read_text(encoding="utf-8"))
            updated_at = str(live_json.get("updated_at", ""))
            active_frames = [int(x) for x in live_json.get("active_frames", [])]
        except Exception:
            pass
    if npz_path.exists():
        data = np.load(npz_path)
        xyz = np.asarray(data.get("xyz", np.zeros((0, 3))), dtype=np.float32)
        rgb = np.asarray(data.get("rgb", np.zeros((xyz.shape[0], 3))), dtype=np.uint8)
        semantic_label = np.asarray(data.get("semantic_label", np.full((xyz.shape[0],), -1)), dtype=np.int32)
        semantic_conf = np.asarray(data.get("semantic_conf", np.zeros((xyz.shape[0],))), dtype=np.float32)
    elif json_path.exists():
        live_json = json.loads(json_path.read_text(encoding="utf-8"))
        pts = np.asarray(live_json.get("points", []), dtype=np.float32)
        xyz = pts[:, :3].astype(np.float32) if pts.size else np.zeros((0, 3), dtype=np.float32)
        rgb = pts[:, 3:6].astype(np.uint8) if pts.size else np.zeros((0, 3), dtype=np.uint8)
        semantic_label = pts[:, 7].astype(np.int32) if pts.shape[1] > 7 else np.full((xyz.shape[0],), -1, dtype=np.int32)
        semantic_conf = pts[:, 8].astype(np.float32) if pts.shape[1] > 8 else np.zeros((xyz.shape[0],), dtype=np.float32)
    else:
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        semantic_label = np.zeros((0,), dtype=np.int32) - 1
        semantic_conf = np.zeros((0,), dtype=np.float32)
    raw_count = int(xyz.shape[0])
    xyz, rgb, semantic_label, semantic_conf = _voxel_clean_sample(
        xyz,
        rgb,
        semantic_label,
        semantic_conf,
        voxel_size=voxel_size,
        min_voxel_points=min_voxel_points,
        max_points=max_points,
    )
    bbox_min = xyz.min(axis=0).astype(float).tolist() if xyz.shape[0] else [-1.0, -1.0, -1.0]
    bbox_max = xyz.max(axis=0).astype(float).tolist() if xyz.shape[0] else [1.0, 1.0, 1.0]
    points = np.concatenate(
        [
            xyz.astype(np.float32),
            rgb.astype(np.float32),
            semantic_label.reshape(-1, 1).astype(np.float32),
            semantic_conf.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    )
    return {
        "points": points.tolist(),
        "raw_point_count": raw_count,
        "shown_point_count": int(xyz.shape[0]),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "updated_at": updated_at,
        "active_frames": active_frames,
        "semantic_summary": _semantic_summary(xyz, semantic_label, semantic_conf),
    }


def _semantic_summary(xyz: np.ndarray, labels: np.ndarray, conf: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for label in sorted(set(int(x) for x in labels.tolist() if int(x) >= 0)):
        mask = labels == label
        if not mask.any():
            continue
        center = xyz[mask].mean(axis=0).astype(float).tolist()
        out.append(
            {
                "id": f"semantic_{label}",
                "label": COCO_NAMES[label] if 0 <= label < len(COCO_NAMES) else f"class {label}",
                "class_id": int(label),
                "position": center,
                "kind": "detected",
                "confidence": float(conf[mask].mean()) if conf.size else 0.0,
                "count": int(mask.sum()),
            }
        )
    return out


def _read_latest_manifest(real2sim_dir: Path) -> dict[str, Any]:
    path = real2sim_dir / "latest_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _nearest_annotation(query: str, annotations: list[dict[str, Any]], semantic_summary: list[dict[str, Any]]) -> dict[str, Any] | None:
    q = query.strip().lower()
    if not q:
        return None
    candidates = annotations + semantic_summary
    exact = [c for c in candidates if str(c.get("label", "")).lower() == q]
    if exact:
        return exact[-1]
    contains = [c for c in candidates if q in str(c.get("label", "")).lower() or str(c.get("label", "")).lower() in q]
    if contains:
        return contains[-1]
    return None


def _build_grid(points: np.ndarray, start: np.ndarray, goal: np.ndarray, resolution: float = 0.18) -> tuple[np.ndarray, np.ndarray, float]:
    if points.shape[0] == 0:
        mn = np.minimum(start[[0, 2]], goal[[0, 2]]) - 2.0
        mx = np.maximum(start[[0, 2]], goal[[0, 2]]) + 2.0
        return np.zeros((32, 32), dtype=bool), mn, max(resolution, float((mx - mn).max() / 31.0))
    xz = points[:, [0, 2]]
    mn = np.minimum(np.percentile(xz, 1, axis=0), np.minimum(start[[0, 2]], goal[[0, 2]])) - 0.6
    mx = np.maximum(np.percentile(xz, 99, axis=0), np.maximum(start[[0, 2]], goal[[0, 2]])) + 0.6
    size = np.maximum(np.ceil((mx - mn) / resolution).astype(int) + 1, 8)
    size = np.minimum(size, 260)
    resolution = float(max((mx - mn).max() / max(size.max() - 1, 1), resolution))
    ij = np.floor((xz - mn) / resolution).astype(np.int32)
    valid = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < size[0]) & (ij[:, 1] < size[1])
    ij = ij[valid]
    y = points[valid, 1]
    floor = float(np.percentile(points[:, 1], 8))
    high = y > floor + 0.35
    occ = np.zeros((int(size[0]), int(size[1])), dtype=np.int32)
    for cell in ij[high]:
        occ[cell[0], cell[1]] += 1
    blocked = occ >= 4
    return blocked, mn, resolution


def _astar_path(points: np.ndarray, start_pos: list[float], goal_pos: list[float]) -> list[list[float]]:
    start = np.asarray(start_pos, dtype=np.float32)
    goal = np.asarray(goal_pos, dtype=np.float32)
    blocked, origin, res = _build_grid(points, start, goal)
    shape = blocked.shape

    def to_cell(pos: np.ndarray) -> tuple[int, int]:
        cell = np.floor((pos[[0, 2]] - origin) / res).astype(int)
        return int(np.clip(cell[0], 0, shape[0] - 1)), int(np.clip(cell[1], 0, shape[1] - 1))

    def to_world(cell: tuple[int, int], y: float) -> list[float]:
        xz = origin + (np.asarray(cell, dtype=np.float32) + 0.5) * res
        return [float(xz[0]), float(y), float(xz[1])]

    s = to_cell(start)
    g = to_cell(goal)
    blocked[s] = False
    blocked[g] = False
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, s)]
    came: dict[tuple[int, int], tuple[int, int] | None] = {s: None}
    cost = {s: 0.0}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == g:
            break
        for dx, dz in dirs:
            nb = (cur[0] + dx, cur[1] + dz)
            if nb[0] < 0 or nb[1] < 0 or nb[0] >= shape[0] or nb[1] >= shape[1] or blocked[nb]:
                continue
            step = math.sqrt(dx * dx + dz * dz)
            new_cost = cost[cur] + step
            if nb not in cost or new_cost < cost[nb]:
                cost[nb] = new_cost
                h = math.hypot(nb[0] - g[0], nb[1] - g[1])
                heapq.heappush(pq, (new_cost + h, nb))
                came[nb] = cur
    if g not in came:
        return [start.astype(float).tolist(), goal.astype(float).tolist()]
    cells = []
    cur: tuple[int, int] | None = g
    while cur is not None:
        cells.append(cur)
        cur = came[cur]
    cells.reverse()
    y = float(start[1])
    path = [start.astype(float).tolist()]
    path.extend(to_world(c, y) for c in cells[1:-1: max(1, len(cells) // 80)])
    path.append(goal.astype(float).tolist())
    return path


def _image_descriptor(path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    small = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).astype(np.float32)
    hist = cv2.normalize(hist, hist).reshape(-1)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=350)
    _, desc = orb.detectAndCompute(gray, None)
    return hist, desc


def _decode_data_url(data_url: str) -> np.ndarray | None:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _match_image_goal(image: np.ndarray, frames: list[FrameRecord], stride: int = 6, max_frames: int = 500) -> dict[str, Any]:
    if image is None:
        raise ValueError("Could not decode uploaded image.")
    query_path = ""
    small = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
    qhist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).astype(np.float32)
    qhist = cv2.normalize(qhist, qhist).reshape(-1)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=350)
    _, qdesc = orb.detectAndCompute(gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best: tuple[float, FrameRecord | None, dict[str, float]] = (-1e9, None, {})
    candidates = frames[-max_frames:: max(1, stride)]
    for frame in candidates:
        if not frame.image_path or not Path(frame.image_path).exists():
            continue
        hist, desc = _image_descriptor(frame.image_path)
        if hist is None:
            continue
        hist_score = float(cv2.compareHist(qhist, hist, cv2.HISTCMP_CORREL))
        match_score = 0.0
        if qdesc is not None and desc is not None and len(qdesc) and len(desc):
            matches = bf.match(qdesc, desc)
            good = [m for m in matches if m.distance < 52]
            match_score = float(len(good)) / max(20.0, float(min(len(qdesc), len(desc))))
        score = hist_score + 1.8 * match_score
        if score > best[0]:
            best = (score, frame, {"hist": hist_score, "orb": match_score})
            query_path = frame.image_path
    if best[1] is None:
        raise ValueError("No comparable history frames found.")
    return {
        "score": float(best[0]),
        "components": best[2],
        "frame": best[1].__dict__,
        "matched_image_path": query_path,
    }


def _build_index_html(gs_url: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Semantic Nav GS Baseline</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080b0d;
      --panel: #10171d;
      --panel2: #16212a;
      --line: #2a3844;
      --text: #edf5f2;
      --muted: #9fb1ac;
      --accent: #57d3ff;
      --green: #7be0a3;
      --warn: #ffd46b;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; height:100vh; overflow:hidden; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, sans-serif; }}
    #app {{ height:100vh; display:grid; grid-template-columns: 330px minmax(0, 1fr) 38vw; }}
    aside {{ background:linear-gradient(180deg,#121b22,#0b1116); border-right:1px solid var(--line); padding:16px; overflow:auto; }}
    main {{ position:relative; min-width:0; background:#050708; }}
    iframe {{ width:100%; height:100%; border:0; background:#050708; border-left:1px solid var(--line); }}
    h1 {{ margin:0; font-size:19px; letter-spacing:0; }}
    .sub {{ color:var(--muted); font-size:13px; line-height:1.45; margin:6px 0 16px; }}
    .section {{ border-top:1px solid rgba(255,255,255,.08); padding-top:12px; margin-top:12px; display:grid; gap:9px; }}
    label {{ display:grid; gap:6px; color:var(--muted); font-size:12px; }}
    input, button, select {{ border:1px solid #334452; background:#14202a; color:var(--text); border-radius:7px; padding:9px 10px; font-size:13px; min-width:0; }}
    button {{ cursor:pointer; }}
    button:hover {{ border-color:var(--accent); }}
    .row {{ display:flex; justify-content:space-between; gap:12px; color:var(--muted); font-size:13px; padding:6px 0; }}
    .row b {{ color:var(--text); font-weight:650; text-align:right; overflow-wrap:anywhere; }}
    #viewer {{ position:absolute; inset:0; }}
    #chips {{ position:absolute; left:12px; bottom:12px; display:flex; flex-wrap:wrap; gap:8px; pointer-events:none; }}
    .chip {{ background:rgba(10,15,20,.78); border:1px solid rgba(255,255,255,.13); border-radius:7px; padding:8px 10px; color:var(--muted); font-size:12px; backdrop-filter:blur(8px); }}
    #results {{ display:grid; gap:7px; }}
    .card {{ border:1px solid rgba(255,255,255,.1); border-radius:7px; background:#101922; padding:9px; font-size:12px; color:var(--muted); }}
    .card b {{ color:var(--text); }}
    a {{ color:var(--accent); }}
    @media (max-width: 1200px) {{ #app {{ grid-template-columns:310px 1fr; }} iframe {{ display:none; }} }}
  </style>
</head>
<body>
  <div id="app">
    <aside>
      <h1>Semantic Nav Baseline</h1>
      <div class="sub">Live RGB map, denoised point cloud, trajectory, semantic updates, text goal and image-goal pose retrieval.</div>
      <div class="section">
        <button id="reload">Reload map</button>
        <button id="reset">Reset camera</button>
        <a href="{gs_url}" target="_blank">Open Gaussian/mesh viewer</a>
      </div>
      <div class="section">
        <label>Text goal
          <input id="goalText" placeholder="chair, desk, target room..." />
        </label>
        <button id="goText">Navigate to semantic target</button>
        <label>Image goal
          <input id="imageGoal" type="file" accept="image/*" />
        </label>
      </div>
      <div class="section">
        <label>Add/update semantic marker
          <input id="annLabel" placeholder="label at current camera pose" />
        </label>
        <button id="addAnn">Add marker at current pose</button>
      </div>
      <div class="section">
        <div class="row"><span>Status</span><b id="status">loading</b></div>
        <div class="row"><span>Points</span><b id="points">-</b></div>
        <div class="row"><span>Trajectory</span><b id="traj">-</b></div>
        <div class="row"><span>Semantics</span><b id="semantics">-</b></div>
        <div class="row"><span>Updated</span><b id="updated">-</b></div>
      </div>
      <div id="results" class="section"></div>
    </aside>
    <main>
      <div id="viewer"></div>
      <div id="chips">
        <div class="chip">drag rotate</div>
        <div class="chip">wheel zoom</div>
        <div class="chip">orange: trajectory</div>
        <div class="chip">green: planned path</div>
      </div>
    </main>
    <iframe src="{gs_url}"></iframe>
  </div>
  <script type="importmap">
    {{"imports":{{"three":"https://unpkg.com/three@0.161.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.161.0/examples/jsm/"}}}}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

    const state = {{ map:null, scene:null, camera:null, renderer:null, controls:null, pointObj:null, trajObj:null, markerGroup:null, pathObj:null }};
    const el = id => document.getElementById(id);
    initThree();
    await reload();
    setInterval(reload, 5000);

    function initThree() {{
      const root = el('viewer');
      state.scene = new THREE.Scene();
      state.scene.background = new THREE.Color(0x050708);
      state.camera = new THREE.PerspectiveCamera(58, root.clientWidth / root.clientHeight, 0.01, 1000);
      state.camera.position.set(0, -3, 4);
      state.renderer = new THREE.WebGLRenderer({{ antialias:true }});
      state.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      state.renderer.setSize(root.clientWidth, root.clientHeight);
      root.appendChild(state.renderer.domElement);
      state.controls = new OrbitControls(state.camera, state.renderer.domElement);
      state.controls.enableDamping = true;
      state.scene.add(new THREE.AmbientLight(0xffffff, .9));
      const grid = new THREE.GridHelper(12, 24, 0x2b4655, 0x15242d);
      grid.rotation.x = Math.PI / 2;
      state.scene.add(grid);
      state.markerGroup = new THREE.Group();
      state.scene.add(state.markerGroup);
      addEventListener('resize', () => {{
        state.camera.aspect = root.clientWidth / root.clientHeight;
        state.camera.updateProjectionMatrix();
        state.renderer.setSize(root.clientWidth, root.clientHeight);
      }});
      animate();
    }}
    function animate() {{
      requestAnimationFrame(animate);
      state.controls.update();
      state.renderer.render(state.scene, state.camera);
    }}
    async function reload() {{
      try {{
        const res = await fetch('/api/map?t=' + Date.now(), {{cache:'no-store'}});
        state.map = await res.json();
        el('status').textContent = 'live';
        el('points').textContent = `${{state.map.shown_point_count}} / ${{state.map.raw_point_count}}`;
        el('traj').textContent = `${{state.map.trajectory.length}} poses`;
        el('semantics').textContent = `${{state.map.annotations.length + state.map.semantic_summary.length}} labels`;
        el('updated').textContent = state.map.updated_at || '-';
        drawMap();
      }} catch (e) {{
        el('status').textContent = 'waiting';
      }}
    }}
    function drawMap() {{
      const m = state.map;
      if (!m) return;
      if (state.pointObj) state.scene.remove(state.pointObj);
      const pts = m.points || [];
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(pts.length * 3);
      const col = new Float32Array(pts.length * 3);
      for (let i=0;i<pts.length;i++) {{
        const p = pts[i];
        pos[i*3+0]=p[0]; pos[i*3+1]=p[1]; pos[i*3+2]=p[2];
        if (p[6] >= 0) {{
          const c = semanticColor(p[6]);
          col[i*3+0]=c[0]; col[i*3+1]=c[1]; col[i*3+2]=c[2];
        }} else {{
          col[i*3+0]=p[3]/255; col[i*3+1]=p[4]/255; col[i*3+2]=p[5]/255;
        }}
      }}
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      state.pointObj = new THREE.Points(geo, new THREE.PointsMaterial({{ size:0.035, vertexColors:true, sizeAttenuation:true }}));
      state.scene.add(state.pointObj);
      drawLine('trajObj', m.trajectory.map(t => t.position), 0xf2a14d, 2);
      drawMarkers();
      fitIfNeeded();
    }}
    function drawLine(key, arr, color, width) {{
      if (state[key]) state.scene.remove(state[key]);
      if (!arr || arr.length < 2) return;
      const geo = new THREE.BufferGeometry().setFromPoints(arr.map(p => new THREE.Vector3(p[0], p[1], p[2])));
      state[key] = new THREE.Line(geo, new THREE.LineBasicMaterial({{ color, linewidth:width }}));
      state.scene.add(state[key]);
    }}
    function drawMarkers() {{
      state.markerGroup.clear();
      const items = [...(state.map.annotations||[]), ...(state.map.semantic_summary||[])];
      for (const item of items) {{
        const p = item.position;
        const color = item.kind === 'manual' ? 0xffd46b : 0x57d3ff;
        const s = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshBasicMaterial({{ color }}));
        s.position.set(p[0], p[1], p[2]);
        state.markerGroup.add(s);
      }}
    }}
    function semanticColor(id) {{
      const palette = [[.95,.2,.3],[.1,.75,.95],[.5,.95,.45],[.95,.8,.2],[.8,.45,.95],[.2,.95,.7]];
      return palette[Math.abs(Math.floor(id)) % palette.length];
    }}
    function fitIfNeeded() {{
      if (state._fit || !state.map) return;
      state._fit = true;
      const mn = state.map.bbox_min, mx = state.map.bbox_max;
      const center = new THREE.Vector3((mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2);
      const span = Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 1);
      state.controls.target.copy(center);
      state.camera.position.copy(center.clone().add(new THREE.Vector3(0, -span*1.4, span*.9)));
      state.camera.near = Math.max(0.01, span/1000);
      state.camera.far = span * 20;
      state.camera.updateProjectionMatrix();
    }}
    function showResult(title, body) {{
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = `<b>${{title}}</b><br>${{body}}`;
      el('results').prepend(div);
    }}
    async function postJson(url, payload) {{
      const res = await fetch(url, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload)}});
      return await res.json();
    }}
    el('reload').onclick = reload;
    el('reset').onclick = () => {{ state._fit=false; fitIfNeeded(); }};
    el('goText').onclick = async () => {{
      const query = el('goalText').value;
      const r = await postJson('/api/goal', {{query}});
      if (r.path) drawLine('pathObj', r.path, 0x7be0a3, 4);
      showResult('Text goal', r.ok ? `${{r.target.label}} · path ${{r.path.length}} waypoints` : r.error);
    }};
    el('addAnn').onclick = async () => {{
      const label = el('annLabel').value.trim();
      if (!label || !state.map || !state.map.current_pose) return;
      const r = await postJson('/api/annotations', {{label, position:state.map.current_pose.position, note:'added from webui'}});
      showResult('Semantic update', r.ok ? `saved ${{label}}` : r.error);
      await reload();
    }};
    el('imageGoal').onchange = async e => {{
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async () => {{
        const r = await postJson('/api/image_goal', {{image:reader.result}});
        if (r.path) drawLine('pathObj', r.path, 0x7be0a3, 4);
        showResult('Image goal', r.ok ? `matched frame ${{r.match.frame.frame_idx}} · score ${{r.match.score.toFixed(3)}}` : r.error);
      }};
      reader.readAsDataURL(file);
    }};
  </script>
</body>
</html>
"""


def _build_monitor_html(gs_url: str) -> str:
    html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live RGB Nav GS Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      --text: #edf7ff;
      --muted: #a8bcc9;
      --cyan: #24d6ff;
      --blue: #168dff;
      --green: #54e89b;
      --red: #ff5264;
      --panel: rgba(3, 7, 10, .78);
      --line: rgba(255,255,255,.16);
    }
    * { box-sizing: border-box; }
    html, body { margin:0; height:100%; overflow:hidden; background:#030507; color:var(--text); font-family:Inter, ui-sans-serif, system-ui, sans-serif; }
    #stage { position:fixed; inset:0; background:#05080b; }
    #rgb { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; background:#080d12; }
    #shade { position:absolute; inset:0; pointer-events:none; background:linear-gradient(180deg, rgba(0,0,0,.34), rgba(0,0,0,.04) 28%, rgba(0,0,0,.25)); }
    .hud { position:absolute; left:18px; top:14px; right:18px; height:42px; display:flex; align-items:center; justify-content:space-between; pointer-events:none; text-shadow:0 2px 8px #000; }
    .brand { font-weight:750; letter-spacing:.04em; font-size:18px; }
    .metrics { display:flex; gap:14px; font-size:13px; color:var(--muted); }
    .metrics b { color:var(--text); font-weight:700; }
    .tile { position:absolute; background:var(--panel); border:1px solid var(--line); overflow:hidden; box-shadow:0 18px 50px rgba(0,0,0,.38); }
    .tile h2 { position:absolute; left:10px; top:7px; z-index:2; margin:0; font-size:12px; font-weight:750; letter-spacing:.05em; text-transform:uppercase; color:#eaf8ff; text-shadow:0 1px 6px #000; }
    #gaussianTile { left:18px; top:66px; width:25vw; height:23vh; min-width:280px; min-height:170px; }
    #cloudTile { left:18px; bottom:24px; width:27vw; height:27vh; min-width:310px; min-height:220px; }
    #mapTile { right:24px; bottom:24px; width:28vw; height:30vh; min-width:330px; min-height:250px; }
    #gaussianFrame { width:100%; height:100%; border:0; transform:scale(1.0); background:#05080b; }
    #cloudView, #mapCanvas { width:100%; height:100%; display:block; }
    #mapCanvas { background:#dfe6ee; }
    #commandBar { position:absolute; left:50%; bottom:28px; transform:translateX(-50%); display:flex; gap:8px; padding:8px; background:rgba(2,5,8,.68); border:1px solid var(--line); backdrop-filter:blur(10px); }
    input, button { border:1px solid rgba(255,255,255,.22); background:rgba(14,24,32,.86); color:var(--text); border-radius:7px; padding:9px 11px; font-size:13px; }
    #goalText { width:240px; }
    button { cursor:pointer; }
    button:hover { border-color:var(--cyan); }
    #log { position:absolute; left:50%; top:64px; transform:translateX(-50%); max-width:520px; color:#dff7ff; background:rgba(0,0,0,.45); border:1px solid rgba(255,255,255,.13); padding:8px 11px; border-radius:7px; font-size:12px; opacity:.95; }
    .mapLegend { position:absolute; right:9px; top:8px; z-index:2; display:grid; gap:4px; color:#10202a; font-size:11px; font-weight:700; text-shadow:0 1px 0 rgba(255,255,255,.55); }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:-1px; }
    @media (max-width: 980px) {
      #gaussianTile { width:36vw; height:22vh; }
      #cloudTile { width:42vw; height:25vh; }
      #mapTile { width:42vw; height:27vh; }
      #commandBar { left:18px; right:18px; transform:none; justify-content:center; }
      #goalText { width:min(42vw, 220px); }
    }
  </style>
</head>
<body>
  <div id="stage">
    <img id="rgb" alt="live rgb" />
    <div id="shade"></div>
    <div class="hud">
      <div class="brand">Live RGB Navigation / Gaussian Monitor</div>
      <div class="metrics">
        <span>RGB <b id="rgbStatus">waiting</b></span>
        <span>Cloud <b id="pointMetric">-</b></span>
        <span>Pose <b id="poseMetric">-</b></span>
        <span>FPS <b id="fpsMetric">-</b></span>
      </div>
    </div>
    <div id="log">initializing live monitor</div>

    <section id="gaussianTile" class="tile">
      <h2>Gaussian render</h2>
      <iframe id="gaussianFrame" src="__GS_URL__"></iframe>
    </section>

    <section id="cloudTile" class="tile">
      <h2>Colored point cloud</h2>
      <div id="cloudView"></div>
    </section>

    <section id="mapTile" class="tile">
      <h2>Nav2 style map</h2>
      <div class="mapLegend">
        <span><i class="dot" style="background:#00c8ff"></i>free/seen</span>
        <span><i class="dot" style="background:#f03752"></i>occupied</span>
        <span><i class="dot" style="background:#1a62ff"></i>trajectory</span>
      </div>
      <canvas id="mapCanvas"></canvas>
    </section>

    <div id="commandBar">
      <input id="goalText" placeholder="target label, e.g. chair" />
      <button id="goText">Navigate</button>
      <input id="imageGoal" type="file" accept="image/*" />
    </div>
  </div>

  <script type="importmap">
    {"imports":{"three":"https://unpkg.com/three@0.161.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.161.0/examples/jsm/"}}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const el = id => document.getElementById(id);
    const state = { map:null, path:null, lastFrames:[], scene:null, camera:null, renderer:null, controls:null, points:null, traj:null, path3d:null, fit:false };
    const rgb = el('rgb');
    let frameCounter = 0, lastFpsT = performance.now();

    initCloud();
    await refresh();
    setInterval(refresh, 5000);
    setInterval(updateRgb, 500);

    function updateRgb() {
      rgb.src = '/api/latest_image?t=' + Date.now();
      el('rgbStatus').textContent = 'live';
      frameCounter++;
      const now = performance.now();
      if (now - lastFpsT > 1000) {
        el('fpsMetric').textContent = frameCounter.toString();
        frameCounter = 0;
        lastFpsT = now;
      }
    }

    async function refresh() {
      try {
        const res = await fetch('/api/map?t=' + Date.now(), {cache:'no-store'});
        state.map = await res.json();
        el('pointMetric').textContent = `${state.map.shown_point_count}/${state.map.raw_point_count}`;
        el('poseMetric').textContent = `${state.map.trajectory.length}`;
        drawCloud();
        drawMap();
      } catch (e) {
        el('log').textContent = 'waiting for map API';
      }
    }

    function initCloud() {
      const root = el('cloudView');
      state.scene = new THREE.Scene();
      state.scene.background = new THREE.Color(0x05080b);
      state.camera = new THREE.PerspectiveCamera(55, root.clientWidth / root.clientHeight, 0.02, 1000);
      state.camera.position.set(0, -3, 2.5);
      state.renderer = new THREE.WebGLRenderer({antialias:true});
      state.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      state.renderer.setSize(root.clientWidth, root.clientHeight);
      root.appendChild(state.renderer.domElement);
      state.controls = new OrbitControls(state.camera, state.renderer.domElement);
      state.controls.enableDamping = true;
      state.scene.add(new THREE.AmbientLight(0xffffff, .85));
      addEventListener('resize', () => {
        state.camera.aspect = root.clientWidth / Math.max(root.clientHeight, 1);
        state.camera.updateProjectionMatrix();
        state.renderer.setSize(root.clientWidth, root.clientHeight);
        drawMap();
      });
      animateCloud();
    }

    function animateCloud() {
      requestAnimationFrame(animateCloud);
      state.controls.update();
      state.renderer.render(state.scene, state.camera);
    }

    function drawCloud() {
      const m = state.map;
      if (!m) return;
      if (state.points) state.scene.remove(state.points);
      if (state.traj) state.scene.remove(state.traj);
      if (state.path3d) state.scene.remove(state.path3d);
      const pts = m.points || [];
      const pos = new Float32Array(pts.length * 3);
      const col = new Float32Array(pts.length * 3);
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        pos[i*3+0] = p[0]; pos[i*3+1] = p[1]; pos[i*3+2] = p[2];
        col[i*3+0] = p[3] / 255; col[i*3+1] = p[4] / 255; col[i*3+2] = p[5] / 255;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      state.points = new THREE.Points(geo, new THREE.PointsMaterial({size:0.035, vertexColors:true, sizeAttenuation:true}));
      state.scene.add(state.points);
      state.traj = makeLine((m.trajectory || []).map(t => t.position), 0x1e8bff);
      if (state.traj) state.scene.add(state.traj);
      if (state.path) {
        state.path3d = makeLine(state.path, 0x55e89b);
        if (state.path3d) state.scene.add(state.path3d);
      }
      if (!state.fit) fitCloud();
    }

    function makeLine(points, color) {
      if (!points || points.length < 2) return null;
      const geo = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
      return new THREE.Line(geo, new THREE.LineBasicMaterial({color}));
    }

    function fitCloud() {
      if (!state.map) return;
      state.fit = true;
      const mn = state.map.bbox_min, mx = state.map.bbox_max;
      const center = new THREE.Vector3((mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2);
      const span = Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 1);
      state.controls.target.copy(center);
      state.camera.position.copy(center.clone().add(new THREE.Vector3(0, -span*1.4, span*.75)));
      state.camera.near = Math.max(0.01, span / 1000);
      state.camera.far = span * 20;
      state.camera.updateProjectionMatrix();
    }

    function drawMap() {
      const m = state.map;
      const canvas = el('mapCanvas');
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
      canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#dfe6ee';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!m) return;
      const pts = m.points || [];
      const traj = m.trajectory || [];
      const path = state.path || [];
      const xs = pts.map(p => p[0]).concat(traj.map(t => t.position[0]), path.map(p => p[0]));
      const zs = pts.map(p => p[2]).concat(traj.map(t => t.position[2]), path.map(p => p[2]));
      if (!xs.length) return;
      const minX = percentile(xs, .02) - .5, maxX = percentile(xs, .98) + .5;
      const minZ = percentile(zs, .02) - .5, maxZ = percentile(zs, .98) + .5;
      const scale = Math.min(canvas.width / Math.max(maxX-minX, .1), canvas.height / Math.max(maxZ-minZ, .1));
      const ox = (canvas.width - (maxX-minX)*scale) / 2;
      const oy = (canvas.height - (maxZ-minZ)*scale) / 2;
      const yVals = pts.map(p => p[1]).sort((a,b)=>a-b);
      const floorY = yVals.length ? yVals[Math.floor(yVals.length * .08)] : 0;
      const to2 = p => [ox + (p[0]-minX)*scale, canvas.height - (oy + (p[2]-minZ)*scale)];
      ctx.globalAlpha = .72;
      for (const p of pts) {
        const q = to2(p);
        const high = p[1] > floorY + .35;
        ctx.fillStyle = high ? '#f03752' : '#18c9f4';
        ctx.fillRect(q[0], q[1], 1.6*devicePixelRatio, 1.6*devicePixelRatio);
      }
      ctx.globalAlpha = 1;
      draw2dLine(ctx, traj.map(t => t.position).map(to2), '#125cff', 2.5*devicePixelRatio);
      draw2dLine(ctx, path.map(to2), '#20c86b', 4*devicePixelRatio);
      if (traj.length) {
        const cur = to2(traj[traj.length-1].position);
        ctx.save();
        ctx.translate(cur[0], cur[1]);
        ctx.rotate(-Math.PI/4);
        ctx.fillStyle = '#111827';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5*devicePixelRatio;
        ctx.fillRect(-7*devicePixelRatio, -5*devicePixelRatio, 14*devicePixelRatio, 10*devicePixelRatio);
        ctx.strokeRect(-7*devicePixelRatio, -5*devicePixelRatio, 14*devicePixelRatio, 10*devicePixelRatio);
        ctx.restore();
      }
    }

    function draw2dLine(ctx, arr, color, width) {
      if (!arr || arr.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      arr.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.stroke();
    }

    function percentile(values, q) {
      const arr = values.filter(Number.isFinite).sort((a,b)=>a-b);
      if (!arr.length) return 0;
      return arr[Math.max(0, Math.min(arr.length-1, Math.floor(q * (arr.length-1))))];
    }

    async function postJson(url, payload) {
      const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      return await res.json();
    }

    el('goText').onclick = async () => {
      const query = el('goalText').value.trim();
      if (!query) return;
      const r = await postJson('/api/goal', {query});
      if (r.ok) {
        state.path = r.path;
        drawCloud();
        drawMap();
        el('log').textContent = `target ${r.target.label}: ${r.path.length} waypoints`;
      } else {
        el('log').textContent = r.error || 'goal failed';
      }
    };

    el('imageGoal').onchange = e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async () => {
        const r = await postJson('/api/image_goal', {image:reader.result});
        if (r.ok) {
          state.path = r.path;
          drawCloud();
          drawMap();
          el('log').textContent = `image goal matched frame ${r.match.frame.frame_idx}, score ${r.match.score.toFixed(3)}`;
        } else {
          el('log').textContent = r.error || 'image goal failed';
        }
      };
      reader.readAsDataURL(file);
    };
  </script>
</body>
</html>
"""
    return html.replace("__GS_URL__", gs_url)


class SemanticNavServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.live_dir = Path(args.live_dir).expanduser().resolve()
        self.worker_dir = Path(args.worker_dir).expanduser().resolve() if args.worker_dir else self.live_dir / "worker"
        self.real2sim_dir = Path(args.real2sim_dir).expanduser().resolve()
        self.annotations_path = Path(args.annotations).expanduser().resolve()
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        self.gs_url = args.gs_url
        self._last_snapshot_time = 0.0
        self._snapshot: dict[str, Any] | None = None
        self._last_image_scan_time = 0.0
        self._latest_image_path: Path | None = None

    def latest_image_path(self) -> Path | None:
        now = time.time()
        if self._latest_image_path is not None and now - self._last_image_scan_time < 0.5:
            return self._latest_image_path
        rgb_dir = self.live_dir / "rgb_stream"
        latest: Path | None = None
        if rgb_dir.exists():
            try:
                latest = max(rgb_dir.glob("*.png"), key=lambda path: path.name)
            except ValueError:
                latest = None
        if latest is None:
            snap = self.snapshot()
            current = snap.get("current_pose") or {}
            candidate = Path(str(current.get("image_path", ""))).expanduser()
            latest = candidate if candidate.exists() else None
        self._latest_image_path = latest
        self._last_image_scan_time = now
        return latest

    def snapshot(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force and self._snapshot is not None and now - self._last_snapshot_time < self.args.cache_sec:
            return self._snapshot
        points = _load_points(
            self.live_dir,
            max_points=self.args.max_points,
            voxel_size=self.args.clean_voxel_size,
            min_voxel_points=self.args.clean_min_voxel_points,
        )
        frames = _load_worker_frames(self.worker_dir, limit_windows=self.args.worker_windows)
        annotations = _load_annotations(self.annotations_path)
        latest_manifest = _read_latest_manifest(self.real2sim_dir)
        trajectory = [frame.__dict__ for frame in frames]
        current_pose = trajectory[-1] if trajectory else None
        points_arr = np.asarray(points["points"], dtype=np.float32)
        xyz = points_arr[:, :3] if points_arr.size else np.zeros((0, 3), dtype=np.float32)
        self._snapshot = {
            **points,
            "schema": "semantic_nav_baseline.v1",
            "trajectory": trajectory,
            "current_pose": current_pose,
            "annotations": annotations,
            "latest_real2sim": latest_manifest,
            "_xyz_cache": xyz,
        }
        self._last_snapshot_time = now
        return self._snapshot

    def public_snapshot(self) -> dict[str, Any]:
        snap = dict(self.snapshot())
        snap.pop("_xyz_cache", None)
        return snap

    def resolve_goal(self, query: str) -> dict[str, Any]:
        snap = self.snapshot(force=True)
        current = snap.get("current_pose")
        if current is None:
            return {"ok": False, "error": "No current camera pose available yet."}
        target = _nearest_annotation(query, snap["annotations"], snap["semantic_summary"])
        if target is None:
            labels = [x["label"] for x in snap["annotations"] + snap["semantic_summary"]]
            return {"ok": False, "error": f"No semantic target matched '{query}'. Known: {', '.join(labels[:20])}"}
        path = _astar_path(snap["_xyz_cache"], current["position"], target["position"])
        return {"ok": True, "target": target, "start": current, "path": path}

    def add_annotation(self, payload: dict[str, Any]) -> dict[str, Any]:
        label = str(payload.get("label", "")).strip()
        pos = payload.get("position")
        if not label or not isinstance(pos, list) or len(pos) != 3:
            return {"ok": False, "error": "Need label and position [x,y,z]."}
        annotations = _load_annotations(self.annotations_path)
        item = {
            "id": f"ann_{int(time.time() * 1000)}",
            "label": label,
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "kind": "manual",
            "confidence": float(payload.get("confidence", 1.0)),
            "note": str(payload.get("note", "")),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        annotations.append(item)
        _save_annotations(self.annotations_path, annotations)
        self._snapshot = None
        return {"ok": True, "annotation": item}

    def image_goal(self, payload: dict[str, Any]) -> dict[str, Any]:
        snap = self.snapshot(force=True)
        current = snap.get("current_pose")
        if current is None:
            return {"ok": False, "error": "No current camera pose available yet."}
        image = _decode_data_url(str(payload.get("image", "")))
        try:
            match = _match_image_goal(image, [FrameRecord(**x) for x in snap["trajectory"]])
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        target = match["frame"]
        path = _astar_path(snap["_xyz_cache"], current["position"], target["position"])
        return {"ok": True, "match": match, "start": current, "target": target, "path": path}


def make_handler(server_state: SemanticNavServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[semantic-nav] {self.address_string()} {fmt % args}", flush=True)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                _html_response(self, _build_monitor_html(server_state.gs_url))
                return
            if parsed.path == "/api/map":
                _json_response(self, server_state.public_snapshot())
                return
            if parsed.path == "/api/latest_image":
                latest = server_state.latest_image_path()
                if latest is None:
                    self.send_error(404)
                    return
                image_path = latest.expanduser().resolve()
                try:
                    image_path.relative_to(server_state.live_dir)
                except ValueError:
                    self.send_error(404)
                    return
                if not image_path.exists():
                    self.send_error(404)
                    return
                raw = image_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if parsed.path == "/api/frame":
                qs = parse_qs(parsed.query)
                path = Path(qs.get("path", [""])[0]).expanduser().resolve()
                try:
                    path.relative_to(server_state.live_dir)
                except ValueError:
                    self.send_error(403)
                    return
                if not path.exists():
                    self.send_error(404)
                    return
                raw = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            try:
                payload = json.loads(_read_body(self).decode("utf-8") or "{}")
            except Exception:
                _json_response(self, {"ok": False, "error": "Invalid JSON body."}, 400)
                return
            if self.path == "/api/goal":
                _json_response(self, server_state.resolve_goal(str(payload.get("query", ""))))
            elif self.path == "/api/annotations":
                _json_response(self, server_state.add_annotation(payload))
            elif self.path == "/api/image_goal":
                _json_response(self, server_state.image_goal(payload))
            else:
                self.send_error(404)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live semantic navigation + Gaussian viewer baseline.")
    parser.add_argument("--live-dir", default="nuc_output/hikrobot_lingbot_ros2_current_cloud_live")
    parser.add_argument("--worker-dir", default="")
    parser.add_argument("--real2sim-dir", default="nuc_output/real2sim_hikrobot_lingbot_live_baseline")
    parser.add_argument("--annotations", default="nuc_output/semantic_nav_baseline/semantic_nav_annotations.json")
    parser.add_argument("--gs-url", default="http://10.209.93.176:19103/real2sim_gs_console_viewer.html")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19105)
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument("--clean-voxel-size", type=float, default=0.045)
    parser.add_argument("--clean-min-voxel-points", type=int, default=2)
    parser.add_argument("--worker-windows", type=int, default=900)
    parser.add_argument("--cache-sec", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = SemanticNavServer(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Semantic nav WebUI: http://{args.host}:{args.port}/", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
