#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from hikrobot_mvs_ros2_publisher import HikRobotCamera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate HikRobot RGB intrinsics from a checkerboard or ChArUco board. "
            "Use checkerboard-cols/rows as inner-corner counts, not square counts."
        )
    )
    parser.add_argument("--target", choices=("checkerboard", "charuco"), default="checkerboard")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--exposure-us", type=float, default=8000.0)
    parser.add_argument("--gain", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--no-camera-roi",
        action="store_true",
        help="Do not set camera Width/Height ROI; capture the full sensor frame.",
    )
    parser.add_argument(
        "--detect-scale",
        type=float,
        default=1.0,
        help="Run board detection on a resized image, then scale corners back. Use e.g. 0.5 for speed.",
    )
    parser.add_argument(
        "--preview-max-width",
        type=int,
        default=1280,
        help="Limit preview window width without changing calibration coordinates.",
    )
    parser.add_argument("--checkerboard-cols", type=int, default=9, help="Inner corners along checkerboard width.")
    parser.add_argument("--checkerboard-rows", type=int, default=6, help="Inner corners along checkerboard height.")
    parser.add_argument("--charuco-cols", type=int, default=11, help="ChArUco square count along board width.")
    parser.add_argument("--charuco-rows", type=int, default=8, help="ChArUco square count along board height.")
    parser.add_argument("--marker-size-m", type=float, default=0.011)
    parser.add_argument("--aruco-dict", default="DICT_4X4_50")
    parser.add_argument("--min-charuco-corners", type=int, default=12)
    parser.add_argument("--square-size-m", type=float, default=0.025)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--min-interval-sec", type=float, default=0.5)
    parser.add_argument(
        "--capture-mode",
        choices=("manual", "auto"),
        default="manual",
        help="manual accepts a calibration view only when Space/s is pressed; auto accepts at --min-interval-sec.",
    )
    parser.add_argument("--output-dir", default="nuc_output/hikrobot_calibration")
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Also save frames where the checkerboard was not detected.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show an OpenCV preview window. Manual mode enables this automatically.",
    )
    return parser.parse_args()


def _make_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    points = np.zeros((rows * cols, 3), np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    points[:, 0] = grid_x.reshape(-1) * float(square_size_m)
    points[:, 1] = grid_y.reshape(-1) * float(square_size_m)
    return points


def _find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok:
            return True, corners.astype(np.float32)
    ok, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined.astype(np.float32)


def _make_charuco_board(args: argparse.Namespace):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build does not include cv2.aruco; install opencv-contrib-python.")
    aruco = cv2.aruco
    dict_name = str(args.aruco_dict)
    if dict_name == "DICT_4X4":
        dict_name = "DICT_4X4_50"
    if not hasattr(aruco, dict_name):
        valid = sorted(name for name in dir(aruco) if name.startswith("DICT_"))
        raise RuntimeError(f"Unknown ArUco dictionary {dict_name}; examples: {', '.join(valid[:8])}")
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    return aruco.CharucoBoard(
        (int(args.charuco_cols), int(args.charuco_rows)),
        float(args.square_size_m),
        float(args.marker_size_m),
        dictionary,
    )


def _make_charuco_detector(board):
    aruco = cv2.aruco
    if hasattr(aruco, "CharucoDetector"):
        return aruco.CharucoDetector(board)
    return None


def _find_charuco_points(
    gray: np.ndarray,
    board,
    detector,
    min_corners: int,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, tuple | None, np.ndarray | None]:
    aruco = cv2.aruco
    if detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    else:
        parameters = aruco.DetectorParameters()
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, board.getDictionary(), parameters=parameters)
        if marker_ids is None or len(marker_ids) == 0:
            return False, None, None, marker_corners, marker_ids
        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < int(min_corners):
        return False, None, None, marker_corners, marker_ids
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if object_points is None or image_points is None or len(object_points) < int(min_corners):
        return False, None, None, marker_corners, marker_ids
    return True, object_points.astype(np.float32), image_points.astype(np.float32), marker_corners, marker_ids


def _mean_reprojection_error(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0
    for obj, img, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        error = cv2.norm(img, projected, cv2.NORM_L2)
        total_error += float(error * error)
        total_points += int(len(obj))
    if total_points <= 0:
        return float("nan")
    return float(np.sqrt(total_error / total_points))


def _draw_charuco_preview(
    preview: np.ndarray,
    object_points: np.ndarray | None,
    image_points: np.ndarray | None,
    marker_corners,
    marker_ids: np.ndarray | None,
) -> None:
    aruco = cv2.aruco
    if marker_ids is not None and marker_corners is not None and len(marker_ids) > 0:
        aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)
    if image_points is None or object_points is None or len(image_points) == 0:
        return
    corners = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
    ids = np.arange(len(corners), dtype=np.int32).reshape(-1, 1)
    if hasattr(aruco, "drawDetectedCornersCharuco"):
        aruco.drawDetectedCornersCharuco(preview, corners, ids, (0, 255, 0))


def _scale_image_points(points: np.ndarray | None, scale: float) -> np.ndarray | None:
    if points is None or abs(float(scale) - 1.0) < 1e-6:
        return points
    return np.asarray(points, dtype=np.float32) / float(scale)


def _resize_for_preview(rgb: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    if int(max_width) <= 0 or rgb.shape[1] <= int(max_width):
        return rgb, 1.0
    scale = float(max_width) / float(rgb.shape[1])
    resized = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def _write_yaml(path: Path, width: int, height: int, K: np.ndarray, dist: np.ndarray) -> None:
    d = [float(v) for v in dist.reshape(-1)[:5]]
    text = f"""image_width: {width}
image_height: {height}
camera_name: hikrobot_camera
camera_matrix:
  rows: 3
  cols: 3
  data: [{K[0,0]:.12g}, 0.0, {K[0,2]:.12g}, 0.0, {K[1,1]:.12g}, {K[1,2]:.12g}, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{d[0]:.12g}, {d[1]:.12g}, {d[2]:.12g}, {d[3]:.12g}, {d[4]:.12g}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [{K[0,0]:.12g}, 0.0, {K[0,2]:.12g}, 0.0, 0.0, {K[1,1]:.12g}, {K[1,2]:.12g}, 0.0, 0.0, 0.0, 1.0, 0.0]
"""
    path.write_text(text, encoding="utf-8")


def _write_env(path: Path, K: np.ndarray, dist: np.ndarray) -> None:
    coeffs = " ".join(f"{float(v):.12g}" for v in dist.reshape(-1)[:5])
    text = f"""export HIKROBOT_CAMERA_FX={float(K[0, 0]):.12g}
export HIKROBOT_CAMERA_FY={float(K[1, 1]):.12g}
export HIKROBOT_CAMERA_CX={float(K[0, 2]):.12g}
export HIKROBOT_CAMERA_CY={float(K[1, 2]):.12g}
export HIKROBOT_DISTORTION_MODEL=plumb_bob
export HIKROBOT_DISTORTION_COEFFS="{coeffs}"
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    accepted_dir = output_dir / "accepted"
    preview_dir = output_dir / "corners"
    rejected_dir = output_dir / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    if args.save_all:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    pattern_size = (int(args.checkerboard_cols), int(args.checkerboard_rows))
    object_template = _make_object_points(pattern_size[0], pattern_size[1], args.square_size_m)
    charuco_board = _make_charuco_board(args) if args.target == "charuco" else None
    charuco_detector = _make_charuco_detector(charuco_board) if charuco_board is not None else None
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    print(
        "Opening HikRobot for calibration: "
        f"{args.width}x{args.height} fps={args.fps} exposure={args.exposure_us} gain={args.gain}",
        flush=True,
    )
    camera = HikRobotCamera(args.index, args.timeout_ms)
    capture_width = None if args.no_camera_roi else (args.width or None)
    capture_height = None if args.no_camera_roi else (args.height or None)
    if args.no_camera_roi:
        print("Camera ROI disabled: capturing full sensor frame and resizing only for detection/preview.", flush=True)
    camera.open(
        args.exposure_us,
        args.gain,
        args.fps,
        capture_width,
        capture_height,
    )
    last_accept_sec = 0.0
    frame_idx = 0
    show_window = bool(args.show or args.capture_mode == "manual")
    if args.capture_mode == "manual":
        print(
            "Manual capture mode: show the board, wait for green detected corners, "
            "then press Space or s to save that view. Press q/Esc to finish early.",
            flush=True,
        )
    try:
        while len(objpoints) < int(args.samples):
            rgb_bytes, width, height = camera.read_rgb()
            rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
            image_size = (int(width), int(height))
            detect_scale = float(args.detect_scale)
            if detect_scale <= 0.0:
                detect_scale = 1.0
            if abs(detect_scale - 1.0) > 1e-6:
                detect_rgb = cv2.resize(rgb, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
            else:
                detect_rgb = rgb
            gray = cv2.cvtColor(detect_rgb, cv2.COLOR_RGB2GRAY)
            marker_corners = None
            marker_ids = None
            if args.target == "charuco":
                ok, object_points, image_points, marker_corners, marker_ids = _find_charuco_points(
                    gray,
                    charuco_board,
                    charuco_detector,
                    args.min_charuco_corners,
                )
                image_points = _scale_image_points(image_points, detect_scale)
                if marker_corners is not None and abs(detect_scale - 1.0) > 1e-6:
                    marker_corners = tuple(np.asarray(c, dtype=np.float32) / detect_scale for c in marker_corners)
                corners = image_points
            else:
                ok, corners = _find_corners(gray, pattern_size)
                corners = _scale_image_points(corners, detect_scale)
                object_points = object_template.copy() if ok and corners is not None else None
                image_points = corners
            now = time.perf_counter()
            preview = rgb.copy()
            if args.target == "charuco":
                _draw_charuco_preview(preview, object_points, image_points, marker_corners, marker_ids)
            elif ok and corners is not None:
                cv2.drawChessboardCorners(preview, pattern_size, corners, ok)
            key = -1
            if show_window:
                preview_display, preview_scale = _resize_for_preview(preview, args.preview_max_width)
                bgr_preview = cv2.cvtColor(preview_display, cv2.COLOR_RGB2BGR)
                point_count = 0
                if image_points is not None:
                    point_count = int(np.asarray(image_points).reshape(-1, 2).shape[0])
                font_scale = max(0.5, 0.72 * preview_scale)
                thickness = max(1, int(round(2 * preview_scale)))
                cv2.putText(
                    bgr_preview,
                    f"{args.capture_mode} accepted {len(objpoints)}/{args.samples} detected {point_count} frame {width}x{height}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0) if ok else (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr_preview,
                    "Space/s save | q/Esc quit",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                cv2.imshow("hikrobot calibration", bgr_preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            valid_detection = bool(ok and object_points is not None and image_points is not None)
            if args.capture_mode == "manual":
                save_requested = key in (ord(" "), ord("s"), ord("S"))
                accept = bool(valid_detection and save_requested)
                if save_requested and not valid_detection:
                    print("save requested, but no valid calibration board is detected in this frame", flush=True)
            else:
                accept = bool(valid_detection and now - last_accept_sec >= float(args.min_interval_sec))
            if accept:
                sample_idx = len(objpoints)
                objpoints.append(object_points.copy())
                imgpoints.append(image_points.copy())
                cv2.imwrite(str(accepted_dir / f"accepted_{sample_idx:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(preview_dir / f"corners_{sample_idx:04d}.png"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                last_accept_sec = now
                point_count = int(np.asarray(image_points).reshape(-1, 2).shape[0])
                print(
                    f"accepted {sample_idx + 1}/{args.samples}: points={point_count}; "
                    "move/tilt the board for a different view",
                    flush=True,
                )
            elif args.save_all and frame_idx % max(1, int(args.fps or 1)) == 0:
                cv2.imwrite(str(rejected_dir / f"frame_{frame_idx:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frame_idx += 1
    finally:
        camera.close()
        if show_window:
            cv2.destroyAllWindows()

    if len(objpoints) < 5 or image_size is None:
        print(f"Need at least 5 accepted calibration views; got {len(objpoints)}.", flush=True)
        return 2

    flags = 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=flags)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    if dist.size < 5:
        dist = np.pad(dist, (0, 5 - dist.size))
    mean_error = _mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    result = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "target": args.target,
        "checkerboard_inner_corners": {"cols": pattern_size[0], "rows": pattern_size[1]},
        "charuco_squares": {"cols": int(args.charuco_cols), "rows": int(args.charuco_rows)},
        "square_size_m": float(args.square_size_m),
        "marker_size_m": float(args.marker_size_m),
        "aruco_dict": str(args.aruco_dict),
        "min_charuco_corners": int(args.min_charuco_corners),
        "samples": len(objpoints),
        "rms_reprojection_error_px": float(rms),
        "mean_reprojection_error_px": float(mean_error),
        "camera_matrix": K.tolist(),
        "distortion_model": "plumb_bob",
        "distortion_coefficients": [float(v) for v in dist[:5]],
        "accepted_dir": str(accepted_dir),
        "corners_dir": str(preview_dir),
    }

    json_path = output_dir / "hikrobot_camera_calibration.json"
    yaml_path = output_dir / "hikrobot_camera_calibration.yaml"
    env_path = output_dir / "hikrobot_calibration.env"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_yaml(yaml_path, image_size[0], image_size[1], K, dist)
    _write_env(env_path, K, dist)

    print(
        "Calibration complete:\n"
        f"  fx={K[0,0]:.3f} fy={K[1,1]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}\n"
        f"  distortion={' '.join(f'{v:.6g}' for v in dist[:5])}\n"
        f"  rms={float(rms):.4f}px mean={mean_error:.4f}px\n"
        f"  json={json_path}\n"
        f"  yaml={yaml_path}\n"
        f"  env={env_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
