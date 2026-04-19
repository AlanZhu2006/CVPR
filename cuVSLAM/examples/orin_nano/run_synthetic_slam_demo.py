import argparse
import os
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import cuvslam as vslam

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_TEST_DIR = REPO_ROOT / "python" / "test"
if str(PYTHON_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_TEST_DIR))

from data_gen import ImageGenerator, generate_stereo_camera


def color_from_id(identifier: int) -> list[int]:
    """Generate a stable pseudo-random color for a track identifier."""
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


def init_rerun(output_dir: Path, mode: str) -> Path:
    """Initialize rerun for GUI, web, or headless file logging."""
    rr.init("cuvslam_synthetic_slam", spawn=False)

    default_blueprint = rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Vertical(
            row_shares=[0.65, 0.35],
            contents=[
                rrb.Spatial3DView(origin="/"),
                rrb.Spatial2DView(origin="rig/cam0"),
            ],
        ),
    )

    if not mode:
        mode = "spawn" if os.environ.get("DISPLAY") else "save"
    mode = mode.lower()

    output_path = output_dir / "synthetic_slam.rrd"
    if mode == "spawn":
        rr.spawn()
    elif mode == "web":
        web_port = int(os.environ.get("CUVSLAM_RERUN_WEB_PORT", "9887"))
        ws_port = int(os.environ.get("CUVSLAM_RERUN_WS_PORT", "9886"))
        rr.serve_web(open_browser=False, web_port=web_port, ws_port=ws_port)
        print(f"Rerun web viewer available at http://127.0.0.1:{web_port}")
    elif mode == "save":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output_path)
        print(f"Saving Rerun recording to {output_path}")
    else:
        raise ValueError("Unsupported rerun mode. Use one of: spawn, web, save.")

    rr.send_blueprint(default_blueprint, make_active=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log(
        "xyz",
        rr.Arrows3D(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"],
        ),
        static=True,
    )
    return output_path


def build_image_from_camera(camera: vslam.Camera) -> np.ndarray:
    """Construct the camera matrix expected by rerun's pinhole helper."""
    return np.array(
        [
            [float(camera.focal[0]), 0.0, float(camera.principal[0])],
            [0.0, float(camera.focal[1]), float(camera.principal[1])],
            [0.0, 0.0, 1.0],
        ]
    )


def build_tracker(rig: vslam.Rig) -> vslam.Tracker:
    """Create a deterministic tracker suitable for map save/load validation."""
    odom_cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_observations_export=True,
        enable_landmarks_export=True,
        enable_final_landmarks_export=True,
        rectified_stereo_camera=True,
    )
    slam_cfg = vslam.Tracker.SlamConfig(sync_mode=True)
    return vslam.Tracker(rig, odom_cfg, slam_cfg)


def log_frame(
    frame_index: int,
    stage: str,
    image_from_camera: np.ndarray,
    image: np.ndarray,
    odom_pose: vslam.Pose,
    slam_pose: vslam.Pose,
    observations: list[vslam.Observation],
    landmarks: list[vslam.Landmark],
    trajectory_odom: list[list[float]],
    trajectory_slam: list[list[float]],
    final_landmarks: list[list[float]],
) -> None:
    """Send the current frame state to rerun."""
    rr.set_time_sequence("frame", frame_index)
    rr.log("status/stage", rr.TextLog(stage))
    rr.log("mapping/trajectory/odometry", rr.LineStrips3D(trajectory_odom))
    rr.log("mapping/trajectory/slam", rr.LineStrips3D(trajectory_slam))

    if final_landmarks:
        rr.log(
            "map/final_landmarks",
            rr.Points3D(final_landmarks, radii=0.03, colors=[[160, 160, 160]]),
        )

    rr.log(
        "rig",
        rr.Transform3D(
            translation=slam_pose.translation,
            quaternion=slam_pose.rotation,
        ),
    )
    rr.log(
        "rig/odom_delta",
        rr.Transform3D(
            translation=odom_pose.translation,
            quaternion=odom_pose.rotation,
        ),
    )
    rr.log(
        "rig/current_landmarks",
        rr.Points3D(
            [landmark.coords for landmark in landmarks],
            radii=0.04,
            colors=[color_from_id(landmark.id) for landmark in landmarks],
        ),
    )
    rr.log(
        "rig/landmark_rays",
        rr.Arrows3D(
            vectors=[landmark.coords for landmark in landmarks],
            radii=0.01,
            colors=[color_from_id(landmark.id) for landmark in landmarks],
        ),
    )
    rr.log(
        "rig/cam0",
        rr.Pinhole(
            image_plane_distance=1.0,
            image_from_camera=image_from_camera,
            width=image.shape[1],
            height=image.shape[0],
        ),
    )
    rr.log("rig/cam0/image", rr.Image(image).compress(jpeg_quality=90))
    rr.log(
        "rig/cam0/observations",
        rr.Points2D(
            [[obs.u, obs.v] for obs in observations],
            radii=4.0,
            colors=[color_from_id(obs.id) for obs in observations],
        ),
    )


def save_map(tracker: vslam.Tracker, map_dir: Path) -> None:
    """Persist a SLAM map to disk and wait for completion."""
    if map_dir.exists():
        shutil.rmtree(map_dir)

    completed = threading.Event()
    result = {"ok": False}

    def callback(success: bool) -> None:
        result["ok"] = success
        completed.set()

    tracker.save_map(str(map_dir), callback)
    if not completed.wait(timeout=5.0):
        raise TimeoutError(f"Timed out while saving map to {map_dir}")
    if not result["ok"]:
        raise RuntimeError(f"cuVSLAM reported map save failure for {map_dir}")


def localize_in_map(
    tracker: vslam.Tracker,
    map_dir: Path,
    images: list[np.ndarray],
    guess_pose: vslam.Pose,
    timestamp_ns: int,
) -> vslam.Pose:
    """Localize in an existing map and return the resulting pose."""
    completed = threading.Event()
    result = {"pose": None, "error": ""}

    settings = vslam.Tracker.SlamLocalizationSettings(
        horizontal_search_radius=0.25,
        vertical_search_radius=0.25,
        horizontal_step=0.0625,
        vertical_step=0.0625,
        angular_step_rads=0.03125,
    )

    def callback(pose: vslam.Pose | None, error_message: str) -> None:
        result["pose"] = pose
        result["error"] = error_message
        completed.set()

    tracker.localize_in_map(str(map_dir), guess_pose, images, settings, callback)

    while not completed.wait(timeout=0.1):
        timestamp_ns += 1_000
        tracker.track(timestamp_ns, images)

    if result["pose"] is None:
        raise RuntimeError(f"Localization failed: {result['error']}")
    return result["pose"]


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(
        description="Run a full synthetic cuVSLAM mapping/save/load/localization demo on Jetson."
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "orin_nano_demo"),
        help="Directory where the map and optional rerun recording will be written.",
    )
    parser.add_argument(
        "--rerun-mode",
        default=os.environ.get("CUVSLAM_RERUN_MODE", ""),
        choices=["", "spawn", "web", "save"],
        help="Rerun output mode. Defaults to spawn with DISPLAY or save on headless systems.",
    )
    parser.add_argument(
        "--mapping-steps",
        type=int,
        default=30,
        help="Number of synthetic frames to use for the mapping pass.",
    )
    parser.add_argument(
        "--localization-start",
        type=int,
        default=10,
        help="Synthetic frame index from which to start the localization pass.",
    )
    parser.add_argument(
        "--localization-steps",
        type=int,
        default=10,
        help="Number of frames to continue tracking after localization.",
    )
    parser.add_argument(
        "--guess-offset-z",
        type=float,
        default=0.05,
        help="Offset added to the initial localization pose guess in meters.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete synthetic SLAM workflow."""
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = output_dir / "synthetic_map"

    recording_path = init_rerun(output_dir, args.rerun_mode)

    cameras = generate_stereo_camera(640, 480, baseline=0.25)
    rig = vslam.Rig(cameras)
    image_from_camera = build_image_from_camera(cameras[0])
    image_generator = ImageGenerator(cameras, steps=max(args.mapping_steps, args.localization_start + args.localization_steps))

    tracker = build_tracker(rig)
    mapping_trajectory_odom: list[list[float]] = []
    mapping_trajectory_slam: list[list[float]] = []

    print("Running synthetic mapping pass...")
    base_timestamp_ns = 1_000_000_000
    for step in range(args.mapping_steps):
        images, _ = image_generator.generate_zoomed_images(step)
        odom_pose_estimate, slam_pose = tracker.track(base_timestamp_ns + step * 1_000_000, images)
        if odom_pose_estimate.world_from_rig is None or slam_pose is None:
            raise RuntimeError(f"Tracking failed during mapping at frame {step}")

        odom_pose = odom_pose_estimate.world_from_rig.pose
        mapping_trajectory_odom.append(list(odom_pose.translation))
        mapping_trajectory_slam.append(list(slam_pose.translation))

        log_frame(
            frame_index=step,
            stage="mapping",
            image_from_camera=image_from_camera,
            image=images[0],
            odom_pose=odom_pose,
            slam_pose=slam_pose,
            observations=tracker.get_last_observations(0),
            landmarks=tracker.get_last_landmarks(),
            trajectory_odom=mapping_trajectory_odom,
            trajectory_slam=mapping_trajectory_slam,
            final_landmarks=list(tracker.get_final_landmarks().values()),
        )

    save_map(tracker, map_dir)
    print(f"Saved synthetic map to {map_dir}")

    print("Running localization pass against the saved map...")
    tracker = build_tracker(rig)
    localization_step = args.localization_start
    localization_timestamp_ns = 2_000_000_000
    images, z_position = image_generator.generate_zoomed_images(localization_step)
    tracker.track(localization_timestamp_ns, images)

    guess_pose = vslam.Pose(
        translation=[0.0, 0.0, z_position + args.guess_offset_z],
        rotation=[0.0, 0.0, 0.0, 1.0],
    )
    localized_pose = localize_in_map(tracker, map_dir, images, guess_pose, localization_timestamp_ns)
    print(f"Localized pose: translation={localized_pose.translation} rotation={localized_pose.rotation}")

    localization_trajectory_slam: list[list[float]] = [list(localized_pose.translation)]
    for offset in range(args.localization_steps):
        step = localization_step + offset
        images, _ = image_generator.generate_zoomed_images(step)
        localization_timestamp_ns += 1_000_000
        odom_pose_estimate, slam_pose = tracker.track(localization_timestamp_ns, images)
        if odom_pose_estimate.world_from_rig is None or slam_pose is None:
            raise RuntimeError(f"Tracking failed during localization at frame {step}")

        localization_trajectory_slam.append(list(slam_pose.translation))
        rr.log("localization/trajectory/slam", rr.LineStrips3D(localization_trajectory_slam))
        rr.log(
            "localization/guess_pose",
            rr.Points3D([guess_pose.translation], radii=0.06, colors=[[255, 180, 0]]),
        )
        log_frame(
            frame_index=args.mapping_steps + offset,
            stage="localization",
            image_from_camera=image_from_camera,
            image=images[0],
            odom_pose=odom_pose_estimate.world_from_rig.pose,
            slam_pose=slam_pose,
            observations=tracker.get_last_observations(0),
            landmarks=tracker.get_last_landmarks(),
            trajectory_odom=mapping_trajectory_odom,
            trajectory_slam=mapping_trajectory_slam,
            final_landmarks=list(tracker.get_final_landmarks().values()),
        )

    print("Synthetic SLAM demo completed successfully.")
    print(f"Map output: {map_dir}")
    if recording_path.exists():
        print(f"Rerun recording: {recording_path}")


if __name__ == "__main__":
    main()
