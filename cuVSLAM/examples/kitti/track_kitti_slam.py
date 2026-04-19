# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import argparse
import os
import shutil
import threading
import time
from pathlib import Path

from PIL import Image
from numpy import array_equal as np_array_equal
from numpy import asarray
from numpy import loadtxt
from numpy import savetxt
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
import cuvslam


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE_PATH = Path(__file__).resolve().parent / "dataset/sequences/06"


def quaternion_to_rotation_matrix(quaternion):
    """Convert a quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    return R.from_quat(quaternion).as_matrix().tolist()


def quaternion_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]."""
    return (R.from_quat(q1) * R.from_quat(q2)).as_quat()


def rotate_vector(vector, rotation_matrix):
    """Rotate a 3D vector using a 3x3 rotation matrix."""
    return R.from_matrix(rotation_matrix).apply(vector)


def combine_poses(initial_pose, relative_pose):
    """Combine an initial world pose and a relative pose."""
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)
    rotated_rel_t = rotate_vector(relative_pose.translation, rotation_matrix)
    absolute_translation = [
        initial_pose.translation[0] + rotated_rel_t[0],
        initial_pose.translation[1] + rotated_rel_t[1],
        initial_pose.translation[2] + rotated_rel_t[2],
    ]
    absolute_rotation = quaternion_multiply(
        initial_pose.rotation, relative_pose.rotation
    )
    return cuvslam.Pose(
        translation=absolute_translation,
        rotation=absolute_rotation,
    )


def transform_landmarks(landmarks, initial_pose):
    """Transform landmarks by an initial pose (rotation + translation)."""
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)
    transformed_landmarks = []
    for landmark in landmarks:
        rotated_landmark = rotate_vector(landmark, rotation_matrix)
        transformed_landmarks.append(
            [
                initial_pose.translation[0] + rotated_landmark[0],
                initial_pose.translation[1] + rotated_landmark[1],
                initial_pose.translation[2] + rotated_landmark[2],
            ]
        )
    return transformed_landmarks


def color_from_id(identifier):
    """Generate a stable pseudo-random color from an integer track id."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256,
    ]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run KITTI stereo SLAM mapping/localization with headless-friendly Rerun output."
    )
    parser.add_argument(
        "--sequence-path",
        default=str(DEFAULT_SEQUENCE_PATH),
        help="Path to the KITTI sequence folder (contains image_0, image_1, calib.txt, times.txt).",
    )
    parser.add_argument(
        "--rerun-mode",
        default=os.environ.get("CUVSLAM_RERUN_MODE", ""),
        choices=["", "spawn", "web", "save"],
        help="Rerun output mode. Defaults to spawn with DISPLAY or save on headless systems.",
    )
    parser.add_argument(
        "--rerun-file",
        default=os.environ.get("CUVSLAM_RERUN_FILE", ""),
        help="Output .rrd file path when rerun mode is save.",
    )
    parser.add_argument(
        "--localization-index",
        type=int,
        default=700,
        help="Starting frame index used when localizing into an existing map.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit the number of frames processed in this run (0 means all frames).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.01,
        help="Sleep duration per frame to let async SLAM catch up.",
    )
    parser.add_argument(
        "--force-remap",
        action="store_true",
        help="Delete any existing map and trajectory before running.",
    )
    parser.add_argument(
        "--async-slam",
        action="store_true",
        help="Run SLAM asynchronously. The default is synchronous for offline Jetson stability.",
    )
    return parser.parse_args()


def init_rerun(mode, rerun_file):
    """Initialize rerun in GUI, web, or save mode."""
    rr.init("kitti", strict=True, spawn=False)

    if not mode:
        mode = "spawn" if os.environ.get("DISPLAY") else "save"
    mode = mode.lower()

    default_blueprint = rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Vertical(
            row_shares=[0.6, 0.4],
            contents=[rrb.Spatial3DView(), rrb.Spatial2DView(origin="car/cam0")],
        ),
    )

    default_rrd = REPO_ROOT / "outputs" / "kitti" / "track_kitti_slam.rrd"
    if mode == "spawn":
        rr.spawn()
        print("Rerun mode: spawn")
    elif mode == "web":
        web_port = int(os.environ.get("CUVSLAM_RERUN_WEB_PORT", "9090"))
        ws_port = int(os.environ.get("CUVSLAM_RERUN_WS_PORT", "9877"))
        rr.serve_web(open_browser=False, web_port=web_port, ws_port=ws_port)
        print(f"Rerun mode: web (viewer http://127.0.0.1:{web_port})")
    elif mode == "save":
        output_file = Path(rerun_file).expanduser() if rerun_file else default_rrd
        output_file.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output_file)
        print(f"Rerun mode: save ({output_file})")
    else:
        raise ValueError("Unsupported rerun mode. Use one of: spawn, web, save.")

    rr.send_blueprint(default_blueprint, make_active=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log(
        "xyz",
        rr.Arrows3D(
            vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["[x]", "[y]", "[z]"],
        ),
        static=True,
    )


def load_sequence(sequence_path):
    """Load KITTI calibration and timestamps."""
    intrinsics = loadtxt(sequence_path / "calib.txt", usecols=range(1, 13))[:4].reshape(
        4, 3, 4
    )
    size = Image.open(sequence_path / "image_0" / "000001.png").size
    timestamps = [
        int(10**9 * float(sec_str))
        for sec_str in open(sequence_path / "times.txt").readlines()
    ]
    return intrinsics, size, timestamps


def build_cameras(intrinsics, size):
    """Build a KITTI stereo rig."""
    cameras = [cuvslam.Camera(), cuvslam.Camera()]
    for i in [0, 1]:
        cameras[i].size = size
        cameras[i].principal = [intrinsics[i][0][2], intrinsics[i][1][2]]
        cameras[i].focal = [intrinsics[i].diagonal()[0], intrinsics[i].diagonal()[1]]
    cameras[1].rig_from_camera.translation[0] = (
        -intrinsics[1][0][3] / intrinsics[1][0][0]
    )
    return cameras


def save_callback_factory(state):
    """Create a callback that records whether the map was saved."""
    def save_callback(success):
        state["map_saved"] = success

    return save_callback


def localization_callback_factory(state, event):
    """Create a callback that stores localization results."""
    def localization_callback(pose, error_message):
        print(f"Localization result: {pose}, {error_message}")
        state["slam_initial_pose"] = pose
        event.set()

    return localization_callback


def main():
    """Entry point."""
    args = parse_args()
    sequence_path = Path(args.sequence_path).expanduser().resolve()
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence path not found: {sequence_path}")

    map_path = sequence_path / "map"
    trajectory_file = sequence_path / "trajectory_tum.txt"

    if args.force_remap:
        if map_path.exists():
            shutil.rmtree(map_path)
        if trajectory_file.exists():
            trajectory_file.unlink()
        print(f"Removed previous map outputs under {sequence_path}")

    init_rerun(args.rerun_mode, args.rerun_file)

    slam_sync_mode = not args.async_slam
    localization_index = args.localization_index
    max_wait_time = 10.0

    intrinsics, size, timestamps = load_sequence(sequence_path)
    cameras = build_cameras(intrinsics, size)

    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        rectified_stereo_camera=True,
    )
    s_cfg = cuvslam.Tracker.SlamConfig(sync_mode=slam_sync_mode)
    tracker = cuvslam.Tracker(cuvslam.Rig(cameras), cfg, s_cfg)
    print(f"SLAM sync mode: {slam_sync_mode}")

    localization_complete = threading.Event()
    state = {
        "slam_initial_pose": None,
        "map_saved": False,
    }

    loc_settings = cuvslam.Tracker.SlamLocalizationSettings(
        horizontal_search_radius=8.0,
        vertical_search_radius=2.0,
        horizontal_step=0.5,
        vertical_step=0.2,
        angular_step_rads=0.03,
    )

    guess_pose = None
    if trajectory_file.exists() and map_path.exists():
        trajectory_data = loadtxt(trajectory_file)
        if localization_index >= len(trajectory_data):
            raise IndexError(
                f"localization index {localization_index} exceeds trajectory length {len(trajectory_data)}"
            )
        guess_tum_pose = trajectory_data[localization_index]
        guess_pose = cuvslam.Pose(
            translation=guess_tum_pose[:3],
            rotation=guess_tum_pose[3:],
        )

    if map_path.exists() and guess_pose is not None:
        init_images = [
            asarray(Image.open(sequence_path / f"image_{cam}" / f"{localization_index:0>6}.png"))
            for cam in [0, 1]
        ]
        tracker.track(timestamps[localization_index], init_images)
        tracker.localize_in_map(
            str(map_path),
            guess_pose,
            init_images,
            loc_settings,
            localization_callback_factory(state, localization_complete),
        )

        wait_time = 0.0
        wait_timestamp_ns = timestamps[localization_index]
        while (
            not slam_sync_mode
            and not localization_complete.wait(timeout=0.5)
            and wait_time < max_wait_time
        ):
            print(f"Waiting for localization... t={wait_time:.1f}s")
            if wait_time < 5.0:
                wait_timestamp_ns += 1_000_000
                _, slam_pose = tracker.track(wait_timestamp_ns, init_images)
                print(f"  slam_pose.t: {[f'{x:.3f}' for x in slam_pose.translation]}")
            wait_time += 0.5

        if not localization_complete.is_set():
            print(f"Localization did not complete within {max_wait_time} seconds")
        localization_index += 1

    if state["slam_initial_pose"] is not None:
        print(f"Localized pose: {state['slam_initial_pose']}")
        wait_time = 0.0
        wait_timestamp_ns = timestamps[min(localization_index, len(timestamps) - 1)]
        init_images = [
            asarray(Image.open(sequence_path / f"image_{cam}" / f"{args.localization_index:0>6}.png"))
            for cam in [0, 1]
        ]
        while not slam_sync_mode and wait_time < max_wait_time:
            time.sleep(0.5)
            wait_timestamp_ns += 1_000_000
            _, slam_pose = tracker.track(wait_timestamp_ns, init_images)
            print(f"  slam_pose.t: {[f'{x:.3f}' for x in slam_pose.translation]}")
            identity_t = all(abs(x) < 1e-6 for x in slam_pose.translation)
            identity_r = all(abs(x) < 1e-6 for x in slam_pose.rotation[:3]) and abs(
                slam_pose.rotation[3] - 1.0
            ) < 1e-6
            if not (identity_t and identity_r):
                break
            wait_time += 0.5
    else:
        print("No prior map detected; starting a fresh mapping pass from frame 0")
        localization_index = 0
        state["slam_initial_pose"] = cuvslam.Pose(
            translation=[0, 0, 0],
            rotation=[0, 0, 0, 1],
        )

    trajectory = []
    trajectory_slam = []
    trajectory_tum = []
    loop_closure_poses = []

    end_frame = len(timestamps)
    if args.max_frames > 0:
        end_frame = min(end_frame, localization_index + args.max_frames)

    print(
        f"Processing frames [{localization_index}, {end_frame}) from {sequence_path}"
    )
    for frame in range(localization_index, end_frame):
        time.sleep(args.sleep_seconds)

        images = [
            asarray(Image.open(sequence_path / f"image_{cam}" / f"{frame:0>6}.png"))
            for cam in [0, 1]
        ]

        odometry_pose_estimate, slam_pose = tracker.track(timestamps[frame], images)
        if odometry_pose_estimate.world_from_rig is None:
            print(f"Warning: Failed to track frame {frame}")
            continue

        odom_pose = odometry_pose_estimate.world_from_rig.pose
        current_pose = combine_poses(state["slam_initial_pose"], odom_pose)

        observations = tracker.get_last_observations(0)
        landmarks = tracker.get_last_landmarks()
        raw_final_landmarks = list(tracker.get_final_landmarks().values())
        final_landmarks = transform_landmarks(
            raw_final_landmarks, state["slam_initial_pose"]
        )

        observations_uv = [[o.u, o.v] for o in observations]
        observations_colors = [color_from_id(o.id) for o in observations]
        landmark_xyz = [l.coords for l in landmarks]
        landmarks_colors = [color_from_id(l.id) for l in landmarks]

        trajectory.append(current_pose.translation)
        trajectory_slam.append(slam_pose.translation)
        trajectory_tum.append(list(slam_pose.translation) + list(slam_pose.rotation))

        current_lc_poses = tracker.get_loop_closure_poses()
        if current_lc_poses and (
            not loop_closure_poses
            or not np_array_equal(
                current_lc_poses[-1].pose.translation, loop_closure_poses[-1]
            )
        ):
            loop_closure_poses.append(current_lc_poses[-1].pose.translation)

        rr.set_time_nanos("timestamp", timestamps[frame])
        rr.log("trajectory", rr.LineStrips3D(trajectory))
        rr.log("trajectory_slam", rr.LineStrips3D(trajectory_slam))
        rr.log("final_landmarks", rr.Points3D(final_landmarks, radii=0.1))
        rr.log(
            "loop_closure_poses",
            rr.Points3D(loop_closure_poses, radii=1.2, colors=[[255, 0, 0]]),
        )
        rr.log(
            "car",
            rr.Transform3D(
                translation=current_pose.translation,
                quaternion=current_pose.rotation,
            ),
        )
        rr.log(
            "car/body",
            rr.Boxes3D(centers=[0, 1.65 / 2, 0], sizes=[[1.6, 1.65, 2.71]]),
        )
        rr.log(
            "car/landmarks_center",
            rr.Points3D(landmark_xyz, radii=0.25, colors=landmarks_colors),
        )
        rr.log(
            "car/landmarks_lines",
            rr.Arrows3D(vectors=landmark_xyz, radii=0.05, colors=landmarks_colors),
        )
        rr.log(
            "car/cam0",
            rr.Pinhole(
                image_plane_distance=1.68,
                image_from_camera=intrinsics[0][:3, :3],
                width=size[0],
                height=size[1],
            ),
        )
        rr.log("car/cam0/image", rr.Image(images[0]).compress(jpeg_quality=80))
        rr.log(
            "car/cam0/observations",
            rr.Points2D(observations_uv, radii=5, colors=observations_colors),
        )

    print(f"Number of loop closure poses: {len(loop_closure_poses)}")

    if guess_pose is None:
        map_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving trajectory to {trajectory_file} with length {len(trajectory_tum)}")
        savetxt(trajectory_file, trajectory_tum)

        tracker.save_map(str(map_path), save_callback_factory(state))

        start_time = time.time()
        while not state["map_saved"] and (time.time() - start_time) < max_wait_time:
            time.sleep(0.1)
            print(
                f"Waiting for map saving to complete... {time.time() - start_time:.1f}s"
            )

        if state["map_saved"]:
            print("Map saved successfully")
        else:
            print("WARNING: Map saving may not have completed")

    print("Script completed")


if __name__ == "__main__":
    main()
