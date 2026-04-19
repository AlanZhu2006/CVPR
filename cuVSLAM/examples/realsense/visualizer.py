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

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import cuvslam as vslam

# Constants
DEFAULT_NUM_VIZ_CAMERAS = 1
POINT_RADIUS = 5.0
ARROW_SCALE = 0.1
GRAVITY_ARROW_SCALE = 0.02


class RerunVisualizer:
    """Rerun-based visualizer for cuVSLAM tracking results."""

    def __init__(self, num_viz_cameras: int = DEFAULT_NUM_VIZ_CAMERAS) -> None:
        """Initialize rerun visualizer.

        Args:
            num_viz_cameras: Number of cameras to visualize
        """
        self.num_viz_cameras = num_viz_cameras
        self._init_rerun()
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        # Set up the visualization layout
        self._setup_blueprint()
        self.track_colors = {}

    def _init_rerun(self) -> None:
        """Initialize rerun in a mode suitable for GUI or headless Jetson setups."""
        rr.init("cuVSLAM Visualizer", spawn=False)

        mode = os.environ.get("CUVSLAM_RERUN_MODE")
        if mode is None:
            mode = "spawn" if os.environ.get("DISPLAY") else "save"
        mode = mode.lower()

        if mode == "spawn":
            rr.spawn()
            print("Rerun mode: spawn")
            return

        if mode == "web":
            web_port = int(os.environ.get("CUVSLAM_RERUN_WEB_PORT", "9877"))
            ws_port = int(os.environ.get("CUVSLAM_RERUN_WS_PORT", "9876"))
            rr.serve_web(open_browser=False, web_port=web_port, ws_port=ws_port)
            print(f"Rerun mode: web (viewer http://127.0.0.1:{web_port})")
            return

        if mode == "save":
            default_file = Path(__file__).resolve().parent / "recordings" / "realsense_tracking.rrd"
            output_file = Path(os.environ.get("CUVSLAM_RERUN_FILE", str(default_file))).expanduser()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            rr.save(output_file)
            print(f"Rerun mode: save ({output_file})")
            return

        raise ValueError(
            "Unsupported CUVSLAM_RERUN_MODE. Use one of: spawn, web, save."
        )

    def _setup_blueprint(self) -> None:
        """Set up the Rerun blueprint for visualization layout."""
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.5, 0.5],
                    contents=[
                        rrb.Vertical(contents=[
                            rrb.Spatial2DView(origin=f'world/camera_{i}')
                            for i in range(self.num_viz_cameras)
                        ]),
                        rrb.Spatial3DView(origin='world')
                    ]
                )
            ),
            make_active=True
        )

    def _log_rig_pose(
        self, rotation_quat: np.ndarray, translation: np.ndarray
    ) -> None:
        """Log rig pose to Rerun.

        Args:
            rotation_quat: Rotation quaternion
            translation: Translation vector
        """
        rr.log(
            "world/camera_0",
            rr.Transform3D(translation=translation, quaternion=rotation_quat),
            rr.Arrows3D(
                vectors=np.eye(3) * ARROW_SCALE,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ
            )
        )

    def _log_observations(
        self,
        observations_main_cam: List[vslam.Observation],
        image: np.ndarray,
        camera_name: str
    ) -> None:
        """Log 2D observations for a specific camera with consistent colors.

        Args:
            observations_main_cam: List of observations
            image: Camera image
            camera_name: Name of the camera for logging
        """
        if not observations_main_cam:
            return

        # Assign random color to new tracks
        for obs in observations_main_cam:
            if obs.id not in self.track_colors:
                self.track_colors[obs.id] = np.random.randint(0, 256, size=3)

        points = np.array([[obs.u, obs.v] for obs in observations_main_cam])
        colors = np.array([
            self.track_colors[obs.id] for obs in observations_main_cam
        ])

        # Handle different image datatypes for compression
        if image.dtype == np.uint8:
            image_log = rr.Image(image).compress()
        else:
            # For other datatypes, don't compress to avoid issues
            image_log = rr.Image(image)

        rr.log(
            f"world/{camera_name}/observations",
            rr.Points2D(positions=points, colors=colors, radii=POINT_RADIUS),
            image_log
        )

    def _log_gravity(self, gravity: np.ndarray) -> None:
        """Log gravity vector to Rerun.

        Args:
            gravity: Gravity vector
        """
        rr.log(
            "world/camera_0/gravity",
            rr.Arrows3D(
                vectors=gravity,
                colors=[[255, 0, 0]],
                radii=GRAVITY_ARROW_SCALE
            )
        )

    def visualize_frame(
        self,
        frame_id: int,
        images: List[np.ndarray],
        pose: vslam.Pose,
        observations_main_cam: List[List[vslam.Observation]],
        trajectory: List[np.ndarray],
        timestamp: int,
        gravity: Optional[np.ndarray] = None
    ) -> None:
        """Visualize current frame state using Rerun.

        Args:
            frame_id: Current frame ID
            images: List of camera images
            pose: Current pose estimate
            observations_main_cam: List of observations for each camera
            trajectory: List of trajectory points
            timestamp: Current timestamp
            gravity: Optional gravity vector
        """
        rr.set_time_sequence("frame", frame_id)
        rr.log("world/trajectory", rr.LineStrips3D(trajectory), static=True)

        self._log_rig_pose(pose.rotation, pose.translation)

        for i in range(self.num_viz_cameras):
            self._log_observations(
                observations_main_cam[i], images[i], f"camera_{i}"
            )

        if gravity is not None:
            self._log_gravity(gravity)

        rr.log("world/timestamp", rr.TextLog(str(timestamp)))
