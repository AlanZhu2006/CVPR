#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from launch import LaunchDescription, LaunchService
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the minimal Nav2 stack used by the GS-SDF baseline.")
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def node(package: str, executable: str, name: str, params_file: str, log_level: str, remappings=None) -> Node:
    return Node(
        package=package,
        executable=executable,
        name=name,
        output="screen",
        parameters=[params_file],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=remappings or [("/tf", "tf"), ("/tf_static", "tf_static")],
    )


def main() -> int:
    args = parse_args()
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    lifecycle_nodes = [
        "map_server",
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
    ]
    description = LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
            node("nav2_map_server", "map_server", "map_server", args.params_file, args.log_level, remappings),
            node(
                "nav2_controller",
                "controller_server",
                "controller_server",
                args.params_file,
                args.log_level,
                remappings + [("cmd_vel", "cmd_vel_nav")],
            ),
            node("nav2_smoother", "smoother_server", "smoother_server", args.params_file, args.log_level, remappings),
            node("nav2_planner", "planner_server", "planner_server", args.params_file, args.log_level, remappings),
            node("nav2_behaviors", "behavior_server", "behavior_server", args.params_file, args.log_level, remappings),
            node("nav2_bt_navigator", "bt_navigator", "bt_navigator", args.params_file, args.log_level, remappings),
            node(
                "nav2_waypoint_follower",
                "waypoint_follower",
                "waypoint_follower",
                args.params_file,
                args.log_level,
                remappings,
            ),
            node(
                "nav2_velocity_smoother",
                "velocity_smoother",
                "velocity_smoother",
                args.params_file,
                args.log_level,
                remappings + [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_nav2_baseline",
                output="screen",
                arguments=["--ros-args", "--log-level", args.log_level],
                parameters=[
                    {"use_sim_time": False},
                    {"autostart": True},
                    {"node_names": lifecycle_nodes},
                ],
            ),
        ]
    )
    service = LaunchService(argv=[])
    service.include_launch_description(description)
    return int(service.run())


if __name__ == "__main__":
    raise SystemExit(main())
