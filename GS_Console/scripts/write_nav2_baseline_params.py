#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact Nav2 params file for the GS-SDF PGM baseline.")
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot-radius", type=float, default=0.28)
    return parser.parse_args()


def read_map_resolution(path: Path) -> float:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() == "resolution":
            return float(value.strip())
    return 0.15


def main() -> None:
    args = parse_args()
    map_yaml = str(Path(args.map_yaml).resolve())
    map_resolution = read_map_resolution(Path(args.map_yaml).resolve())
    robot_radius = float(args.robot_radius)
    local_inflation_radius = max(robot_radius + 0.12, robot_radius * 1.35, 0.02)
    global_inflation_radius = max(robot_radius + 0.10, robot_radius * 1.30, 0.02)
    text = f"""map_server:
  ros__parameters:
    use_sim_time: false
    yaml_filename: "{map_yaml}"

planner_server:
  ros__parameters:
    use_sim_time: false
    expected_planner_frequency: 5.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_smac_planner/SmacPlanner2D"
      tolerance: 0.45
      allow_unknown: false
      downsample_costmap: false
      max_iterations: 100000
      max_on_approach_iterations: 1000
      max_planning_time: 2.0
      cost_travel_multiplier: 2.5
      use_final_approach_orientation: false

controller_server:
  ros__parameters:
    use_sim_time: false
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 1.0
    progress_checker_plugin: "progress_checker"
    goal_checker_plugins: ["goal_checker"]
    controller_plugins: ["FollowPath"]
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.02
      movement_time_allowance: 30.0
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.22
      yaw_goal_tolerance: 0.6
      stateful: true
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.60
      lookahead_dist: 0.45
      min_lookahead_dist: 0.20
      max_lookahead_dist: 0.75
      lookahead_time: 1.5
      rotate_to_heading_angular_vel: 0.8
      max_angular_accel: 2.2
      transform_tolerance: 0.35
      use_velocity_scaled_lookahead_dist: false
      min_approach_linear_velocity: 0.08
      approach_velocity_scaling_dist: 0.45
      use_collision_detection: true
      max_allowed_time_to_collision_up_to_carrot: 1.0
      use_rotate_to_heading: true
      allow_reversing: false
      xy_goal_tolerance: 0.22
      stateful: true

local_costmap:
  local_costmap:
    ros__parameters:
      use_sim_time: false
      update_frequency: 8.0
      publish_frequency: 4.0
      global_frame: odom
      robot_base_frame: base_link
      rolling_window: true
      width: 5
      height: 5
      resolution: 0.05
      robot_radius: {robot_radius:.3f}
      plugins: ["obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 8.0
          raytrace_min_range: 0.0
          obstacle_max_range: 7.0
          obstacle_min_range: 0.0
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: {local_inflation_radius:.3f}
      always_send_full_costmap: true

global_costmap:
  global_costmap:
    ros__parameters:
      use_sim_time: false
      update_frequency: 2.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      robot_radius: {robot_radius:.3f}
      resolution: {map_resolution:.6f}
      track_unknown_space: false
      plugins: ["static_layer", "inflation_layer"]
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 2.5
        inflation_radius: {global_inflation_radius:.3f}
      always_send_full_costmap: true

smoother_server:
  ros__parameters:
    use_sim_time: false
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: true

behavior_server:
  ros__parameters:
    use_sim_time: false
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "drive_on_heading", "wait"]
    spin:
      plugin: "nav2_behaviors/Spin"
    backup:
      plugin: "nav2_behaviors/BackUp"
    drive_on_heading:
      plugin: "nav2_behaviors/DriveOnHeading"
    wait:
      plugin: "nav2_behaviors/Wait"
    global_frame: odom
    robot_base_frame: base_link
    transform_tolerance: 0.35
    simulate_ahead_time: 2.0
    max_rotational_vel: 1.0
    min_rotational_vel: 0.25
    rotational_acc_lim: 2.0

bt_navigator:
  ros__parameters:
    use_sim_time: false
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    wait_for_service_timeout: 1000
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_smooth_path_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_drive_on_heading_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_goal_updated_condition_bt_node
      - nav2_globally_updated_goal_condition_bt_node
      - nav2_is_path_valid_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_truncate_path_local_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_path_expiring_timer_condition
      - nav2_single_trigger_bt_node
      - nav2_goal_updated_controller_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_navigate_through_poses_action_bt_node
      - nav2_navigate_to_pose_action_bt_node
      - nav2_remove_passed_goals_action_bt_node
      - nav2_planner_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_checker_selector_bt_node
      - nav2_controller_cancel_bt_node
      - nav2_path_longer_on_approach_bt_node
      - nav2_wait_cancel_bt_node
      - nav2_spin_cancel_bt_node
      - nav2_back_up_cancel_bt_node
      - nav2_assisted_teleop_action_bt_node
      - nav2_assisted_teleop_cancel_bt_node
      - nav2_drive_on_heading_cancel_bt_node

bt_navigator_navigate_to_pose_rclcpp_node:
  ros__parameters:
    use_sim_time: false

waypoint_follower:
  ros__parameters:
    use_sim_time: false
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      waypoint_pause_duration: 0

velocity_smoother:
  ros__parameters:
    use_sim_time: false
    smoothing_frequency: 20.0
    scale_velocities: false
    feedback: "OPEN_LOOP"
    max_velocity: [0.75, 0.0, 1.9]
    min_velocity: [-0.15, 0.0, -1.7]
    max_accel: [1.8, 0.0, 2.8]
    max_decel: [-2.0, 0.0, -3.2]
    odom_topic: "odom"
    odom_duration: 0.1
    deadband_velocity: [0.0, 0.0, 0.0]
    velocity_timeout: 1.0
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(output)


if __name__ == "__main__":
    main()
