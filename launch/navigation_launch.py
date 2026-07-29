# Copyright (c) 2018 Intel Corporation
# Copyright (c) 2026 Viam
#
# Licensed under the Apache License, Version 2.0.
#
# Minimal Nav2 bringup for nav-stack.
#
# Stock nav2_bringup/navigation_launch.py always starts collision_monitor,
# docking_server, route_server, and a lifecycle manager that waits on all of
# them. collision_monitor then fails/loops on our config, and killing those
# extras races activation (nodes present, lifecycle get timeouts, zero action
# servers). This launch only starts the servers we actually use and one
# lifecycle manager that owns them.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    log_level = LaunchConfiguration("log_level")

    # Core navigation only — no collision_monitor / docking / route.
    lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
    ]

    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    configured_params = ParameterFile(params_file, allow_substs=True)

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
            DeclareLaunchArgument(
                "params_file",
                description="Full path to the ROS2 parameters file",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation clock if true",
            ),
            DeclareLaunchArgument(
                "autostart",
                default_value="true",
                description="Configure and activate lifecycle nodes on startup",
            ),
            DeclareLaunchArgument(
                "log_level", default_value="info", description="log level"
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
            ),
            Node(
                package="nav2_smoother",
                executable="smoother_server",
                name="smoother_server",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
            ),
            Node(
                package="nav2_velocity_smoother",
                executable="velocity_smoother",
                name="velocity_smoother",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                arguments=["--ros-args", "--log-level", log_level],
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": autostart,
                        # Bond misses on a loaded Pi deactivate the whole stack.
                        "bond_timeout": 0.0,
                        "node_names": lifecycle_nodes,
                    }
                ],
            ),
        ]
    )
