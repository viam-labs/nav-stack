"""ROS-free adapter the bridge (and builtin scan readers) use to talk to Viam.

Kept out of ``bridge.py`` so ``build_io_provider`` / ``global_localize`` can run
when ``rclpy`` is not installed (``slam_backend: builtin``).
"""
from __future__ import annotations


class IOProvider:
    """Adapter the bridge uses to talk to Viam components.

    The navigation/SLAM models supply concrete async callables; the bridge stays
    free of any Viam SDK imports.
    """

    def __init__(self, read_lidar_points, read_odometry, drive_base, stop_base):
        self.read_lidar_points = read_lidar_points  # async (name) -> lidar points
        self.read_odometry = read_odometry  # async () -> conv.OdomReading
        self.drive_base = drive_base  # async (vx, vy, vtheta) -> None
        self.stop_base = stop_base  # async () -> None
