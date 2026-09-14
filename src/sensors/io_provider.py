"""Adapter SLAM / nav use to talk to Viam components without SDK coupling."""
from __future__ import annotations


class IOProvider:
    """Adapter the bridge uses to talk to Viam components.

    The navigation/SLAM models supply concrete async callables so sensor IO stays
    free of Viam SDK imports at this layer.
    """

    def __init__(self, read_lidar_points, read_odometry, drive_base, stop_base):
        self.read_lidar_points = read_lidar_points  # async (name) -> lidar points
        self.read_odometry = read_odometry  # async () -> conv.OdomReading
        self.drive_base = drive_base  # async (vx, vy, vtheta) -> None
        self.stop_base = stop_base  # async () -> None
