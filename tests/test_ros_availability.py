"""Tests for ROS availability helpers."""
from __future__ import annotations

import pytest

from src.ros import availability


def test_require_rclpy_raises_when_missing(monkeypatch):
    monkeypatch.setattr(availability, "rclpy_available", lambda: False)
    with pytest.raises(RuntimeError, match="REQUIRE_ROS=1"):
        availability.require_rclpy("slam_backend=slam_toolbox")


def test_require_rclpy_ok_when_present(monkeypatch):
    monkeypatch.setattr(availability, "rclpy_available", lambda: True)
    availability.require_rclpy("nav_backend=nav2")
