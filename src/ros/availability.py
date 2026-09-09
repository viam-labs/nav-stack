"""Detect whether ROS 2 Python bindings are importable."""
from __future__ import annotations


def rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return False
    return True


def require_rclpy(feature: str) -> None:
    """Raise a clear error when a ROS-backed backend is configured without ROS."""
    if rclpy_available():
        return
    raise RuntimeError(
        f"{feature} requires ROS 2 (rclpy). "
        "Builtin slam/nav need no ROS. For slam_toolbox or Nav2: set "
        "REQUIRE_ROS=1 in the module env, re-run setup on Ubuntu "
        "22.04/24.04/26.04, then restart the module."
    )
