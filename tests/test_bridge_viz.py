"""Tests for BridgeNode nav-camera viz state (plan trail + snapshot).

Uses the same ROS-stub approach as test_bridge_nav: stub the ROS modules so the
bridge imports without a ROS install, then drive the pure viz methods with a
SimpleNamespace stand-in (no BridgeNode __init__, which needs a live node).
"""
import sys
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock


class _FakeNode:
    pass


sys.modules.setdefault("rclpy", MagicMock())
sys.modules.setdefault("rclpy.node", MagicMock(Node=_FakeNode))
sys.modules.setdefault("rclpy.qos", MagicMock())
sys.modules.setdefault("rclpy.action", MagicMock())
sys.modules.setdefault("rclpy.time", MagicMock())
for _mod in (
    "geometry_msgs",
    "geometry_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
    "tf2_ros",
    "nav2_msgs",
    "nav2_msgs.action",
):
    sys.modules.setdefault(_mod, MagicMock())

from src.ros.bridge import BridgeNode  # noqa: E402


def _path(points):
    """A fake nav_msgs/Path with the given (x, y) points."""
    poses = [
        SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)))
        for (x, y) in points
    ]
    return SimpleNamespace(poses=poses)


def _viz_stub():
    return SimpleNamespace(
        _viz_lock=threading.Lock(),
        _viz_global_plan=(),
        _viz_local_plan=(),
        _viz_plan_history=deque(maxlen=8),
        _viz_global_costmap=None,
        _viz_local_costmap=None,
        _viz_footprint=(),
        _viz_goal=None,
        _latest_map=None,
        _path_points=BridgeNode._path_points,
    )


def test_plan_trail_records_superseded_plans():
    b = _viz_stub()
    a = [(0.0, 0.0), (1.0, 1.0)]
    bb = [(0.0, 0.0), (1.0, 1.2)]
    c = [(0.0, 0.0), (1.0, 1.4)]

    BridgeNode._on_viz_global_plan(b, _path(a))
    assert b._viz_global_plan == tuple(a)
    assert list(b._viz_plan_history) == []  # first plan: nothing superseded

    BridgeNode._on_viz_global_plan(b, _path(a))  # republished, unchanged
    assert list(b._viz_plan_history) == []  # no-op, no duplicate in trail

    BridgeNode._on_viz_global_plan(b, _path(bb))
    assert b._viz_global_plan == tuple(bb)
    assert list(b._viz_plan_history) == [tuple(a)]

    BridgeNode._on_viz_global_plan(b, _path(c))
    assert b._viz_global_plan == tuple(c)
    assert list(b._viz_plan_history) == [tuple(a), tuple(bb)]


def test_plan_trail_is_bounded():
    b = _viz_stub()
    b._viz_plan_history = deque(maxlen=3)
    for i in range(10):
        BridgeNode._on_viz_global_plan(b, _path([(0.0, 0.0), (1.0, float(i))]))
    assert len(b._viz_plan_history) == 3  # bounded, keeps the most recent


def test_local_plan_and_footprint_cached():
    b = _viz_stub()
    BridgeNode._on_viz_local_plan(b, _path([(0.0, 0.0), (0.5, 0.5)]))
    assert b._viz_local_plan == ((0.0, 0.0), (0.5, 0.5))

    poly = SimpleNamespace(
        polygon=SimpleNamespace(
            points=[SimpleNamespace(x=0.0, y=0.0), SimpleNamespace(x=0.1, y=0.0)]
        )
    )
    BridgeNode._on_viz_footprint(b, poly)
    assert b._viz_footprint == ((0.0, 0.0), (0.1, 0.0))


def test_viz_snapshot_shape():
    b = _viz_stub()
    b._viz_global_plan = ((0.0, 0.0),)
    b._viz_goal = (1.0, 2.0, 0.5)
    b.get_pose_in_map = lambda: SimpleNamespace(x=1.0, y=2.0, theta=0.5)

    snap = BridgeNode.viz_snapshot(b)
    assert set(snap) == {
        "costmap",
        "local_costmap",
        "map",
        "global_plan",
        "plan_history",
        "local_plan",
        "footprint",
        "goal",
        "pose",
    }
    assert snap["pose"] == (1.0, 2.0, 0.5)
    assert snap["goal"] == (1.0, 2.0, 0.5)
    assert isinstance(snap["plan_history"], list)


def test_viz_snapshot_handles_missing_pose():
    b = _viz_stub()
    b.get_pose_in_map = lambda: None
    snap = BridgeNode.viz_snapshot(b)
    assert snap["pose"] is None
