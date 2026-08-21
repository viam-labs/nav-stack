"""Bridge an arbitrary Viam SLAM service into the ROS graph for Nav2.

The built-in path runs slam_toolbox, which publishes ``/map`` and the
``map -> odom`` TF natively. An external ``rdk:service:slam`` does not speak ROS,
so this publisher polls it and republishes what Nav2 needs:

* ``GetPosition()``            -> ``map -> odom`` TF
* ``get_grid`` DoCommand       -> ``/map`` OccupancyGrid

It attaches to the already-running :class:`~.bridge.BridgeNode` so it shares the
rclpy context, the module event loop (via ``node._run``), the TF broadcaster,
and — crucially — the bridge's live ``odom -> base_link`` estimate, which keeps
``map -> odom`` coherent with the odom TF the bridge publishes.

``map -> odom`` derivation (all 2D transforms):

    map_to_base  = SLAM GetPosition (robot pose in the map frame)
    odom_to_base = bridge's current odom pose
    map_to_odom  = map_to_base ∘ odom_to_base⁻¹
"""
from __future__ import annotations

import base64
import struct
from typing import Callable, Optional

from . import conversions as conv

# ROS message/QoS imports are done lazily inside methods (see manager.py's lazy
# ``import rclpy``) so the pure helpers below stay importable without a sourced
# ROS environment — keeps parse_get_grid / slam_pose_to_pose2d unit-testable.


def _latched_map_qos():
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )

    # Latched (transient-local) so a late-joining Nav2 costmap still gets the
    # most recent map without waiting for the next publish.
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def slam_pose_to_pose2d(pose) -> conv.Pose2D:
    """Project a Viam SLAM ``Pose`` (mm + orientation vector) onto the ground.

    Position mm -> m; yaw from the orientation vector (o_x, o_y, o_z, theta_deg),
    so a tilted 3D SLAM pose still yields the correct planar heading.
    """
    _roll, _pitch, yaw = conv.euler_from_orientation_vector(
        pose.o_x, pose.o_y, pose.o_z, pose.theta
    )
    return conv.Pose2D(conv.mm_to_m(pose.x), conv.mm_to_m(pose.y), yaw)


def _decode_grid_cells(data) -> list:
    """Decode a ``get_grid`` ``data`` field into a list of int8 occupancy cells.

    Viam DoCommand payloads travel as a protobuf ``Struct``, which has no bytes
    type, so services encode the raw int8 mat as one of: base64 string, a bytes
    object (some transports), or a plain list of ints. Accept all three.
    """
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    elif isinstance(data, str):
        raw = base64.b64decode(data)
    elif isinstance(data, (list, tuple)):
        return [int(v) for v in data]
    else:
        raise TypeError(f"unsupported get_grid 'data' type: {type(data).__name__}")
    return list(struct.unpack(f"{len(raw)}b", raw))  # 'b' = signed int8


class ExternalSlamPublisher:
    """Poll a Viam SLAM service and publish ``/map`` + ``map -> odom`` into ROS."""

    def __init__(
        self,
        node,
        slam,
        frames,
        get_odom: Callable[[], Optional[conv.Pose2D]],
        *,
        pose_rate_hz: float = 10.0,
        grid_rate_hz: float = 1.5,
        transform_timeout_s: float = 0.2,
        logger=None,
    ):
        self._node = node
        self._slam = slam
        self._frames = frames
        self._get_odom = get_odom
        self._transform_timeout_s = float(transform_timeout_s)
        self._logger = logger
        # Shared with the SlamRuntime so navigation's get_status can report it.
        self.localization_check: dict = {"status": "starting"}
        self._last_grid_key = None

        from nav_msgs.msg import OccupancyGrid
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

        self._map_pub = node.create_publisher(OccupancyGrid, "map", _latched_map_qos())
        # Separate callback groups: the grid poll blocks on a remote get_grid
        # (up to the _run 5s timeout), so it must not share a mutually-exclusive
        # group with the pose timer or a slow grid read would starve the
        # map->odom broadcast and stall Nav2's TF chain.
        self._pose_cb_group = MutuallyExclusiveCallbackGroup()
        self._grid_cb_group = MutuallyExclusiveCallbackGroup()
        node.create_timer(
            1.0 / max(pose_rate_hz, 1.0),
            node._guarded(self._on_pose),
            callback_group=self._pose_cb_group,
        )
        node.create_timer(
            1.0 / max(grid_rate_hz, 0.2),
            node._guarded(self._on_grid),
            callback_group=self._grid_cb_group,
        )
        self._log(
            f"external SLAM publisher started (pose {pose_rate_hz} Hz, grid {grid_rate_hz} Hz)"
        )

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)

    # -- map -> odom TF ------------------------------------------------------
    def _on_pose(self) -> None:
        odom_to_base = self._get_odom()
        if odom_to_base is None:
            return  # odom not flowing yet; nothing to anchor map->odom to
        pose = self._node._run(self._slam.get_position())
        if pose is None:
            self._set_loc_check({"status": "no_pose"})
            return
        map_to_base = slam_pose_to_pose2d(pose)
        map_to_odom = conv.compose_poses(map_to_base, conv.invert_pose(odom_to_base))
        self._broadcast_map_to_odom(map_to_odom)
        self._set_loc_check(
            {
                "status": "ok",
                "map_to_base": {
                    "x": map_to_base.x,
                    "y": map_to_base.y,
                    "theta": map_to_base.theta,
                },
            }
        )

    def _set_loc_check(self, values: dict) -> None:
        # Mutate in place: the SlamRuntime holds a reference to this same dict,
        # so navigation's get_status sees updates without re-fetching.
        self.localization_check.clear()
        self.localization_check.update(values)

    def _broadcast_map_to_odom(self, mo: conv.Pose2D) -> None:
        from geometry_msgs.msg import TransformStamped
        from rclpy.duration import Duration

        now = self._node.get_clock().now()
        # Stamp slightly in the future (transform_timeout), matching slam_toolbox:
        # map->odom must be newer than odom->base_link or TF drops the chain.
        stamp = (now + Duration(seconds=self._transform_timeout_s)).to_msg()
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._frames.map
        t.child_frame_id = self._frames.odom
        t.transform.translation.x = float(mo.x)
        t.transform.translation.y = float(mo.y)
        t.transform.translation.z = 0.0
        qx, qy, qz, qw = conv.yaw_to_quaternion(mo.theta)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._node._tf_broadcaster.sendTransform(t)

    # -- /map OccupancyGrid --------------------------------------------------
    def _on_grid(self) -> None:
        resp = self._node._run(
            self._slam.do_command({"command": "get_grid"}), timeout=5.0
        )
        if not isinstance(resp, dict):
            return
        # Skip the rebuild+publish when the grid is unchanged: /map is latched
        # (transient-local), so a late subscriber still gets the last map. Avoids
        # a multi-MB decode+serialize spike every cycle on large, static maps.
        key = _grid_key(resp)
        if key is not None and key == self._last_grid_key:
            return
        grid = self._occupancy_from_get_grid(resp)
        if grid is not None:
            self._map_pub.publish(grid)
            self._last_grid_key = key

    def _occupancy_from_get_grid(self, resp: dict) -> Optional["OccupancyGrid"]:  # noqa: F821 - lazy rclpy import
        parsed = parse_get_grid(resp)
        if parsed is None:
            self._log("get_grid: malformed/inconsistent response; skipping")
            return None
        rows, cols, cell_size, x_min, y_min, cells = parsed
        from nav_msgs.msg import OccupancyGrid

        grid = OccupancyGrid()
        grid.header.stamp = self._node.get_clock().now().to_msg()
        grid.header.frame_id = self._frames.map
        grid.info.resolution = cell_size
        grid.info.width = cols
        grid.info.height = rows
        grid.info.origin.position.x = x_min
        grid.info.origin.position.y = y_min
        grid.info.origin.orientation.w = 1.0
        # NOTE: assumes get_grid cells are row-major, bottom row first (ROS
        # OccupancyGrid convention, origin at the min corner). If a SLAM service
        # emits image-convention rows (top row first), /map is flipped in Y and
        # obstacles land mirrored — verify against the room in rviz and apply a
        # per-row flip here if so. (Left un-flipped until confirmed on hardware.)
        grid.data = cells
        return grid


def parse_get_grid(resp: dict):
    """Validate + decode a ``get_grid`` response into OccupancyGrid inputs.

    Returns ``(rows, cols, cell_size, x_min, y_min, cells)`` or ``None`` when the
    response is malformed or the cell count doesn't match ``rows * cols``.
    Accepts both camelCase (``xMin``/``cellSize``) and snake_case keys.
    """
    if not isinstance(resp, dict):
        return None
    try:
        rows = int(resp["rows"])
        cols = int(resp["cols"])
        cell_size = float(resp.get("cellSize", resp.get("cell_size")))
        x_min = float(resp.get("xMin", resp.get("x_min")))
        y_min = float(resp.get("yMin", resp.get("y_min")))
        cells = _decode_grid_cells(resp["data"])
    except (KeyError, TypeError, ValueError):
        return None
    if rows <= 0 or cols <= 0 or cell_size <= 0 or len(cells) != rows * cols:
        return None
    return rows, cols, cell_size, x_min, y_min, _clamp_occupancy(cells)


def _clamp_occupancy(cells: list) -> list:
    """Clamp cells to ROS OccupancyGrid semantics: -1 (unknown) or 0..100.

    ``OccupancyGrid.data`` is ``int8[]``; a source that sends out-of-range values
    (e.g. a 0-255 probability grid instead of the documented -1/0/100) would
    overflow int8 on assignment and raise — swallowed by the _guarded timer, so
    ``/map`` would silently never publish. Clamp defensively.
    """
    try:
        import numpy as np

        return np.clip(np.asarray(cells, dtype=np.int32), -1, 100).astype(
            np.int8
        ).tolist()
    except Exception:
        return [(-1 if c < -1 else 100 if c > 100 else int(c)) for c in cells]


def _grid_key(resp: dict):
    """Cheap change key for a get_grid response, or None if it can't be computed.

    A None key never matches (always republishes), so a key we can't hash is
    safe — it just skips the change-detection optimization for that cycle.
    """
    try:
        data = resp["data"]
        if isinstance(data, (bytes, bytearray)):
            data_hash = hash(bytes(data))
        elif isinstance(data, str):
            data_hash = hash(data)
        else:
            data_hash = hash(tuple(data))
        return (
            resp.get("rows"),
            resp.get("cols"),
            resp.get("cellSize", resp.get("cell_size")),
            resp.get("xMin", resp.get("x_min")),
            resp.get("yMin", resp.get("y_min")),
            data_hash,
        )
    except Exception:
        return None
