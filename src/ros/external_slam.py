"""Pure helpers for consuming a Viam SLAM ``get_grid`` / pose response.

Used by builtin nav (``ViamWorldIO``) and tests. No rclpy / ROS publishers.
"""
from __future__ import annotations

import base64
import struct

from . import conversions as conv


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
    """Clamp cells to occupancy-grid semantics: -1 (unknown) or 0..100."""
    try:
        import numpy as np

        return np.clip(np.asarray(cells, dtype=np.int32), -1, 100).astype(
            np.int8
        ).tolist()
    except Exception:
        return [(-1 if c < -1 else 100 if c > 100 else int(c)) for c in cells]


def _grid_key(resp: dict):
    """Cheap change key for a get_grid response, or None if it can't be computed.

    A None key never matches (always treated as changed), so a key we can't hash
    is safe — it just skips the change-detection optimization for that cycle.
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
