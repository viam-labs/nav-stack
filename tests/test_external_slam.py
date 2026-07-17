import base64
import math
import struct

import pytest

pytest.importorskip("viam")
pytest.importorskip("viam.spatialmath")

from viam.proto.common import Pose

from src.ros import conversions as conv
from src.ros.external_slam import (
    _decode_grid_cells,
    _grid_key,
    parse_get_grid,
    slam_pose_to_pose2d,
)


def test_slam_pose_to_pose2d_mm_and_orientation_vector():
    # 1 m forward, 2 m left, yaw 90 deg (OV = +z axis, theta 90).
    pose = Pose(x=1000.0, y=2000.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=90.0)
    p = slam_pose_to_pose2d(pose)
    assert p.x == pytest.approx(1.0)
    assert p.y == pytest.approx(2.0)
    assert p.theta == pytest.approx(math.pi / 2)


def test_decode_grid_cells_accepts_bytes_base64_and_list():
    values = [-1, 0, 100, 50]
    raw = struct.pack(f"{len(values)}b", *values)
    assert _decode_grid_cells(raw) == values
    assert _decode_grid_cells(base64.b64encode(raw).decode()) == values
    assert _decode_grid_cells(list(values)) == values


def test_decode_grid_cells_rejects_unknown_type():
    with pytest.raises(TypeError):
        _decode_grid_cells(42)


def test_parse_get_grid_camelcase():
    cells = [0, 100, -1, 0, 0, 100]
    resp = {
        "rows": 2,
        "cols": 3,
        "cellSize": 0.05,
        "xMin": -1.5,
        "yMin": -2.0,
        "encoding": "raw_int8",
        "data": cells,
    }
    rows, cols, cell_size, x_min, y_min, out = parse_get_grid(resp)
    assert (rows, cols) == (2, 3)
    assert cell_size == pytest.approx(0.05)
    assert (x_min, y_min) == (-1.5, -2.0)
    assert out == cells


def test_parse_get_grid_snake_case_fallback():
    resp = {"rows": 1, "cols": 2, "cell_size": 0.1, "x_min": 0.0, "y_min": 0.0, "data": [0, 0]}
    assert parse_get_grid(resp)[2] == pytest.approx(0.1)


def test_parse_get_grid_length_mismatch_returns_none():
    resp = {"rows": 2, "cols": 3, "cellSize": 0.05, "xMin": 0, "yMin": 0, "data": [0, 0]}
    assert parse_get_grid(resp) is None


def test_parse_get_grid_malformed_returns_none():
    assert parse_get_grid({"rows": 2}) is None
    assert parse_get_grid("nope") is None
    assert parse_get_grid({"rows": 0, "cols": 0, "cellSize": 0.05, "xMin": 0, "yMin": 0, "data": []}) is None


def test_parse_get_grid_rejects_nonpositive_resolution():
    base = {"rows": 1, "cols": 2, "xMin": 0, "yMin": 0, "data": [0, 0]}
    assert parse_get_grid({**base, "cellSize": 0.0}) is None
    assert parse_get_grid({**base, "cellSize": -0.05}) is None
    assert parse_get_grid({**base, "cellSize": 0.05}) is not None


def test_parse_get_grid_clamps_out_of_range_cells():
    # A 0-255 probability grid (or stray negatives) must be clamped to -1..100
    # so int8 assignment can't overflow and silently drop /map.
    resp = {
        "rows": 1,
        "cols": 6,
        "cellSize": 0.05,
        "xMin": 0,
        "yMin": 0,
        "data": [0, 200, -5, 100, 255, -1],
    }
    out = parse_get_grid(resp)[5]
    assert out == [0, 100, -1, 100, 100, -1]
    assert all(-1 <= v <= 100 for v in out)


def test_grid_key_stable_and_change_sensitive():
    resp = {"rows": 2, "cols": 2, "cellSize": 0.05, "xMin": 0, "yMin": 0, "data": [0, 0, 0, 100]}
    assert _grid_key(resp) == _grid_key(dict(resp))  # same content -> same key
    changed = {**resp, "data": [0, 0, 100, 100]}
    assert _grid_key(resp) != _grid_key(changed)  # different cells -> different key
    moved = {**resp, "xMin": 1.0}
    assert _grid_key(resp) != _grid_key(moved)  # different origin -> different key


def test_map_to_odom_derivation_identity():
    # map_to_odom ∘ odom_to_base must reconstruct map_to_base exactly.
    map_to_base = conv.Pose2D(3.0, -1.0, math.radians(40))
    odom_to_base = conv.Pose2D(0.5, 0.25, math.radians(15))
    map_to_odom = conv.compose_poses(map_to_base, conv.invert_pose(odom_to_base))
    recon = conv.compose_poses(map_to_odom, odom_to_base)
    assert recon.x == pytest.approx(map_to_base.x)
    assert recon.y == pytest.approx(map_to_base.y)
    assert conv.normalize_angle(recon.theta - map_to_base.theta) == pytest.approx(0.0)
