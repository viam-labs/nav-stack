import math

import numpy as np

from src.ros import conversions as conv


def test_unit_conversions():
    assert conv.m_to_mm(1.5) == 1500.0
    assert conv.mm_to_m(2500.0) == 2.5


def test_yaw_quaternion_roundtrip():
    for yaw in (-3.0, -1.0, 0.0, 0.5, 1.5, 3.0):
        x, y, z, w = conv.yaw_to_quaternion(yaw)
        assert math.isclose(conv.quaternion_to_yaw(x, y, z, w), yaw, abs_tol=1e-9)


def test_viam_pose_roundtrip():
    p = conv.viam_pose_to_pose2d(1000.0, -2000.0, 90.0)
    assert math.isclose(p.x, 1.0)
    assert math.isclose(p.y, -2.0)
    assert math.isclose(p.theta, math.pi / 2)
    x_mm, y_mm, theta_deg = conv.pose2d_to_viam_pose(p)
    assert math.isclose(x_mm, 1000.0)
    assert math.isclose(y_mm, -2000.0)
    assert math.isclose(theta_deg, 90.0)


def test_pcd_roundtrip():
    pts = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    raw = conv.points_to_pcd(pts)
    back = conv.parse_pcd(raw)
    assert back.shape == (2, 3)
    assert np.allclose(back, pts)


def test_parse_pcd_ascii():
    ascii_pcd = (
        b"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        b"WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n1 2 3\n4 5 6\n"
    )
    pts = conv.parse_pcd(ascii_pcd)
    assert np.allclose(pts, [[1, 2, 3], [4, 5, 6]])


def test_occupancy_grid_to_pcd_picks_occupied():
    grid = np.array([[0, 100], [-1, 80]], dtype=np.int16)
    raw = conv.occupancy_grid_to_pcd(grid, resolution=1.0, origin_x=0.0, origin_y=0.0)
    pts = conv.parse_pcd(raw)
    # Two cells exceed the default threshold (100 and 80).
    assert pts.shape[0] == 2
    assert np.allclose(pts[:, 2], 0.0)


def test_points_to_scan_keeps_nearest():
    # Two points in the same direction; nearest should win.
    pts = np.array([[2.0, 0.0], [1.0, 0.0]])
    scan = conv.points_to_scan(pts, num_bins=360, range_min=0.1, range_max=10.0)
    finite = scan.ranges[np.isfinite(scan.ranges)]
    assert math.isclose(finite.min(), 1.0, abs_tol=1e-6)


def test_merge_scans_combines_lidars():
    front = conv.LaserScan2D(
        ranges=np.array([1.0]), angle_min=0.0, angle_increment=0.1,
        sensor_pose=conv.Pose2D(0.0, 0.0, 0.0),
    )
    rear = conv.LaserScan2D(
        ranges=np.array([1.0]), angle_min=math.pi, angle_increment=0.1,
        sensor_pose=conv.Pose2D(0.0, 0.0, math.pi),
    )
    merged = conv.merge_scans([front, rear], num_bins=360)
    assert np.isfinite(merged.ranges).sum() >= 1


def test_pointcloud_to_scan_filters_height():
    pts = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 5.0]])
    scan = conv.pointcloud_to_scan(pts, z_min=-0.2, z_max=2.0, num_bins=360)
    assert np.isfinite(scan.ranges).sum() == 1


def test_parse_pcd_malformed_no_fields():
    # Binary PCD with DATA section but no FIELDS line — must not raise ZeroDivisionError.
    raw = b"VERSION .7\nWIDTH 1\nHEIGHT 1\nPOINTS 0\nDATA binary\n"
    pts = conv.parse_pcd(raw)
    assert pts.shape == (0, 3)


def test_parse_pcd_malformed_missing_xyz():
    raw = (
        b"VERSION .7\nFIELDS intensity\nSIZE 4\nTYPE F\nCOUNT 1\n"
        b"WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n\x00\x00\x00\x00"
    )
    pts = conv.parse_pcd(raw)
    assert pts.shape == (0, 3)


def test_parse_odom_pose_from_readings_flat():
    pose = conv.parse_odom_pose_from_readings(
        {"x": 1.5, "y": -2.0, "theta": 90.0}
    )
    assert pose is not None
    assert pose.x == 1.5
    assert pose.y == -2.0
    assert math.isclose(pose.theta, math.radians(90.0))


def test_parse_odom_pose_from_readings_nested_pose():
    pose = conv.parse_odom_pose_from_readings(
        {"pose": {"x": 0.5, "y": 1.0, "theta": 0.25}}
    )
    assert pose is not None
    assert math.isclose(pose.theta, 0.25)


def test_parse_odom_pose_from_readings_quaternion():
    x, y, z, w = conv.yaw_to_quaternion(math.pi / 2)
    pose = conv.parse_odom_pose_from_readings(
        {"x": 0.0, "y": 0.0, "orientation": {"x": x, "y": y, "z": z, "w": w}}
    )
    assert pose is not None
    assert math.isclose(pose.theta, math.pi / 2, abs_tol=1e-6)


def test_parse_odom_from_readings_mir_base():
    """mir-base yaw_deg is map-fused; must not drive /odom without odom_* fields."""
    reading = conv.parse_odom_from_readings(
        {
            "source": "odom+/odom+/status",
            "position_x_m": 10.0,
            "position_y_m": 20.0,
            "yaw_deg": 45.0,
            "linear_velocity_mps": {"x": 0.5, "y": 0.1, "z": 0.0},
            "angular_velocity_dps": {"x": 0.0, "y": 0.0, "z": 5.0},
        }
    )
    assert reading.pose is None
    assert reading.heading_rad is None
    assert reading.vx == 0.5
    assert reading.vy == 0.1
    assert math.isclose(reading.vtheta, math.radians(5.0))


def test_parse_odom_from_readings_mir_base_odom_fields():
    reading = conv.parse_odom_from_readings(
        {
            "odom_position_x_m": 1.0,
            "odom_position_y_m": 2.0,
            "odom_yaw_deg": 90.0,
            "linear_velocity_mps": {"x": 0.2, "y": 0.0, "z": 0.0},
            "angular_velocity_dps": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    )
    assert reading.pose is not None
    assert reading.pose.x == 1.0
    assert reading.pose.y == 2.0
    assert math.isclose(reading.pose.theta, math.radians(90.0))
    assert reading.heading_rad is None


def test_parse_odom_from_readings_ros_odom_message():
    reading = conv.parse_odom_from_readings(
        {
            "pose": {
                "pose": {
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            },
            "twist": {
                "twist": {
                    "linear": {"x": 0.3, "y": 0.0, "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": 0.2},
                }
            },
        }
    )
    assert reading.pose is not None
    assert reading.pose.x == 1.0
    assert reading.pose.y == 2.0
    assert reading.heading_rad is None
    assert reading.vx == 0.3
    assert math.isclose(reading.vtheta, 0.2)


def test_mir_laser_scan_payload_skips_zero_and_out_of_range():
    payload = {
        "scans": [
            {
                "topic": "/f_raw_scan",
                "frame_id": "front_laser_link",
                "message": {
                    "header": {"frame_id": "front_laser_link"},
                    "angle_min": 0.0,
                    "angle_increment": math.pi / 2,
                    "range_min": 0.1,
                    "range_max": 40.0,
                    "ranges": [0, 0, 2.0, 41.0],
                },
            }
        ]
    }
    pts = conv.points_from_mir_laser_scan_payload(payload)
    assert pts.sensor.shape[0] == 1
    assert pts.base_link.shape[0] == 1
    assert pts.sensor_scan is not None
    assert conv.scan_has_returns(pts.sensor_scan)
    scan = conv.pointcloud_to_scan(pts.base_link, num_bins=720)
    assert conv.scan_has_returns(scan)


def test_mir_native_scan_preserves_negative_angle_increment():
    msg = {
        "header": {"frame_id": "front_laser_link"},
        "angle_min": 2.4,
        "angle_max": -2.4,
        "angle_increment": -0.01,
        "range_min": 0.1,
        "range_max": 40.0,
        "ranges": [0, 0, 6.0, 0],
    }
    scan = conv.mir_laser_scan_message_to_scan2d(msg)
    assert conv.scan_has_returns(scan)
    assert math.isclose(float(scan.ranges[2]), 6.0)
