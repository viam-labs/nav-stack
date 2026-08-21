import math

import numpy as np
import pytest

from src.ros import conversions as conv


def test_unit_conversions():
    assert conv.m_to_mm(1.5) == 1500.0
    assert conv.mm_to_m(2500.0) == 2.5


def test_forward_sector_min_range_detects_obstacle_ahead():
    ranges = np.full(720, np.inf)
    # angle_min=-pi, increment=2pi/720 → bin for angle≈0 is 360
    ranges[360] = 1.5
    ranges[100] = 0.4  # far off to the side; must not win
    scan = conv.LaserScan2D(
        ranges, -math.pi, (2.0 * math.pi) / 720, 0.05, 25.0
    )
    assert math.isclose(
        conv.forward_sector_min_range(scan, half_width_rad=math.radians(15.0)),
        1.5,
        abs_tol=1e-9,
    )


def test_nearest_return_bearing_deg_reports_offset_wall():
    ranges = np.full(720, np.inf)
    # Put nearest hit at about +45° (bin ≈ 360 + 90 = 450 for 720 bins).
    ranges[450] = 1.0
    ranges[360] = 3.0  # forward is farther — nearest bearing should be ~45°
    scan = conv.LaserScan2D(
        ranges, -math.pi, (2.0 * math.pi) / 720, 0.05, 25.0
    )
    bearing = conv.nearest_return_bearing_deg(scan)
    assert bearing is not None
    assert abs(bearing - 45.0) < 1.0


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


def test_pointcloud_to_scan_excludes_floor_for_livox_height_band():
    # Floor hits in base_link sit near z=0; a 3D lidar height band should drop
    # them so slam_toolbox does not see a floor arc.
    pts = np.array(
        [
            [3.0, 0.0, 0.02],
            [3.0, 0.0, 0.05],
            [2.0, 0.0, 0.8],
            [1.5, 1.0, 1.0],
        ]
    )
    scan = conv.pointcloud_to_scan(pts, z_min=0.12, z_max=1.5, num_bins=720)
    valid = np.asarray(scan.ranges, dtype=float)
    valid = valid[np.isfinite(valid)]
    assert valid.size == 2
    assert np.all(valid <= 2.05)
    assert np.all(valid >= 1.4)


def test_filter_points_by_z():
    pts = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 0.5], [0.0, 0.0, 2.0]])
    kept = conv.filter_points_by_z(pts, 0.12, 1.5)
    assert kept.shape == (1, 3)
    assert kept[0, 2] == 0.5


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


def test_parse_odom_from_readings_mir_base_ignores_imu_fields():
    """mir-base wheel odom must win even if IMU-like fields are also present."""
    reading = conv.parse_odom_from_readings(
        {
            "source": "odom+/odom+/status",
            "position_x_m": 10.0,
            "position_y_m": 20.0,
            "yaw_deg": 45.0,
            "linear_velocity_mps": {"x": 0.5, "y": 0.1, "z": 0.0},
            "angular_velocity_dps": {"x": 0.0, "y": 0.0, "z": 5.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 99.0},
            "orientation": {"yaw": 1.2, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 1.0, "y": 0.0, "z": 9.8},
        }
    )
    assert reading.pose is None
    assert reading.heading_rad is None
    assert reading.ax is None and reading.ay is None
    assert reading.vx == 0.5
    assert reading.vy == 0.1
    assert math.isclose(reading.vtheta, math.radians(5.0))


def test_parse_odom_from_readings_wheeled_odometry_y_forward():
    """rdk:builtin:wheeled-odometry reports forward velocity on +Y (compass
    frame); it must be remapped to ROS vx, not treated as lateral motion."""
    reading = conv.parse_odom_from_readings(
        {
            "position_meters_X": 0.0,
            "position_meters_Y": 1.2,
            "linear_velocity": {"x": 0.0, "y": 0.5, "z": 0.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 10.0},
            "orientation": {"ox": 0.0, "oy": 0.0, "oz": 1.0, "theta": 15.0},
        }
    )
    assert math.isclose(reading.vx, 0.5)
    assert math.isclose(reading.vy, 0.0, abs_tol=1e-12)
    assert math.isclose(reading.vtheta, math.radians(10.0))
    assert reading.pose is None


def test_parse_odom_from_readings_wheeled_odometry_parked():
    reading = conv.parse_odom_from_readings(
        {
            "position_meters_X": 0.0,
            "position_meters_Y": 0.0,
            "linear_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    )
    assert reading.vx == 0.0
    assert reading.vy == 0.0
    assert reading.vtheta == 0.0


def test_parse_odom_from_readings_viam_imu():
    """Wit-style IMU: euler orientation (rad) + angular_velocity (deg/s)."""
    reading = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 14.324},  # ~0.25 rad/s
            "orientation": {
                "yaw": -1.4876737428513436,
                "pitch": -0.014956312681885001,
                "roll": -0.008436894333371027,
            },
            "linear_acceleration": {"x": 0.14, "y": -0.09, "z": 9.83},
        }
    )
    assert reading.pose is None
    assert reading.heading_rad is not None
    assert math.isclose(reading.heading_rad, -1.4876737428513436, abs_tol=1e-6)
    assert reading.vx == 0.0
    assert reading.vy == 0.0
    assert math.isclose(reading.vtheta, math.radians(14.324), abs_tol=1e-4)


def test_parse_odom_angular_velocity_is_degrees_per_sec():
    """Viam GetAngularVelocity is deg/s — same as mir-base angular_velocity_dps."""
    reading = conv.parse_odom_from_readings(
        {"angular_velocity": {"x": 0.0, "y": 0.0, "z": 90.0}}
    )
    assert math.isclose(reading.vtheta, math.pi / 2, abs_tol=1e-6)


def test_parse_odom_from_readings_viam_imu_orientation_only():
    reading = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.5, "pitch": 0.0, "roll": 0.0},
        }
    )
    assert reading.heading_rad is not None
    assert math.isclose(reading.heading_rad, 0.5)
    assert reading.vtheta == 0.0
    assert reading.ax is None and reading.ay is None


def test_parse_odom_body_accel_removes_gravity_at_rest():
    reading = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {
                "yaw": -1.4876737428513436,
                "pitch": -0.014956312681885001,
                "roll": -0.008436894333371027,
            },
            "linear_acceleration": {
                "x": 0.143652099609375,
                "y": -0.086191259765625,
                "z": 9.82580361328125,
            },
        }
    )
    assert reading.ax is not None and reading.ay is not None
    assert abs(reading.ax) < 0.25
    assert abs(reading.ay) < 0.25


def test_merge_odom_heading_with_wheel_pose():
    wheel = conv.OdomReading(
        0.5,
        0.0,
        0.1,
        pose=conv.Pose2D(1.0, 2.0, 0.0),
    )
    merged = conv.merge_odom_heading(wheel, 1.5)
    assert merged.pose is not None
    assert merged.pose.x == 1.0
    assert merged.pose.y == 2.0
    assert merged.pose.theta == 1.5
    assert merged.heading_rad is None


def test_merge_odom_heading_with_wheel_twist_only():
    wheel = conv.OdomReading(0.4, 0.0, 0.05)
    merged = conv.merge_odom_heading(wheel, 0.8)
    assert merged.pose is None
    assert merged.heading_rad == 0.8
    assert merged.vx == 0.4


def test_apply_sensor_mount_yaw_rotates_accel_and_heading():
    """IMU mounted -90 deg about +z: forward accel shows up on its +y axis."""
    imu = conv.OdomReading(
        0.0,
        0.0,
        0.1,
        heading_rad=0.5,
        ax=0.0,
        ay=1.0,  # robot's forward accel, seen on the rotated IMU's y axis
    )
    fixed = conv.apply_sensor_mount_yaw(imu, math.radians(-90.0))
    assert math.isclose(fixed.ax, 1.0, abs_tol=1e-9)
    assert math.isclose(fixed.ay, 0.0, abs_tol=1e-9)
    # yaw offset removed so heading refers to robot forward
    assert math.isclose(fixed.heading_rad, 0.5 + math.pi / 2, abs_tol=1e-9)
    assert fixed.vtheta == 0.1


def test_apply_sensor_upside_down_flips_yaw_axes():
    imu = conv.OdomReading(
        0.2,
        0.1,
        0.3,
        heading_rad=1.0,
        ax=0.5,
        ay=0.2,
    )
    fixed = conv.apply_sensor_upside_down(imu)
    assert fixed.vx == 0.2
    assert fixed.vy == -0.1
    assert fixed.vtheta == -0.3
    assert math.isclose(fixed.heading_rad, -1.0)
    assert fixed.ax == 0.5
    assert fixed.ay == -0.2


def test_apply_sensor_mount_yaw_zero_is_noop():
    imu = conv.OdomReading(0.2, 0.0, 0.1, heading_rad=0.5, ax=0.3, ay=0.1)
    assert conv.apply_sensor_mount_yaw(imu, 0.0) is imu


def test_apply_sensor_mount_yaw_rotates_pose_theta_only():
    wheel = conv.OdomReading(
        0.5,
        0.0,
        0.0,
        pose=conv.Pose2D(1.0, 2.0, 1.0),
    )
    fixed = conv.apply_sensor_mount_yaw(wheel, math.radians(-90.0))
    assert fixed.pose.x == 1.0
    assert fixed.pose.y == 2.0
    assert math.isclose(fixed.pose.theta, 1.0 + math.pi / 2, abs_tol=1e-9)


def test_parse_heading_sensor_readings_euler():
    heading = conv.parse_heading_sensor_readings(
        {"orientation": {"yaw": 1.2, "pitch": 0.0, "roll": 0.0}}
    )
    assert math.isclose(heading, 1.2)


def test_parse_heading_small_degree_values_convert():
    """Keys named *_deg are always degrees, even below the 2*pi heuristic."""
    heading = conv.parse_heading_sensor_readings({"yaw_deg": 5.0})
    assert math.isclose(heading, math.radians(5.0))
    heading = conv._parse_heading_rad_from_readings(
        {"odom_yaw_deg": 3.0}, odom_only=True
    )
    assert math.isclose(heading, math.radians(3.0))


def test_parse_odom_twist_prefers_angular_velocity_for_yaw_rate():
    """linear_velocity.z is vertical speed, not turn rate; use angular_velocity."""
    vx, vy, vtheta = conv.parse_odom_twist_from_readings(
        {
            "linear_velocity": {"x": 1.0, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 30.0},  # deg/s
        }
    )
    assert math.isclose(vx, 1.0)
    assert math.isclose(vy, 0.0)
    assert math.isclose(vtheta, math.radians(30.0))


def test_parse_odom_from_readings_viam_vector3_objects():
    """Viam often returns Vector3/Euler objects, not plain dicts."""

    class _Vec:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Euler:
        def __init__(self, yaw, pitch=0.0, roll=0.0):
            self.yaw, self.pitch, self.roll = yaw, pitch, roll

    reading = conv.parse_odom_from_readings(
        {
            "angular_velocity": _Vec(0.0, 0.0, 5.73),  # deg/s ≈ 0.1 rad/s
            "orientation": _Euler(-0.95, -0.01, -0.008),
            "linear_acceleration": _Vec(0.14, -0.09, 9.83),
        }
    )
    assert reading.has_heading if False else reading.heading_rad is not None
    assert reading.ax is not None and reading.ay is not None
    assert math.isclose(reading.vtheta, math.radians(5.73), abs_tol=1e-3)


def test_gravity_compensation_independent_of_yaw():
    # Same roll/pitch at different yaw must yield the same body gravity.
    g0 = conv._gravity_specific_force_body(0.01, -0.02, 0.0)
    g1 = conv._gravity_specific_force_body(0.01, -0.02, 1.5)
    assert math.isclose(g0[0], g1[0], abs_tol=1e-9)
    assert math.isclose(g0[1], g1[1], abs_tol=1e-9)
    assert math.isclose(g0[2], g1[2], abs_tol=1e-9)


def test_transform_base_link_to_lidar_mount_roundtrip():
    pts = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    mount = dict(x=0.463, y=0.0, z=1.129, theta=0.1)
    base = conv.transform_lidar_mount_to_base_link(pts, **mount)
    back = conv.transform_base_link_to_lidar_mount(base, **mount)
    assert np.allclose(pts, back, atol=1e-6)


def test_transform_base_link_to_lidar_mount_roundtrip_with_tilt():
    pts = np.array([[15.0, 1.0, 0.2], [-3.0, -2.0, 0.9]])
    mount = dict(x=0.463, y=0.0, z=1.129, theta=0.1, pitch=0.035, roll=-0.01)
    base = conv.transform_lidar_mount_to_base_link(pts, **mount)
    back = conv.transform_base_link_to_lidar_mount(base, **mount)
    assert np.allclose(pts, back, atol=1e-6)


def test_transform_lidar_mount_pitch_levels_far_points():
    """A sensor pitched down 2 deg sees the floor 'rise' ahead; leveling with
    the mount pitch must restore the true base_link height at long range."""
    pitch = math.radians(2.0)
    sensor_z = 1.129
    # A point on the floor 15 m ahead, as seen by the tilted sensor: rotate the
    # true base_link point back into the sensor frame.
    floor_base = np.array([[15.0, 0.0, 0.0]])
    sensor_pts = conv.transform_base_link_to_lidar_mount(
        floor_base, x=0.0, y=0.0, z=sensor_z, theta=0.0, pitch=pitch
    )
    # Without tilt compensation the floor point lands ~0.5 m too high and
    # would pass a z_min=0.4 filter as an obstacle.
    uncompensated = conv.transform_lidar_mount_to_base_link(
        sensor_pts, x=0.0, y=0.0, z=sensor_z, theta=0.0
    )
    assert uncompensated[0, 2] > 0.4
    # With the configured pitch it comes back to floor height.
    leveled = conv.transform_lidar_mount_to_base_link(
        sensor_pts, x=0.0, y=0.0, z=sensor_z, theta=0.0, pitch=pitch
    )
    assert abs(leveled[0, 2]) < 1e-6


def test_base_link_cloud_to_lidar_scan():
    pts = np.array([[2.0, 0.0, 0.8], [0.0, 2.0, 0.8]])
    scan = conv.base_link_cloud_to_lidar_scan(
        pts,
        x=0.0,
        y=0.0,
        z=0.0,
        theta=0.0,
        z_min=0.12,
        z_max=1.5,
        num_bins=360,
        range_min=0.1,
        range_max=30.0,
    )
    assert conv.scan_has_returns(scan)


def test_merge_accumulated_rotation_only():
    pts = np.array([[2.0, 0.0, 1.0]])
    old = conv.Pose2D(0.0, 0.0, math.pi / 4)
    merged = conv.merge_accumulated_rotation_only([(pts, old)], 0.0)
    assert merged.shape == (1, 3)
    assert math.isclose(merged[0, 0], math.sqrt(2.0), abs_tol=0.01)


def test_transform_points_between_poses():
    pts = np.array([[1.0, 0.0, 0.5]])
    old = conv.Pose2D(0.0, 0.0, 0.0)
    new = conv.Pose2D(1.0, 0.0, 0.0)
    out = conv.transform_points_between_poses(pts, old, new)
    assert out.shape == (1, 3)
    assert math.isclose(out[0, 0], 0.0, abs_tol=1e-6)
    assert math.isclose(out[0, 1], 0.0, abs_tol=1e-6)
    assert math.isclose(out[0, 2], 0.5, abs_tol=1e-6)


def test_merge_accumulated_point_clouds():
    pts = np.array([[2.0, 0.0, 1.0]])
    old = conv.Pose2D(0.0, 0.0, 0.0)
    current = conv.Pose2D(0.5, 0.0, 0.0)
    merged = conv.merge_accumulated_point_clouds([(pts, old)], current)
    assert merged.shape == (1, 3)
    assert math.isclose(merged[0, 0], 1.5, abs_tol=1e-6)


def test_estimate_forward_range_flow_detects_forward_motion():
    n = 720
    angle_min = -math.pi
    angle_inc = 2 * math.pi / n
    angles = angle_min + np.arange(n) * angle_inc
    # Wall at x=5; drive forward 0.15 m.
    prev_r = np.where(
        np.abs(angles) < math.pi / 2,
        5.0 / np.maximum(np.cos(angles), 0.25),
        np.inf,
    )
    curr_r = np.where(
        np.abs(angles) < math.pi / 2,
        4.85 / np.maximum(np.cos(angles), 0.25),
        np.inf,
    )
    prev = conv.LaserScan2D(prev_r, angle_min, angle_inc, 0.1, 30.0)
    curr = conv.LaserScan2D(curr_r, angle_min, angle_inc, 0.1, 30.0)

    flow = conv.estimate_forward_range_flow(prev, curr, dtheta=0.0)
    assert flow is not None
    assert math.isclose(flow.dx, 0.15, abs_tol=0.06)
    assert flow.method == "range_flow"


def test_estimate_scan_motion_uses_range_flow_fallback():
    n = 720
    angle_min = -math.pi
    angle_inc = 2 * math.pi / n
    angles = angle_min + np.arange(n) * angle_inc
    prev_r = np.where(
        np.abs(angles) < math.pi / 2,
        5.0 / np.maximum(np.cos(angles), 0.25),
        np.inf,
    )
    curr_r = np.where(
        np.abs(angles) < math.pi / 2,
        4.85 / np.maximum(np.cos(angles), 0.25),
        np.inf,
    )
    prev = conv.LaserScan2D(prev_r, angle_min, angle_inc, 0.1, 30.0)
    curr = conv.LaserScan2D(curr_r, angle_min, angle_inc, 0.1, 30.0)

    motion = conv.estimate_scan_motion(prev, curr, dtheta=0.0)
    assert motion is not None
    assert math.isclose(motion.dx, 0.15, abs_tol=0.06)


def test_estimate_scan_motion_rejects_lateral_by_default():
    n = 90
    angle_min = -math.pi / 3
    angle_inc = (2 * math.pi / 3) / (n - 1)
    angles = angle_min + np.arange(n) * angle_inc
    # Pure lateral shift of a front wall is ambiguous; with allow_lateral=False
    # we only search dx, so a true dy motion should not invent a large dx.
    prev = conv.LaserScan2D(5.0 / np.cos(angles), angle_min, angle_inc, 0.1, 30.0)
    curr = conv.LaserScan2D(4.8 / np.cos(angles), angle_min, angle_inc, 0.1, 30.0)
    motion = conv.estimate_scan_motion(prev, curr, allow_lateral=False)
    assert motion is not None
    assert motion.dy == 0.0


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


def test_mir_laser_scan_payload_reports_age_s():
    payload = {
        "scans": [
            {
                "topic": "/f_raw_scan",
                "age_s": 0.12,
                "message": {
                    "header": {"frame_id": "front_laser_link"},
                    "angle_min": 0.0,
                    "angle_increment": math.pi / 2,
                    "range_min": 0.1,
                    "range_max": 40.0,
                    "ranges": [2.0, 3.0],
                },
            },
            {
                "topic": "/b_raw_scan",
                "age_s": 0.31,
                "message": {
                    "header": {"frame_id": "back_laser_link"},
                    "angle_min": 0.0,
                    "angle_increment": math.pi / 2,
                    "range_min": 0.1,
                    "range_max": 40.0,
                    "ranges": [2.0, 3.0],
                },
            },
        ]
    }
    pts = conv.points_from_mir_laser_scan_payload(payload)
    # Oldest scan wins so the merged cloud is stamped conservatively.
    assert pts.age_s == 0.31


def test_mir_laser_scan_payload_age_s_absent_is_none():
    payload = {
        "scans": [
            {
                "topic": "/f_raw_scan",
                "message": {
                    "header": {"frame_id": "front_laser_link"},
                    "angle_min": 0.0,
                    "angle_increment": math.pi / 2,
                    "range_min": 0.1,
                    "range_max": 40.0,
                    "ranges": [2.0, 3.0],
                },
            }
        ]
    }
    pts = conv.points_from_mir_laser_scan_payload(payload)
    assert pts.age_s is None


def test_mir_laser_scan_payload_top_level_age_s_fallback():
    payload = {
        "age_s": 0.2,
        "scans": [
            {
                "topic": "/f_raw_scan",
                "message": {
                    "header": {"frame_id": "front_laser_link"},
                    "angle_min": 0.0,
                    "angle_increment": math.pi / 2,
                    "range_min": 0.1,
                    "range_max": 40.0,
                    "ranges": [2.0, 3.0],
                },
            }
        ],
    }
    pts = conv.points_from_mir_laser_scan_payload(payload)
    assert pts.age_s == 0.2


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


def _side_wall_scan(
    *,
    y: float = 1.5,
    x0: float = -2.0,
    x1: float = 2.0,
    n: int = 80,
    yaw_bias_rad: float = 0.0,
) -> conv.LaserScan2D:
    """Synthetic left-side wall (y≈const) optionally rotated by ``yaw_bias_rad``."""
    xs = np.linspace(x0, x1, n)
    ys = np.full(n, y)
    c, s = math.cos(yaw_bias_rad), math.sin(yaw_bias_rad)
    xr = c * xs - s * ys
    yr = s * xs + c * ys
    pts = np.stack([xr, yr], axis=1)
    return conv.points_to_scan(pts, num_bins=720, range_min=0.05, range_max=25.0)


def test_extract_dominant_wall_parallel_left_wall():
    scan = _side_wall_scan(yaw_bias_rad=0.0)
    obs = conv.extract_dominant_wall(scan, min_length_m=2.0, min_inliers=20, seed=1)
    assert obs is not None
    assert obs.side == "left"
    assert abs(obs.wall_yaw_body) < math.radians(5.0)
    assert obs.length_m >= 2.0


def test_extract_dominant_wall_rejects_short_clutter():
    # Tiny cluster on the left — below min_length.
    pts = np.array([[0.1, 1.0], [0.2, 1.01], [0.3, 0.99]])
    scan = conv.points_to_scan(pts, num_bins=720, range_min=0.05, range_max=25.0)
    assert conv.extract_dominant_wall(scan, min_length_m=2.0, min_inliers=5) is None


def test_wall_yaw_correction_delta_caps_and_signs():
    # Wall appears +10° in body → rotate body by −10° (scaled/capped).
    delta = conv.wall_yaw_correction_delta(
        math.radians(10.0),
        max_step_rad=math.radians(2.0),
        blend=1.0,
    )
    assert delta == pytest.approx(-math.radians(2.0))
    half = conv.wall_yaw_correction_delta(
        math.radians(4.0),
        max_step_rad=math.radians(5.0),
        blend=0.5,
    )
    assert half == pytest.approx(-math.radians(2.0))


def test_compose_and_invert_pose_roundtrip():
    a = conv.Pose2D(1.0, -2.0, math.radians(30.0))
    b = conv.Pose2D(0.5, 0.25, math.radians(-75.0))
    ab = conv.compose_poses(a, b)
    back = conv.compose_poses(conv.invert_pose(a), ab)
    assert back.x == pytest.approx(b.x)
    assert back.y == pytest.approx(b.y)
    assert back.theta == pytest.approx(b.theta)


def test_map_pose_to_odom_pose_lands_prior_on_map_pose():
    map_to_odom = conv.Pose2D(3.0, -1.0, math.radians(20.0))
    desired_map_pose = conv.Pose2D(7.5, 2.0, math.radians(-40.0))
    odom_pose = conv.map_pose_to_odom_pose(desired_map_pose, map_to_odom)
    # map_to_odom ∘ odom_pose must reproduce the desired map pose exactly.
    got = conv.compose_poses(map_to_odom, odom_pose)
    assert got.x == pytest.approx(desired_map_pose.x)
    assert got.y == pytest.approx(desired_map_pose.y)
    assert got.theta == pytest.approx(desired_map_pose.theta)


def test_extract_dominant_wall_angled_then_corrects():
    bias = math.radians(8.0)
    scan = _side_wall_scan(yaw_bias_rad=bias)
    obs = conv.extract_dominant_wall(
        scan, min_length_m=2.0, min_inliers=20, parallel_tol_rad=math.radians(40), seed=2
    )
    assert obs is not None
    assert abs(obs.wall_yaw_body - bias) < math.radians(3.0)
    delta = conv.wall_yaw_correction_delta(
        obs.wall_yaw_body, max_step_rad=math.radians(2.0), blend=0.5
    )
    # Soft step toward canceling the bias (negative of wall yaw).
    assert delta < 0.0
    assert abs(delta) <= math.radians(2.0) + 1e-9
