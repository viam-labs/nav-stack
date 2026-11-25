"""Offline vision evaluation: dataset reading and scoring rules."""

from __future__ import annotations

import math
import zipfile

import numpy as np
import pytest

from src.geom import conversions as conv
from src.semantic.types import CameraIntrinsics
from tools.fetch_lilocbench import choose_members
from tools.vision_eval.lilocbench import (
    interpolate_pose,
    mount_from_matrix,
    register_depth_to_color,
)
from tools.vision_eval.place_eval import (
    PlaceRanking,
    pose_errors,
    recall_at_k,
    select_keyframes,
)
from tools.vision_eval.reloc_eval import Attempt, summarize


def _info(name: str) -> zipfile.ZipInfo:
    return zipfile.ZipInfo(name)


def test_fetch_keeps_small_files_and_thins_one_camera_to_the_asked_rate():
    infos = [_info("transformations.yaml"), _info("laser_scan_front/ranges.txt")]
    for i in range(30):  # 15 Hz for two seconds
        stamp = 100.0 + i / 15.0
        infos.append(_info(f"camera_front/color/images/{stamp:.9f}.png"))
        infos.append(_info(f"camera_front/depth/images/{stamp + 0.001:.9f}.png"))
        infos.append(_info(f"camera_left/color/images/{stamp:.9f}.png"))
    keep, counts = choose_members(infos, cameras=["front"], rate_hz=2.0)
    names = [m.filename for m in keep]
    assert "transformations.yaml" in names and "laser_scan_front/ranges.txt" in names
    assert not any(n.startswith("camera_left") for n in names)
    assert counts == {"front": 4}
    assert sum(n.startswith("camera_front/color") for n in names) == 4
    assert sum(n.startswith("camera_front/depth") for n in names) == 4


def test_fetch_skips_a_colour_frame_with_no_depth_close_in_time():
    infos = [
        _info("camera_front/color/images/100.000000000.png"),
        _info("camera_front/depth/images/100.300000000.png"),
    ]
    keep, counts = choose_members(infos, cameras=["front"], rate_hz=2.0)
    assert counts == {"front": 0} and keep == []


def test_pose_is_interpolated_across_the_heading_wrap_and_refused_across_a_gap():
    stamps = np.array([0.0, 0.1, 5.0])
    poses = np.array(
        [[0.0, 0.0, math.radians(175)], [1.0, 2.0, math.radians(-175)], [9.0, 9.0, 0.0]]
    )
    mid = interpolate_pose(stamps, poses, 0.05)
    assert mid is not None
    assert abs(mid.x - 0.5) < 1e-9 and abs(mid.y - 1.0) < 1e-9
    assert abs(abs(math.degrees(mid.theta)) - 180.0) < 1e-6
    assert interpolate_pose(stamps, poses, 2.0) is None
    assert interpolate_pose(stamps, poses, -1.0) is None


def test_mount_angles_rebuild_the_rotation_the_stack_uses():
    want = conv._mount_rotation(0.3, 0.05, -0.02)  # noqa: SLF001
    mat = np.eye(4)
    mat[:3, :3] = want
    mat[:3, 3] = (0.1, -0.2, 0.66)
    x, y, z, theta, pitch, roll = mount_from_matrix(mat)
    assert (x, y, z) == (0.1, -0.2, 0.66)
    assert np.allclose(conv._mount_rotation(theta, pitch, roll), want)  # noqa: SLF001


def test_depth_is_shifted_into_the_colour_camera():
    k = CameraIntrinsics(fx=100.0, fy=100.0, cx=32.0, cy=24.0, width=64, height=48)
    depth = np.zeros((48, 64), dtype=np.float32)
    depth[24, 32] = 2.0
    shift = np.eye(4)
    shift[0, 3] = 0.1  # depth sensor 10 cm to the right of colour
    out = register_depth_to_color(depth, k, k, shift)
    # 0.1 m at 2 m and fx=100 is 5 px.
    assert out[24, 37] == 2.0 and np.count_nonzero(out) == 1


def test_keyframes_follow_distance_or_turning():
    poses = [conv.Pose2D(0.1 * i, 0.0, 0.0) for i in range(20)]
    assert select_keyframes(poses) == [0, 8, 16]
    spin = [conv.Pose2D(0.0, 0.0, math.radians(10 * i)) for i in range(7)]
    assert select_keyframes(spin) == [0, 3, 6]


def test_recall_counts_a_close_keyframe_in_the_top_k_and_reports_the_ceiling():
    keyframes = [
        conv.Pose2D(0.0, 0.0, 0.0),
        conv.Pose2D(5.0, 0.0, 0.0),
        conv.Pose2D(0.5, 0.0, math.pi),
    ]
    queries = [conv.Pose2D(0.2, 0.0, 0.0), conv.Pose2D(20.0, 0.0, 0.0)]
    dist, dyaw = pose_errors(keyframes, queries)
    # Query 0 ranks the far keyframe first, then the reversed one, then the right one.
    ranking = PlaceRanking("t", np.array([[1, 2, 0], [0, 1, 2]]), np.zeros((2, 3)))
    any_heading = recall_at_k(ranking, dist, dyaw, ks=(1, 2, 3), max_dist_m=1.5)
    assert any_heading == {"possible": 0.5, "top1": 0.0, "top2": 0.5, "top3": 0.5}
    same = recall_at_k(
        ranking, dist, dyaw, ks=(1, 2, 3), max_dist_m=1.5, max_yaw_deg=45.0
    )
    assert same == {"possible": 0.5, "top1": 0.0, "top2": 0.0, "top3": 0.5}


def test_summary_separates_a_wrong_pose_that_would_be_applied():
    attempts = [
        Attempt(0.1, 2.0, True, 0.6),  # right, applied
        Attempt(0.1, 2.0, False, 0.2),  # right, refused
        Attempt(8.0, 90.0, True, 0.6),  # wrong, applied
        None,  # no match at all
    ]
    out = summarize(attempts)
    assert out == {
        "queries": 4,
        "right": 0.5,
        "accepted_right": 0.25,
        "accepted_wrong": 0.25,
        "refused": 0.5,
    }


# --- Replay through the SLAM engine --------------------------------------------

import time  # noqa: E402

from tools.vision_eval.replay import (  # noqa: E402
    ReplayClock,
    ReplaySensors,
    Trajectory,
    align_se2,
    replay_mapping,
)


def test_alignment_removes_the_frame_difference_and_leaves_the_real_error():
    rng = np.random.default_rng(0)
    truth = np.column_stack(
        [np.cumsum(rng.normal(0.1, 0.05, 80)), np.cumsum(rng.normal(0.0, 0.08, 80))]
    )
    c, s = math.cos(0.7), math.sin(0.7)
    est = (truth - (3.0, -1.0)) @ np.array(
        [[c, -s], [s, c]]
    )  # same path, another frame
    rot, shift, yaw = align_se2(est, truth)
    assert np.allclose(est @ rot.T + shift, truth, atol=1e-9)
    assert abs(yaw - 0.7) < 1e-9

    stamps = np.arange(80.0)
    same = Trajectory(
        stamps,
        np.column_stack([est, np.zeros(80) - 0.7]),
        np.column_stack([truth, np.zeros(80)]),
    )
    assert (
        same.summary()["ate_rmse_m"] < 1e-9
        and same.summary()["heading_rmse_deg"] < 1e-6
    )


def test_drift_is_measured_from_a_known_start_not_fitted_away():
    stamps = np.arange(0.0, 100.0)
    truth = np.column_stack([stamps * 0.1, np.zeros(100), np.zeros(100)])
    est = truth.copy()
    est[:, 1] = np.where(
        stamps > 20.0, (stamps - 20.0) * 0.01, 0.0
    )  # slides sideways after 20 s
    path = Trajectory(stamps, est, truth)
    drift, _yaw = path.errors(anchor_s=20.0)
    assert drift[:21].max() < 1e-9 and drift[-1] == pytest.approx(0.79, abs=1e-6)
    fitted, _yaw = path.errors()
    assert fitted.max() < drift[-1]  # a whole-path fit hides part of it
    assert path.summary()["drift_end_m"] == pytest.approx(0.79, abs=1e-6)


def test_replay_clock_stands_in_for_monotonic_and_is_put_back():
    real = time.monotonic
    with ReplayClock(first_stamp=5000.0) as clock:
        clock.now = clock.of(5012.5)
        assert time.monotonic() == clock.now
        assert clock.stamp(time.monotonic()) == pytest.approx(5012.5)
    assert time.monotonic is real


class _RoomRun:
    """A robot driving along a 6 m x 4 m room, as a recorded run would describe it."""

    name = "room"
    camera = "front"
    stamps: list = []

    def __init__(self, seconds: float = 8.0, odom_scale: float = 1.0) -> None:
        self._t = np.arange(0.0, seconds, 1.0 / 15.0) + 100.0
        self.gt_stamps = self._t
        self._odom_scale = odom_scale

    def _truth(self, stamp: float) -> conv.Pose2D:
        return conv.Pose2D(-1.5 + 0.3 * (stamp - 100.0), 0.2, 0.0)

    def gt_pose(self, stamp: float):
        return self._truth(stamp) if self._t[0] <= stamp <= self._t[-1] else None

    def scan_stamps(self) -> np.ndarray:
        return self._t

    def scan_before(self, stamp: float):
        i = int(np.searchsorted(self._t, stamp, side="right")) - 1
        if i < 0:
            return None, float("nan")
        taken = float(self._t[i])
        pose = self._truth(taken)
        angles = -math.pi + np.arange(360) * (2 * math.pi / 360)
        dx, dy = np.cos(angles + pose.theta), np.sin(angles + pose.theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (3.0 - pose.x) / dx, (-3.0 - pose.x) / dx)
            ty = np.where(dy > 0, (2.0 - pose.y) / dy, (-2.0 - pose.y) / dy)
        ranges = np.minimum(
            np.where(np.isfinite(tx), tx, np.inf), np.where(np.isfinite(ty), ty, np.inf)
        )
        return conv.LaserScan2D(ranges, -math.pi, 2 * math.pi / 360, 0.05, 25.0), taken

    def odometry(self):
        x = (self._t - 100.0) * 0.3 * self._odom_scale
        poses = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
        twists = np.column_stack(
            [
                np.full_like(x, 0.3 * self._odom_scale),
                np.zeros_like(x),
                np.zeros_like(x),
            ]
        )
        return self._t, poses, twists


def test_replay_sensors_hand_out_each_scan_once_and_the_latest_odometry():
    run = _RoomRun()
    with ReplayClock(run.scan_stamps()[0]) as clock:
        sensors = ReplaySensors(run, clock)
        clock.now = clock.of(100.0)
        first = sensors.get_scan()
        clock.now = clock.of(100.03)  # before the next scan
        assert sensors.get_scan() is first
        clock.now = clock.of(100.07)
        second = sensors.get_scan()
        assert second is not first
        assert sensors.scan_captured_at() == pytest.approx(clock.of(100.0 + 1.0 / 15.0))
        assert sensors.get_odom().pose.x == pytest.approx(0.3 / 15.0)
        clock.now = clock.of(105.0)
        assert sensors.get_scan(max_age_s=0.5) is not None
        clock.now = clock.of(200.0)  # long after the recording ends
        assert sensors.get_scan(max_age_s=0.5) is None


def test_replaying_a_run_maps_it_and_scores_the_pose_against_truth():
    # Wheel odometry over-reads distance by 10%; the scan match has to hold the pose.
    result = replay_mapping(_RoomRun(odom_scale=1.1), log=lambda *_: None)
    summary = result.summary()
    assert result.ticks > 100 and summary["travel_m"] > 1.5
    assert summary["live"]["ate_rmse_m"] < 0.08
    assert summary["final_map"]["ate_rmse_m"] < 0.08
    # Odometry alone would have ended 0.24 m long.
    assert summary["live"]["drift_max_m"] < 0.15
    grid = np.asarray(result.map["grid"])
    assert (grid >= 65).sum() > 100 and (grid == 0).sum() > 1000
    assert time.monotonic() < 1e9 and callable(time.monotonic)


# --- Pose tracking in a known map -----------------------------------------------

from src.nav.localize.global_localize import OccupancyMap  # noqa: E402
from tools.vision_eval.tracking_eval import (  # noqa: E402
    START_STD,
    evaluate_tracking,
    perturbed_starts,
    track,
)


def _room_map() -> OccupancyMap:
    """The 6 m x 4 m room ``_RoomRun`` drives in, as the dataset's map would draw it."""
    res = 0.05
    grid = np.full((120, 160), -1, dtype=np.int16)  # 8 m x 6 m around the room
    ox, oy = -4.0, -3.0
    xs = ox + (np.arange(160) + 0.5) * res
    ys = oy + (np.arange(120) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    inside = (np.abs(gx) < 3.0) & (np.abs(gy) < 2.0)
    grid[inside] = 0
    wall = inside & ((np.abs(gx) > 3.0 - res) | (np.abs(gy) > 2.0 - res))
    grid[wall] = 100
    return OccupancyMap(grid=grid, resolution=res, origin_x=ox, origin_y=oy)


def test_tracking_follows_the_robot_in_the_map_and_scores_without_alignment():
    # Odometry over-reads distance by 10%; the map has to hold the pose.
    run = track(_RoomRun(odom_scale=1.1), _room_map())
    summary = run.summary()
    assert summary["samples"] > 100
    assert summary["rmse_cm"] < 5.0 and summary["under_10cm_pct"] > 95.0
    assert summary["heading_rmse_deg"] < 2.0
    # Odometry alone would have ended 0.24 m long.
    assert np.hypot(*(run.est[-1, :2] - run.truth[-1, :2])) < 0.08
    # Scored against the truth at the time of the data, not re-fitted.
    pos, _yaw = run.errors()
    assert pos.size == run.stamps.size == run.truth.shape[0]


def test_a_start_off_by_the_benchmarks_spread_is_pulled_onto_the_map():
    room = _room_map()
    first = _RoomRun().gt_pose(100.0 + 1.0 / 15.0)
    start = conv.Pose2D(first.x + 0.15, first.y - 0.1, first.theta + math.radians(6.0))
    run = track(_RoomRun(), room, start)
    pos, yaw = run.errors()
    assert pos[0] > 0.1  # it did start off
    assert pos[-20:].max() < 0.05 and yaw[-20:].max() < 2.0


def test_drawn_starts_follow_the_benchmarks_spread_and_repeat_with_the_seed():
    first = conv.Pose2D(1.0, 2.0, 0.5)
    starts = perturbed_starts(first, 400, seed=3)
    dx = np.array([s.x - first.x for s in starts])
    dth = np.array([conv.normalize_angle(s.theta - first.theta) for s in starts])
    assert np.std(dx) == pytest.approx(START_STD[0], rel=0.15)
    assert np.std(dth) == pytest.approx(START_STD[2], rel=0.15)
    again = perturbed_starts(first, 400, seed=3)
    assert [(s.x, s.y, s.theta) for s in again] == [(s.x, s.y, s.theta) for s in starts]


def test_evaluation_reports_the_true_start_and_the_spread_over_drawn_starts():
    out = evaluate_tracking(_RoomRun(), _room_map(), starts=2, log=lambda *_: None)
    assert out["true_start"]["rmse_cm"] < 5.0
    drawn = out["drawn_starts"]
    assert len(drawn["runs"]) == 2 and "rmse_cm" in drawn["mean"]
    assert all(r["start_offset_m"] >= 0.0 for r in drawn["runs"])


def test_with_the_service_a_restart_a_meter_off_is_put_right():
    """The engine alone cannot pull a pose 1 m off back onto the walls.

    A robot restarting from a saved pose that far off is put right by the
    SLAM service's startup localize, which the replay runs on the
    recording's clock.
    """
    room = _room_map()
    first = _RoomRun(seconds=10.0).gt_pose(100.0 + 1.0 / 15.0)
    start = conv.Pose2D(first.x + 1.0, first.y, first.theta)

    alone = track(_RoomRun(seconds=10.0), room, start)
    pos, _ = alone.errors()
    assert pos[-15:].max() > 0.5 and alone.service_moves == []

    served = track(_RoomRun(seconds=10.0), room, start, service=True)
    pos, _ = served.errors()
    assert pos[-15:].max() < 0.05
    assert len(served.service_moves) == 1
    stamp, before, after = served.service_moves[0]
    assert np.hypot(before.x - after.x, before.y - after.y) > 0.8
    assert any("startup" in line for line in served.service_log)


def test_the_service_replay_repeats_exactly():
    room = _room_map()
    first = _RoomRun(seconds=10.0).gt_pose(100.0 + 1.0 / 15.0)
    start = conv.Pose2D(first.x, first.y + 0.8, first.theta)
    a = track(_RoomRun(seconds=10.0), room, start, service=True)
    b = track(_RoomRun(seconds=10.0), room, start, service=True)
    assert np.array_equal(a.est, b.est) and a.service_log == b.service_log
    # The service's loggers are put back afterwards.
    from src.models import slam as slam_module

    assert type(slam_module.LOGGER).__name__ != "_Recorder"


# --- The challenge's scoring and format --------------------------------------------

from tools.vision_eval.challenge import (  # noqa: E402
    load_tum,
    pose_scores,
    run_errors,
    write_tum,
)


def test_each_pose_earns_the_score_of_the_tightest_band_it_is_in():
    pos = np.array([0.01, 0.03, 0.03, 0.15, 0.5])
    yaw = np.array([0.5, 1.5, 4.5, 9.0, 0.0])
    # 2 cm/1 deg -> 100; 4 cm/2 deg -> 80; 3 cm but 4.5 deg -> 20 (10 cm/5 deg);
    # 15 cm/9 deg -> 10; 50 cm -> 0.
    assert list(pose_scores(pos, yaw)) == [100.0, 80.0, 20.0, 10.0, 0.0]


def test_a_submission_round_trips_with_one_pose_per_stamp(tmp_path):
    stamps = np.array([1.0, 1.0, 2.0, 3.0])  # two ticks shared an odometry sample
    poses = np.array(
        [[0.0, 0.0, 0.1], [0.5, 0.0, 0.2], [1.0, 0.0, -3.0], [2.0, 1.0, 3.0]]
    )
    write_tum(tmp_path / "run_1.txt", stamps, poses)
    t, p = load_tum(tmp_path / "run_1.txt")
    assert list(t) == [1.0, 2.0, 3.0]
    assert p[0] == pytest.approx([0.5, 0.0, 0.2])  # the later estimate is kept
    assert p[2, 2] == pytest.approx(3.0)


def test_run_errors_hold_the_localizer_before_its_first_pose_and_score_from_the_start():
    gt_t = np.arange(0.0, 10.0, 0.5)
    gt_p = np.column_stack([gt_t, np.zeros_like(gt_t), np.zeros_like(gt_t)])
    loc_t = np.array([4.0, 9.5])
    loc_p = np.array([[4.0, 0.0, 0.0], [9.5, 0.0, 0.0]])
    pos, yaw = run_errors(gt_t, gt_p, 2.0, loc_t, loc_p)
    assert pos.size == (gt_t >= 2.0).sum()
    # Before its first pose the localizer is held there: 2 m off at t = 2.
    assert pos[0] == pytest.approx(2.0) and pos[-1] == pytest.approx(0.0)
    assert yaw.max() == pytest.approx(0.0)


def test_the_lost_filter_does_not_commit_in_a_room_that_looks_the_same_both_ways():
    """A bare rectangle looks identical turned half a turn: two equal clusters.

    The filter must keep both rather than hand one to the tracker.
    """
    room = _room_map()
    run = track(
        _RoomRun(seconds=10.0), room, service=False, global_start=True, lost_filter=True
    )
    assert not [line for line in run.service_log if "handed over" in line]
