"""Offline vision evaluation: dataset reading and scoring rules."""

from __future__ import annotations

import math
import zipfile

import numpy as np

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
    keyframes = [conv.Pose2D(0.0, 0.0, 0.0), conv.Pose2D(5.0, 0.0, 0.0), conv.Pose2D(0.5, 0.0, math.pi)]
    queries = [conv.Pose2D(0.2, 0.0, 0.0), conv.Pose2D(20.0, 0.0, 0.0)]
    dist, dyaw = pose_errors(keyframes, queries)
    # Query 0 ranks the far keyframe first, then the reversed one, then the right one.
    ranking = PlaceRanking("t", np.array([[1, 2, 0], [0, 1, 2]]), np.zeros((2, 3)))
    any_heading = recall_at_k(ranking, dist, dyaw, ks=(1, 2, 3), max_dist_m=1.5)
    assert any_heading == {"possible": 0.5, "top1": 0.0, "top2": 0.5, "top3": 0.5}
    same = recall_at_k(ranking, dist, dyaw, ks=(1, 2, 3), max_dist_m=1.5, max_yaw_deg=45.0)
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
