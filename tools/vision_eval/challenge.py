"""The LILocBench challenge's scoring, and its submission format.

The challenge (its Codabench page and evaluation tool) runs a localizer
several times per run, each time from a later start (0, 120 and 240 s into
the public runs) with no initial pose, until the end of the run. Every ground
truth pose from that start on is compared with the localizer's pose at the
same moment (interpolated; held before its first and after its last pose),
with no alignment. A pose earns the score of the tightest band it falls in:
100 within 2 cm and 1°, then 80, 60, 40, 20 and 10 out to 20 cm and 10°,
else 0. Scores are averaged over poses, weighted by pose count, per run,
per sequence, per group (static, dynamics, ...) and overall.

Written from the challenge's description and evaluation tool's behaviour.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# (meters, degrees, score), tightest first, as the public configuration sets them.
BANDS: Tuple[Tuple[float, float, float], ...] = (
    (0.02, 1.0, 100.0),
    (0.04, 2.0, 80.0),
    (0.06, 3.0, 60.0),
    (0.08, 4.0, 40.0),
    (0.10, 5.0, 20.0),
    (0.20, 10.0, 10.0),
)
START_TIMES = {
    "static_0": (0.0, 120.0, 240.0),
    "dynamics_0": (0.0, 30.0, 60.0),
    "lt_changes_0": (0.0, 120.0, 240.0),
    "lt_changes_dynamics_0": (0.0, 120.0, 240.0),
}


def load_tum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Stamps and ``(N, 3)`` x, y, yaw from a TUM pose file."""
    a = np.loadtxt(path, ndmin=2)
    qx, qy, qz, qw = a[:, 4], a[:, 5], a[:, 6], a[:, 7]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return a[:, 0], np.column_stack([a[:, 1], a[:, 2], yaw])


def write_tum(path: Path, stamps: np.ndarray, poses: np.ndarray) -> None:
    """One pose per stamp (the latest written for it), stamps increasing."""
    stamps, poses = np.asarray(stamps, dtype=float), np.asarray(poses, dtype=float)
    # Two ticks can share an odometry sample; keep the later estimate.
    _, last = np.unique(stamps[::-1], return_index=True)
    keep = np.sort(len(stamps) - 1 - last)
    stamps, poses = stamps[keep], poses[keep]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    half = poses[:, 2] / 2.0
    rows = np.column_stack(
        [
            stamps,
            poses[:, 0],
            poses[:, 1],
            np.zeros(len(stamps)),
            np.zeros(len(stamps)),
            np.zeros(len(stamps)),
            np.sin(half),
            np.cos(half),
        ]
    )
    np.savetxt(path, rows, fmt="%.9f")


def _at(times: np.ndarray, stamps: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Poses at ``times``: linear in position and heading, held at both ends."""
    yaw = np.unwrap(poses[:, 2])
    return np.column_stack(
        [
            np.interp(times, stamps, poses[:, 0]),
            np.interp(times, stamps, poses[:, 1]),
            np.interp(times, stamps, yaw),
        ]
    )


def run_errors(
    gt_stamps: np.ndarray,
    gt_poses: np.ndarray,
    start_s: float,
    stamps: np.ndarray,
    poses: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Position (m) and heading (deg) error at every ground-truth pose from the start."""
    keep = gt_stamps >= gt_stamps[0] + start_s
    t, truth = gt_stamps[keep], gt_poses[keep]
    est = _at(t, stamps, poses)
    pos = np.hypot(est[:, 0] - truth[:, 0], est[:, 1] - truth[:, 1])
    d = est[:, 2] - truth[:, 2]
    yaw = np.abs(np.degrees(np.arctan2(np.sin(d), np.cos(d))))
    return pos, yaw


def pose_scores(pos: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Each pose's score: that of the tightest band it falls in, else 0."""
    out = np.zeros(pos.shape)
    for meters, degrees, score in reversed(BANDS):
        out[(pos <= meters) & (yaw <= degrees)] = score
    return out


def score_submission(submission: Path, gt_dir: Path) -> Dict:
    """Challenge scores for a submission folder (``<sequence>/run_<k>.txt``)."""
    out: Dict = {"sequences": {}, "groups": {}}
    group_tot: Dict[str, List[float]] = {}
    for seq, starts in START_TIMES.items():
        gt_file = Path(gt_dir) / seq / "gt_poses.txt"
        if not gt_file.is_file():
            continue
        gt_t, gt_p = load_tum(gt_file)
        runs = {}
        total = size = 0.0
        for k, start in enumerate(starts, 1):
            f = Path(submission) / seq / f"run_{k}.txt"
            if not f.is_file():
                continue
            t, p = load_tum(f)
            pos, yaw = run_errors(gt_t, gt_p, start, t, p)
            s = pose_scores(pos, yaw)
            runs[f"run_{k}"] = {
                "score": float(s.mean()),
                "size": int(s.size),
                "rmse_cm": float(100.0 * math.sqrt(float(np.mean(pos**2)))),
                "median_cm": float(100.0 * np.median(pos)),
                "under_10cm_pct": float(100.0 * np.mean(pos < 0.10)),
            }
            total += float(s.sum())
            size += s.size
        if not size:
            continue
        out["sequences"][seq] = {"score": total / size, "size": int(size), "runs": runs}
        group = seq.rsplit("_", 1)[0]
        acc = group_tot.setdefault(group, [0.0, 0.0])
        acc[0] += total
        acc[1] += size
    for group, (total, size) in group_tot.items():
        out["groups"][group] = total / size
    all_total = sum(v[0] for v in group_tot.values())
    all_size = sum(v[1] for v in group_tot.values())
    out["overall"] = all_total / all_size if all_size else float("nan")
    return out


def describe(scores: Dict) -> str:
    lines = [f"overall {scores['overall']:.1f}"]
    for seq, s in scores["sequences"].items():
        runs = ", ".join(
            f"{name} {r['score']:.1f} (rmse {r['rmse_cm']:.0f} cm, median {r['median_cm']:.1f} cm)"
            for name, r in s["runs"].items()
        )
        lines.append(f"  {seq}: {s['score']:.1f}  [{runs}]")
    return "\n".join(lines)


def groups_line(
    scores: Dict,
    order: Sequence[str] = ("static", "dynamics", "lt_changes", "lt_changes_dynamics"),
) -> str:
    return " | ".join(f"{g} {scores['groups'].get(g, float('nan')):.1f}" for g in order)
