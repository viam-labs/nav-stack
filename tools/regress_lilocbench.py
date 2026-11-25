"""Regression check on LILocBench: mapping and localization where nav-stack stands out.

    python tools/regress_lilocbench.py              # run, compare with the baselines, exit 1 on a regression
    python tools/regress_lilocbench.py --update     # run and store the results as the new baselines
    python tools/regress_lilocbench.py --full       # also front-lidar tracking and both-lidar challenge

What it runs (each a separate tools/eval_vision.py process, ``--jobs`` at a time):

- **mapping** from scratch on ``mapping``, ``static_0`` and ``dynamics_0``
  (people walking, where slam_toolbox fed both lidars broke): live pose and
  final keyframe error.
- **tracking** (the LILocBench paper's protocol, true start, the SLAM
  service as on a robot) on ``static_0``, ``dynamics_0`` (people walking),
  ``lt_changes_0`` (furniture moved) and ``lt_changes_dynamics_0`` (both):
  position error.
- **challenge** (no initial pose, banded score) on the same four runs, with
  the service's lost filter: score per run and overall.

A replay is deterministic on one machine, so any change in a number is the
code's. A result counts as worse when an error grows by more than 0.5 cm and
5%, a run's challenge score drops by more than 2 points, or the overall
score by more than 1. Heading, shares within 5 cm and seconds taken are shown
but not judged (time depends on the machine and its load).

The baselines live next to the dataset (``<data>/regression_baselines.json``),
not in the repo: the dataset is CC BY-NC-SA and stays out of it. Fetch it
with tools/fetch_lilocbench.py (``lt_changes_dynamics_0`` too).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.vision_eval.challenge import score_submission  # noqa: E402

DEFAULT_DATA = Path.home() / "datasets" / "lilocbench"
MAPPING_RUNS = ("mapping", "static_0", "dynamics_0")
LOCALIZE_RUNS = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")

# (worse by more than this many cm, and by more than this share)
ERROR_SLACK = (0.5, 0.05)
SCORE_SLACK = 2.0
OVERALL_SLACK = 1.0


class Job:
    def __init__(self, key: str, args: List[str], runs: Tuple[str, ...]) -> None:
        self.key = key
        self.args = args
        self.runs = runs
        self.data: Optional[Path] = None
        self.wall_s = 0.0
        self.error = ""


def _jobs(full: bool) -> List[Job]:
    jobs = [
        Job(
            f"mapping {run}",
            ["mapping", "--runs", run, "--variants", "lidar"],
            ("map_office", run),
        )
        for run in MAPPING_RUNS
    ]
    for lidars in ("both", "front") if full else ("both",):
        jobs += [
            Job(
                f"tracking {run} {lidars}",
                [
                    "tracking",
                    "--queries",
                    run,
                    "--lidars",
                    lidars,
                    "--localizer",
                    "service",
                    "--starts",
                    "0",
                    "--lost-filter",
                ],
                ("map_office", run),
            )
            for run in LOCALIZE_RUNS
        ]
    for lidars in ("front", "both") if full else ("front",):
        jobs += [
            Job(
                f"challenge {run} {lidars}",
                [
                    "challenge",
                    "--queries",
                    run,
                    "--lidars",
                    lidars,
                    "--localizer",
                    "service",
                    "--lost-filter",
                ],
                ("map_office", run),
            )
            for run in LOCALIZE_RUNS
        ]
    return jobs


def _run(job: Job, data: Path, work: Path) -> Job:
    job.data = work / job.key.replace(" ", "_")
    (job.data / "cache").mkdir(parents=True, exist_ok=True)
    for name in job.runs:
        (job.data / name).symlink_to(data / name)
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "tools/eval_vision.py", *job.args, "--data", str(job.data)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    job.wall_s = time.monotonic() - t0
    (job.data / "log.txt").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        job.error = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["failed"]
        job.error = job.error[0]
    return job


def _results(jobs: List[Job], data: Path, work: Path) -> Dict[str, float]:
    """Flat ``metric -> value`` from every job's results."""
    out: Dict[str, float] = {}
    challenge: Dict[str, List[Path]] = {}
    for job in jobs:
        kind, run, *rest = job.key.split()
        cache = job.data / "cache"
        if kind == "mapping":
            r = json.loads((cache / "results_mapping.json").read_text())[run]["lidar"]
            out[f"mapping/{run}/live_cm"] = 100.0 * r["live"]["ate_rmse_m"]
            out[f"mapping/{run}/final_cm"] = 100.0 * r["final_map"]["ate_rmse_m"]
            out[f"mapping/{run}/heading_deg"] = r["live"]["heading_rmse_deg"]
            out[f"mapping/{run}/wall_s"] = float(r["wall_s"])
        elif kind == "tracking":
            lidars = rest[0]
            r = json.loads((cache / "results_tracking.json").read_text())[run]
            t = r[f"{lidars} service"]["true_start"]
            out[f"tracking/{run}/{lidars}/rmse_cm"] = t["rmse_cm"]
            out[f"tracking/{run}/{lidars}/under_5cm_pct"] = t["under_5cm_pct"]
        else:
            lidars = rest[0]
            folder = cache / "challenge" / f"{lidars}_service_lost_filter"
            s = score_submission(folder, data)
            out[f"challenge/{run}/{lidars}/score"] = s["sequences"][run]["score"]
            challenge.setdefault(lidars, []).append(folder / run)
    for lidars, folders in challenge.items():
        merged = work / f"challenge_{lidars}"
        for f in folders:
            shutil.copytree(f, merged / f.name, dirs_exist_ok=True)
        out[f"challenge/overall/{lidars}/score"] = score_submission(merged, data)[
            "overall"
        ]
    return out


def _verdict(metric: str, base: float, now: float) -> str:
    if metric.endswith(("heading_deg", "under_5cm_pct", "wall_s")):
        return ""
    if not (math.isfinite(base) and math.isfinite(now)):
        return "WORSE" if math.isfinite(base) else ""
    if metric.endswith("_cm"):
        cm, share = ERROR_SLACK
        if now > base + cm and now > base * (1.0 + share):
            return "WORSE"
        return "better" if now < base - cm else "ok"
    slack = OVERALL_SLACK if "/overall/" in metric else SCORE_SLACK
    if now < base - slack:
        return "WORSE"
    return "better" if now > base + slack else "ok"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--full", action="store_true")
    ap.add_argument(
        "--update", action="store_true", help="store these results as the baselines"
    )
    ap.add_argument("--keep", action="store_true", help="keep the work folder")
    args = ap.parse_args(argv)
    data = args.data.expanduser().resolve()
    missing = [
        r
        for r in ("map_office", *MAPPING_RUNS, *LOCALIZE_RUNS)
        if not (data / r).exists()
    ]
    if missing:
        print(
            f"{data} lacks {', '.join(missing)}; fetch with tools/fetch_lilocbench.py"
        )
        return 2
    baselines_path = data / "regression_baselines.json"
    baselines = (
        json.loads(baselines_path.read_text()) if baselines_path.is_file() else {}
    )

    work = Path(tempfile.mkdtemp(prefix="nav-regress-"))
    jobs = _jobs(args.full)
    t0 = time.monotonic()
    print(f"{len(jobs)} runs, {args.jobs} at a time ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for job in pool.map(lambda j: _run(j, data, work), jobs):
            state = f"failed: {job.error}" if job.error else f"{job.wall_s:.0f} s"
            print(f"  {job.key}: {state}", flush=True)
    failed = [j for j in jobs if j.error]
    if failed:
        print(f"{len(failed)} runs failed; logs in {work}")
        return 2
    now = _results(jobs, data, work)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()

    base = baselines.get("metrics", {})
    worse = 0
    print(f"\n{'metric':50s} {'baseline':>9s} {'now':>9s} {'change':>8s}")
    for metric in sorted(now):
        b = base.get(metric, math.nan)
        v = now[metric]
        verdict = _verdict(metric, b, v) if metric in base else "new"
        worse += verdict == "WORSE"
        delta = f"{v - b:+8.2f}" if math.isfinite(b) else ""
        print(f"{metric:50s} {b:9.2f} {v:9.2f} {delta:>8s}  {verdict}")
    print(
        f"\n{time.monotonic() - t0:.0f} s; baselines from {baselines.get('commit', 'nowhere yet')}"
    )
    if args.update:
        baselines = {
            "commit": commit,
            "date": time.strftime("%Y-%m-%d"),
            "full": bool(args.full),
            "metrics": {**base, **now},
        }
        baselines_path.write_text(json.dumps(baselines, indent=1, sort_keys=True))
        print(f"baselines stored in {baselines_path} (commit {commit})")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    if worse and not args.update:
        print(f"{worse} results got worse")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
