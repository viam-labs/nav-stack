#!/usr/bin/env python3
"""Measure the vision side-channel on a recorded dataset.

    python tools/fetch_lilocbench.py            # once: ~2 GB subset
    python tools/eval_vision.py place           # place-recognition recall
    python tools/eval_vision.py reloc           # lost-pose recovery, lidar vs camera+lidar
    python tools/eval_vision.py mapping         # replay a run through SLAM, score the trajectory

``place`` builds a place index from the mapping run, then asks for each frame
of a later run whether a keyframe near the true pose is among the top guesses.
``reloc`` treats each sampled frame as a kidnapped robot and compares the
stack's whole-map lidar search with a lidar search seeded by place guesses.

``mapping`` feeds a run's odometry and scans through the SLAM engine in
mapping mode, on a clock that follows the recording, and compares the pose it
publishes and the keyframes it ends with against ground truth: with lidar loop
closure off and on, and with the vision supervisor (person mask) alongside.

Results are printed and written as JSON next to the dataset cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.vision_eval.features import ensure_features  # noqa: E402
from tools.vision_eval.lilocbench import LilocSequence, load_reference_map  # noqa: E402
from tools.vision_eval.place_eval import evaluate_place  # noqa: E402
from tools.vision_eval.reloc_eval import evaluate_reloc  # noqa: E402
from tools.vision_eval.replay import odometry_trajectory, replay_mapping  # noqa: E402

DEFAULT_DATA = Path.home() / "datasets" / "lilocbench"
DEFAULT_QUERIES = ("dynamics_0", "lt_changes_0")


def _pct(value: Optional[float]) -> str:
    return "   - " if value is None else f"{100.0 * value:4.0f}%"


def _print_place(name: str, result: dict) -> None:
    first = next(iter(result.values()))
    print(
        f"\n{name}: {first['queries']} query frames against "
        f"{first['keyframes']} mapping keyframes"
    )
    print(
        "  a guess counts when it is within 1.5 m of the true pose "
        "(and, in the right-hand block, facing within 45°)"
    )
    print(f"  {'method':<18} {'any heading':<26} {'same heading':<26}")
    print(
        f"  {'':<18} {'top1  top3  top5  ceiling':<26} {'top1  top3  top5  ceiling':<26}"
    )
    for method, row in result.items():
        a, s = row["any_heading"], row["same_heading"]
        print(
            f"  {method:<18} "
            f"{_pct(a['top1'])} {_pct(a['top3'])} {_pct(a['top5'])} {_pct(a['possible'])}    "
            f"{_pct(s['top1'])} {_pct(s['top3'])} {_pct(s['top5'])} {_pct(s['possible'])}"
        )


def _print_reloc(name: str, result: dict) -> None:
    first = next(iter(result.values()))
    print(
        f"\n{name}: {first['queries']} kidnapped frames (right = within 0.5 m and 15°)"
    )
    print(
        f"  {'method':<24} {'right':>6} {'accepted+right':>15} {'accepted+wrong':>15} {'refused':>8}"
    )
    for method, row in result.items():
        print(
            f"  {method:<24} {_pct(row['right']):>6} {_pct(row['accepted_right']):>15} "
            f"{_pct(row['accepted_wrong']):>15} {_pct(row['refused']):>8}"
        )


MAPPING_VARIANTS = {
    "odometry": "wheel odometry alone",
    "lidar_no_loops": "SLAM, lidar loop closure off",
    "lidar": "SLAM as shipped without a camera",
    "vision": "SLAM + vision (person mask), result applied at once",
    "vision_late": "SLAM + vision, result 0.3 s after the frame",
}


def _run_mapping(seq, feats, cache, variants: Sequence[str]) -> dict:
    out: dict = {}
    for name in variants:
        if name == "odometry":
            out[name] = {"live": odometry_trajectory(seq).summary()}
            continue
        kwargs: dict = {}
        if name == "lidar_no_loops":
            kwargs["config"] = {"builtin_loop_closure": False}
        if name.startswith("vision"):
            kwargs.update(feats=feats, cache_dir=cache, vision=True)
        if name == "vision_late":
            # A camera result this late misses the 0.2 s mask window.
            kwargs["vision_latency_s"] = 0.3
        out[name] = replay_mapping(seq, log=lambda *_: None, **kwargs).summary()
        print(f"  {seq.name}: {name} done in {out[name]['wall_s']:.0f} s", flush=True)
    return out


def _print_mapping(name: str, result: dict) -> None:
    travel = next((r["travel_m"] for r in result.values() if "travel_m" in r), None)
    length = f", {travel} m of travel" if travel is not None else ""
    print(f"\n{name}: mapped from scratch{length}. Errors against ground truth.")
    print(
        f"  {'variant':<28} {'live pose':<26} {'final keyframes':<18} "
        f"{'lidar loops':>11} {'frames masked':>14}"
    )
    print(f"  {'':<28} {'rmse    max   heading':<26} {'rmse    max':<18}")
    for variant, row in result.items():
        live, final = row["live"], row.get("final_map")
        final_txt = (
            f"{final['ate_rmse_m']:.2f} m  {final['ate_max_m']:.2f} m" if final else "-"
        )
        masked = (row.get("vision_trace") or {}).get("frames_masked_in_time", "-")
        print(
            f"  {variant:<28} "
            f"{live['ate_rmse_m']:.2f} m  {live['ate_max_m']:.2f} m  {live['heading_rmse_deg']:4.1f}°    "
            f"{final_txt:<18} "
            f"{row.get('loop_closures', '-')!s:>11} {masked!s:>14}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment", choices=["place", "reloc", "mapping"])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--mapping", default="mapping")
    ap.add_argument("--queries", nargs="+", default=list(DEFAULT_QUERIES))
    ap.add_argument("--camera", default="front")
    ap.add_argument(
        "--stride", type=int, default=None, help="use every Nth query frame"
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--place-methods",
        nargs="+",
        default=["runtime"],
        help="reloc only: 'runtime' is the shipped recovery path; other names "
        "seed a plain lidar search with that place method (append +patches "
        "to re-rank by matched patches, +fit to refine each guess by camera)",
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        default=["mapping"],
        help="mapping only: runs to map from scratch",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=list(MAPPING_VARIANTS),
        choices=list(MAPPING_VARIANTS),
        help="mapping only: "
        + "; ".join(f"{k} = {v}" for k, v in MAPPING_VARIANTS.items()),
    )
    args = ap.parse_args(argv)

    cache = args.data / "cache"
    if args.experiment == "mapping":
        report = {}
        t0 = time.monotonic()
        for name in args.runs:
            seq = LilocSequence(args.data, name, camera=args.camera)
            needs_vision = any(v.startswith("vision") for v in args.variants)
            feats = ensure_features(seq, cache) if needs_vision else None
            report[name] = _run_mapping(seq, feats, cache, args.variants)
            _print_mapping(name, report[name])
        out = cache / "results_mapping.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\n{time.monotonic() - t0:.0f} s; results written to {out}")
        return 0
    mapping = LilocSequence(args.data, args.mapping, camera=args.camera)
    map_feats = ensure_features(mapping, cache)
    occ = load_reference_map(args.data) if args.experiment == "reloc" else None
    report = {}
    t0 = time.monotonic()
    for name in args.queries:
        query = LilocSequence(args.data, name, camera=args.camera)
        feats = ensure_features(query, cache)
        if args.experiment == "place":
            result = evaluate_place(
                mapping, map_feats, query, feats, query_stride=args.stride or 1
            )
            _print_place(name, result)
        else:
            result = evaluate_reloc(
                occ,
                mapping,
                map_feats,
                query,
                feats,
                list(args.place_methods),
                query_stride=args.stride or 4,
                top_k=args.top_k,
            )
            _print_reloc(name, result)
        report[name] = result
    out = cache / f"results_{args.experiment}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n{time.monotonic() - t0:.0f} s; results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
