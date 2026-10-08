#!/usr/bin/env python3
"""Poll nav + SLAM status on a live machine during a route and log it.

Captures the fields that reset between goals (replan reasons) and the
localization quality signals (match score, ray error, correction counters)
at ~1 Hz, so a stuck episode can be reconstructed afterwards.

Usage (API key from the app's Connect tab, or ``viam machines api-key create``):

    export VIAM_ADDRESS=tracer2a-main.xxxx.viam.cloud
    export VIAM_API_KEY_ID=...
    export VIAM_API_KEY=...
    python scripts/monitor_nav.py --nav navigation --slam slam \
        --out .local/monitor/route-$(date +%Y%m%d-%H%M%S).jsonl

Stop with Ctrl-C; a summary is printed and appended to ``<out>.summary.txt``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    from viam.robot.client import RobotClient
    from viam.services.motion import MotionClient
    from viam.services.slam import SLAMClient
except ImportError:  # pragma: no cover
    print("viam-sdk not installed; run inside the project venv", file=sys.stderr)
    raise


def _get(d: Any, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "-"
        return f"{v:.{nd}f}"
    return str(v)


class Monitor:
    def __init__(self, nav: MotionClient, slam: SLAMClient, out: Path, period_s: float):
        self.nav = nav
        self.slam = slam
        self.out = out
        self.period_s = period_s
        self.samples: list[dict] = []
        self._stop = False
        self._prev_slam: Optional[dict] = None
        self._last_replan_sig: Optional[str] = None
        self._goal_start: Optional[float] = None
        self._goals: list[dict] = []

    def stop(self):
        self._stop = True

    async def sample(self) -> dict:
        t0 = time.time()
        nav_task = self.nav.do_command({"command": "get_status"})
        slam_task = self.slam.do_command({"command": "get_status"})
        nav_res, slam_res = await asyncio.gather(nav_task, slam_task, return_exceptions=True)
        row: dict[str, Any] = {"t": t0}
        if isinstance(nav_res, Exception):
            row["nav_error"] = str(nav_res)
            nav_res = {}
        if isinstance(slam_res, Exception):
            row["slam_error"] = str(slam_res)
            slam_res = {}
        nav = dict(nav_res) if isinstance(nav_res, dict) else {}
        slam = dict(slam_res) if isinstance(slam_res, dict) else {}
        prog = nav.get("progress") or {}
        pose = nav.get("pose") or {}

        row["nav"] = {
            "state": nav.get("state"),
            "active": nav.get("active"),
            "x": pose.get("x"),
            "y": pose.get("y"),
            "theta": pose.get("theta"),
            "dist_remaining_m": prog.get("distance_remaining_m"),
            "obstacle": prog.get("obstacle"),
            "local_blocked": prog.get("local_blocked"),
            "nose_clear": prog.get("nose_clear"),
            "local_planner": prog.get("local_planner"),
            "path_cost_ahead": prog.get("path_cost_ahead"),
            "spin_blocked": prog.get("spin_blocked"),
            "forward_clearance_m": prog.get("forward_clearance_m"),
            "cmd_vx": prog.get("cmd_vx_mps"),
            "cmd_w": prog.get("cmd_vtheta_rad_s"),
            "failed_replan_while_blocked": prog.get("failed_replan_while_blocked"),
            "local_replan_cooldown_s": prog.get("local_replan_cooldown_s"),
            "last_replan_trigger": prog.get("last_replan_trigger"),
            "last_replan_error": prog.get("last_replan_error"),
            "last_replan_info": prog.get("last_replan_info"),
            "localization_hold": prog.get("localization_hold"),
            "stuck_pose_refine": prog.get("stuck_pose_refine"),
            # New in fix/localization-yield-deadlock: goal-blocked finish and
            # localization yield (absent on older module builds).
            "goal_blocked": nav.get("goal_blocked"),
            "goal_offset_m": nav.get("goal_offset_m"),
            "localization_yield": nav.get("localization_yield"),
            "loc_yield": prog.get("loc_yield"),
            "error_msg": nav.get("error_msg"),
            "length_m": nav.get("length_m"),
            "drive": nav.get("drive"),
            "control_loop": nav.get("control_loop"),
        }
        lc = slam.get("localization_check") or {}
        row["slam"] = {
            "ticks": slam.get("ticks"),
            "match_accepts": slam.get("match_accepts"),
            "match_rejects": slam.get("match_rejects"),
            "nav_track_applies": slam.get("nav_track_applies"),
            "last_match_score": slam.get("last_match_score"),
            "last_prior_score": slam.get("last_prior_score"),
            "last_scan_age_s": slam.get("last_scan_age_s"),
            "yaw_rate_deg_s": slam.get("yaw_rate_deg_s"),
            "lc_status": lc.get("status"),
            "lc_score": lc.get("score"),
            "lc_ray_mae_m": lc.get("ray_mae_m"),
            "lc_tick_match_score": lc.get("tick_match_score"),
            "lc_shift_m": lc.get("shift_m"),
            "lc_corrected": lc.get("corrected"),
            "odom_gap_hold_events": _get(slam, "odom", "gap_hold_events"),
            "odom_last_gap_s": _get(slam, "odom", "last_sample_gap_s"),
            "pose": slam.get("pose"),
        }
        # Deltas since previous sample (rates while moving vs still).
        prev = self._prev_slam
        if prev:
            dt = max(1e-6, t0 - prev["t"])
            for k in ("ticks", "match_accepts", "match_rejects", "nav_track_applies"):
                a, b = prev.get(k), row["slam"].get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    row["slam"][f"d_{k}"] = b - a
                    row["slam"][f"rate_{k}_hz"] = round((b - a) / dt, 2)
        self._prev_slam = {"t": t0, **{k: row["slam"].get(k) for k in ("ticks", "match_accepts", "match_rejects", "nav_track_applies")}}
        return row

    def _track_goals(self, row: dict) -> None:
        nav = row["nav"]
        active = bool(nav.get("active"))
        if active and self._goal_start is None:
            self._goal_start = row["t"]
            self._goals.append({"start": row["t"], "length_m": nav.get("length_m"), "replans": [], "states": {}})
        if self._goal_start is not None:
            g = self._goals[-1]
            obs = str(nav.get("obstacle") or "")
            g["states"][obs] = g["states"].get(obs, 0) + 1
            sig = json.dumps(nav.get("last_replan_info"), sort_keys=True) if nav.get("last_replan_info") else None
            if sig and sig != self._last_replan_sig:
                self._last_replan_sig = sig
                g["replans"].append({"t": row["t"], "info": nav.get("last_replan_info"), "error": nav.get("last_replan_error")})
            if not active:
                g["end"] = row["t"]
                g["final_state"] = nav.get("state")
                g["error_msg"] = nav.get("error_msg")
                self._goal_start = None
                self._last_replan_sig = None

    def _print_line(self, row: dict) -> None:
        n, s = row["nav"], row["slam"]
        ts = time.strftime("%H:%M:%S", time.localtime(row["t"]))
        rp = ""
        if n.get("last_replan_error"):
            rp = f" REPLAN_ERR={n['last_replan_error'][:80]!r}"
        elif n.get("last_replan_trigger"):
            rp = f" replan={n['last_replan_trigger'][:40]!r}"
        print(
            f"{ts} nav={n.get('state')}{'*' if n.get('active') else ' '} "
            f"pos=({_fmt(n.get('x'))},{_fmt(n.get('y'))},{_fmt(n.get('theta'))}) "
            f"rem={_fmt(n.get('dist_remaining_m'))}m obs={n.get('obstacle')} "
            f"pc={n.get('path_cost_ahead')} lb={int(bool(n.get('local_blocked')))} "
            f"nose={int(bool(n.get('nose_clear')))} v={_fmt(n.get('cmd_vx'))} w={_fmt(n.get('cmd_w'))} | "
            f"slam score={_fmt(s.get('last_match_score'))}/{_fmt(s.get('last_prior_score'))} "
            f"tick={_fmt(s.get('lc_tick_match_score'))} lc={s.get('lc_status')} "
            f"lcs={_fmt(s.get('lc_score'))} mae={_fmt(s.get('lc_ray_mae_m'))} "
            f"acc+{s.get('d_match_accepts', '-')} rej+{s.get('d_match_rejects', '-')} "
            f"trk+{s.get('d_nav_track_applies', '-')}{rp}",
            flush=True,
        )

    def summary(self) -> str:
        if not self.samples:
            return "no samples"
        lines = [f"samples: {len(self.samples)} over {self.samples[-1]['t'] - self.samples[0]['t']:.0f}s"]
        moving = [r for r in self.samples if r["nav"].get("active")]
        still = [r for r in self.samples if not r["nav"].get("active")]

        def agg(rows, key, sub="slam"):
            vals = [r[sub].get(key) for r in rows]
            vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
            if not vals:
                return "-"
            vals.sort()
            return f"min={vals[0]:.2f} p50={vals[len(vals)//2]:.2f} max={vals[-1]:.2f} n={len(vals)}"

        for label, rows in (("while navigating", moving), ("while idle", still)):
            if not rows:
                continue
            lines.append(f"[{label}] samples={len(rows)}")
            lines.append(f"  last_match_score   {agg(rows, 'last_match_score')}")
            lines.append(f"  last_prior_score   {agg(rows, 'last_prior_score')}")
            lines.append(f"  tick_match_score   {agg(rows, 'lc_tick_match_score')}")
            lines.append(f"  loc_check score    {agg(rows, 'lc_score')}")
            lines.append(f"  loc_check ray_mae  {agg(rows, 'lc_ray_mae_m')}")
            acc = sum(r["slam"].get("d_match_accepts") or 0 for r in rows)
            rej = sum(r["slam"].get("d_match_rejects") or 0 for r in rows)
            trk = sum(r["slam"].get("d_nav_track_applies") or 0 for r in rows)
            tot = acc + rej
            lines.append(f"  matches: {tot} (accepted {acc}, not applied {rej}, nav-track nudges {trk})")
            if label == "while navigating":
                obs: dict[str, int] = {}
                for r in rows:
                    o = str(r["nav"].get("obstacle") or "")
                    obs[o] = obs.get(o, 0) + 1
                lines.append(f"  obstacle states: {obs}")
        for i, g in enumerate(self._goals, 1):
            dur = (g.get("end") or self.samples[-1]["t"]) - g["start"]
            lines.append(
                f"goal {i}: {dur:.0f}s len={_fmt(g.get('length_m'))}m final={g.get('final_state')} "
                f"err={g.get('error_msg')!r} replans={len(g['replans'])} states={g['states']}"
            )
            for rp in g["replans"]:
                lines.append(f"    replan @{time.strftime('%H:%M:%S', time.localtime(rp['t']))}: {json.dumps(rp['info'])[:400]}")
        return "\n".join(lines)

    async def run(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open("a") as fh:
            while not self._stop:
                t0 = time.monotonic()
                try:
                    row = await self.sample()
                except Exception as exc:  # noqa: BLE001 - keep sampling
                    row = {"t": time.time(), "error": str(exc), "nav": {}, "slam": {}}
                self.samples.append(row)
                self._track_goals(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                self._print_line(row) if "error" not in row else print(f"sample error: {row['error']}")
                await asyncio.sleep(max(0.0, self.period_s - (time.monotonic() - t0)))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default=os.environ.get("VIAM_ADDRESS"))
    ap.add_argument("--api-key-id", default=os.environ.get("VIAM_API_KEY_ID"))
    ap.add_argument("--api-key", default=os.environ.get("VIAM_API_KEY"))
    ap.add_argument("--nav", default="navigation")
    ap.add_argument("--slam", default="slam")
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--out", default=f".local/monitor/route-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    args = ap.parse_args()
    if not (args.address and args.api_key_id and args.api_key):
        ap.error("need --address/--api-key-id/--api-key or VIAM_ADDRESS/VIAM_API_KEY_ID/VIAM_API_KEY")

    opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    robot = await RobotClient.at_address(args.address, opts)

    def _client(ctor, name):
        # ``from_robot`` needs the name in ``resource_names``; modular services
        # mid-reconfigure can be missing there. The DoCommand channel works
        # regardless, so fall back to constructing on the channel directly.
        try:
            return ctor.from_robot(robot, name)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: from_robot failed ({type(exc).__name__}); using direct channel client")
            return ctor(name, robot._channel)  # noqa: SLF001

    try:
        nav = _client(MotionClient, args.nav)
        slam = _client(SLAMClient, args.slam)
        mon = Monitor(nav, slam, Path(args.out), args.period)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, mon.stop)
        print(f"monitoring {args.address} nav={args.nav} slam={args.slam} -> {args.out} (Ctrl-C to stop)")
        await mon.run()
        summ = mon.summary()
        print("\n" + summ)
        Path(str(args.out) + ".summary.txt").write_text(summ + "\n")
    finally:
        await robot.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
