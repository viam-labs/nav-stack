#!/usr/bin/env python3
"""Drive a loop of named locations while logging nav + SLAM status at 1 Hz.

Builds on ``monitor_nav.Monitor`` (same sampling / JSONL / summary) and adds a
goal controller with an explicit recovery policy for the case where the robot
cannot finish a leg:

    leg fails or stalls
      -> log the final status (error_msg, last_replan_info, obstacle histogram,
         SLAM localization_check)
      -> recovery 1: ``slam check_localization`` (stack's own apply policy),
         wait, retry the same goal once
      -> recovery 2: navigate back to the previous (known-good) location, then
         continue with the next leg
      -> if the fallback fails too: ``cancel``, stop the loop, leave the robot
         stopped, print the summary

"Stalled" means: still active but the pose moved < ``--stall-move-m`` for
``--stall-s`` seconds, or the leg exceeded ``--leg-timeout-s``. Both cancel
the goal first.

Transport loss (viam-server restart, network drop) is **not** a navigation
failure: the runner reconnects for up to ``--reconnect-s`` and resumes the leg
it was on. Time spent disconnected does not count toward leg timeouts.

Usage:
    source .local/monitor/env.sh
    .venv/bin/python scripts/route_loop.py --locations kevin-desk seanp dock_test \
        --laps 3 --dwell-s 20 --first-hold-s 60 --last-hold-s 60
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor_nav import Monitor, _fmt  # noqa: E402

from viam.robot.client import RobotClient  # noqa: E402
from viam.services.motion import MotionClient  # noqa: E402
from viam.services.slam import SLAMClient  # noqa: E402


class Conn:
    """Robot connection that can be rebuilt after viam-server restarts."""

    def __init__(self, address: str, key_id: str, key: str, nav_name: str, slam_name: str):
        self.address, self.key_id, self.key = address, key_id, key
        self.nav_name, self.slam_name = nav_name, slam_name
        self.robot: Optional[RobotClient] = None
        self.nav: Optional[MotionClient] = None
        self.slam: Optional[SLAMClient] = None
        self.reconnects = 0

    async def connect(self) -> None:
        await self.close()
        opts = RobotClient.Options.with_api_key(api_key=self.key, api_key_id=self.key_id)
        self.robot = await asyncio.wait_for(RobotClient.at_address(self.address, opts), timeout=30)
        self.nav = MotionClient(self.nav_name, self.robot._channel)  # noqa: SLF001
        self.slam = SLAMClient(self.slam_name, self.robot._channel)  # noqa: SLF001

    async def close(self) -> None:
        if self.robot is not None:
            try:
                await self.robot.close()
            except Exception:  # noqa: BLE001
                pass
        self.robot = self.nav = self.slam = None

    async def reconnect_until(self, timeout_s: float, log) -> bool:
        """Rebuild the connection until nav get_status answers, or give up."""
        deadline = time.monotonic() + timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                await self.connect()
                assert self.nav is not None
                await asyncio.wait_for(self.nav.do_command({"command": "get_status"}), timeout=10)
                self.reconnects += 1
                log("reconnected", attempt=attempt)
                return True
            except Exception as exc:  # noqa: BLE001
                log("reconnect_attempt_failed", attempt=attempt, error=f"{type(exc).__name__}: {str(exc)[:120]}")
                await asyncio.sleep(5)
        return False


class _Proxy:
    """Stable handle for Monitor; delegates to whatever client Conn holds now."""

    def __init__(self, conn: Conn, attr: str):
        self._conn, self._attr = conn, attr

    async def do_command(self, command: dict, **kw):
        client = getattr(self._conn, self._attr)
        if client is None:
            raise ConnectionError("not connected")
        return await asyncio.wait_for(client.do_command(command, **kw), timeout=10)


class RouteLoop:
    def __init__(self, conn: Conn, mon: Monitor, *, events_path: Path, leg_timeout_s: float,
                 stall_s: float, stall_move_m: float, dwell_s: float,
                 first_hold_s: float, last_hold_s: float, retry_wait_s: float, reconnect_s: float,
                 yaw_freeze_s: float = 6.0):
        self.conn = conn
        self.yaw_freeze_s = yaw_freeze_s
        self.fatal: Optional[str] = None
        self.mon = mon
        self.events_path = events_path
        self.leg_timeout_s = leg_timeout_s
        self.stall_s = stall_s
        self.stall_move_m = stall_move_m
        self.dwell_s = dwell_s
        self.first_hold_s = first_hold_s
        self.last_hold_s = last_hold_s
        self.retry_wait_s = retry_wait_s
        self.reconnect_s = reconnect_s
        self._stop = False
        self.legs: list[dict] = []
        self._locations: dict[str, dict] = {}
        self._consecutive_sample_errors = 0

    def stop(self):
        self._stop = True
        self.mon.stop()

    # -- helpers -----------------------------------------------------------
    def event(self, kind: str, **data: Any) -> None:
        row = {"t": time.time(), "event": kind, **data}
        with self.events_path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        ts = time.strftime("%H:%M:%S", time.localtime(row["t"]))
        short = {k: v for k, v in data.items() if k not in ("status", "sample")}
        print(f"=== {ts} {kind} {json.dumps(short, default=str)[:500]}", flush=True)

    async def one_sample(self) -> dict:
        row = await self.mon.sample()
        self.mon.samples.append(row)
        with self.mon.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if row.get("nav_error") and row.get("slam_error"):
            self._consecutive_sample_errors += 1
            row["conn_lost"] = True
            if self._consecutive_sample_errors == 3:
                self.event("connection_lost", nav_error=str(row.get("nav_error"))[:120])
            if self._consecutive_sample_errors >= 3:
                ok = await self.conn.reconnect_until(self.reconnect_s, self.event)
                if not ok:
                    self.event("reconnect_gave_up", after_s=self.reconnect_s)
                    self.stop()
                else:
                    self._consecutive_sample_errors = 0
        else:
            self._consecutive_sample_errors = 0
            self.mon._track_goals(row)  # noqa: SLF001
            self.mon._print_line(row)  # noqa: SLF001
        return row

    async def hold(self, seconds: float, label: str) -> None:
        self.event("hold_start", label=label, seconds=seconds)
        t_end = time.monotonic() + seconds
        while not self._stop and time.monotonic() < t_end:
            t0 = time.monotonic()
            await self.one_sample()
            await asyncio.sleep(max(0.0, self.mon.period_s - (time.monotonic() - t0)))
        self.event("hold_end", label=label)

    async def load_locations(self) -> None:
        res = await self.mon.nav.do_command({"command": "list_locations"})
        for loc in res.get("locations") or []:
            self._locations[str(loc["name"])] = dict(loc)

    def _dist_to(self, row: dict, name: str) -> Optional[float]:
        loc = self._locations.get(name)
        x, y = row["nav"].get("x"), row["nav"].get("y")
        if not loc or x is None or y is None:
            return None
        return math.hypot(float(loc["x"]) - float(x), float(loc["y"]) - float(y))

    async def _send(self, command: dict, what: str) -> Optional[dict]:
        """DoCommand with reconnect-and-retry on transport errors."""
        for attempt in (1, 2):
            try:
                return await self.mon.nav.do_command(command)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {str(exc)[:160]}"
                # A ValueError from the module (unknown location etc.) is a real
                # failure; anything else is treated as transport.
                if "ValueError" in msg or "KeyError" in msg:
                    self.event(f"{what}_rejected", error=msg)
                    return None
                self.event(f"{what}_transport_error", attempt=attempt, error=msg)
                if attempt == 1:
                    ok = await self.conn.reconnect_until(self.reconnect_s, self.event)
                    if not ok:
                        return None
        return None

    # -- goal control ------------------------------------------------------
    async def navigate(self, name: str, *, attempt: int, lap: int) -> dict:
        """Send one goal and poll until it ends. Returns a leg record."""
        leg = {"lap": lap, "target": name, "attempt": attempt, "start": time.time(),
               "outcome": None, "error_msg": None, "replans": 0, "states": {},
               "min_tick_score": None, "min_lc_score": None, "max_ray_mae": None,
               "disconnected_s": 0.0}
        pre = await self.one_sample()
        leg["start_dist_m"] = self._dist_to(pre, name)
        ack = await self._send({"command": "navigate_to_location", "name": name}, "navigate")
        if ack is None:
            leg["outcome"] = "send_failed"
            leg["end"] = time.time()
            self.event("leg_send_failed", target=name)
            self.legs.append(leg)
            return leg
        self.event("leg_start", lap=lap, target=name, attempt=attempt,
                   start_dist_m=leg["start_dist_m"], ack=ack)
        last_move_pose = None
        last_move_at = time.monotonic()
        seen_active = False
        last_replan_sig = None
        t_start = time.monotonic()
        lost_s = 0.0
        last_theta: Optional[float] = None
        last_theta_change_at = time.monotonic()
        yaw_cmd_since: Optional[float] = None
        while not self._stop:
            t0 = time.monotonic()
            row = await self.one_sample()
            if row.get("conn_lost"):
                lost_s += time.monotonic() - t0
                last_move_at = time.monotonic()  # do not count stall while blind
                continue
            n, s = row["nav"], row["slam"]
            active = bool(n.get("active"))
            obs = str(n.get("obstacle") or "")
            leg["states"][obs] = leg["states"].get(obs, 0) + 1
            for key, agg in (("lc_tick_match_score", "min_tick_score"), ("lc_score", "min_lc_score")):
                v = s.get(key)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    leg[agg] = v if leg[agg] is None else min(leg[agg], v)
            v = s.get("lc_ray_mae_m")
            if isinstance(v, (int, float)) and math.isfinite(v):
                leg["max_ray_mae"] = v if leg["max_ray_mae"] is None else max(leg["max_ray_mae"], v)
            info = n.get("last_replan_info")
            sig = json.dumps(info, sort_keys=True) if info else None
            if sig and sig != last_replan_sig:
                last_replan_sig = sig
                leg["replans"] += 1
                self.event("replan", target=name, trigger=n.get("last_replan_trigger"),
                           error=n.get("last_replan_error"), info=info)
            if obs in ("loc_hold", "loc_refine", "planning", "backup", "narrow_reverse", "wait", "loc_yield") and leg["states"][obs] == 1:
                self.event("nav_state", target=name, obstacle=obs,
                           tick_score=s.get("lc_tick_match_score"), lc=s.get("lc_status"))
            if n.get("localization_yield") and not leg.get("yielded"):
                leg["yielded"] = True
                self.event("loc_yield_start", target=name, tick_score=s.get("lc_tick_match_score"))
            ly = n.get("loc_yield")
            if ly and ly != leg.get("last_loc_yield"):
                leg["last_loc_yield"] = ly
                self.event("loc_yield_result", target=name, info=ly)
            if active:
                seen_active = True
            x, y = n.get("x"), n.get("y")
            if x is not None and y is not None:
                if last_move_pose is None or math.hypot(x - last_move_pose[0], y - last_move_pose[1]) >= self.stall_move_m:
                    last_move_pose = (x, y)
                    last_move_at = time.monotonic()
            if seen_active and not active:
                leg["outcome"] = str(n.get("state") or "unknown")
                leg["error_msg"] = n.get("error_msg") or None
                leg["goal_blocked"] = bool(n.get("goal_blocked"))
                leg["goal_offset_m"] = n.get("goal_offset_m")
                if leg["goal_blocked"]:
                    self.event("goal_blocked_finish", target=name, offset_m=n.get("goal_offset_m"))
                break
            elapsed = time.monotonic() - t_start - lost_s
            if not seen_active and elapsed > 10.0:
                leg["outcome"] = f"never_active:{n.get('state')}"
                leg["error_msg"] = n.get("error_msg") or None
                break
            # Yaw-freeze guard: nav is commanding a turn but the published
            # heading is not changing at all. Live 2026-10-07: IMU port baud
            # reset by another module -> SLAM yaw frozen -> robot spun for
            # 50 s. Abort fast; a retry would spin again.
            th = n.get("theta")
            cmd_w = float(n.get("cmd_w") or 0.0)
            if th is not None and active:
                if last_theta is None or abs(th - last_theta) > 0.01:
                    last_theta, last_theta_change_at = th, time.monotonic()
                if abs(cmd_w) >= 0.2 and abs(float(n.get("cmd_vx") or 0.0)) < 0.02:
                    if yaw_cmd_since is None:
                        yaw_cmd_since = time.monotonic()
                    if (time.monotonic() - yaw_cmd_since) >= self.yaw_freeze_s and (time.monotonic() - last_theta_change_at) >= self.yaw_freeze_s:
                        self.event("leg_cancel", target=name, reason="pose_not_tracking_yaw", cmd_w=cmd_w,
                                   theta=th, frozen_s=round(time.monotonic() - last_theta_change_at, 1),
                                   tick_score=s.get("lc_tick_match_score"))
                        await self._send({"command": "cancel"}, "cancel")
                        leg["outcome"] = "pose_not_tracking_yaw"
                        leg["error_msg"] = "published yaw did not change while a turn was commanded"
                        self.fatal = "pose_not_tracking_yaw"
                        break
                else:
                    yaw_cmd_since = None
            stalled = active and (time.monotonic() - last_move_at) >= self.stall_s and (n.get("dist_remaining_m") or 1.0) > 0.3
            if elapsed >= self.leg_timeout_s or stalled:
                reason = "loop_timeout" if elapsed >= self.leg_timeout_s else "stalled"
                self.event("leg_cancel", target=name, reason=reason, elapsed_s=round(elapsed, 1),
                           obstacle=obs, local_blocked=n.get("local_blocked"), nose_clear=n.get("nose_clear"),
                           path_cost_ahead=n.get("path_cost_ahead"), last_replan_error=n.get("last_replan_error"),
                           last_replan_info=n.get("last_replan_info"))
                await self._send({"command": "cancel"}, "cancel")
                leg["outcome"] = reason
                leg["error_msg"] = n.get("error_msg") or None
                break
            await asyncio.sleep(max(0.0, self.mon.period_s - (time.monotonic() - t0)))
        leg["end"] = time.time()
        leg["disconnected_s"] = round(lost_s, 1)
        post = await self.one_sample()
        leg["end_dist_m"] = self._dist_to(post, name)
        leg["final_status"] = post["nav"]
        leg["final_slam"] = post["slam"]
        self.event("leg_end", lap=lap, target=name, attempt=attempt, outcome=leg["outcome"],
                   error=leg["error_msg"], duration_s=round(leg["end"] - leg["start"], 1),
                   goal_blocked=leg.get("goal_blocked"), goal_offset_m=leg.get("goal_offset_m"),
                   yielded=leg.get("yielded", False),
                   disconnected_s=leg["disconnected_s"], end_dist_m=leg["end_dist_m"],
                   replans=leg["replans"], states=leg["states"], min_tick_score=leg["min_tick_score"],
                   min_lc_score=leg["min_lc_score"], max_ray_mae=leg["max_ray_mae"])
        self.legs.append(leg)
        return leg

    async def _wait_nav_idle(self, timeout_s: float = 10.0) -> bool:
        """SLAM skips localization checks while a goal is active; wait it out."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            row = await self.one_sample()
            if not row.get("conn_lost") and not row["nav"].get("active"):
                return True
            await asyncio.sleep(0.5)
        return False

    async def recover_localization(self, reason: str) -> None:
        """Let SLAM fix the pose while parked.

        Mid-nav the stack refuses corrections over 1 m. Once nav is idle a
        large jump is allowed but needs two agreeing matches, so ask up to
        three times and stop as soon as one is applied.
        """
        self.event("recovery_check_localization", reason=reason)
        if not await self._wait_nav_idle():
            self.event("recovery_check_localization_skipped", reason="nav still active")
        else:
            for i in range(1, 4):
                try:
                    res = await self.mon.slam.do_command({"command": "check_localization"})
                except Exception as exc:  # noqa: BLE001
                    self.event("recovery_check_localization_failed", attempt=i, error=str(exc)[:300])
                    break
                self.event("recovery_check_localization_result", attempt=i,
                           status=res.get("status"), reason=res.get("reason"), score=res.get("score"),
                           prior_score=res.get("prior_score"), ray_mae_m=res.get("ray_mae_m"),
                           shift_m=res.get("shift_m"), shift_deg=res.get("shift_deg"),
                           corrected=res.get("corrected"), confirm=res.get("confirm_count"))
                if res.get("corrected") or str(res.get("status")) not in ("awaiting_confirm", "skipped"):
                    break
                await asyncio.sleep(2.0)
        await self.hold(self.retry_wait_s, "recovery_wait")

    async def run(self, locations: list[str], laps: int) -> None:
        await self.load_locations()
        missing = [n for n in locations if n not in self._locations]
        if missing:
            raise SystemExit(f"unknown locations: {missing}; known: {sorted(self._locations)}")
        self.event("loop_start", locations=locations, laps=laps,
                   leg_timeout_s=self.leg_timeout_s, stall_s=self.stall_s)
        await self.hold(self.first_hold_s, "baseline_start")
        prev_good: Optional[str] = None
        consecutive_failures = 0
        for lap in range(1, laps + 1):
            for name in locations:
                if self._stop:
                    break
                leg = await self.navigate(name, attempt=1, lap=lap)
                if leg["outcome"] == "succeeded":
                    prev_good = name
                    consecutive_failures = 0
                    await self.hold(self.dwell_s, f"dwell@{name}")
                    continue
                if self._stop:
                    break
                if self.fatal:
                    # Pose is not tracking commands: retrying would repeat the
                    # spin. Stop everything and hand back to the operator.
                    self.event("loop_abort", reason=self.fatal, target=name)
                    await self._safe_cancel()
                    return
                # ---- recovery 1: localization check + retry once ----
                await self.recover_localization(f"leg to {name} ended {leg['outcome']}")
                leg2 = await self.navigate(name, attempt=2, lap=lap)
                if leg2["outcome"] == "succeeded":
                    prev_good = name
                    consecutive_failures = 0
                    self.event("recovery_retry_succeeded", target=name)
                    await self.hold(self.dwell_s, f"dwell@{name}")
                    continue
                consecutive_failures += 1
                # ---- recovery 2: fall back to last known-good location ----
                if prev_good and prev_good != name:
                    self.event("recovery_fallback", to=prev_good, after_failed=name)
                    fb = await self.navigate(prev_good, attempt=3, lap=lap)
                    if fb["outcome"] != "succeeded":
                        self.event("loop_abort", reason="fallback_failed", at=prev_good,
                                   details=fb.get("error_msg"))
                        await self._safe_cancel()
                        return
                    self.event("recovery_fallback_succeeded", at=prev_good)
                    await self.hold(self.dwell_s, f"dwell@{prev_good}")
                if consecutive_failures >= 2:
                    self.event("loop_abort", reason="two_consecutive_leg_failures")
                    await self._safe_cancel()
                    return
            if self._stop:
                break
        if not self._stop:
            await self.hold(self.last_hold_s, "baseline_end")
        self.event("loop_end", legs=len(self.legs), reconnects=self.conn.reconnects)

    async def _safe_cancel(self) -> None:
        try:
            await self.mon.nav.do_command({"command": "cancel"})
        except Exception:  # noqa: BLE001
            pass

    def summary(self) -> str:
        lines = [self.mon.summary(), "", f"reconnects: {self.conn.reconnects}", "legs:"]
        for leg in self.legs:
            lines.append(
                f"  lap{leg['lap']} -> {leg['target']} (try {leg['attempt']}): {leg['outcome']} "
                f"{_fmt((leg.get('end') or 0) - leg['start'], 0)}s err={leg.get('error_msg')!r} "
                f"start_d={_fmt(leg.get('start_dist_m'))} end_d={_fmt(leg.get('end_dist_m'))} "
                f"replans={leg['replans']} min_tick={_fmt(leg['min_tick_score'])} "
                f"min_lc={_fmt(leg['min_lc_score'])} max_mae={_fmt(leg['max_ray_mae'])} "
                f"offline={leg.get('disconnected_s')}s states={leg['states']}"
            )
        return "\n".join(lines)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default=os.environ.get("VIAM_ADDRESS"))
    ap.add_argument("--api-key-id", default=os.environ.get("VIAM_API_KEY_ID"))
    ap.add_argument("--api-key", default=os.environ.get("VIAM_API_KEY"))
    ap.add_argument("--nav", default="navigation")
    ap.add_argument("--slam", default="slam")
    ap.add_argument("--locations", nargs="+", required=True)
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--dwell-s", type=float, default=20.0)
    ap.add_argument("--first-hold-s", type=float, default=60.0)
    ap.add_argument("--last-hold-s", type=float, default=60.0)
    ap.add_argument("--leg-timeout-s", type=float, default=300.0)
    ap.add_argument("--stall-s", type=float, default=45.0)
    ap.add_argument("--stall-move-m", type=float, default=0.05)
    ap.add_argument("--retry-wait-s", type=float, default=5.0)
    ap.add_argument("--reconnect-s", type=float, default=240.0)
    ap.add_argument("--yaw-freeze-s", type=float, default=6.0,
                    help="abort when a turn is commanded but published yaw is frozen this long")
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--out", default=f".local/monitor/loop-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    args = ap.parse_args()
    if not (args.address and args.api_key_id and args.api_key):
        ap.error("need VIAM_ADDRESS / VIAM_API_KEY_ID / VIAM_API_KEY")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = Conn(args.address, args.api_key_id, args.api_key, args.nav, args.slam)
    await conn.connect()
    mon = Monitor(_Proxy(conn, "nav"), _Proxy(conn, "slam"), out, args.period)
    runner = RouteLoop(
        conn, mon, events_path=Path(str(out) + ".events.jsonl"), leg_timeout_s=args.leg_timeout_s,
        stall_s=args.stall_s, stall_move_m=args.stall_move_m, dwell_s=args.dwell_s,
        first_hold_s=args.first_hold_s, last_hold_s=args.last_hold_s, retry_wait_s=args.retry_wait_s,
        reconnect_s=args.reconnect_s, yaw_freeze_s=args.yaw_freeze_s,
    )
    ev_loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        ev_loop.add_signal_handler(sig, runner.stop)
    print(f"route loop on {args.address}: {args.locations} x{args.laps} -> {out}", flush=True)
    try:
        await runner.run(args.locations, args.laps)
    except Exception as exc:  # noqa: BLE001
        runner.event("runner_crashed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
    finally:
        try:
            if runner._stop:  # noqa: SLF001 - Ctrl-C / give-up: make sure the base is stopped
                await runner._safe_cancel()  # noqa: SLF001
        finally:
            summ = runner.summary()
            print("\n" + summ, flush=True)
            Path(str(out) + ".summary.txt").write_text(summ + "\n")
            await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
