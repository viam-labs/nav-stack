#!/usr/bin/env python3
"""Poll nav ``describe_motion`` once per second; print only when the story changes.

Duration wording ("for about 3.1 s") is ignored for change detection so a
stable maneuver does not spam the console as the timer ticks.

Defaults match the miti-nav2 machine. Prefer env vars over hardcoding keys:

  export VIAM_API_KEY='...'
  export VIAM_API_KEY_ID='...'
  python scripts/watch_describe_motion.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from datetime import datetime

from viam.robot.client import RobotClient
from viam.services.generic import Generic as GenericService

# Match trailing / mid-phrase duration clauses produced by motion_summary.
_DURATION_RE = re.compile(
    r"\s+for about \d+(?:\.\d+)? s\b|\s+for \d+(?:\.\d+)? s\b"
)

ADDRESS = os.environ.get(
    "VIAM_MACHINE_ADDRESS", "miti-nav2-main.q0s2f7mco8.viam.cloud"
)
NAV_NAME = os.environ.get("NAV_SERVICE_NAME", "nav")
INTERVAL_S = float(os.environ.get("WATCH_INTERVAL_S", "1"))

# Prefer env; fall back to the values from your Viam sample script.
API_KEY = os.environ.get("VIAM_API_KEY", "s5x9mjenwda4s7dz4smudgzzerknomd3")
API_KEY_ID = os.environ.get(
    "VIAM_API_KEY_ID", "2028ab21-233d-4fec-97e4-36988dce7398"
)


def fingerprint(summary: str) -> str:
    """Normalize a summary so duration-only edits compare equal."""
    return _DURATION_RE.sub("", summary or "").strip()


async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=API_KEY,
        api_key_id=API_KEY_ID,
    )
    return await RobotClient.at_address(ADDRESS, opts)


async def main() -> None:
    async with await connect() as machine:
        nav = GenericService.from_robot(machine, NAV_NAME)
        last_fp: str | None = None
        print(
            f"watching {NAV_NAME!r} on {ADDRESS} every {INTERVAL_S:g}s "
            f"(Ctrl-C to stop; duration ignored for diffs)",
            file=sys.stderr,
            flush=True,
        )
        while True:
            try:
                result = await nav.do_command({"command": "describe_motion"})
            except Exception as exc:  # noqa: BLE001 - keep polling through transient RPC errors
                print(f"{datetime.now().strftime('%H:%M:%S')} error: {exc}", flush=True)
                await asyncio.sleep(INTERVAL_S)
                continue

            summary = str(result.get("summary") or "")
            fp = fingerprint(summary)
            if fp and fp != last_fp:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"{ts}  {summary}", flush=True)
                last_fp = fp
            await asyncio.sleep(INTERVAL_S)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
