#!/usr/bin/env python3
"""Poll nav ``describe_motion`` once per second; print only when the story changes.

Duration wording ("for about 3.1 s") is ignored for change detection so a
stable maneuver does not spam the console as the timer ticks.

Required environment:

  export VIAM_MACHINE_ADDRESS='<machine>.viam.cloud'
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
from viam.services.motion import MotionClient

# Match trailing / mid-phrase duration clauses produced by motion_summary.
_DURATION_RE = re.compile(
    r"\s+for about \d+(?:\.\d+)? s\b|\s+for \d+(?:\.\d+)? s\b"
)

NAV_NAME = os.environ.get("NAV_SERVICE_NAME", "nav")
INTERVAL_S = float(os.environ.get("WATCH_INTERVAL_S", "1"))


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"error: {name} environment variable is required")
    return value


ADDRESS = _require_env("VIAM_MACHINE_ADDRESS")
API_KEY = _require_env("VIAM_API_KEY")
API_KEY_ID = _require_env("VIAM_API_KEY_ID")


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
        # nav-stack navigation models register as rdk:service:motion.
        nav = MotionClient.from_robot(machine, NAV_NAME)
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
