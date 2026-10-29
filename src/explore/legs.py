"""Execute one navigation leg and classify how it ended."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from .clients import NavClient, Pose


@dataclass
class LegResult:
    ok: bool
    reason: str
    end_pose: Optional[Pose]
    elapsed_s: float


def leg_timeout_s(planned_length_m: float, max_linear_mps: float) -> float:
    return max(20.0, 3.0 * planned_length_m / max_linear_mps)


class LegRunner:
    def __init__(
        self,
        nav: NavClient,
        *,
        max_linear_mps: float,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
        poll_s: float = 0.5,
        should_stop: Callable[[], bool] = lambda: False,
    ):
        self._nav = nav
        self._max_linear_mps = max_linear_mps
        self._clock = clock
        self._sleep = sleep
        self._poll_s = poll_s
        self._should_stop = should_stop

    async def run(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        planned: bool,
        planned_length_m: float,
        goal_tolerance_m: float = 1.0,
    ) -> LegResult:
        start = self._clock()
        timeout = leg_timeout_s(planned_length_m, self._max_linear_mps)
        if planned:
            await self._nav.navigate_to_point(x, y, theta)
        else:
            await self._nav.go_to_point(x, y, theta, velocity_mps=self._max_linear_mps)

        def done(ok: bool, reason: str, pose: Optional[Pose]) -> LegResult:
            return LegResult(ok, reason, pose, self._clock() - start)

        while True:
            status = await self._nav.status()
            if self._should_stop():
                await self._nav.cancel()
                return done(False, "canceled", status.pose)
            if not status.localized:
                await self._nav.cancel()
                return done(False, "lost_pose", status.pose)
            if not status.active:
                if status.state in ("failed", "canceled"):
                    return done(False, status.state, status.pose)
                if status.state == "succeeded":
                    pose = status.pose
                    if pose is None or math.hypot(pose.x - x, pose.y - y) > goal_tolerance_m:
                        return done(False, "far_from_goal", pose)
                    return done(True, "", pose)
            if self._clock() - start > timeout:
                await self._nav.cancel()
                return done(False, "timeout", status.pose)
            await self._sleep(self._poll_s)
