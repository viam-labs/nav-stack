"""Exploration-aware nav view: ``viam-labs:nav-stack:explore-view``.

Same base render as ``nav-camera`` (costmap + plans + pose + footprint + goal),
with a HUD stamped on top summarising the running explore service's state:
phase, elapsed time (and budget, when set), map coverage, keyframe count, goal
counts and frontiers left.
Falls back to the plain nav render when explore isn't started yet or the
status call times out.
"""

from __future__ import annotations

import asyncio
import io
from typing import ClassVar, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

from ..config import ExploreViewConfig
from ..runtime import get_nav_view
from ..viz.nav_view import NavViewOptions, placeholder_png, render_nav_view

LOGGER = getLogger(__name__)

_HUD_BG = (0, 0, 0, 170)
_HUD_FG = (235, 235, 235, 255)
_HUD_MUTED = (170, 170, 170, 255)
_PHASE_COLORS = {
    "idle": (170, 170, 170, 255),
    # Navigation has no pose it trusts yet; nothing moves until it does.
    "waiting_for_pose": (235, 200, 80, 255),
    "exploring": (120, 220, 120, 255),
    # Scans stopped fitting the map; getting the robot back onto it.
    "relocating": (235, 90, 90, 255),
    "finishing": (230, 140, 60, 255),
    "done": (200, 200, 200, 255),
}
_HUD_LINE_H = 16
_HUD_PAD = 8
_HUD_WIDTH = 220


class ExploreView(Camera):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam-labs", "nav-stack"), "explore-view"
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[ExploreViewConfig] = None
        self._opts = NavViewOptions()
        self._explore: Optional[Generic] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        cam = cls(config.name)
        cam.reconfigure(config, dependencies)
        return cam

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        cfg = ExploreViewConfig.from_dict(struct_to_dict(config.attributes))
        return cfg.required_dependencies(), []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        cfg = ExploreViewConfig.from_dict(struct_to_dict(config.attributes))
        self._cfg = cfg
        self._opts = NavViewOptions(
            max_dim=cfg.max_dim,
            show_global_plan=cfg.show_global_plan,
            show_local_plan=cfg.show_local_plan,
            show_pose=cfg.show_pose,
            show_footprint=cfg.show_footprint,
            show_goal=cfg.show_goal,
            show_history=cfg.show_history,
            robot_radius_m=cfg.robot_radius_m,
            window_mode=cfg.window_mode,
            window_size_m=cfg.window_size_m,
            window_min_x=cfg.window_min_x,
            window_min_y=cfg.window_min_y,
            window_max_x=cfg.window_max_x,
            window_max_y=cfg.window_max_y,
        )
        explore_dep = dependencies.get(Generic.get_resource_name(cfg.explore))
        if explore_dep is None:
            LOGGER.warning(
                "explore-view '%s' missing explore dependency %r; HUD will show no run",
                self.name,
                cfg.explore,
            )
            self._explore = None
        else:
            self._explore = explore_dep  # type: ignore[assignment]
        LOGGER.info(
            "nav-stack explore-view '%s' rendering navigation %r + explore %r",
            self.name,
            cfg.navigation,
            cfg.explore,
        )

    # -- rendering -----------------------------------------------------------
    def _snapshot(self) -> Optional[dict]:
        cfg = self._cfg
        if cfg is None:
            return None
        view = get_nav_view(cfg.navigation)
        if view is None:
            return None
        if hasattr(view, "enable_viz"):
            view.enable_viz(cfg.plan_history_len)
        return view.viz_snapshot() if hasattr(view, "viz_snapshot") else view.snapshot()

    def _render_base(self, snapshot: dict) -> bytes:
        return render_nav_view(snapshot, self._opts)

    async def _fetch_status(self) -> Optional[Mapping[str, object]]:
        cfg = self._cfg
        if cfg is None or self._explore is None:
            return None
        try:
            return await asyncio.wait_for(
                self._explore.do_command({"command": "status"}),
                timeout=cfg.status_timeout_s,
            )
        except asyncio.TimeoutError:
            LOGGER.debug("explore-view '%s' status timed out", self.name)
            return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "explore-view '%s' status failed: %s: %s",
                self.name,
                type(exc).__name__,
                exc,
            )
            return None

    async def get_image(
        self,
        mime_type: str = "",
        *,
        extra=None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ViamImage:
        cfg = self._cfg
        if cfg is None:
            return ViamImage(
                placeholder_png("explore-view: not configured"), CameraMimeType.PNG
            )
        snapshot = await asyncio.to_thread(self._snapshot)
        if snapshot is None:
            return ViamImage(
                placeholder_png(
                    f"explore-view: navigation {cfg.navigation!r} not running yet"
                ),
                CameraMimeType.PNG,
            )
        status = await self._fetch_status()
        png = await asyncio.to_thread(self._compose, snapshot, status)
        return ViamImage(png, CameraMimeType.PNG)

    async def get_images(
        self, *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[list[NamedImage], ResponseMetadata]:
        img = await self.get_image()
        return [NamedImage(self.name, img.data, CameraMimeType.PNG)], ResponseMetadata()

    async def get_point_cloud(
        self, *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[bytes, str]:
        raise NotImplementedError("explore-view renders images only, not point clouds")

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        return Camera.Properties(supports_pcd=False, mime_types=[CameraMimeType.PNG])

    async def do_command(
        self,
        command: Mapping[str, object],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, object]:
        cmd = command.get("command")
        if cmd == "stats":
            snap = await asyncio.to_thread(self._snapshot)
            status = await self._fetch_status()
            known, area_m2, shape, resolution = self._coverage(snap)
            return {
                "mapped_m2": area_m2,
                "known_cells": known,
                "grid_shape": shape,
                "grid_resolution_m": resolution,
                "status": status,
            }
        return {"error": "unknown command"}

    # -- compose/HUD ---------------------------------------------------------
    def _compose(self, snapshot: dict, status: Optional[Mapping[str, object]]) -> bytes:
        base_bytes = self._render_base(snapshot)
        img = Image.open(io.BytesIO(base_bytes)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        lines = self._hud_lines(snapshot, status)
        height = _HUD_PAD * 2 + _HUD_LINE_H * len(lines)
        width = min(_HUD_WIDTH, img.size[0] - 8)
        draw.rectangle([(4, 4), (4 + width, 4 + height)], fill=_HUD_BG)
        y = 4 + _HUD_PAD
        for text, color in lines:
            draw.text((4 + _HUD_PAD, y), text, fill=color)
            y += _HUD_LINE_H

        # Progress bar below the Time line when we have one.
        pct = self._progress_pct(status)
        if pct is not None:
            bar_y = 4 + _HUD_PAD + _HUD_LINE_H * 2 - 2
            bar_w = width - _HUD_PAD * 2
            draw.rectangle(
                [(4 + _HUD_PAD, bar_y), (4 + _HUD_PAD + bar_w, bar_y + 3)],
                fill=(80, 80, 80, 200),
            )
            fill_w = int(round(bar_w * pct))
            if fill_w > 0:
                draw.rectangle(
                    [(4 + _HUD_PAD, bar_y), (4 + _HUD_PAD + fill_w, bar_y + 3)],
                    fill=(90, 170, 255, 230),
                )

        out = Image.alpha_composite(img, overlay).convert("RGB")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    def _hud_lines(
        self, snapshot: dict, status: Optional[Mapping[str, object]]
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        if status is None:
            return [("no explore run", _HUD_MUTED)]
        phase = str(status.get("phase", "?"))
        elapsed = float(status.get("elapsed_s") or 0.0)
        duration = status.get("duration_s") or None  # absent (or 0): no budget
        goals = int(status.get("goals") or 0)
        goals_failed = int(status.get("goals_failed") or 0)
        keyframes = int(status.get("keyframes_used") or 0)
        frontiers = int(status.get("frontier_count") or 0)
        unreachable = int(status.get("unreachable_frontiers") or 0)
        known, area_m2, _shape, _resolution = self._coverage(snapshot)

        lines: list[tuple[str, tuple[int, int, int, int]]] = []
        lines.append(
            (
                f"Phase:    {phase}",
                _PHASE_COLORS.get(phase, _HUD_FG),
            )
        )
        if duration is None:
            lines.append((f"Time:     {_fmt_time(elapsed)}", _HUD_FG))
        else:
            lines.append(
                (
                    f"Time:     {_fmt_time(elapsed)} / {_fmt_time(float(duration))}",
                    _HUD_FG,
                )
            )
            # Reserve a slot for the progress bar; filled with the bar itself.
            lines.append(("", _HUD_FG))
        if area_m2 is not None and known is not None:
            lines.append((f"Mapped:   {area_m2:.1f} m^2 ({known} cells)", _HUD_FG))
        else:
            lines.append(("Mapped:   n/a", _HUD_MUTED))
        lines.append((f"Fidelity: {keyframes} kf used", _HUD_FG))
        lines.append((f"Goals:    {goals} sent, {goals_failed} failed", _HUD_FG))
        frontier_line = f"Frontier: {frontiers} open"
        if unreachable:
            frontier_line += f", {unreachable} unreachable"
        lines.append((frontier_line, _HUD_FG))
        return lines

    @staticmethod
    def _progress_pct(status: Optional[Mapping[str, object]]) -> Optional[float]:
        if status is None:
            return None
        elapsed = float(status.get("elapsed_s") or 0.0)
        duration = status.get("duration_s")
        if not duration:
            return None  # no budget: runs until nothing is left to explore
        total = float(duration)
        if total <= 0:
            return None
        return max(0.0, min(1.0, elapsed / total))

    @staticmethod
    def _coverage(
        snapshot: Optional[dict],
    ) -> tuple[
        Optional[int], Optional[float], Optional[tuple[int, int]], Optional[float]
    ]:
        """Known-cell count and absolute area, both monotonic as mapping progresses.

        Returns (known_cells, area_m2, grid_shape, resolution_m). Any value may be
        None when the snapshot has no grid yet or the grid's resolution is missing.
        A percentage is deliberately avoided: nav-stack's costmap is a fixed-size
        canvas much larger than the explorable area, so "known / total" never
        matches what the operator sees filling in.
        """
        if snapshot is None:
            return (None, None, None, None)
        # Prefer the SLAM map (what the operator sees filling in); fall back to
        # the costmap, which has the same known-cell set wrapped in extra
        # inflation on top.
        grid_info = snapshot.get("map") or snapshot.get("costmap")
        if grid_info is None or grid_info.get("grid") is None:
            return (None, None, None, None)
        grid = np.asarray(grid_info["grid"])
        shape = (int(grid.shape[0]), int(grid.shape[1])) if grid.ndim >= 2 else None
        resolution = grid_info.get("resolution")
        resolution_f = float(resolution) if resolution else None
        known = int(np.count_nonzero(grid >= 0))
        area = known * resolution_f * resolution_f if resolution_f is not None else None
        return (known, area, shape, resolution_f)


def _fmt_time(s: float) -> str:
    s = max(0.0, s)
    m = int(s // 60)
    r = int(s - m * 60)
    return f"{m}:{r:02d}"


Registry.register_resource_creator(
    Camera.API,
    ExploreView.MODEL,
    ResourceCreatorRegistration(ExploreView.new, ExploreView.validate_config),
)
