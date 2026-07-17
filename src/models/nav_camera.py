"""Navigation visualization camera: ``viam-labs:nav-stack:nav-camera``.

A read-only Viam ``rdk:component:camera`` that renders what the navigation
service is doing — the Nav2 **global costmap** (so you see the inflated cost
surface the planner actually reasons over) with the **active global plan**, the
**local plan**, a fading **history of superseded plans** for the current goal,
the **robot pose + footprint**, and the **goal marker** drawn on top.

It reads straight from the running navigation service's in-process
:class:`~..ros.bridge.BridgeNode` (found via the process-global bridge registry
keyed by the ``navigation`` config attribute), so there is no extra ROS process
and no RPC round-trip. Because it consumes only Nav2's standard costmap/plan
topics, it works with **any** SLAM backend (built-in slam_toolbox or an external
``rdk:service:slam``), not just one algorithm.

Point in the Viam app's camera stream at this component to watch planning live.
"""
from __future__ import annotations

import asyncio
from typing import ClassVar, Mapping, Optional, Sequence

from typing_extensions import Self

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from ..config import NavCameraConfig
from ..ros.nav_view import (
    NavViewOptions,
    legend_text,
    placeholder_png,
    render_nav_view,
)
from ..runtime import get_bridge

LOGGER = getLogger(__name__)


class NavCamera(Camera):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "nav-camera")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[NavCameraConfig] = None
        self._opts = NavViewOptions()

    # -- registration --------------------------------------------------------
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
        cfg = NavCameraConfig.from_dict(struct_to_dict(config.attributes))
        return cfg.required_dependencies(), []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        cfg = NavCameraConfig.from_dict(struct_to_dict(config.attributes))
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
        # The bridge is resolved lazily per frame: Nav2 bringup is asynchronous,
        # so the bridge/costmap may not exist yet when this camera configures.
        LOGGER.info(
            f"nav-stack nav-camera '{self.name}' rendering navigation "
            f"{cfg.navigation!r}"
        )

    # -- rendering -----------------------------------------------------------
    def _render(self) -> bytes:
        """Build + render one frame. Runs off the event loop (blocking PIL/TF)."""
        cfg = self._cfg
        if cfg is None:
            return placeholder_png("nav-camera: not configured")
        bridge = get_bridge(cfg.navigation)
        if bridge is None:
            return placeholder_png(
                f"nav-camera: navigation {cfg.navigation!r} not running yet"
            )
        bridge.enable_viz(cfg.plan_history_len)
        snapshot = bridge.viz_snapshot()
        return render_nav_view(snapshot, self._opts)

    async def get_image(
        self, mime_type: str = "", *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> ViamImage:
        png = await asyncio.to_thread(self._render)
        return ViamImage(png, CameraMimeType.PNG)

    async def get_images(
        self, *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[list[NamedImage], ResponseMetadata]:
        png = await asyncio.to_thread(self._render)
        return [NamedImage(self.name, png, CameraMimeType.PNG)], ResponseMetadata()

    async def get_point_cloud(
        self, *, extra=None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[bytes, str]:
        raise NotImplementedError("nav-camera renders images only, not point clouds")

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        return Camera.Properties(
            supports_pcd=False, mime_types=[CameraMimeType.PNG]
        )

    async def do_command(
        self, command: Mapping[str, object], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, object]:
        """Debug/introspection surface (no video stream needed).

        ``{"command": "legend"}`` returns the colour key; anything else (e.g.
        ``{"command": "stats"}``) returns a summary of the current view.
        """
        if command.get("command") == "legend":
            return {"legend": legend_text()}

        cfg = self._cfg
        if cfg is None:
            return {"configured": False}
        bridge = get_bridge(cfg.navigation)
        if bridge is None:
            return {"navigation": cfg.navigation, "bridge": False}
        bridge.enable_viz(cfg.plan_history_len)
        snap = bridge.viz_snapshot()

        def _n(v) -> int:
            return len(v) if v else 0

        return {
            "navigation": cfg.navigation,
            "bridge": True,
            "has_costmap": snap.get("costmap") is not None,
            "has_map": snap.get("map") is not None,
            "global_plan_points": _n(snap.get("global_plan")),
            "local_plan_points": _n(snap.get("local_plan")),
            "plan_history": _n(snap.get("plan_history")),
            "footprint_points": _n(snap.get("footprint")),
            "goal": list(snap["goal"]) if snap.get("goal") else None,
            "pose": list(snap["pose"]) if snap.get("pose") else None,
        }


Registry.register_resource_creator(
    Camera.API,
    NavCamera.MODEL,
    ResourceCreatorRegistration(NavCamera.new, NavCamera.validate_config),
)
