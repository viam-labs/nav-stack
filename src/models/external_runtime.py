"""Shared builder for the external-SLAM ROS runtime.

Both ``navigation-external`` (generic DoCommand API) and ``navigation-service``
(``rdk:service:navigation`` API) drive Nav2 from an external Viam SLAM service,
so they need the same stack: a :class:`~..ros.manager.RosManager` running the
sensor bridge + :class:`~..ros.external_slam.ExternalSlamPublisher`, a nav-stack
``MapStore`` for locations/zones, and a ``SlamRuntime`` handle for
:class:`~.nav_core.NavCoreMixin`. This assembles that stack from an
``ExternalNavConfig`` + the resolved Viam dependencies; the caller owns Nav2
bringup and teardown.
"""
from __future__ import annotations

import asyncio
from typing import Mapping, Tuple, cast

from viam.components.base import Base
from viam.components.camera import Camera
from viam.components.movement_sensor import MovementSensor
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.services.slam import SLAM

from ..config import ExternalNavConfig
from ..nav.maps import MapStore
from ..ros.manager import RosManager
from ..ros.odom_source import TypedMovementSensorOdom, TypedOdomConfig
from ..ros.sensor_io import build_io_provider
from ..runtime import SlamRuntime


def build_external_runtime(
    ext: ExternalNavConfig,
    dependencies: Mapping[ResourceName, ResourceBase],
    *,
    loop: asyncio.AbstractEventLoop,
    logger,
) -> Tuple[Base, RosManager, SlamRuntime, SLAM]:
    """Build + start the external-SLAM ROS runtime.

    Returns ``(base, manager, runtime, slam)`` (``slam`` is the Viam SLAM dep, so
    the caller can query its annotations). The manager's bridge is started (with
    the external SLAM publisher attached) but Nav2 is *not* launched — the caller
    calls ``manager.set_nav_config`` + ``ensure_nav2_async``. The caller also owns
    shutting down any prior manager before calling this.
    """
    bridge_cfg = ext.bridge
    base = cast(Base, dependencies[Base.get_resource_name(ext.nav.base)])
    slam = cast(SLAM, dependencies[SLAM.get_resource_name(ext.slam_service)])
    cameras = {
        lidar.name: cast(Camera, dependencies[Camera.get_resource_name(lidar.name)])
        for lidar in bridge_cfg.lidars
    }
    movement_sensor = (
        cast(
            MovementSensor,
            dependencies[MovementSensor.get_resource_name(bridge_cfg.movement_sensor)],
        )
        if bridge_cfg.movement_sensor
        else None
    )
    heading_sensor = (
        cast(
            MovementSensor,
            dependencies[MovementSensor.get_resource_name(bridge_cfg.heading_sensor)],
        )
        if bridge_cfg.heading_sensor
        else None
    )

    odom_reader = (
        TypedMovementSensorOdom(
            movement_sensor,
            TypedOdomConfig(
                trust_pose=ext.trust_movement_sensor_pose,
                snap_heading=ext.snap_heading,
            ),
            logger=logger,
        )
        if movement_sensor is not None
        else None
    )
    io = build_io_provider(
        base=base,
        cameras=cameras,
        cfg=bridge_cfg,
        movement_sensor=movement_sensor,
        heading_sensor=heading_sensor,
        skip_get_laser_scan=set(),
        odom_reader=odom_reader,
        logger=logger,
    )

    # external_slam -> the bridge starts the ExternalSlamPublisher (publishes
    # /map + map->odom from the SLAM service); slam_toolbox is never launched.
    manager = RosManager(bridge_cfg, logger=logger, external_slam=slam)
    manager.start(io, loop, nav_cfg=ext.nav)

    node = manager.node
    if node is not None:
        # Wire cmd_vel history recording now the bridge node exists.
        node._io = build_io_provider(
            base=base,
            cameras=cameras,
            cfg=bridge_cfg,
            movement_sensor=movement_sensor,
            heading_sensor=heading_sensor,
            skip_get_laser_scan=set(),
            odom_reader=odom_reader,
            logger=logger,
            record_cmd_vel=node.record_cmd_vel,
        )

    # Locations/zones live in a nav-stack-managed map store (the external SLAM
    # owns the occupancy grid, delivered live via get_grid).
    map_store = MapStore(bridge_cfg.maps_dir)
    active = bridge_cfg.active_map or map_store.get_active_map_name() or "default"
    map_store.get_or_create_map(active)
    map_store.set_active_map(active)

    loc_check = (
        node._external.localization_check
        if node is not None and node._external is not None
        else {"status": "unknown"}
    )
    runtime = SlamRuntime(manager, map_store, bridge_cfg, loc_check)
    return base, manager, runtime, slam
