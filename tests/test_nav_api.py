import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("viam")

from viam.proto.service.navigation import MapType, Mode
from viam.services.navigation import Navigation

from src.config import NavConfig
from src.ros import conversions as conv
from src.models.nav_api import (
    NavApiMixin,
    geopoint_to_xy,
    map_geopoint,
    waypoints_to_path,
    zone_to_geo_geometry,
)
from src.models.nav_core import NavCoreMixin


# Concrete test double: the same mixin stack the real models use, with a stubbed
# runtime (no ROS/rclpy needed — nav_api/nav_core/Navigation don't pull the bridge).
class _NavT(NavApiMixin, NavCoreMixin, Navigation):
    def __init__(self, name, runtime):
        super().__init__(name)
        self._cfg = NavConfig.from_dict({"slam_service": "s", "base": "b"})
        self._runtime = runtime

    def _resolve_runtime(self):
        return self._runtime


def _svc(pose=None):
    pose = pose if pose is not None else conv.Pose2D(1.0, 2.0, 0.5)
    mgr = SimpleNamespace(
        get_pose_in_map=lambda: pose,
        cancel=MagicMock(),
        navigate=MagicMock(),
        nav_status=MagicMock(return_value={"active": False, "state": "succeeded"}),
    )
    runtime = SimpleNamespace(manager=mgr, map_store=None, slam_cfg=None, localization_check={})
    return _NavT("nav", runtime), mgr


# -- pure coordinate overload (latitude = x, longitude = y) -------------------
def test_map_geopoint_and_roundtrip():
    gp = map_geopoint(1.5, -2.5)
    assert (gp.latitude, gp.longitude) == (1.5, -2.5)
    assert geopoint_to_xy(gp) == (1.5, -2.5)


def test_waypoints_to_path():
    from viam.proto.service.navigation import Waypoint

    assert waypoints_to_path([]) == []
    wps = [Waypoint(id="0", location=map_geopoint(1, 2)), Waypoint(id="1", location=map_geopoint(3, 4))]
    paths = waypoints_to_path(wps)
    assert len(paths) == 1
    assert paths[0].destination_waypoint_id == "1"
    assert (paths[0].geopoints[1].latitude, paths[0].geopoints[1].longitude) == (3, 4)


def test_zone_to_geo_geometry_circle_box_polygon():
    circle = SimpleNamespace(name="c", geometry={"type": "circle", "center": [1.0, 2.0], "radius": 0.5})
    gg = zone_to_geo_geometry(circle)
    assert (gg.location.latitude, gg.location.longitude) == (1.0, 2.0)
    assert gg.geometries[0].box.dims_mm.x == pytest.approx(1000.0)

    box = SimpleNamespace(name="b", geometry={"type": "box", "center": [0.0, 0.0], "size": [2.0, 1.0]})
    assert zone_to_geo_geometry(box).geometries[0].box.dims_mm.y == pytest.approx(1000.0)

    poly = SimpleNamespace(name="p", geometry={"type": "polygon", "points": [[0, 0], [4, 0], [4, 2], [0, 2]]})
    gg = zone_to_geo_geometry(poly)
    assert (gg.location.latitude, gg.location.longitude) == (2.0, 1.0)
    assert gg.geometries[0].box.dims_mm.x == pytest.approx(4000.0)

    assert zone_to_geo_geometry(SimpleNamespace(name="x", geometry={"type": "blob"})) is None


# -- Navigation API over the mixin stack --------------------------------------
def test_get_location_maps_pose_lat_x_lng_y():
    svc, _ = _svc(conv.Pose2D(3.0, -4.0, 0.0))
    gp = asyncio.run(svc.get_location())
    assert (gp.latitude, gp.longitude) == (3.0, -4.0)


def test_get_properties_is_none_maptype():
    svc, _ = _svc()
    assert asyncio.run(svc.get_properties()) == MapType.MAP_TYPE_NONE


def test_waypoint_crud_and_paths():
    svc, _ = _svc()

    async def body():
        await svc.add_waypoint(map_geopoint(1, 2))
        await svc.add_waypoint(map_geopoint(3, 4))
        assert [w.id for w in await svc.get_waypoints()] == ["0", "1"]
        await svc.remove_waypoint("0")
        wps = await svc.get_waypoints()
        assert [w.id for w in wps] == ["1"]
        assert (await svc.get_paths())[0].destination_waypoint_id == "1"

    asyncio.run(body())


def test_set_mode_explore_rejected():
    svc, _ = _svc()
    with pytest.raises(ValueError, match="explore"):
        asyncio.run(svc.set_mode(Mode.MODE_EXPLORE))


def test_set_mode_manual_cancels_nav():
    svc, mgr = _svc()
    asyncio.run(svc.set_mode(Mode.MODE_MANUAL))
    assert svc._mode == Mode.MODE_MANUAL
    mgr.cancel.assert_called_once()


def _nogo(coords):
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"kind": "no_go"}}


def _slowdown(coords, mps):
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"kind": "slow_down", "max_speed_m_s": mps}}


def _label(name, xy):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": list(xy)},
            "properties": {"kind": "label", "label": name}}


def _svc_with_map(tmp_path, annotations=None):
    import numpy as np

    from src.nav.maps import MapStore

    from src.config import SlamConfig

    ms = MapStore(str(tmp_path))
    ms.get_or_create_map("default")
    ms.set_active_map("default")
    grid = {"grid": np.zeros((4, 4), dtype=np.int8), "resolution": 0.05, "origin_x": 0.0, "origin_y": 0.0}
    mgr = SimpleNamespace(
        node=SimpleNamespace(get_map=lambda: grid),
        navigate=MagicMock(),
        publish_zone_masks=MagicMock(),
        get_pose_in_map=lambda: conv.Pose2D(0.0, 0.0, 0.0),
        cancel=MagicMock(),
        nav_status=MagicMock(return_value={"active": False, "state": "succeeded"}),
        motors_enabled=MagicMock(return_value=True),
        set_motors_enabled=MagicMock(),
        set_nav_config=MagicMock(),
        stop_nav2=MagicMock(),
        ensure_nav2=MagicMock(),
    )
    slam_cfg = SlamConfig.from_dict({"base": "b", "lidar": "l", "maps_dir": str(tmp_path)})
    runtime = SimpleNamespace(manager=mgr, map_store=ms, slam_cfg=slam_cfg, localization_check={})
    fc = annotations or {"type": "FeatureCollection", "features": []}

    class _T(NavApiMixin, NavCoreMixin, Navigation):
        def __init__(self):
            super().__init__("nav")
            self._cfg = NavConfig.from_dict({"slam_service": "s", "base": "b", "max_vel_x": 0.5})
            self._runtime = runtime

        def _resolve_runtime(self):
            return runtime

        async def _get_annotations(self):
            return fc

    return _T(), mgr


def test_annotation_zones_conversion(tmp_path):
    square = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    fc = {"type": "FeatureCollection", "features": [_nogo(square), _slowdown(square, 0.25)]}
    svc, _ = _svc_with_map(tmp_path, fc)
    zones = asyncio.run(svc._annotation_zones())
    assert sorted(z.type for z in zones) == ["keepout", "speed_limit"]
    speed = next(z for z in zones if z.type == "speed_limit")
    assert speed.speed_pct == pytest.approx(50.0)  # 0.25 / max_vel_x(0.5) * 100
    assert speed.geometry["type"] == "polygon"


def test_plan_to_label_navigates_to_annotation(tmp_path):
    fc = {"type": "FeatureCollection", "features": [_label("charger", (2.0, 3.0))]}
    svc, mgr = _svc_with_map(tmp_path, fc)
    res = asyncio.run(svc.do_command({"command": "plan_to_label", "label": "charger"}))
    assert res["status"] == "navigating" and res["label"] == "charger"
    mgr.navigate.assert_called_once()
    assert mgr.navigate.call_args.args[:2] == (2.0, 3.0)
    mgr.publish_zone_masks.assert_called()  # masks refreshed before the goal


def test_plan_to_label_unknown_raises(tmp_path):
    svc, _ = _svc_with_map(tmp_path, {"type": "FeatureCollection", "features": []})
    with pytest.raises(ValueError, match="label"):
        asyncio.run(svc.do_command({"command": "plan_to_label", "label": "nope"}))


def test_get_obstacles_includes_annotations(tmp_path):
    square = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    fc = {"type": "FeatureCollection", "features": [_nogo(square)]}
    svc, _ = _svc_with_map(tmp_path, fc)
    obs = asyncio.run(svc.get_obstacles())
    assert len(obs) == 1  # the no_go annotation, as GeoGeometry


# -- Phase 3 DoCommand extensions ---------------------------------------------
def test_clear_waypoints(tmp_path):
    svc, _ = _svc_with_map(tmp_path)

    async def body():
        await svc.add_waypoint(map_geopoint(1, 2))
        assert await svc.do_command({"command": "clear_waypoints"}) == {"cleared": True}
        assert await svc.get_waypoints() == []

    asyncio.run(body())


def test_get_set_motors_enabled(tmp_path):
    svc, mgr = _svc_with_map(tmp_path)
    assert asyncio.run(svc.do_command({"command": "get_motors_enabled"})) == {"enabled": True}
    res = asyncio.run(svc.do_command({"command": "set_motors_enabled", "enabled": False}))
    assert res == {"enabled": False}
    mgr.set_motors_enabled.assert_called_once_with(False)


def test_replan_refreshes_masks_and_returns_points(tmp_path):
    svc, mgr = _svc_with_map(tmp_path)

    async def body():
        await svc.add_waypoint(map_geopoint(1.0, 2.0))
        res = await svc.do_command({"command": "replan"})
        assert res["ok"] is True and res["points"] == [[1.0, 2.0]]
        mgr.publish_zone_masks.assert_called()

    asyncio.run(body())


def test_set_planner_config_updates_and_restarts(tmp_path):
    svc, mgr = _svc_with_map(tmp_path)
    res = asyncio.run(
        svc.do_command(
            {"command": "set_planner_config", "inflation_radius": 0.7, "allow_unknown": False}
        )
    )
    assert res["inflation_radius"] == pytest.approx(0.7)
    assert res["allow_unknown"] is False
    mgr.stop_nav2.assert_called_once()
    mgr.ensure_nav2.assert_called_once()
    # round-trips through get_planner_config
    got = asyncio.run(svc.do_command({"command": "get_planner_config"}))
    assert got["inflation_radius"] == pytest.approx(0.7) and got["allow_unknown"] is False


def test_do_command_delegates_unknown_to_core(tmp_path):
    svc, _ = _svc_with_map(tmp_path)
    with pytest.raises(ValueError, match="unknown command"):
        asyncio.run(svc.do_command({"command": "definitely_not_a_command"}))


def test_reset_nav_state_clears_queue():
    svc, _ = _svc()

    async def body():
        await svc.add_waypoint(map_geopoint(1, 2))
        svc._mode = Mode.MODE_WAYPOINT
        svc._reset_nav_state()
        assert svc._mode == Mode.MODE_MANUAL
        assert await svc.get_waypoints() == []

    asyncio.run(body())
