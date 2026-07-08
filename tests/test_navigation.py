import asyncio

import pytest

pytest.importorskip("viam")

from src.models.navigation import RosNavigation, _sync_mppi_model_dt


def test_do_command_raises_when_unconfigured():
    nav = RosNavigation("nav")
    with pytest.raises(RuntimeError, match="not configured"):
        asyncio.run(nav.do_command({"command": "get_status"}))


def test_refresh_zone_masks_raises_when_unconfigured():
    nav = RosNavigation("nav")
    with pytest.raises(RuntimeError, match="not configured"):
        nav._refresh_zone_masks()


def _mppi_params(controller_frequency, model_dt):
    return {
        "controller_server": {
            "ros__parameters": {
                "controller_frequency": controller_frequency,
                "FollowPath": {
                    "plugin": "nav2_mppi_controller::MPPIController",
                    "model_dt": model_dt,
                },
            }
        }
    }


def test_sync_mppi_model_dt_raises_to_controller_period():
    params = _mppi_params(5.0, 0.1)
    _sync_mppi_model_dt(params)
    fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    assert fp["model_dt"] == pytest.approx(0.2)


def test_sync_mppi_model_dt_keeps_smaller_dt_when_freq_high():
    params = _mppi_params(20.0, 0.1)  # period 0.05 < model_dt 0.1 -> leave as-is
    _sync_mppi_model_dt(params)
    fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    assert fp["model_dt"] == pytest.approx(0.1)


def test_sync_mppi_model_dt_ignores_non_mppi():
    params = _mppi_params(5.0, 0.1)
    fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    fp["plugin"] = "some_other_controller::Plugin"
    _sync_mppi_model_dt(params)
    assert fp["model_dt"] == pytest.approx(0.1)


def test_sync_mppi_model_dt_noop_without_controller_server():
    params = {"planner_server": {"ros__parameters": {}}}
    _sync_mppi_model_dt(params)  # must not raise
    assert "controller_server" not in params
