"""Tests for DDS discovery isolation helpers."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.ros.dds_env import (
    apply_dds_isolation,
    dds_status,
    stable_domain_id,
)


@pytest.fixture(autouse=True)
def _clear_dds_env(monkeypatch):
    for key in (
        "ROS_DOMAIN_ID",
        "ROS_AUTOMATIC_DISCOVERY_RANGE",
        "ROS_LOCALHOST_ONLY",
        "RMW_IMPLEMENTATION",
    ):
        monkeypatch.delenv(key, raising=False)


def test_stable_domain_id_in_safe_range():
    for seed in ("abc", "machine-1", "machine-2", ""):
        value = stable_domain_id(seed)
        assert 1 <= value <= 101


def test_stable_domain_id_deterministic():
    assert stable_domain_id("same") == stable_domain_id("same")
    assert stable_domain_id("a") != stable_domain_id("b")


def test_apply_dds_isolation_sets_defaults_and_persists(tmp_path, monkeypatch):
    persist = tmp_path / "ros_domain_id"
    status = apply_dds_isolation(persist)
    assert status["ros_automatic_discovery_range"] == "LOCALHOST"
    assert status["ros_localhost_only"] == "1"
    assert status["rmw_implementation"] == "rmw_fastrtps_cpp"
    domain = int(status["ros_domain_id"])
    assert 1 <= domain <= 101
    assert persist.read_text(encoding="utf-8").strip() == str(domain)

    # Second call keeps the persisted id even if env is cleared.
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    status2 = apply_dds_isolation(persist)
    assert status2["ros_domain_id"] == str(domain)


def test_apply_dds_isolation_respects_explicit_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "0")
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "SUBNET")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "0")
    persist = tmp_path / "ros_domain_id"
    status = apply_dds_isolation(persist)
    assert status["ros_domain_id"] == "0"
    assert status["ros_automatic_discovery_range"] == "SUBNET"
    assert status["ros_localhost_only"] == "0"
    assert not persist.exists()


def test_dds_status_reads_environ(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    assert dds_status() == {
        "ros_domain_id": "42",
        "ros_automatic_discovery_range": "LOCALHOST",
        "ros_localhost_only": "1",
        "rmw_implementation": "rmw_fastrtps_cpp",
    }
