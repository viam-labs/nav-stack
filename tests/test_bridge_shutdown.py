"""Shutdown / in-flight RPC cancellation for the ROS bridge."""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _FakeNode:
    pass


sys.modules.setdefault("rclpy", MagicMock())
sys.modules.setdefault("rclpy.node", MagicMock(Node=_FakeNode))
sys.modules.setdefault("rclpy.qos", MagicMock())
sys.modules.setdefault("rclpy.action", MagicMock())
sys.modules.setdefault("rclpy.time", MagicMock())
sys.modules.setdefault("rclpy.duration", MagicMock(Duration=MagicMock()))
for _mod in (
    "geometry_msgs",
    "geometry_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
    "tf2_ros",
    "nav2_msgs",
    "nav2_msgs.action",
):
    sys.modules.setdefault(_mod, MagicMock())

from src.ros.bridge import BridgeNode


@pytest.fixture
def bridge_with_loop():
    """Minimal object with BridgeNode._run / begin_shutdown bound."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    node = SimpleNamespace(
        _closing=False,
        _inflight_futures=set(),
        _inflight_lock=threading.Lock(),
        _loop=loop,
    )
    node.begin_shutdown = types.MethodType(BridgeNode.begin_shutdown, node)
    node._run = types.MethodType(BridgeNode._run, node)

    yield node

    node.begin_shutdown()
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)
    loop.close()


def test_run_cancels_future_on_timeout(bridge_with_loop):
    node = bridge_with_loop

    async def slow():
        await asyncio.sleep(5.0)
        return "done"

    with pytest.raises(TimeoutError):
        node._run(slow(), timeout=0.05)

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with node._inflight_lock:
            if not node._inflight_futures:
                break
        time.sleep(0.02)
    with node._inflight_lock:
        assert node._inflight_futures == set()


def test_begin_shutdown_cancels_inflight(bridge_with_loop):
    node = bridge_with_loop
    started = threading.Event()

    async def block():
        started.set()
        await asyncio.sleep(30.0)
        return "nope"

    future = asyncio.run_coroutine_threadsafe(block(), node._loop)
    with node._inflight_lock:
        node._inflight_futures.add(future)
    assert started.wait(timeout=1.0)

    node.begin_shutdown()
    assert node._closing is True
    with pytest.raises((concurrent.futures.CancelledError, asyncio.CancelledError)):
        future.result(timeout=2.0)


def test_run_rejects_work_after_shutdown(bridge_with_loop):
    node = bridge_with_loop
    node.begin_shutdown()

    async def quick():
        return 1

    with pytest.raises(RuntimeError, match="shutting down"):
        node._run(quick(), timeout=0.5)
