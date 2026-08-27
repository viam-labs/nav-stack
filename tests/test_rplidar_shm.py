from __future__ import annotations

import math
import time
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.lidar import rplidar_protocol as proto
from src.lidar.rplidar_serial import RPLidarSerial
from src.ros import conversions as conv
from src.ros import pcshm
from src.ros.shm_lidar import ShmPointCloudClient


def test_node_round_trip():
    raw = proto.encode_node(new_scan=True, quality=15, angle_deg=90.0, distance_mm=1000.0)
    new_scan, quality, angle, dist = proto.decode_node(raw)
    assert new_scan is True
    assert quality == 15
    assert angle == pytest.approx(90.0)
    assert dist == pytest.approx(1000.0)


def test_polar_to_xyz_matches_viam_flip():
    # 0 deg → -X; 90 deg → +Y
    x0, y0, z0 = proto.polar_to_xyz_m(0.0, 1000.0)
    assert x0 == pytest.approx(-1.0)
    assert y0 == pytest.approx(0.0)
    assert z0 == 0.0
    x90, y90, _ = proto.polar_to_xyz_m(90.0, 2000.0)
    assert x90 == pytest.approx(0.0, abs=1e-9)
    assert y90 == pytest.approx(2.0)


class ScriptedSerial:
    """Minimal serial stand-in: replies to INFO/HEALTH/SCAN like an A1."""

    def __init__(self, scans: List[List[Tuple[int, float, float]]]):
        self.dtr = True
        self.writes: List[bytes] = []
        self._buf = bytearray()
        self._scans = scans
        self.in_waiting = 0

    def write(self, data: bytes) -> int:
        data = bytes(data)
        self.writes.append(data)
        if data == proto.command(proto.CMD_GET_INFO):
            self._buf += proto.descriptor(proto.INFO_LEN, True, proto.INFO_TYPE)
            self._buf += proto.encode_info(model=proto.MODEL_A1)
        elif data == proto.command(proto.CMD_GET_HEALTH):
            self._buf += proto.descriptor(proto.HEALTH_LEN, True, proto.HEALTH_TYPE)
            self._buf += proto.encode_health(0, 0)
        elif data == proto.command(proto.CMD_SCAN):
            self._buf += proto.descriptor(proto.NODE_LEN, False, proto.SCAN_TYPE)
            self._buf += proto.encode_scan_stream(self._scans)
        self.in_waiting = len(self._buf)
        return len(data)

    def read(self, n: int) -> bytes:
        chunk = bytes(self._buf[:n])
        del self._buf[:n]
        self.in_waiting = len(self._buf)
        return chunk

    def read_all(self) -> bytes:
        return self.read(len(self._buf))

    def close(self) -> None:
        pass


class JunkPrefixSerial(ScriptedSerial):
    def write(self, data: bytes) -> int:
        data = bytes(data)
        self.writes.append(data)
        if data == proto.command(proto.CMD_GET_INFO):
            self._buf += b"\x00\xff\x13junk"
            self._buf += proto.descriptor(proto.INFO_LEN, True, proto.INFO_TYPE)
            self._buf += proto.encode_info(model=proto.MODEL_A1)
        elif data == proto.command(proto.CMD_GET_HEALTH):
            self._buf += b"\x99"
            self._buf += proto.descriptor(proto.HEALTH_LEN, True, proto.HEALTH_TYPE)
            self._buf += proto.encode_health(0, 0)
        elif data == proto.command(proto.CMD_SCAN):
            self._buf += b"\x01\x02"
            self._buf += proto.descriptor(proto.NODE_LEN, False, proto.SCAN_TYPE)
            self._buf += proto.encode_scan_stream(self._scans)
        self.in_waiting = len(self._buf)
        return len(data)


def _circle_scan(n: int = 36, radius_mm: float = 2000.0) -> List[Tuple[int, float, float]]:
    step = 360.0 / n
    return [(20, i * step, radius_mm) for i in range(n)]


def test_serial_handshake_and_one_scan():
    # Two revolutions so iter_scans can yield the first complete one.
    fake = ScriptedSerial([_circle_scan(), _circle_scan(), _circle_scan()])
    lidar = RPLidarSerial(
        "/dev/null",
        serial_port=fake,
        motor_warmup_s=0.0,
        reset_settle_s=0.0,
    )
    lidar.open()
    assert fake.writes[0] == proto.command(proto.CMD_GET_INFO)
    assert lidar.info["model"] == proto.MODEL_A1
    scans = []
    for meas in lidar.iter_scans(min_points=20):
        scans.append(meas)
        assert fake.dtr is False  # A1 motor via DTR
        break
    lidar.close()
    assert len(scans[0]) == 36
    xyz = proto.scan_to_xyz_m(scans[0])
    assert len(xyz) == 36
    rs = [math.hypot(x, y) for x, y, _ in xyz]
    assert min(rs) == pytest.approx(2.0, abs=0.02)


def test_serial_handshake_resyncs_after_junk_prefix():
    fake = JunkPrefixSerial([_circle_scan(), _circle_scan(), _circle_scan()])
    lidar = RPLidarSerial(
        "/dev/null",
        serial_port=fake,
        motor_warmup_s=0.0,
        reset_settle_s=0.0,
    )
    lidar.open()
    assert lidar.info["model"] == proto.MODEL_A1
    meas = next(lidar.iter_scans(min_points=20))
    lidar.close()
    assert len(meas) == 36


def test_open_first_working_tries_ports():
    class SilentSerial:
        dtr = True

        def write(self, data: bytes) -> int:
            return len(data)

        def read(self, n: int) -> bytes:
            return b""

        def close(self) -> None:
            pass

    mocks = {
        "/dev/ttyUSB0": SilentSerial(),
        "/dev/ttyUSB1": ScriptedSerial([_circle_scan(), _circle_scan()]),
    }

    def factory(port, **kwargs):
        return mocks[port]

    import src.lidar.rplidar_serial as rs

    old = rs._pyserial
    rs._pyserial = type(
        "M",
        (),
        {
            "Serial": staticmethod(factory),
            "PARITY_NONE": 0,
            "STOPBITS_ONE": 1,
        },
    )
    try:
        lidar = RPLidarSerial.open_first_working(
            ["/dev/ttyUSB0", "/dev/ttyUSB1"],
            motor_warmup_s=0.0,
            reset_settle_s=0.0,
        )
        assert lidar.port == "/dev/ttyUSB1"
        lidar.close()
    finally:
        rs._pyserial = old


def test_scan_loop_reconnects_after_uart_exit():
    pytest.importorskip("viam")
    import threading

    from src.models.rplidar_shm import RPLidarShm

    class OneShotDevice:
        def __init__(self):
            self.calls = 0
            self.port = "/dev/fake"
            self.baudrate = 115200
            self.info = {"model": proto.MODEL_A1}

        def iter_scans(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield _circle_scan()
                raise proto.RPLidarError("short read 0/1")
            while True:
                yield _circle_scan()

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    cam = RPLidarShm("lidar")
    cam._stop = threading.Event()
    cam._warmup_scans = 0
    cam._reconnect_backoff_s = 0.01
    cam._max_reconnect_backoff_s = 0.05
    cam._shm_name = "/viam-pc-recon"
    cam._shm = pcshm.open_writer(cam._shm_name)
    fake = OneShotDevice()

    def fake_open():
        cam._device = fake
        cam._info = dict(fake.info)

    cam._open_device = fake_open  # type: ignore[method-assign]
    fake_open()
    thread = threading.Thread(target=cam._scan_loop, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2.0
    try:
        while time.monotonic() < deadline:
            if cam._reconnects >= 1 and cam._scans >= 2:
                break
            time.sleep(0.05)
        assert cam._reconnects >= 1
        assert cam._scans >= 2
        assert fake.calls >= 2
    finally:
        cam._stop.set()
        thread.join(timeout=2.0)
        cam.close_sync()


def test_iter_scans_raises_when_stalled():
    class StuckSerial(ScriptedSerial):
        def write(self, data: bytes) -> int:
            data = bytes(data)
            self.writes.append(data)
            if data == proto.command(proto.CMD_SCAN):
                self._buf += proto.descriptor(proto.NODE_LEN, False, proto.SCAN_TYPE)
                self._streaming = True
            elif data == proto.command(proto.CMD_GET_INFO):
                self._buf += proto.descriptor(proto.INFO_LEN, True, proto.INFO_TYPE)
                self._buf += proto.encode_info(model=proto.MODEL_A1)
            elif data == proto.command(proto.CMD_GET_HEALTH):
                self._buf += proto.descriptor(proto.HEALTH_LEN, True, proto.HEALTH_TYPE)
                self._buf += proto.encode_health(0, 0)
            self.in_waiting = len(self._buf)
            return len(data)

        def read(self, n: int) -> bytes:
            if getattr(self, "_streaming", False):
                while len(self._buf) < n:
                    self._buf += proto.encode_node(
                        new_scan=False, quality=10, angle_deg=0.0, distance_mm=1000.0
                    )
            return super().read(n)

    fake = StuckSerial([_circle_scan()])
    lidar = RPLidarSerial(
        "/dev/null",
        serial_port=fake,
        motor_warmup_s=0.0,
        reset_settle_s=0.0,
    )
    lidar.open()
    with pytest.raises(proto.RPLidarError, match="no complete scan"):
        next(lidar.iter_scans(min_points=20, max_stall_s=0.2))
    lidar.close()


def test_iter_scans_honors_abort_check():
    fake = ScriptedSerial([_circle_scan(), _circle_scan(), _circle_scan()])
    lidar = RPLidarSerial(
        "/dev/null",
        serial_port=fake,
        motor_warmup_s=0.0,
        reset_settle_s=0.0,
    )
    lidar.open()
    aborted = False

    def abort() -> bool:
        return aborted

    gen = lidar.iter_scans(min_points=20, abort_check=abort)
    next(gen)
    aborted = True
    with pytest.raises(proto.RPLidarError, match="scan aborted"):
        next(gen)
    lidar.close()


def test_stall_watchdog_signals_abort_without_closing_device():
    pytest.importorskip("viam")
    import threading

    from src.models.rplidar_shm import RPLidarShm

    class BlockingDevice:
        def __init__(self):
            self.port = "/dev/fake"
            self.baudrate = 115200
            self.info = {"model": proto.MODEL_A1}
            self.close_calls = 0
            self._released = threading.Event()

        def iter_scans(self, **kwargs):
            abort_check = kwargs.get("abort_check")
            while abort_check is None or not abort_check():
                time.sleep(0.05)
            raise proto.RPLidarError("scan aborted")

        def stop(self) -> None:
            pass

        def close(self) -> None:
            self.close_calls += 1
            self._released.set()

    cam = RPLidarShm("lidar")
    cam._stop = threading.Event()
    cam._warmup_scans = 0
    cam._max_publish_gap_s = 0.2
    cam._reconnect_backoff_s = 0.05
    cam._max_reconnect_backoff_s = 0.1
    cam._last_publish_wall = time.monotonic() - 1.0
    fake = BlockingDevice()
    cam._device = fake
    cam._info = dict(fake.info)

    watchdog = threading.Thread(target=cam._stall_watchdog, daemon=True)
    watchdog.start()
    deadline = time.monotonic() + 2.0
    try:
        while time.monotonic() < deadline:
            if cam._stall_abort.is_set():
                break
            time.sleep(0.05)
        assert cam._stall_abort.is_set()
        assert fake.close_calls == 0
    finally:
        cam._stop.set()
        cam._stall_abort.set()
        watchdog.join(timeout=2.0)


def test_kick_scan_loop_closes_device_without_spawning_thread():
    pytest.importorskip("viam")
    import threading

    from src.models.rplidar_shm import RPLidarShm

    cam = RPLidarShm("lidar")
    cam._stop = threading.Event()
    fake = MagicMock()
    fake.baudrate = 115200
    cam._device = fake

    cam._kick_scan_loop("test")
    assert cam._kick_count == 1
    assert cam._stall_abort.is_set()
    assert cam._device is None
    fake.stop.assert_called_once()
    fake.close.assert_called_once()


def test_close_sync_interrupts_long_backoff():
    pytest.importorskip("viam")
    import threading

    from src.models.rplidar_shm import RPLidarShm

    class FailDevice:
        port = "/dev/fake"
        baudrate = 115200
        info = {"model": proto.MODEL_A1}

        def iter_scans(self, **kwargs):
            raise proto.RPLidarError("uart died")

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    cam = RPLidarShm("lidar")
    stop = cam._stop
    cam._warmup_scans = 0
    cam._reconnect_backoff_s = 15.0
    cam._max_reconnect_backoff_s = 15.0
    cam._timeout_s = 2.0
    cam._motor_warmup_s = 0.0
    cam._shm_name = "/viam-pc-close-backoff"
    cam._shm = pcshm.open_writer(cam._shm_name)
    fake = FailDevice()
    cam._device = fake
    cam._info = dict(fake.info)

    thread = threading.Thread(target=cam._scan_loop, daemon=True)
    cam._thread = thread
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and cam._errors < 1:
        time.sleep(0.05)
    assert cam._errors >= 1

    t0 = time.monotonic()
    cam.close_sync()
    elapsed = time.monotonic() - t0

    assert cam._stop is stop
    assert elapsed < 2.0
    assert not thread.is_alive()
    assert cam._thread is None


def test_join_timeout_covers_reconnect_backoff():
    pytest.importorskip("viam")

    from src.models.rplidar_shm import RPLidarShm

    cam = RPLidarShm("lidar")
    cam._max_reconnect_backoff_s = 15.0
    cam._timeout_s = 2.0
    cam._motor_warmup_s = 1.0
    assert cam._join_timeout_s() >= cam._max_reconnect_backoff_s + cam._timeout_s


def test_reconfigure_reuses_stop_event():
    pytest.importorskip("viam")

    from src.models.rplidar_shm import RPLidarShm

    cam = RPLidarShm("lidar")
    stop = cam._stop
    cam._stop.set()
    cam.close_sync()
    cam._stop.clear()
    assert cam._stop is stop
    assert not cam._stop.is_set()


def test_publish_scan_writes_shm_pcd():
    pytest.importorskip("viam")
    from src.models.rplidar_shm import RPLidarShm

    name = "/viam-pc-rpt"
    cam = RPLidarShm("lidar")
    cam._shm = pcshm.open_writer(name)
    cam._shm_name = name
    try:
        cam._publish_scan(_circle_scan())
        client = ShmPointCloudClient()
        try:
            got = client.try_read(name)
            assert got is not None
            raw, _age = got
            pts = conv.parse_pcd(raw)
            assert pts.shape[0] == 36
            assert float(np.median(np.hypot(pts[:, 0], pts[:, 1]))) == pytest.approx(
                2.0, abs=0.05
            )
        finally:
            client.close()
    finally:
        cam.close_sync()
