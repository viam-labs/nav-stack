from __future__ import annotations

import math
from typing import List, Tuple

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
