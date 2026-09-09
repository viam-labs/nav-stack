"""Unit tests for RPLIDAR model / baud / dense-express helpers (S2/S3)."""
from __future__ import annotations

from src.lidar import rplidar_protocol as proto
from src.lidar.rplidar_serial import RPLidarSerial


def test_model_bytes_and_names():
    assert proto.MODEL_A1 == 24
    assert proto.MODEL_A3 == 49
    assert proto.MODEL_S1 == 97
    assert proto.MODEL_S2 == 113
    assert proto.MODEL_S3 == 129
    assert proto.model_name(proto.MODEL_S2) == "S2"
    assert proto.model_name(proto.MODEL_S3) == "S3"
    assert proto.model_name(0).startswith("unknown")


def test_is_s_series_and_tof():
    assert not proto.is_s_series(proto.MODEL_A1)
    assert not proto.is_tof_lidar(proto.MODEL_A1)
    assert not proto.is_tof_lidar(proto.MODEL_A3)
    assert proto.is_s_series(proto.MODEL_S1)
    assert proto.is_tof_lidar(proto.MODEL_S1)
    assert proto.is_tof_lidar(proto.MODEL_S2)
    assert proto.is_tof_lidar(proto.MODEL_S3)


def test_baudrates_probe_1m_first():
    assert proto.BAUDRATES[0] == 1000000
    assert 256000 in proto.BAUDRATES
    assert 115200 in proto.BAUDRATES


def test_s_series_skips_dtr_motor():
    class _Ser:
        def __init__(self):
            self.dtr = True

    for model in (proto.MODEL_S1, proto.MODEL_S2, proto.MODEL_S3):
        lidar = RPLidarSerial("/dev/null", serial_port=_Ser())
        lidar.info = {"model": model}
        lidar.start_motor()
        assert lidar._ser.dtr is True
        lidar._ser.dtr = False
        lidar.stop_motor()
        assert lidar._ser.dtr is False


def test_a1_toggles_dtr_motor():
    class _Ser:
        def __init__(self):
            self.dtr = True

    lidar = RPLidarSerial("/dev/null", serial_port=_Ser())
    lidar.info = {"model": proto.MODEL_A1}
    lidar.start_motor()
    assert lidar._ser.dtr is False
    lidar.stop_motor()
    assert lidar._ser.dtr is True


def test_command_with_payload_checksum():
    payload = proto.encode_express_scan_payload(3)
    pkt = proto.command_with_payload(proto.CMD_EXPRESS_SCAN, payload)
    assert pkt[0] == proto.SYNC
    assert pkt[1] == (proto.CMD_EXPRESS_SCAN | 0x80)
    assert pkt[2] == len(payload)
    assert pkt[3:-1] == payload
    checksum = proto.SYNC ^ (proto.CMD_EXPRESS_SCAN | 0x80) ^ len(payload)
    for b in payload:
        checksum ^= b
    assert pkt[-1] == (checksum & 0xFF)


def _make_dense_capsule(start_angle_deg: float, distances_mm, *, sync_bit: bool = False) -> bytes:
    angle_q6 = int(round(start_angle_deg * 64.0)) & 0x7FFF
    if sync_bit:
        angle_q6 |= proto.EXP_SYNC_BIT
    body = angle_q6.to_bytes(2, "little")
    for d in distances_mm:
        body += int(d).to_bytes(2, "little")
    assert len(body) == 2 + 80
    checksum = 0
    for b in body:
        checksum ^= b
    b0 = (proto.EXP_SYNC_1 << 4) | (checksum & 0x0F)
    b1 = (proto.EXP_SYNC_2 << 4) | ((checksum >> 4) & 0x0F)
    raw = bytes((b0, b1)) + body
    assert proto.dense_capsule_checksum_ok(raw)
    return raw


def test_dense_capsule_decode_yields_40_samples():
    prev = _make_dense_capsule(0.0, [1000] * 40, sync_bit=True)
    curr = _make_dense_capsule(10.0, [1100] * 40)
    samples = proto.decode_dense_capsule_pair(prev, curr)
    assert len(samples) == 40
    assert samples[0][2] == 1000.0
    angles = [s[1] for s in samples]
    assert angles[0] == 0.0
    assert max(angles) < 10.0 + 1e-6
