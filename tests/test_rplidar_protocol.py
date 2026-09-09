"""Unit tests for RPLIDAR model / baud helpers (S2/S3 support)."""
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


def test_is_s_series():
    assert not proto.is_s_series(proto.MODEL_A1)
    assert not proto.is_s_series(proto.MODEL_A3)
    assert proto.is_s_series(proto.MODEL_S1)
    assert proto.is_s_series(proto.MODEL_S2)
    assert proto.is_s_series(proto.MODEL_S3)


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
