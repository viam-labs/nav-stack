from __future__ import annotations

from unittest.mock import patch

from src.lidar.serial_ports import _dedupe_by_realpath, list_candidate_serial_ports


def test_candidate_ports_prefer_cp2102_by_id():
    fake = [
        "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
        "/dev/ttyUSB0",
        "/dev/ttyUSB2",
    ]

    def fake_glob(pat: str):
        if "by-id" in pat:
            return [p for p in fake if "by-id" in p]
        if "ttyUSB" in pat:
            return [p for p in fake if "ttyUSB" in p]
        return []

    with (
        patch("src.lidar.serial_ports.glob.glob", side_effect=fake_glob),
        patch("src.lidar.serial_ports.os.path.exists", return_value=True),
        patch(
            "src.lidar.serial_ports.os.path.realpath",
            side_effect=lambda p: p,
        ),
    ):
        ports = list_candidate_serial_ports()

    assert ports[0].startswith("/dev/serial/by-id/usb-Silicon_Labs")


def test_dedupe_collapses_by_id_and_by_path_aliases():
    paths = [
        "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        "/dev/serial/by-path/platform-xhci-hcd.0-usb-0:2:1.0-port0",
        "/dev/serial/by-path/platform-xhci-hcd.0-usbv2-0:2:1.0-port0",
        "/dev/ttyUSB2",
    ]

    def realpath(p: str) -> str:
        if "ttyUSB2" in p or "1a86" in p or "0:2" in p:
            return "/dev/ttyUSB2"
        return p

    with (
        patch("src.lidar.serial_ports.os.path.exists", return_value=True),
        patch("src.lidar.serial_ports.os.path.realpath", side_effect=realpath),
    ):
        out = _dedupe_by_realpath(paths)

    assert out == ["/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"]


def test_serial_claim_registry_and_busy_detection():
    from src.lidar.serial_ports import (
        claim_serial_port,
        is_port_busy_error,
        is_serial_claimed_by_other,
        release_serial_port,
    )

    claim_serial_port("imu", "/dev/ttyUSB0")
    try:
        assert is_serial_claimed_by_other("lidar", "/dev/ttyUSB0")
        assert not is_serial_claimed_by_other("imu", "/dev/ttyUSB0")
    finally:
        release_serial_port("/dev/ttyUSB0")

    assert is_port_busy_error(
        Exception(
            "Could not exclusively lock port: [Errno 11] Resource temporarily unavailable"
        )
    )


def test_chip_filter_separates_cp210_and_ch340():
    fake_id = [
        "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
    ]
    fake_path = [
        "/dev/serial/by-path/platform-xhci-hcd.0-usb-0:2:1.0-port0",
        "/dev/serial/by-path/platform-xhci-hcd.1-usb-0:1:1.0-port0",
    ]

    def fake_glob(pat: str):
        if "by-id" in pat:
            return list(fake_id)
        if "by-path" in pat:
            return list(fake_path)
        return []

    def realpath(p: str) -> str:
        if "1a86" in p or "0:2" in p:
            return "/dev/ttyUSB2"
        if "Silicon" in p or "0:1" in p:
            return "/dev/ttyUSB1"
        return p

    with (
        patch("src.lidar.serial_ports.glob.glob", side_effect=fake_glob),
        patch("src.lidar.serial_ports.os.path.exists", return_value=True),
        patch("src.lidar.serial_ports.os.path.realpath", side_effect=realpath),
    ):
        imu_ports = list_candidate_serial_ports(prefer_cp210=True, chip="cp210")
        lidar_ports = list_candidate_serial_ports(prefer_cp210=False, chip="ch340")

    assert imu_ports[0].startswith("/dev/serial/by-id/usb-Silicon_Labs")
    assert all(realpath(p) == "/dev/ttyUSB1" for p in imu_ports)
    assert all(realpath(p) == "/dev/ttyUSB2" for p in lidar_ports)


def test_sort_unclaimed_first_still_includes_claimed():
    from src.lidar.serial_ports import (
        claim_serial_port,
        release_serial_port,
        sort_unclaimed_first,
    )

    claim_serial_port("imu", "/dev/ttyUSB2")
    try:
        ordered = sort_unclaimed_first(
            "lidar", ["/dev/ttyUSB2", "/dev/ttyUSB1"]
        )
        assert ordered == ["/dev/ttyUSB1", "/dev/ttyUSB2"]
    finally:
        release_serial_port("/dev/ttyUSB2")
