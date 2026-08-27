from __future__ import annotations

from unittest.mock import patch

from src.lidar.serial_ports import list_candidate_serial_ports


def test_candidate_ports_prefer_cp2102_by_id():
    fake = [
        "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
        "/dev/ttyUSB0",
        "/dev/ttyUSB2",
    ]

    with patch("src.lidar.serial_ports.glob.glob", side_effect=lambda pat: fake if "by-id" in pat else []):
        ports = list_candidate_serial_ports()

    assert ports[0].startswith("/dev/serial/by-id/usb-Silicon_Labs")
