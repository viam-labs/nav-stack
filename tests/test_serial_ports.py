from __future__ import annotations

from unittest.mock import patch

from src.lidar.serial_ports import (
    _dedupe_by_realpath,
    is_safe_sensor_serial_port,
    list_candidate_serial_ports,
    normalize_exclude_list,
)


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
        return []

    with (
        patch("src.lidar.serial_ports.glob.glob", side_effect=fake_glob),
        patch("src.lidar.serial_ports.os.path.exists", return_value=True),
        patch(
            "src.lidar.serial_ports.os.path.realpath",
            side_effect=lambda p: p,
        ),
        patch("src.lidar.serial_ports._usb_vid_pid_for_port", return_value=None),
        patch("src.lidar.serial_ports._tty_driver_name", return_value="cp210x"),
        patch("src.lidar.serial_ports._tty_bound_to_can_netdev", return_value=False),
    ):
        ports = list_candidate_serial_ports()

    assert ports[0].startswith("/dev/serial/by-id/usb-Silicon_Labs")
    assert all("/ttyUSB" not in p for p in ports)


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

    def fake_glob(pat: str):
        if "by-id" in pat:
            return list(fake_id)
        return []

    def realpath(p: str) -> str:
        if "1a86" in p:
            return "/dev/ttyUSB2"
        if "Silicon" in p:
            return "/dev/ttyUSB1"
        return p

    with (
        patch("src.lidar.serial_ports.glob.glob", side_effect=fake_glob),
        patch("src.lidar.serial_ports.os.path.exists", return_value=True),
        patch("src.lidar.serial_ports.os.path.realpath", side_effect=realpath),
        patch("src.lidar.serial_ports._usb_vid_pid_for_port", return_value=None),
        patch("src.lidar.serial_ports._tty_driver_name", return_value="cp210x"),
        patch("src.lidar.serial_ports._tty_bound_to_can_netdev", return_value=False),
    ):
        imu_ports = list_candidate_serial_ports(prefer_cp210=True, chip="cp210")
        lidar_ports = list_candidate_serial_ports(prefer_cp210=False, chip="ch340")

    assert imu_ports[0].startswith("/dev/serial/by-id/usb-Silicon_Labs")
    assert all(realpath(p) == "/dev/ttyUSB1" for p in imu_ports)
    assert all(realpath(p) == "/dev/ttyUSB2" for p in lidar_ports)


def test_prefer_cp210_false_puts_ch340_first():
    fake_id = [
        "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
    ]

    def fake_glob(pat: str):
        if "by-id" in pat:
            return list(fake_id)
        return []

    def realpath(p: str) -> str:
        if "1a86" in p:
            return "/dev/ttyUSB2"
        if "Silicon" in p:
            return "/dev/ttyUSB1"
        return p

    with (
        patch("src.lidar.serial_ports.glob.glob", side_effect=fake_glob),
        patch("src.lidar.serial_ports.os.path.exists", return_value=True),
        patch("src.lidar.serial_ports.os.path.realpath", side_effect=realpath),
        patch("src.lidar.serial_ports._usb_vid_pid_for_port", return_value=None),
        patch("src.lidar.serial_ports._tty_driver_name", return_value="ch341"),
        patch("src.lidar.serial_ports._tty_bound_to_can_netdev", return_value=False),
    ):
        ports = list_candidate_serial_ports(prefer_cp210=False)

    assert ports[0].startswith("/dev/serial/by-id/usb-1a86")
    assert ports[1].startswith("/dev/serial/by-id/usb-Silicon_Labs")


def test_eio_counts_as_missing_for_retry():
    from src.lidar.serial_ports import is_port_missing_error

    assert is_port_missing_error(
        Exception(
            "SerialException(5, \"could not open port: [Errno 5] Input/output error\")"
        )
    )


def test_autodetect_never_lists_openmoko_can_or_by_path():
    fake_id = [
        "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        "/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_abc-if00-port0",
        "/dev/serial/by-id/usb-OpenMoko_Inc._Geschwister_Schneider_CAN_adapter_123-if00-port0",
        "/dev/serial/by-id/usb-1d50_606f_CAN_if00-port0",
    ]
    fake_path = [
        "/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-port0",
    ]

    def fake_glob(pat: str):
        if "by-id" in pat:
            return list(fake_id)
        if "by-path" in pat:
            return list(fake_path)
        if "ttyACM" in pat:
            return ["/dev/ttyACM0"]
        return []

    def realpath(p: str) -> str:
        if "1a86" in p:
            return "/dev/ttyUSB0"
        if "Silicon" in p:
            return "/dev/ttyUSB1"
        if "OpenMoko" in p or "1d50" in p or "ttyACM" in p:
            return "/dev/ttyACM0"
        if "by-path" in p:
            return "/dev/ttyACM0"
        return p

    with (
        patch("src.lidar.serial_ports.glob.glob", side_effect=fake_glob),
        patch("src.lidar.serial_ports.os.path.exists", return_value=True),
        patch("src.lidar.serial_ports.os.path.realpath", side_effect=realpath),
        patch("src.lidar.serial_ports._usb_vid_pid_for_port", return_value=None),
        patch("src.lidar.serial_ports._tty_driver_name", return_value="cp210x"),
        patch("src.lidar.serial_ports._tty_bound_to_can_netdev", return_value=False),
    ):
        ports = list_candidate_serial_ports(prefer_cp210=True)
        lidar = list_candidate_serial_ports(prefer_cp210=True, chip="cp210")
        imu = list_candidate_serial_ports(prefer_cp210=False, chip="ch340")

    joined = " ".join(ports)
    assert "OpenMoko" not in joined
    assert "1d50" not in joined
    assert "by-path" not in joined
    assert "ttyACM" not in joined
    assert lidar[0].startswith("/dev/serial/by-id/usb-Silicon_Labs")
    assert imu[0].startswith("/dev/serial/by-id/usb-1a86")


def test_refuse_openmoko_vid_pid():
    with patch(
        "src.lidar.serial_ports._usb_vid_pid_for_port",
        return_value=("1d50", "606f"),
    ):
        assert not is_safe_sensor_serial_port("/dev/ttyACM0")


def test_serial_exclude_extra_substring():
    fake_id = [
        "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
    ]

    def fake_glob(pat: str):
        if "by-id" in pat:
            return list(fake_id)
        return []

    with (
        patch("src.lidar.serial_ports.glob.glob", side_effect=fake_glob),
        patch("src.lidar.serial_ports.os.path.exists", return_value=True),
        patch(
            "src.lidar.serial_ports.os.path.realpath",
            side_effect=lambda p: (
                "/dev/ttyUSB0" if "1a86" in p else "/dev/ttyUSB1" if "Silicon" in p else p
            ),
        ),
        patch("src.lidar.serial_ports._usb_vid_pid_for_port", return_value=None),
        patch("src.lidar.serial_ports._tty_driver_name", return_value="cp210x"),
        patch("src.lidar.serial_ports._tty_bound_to_can_netdev", return_value=False),
    ):
        ports = list_candidate_serial_ports(exclude=["1a86"])

    assert len(ports) == 1
    assert "Silicon_Labs" in ports[0]


def test_shares_usb_hub_with_can_detects_sibling():
    from src.lidar.serial_ports import shares_usb_hub_with_can

    def read_text(p: str) -> str:
        if p.endswith("idVendor") and "1-1.4" in p:
            return "1d50"
        if p.endswith("idProduct") and "1-1.4" in p:
            return "606f"
        if p.endswith("idVendor"):
            return "10c4"
        if p.endswith("idProduct"):
            return "ea60"
        return ""

    with (
        patch(
            "src.lidar.serial_ports._usb_device_sysfs",
            return_value="/sys/bus/usb/devices/1-1.3",
        ),
        patch("src.lidar.serial_ports.os.listdir", return_value=["1-1.3", "1-1.4"]),
        patch("src.lidar.serial_ports._read_text", side_effect=read_text),
        patch("src.lidar.serial_ports.glob.glob", return_value=[]),
    ):
        assert shares_usb_hub_with_can("/dev/ttyUSB0")


def test_normalize_exclude_list():
    assert normalize_exclude_list("can0, ttyACM") == ["can0", "ttyACM"]
    assert normalize_exclude_list(["a", "b"]) == ["a", "b"]


def test_chip_filter_does_not_fallback_to_other_chips():
    fake_id = [
        "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
    ]

    def fake_glob(pat: str):
        if "by-id" in pat:
            return list(fake_id)
        return []

    with (
        patch("src.lidar.serial_ports.glob.glob", side_effect=fake_glob),
        patch("src.lidar.serial_ports.os.path.exists", return_value=True),
        patch(
            "src.lidar.serial_ports.os.path.realpath",
            side_effect=lambda p: "/dev/ttyUSB0" if "1a86" in p else p,
        ),
        patch("src.lidar.serial_ports._usb_vid_pid_for_port", return_value=None),
        patch("src.lidar.serial_ports._tty_driver_name", return_value="ch341"),
        patch("src.lidar.serial_ports._tty_bound_to_can_netdev", return_value=False),
    ):
        # Lidar asking for CP210 must not fall back to CH340.
        assert list_candidate_serial_ports(chip="cp210") == []
