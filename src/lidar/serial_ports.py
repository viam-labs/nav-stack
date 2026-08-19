"""Helpers for finding likely RPLIDAR serial devices on Linux."""

from __future__ import annotations

import glob
from typing import List


def list_candidate_serial_ports() -> List[str]:
    """Return stable by-id paths first, then numbered ttyUSB/ttyACM."""
    seen: set[str] = set()
    out: List[str] = []
    for pattern in (
        "/dev/serial/by-id/usb-*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ):
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            out.append(path)
    return out
