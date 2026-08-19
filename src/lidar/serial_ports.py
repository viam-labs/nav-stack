"""Helpers for finding likely RPLIDAR serial devices on Linux."""

from __future__ import annotations

import glob
from typing import List


def list_candidate_serial_ports() -> List[str]:
    """Return CP2102 by-id paths first, then other by-id, then ttyUSB/ttyACM."""
    seen: set[str] = set()
    out: List[str] = []

    def add(path: str) -> None:
        if path in seen:
            return
        seen.add(path)
        out.append(path)

    by_id = sorted(glob.glob("/dev/serial/by-id/usb-*"))
    for path in by_id:
        if "Silicon_Labs" in path or "CP210" in path:
            add(path)
    for path in by_id:
        add(path)
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        for path in sorted(glob.glob(pattern)):
            add(path)
    return out
