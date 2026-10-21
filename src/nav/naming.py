"""Shared name validation for maps, locations, zones, and routes."""
from __future__ import annotations

import re

# Maps / routes / zones stay filesystem-safe (no `.` / `@`).
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")
# Locations may include `.` and `@` (e.g. email-like or dotted labels).
_LOCATION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-@]{0,63}$")


def validate_name(name: str, kind: str) -> str:
    """Strip and validate a user-supplied name; raises ValueError when invalid."""
    name = (name or "").strip()
    pattern = _LOCATION_NAME_RE if kind == "location" else _NAME_RE
    if not pattern.match(name):
        allowed = (
            "letters, digits, space, dash, underscore, period or @"
            if kind == "location"
            else "letters, digits, space, dash or underscore"
        )
        raise ValueError(
            f"invalid {kind} name {name!r}: use 1-64 chars of {allowed}"
        )
    return name
