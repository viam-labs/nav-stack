"""Shared name validation for maps, locations, and zones."""
from __future__ import annotations

import re

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")


def validate_name(name: str, kind: str) -> str:
    """Strip and validate a user-supplied name; raises ValueError when invalid."""
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid {kind} name {name!r}: use 1-64 chars of letters, digits, "
            "space, dash or underscore"
        )
    return name
