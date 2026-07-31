"""DDS discovery isolation for nav-stack's private ROS graph.

Cross-machine ``/map`` crosstalk is possible when multiple robots share
``ROS_DOMAIN_ID=0`` and subnet discovery. We default to:

* ``ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`` — do not multicast-discover peers
  on the LAN (Jazzy+ / Iron+).
* ``ROS_LOCALHOST_ONLY=1`` — older Humble-oriented equivalent.
* a stable non-zero ``ROS_DOMAIN_ID`` (1..101) when unset — hard isolation even
  if discovery range is widened or ignored by the RMW.

Explicit env values always win (setdefault only). The chosen domain id is
persisted so restarts and CLI debugging stay consistent.
"""
from __future__ import annotations

import hashlib
import os
import socket
from pathlib import Path
from typing import Dict, Optional, Union

_DOMAIN_MIN = 1
_DOMAIN_MAX = 101  # ROS 2 safe domain range on Linux (avoid ephemeral ports)
_PERSIST_NAME = "ros_domain_id"


def _machine_seed() -> str:
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return socket.gethostname() or "nav-stack"


def stable_domain_id(seed: Optional[str] = None) -> int:
    """Map a seed string to a ROS domain id in ``[1, 101]``."""
    raw = f"nav-stack:{seed if seed is not None else _machine_seed()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return _DOMAIN_MIN + (int(digest[:8], 16) % (_DOMAIN_MAX - _DOMAIN_MIN + 1))


def _read_persisted_domain(path: Path) -> Optional[int]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    if _DOMAIN_MIN <= value <= _DOMAIN_MAX:
        return value
    return None


def _write_persisted_domain(path: Path, domain_id: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{int(domain_id)}\n", encoding="utf-8")
    except OSError:
        pass


def resolve_persist_path(persist_path: Optional[Union[Path, str]] = None) -> Path:
    if persist_path is not None:
        return Path(persist_path)
    # Module root (parent of ``src/``) / ``.ros_domain_id``.
    return Path(__file__).resolve().parents[2] / f".{_PERSIST_NAME}"


def apply_dds_isolation(
    persist_path: Optional[Union[Path, str]] = None,
) -> Dict[str, str]:
    """Ensure isolation env vars are set on ``os.environ``; return the effective set.

    Safe to call multiple times. Does not override variables the operator already
    exported (including ``ROS_DOMAIN_ID=0`` if they really want the default domain).
    """
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
    # FASTDDS_BUILTIN_TRANSPORTS is deliberately not defaulted: it replaces the
    # participant's whole builtin transport set, conflicting with the one
    # ROS_LOCALHOST_ONLY installs. Operators opt in via the module env.

    if "ROS_DOMAIN_ID" not in os.environ or os.environ.get("ROS_DOMAIN_ID", "") == "":
        path = resolve_persist_path(persist_path)
        domain = _read_persisted_domain(path)
        if domain is None:
            domain = stable_domain_id()
            _write_persisted_domain(path, domain)
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    return dds_status()


def dds_status() -> Dict[str, str]:
    """Snapshot of DDS isolation settings for ``get_status`` / diagnostics."""
    return {
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "ros_automatic_discovery_range": os.environ.get(
            "ROS_AUTOMATIC_DISCOVERY_RANGE", ""
        ),
        "ros_localhost_only": os.environ.get("ROS_LOCALHOST_ONLY", ""),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", ""),
        "fastdds_builtin_transports": os.environ.get(
            "FASTDDS_BUILTIN_TRANSPORTS", ""
        ),
        "fastrtps_default_profiles_file": os.environ.get(
            "FASTRTPS_DEFAULT_PROFILES_FILE", ""
        ),
    }
