"""Local multi-map store.

Layout on disk::

    <maps_dir>/
        state.json                 # { "active_map": "<name>" }
        <map_name>/
            metadata.json          # name, timestamps, resolution, frame
            map.posegraph          # slam_toolbox serialized pose-graph (written by ROS)
            map.data               # slam_toolbox serialized data (written by ROS)
            map.yaml / map.pgm     # occupancy grid (optional, for export)
            locations.json         # named locations (scoped to this map)
            zones.json             # keepout / speed_limit zones (scoped to this map)

Locations and zones are intentionally scoped to a map: switching the active map
switches its locations and zones too.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")


def validate_map_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid map name {name!r}: use 1-64 chars of letters, digits, "
            "space, dash or underscore"
        )
    return name


@dataclass
class MapMetadata:
    name: str
    created_unix: float = field(default_factory=time.time)
    updated_unix: float = field(default_factory=time.time)
    resolution: float = 0.05  # meters/cell
    frame: str = "map"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MapMetadata":
        return cls(**{k: d[k] for k in d if k in cls.__annotations__})


class MapHandle:
    """Filesystem handle for a single map's directory."""

    def __init__(self, root: Path):
        self.root = root

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def posegraph_path(self) -> Path:
        # slam_toolbox serialization writes <stem>.posegraph and <stem>.data.
        return self.root / "map.posegraph"

    @property
    def serialization_stem(self) -> Path:
        return self.root / "map"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def locations_path(self) -> Path:
        return self.root / "locations.json"

    @property
    def zones_path(self) -> Path:
        return self.root / "zones.json"

    @property
    def occupancy_yaml_path(self) -> Path:
        return self.root / "map.yaml"

    def exists(self) -> bool:
        return self.root.is_dir()

    def has_serialized_map(self) -> bool:
        return self.posegraph_path.exists()

    def metadata(self) -> MapMetadata:
        if self.metadata_path.exists():
            return MapMetadata.from_dict(json.loads(self.metadata_path.read_text()))
        return MapMetadata(name=self.name)

    def write_metadata(self, meta: MapMetadata) -> None:
        meta.updated_unix = time.time()
        self.metadata_path.write_text(json.dumps(meta.to_dict(), indent=2))


class MapStore:
    def __init__(self, maps_dir: str):
        self.maps_dir = Path(maps_dir).expanduser()
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.maps_dir / "state.json"

    # -- state ---------------------------------------------------------------
    def _read_state(self) -> Dict:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text())
            except json.JSONDecodeError:
                pass
        return {}

    def _write_state(self, state: Dict) -> None:
        self._state_path.write_text(json.dumps(state, indent=2))

    # -- queries -------------------------------------------------------------
    def list_maps(self) -> List[Dict]:
        out: List[Dict] = []
        active = self.get_active_map_name()
        for child in sorted(self.maps_dir.iterdir()):
            if not child.is_dir():
                continue
            handle = MapHandle(child)
            meta = handle.metadata()
            out.append(
                {
                    **meta.to_dict(),
                    "active": child.name == active,
                    "has_map": handle.has_serialized_map(),
                }
            )
        return out

    def handle(self, name: str) -> MapHandle:
        return MapHandle(self.maps_dir / validate_map_name(name))

    def get_active_map_name(self) -> Optional[str]:
        return self._read_state().get("active_map")

    def active_handle(self) -> Optional[MapHandle]:
        name = self.get_active_map_name()
        if not name:
            return None
        handle = self.handle(name)
        return handle if handle.exists() else None

    # -- mutations -----------------------------------------------------------
    def create_map(self, name: str, resolution: float = 0.05) -> MapHandle:
        name = validate_map_name(name)
        handle = self.handle(name)
        if handle.exists():
            raise ValueError(f"map {name!r} already exists")
        handle.root.mkdir(parents=True)
        handle.write_metadata(MapMetadata(name=name, resolution=resolution))
        return handle

    def get_or_create_map(self, name: str, resolution: float = 0.05) -> MapHandle:
        handle = self.handle(name)
        if not handle.exists():
            return self.create_map(name, resolution=resolution)
        return handle

    def set_active_map(self, name: str) -> MapHandle:
        handle = self.handle(name)
        if not handle.exists():
            raise ValueError(f"map {name!r} does not exist")
        state = self._read_state()
        state["active_map"] = handle.name
        self._write_state(state)
        return handle

    def rename_map(self, old: str, new: str) -> MapHandle:
        old_handle = self.handle(old)
        if not old_handle.exists():
            raise ValueError(f"map {old!r} does not exist")
        new_name = validate_map_name(new)
        new_handle = self.handle(new_name)
        if new_handle.exists():
            raise ValueError(f"map {new_name!r} already exists")
        old_handle.root.rename(new_handle.root)
        meta = new_handle.metadata()
        meta.name = new_name
        new_handle.write_metadata(meta)
        state = self._read_state()
        if state.get("active_map") == old_handle.name:
            state["active_map"] = new_name
            self._write_state(state)
        return new_handle

    def delete_map(self, name: str) -> None:
        handle = self.handle(name)
        if not handle.exists():
            raise ValueError(f"map {name!r} does not exist")
        shutil.rmtree(handle.root)
        state = self._read_state()
        if state.get("active_map") == handle.name:
            state.pop("active_map", None)
            self._write_state(state)
