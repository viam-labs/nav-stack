"""Named-locations store (CRUD), scoped to a single map.

A location is a named pose in the map frame, stored in meters/radians (ROS
convention) in the map's ``locations.json``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")


def validate_location_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid location name {name!r}: use 1-64 chars of letters, digits, "
            "space, dash or underscore"
        )
    return name


@dataclass
class Location:
    name: str
    x: float  # meters, map frame
    y: float  # meters, map frame
    theta: float = 0.0  # radians, map frame

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Location":
        return cls(
            name=d["name"],
            x=float(d["x"]),
            y=float(d["y"]),
            theta=float(d.get("theta", 0.0)),
        )


class LocationStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._locations: Dict[str, Location] = {}
        self._load()

    def _load(self) -> None:
        self._locations = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                raw = []
            for item in raw:
                loc = Location.from_dict(item)
                self._locations[loc.name] = loc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [loc.to_dict() for loc in self._locations.values()]
        self.path.write_text(json.dumps(data, indent=2))

    # -- CRUD ----------------------------------------------------------------
    def add(self, name: str, x: float, y: float, theta: float = 0.0) -> Location:
        name = validate_location_name(name)
        if name in self._locations:
            raise ValueError(f"location {name!r} already exists")
        loc = Location(name=name, x=float(x), y=float(y), theta=float(theta))
        self._locations[name] = loc
        self._save()
        return loc

    def get(self, name: str) -> Location:
        if name not in self._locations:
            raise KeyError(f"location {name!r} not found")
        return self._locations[name]

    def list(self) -> List[Location]:
        return list(self._locations.values())

    def update(
        self,
        name: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        theta: Optional[float] = None,
        new_name: Optional[str] = None,
    ) -> Location:
        loc = self.get(name)
        if x is not None:
            loc.x = float(x)
        if y is not None:
            loc.y = float(y)
        if theta is not None:
            loc.theta = float(theta)
        if new_name is not None and new_name != name:
            new_name = validate_location_name(new_name)
            if new_name in self._locations:
                raise ValueError(f"location {new_name!r} already exists")
            del self._locations[name]
            loc.name = new_name
            self._locations[new_name] = loc
        self._save()
        return loc

    def delete(self, name: str) -> None:
        if name not in self._locations:
            raise KeyError(f"location {name!r} not found")
        del self._locations[name]
        self._save()

    def delete_all(self) -> None:
        self._locations.clear()
        self._save()
