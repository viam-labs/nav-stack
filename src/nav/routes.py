"""Named-routes store (CRUD), scoped to a single map.

A **route** is an ordered series of **waypoints**. Each waypoint is the name of
a location on the same map (see ``locations.py``). Persistence is
``routes.json`` next to ``locations.json`` / ``zones.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .naming import validate_name


def validate_route_name(name: str) -> str:
    return validate_name(name, "route")


@dataclass
class Route:
    name: str
    # Ordered location names on this map.
    waypoints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"name": self.name, "waypoints": list(self.waypoints)}

    @classmethod
    def from_dict(cls, d: Dict) -> "Route":
        wps = d.get("waypoints") or []
        if not isinstance(wps, list):
            raise ValueError("route waypoints must be a list of location names")
        return cls(
            name=str(d["name"]),
            waypoints=[str(w) for w in wps],
        )


class RouteStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._routes: Dict[str, Route] = {}
        self._load()

    def _load(self) -> None:
        self._routes = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                raw = []
            if not isinstance(raw, list):
                raw = []
            for item in raw:
                route = Route.from_dict(item)
                self._routes[route.name] = route

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [route.to_dict() for route in self._routes.values()]
        self.path.write_text(json.dumps(data, indent=2))

    # -- route CRUD ----------------------------------------------------------
    def add(
        self, name: str, waypoints: Optional[Sequence[str]] = None
    ) -> Route:
        name = validate_route_name(name)
        if name in self._routes:
            raise ValueError(f"route {name!r} already exists")
        wps = [str(w) for w in (waypoints or [])]
        route = Route(name=name, waypoints=wps)
        self._routes[name] = route
        self._save()
        return route

    def get(self, name: str) -> Route:
        if name not in self._routes:
            raise KeyError(f"route {name!r} not found")
        return self._routes[name]

    def list(self) -> List[Route]:
        return list(self._routes.values())

    def update(self, name: str, new_name: Optional[str] = None) -> Route:
        route = self.get(name)
        if new_name is not None and new_name != name:
            new_name = validate_route_name(new_name)
            if new_name in self._routes:
                raise ValueError(f"route {new_name!r} already exists")
            del self._routes[name]
            route.name = new_name
            self._routes[new_name] = route
        self._save()
        return route

    def delete(self, name: str) -> None:
        if name not in self._routes:
            raise KeyError(f"route {name!r} not found")
        del self._routes[name]
        self._save()

    def delete_all(self) -> None:
        self._routes.clear()
        self._save()

    # -- waypoint CRUD -------------------------------------------------------
    def list_waypoints(self, name: str) -> List[str]:
        return list(self.get(name).waypoints)

    def set_waypoints(self, name: str, waypoints: Sequence[str]) -> Route:
        """Replace the full waypoint list (order preserved)."""
        route = self.get(name)
        route.waypoints = [str(w) for w in waypoints]
        self._save()
        return route

    def clear_waypoints(self, name: str) -> Route:
        return self.set_waypoints(name, [])

    def add_waypoint(
        self,
        name: str,
        location: str,
        *,
        index: Optional[int] = None,
    ) -> Route:
        """Append ``location`` (or insert at ``index``)."""
        route = self.get(name)
        loc = str(location)
        if index is None:
            route.waypoints.append(loc)
        else:
            idx = int(index)
            if idx < 0 or idx > len(route.waypoints):
                raise ValueError(
                    f"waypoint index {idx} out of range "
                    f"(0..{len(route.waypoints)} for insert)"
                )
            route.waypoints.insert(idx, loc)
        self._save()
        return route

    def update_waypoint(
        self, name: str, index: int, location: str
    ) -> Route:
        """Replace the location name at ``index``."""
        route = self.get(name)
        idx = int(index)
        if idx < 0 or idx >= len(route.waypoints):
            raise ValueError(
                f"waypoint index {idx} out of range "
                f"(0..{len(route.waypoints) - 1})"
            )
        route.waypoints[idx] = str(location)
        self._save()
        return route

    def remove_waypoint(
        self,
        name: str,
        *,
        index: Optional[int] = None,
        location: Optional[str] = None,
    ) -> Route:
        """Remove by ``index``, or the first matching ``location`` name."""
        route = self.get(name)
        if index is not None:
            idx = int(index)
            if idx < 0 or idx >= len(route.waypoints):
                raise ValueError(
                    f"waypoint index {idx} out of range "
                    f"(0..{len(route.waypoints) - 1})"
                )
            del route.waypoints[idx]
        elif location is not None:
            loc = str(location)
            try:
                route.waypoints.remove(loc)
            except ValueError as exc:
                raise KeyError(
                    f"location {loc!r} is not a waypoint on route {name!r}"
                ) from exc
        else:
            raise ValueError("remove_waypoint requires index or location")
        self._save()
        return route

    def move_waypoint(
        self, name: str, from_index: int, to_index: int
    ) -> Route:
        """Reorder: take waypoint at ``from_index`` and insert at ``to_index``."""
        route = self.get(name)
        n = len(route.waypoints)
        src = int(from_index)
        dst = int(to_index)
        if src < 0 or src >= n:
            raise ValueError(
                f"from_index {src} out of range (0..{n - 1})"
            )
        if dst < 0 or dst >= n:
            raise ValueError(
                f"to_index {dst} out of range (0..{n - 1})"
            )
        loc = route.waypoints.pop(src)
        route.waypoints.insert(dst, loc)
        self._save()
        return route

    # -- location reference maintenance --------------------------------------
    def rename_location_refs(self, old: str, new: str) -> int:
        """Rewrite waypoint names after a location rename. Returns count."""
        changed = 0
        for route in self._routes.values():
            for i, wp in enumerate(route.waypoints):
                if wp == old:
                    route.waypoints[i] = new
                    changed += 1
        if changed:
            self._save()
        return changed

    def remove_location_refs(self, location: str) -> int:
        """Drop all waypoints that reference ``location``. Returns count."""
        changed = 0
        for route in self._routes.values():
            before = len(route.waypoints)
            route.waypoints = [w for w in route.waypoints if w != location]
            changed += before - len(route.waypoints)
        if changed:
            self._save()
        return changed
