"""GeoJSON-shaped annotation store: ``no_go`` / ``slow_down`` / ``label``.

Mirrors the RTAB-Map module's annotation schema (``docs/annotations.md`` there)
so a webapp built against it can CRUD annotations on nav-stack's SLAM service
unchanged. Persisted per-map as a GeoJSON ``FeatureCollection`` in **local
map-frame metres** (not WGS84). Each annotation is a ``Feature`` with a stable
``id`` (UUIDv4, server-assigned), a ``geometry``, and ``properties.kind`` (+
optional ``label``).

The navigation side derives what it needs from the collection: ``no_go`` polygons
-> keepout costmap mask + ``GetObstacles``; ``slow_down`` polygons (+
``max_speed_m_s``) -> speed mask; ``label`` points -> named goals for
``plan_to_label``.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

NO_GO = "no_go"
SLOW_DOWN = "slow_down"
LABEL = "label"
KINDS = {NO_GO, SLOW_DOWN, LABEL}

_GEOMETRY_TYPES = {"Polygon", "MultiPolygon", "Point"}


def _validate_feature(feature: Dict) -> Dict:
    if not isinstance(feature, dict) or feature.get("type") != "Feature":
        raise ValueError("annotation must be a GeoJSON Feature")
    geom = feature.get("geometry")
    if not isinstance(geom, dict) or "coordinates" not in geom:
        raise ValueError("Feature requires a geometry with type + coordinates")
    if geom.get("type") not in _GEOMETRY_TYPES:
        raise ValueError(f"unsupported geometry type {geom.get('type')!r}")
    if not isinstance(feature.get("properties") or {}, dict):
        raise ValueError("Feature.properties must be an object")
    return feature


class AnnotationStore:
    """Per-map GeoJSON FeatureCollection with UUID-keyed CRUD."""

    def __init__(self, path):
        self.path = Path(path)
        self._features: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        self._features = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            raw = {}
        # Degrade to "no annotations" on structurally-wrong JSON (hand-edit, wrong
        # schema, partial migration) rather than crashing every navigate for the map.
        if not isinstance(raw, dict):
            return
        for feat in raw.get("features") or []:
            if isinstance(feat, dict) and feat.get("id"):
                self._features[str(feat["id"])] = feat

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.feature_collection(), indent=2))

    def feature_collection(self) -> Dict:
        return {"type": "FeatureCollection", "features": list(self._features.values())}

    def add(self, feature: Dict) -> str:
        feature = _validate_feature(dict(feature))
        fid = str(feature.get("id") or uuid.uuid4())
        feature["id"] = fid
        self._features[fid] = feature
        self._save()
        return fid

    def update(self, feature: Dict) -> bool:
        feature = _validate_feature(dict(feature))
        fid = feature.get("id")
        if not fid or str(fid) not in self._features:
            return False
        feature["id"] = str(fid)
        self._features[str(fid)] = feature
        self._save()
        return True

    def delete(self, fid: str) -> bool:
        existed = str(fid) in self._features
        self._features.pop(str(fid), None)
        if existed:
            self._save()
        return existed

    def set_all(self, feature_collection: Dict) -> List[str]:
        """Bulk replace. Features with an existing id keep it; others get a
        fresh UUID. Returns the ids in input order."""
        new: Dict[str, Dict] = {}
        ids: List[str] = []
        for feat in (feature_collection or {}).get("features") or []:
            feat = _validate_feature(dict(feat))
            fid = str(feat.get("id") or uuid.uuid4())
            feat["id"] = fid
            new[fid] = feat
            ids.append(fid)
        self._features = new
        self._save()
        return ids


# ---------------------------------------------------------------------------
# Derivations for the navigation side
# ---------------------------------------------------------------------------
def _outer_ring_xy(geometry: Dict) -> Optional[List[List[float]]]:
    """Outer ring [[x, y], ...] for a Polygon / first-polygon of a MultiPolygon."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    try:
        if gtype == "Polygon":
            ring = coords[0]
        elif gtype == "MultiPolygon":
            ring = coords[0][0]
        else:
            return None
        return [[float(p[0]), float(p[1])] for p in ring]
    except (TypeError, IndexError, ValueError):
        return None


def _features_of_kind(fc: Dict, kind: str) -> List[Dict]:
    return [
        f for f in fc.get("features", []) if (f.get("properties") or {}).get("kind") == kind
    ]


def no_go_polygons(fc: Dict) -> List[Dict]:
    """``no_go`` features -> nav-stack polygon geometry dicts (for rasterization)."""
    out = []
    for f in _features_of_kind(fc, NO_GO):
        ring = _outer_ring_xy(f.get("geometry", {}))
        if ring and len(ring) >= 3:
            out.append({"type": "polygon", "points": ring})
    return out


def slow_down_regions(fc: Dict) -> List[Tuple[Dict, float]]:
    """``slow_down`` features -> ``(polygon geometry, max_speed_m_s)`` pairs."""
    out = []
    for f in _features_of_kind(fc, SLOW_DOWN):
        ring = _outer_ring_xy(f.get("geometry", {}))
        if not (ring and len(ring) >= 3):
            continue
        try:
            speed = float((f.get("properties") or {}).get("max_speed_m_s", 0.0))
        except (TypeError, ValueError):
            continue
        # A slow_down with no positive cap is meaningless (would map to a ~1%
        # near-keepout); skip it rather than crawl the robot.
        if speed > 0:
            out.append(({"type": "polygon", "points": ring}, speed))
    return out


def labels(fc: Dict) -> Dict[str, Tuple[float, float]]:
    """``label`` Point features -> ``{label: (x, y)}``."""
    out: Dict[str, Tuple[float, float]] = {}
    for f in _features_of_kind(fc, LABEL):
        geom = f.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        lbl = (f.get("properties") or {}).get("label")
        if not (lbl and coords):
            continue
        try:
            out[str(lbl)] = (float(coords[0]), float(coords[1]))
        except (TypeError, IndexError, ValueError):
            continue  # malformed Point coordinates -> skip this label
    return out
