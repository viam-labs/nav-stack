import json

import pytest

from src.nav.annotations import (
    AnnotationStore,
    labels,
    no_go_polygons,
    slow_down_regions,
)


def _poly(kind, coords, **props):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"kind": kind, **props},
    }


def _point(label, xy):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": list(xy)},
        "properties": {"kind": "label", "label": label},
    }


def test_add_assigns_uuid_and_persists(tmp_path):
    p = tmp_path / "annotations.json"
    store = AnnotationStore(p)
    fid = store.add(_poly("no_go", [[0, 0], [1, 0], [1, 1], [0, 0]], label="wall"))
    assert fid and isinstance(fid, str)
    # persisted + reloadable
    fc = json.loads(p.read_text())
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 1
    assert AnnotationStore(p).feature_collection()["features"][0]["id"] == fid


def test_update_and_delete(tmp_path):
    store = AnnotationStore(tmp_path / "a.json")
    fid = store.add(_point("charger", (2.0, 3.0)))
    feat = store.feature_collection()["features"][0]
    feat["properties"]["label"] = "dock"
    assert store.update(feat) is True
    assert store.feature_collection()["features"][0]["properties"]["label"] == "dock"
    assert store.update({**feat, "id": "nope"}) is False
    assert store.delete(fid) is True
    assert store.delete(fid) is False
    assert store.feature_collection()["features"] == []


def test_set_all_bulk_replace_keeps_existing_ids(tmp_path):
    store = AnnotationStore(tmp_path / "a.json")
    kept = _poly("no_go", [[0, 0], [1, 0], [1, 1], [0, 0]])
    kept["id"] = "keep-me"
    ids = store.set_all({"type": "FeatureCollection", "features": [kept, _point("x", (1, 1))]})
    assert ids[0] == "keep-me"
    assert len(ids) == 2 and ids[1] != "keep-me"


def test_validation_rejects_bad_features(tmp_path):
    store = AnnotationStore(tmp_path / "a.json")
    with pytest.raises(ValueError):
        store.add({"type": "NotAFeature"})
    with pytest.raises(ValueError):
        store.add({"type": "Feature", "geometry": {"type": "LineString", "coordinates": []}, "properties": {}})


def test_derivations():
    fc = {
        "type": "FeatureCollection",
        "features": [
            _poly("no_go", [[0, 0], [2, 0], [2, 2], [0, 0]]),
            _poly("slow_down", [[3, 3], [4, 3], [4, 4], [3, 3]], max_speed_m_s=0.25),
            _point("charger", (5.0, 6.0)),
        ],
    }
    ngs = no_go_polygons(fc)
    assert len(ngs) == 1 and ngs[0]["type"] == "polygon"
    assert ngs[0]["points"][1] == [2.0, 0.0]

    sds = slow_down_regions(fc)
    assert len(sds) == 1 and sds[0][1] == pytest.approx(0.25)

    assert labels(fc) == {"charger": (5.0, 6.0)}


def test_unknown_kind_ignored_by_derivations():
    fc = {"type": "FeatureCollection", "features": [_poly("future_kind", [[0, 0], [1, 0], [1, 1], [0, 0]])]}
    assert no_go_polygons(fc) == []
    assert slow_down_regions(fc) == []
    assert labels(fc) == {}
