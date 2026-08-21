import pytest

from src.nav.locations import LocationStore
from src.nav.maps import MapStore
from src.nav import zones as zmod
from src.nav.zones import ZoneStore


# -- locations ---------------------------------------------------------------
def test_location_crud(tmp_path):
    store = LocationStore(tmp_path / "locations.json")
    store.add("kitchen", 1.0, 2.0, 0.5)
    assert store.get("kitchen").x == 1.0
    with pytest.raises(ValueError):
        store.add("kitchen", 0, 0)

    store.update("kitchen", x=3.0, new_name="galley")
    assert store.get("galley").x == 3.0
    with pytest.raises(KeyError):
        store.get("kitchen")

    # Persistence across reload.
    reloaded = LocationStore(tmp_path / "locations.json")
    assert reloaded.get("galley").x == 3.0

    reloaded.delete("galley")
    assert reloaded.list() == []


def test_location_invalid_name(tmp_path):
    store = LocationStore(tmp_path / "loc.json")
    with pytest.raises(ValueError):
        store.add("bad/name", 0, 0)


# -- maps --------------------------------------------------------------------
def test_map_store_lifecycle(tmp_path):
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.set_active_map("floor1")
    assert store.get_active_map_name() == "floor1"

    store.get_or_create_map("floor2")
    names = {m["name"] for m in store.list_maps()}
    assert names == {"floor1", "floor2"}

    store.rename_map("floor1", "ground")
    assert store.get_active_map_name() == "ground"

    store.delete_map("ground")
    assert store.get_active_map_name() is None
    assert {m["name"] for m in store.list_maps()} == {"floor2"}


def test_map_duplicate_rejected(tmp_path):
    store = MapStore(str(tmp_path))
    store.create_map("a")
    with pytest.raises(ValueError):
        store.create_map("a")


# -- zones -------------------------------------------------------------------
def test_zone_crud_and_validation(tmp_path):
    store = ZoneStore(tmp_path / "zones.json")
    store.add("rug", zmod.KEEPOUT, {"type": "circle", "center": [0, 0], "radius": 1.0})
    store.add(
        "lobby",
        zmod.SPEED_LIMIT,
        {"type": "box", "center": [2, 2], "size": [1, 1]},
        speed_pct=30,
    )
    assert len(store.list()) == 2
    assert len(store.list(zmod.KEEPOUT)) == 1

    with pytest.raises(ValueError):
        store.add("nospeed", zmod.SPEED_LIMIT, {"type": "circle", "center": [0, 0], "radius": 1})

    with pytest.raises(ValueError):
        store.add("badtype", "lava", {"type": "circle", "center": [0, 0], "radius": 1})

    store.delete("rug")
    assert len(store.list()) == 1


def test_zone_rasterize_keepout():
    zones = [
        zmod.Zone("box", zmod.KEEPOUT, {"type": "box", "center": [0.5, 0.5], "size": [1.0, 1.0]})
    ]
    mask = zmod.rasterize_zones(zones, zmod.KEEPOUT, width=2, height=2, resolution=1.0,
                                origin_x=0.0, origin_y=0.0)
    assert mask.shape == (2, 2)
    # The box covers the cell whose center is (0.5, 0.5) -> index (0,0).
    assert mask[0, 0] == 100


def test_zone_rasterize_speed_most_restrictive():
    zones = [
        zmod.Zone("a", zmod.SPEED_LIMIT, {"type": "circle", "center": [0.5, 0.5], "radius": 5}, speed_pct=50),
        zmod.Zone("b", zmod.SPEED_LIMIT, {"type": "circle", "center": [0.5, 0.5], "radius": 5}, speed_pct=20),
    ]
    mask = zmod.rasterize_zones(zones, zmod.SPEED_LIMIT, width=1, height=1, resolution=1.0,
                               origin_x=0.0, origin_y=0.0)
    assert mask[0, 0] == 20


def test_zone_rasterize_polygon():
    poly = {"type": "polygon", "points": [[0, 0], [3, 0], [3, 3], [0, 3]]}
    zones = [zmod.Zone("p", zmod.KEEPOUT, poly)]
    mask = zmod.rasterize_zones(zones, zmod.KEEPOUT, width=3, height=3, resolution=1.0,
                               origin_x=0.0, origin_y=0.0)
    # All 9 cell centers (0.5/1.5/2.5) fall inside the 3x3 polygon.
    assert (mask == 100).sum() == 9
