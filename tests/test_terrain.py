import json

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import LineString, Polygon

from swarmsim.config import load_config
from swarmsim.terrain import TerrainParams, World, generate_and_export, generate_world
from swarmsim.terrain import osm, satellite
from swarmsim.terrain.layout import road_surface


def small(**kw):
    p = TerrainParams(size_m=300, **kw)
    p.textures.ground_px = 512
    return p


def test_same_seed_same_city_different_seed_different_city():
    a, _, _ = generate_world(small(seed=1), verbose=False)
    b, _, _ = generate_world(small(seed=1), verbose=False)
    c, _, _ = generate_world(small(seed=2), verbose=False)
    assert [x.footprint for x in a.buildings] == [x.footprint for x in b.buildings]
    assert [x.footprint for x in a.buildings] != [x.footprint for x in c.buildings]


@pytest.mark.parametrize("pattern", ["grid", "organic", "radial"])
def test_layout_is_consistent(pattern):
    p = small(seed=3)
    p.roads.pattern = pattern
    w, _, _ = generate_world(p, verbose=False)
    assert len(w.buildings) > 20 and len(w.trees) > 20 and len(w.roads) > 10
    roads = road_surface(w.roads)
    water = [Polygon(r) for r in w.water]
    for b in w.buildings:
        fp = Polygon(b.footprint)
        assert fp.is_valid and fp.area > 10
        assert fp.intersection(roads).area < 1.0            # no building on a road
        assert all(fp.intersection(wp).area < 1.0 for wp in water)
        assert b.height >= 3.0 and b.base_z <= b.ground_z
    floors = np.array([b.height for b in w.buildings]) / p.floor_height_m
    assert np.allclose(floors, np.round(floors))             # heights are whole floors
    for t in w.trees:                                        # trees stand on the ground, off the roads
        assert abs(t.z - w.height_at(t.x, t.y)) < 1e-3


def test_hyperparameters_change_the_city():
    base, _, _ = generate_world(small(seed=5), verbose=False)
    p = small(seed=5)
    p.zones["residential"].height_median_m = 30.0
    p.zones["residential"].height_max_m = 60.0
    p.zones["residential"].roof = "flat"
    tall, _, _ = generate_world(p, verbose=False)
    mean = lambda w: np.mean([b.height for b in w.buildings if b.zone == "residential"])   # noqa: E731
    assert mean(tall) > 2.5 * mean(base)
    p = small(seed=5)
    p.trees.enabled = False
    p.relief.amplitude_m = 0.0
    p.water.river = False
    flat, _, _ = generate_world(p, verbose=False)
    assert not flat.trees and not flat.water and float(np.ptp(flat.heightmap)) == 0.0
    p = small(seed=5)
    p.roads.block_size_m = 60.0
    dense, _, _ = generate_world(p, verbose=False)
    assert len(dense.roads) > 1.3 * len(base.roads)


def test_export_and_reload(tmp_path):
    w = generate_and_export(small(seed=6), str(tmp_path / "city"), verbose=False)
    manifest = json.loads((tmp_path / "city" / "scene.json").read_text())
    for e in manifest["meshes"]:
        assert (tmp_path / "city" / e["file"]).stat().st_size > 100
        assert (tmp_path / "city" / e["texture"]).exists()
    r = World.load(str(tmp_path / "city"))
    assert len(r.buildings) == len(w.buildings) and len(r.trees) == len(w.trees)
    b = max(r.buildings, key=lambda q: q.top_z)
    c = Polygon(b.footprint).representative_point()
    assert r.obstacle_top_at(c.x, c.y) == pytest.approx(b.top_z, abs=0.5)
    assert not r.is_free(c.x, c.y, b.top_z - 2.0) and r.is_free(c.x, c.y, r.max_height + 20.0)
    road = LineString(r.roads[0].pts).interpolate(0.5, normalized=True)
    assert r.obstacle_top_at(road.x, road.y) - r.height_at(road.x, road.y) < 20.0


def test_osm_pipeline_offline(tmp_path):
    """Round trip: a generated city written in Overpass JSON shape, read back through every OSM mode."""
    src, _, _ = generate_world(small(seed=8), verbose=False)
    data, bbox = osm.world_to_overpass(src)
    path = tmp_path / "area.json"
    path.write_text(json.dumps(data))
    parsed = osm.parse(data, bbox, 300.0)
    assert abs(len(parsed.buildings) - sum(not b.stacked for b in src.buildings)) <= 3
    assert len(parsed.roads) >= 0.9 * len(src.roads)
    fit = osm.fit_profile(parsed, small())
    assert 40 < fit["overrides"]["roads"]["block_size_m"] < 200
    assert abs(fit["overrides"]["roads"]["rotation_deg"] - 12.0) < 8.0     # recovers the grid angle
    for mode in ("profile", "roads", "full"):
        p = small(seed=9)
        p.osm.mode, p.osm.bbox, p.osm.json_path = mode, bbox, str(path)
        w, _, _ = generate_world(p, verbose=False)
        assert len(w.buildings) > 20
        if mode == "full":
            assert abs(len(w.buildings) - len(parsed.buildings)) <= 5


def test_satellite_palette(tmp_path):
    arr = np.zeros((64, 64, 3), np.uint8)
    arr[:, :32] = (60, 140, 50)      # vegetation half
    arr[:32, 32:] = (170, 80, 60)    # red roofs
    arr[32:, 32:] = (40, 60, 120)    # water
    img = tmp_path / "sat.png"
    Image.fromarray(arr).save(img)
    assert satellite.vegetation_fraction(Image.open(img)) == pytest.approx(0.5, abs=0.05)
    pal = satellite.palette_from_image(Image.open(img), 6, np.random.default_rng(0))
    assert pal.grass[1] > pal.grass[0] and pal.water[2] > pal.water[0]
    assert all(c[0] > c[2] for c in pal.roof_gable)
    p = small(seed=2)
    p.satellite.enabled, p.satellite.image_path = True, str(img)
    w, pal2, _ = generate_world(p, verbose=False)
    assert pal2.source == "satellite" and w.meta["satellite"]["tree_density_scale"] > 1.5
    assert satellite._tile_xy(0.0, 0.0, 1) == pytest.approx((1.0, 1.0))


def test_config_rejects_typos():
    with pytest.raises(KeyError):
        load_config(None, {"terrain": {"roads": {"blok_size_m": 50}}})
    cfg = load_config(None, {"terrain": {"zones": {"downtown": {"height_median_m": 80}}}})
    assert cfg.terrain.zones["downtown"].height_median_m == 80
    assert cfg.terrain.zones["downtown"].tower_prob == 0.45      # other defaults survive
