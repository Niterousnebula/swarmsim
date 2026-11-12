"""OpenStreetMap as a reference dataset (Overpass API).

Three uses, chosen with ``terrain.osm.mode``:
  profile - measure a real neighbourhood and feed the statistics to the
            procedural generator (new city, real-world proportions)
  roads   - reuse the real road network
  full    - reuse roads, building footprints, parks and water

OSM data is (c) OpenStreetMap contributors, ODbL 1.0. Keep that credit with any
map you publish.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union

from .layout import polys_of
from .params import OSMParams, TerrainParams
from .roads import _clip
from .world import Road

ATTRIBUTION = "Map data (c) OpenStreetMap contributors, ODbL 1.0"

HIGHWAY_WIDTH = {
    "motorway": 22.0, "trunk": 20.0, "primary": 18.0, "secondary": 14.0, "tertiary": 11.0,
    "motorway_link": 9.0, "trunk_link": 9.0, "primary_link": 9.0, "secondary_link": 8.0,
    "residential": 8.0, "unclassified": 8.0, "living_street": 6.0, "service": 5.0, "pedestrian": 5.0,
}
ARTERIAL = {"motorway", "trunk", "primary", "secondary"}
GREEN_TAGS = {
    "leisure": {"park", "garden", "pitch", "golf_course", "nature_reserve"},
    "landuse": {"grass", "forest", "meadow", "recreation_ground", "village_green", "cemetery"},
    "natural": {"wood", "scrub", "grassland"},
}
ZONE_OF_LANDUSE = {"residential": "residential", "commercial": "midrise", "retail": "midrise",
                   "industrial": "industrial"}


@dataclass
class OSMData:
    size: float
    origin: Tuple[float, float]                    # lat, lon of the map centre
    roads: List[Road] = field(default_factory=list)
    buildings: List[Tuple[Polygon, Optional[float]]] = field(default_factory=list)
    parks: List[Polygon] = field(default_factory=list)
    water: List[Polygon] = field(default_factory=list)
    trees: List[Tuple[float, float]] = field(default_factory=list)
    landuse: List[Tuple[Polygon, str]] = field(default_factory=list)


# -------------------------------------------------------------------- download
def overpass_query(bbox: List[float], timeout_s: int = 90) -> str:
    s, w, n, e = bbox
    b = f"{s},{w},{n},{e}"
    return (f"[out:json][timeout:{timeout_s}];("
            f'way["highway"]({b});way["building"]({b});relation["building"]({b});'
            f'way["leisure"]({b});way["landuse"]({b});way["natural"]({b});'
            f'relation["natural"="water"]({b});way["waterway"]({b});node["natural"="tree"]({b});'
            f");out body geom;")


def load_or_fetch(p: OSMParams) -> dict:
    """Return Overpass JSON from ``json_path``, the cache, or the network (in that order)."""
    if p.json_path:
        with open(p.json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    key = hashlib.md5(json.dumps(p.bbox).encode()).hexdigest()[:12]
    cache = os.path.join(p.cache_dir, f"overpass_{key}.json")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as fh:
            return json.load(fh)
    import requests
    resp = requests.post(p.overpass_url, data={"data": overpass_query(p.bbox, p.timeout_s)},
                         timeout=p.timeout_s + 30, headers={"User-Agent": "swarmsim-terrain/0.1"})
    resp.raise_for_status()
    data = resp.json()
    os.makedirs(p.cache_dir, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


# --------------------------------------------------------------------- parsing
def _num(text) -> Optional[float]:
    if text is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", str(text).replace(",", "."))
    return float(m.group()) if m else None


def bbox_extent_m(bbox: List[float]) -> Tuple[float, float]:
    s, w, n, e = bbox
    lat0 = math.radians((s + n) / 2)
    return (e - w) * 111320.0 * math.cos(lat0), (n - s) * 110540.0


def parse(data: dict, bbox: List[float], size: float, floor_h: float = 3.2) -> OSMData:
    """Project Overpass JSON to local metres around the bbox centre and clip to the map square."""
    s, w, n, e = bbox
    lat0, lon0 = (s + n) / 2, (w + e) / 2
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    frame = box(-size / 2, -size / 2, size / 2, size / 2)
    out = OSMData(size=size, origin=(lat0, lon0))

    def proj(geometry) -> List[Tuple[float, float]]:
        return [((g["lon"] - lon0) * kx, (g["lat"] - lat0) * ky) for g in geometry if g]

    def add_area(pts, tags, height=None):
        if len(pts) < 4 or pts[0] != pts[-1]:
            return
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        for part in polys_of(poly.intersection(frame)):
            if part.area < 8.0:
                continue
            if "building" in tags:
                out.buildings.append((part, height))
            elif tags.get("natural") == "water" or tags.get("waterway") == "riverbank" \
                    or tags.get("landuse") in ("reservoir", "basin"):
                out.water.append(part)
            elif any(tags.get(k) in v for k, v in GREEN_TAGS.items()):
                out.parks.append(part)
            elif tags.get("landuse") in ZONE_OF_LANDUSE:
                out.landuse.append((part, ZONE_OF_LANDUSE[tags["landuse"]]))

    def building_height(tags) -> Optional[float]:
        h = _num(tags.get("height"))
        if h is None and _num(tags.get("building:levels")) is not None:
            h = _num(tags.get("building:levels")) * floor_h
        return h if h and 2.0 < h < 600.0 else None

    raw_roads: List[Road] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        kind = el.get("type")
        if kind == "node":
            if tags.get("natural") == "tree" and "lat" in el:
                x, y = (el["lon"] - lon0) * kx, (el["lat"] - lat0) * ky
                if abs(x) < size / 2 and abs(y) < size / 2:
                    out.trees.append((x, y))
            continue
        if kind == "way" and "geometry" in el:
            pts = proj(el["geometry"])
            hw = tags.get("highway")
            if hw in HIGHWAY_WIDTH and len(pts) >= 2 and tags.get("tunnel") != "yes":
                width = _num(tags.get("width"))
                if width is None and _num(tags.get("lanes")) is not None:
                    width = _num(tags.get("lanes")) * 3.5 + 1.0
                width = float(np.clip(width if width else HIGHWAY_WIDTH[hw], 4.0, 40.0))
                raw_roads.append(Road([list(p) for p in pts], width, "arterial" if hw in ARTERIAL else "street"))
            elif tags.get("waterway") in ("river", "stream", "canal") and len(pts) >= 2:
                wd = _num(tags.get("width")) or (14.0 if tags["waterway"] == "river" else 4.0)
                for part in polys_of(LineString(pts).buffer(wd / 2.0).intersection(frame)):
                    out.water.append(part)
            else:
                add_area(pts, tags, building_height(tags))
        elif kind == "relation":
            outers = [LineString(proj(m["geometry"])) for m in el.get("members", [])
                      if m.get("role") in ("outer", "") and len(m.get("geometry", [])) >= 2]
            inners = [LineString(proj(m["geometry"])) for m in el.get("members", [])
                      if m.get("role") == "inner" and len(m.get("geometry", [])) >= 2]
            if not outers:
                continue
            shape = unary_union(list(polygonize(unary_union(outers))))
            if inners:
                shape = shape.difference(unary_union(list(polygonize(unary_union(inners)))))
            for part in polys_of(shape):
                ring = list(part.exterior.coords)
                poly_tags = dict(tags)
                if part.interiors and "building" in tags:
                    for clipped in polys_of(part.intersection(frame)):
                        if clipped.area >= 8.0:
                            out.buildings.append((clipped, building_height(tags)))
                else:
                    add_area(ring, poly_tags, building_height(tags))
    out.roads = _clip(raw_roads, size)
    return out


# --------------------------------------------------------------- profile fitting
def fit_profile(osm: OSMData, params: TerrainParams) -> dict:
    """Measure the real area and return config overrides for the procedural generator."""
    area = osm.size ** 2
    over: dict = {"roads": {}, "zones": {}, "zoning": {}, "water": {}}
    report: dict = {}

    if osm.roads:
        lens, widths_a, widths_s, bearings, weights = 0.0, [], [], [], []
        for r in osm.roads:
            pts = np.asarray(r.pts)
            seg = np.diff(pts, axis=0)
            seg_len = np.linalg.norm(seg, axis=1)
            lens += float(seg_len.sum())
            (widths_a if r.kind == "arterial" else widths_s).append((r.width, float(seg_len.sum())))
            bearings.extend(np.degrees(np.arctan2(seg[:, 1], seg[:, 0])) % 90.0)
            weights.extend(seg_len)
        if lens > 0:
            over["roads"]["block_size_m"] = float(np.clip(2.0 * area / lens, 45.0, 260.0))
        for key, vals in (("arterial_width_m", widths_a), ("street_width_m", widths_s)):
            if vals:
                w_, l_ = np.array(vals).T
                over["roads"][key] = float(np.average(w_, weights=l_))
        if weights:
            hist, edges = np.histogram(bearings, bins=18, range=(0, 90), weights=weights)
            k = int(np.argmax(hist))
            over["roads"]["rotation_deg"] = float((edges[k] + edges[k + 1]) / 2)
            report["grid_regularity"] = float(hist[k] / max(hist.sum(), 1e-9))
        if widths_a and widths_s:
            la, ls = sum(v[1] for v in widths_a), sum(v[1] for v in widths_s)
            over["roads"]["arterial_every"] = int(np.clip(round((la + ls) / max(la, 1.0)), 2, 8))

    z = params.zoning
    half = osm.size / 2.0
    bands = {"downtown": (0.0, z.downtown_radius_frac), "midrise": (z.downtown_radius_frac, z.midrise_radius_frac),
             "residential": (z.midrise_radius_frac, 9.9)}
    for zone, (lo, hi) in bands.items():
        sel = [(p, h) for p, h in osm.buildings if lo <= math.hypot(p.centroid.x, p.centroid.y) / half < hi]
        if len(sel) < 15:
            continue
        prof = params.zones[zone]
        side = float(np.median([math.sqrt(p.area) for p, _ in sel]))
        lot = side / math.sqrt(prof.coverage) + 2 * prof.setback_m
        zo = {"lot_min_m": float(np.clip(lot * 0.8, 8, 120)), "lot_max_m": float(np.clip(lot * 1.25, 12, 160))}
        hs = np.array([h for _, h in sel if h])
        if len(hs) >= 8:
            zo["height_median_m"] = float(np.median(hs))
            zo["height_sigma"] = float(np.clip(np.std(np.log(hs)), 0.1, 0.8))
            zo["height_min_m"] = float(max(3.0, np.percentile(hs, 2)))
            zo["height_max_m"] = float(max(np.percentile(hs, 99), zo["height_median_m"] * 1.5))
        over["zones"][zone] = zo
        report[f"{zone}_buildings"] = len(sel)

    park_area = unary_union(osm.parks).area if osm.parks else 0.0
    over["zoning"]["park_fraction"] = float(np.clip(park_area / area, 0.0, 0.4))
    water_area = unary_union(osm.water).area if osm.water else 0.0
    over["water"]["river"] = bool(water_area / area > 0.015)
    report.update(roads=len(osm.roads), buildings=len(osm.buildings), tagged_trees=len(osm.trees),
                  park_fraction=over["zoning"]["park_fraction"], water_fraction=water_area / area)
    over = {k: v for k, v in over.items() if v}
    return {"overrides": over, "report": report}


# ------------------------------------------------------------------- test data
def world_to_overpass(world, origin=(28.6310, 77.2180)) -> Tuple[dict, List[float]]:
    """Write a generated world back out in Overpass JSON shape.

    Lets the whole OSM code path be exercised offline (tests, demos without a
    network connection). Returns (json, bbox).
    """
    lat0, lon0 = origin
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    geo = lambda pts: [{"lat": lat0 + y / ky, "lon": lon0 + x / kx} for x, y in pts]   # noqa: E731
    elements, nid = [], 1
    for r in world.roads:
        hw = {"arterial": "primary", "avenue": "secondary"}.get(r.kind, "residential")
        elements.append({"type": "way", "id": nid, "tags": {"highway": hw, "width": f"{r.width:.1f}"},
                         "geometry": geo(r.pts)})
        nid += 1
    for b in world.buildings:
        if b.stacked:
            continue
        ring_pts = b.footprint + [b.footprint[0]]
        elements.append({"type": "way", "id": nid, "geometry": geo(ring_pts),
                         "tags": {"building": "yes", "building:levels": str(int(round(b.height / 3.2)))}})
        nid += 1
    for blk in world.blocks:
        if blk.zone == "park":
            elements.append({"type": "way", "id": nid, "tags": {"leisure": "park"},
                             "geometry": geo(blk.outline + [blk.outline[0]])})
            nid += 1
    for poly in world.water:
        elements.append({"type": "way", "id": nid, "tags": {"natural": "water"},
                         "geometry": geo(poly + [poly[0]])})
        nid += 1
    for t in world.trees[::4]:
        elements.append({"type": "node", "id": nid, "lat": lat0 + t.y / ky, "lon": lon0 + t.x / kx,
                         "tags": {"natural": "tree"}})
        nid += 1
    h = world.half
    bbox = [lat0 - h / ky, lon0 - h / kx, lat0 + h / ky, lon0 + h / kx]
    return {"version": 0.6, "generator": "swarmsim offline fixture", "elements": elements}, bbox
