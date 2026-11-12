"""From roads to blocks, zones, lots and buildings."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .params import TerrainParams, ZoneProfile
from .world import Building, Road


# --------------------------------------------------------------------- helpers
def polys_of(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and not g.is_empty]


def largest(geom) -> Optional[Polygon]:
    ps = polys_of(geom)
    return max(ps, key=lambda g: g.area) if ps else None


def ring(poly: Polygon) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in list(poly.exterior.coords)[:-1]]


def hole_rings(poly: Polygon) -> List[List[List[float]]]:
    return [[[float(x), float(y)] for x, y in list(i.coords)[:-1]] for i in poly.interiors]


def obb(poly: Polygon):
    """Oriented bounding box: centre, long direction, long length, short length, corners."""
    rect = poly.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return None
    c = np.array(rect.exterior.coords)[:4]
    e0, e1 = c[1] - c[0], c[2] - c[1]
    l0, l1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    if min(l0, l1) < 1e-6:
        return None
    if l0 >= l1:
        return c.mean(0), e0 / l0, l0, l1, c
    return c.mean(0), e1 / l1, l1, l0, c


# ------------------------------------------------------------------------ land
def road_surface(roads: List[Road]):
    return unary_union([LineString(r.pts).buffer(r.width / 2.0, cap_style="round", quad_segs=3) for r in roads])


def land_pieces(surface, water, size: float, min_area: float = 80.0) -> List[Polygon]:
    land = box(-size / 2, -size / 2, size / 2, size / 2).difference(surface)
    if water is not None and not water.is_empty:
        land = land.difference(water.buffer(5.0))
    return [p for p in polys_of(land) if p.area >= min_area]


def make_zoner(params: TerrainParams, centre: Tuple[float, float], rng: np.random.Generator) -> Callable[[Polygon], str]:
    z = params.zoning
    half = params.size_m / 2.0
    ind_angle = rng.uniform(0, 2 * math.pi)
    ind_halfwidth = math.pi * min(0.9, max(0.0, z.industrial_fraction))

    def zone_of(piece: Polygon) -> str:
        c = piece.centroid
        d = math.hypot(c.x - centre[0], c.y - centre[1]) / half
        if d > z.downtown_radius_frac * 0.6 and rng.random() < z.park_fraction:
            return "park"
        if d < z.downtown_radius_frac:
            return "downtown"
        if d < z.midrise_radius_frac:
            return "midrise"
        ang = math.atan2(c.y - centre[1], c.x - centre[0])
        diff = abs((ang - ind_angle + math.pi) % (2 * math.pi) - math.pi)
        if diff < ind_halfwidth:
            return "industrial"
        return "residential"

    return zone_of


# ------------------------------------------------------------------------ lots
def subdivide(poly: Polygon, target: float, rng: np.random.Generator, depth: int = 0) -> List[Polygon]:
    """Recursive oriented-bounding-box split until pieces are about ``target`` wide."""
    if poly.is_empty or poly.area < 30.0:
        return []
    info = obb(poly)
    if info is None:
        return []
    centre, d, long_len, short_len, _ = info
    if depth > 14 or long_len <= target * 1.45:
        return [poly]
    n = np.array([-d[1], d[0]])
    m = centre + d * long_len * rng.uniform(-0.12, 0.12)
    big = long_len + short_len + 10.0
    half_a = Polygon([m + n * big, m - n * big, m - n * big - d * big, m + n * big - d * big])
    half_b = Polygon([m + n * big, m - n * big, m - n * big + d * big, m + n * big + d * big])
    out: List[Polygon] = []
    for half in (half_a, half_b):
        for g in polys_of(poly.intersection(half)):
            out.extend(subdivide(g, target, rng, depth + 1))
    return out


def has_frontage(lot: Polygon, block_inner: Polygon) -> bool:
    return lot.buffer(0.25).intersection(block_inner.exterior).length > 4.0


# ------------------------------------------------------------------- buildings
FLAT_ROOF_COLORS = list(range(0, 8))
GABLE_ROOF_COLORS = list(range(8, 16))


def footprint_for_lot(lot: Polygon, prof: ZoneProfile, rng: np.random.Generator) -> Optional[Polygon]:
    inner = largest(lot.buffer(-prof.setback_m, join_style="mitre"))
    if inner is None or inner.area < 25.0:
        return None
    k = math.sqrt(min(1.0, max(0.05, prof.coverage)))
    rect = affinity.scale(inner.minimum_rotated_rectangle, k, k, origin=inner.centroid)
    fp = largest(rect.intersection(inner))
    if fp is None:
        return None
    fp = fp.simplify(0.3)
    if fp.geom_type != "Polygon" or fp.area < 20.0:
        return None
    coords = np.array(fp.exterior.coords)[:-1]
    if len(coords) == 4 and rng.random() < prof.l_shape_prob:
        e0, e1 = coords[1] - coords[0], coords[3] - coords[0]
        if min(np.linalg.norm(e0), np.linalg.norm(e1)) > 14.0:
            i = int(rng.integers(4))
            p0 = coords[i]
            a = (coords[(i + 1) % 4] - p0) * rng.uniform(0.3, 0.5)
            b = (coords[(i - 1) % 4] - p0) * rng.uniform(0.3, 0.5)
            cut = Polygon([p0 - 0.01 * (a + b), p0 + a, p0 + a + b, p0 + b])
            trimmed = largest(fp.difference(cut))
            if trimmed is not None and trimmed.area > 20.0:
                fp = trimmed
    return orient(fp, 1.0)


def sample_height(prof: ZoneProfile, dist_frac: float, downtown_frac: float,
                  floor_h: float, rng: np.random.Generator) -> float:
    h = prof.height_median_m * math.exp(prof.height_sigma * rng.normal())
    h *= 1.0 + prof.centre_boost * max(0.0, 1.0 - dist_frac / max(downtown_frac, 1e-6))
    h = float(np.clip(h, prof.height_min_m, prof.height_max_m))
    return max(1, round(h / floor_h)) * floor_h


def buildings_from_footprint(fp: Polygon, zone: str, prof: ZoneProfile, params: TerrainParams,
                             centre: Tuple[float, float], height_fn: Callable, rng: np.random.Generator,
                             height: Optional[float] = None) -> List[Building]:
    """Turn one footprint into one building (or a podium + tower pair)."""
    fp = orient(fp, 1.0)
    pts = np.array(fp.exterior.coords)[:-1]
    ground = height_fn(pts[:, 0], pts[:, 1])
    ground_z = float(np.max(ground))
    base_z = float(np.min(ground)) - 0.6
    floor_h = params.floor_height_m
    c = fp.centroid
    dist = math.hypot(c.x - centre[0], c.y - centre[1]) / (params.size_m / 2.0)
    if height is None:
        height = sample_height(prof, dist, params.zoning.downtown_radius_frac, floor_h, rng)
    else:
        height = max(1, round(height / floor_h)) * floor_h
    style = str(rng.choice(prof.styles)) if prof.styles else "office_concrete"

    is_rect = len(pts) == 4 and not fp.interiors
    roof = "flat"
    if prof.roof in ("gable", "mixed") and is_rect and height <= 12.5:
        if prof.roof == "gable" or rng.random() < 0.7:
            roof = "gable"
    roof_color = int(rng.choice(GABLE_ROOF_COLORS if roof == "gable" else FLAT_ROOF_COLORS))

    common = dict(zone=zone, style=style)
    if rng.random() < prof.tower_prob and height > 40.0 and fp.area > 450.0 and not fp.interiors:
        podium_h = float(rng.integers(3, 7)) * floor_h
        k = rng.uniform(0.6, 0.78)
        tower_fp = orient(affinity.scale(fp, k, k, origin=c), 1.0)
        return [
            Building(ring(fp), podium_h, ground_z, base_z, roof="flat", roof_color=roof_color, **common),
            Building(ring(tower_fp), height - podium_h, ground_z + podium_h, ground_z + podium_h,
                     roof="flat", roof_color=int(rng.choice(FLAT_ROOF_COLORS)), stacked=True, **common),
        ]
    return [Building(ring(fp), height, ground_z, base_z, roof=roof, roof_color=roof_color,
                     holes=hole_rings(fp), **common)]


def build_blocks(pieces: List[Polygon], zone_of: Callable[[Polygon], str], params: TerrainParams,
                 centre: Tuple[float, float], height_fn: Callable, rng: np.random.Generator):
    """Zone every land piece, cut it into lots and place buildings.

    Returns (zones, lots, buildings) where ``lots`` is a list of
    (polygon, zone, built) and ``zones`` is aligned with ``pieces``.
    """
    zones: List[str] = []
    lots: List[Tuple[Polygon, str, bool]] = []
    buildings: List[Building] = []
    sidewalk = params.roads.sidewalk_width_m
    for piece in pieces:
        zone = zone_of(piece)
        zones.append(zone)
        prof = params.zones.get(zone)
        if prof is None or zone == "park":
            continue
        for inner in polys_of(piece.buffer(-sidewalk, join_style="mitre")):
            target = rng.uniform(prof.lot_min_m, prof.lot_max_m)
            for lot in subdivide(inner, target, rng):
                p_build = prof.build_prob if has_frontage(lot, inner) else prof.interior_build_prob
                fp = footprint_for_lot(lot, prof, rng) if rng.random() < p_build else None
                if fp is None:
                    lots.append((lot, zone, False))
                    continue
                lots.append((lot, zone, True))
                buildings.extend(buildings_from_footprint(fp, zone, prof, params, centre, height_fn, rng))
    return zones, lots, buildings
