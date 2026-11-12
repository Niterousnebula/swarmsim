"""Tree placement: street trees, parks, yards."""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .noise import ValueNoise
from .params import TerrainParams
from .world import Building, Road, Tree


def _points_in(poly: Polygon, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or poly.is_empty:
        return np.zeros((0, 2))
    minx, miny, maxx, maxy = poly.bounds
    pts = np.column_stack([rng.uniform(minx, maxx, n * 3), rng.uniform(miny, maxy, n * 3)])
    keep = shapely.contains_xy(poly, pts[:, 0], pts[:, 1])
    return pts[keep][:n]


def place_trees(params: TerrainParams, roads: List[Road], surface, water,
                parks: List[Polygon], lots: List[Tuple[Polygon, str, bool]],
                buildings: List[Building], height_fn: Callable,
                rng: np.random.Generator, density_scale: float = 1.0) -> List[Tree]:
    tp = params.trees
    if not tp.enabled:
        return []
    candidates: List[Tuple[float, float, float]] = []   # x, y, size factor

    # --- street trees on the sidewalk strip
    off = params.roads.sidewalk_width_m * 0.5
    prob = min(1.0, tp.street_tree_prob * density_scale)
    for r in roads:
        line = LineString(r.pts)
        p_road = prob if r.kind != "street" else prob * 0.6
        for side in (-1.0, 1.0):
            curve = line.offset_curve(side * (r.width / 2.0 + off))
            for part in ([curve] if curve.geom_type == "LineString" else getattr(curve, "geoms", [])):
                if part.is_empty or part.length < tp.street_spacing_m:
                    continue
                dists = np.arange(tp.street_spacing_m * rng.uniform(0.2, 0.8), part.length, tp.street_spacing_m)
                for d in dists:
                    if rng.random() < p_road:
                        pt = part.interpolate(float(d))
                        candidates.append((pt.x, pt.y, 0.85))

    # --- parks: clustered with clearings
    noise = ValueNoise(rng)
    for park in parks:
        n = int(tp.park_density_per_ha * density_scale * park.area / 1e4)
        pts = _points_in(park.buffer(-3.0), n * 2, rng)
        if len(pts):
            keep = noise.fbm(pts[:, 0] / 45.0, pts[:, 1] / 45.0, 2) > 0.42
            for x, y in pts[keep][:n]:
                candidates.append((x, y, 1.15))

    # --- yards and empty lots
    for lot, zone, built in lots:
        prof = params.zones.get(zone)
        if prof is None:
            continue
        if built:
            k = int(rng.random() < prof.yard_tree_prob * density_scale) + int(rng.random() < prof.yard_tree_prob * 0.4)
        else:
            k = int(rng.integers(1, 4)) if prof.ground == "grass" else int(rng.random() < 0.3)
        for x, y in _points_in(lot.buffer(-1.5), k, rng):
            candidates.append((x, y, 0.9))

    if not candidates:
        return []
    arr = np.array(candidates)

    # --- drop anything on a road, in water or too close to a building
    blocked = [surface.buffer(0.4)]
    if water is not None and not water.is_empty:
        blocked.append(water.buffer(2.0))
    if buildings:
        blocked.append(unary_union([Polygon(b.footprint).buffer(2.2) for b in buildings if not b.stacked]))
    forbidden = unary_union(blocked)
    shapely.prepare(forbidden)
    ok = ~shapely.contains_xy(forbidden, arr[:, 0], arr[:, 1])
    half = params.size_m / 2.0 - 3.0
    ok &= (np.abs(arr[:, 0]) < half) & (np.abs(arr[:, 1]) < half)
    arr = arr[ok]
    if len(arr) > tp.max_trees:
        arr = arr[rng.choice(len(arr), tp.max_trees, replace=False)]

    trees: List[Tree] = []
    zs = height_fn(arr[:, 0], arr[:, 1]) if len(arr) else []
    for (x, y, f), z in zip(arr, zs):
        conifer = rng.random() < tp.conifer_fraction
        h = float(np.clip(rng.uniform(tp.height_min_m, tp.height_max_m) * f, 3.0, tp.height_max_m * 1.3))
        r = h * tp.canopy_ratio * rng.uniform(0.8, 1.2) * (0.6 if conifer else 1.0)
        trees.append(Tree(float(x), float(y), float(z), h, float(r),
                          "conifer" if conifer else "broadleaf", int(rng.integers(0, 8))))
    return trees
