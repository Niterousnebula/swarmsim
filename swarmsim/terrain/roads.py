"""Procedural road networks: grid, organic (warped grid) and radial."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, box

from .noise import ValueNoise
from .params import RoadParams, WaterParams
from .world import Road


def _lattice(rng: np.random.Generator, half_extent: float, spacing: float, jitter: float) -> np.ndarray:
    pos = [-half_extent]
    while pos[-1] < half_extent:
        pos.append(pos[-1] + spacing * (1.0 + jitter * rng.uniform(-1, 1)))
    return np.array(pos)


def _clip(roads: List[Road], size: float) -> List[Road]:
    """Cut every road to the map square."""
    frame = box(-size / 2, -size / 2, size / 2, size / 2)
    out: List[Road] = []
    for r in roads:
        geom = LineString(r.pts).intersection(frame)
        if geom.is_empty:
            continue
        parts = [geom] if geom.geom_type == "LineString" else [g for g in getattr(geom, "geoms", []) if g.geom_type == "LineString"]
        for part in parts:
            if part.length > 2.0:
                out.append(Road([list(map(float, c)) for c in part.coords], r.width, r.kind))
    return out


def _grid(p: RoadParams, size: float, rng: np.random.Generator, organic: bool) -> List[Road]:
    half = size * 0.78                       # big enough to cover the map after rotation
    us = _lattice(rng, half, p.block_size_m, p.block_jitter)
    vs = _lattice(rng, half, p.block_size_m, p.block_jitter)
    ang = math.radians(p.rotation_deg)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    nodes = np.stack([uu, vv], axis=-1) @ rot.T
    nodes += rng.normal(0, p.block_size_m * 0.02, nodes.shape)
    if organic:
        n1, n2 = ValueNoise(rng), ValueNoise(rng)
        scale = 1.0 / (p.block_size_m * 3.2)
        dx = (n1.fbm(nodes[..., 0] * scale + 11, nodes[..., 1] * scale + 5, 2) - 0.5) * 2
        dy = (n2.fbm(nodes[..., 0] * scale + 3, nodes[..., 1] * scale + 17, 2) - 0.5) * 2
        nodes = nodes + np.stack([dx, dy], axis=-1) * p.organic_warp_m

    every = max(1, p.arterial_every)
    pu, pv = int(rng.integers(every)), int(rng.integers(every))
    roads: List[Road] = []
    nu, nv = len(us), len(vs)
    for j in range(nv):                       # roads running along u
        arterial = (j % every) == pv
        for i in range(nu - 1):
            if not arterial and rng.random() < p.removal_prob:
                continue
            roads.append(Road([nodes[i, j].tolist(), nodes[i + 1, j].tolist()],
                              p.arterial_width_m if arterial else p.street_width_m,
                              "arterial" if arterial else "street"))
    for i in range(nu):                       # roads running along v
        arterial = (i % every) == pu
        for j in range(nv - 1):
            if not arterial and rng.random() < p.removal_prob:
                continue
            roads.append(Road([nodes[i, j].tolist(), nodes[i, j + 1].tolist()],
                              p.arterial_width_m if arterial else p.street_width_m,
                              "arterial" if arterial else "street"))
    return roads


def _radial(p: RoadParams, size: float, rng: np.random.Generator, centre: Tuple[float, float]) -> List[Road]:
    cx, cy = centre
    radii = np.cumsum(p.block_size_m * (1.0 + p.block_jitter * rng.uniform(-1, 1, 64)))
    radii = radii[radii < size * 1.1]
    n = max(4, p.radial_spokes)
    theta0 = math.radians(p.rotation_deg)
    thetas = theta0 + (np.arange(n) + rng.uniform(-0.12, 0.12, n)) * 2 * math.pi / n
    every = max(1, p.arterial_every)
    roads: List[Road] = []

    def pt(r, th):
        return [cx + r * math.cos(th), cy + r * math.sin(th)]

    # primary spokes from the centre, secondary spokes start further out
    split_ring = 3
    for j, th in enumerate(thetas):
        arterial = (j % every) == 0
        w = p.arterial_width_m if arterial else p.street_width_m
        kind = "arterial" if arterial else "street"
        prev = 0.0 if arterial or j % 2 == 0 else radii[0]
        for r in radii:
            if r <= prev:
                continue
            if not arterial and rng.random() < p.removal_prob * 0.5:
                prev = r
                continue
            roads.append(Road([pt(prev, th), pt(r, th)], w, kind))
            prev = r
        th_mid = th + math.pi / n
        prev = radii[min(split_ring, len(radii) - 1)]
        for r in radii[split_ring + 1:]:
            if rng.random() >= p.removal_prob:
                roads.append(Road([pt(prev, th_mid), pt(r, th_mid)], p.street_width_m, "street"))
            prev = r
    # rings, drawn sector by sector
    for k, r in enumerate(radii):
        arterial = (k % every) == every - 1
        w = p.arterial_width_m if arterial else p.street_width_m
        for j in range(n):
            if not arterial and rng.random() < p.removal_prob:
                continue
            a0 = thetas[j]
            a1 = thetas[(j + 1) % n] + (2 * math.pi if j == n - 1 else 0.0)
            steps = max(3, int(r * (a1 - a0) / 18.0))
            arc = [pt(r, a0 + (a1 - a0) * s / steps) for s in range(steps + 1)]
            roads.append(Road(arc, w, "arterial" if arterial else "street"))
    return roads


def generate_roads(p: RoadParams, size: float, rng: np.random.Generator,
                   centre: Tuple[float, float] = (0.0, 0.0)) -> List[Road]:
    if p.pattern == "grid":
        roads = _grid(p, size, rng, organic=False)
    elif p.pattern == "organic":
        roads = _grid(p, size, rng, organic=True)
    elif p.pattern == "radial":
        roads = _radial(p, size, rng, centre)
    else:
        raise ValueError(f"Unknown road pattern '{p.pattern}' (grid | organic | radial)")

    for _ in range(max(0, p.diagonal_avenues)):
        ang = math.radians(p.rotation_deg + rng.uniform(28, 62) * rng.choice([-1, 1]))
        off = np.array([centre[0], centre[1]]) + rng.uniform(-0.12, 0.12, 2) * size
        d = np.array([math.cos(ang), math.sin(ang)]) * size * 1.5
        roads.append(Road([(off - d).tolist(), (off + d).tolist()], p.avenue_width_m, "avenue"))
    return _clip(roads, size)


def generate_river(p: WaterParams, size: float, rng: np.random.Generator,
                   centre: Tuple[float, float] = (0.0, 0.0)):
    """A meandering river that crosses the map but stays off the city centre."""
    if not p.river:
        return None
    t = np.linspace(-0.75, 0.75, 60)
    side = rng.choice([-1.0, 1.0])
    base_offset = side * rng.uniform(0.22, 0.36) * size
    phase = rng.uniform(0, 2 * math.pi)
    lateral = base_offset + p.meander * size * (np.sin(2 * math.pi * 0.9 * t + phase)
                                               + 0.35 * np.sin(2 * math.pi * 2.3 * t + 2 * phase))
    pts = np.stack([t * size, lateral], axis=1)
    ang = rng.uniform(0, 2 * math.pi)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    pts = pts @ rot.T + np.asarray(centre)
    frame = box(-size / 2, -size / 2, size / 2, size / 2)
    return LineString(pts).buffer(p.width_m / 2.0, cap_style="round").intersection(frame)
