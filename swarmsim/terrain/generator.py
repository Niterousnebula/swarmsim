"""Module 1 entry point: parameters in, a ready-to-load scene folder out.

    world = generate_and_export(TerrainParams(seed=3), "worlds/demo")

The scene folder holds meshes, textures, ``world.json`` (roads, buildings,
trees as plain data) and ``scene.json`` (what the physics module should load).
Nothing here imports PyBullet.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import shapely
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from ..utils.cfg import apply_overrides, to_dict
from . import osm as osm_mod
from . import satellite as sat_mod
from .layout import (build_blocks, buildings_from_footprint, hole_rings, land_pieces, make_zoner,
                     polys_of, ring, road_surface)
from .mesh import export_meshes, rasterize_obstacles
from .noise import ValueNoise
from .params import TerrainParams
from .roads import generate_river, generate_roads
from .textures import (FACADE_STYLES, Palette, facade_texture, render_ground, roof_atlas,
                       vegetation_atlas)
from .vegetation import place_trees
from .world import Block, Lot, Tree, World


def _heightmap(params: TerrainParams, rng: np.random.Generator, water) -> np.ndarray:
    r = params.relief
    n = max(9, int(r.grid_n))
    half = params.size_m / 2.0
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)
    if r.amplitude_m <= 0:
        return np.zeros((n, n), dtype=np.float32)
    noise = ValueNoise(rng)
    h = noise.fbm(X / r.wavelength_m + 7.3, Y / r.wavelength_m + 2.1, r.octaves)
    h = (h - h.min()) / max(float(np.ptp(h)), 1e-9) * r.amplitude_m
    if water is not None and not water.is_empty:      # valley floor along the water
        d = shapely.distance(water, shapely.points(X.ravel(), Y.ravel())).reshape(n, n)
        t = np.clip(d / 110.0, 0.0, 1.0)
        h = h * (t * t * (3 - 2 * t))
    return h.astype(np.float32)


def generate_world(params: TerrainParams, verbose: bool = True) -> Tuple[World, Palette, Optional[Image.Image]]:
    """Run the full pipeline in memory. ``params`` is not modified."""
    params = copy.deepcopy(params)
    rng = np.random.default_rng(params.seed)
    t0 = time.time()
    log = (lambda *a: print("[terrain]", *a)) if verbose else (lambda *a: None)
    meta: dict = {"seed": params.seed, "attribution": []}

    # ---- reference data ---------------------------------------------------
    osm = None
    mode = params.osm.mode
    if mode not in ("off", "profile", "roads", "full"):
        raise ValueError(f"terrain.osm.mode must be off | profile | roads | full, got '{mode}'")
    if mode != "off":
        raw = osm_mod.load_or_fetch(params.osm)
        ex, ey = osm_mod.bbox_extent_m(params.osm.bbox)
        if mode in ("roads", "full"):
            params.size_m = float(min(params.size_m, ex, ey))
        osm = osm_mod.parse(raw, params.osm.bbox, min(ex, ey) if mode == "profile" else params.size_m,
                            params.floor_height_m)
        meta["attribution"].append(osm_mod.ATTRIBUTION)
        log(f"OSM: {len(osm.roads)} road segments, {len(osm.buildings)} buildings, mode={mode}")
        if mode == "profile":
            fit = osm_mod.fit_profile(osm, params)
            apply_overrides(params, fit["overrides"])
            meta["osm_profile"] = fit
            log("fitted profile:", json.dumps(fit["overrides"]))

    palette = Palette()
    sat_img = None
    density_scale = 1.0
    if params.satellite.enabled:
        sat_img = sat_mod.load_or_fetch(params.satellite, params.osm.bbox)
        palette = sat_mod.palette_from_image(sat_img, params.satellite.palette_k, rng, palette)
        veg = sat_mod.vegetation_fraction(sat_img)
        if params.satellite.drive_tree_density:
            density_scale = float(np.clip(veg / 0.22, 0.35, 2.2))
        meta["satellite"] = {"vegetation_fraction": veg, "tree_density_scale": density_scale}
        if not params.satellite.image_path:
            meta["attribution"].append(params.satellite.attribution)
        log(f"satellite: vegetation cover {veg:.0%}, tree density x{density_scale:.2f}")

    size = params.size_m
    use_osm_geo = mode in ("roads", "full")
    off = params.zoning.centre_offset_frac * size / 2.0
    centre = (0.0, 0.0) if use_osm_geo else tuple(float(v) for v in rng.uniform(-off, off, 2))

    # ---- water, relief, roads ---------------------------------------------
    if use_osm_geo:
        water = unary_union(osm.water) if osm.water else None
    else:
        water = generate_river(params.water, size, rng, centre)
    heightmap = _heightmap(params, rng, water)
    world = World(size, heightmap)
    world.centre = list(centre)

    if use_osm_geo:
        roads = osm.roads
    else:
        roads = generate_roads(params.roads, size, rng, centre)
        if water is not None and not water.is_empty:   # only wide roads get a bridge
            roads = [r for r in roads if r.kind != "street" or not LineString(r.pts).intersects(water)]
    if not roads:
        raise RuntimeError("No roads were produced; check the bbox / road parameters.")
    surface = road_surface(roads)
    pieces = land_pieces(surface, water, size)

    # ---- zoning, lots, buildings ------------------------------------------
    zone_radial = make_zoner(params, centre, rng)
    if use_osm_geo:
        park_union = unary_union(osm.parks) if osm.parks else None

        def zone_of(piece: Polygon) -> str:
            if park_union is not None and piece.intersection(park_union).area > 0.5 * piece.area:
                return "park"
            for poly, zone in osm.landuse:
                if poly.contains(piece.representative_point()):
                    return zone
            z = zone_radial(piece)
            return "midrise" if z == "park" else z
    else:
        zone_of = zone_radial

    if mode == "full":
        zones = [zone_of(p) for p in pieces]
        lots, buildings = [], []
        for fp, h in osm.buildings:
            fp = fp.simplify(0.4)
            if fp.geom_type != "Polygon" or fp.area < 12.0:
                continue
            probe = Polygon(fp.exterior)
            zone = zone_radial(probe)
            zone = "midrise" if zone == "park" else zone
            for poly, lz in osm.landuse:
                if poly.contains(fp.representative_point()):
                    zone = lz
                    break
            buildings.extend(buildings_from_footprint(fp, zone, params.zones[zone], params, centre,
                                                      world.height_at, rng, height=h))
    else:
        zones, lots, buildings = build_blocks(pieces, zone_of, params, centre, world.height_at, rng)

    # ---- vegetation --------------------------------------------------------
    parks = [p for p, z in zip(pieces, zones) if z == "park"]
    trees = place_trees(params, roads, surface, water, parks, lots, buildings, world.height_at, rng, density_scale)
    if mode == "full" and osm.trees and params.trees.enabled:
        for x, y in osm.trees:
            h = float(rng.uniform(params.trees.height_min_m, params.trees.height_max_m))
            trees.append(Tree(x, y, float(world.height_at(x, y)), h, h * params.trees.canopy_ratio,
                              "broadleaf", int(rng.integers(0, 8))))

    # ---- assemble ----------------------------------------------------------
    world.roads = roads
    world.blocks = [Block(ring(p), z, hole_rings(p)) for p, z in zip(pieces, zones)]
    world.lots = [Lot(ring(p), z, built) for p, z, built in lots]
    world.buildings = buildings
    world.trees = trees
    world.water = [ring(p) for p in polys_of(water)] if water is not None else []
    world.obstacle_height = rasterize_obstacles(world, world.obstacle_res)
    by_zone: dict = {}
    for b in buildings:
        by_zone[b.zone] = by_zone.get(b.zone, 0) + 1
    meta.update({
        "params": to_dict(params), "palette_source": palette.source,
        "counts": {"roads": len(roads), "blocks": len(pieces), "buildings": len(buildings),
                   "trees": len(trees), "buildings_by_zone": by_zone},
        "tallest_building_m": max((b.height + (b.ground_z - world.height_at(*b.footprint[0])) for b in buildings), default=0.0),
        "max_obstacle_z": world.max_obstacle_z(),
        "generation_seconds": round(time.time() - t0, 2),
    })
    world.meta = meta
    log(f"{len(roads)} road segments, {len(pieces)} blocks, {len(buildings)} buildings, "
        f"{len(trees)} trees in {meta['generation_seconds']} s")
    return world, palette, sat_img


def render_map(world: World, ground: Image.Image, px: int = 1400) -> Image.Image:
    """2-D preview: ground texture with building roofs and tree crowns on top."""
    img = ground.convert("RGB").resize((px, px), Image.LANCZOS)
    draw = ImageDraw.Draw(img, "RGBA")
    s = px / world.size
    to = lambda pts: [((x + world.half) * s, (world.half - y) * s) for x, y in pts]   # noqa: E731
    hmax = max((b.top_z for b in world.buildings), default=1.0)
    for t in world.trees:
        cx, cy = to([(t.x, t.y)])[0]
        r = t.canopy_r * s
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(50, 104, 54, 235))
    for b in sorted(world.buildings, key=lambda q: q.top_z):
        shade = int(120 + 120 * (b.top_z / hmax) ** 0.6)
        tint = (shade, int(shade * 0.78), int(shade * 0.7)) if b.roof == "gable" else (shade, shade, int(shade * 1.04))
        draw.polygon(to(b.footprint), fill=tuple(min(255, c) for c in tint) + (255,), outline=(40, 40, 44, 255))
    return img


def export_scene(world: World, params: TerrainParams, palette: Palette, out_dir: str,
                 underlay: Optional[Image.Image] = None, verbose: bool = True) -> str:
    """Write meshes, textures, world.json and scene.json into ``out_dir``."""
    rng = np.random.default_rng(params.seed + 1001)
    tex_dir = os.path.join(out_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    tp = params.textures

    ground = render_ground(world, params, palette, rng, underlay)
    ground.save(os.path.join(tex_dir, "ground.png"))
    for style in sorted({b.style for b in world.buildings if b.style in FACADE_STYLES} | {"office_concrete"}):
        facade_texture(style, tp.facade_px, tp.window_lit_fraction, rng).save(
            os.path.join(tex_dir, f"facade_{style}.png"))
    roof_atlas(palette, rng).save(os.path.join(tex_dir, "roofs.png"))
    vegetation_atlas(palette, rng).save(os.path.join(tex_dir, "vegetation.png"))

    entries = export_meshes(world, params.floor_height_m, out_dir, rng)
    world.save(out_dir)
    render_map(world, ground).save(os.path.join(out_dir, "preview_map.png"))
    manifest = {
        "size": world.size, "centre": world.centre, "meshes": entries,
        "attribution": world.meta.get("attribution", []),
        "counts": world.meta.get("counts", {}),
    }
    with open(os.path.join(out_dir, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    if verbose:
        faces = sum(e["faces"] for e in entries)
        print(f"[terrain] wrote {len(entries)} meshes ({faces} triangles) to {out_dir}")
    return out_dir


def generate_and_export(params: TerrainParams, out_dir: str, verbose: bool = True) -> World:
    world, palette, sat = generate_world(params, verbose)
    export_scene(world, params, palette, out_dir, sat, verbose)
    return world
