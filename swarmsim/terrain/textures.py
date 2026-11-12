"""Procedural textures: building facades, roof and vegetation swatches, ground map."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import LineString, Polygon

from .params import TerrainParams
from .world import World

RGB = Tuple[int, int, int]


@dataclass
class Palette:
    """Every colour the generator uses. Can be re-derived from a satellite image."""
    asphalt: RGB = (62, 64, 68)
    sidewalk: RGB = (170, 168, 162)
    paved: RGB = (146, 143, 136)
    grass: RGB = (104, 134, 76)
    park: RGB = (86, 126, 64)
    dirt: RGB = (154, 136, 108)
    water: RGB = (66, 102, 130)
    lane: RGB = (226, 222, 204)
    roof_flat: List[RGB] = field(default_factory=lambda: [
        (150, 150, 148), (120, 120, 122), (188, 186, 180), (96, 98, 102),
        (206, 204, 198), (134, 128, 120), (164, 158, 150), (110, 114, 120)])
    roof_gable: List[RGB] = field(default_factory=lambda: [
        (150, 72, 54), (128, 62, 50), (104, 84, 72), (88, 90, 96),
        (164, 92, 66), (116, 100, 88), (70, 74, 82), (140, 80, 60)])
    tree_broadleaf: List[RGB] = field(default_factory=lambda: [
        (58, 110, 52), (70, 124, 58), (50, 98, 48), (84, 132, 60),
        (64, 116, 70), (92, 128, 54), (46, 104, 60), (76, 118, 48)])
    tree_conifer: List[RGB] = field(default_factory=lambda: [
        (34, 78, 50), (40, 86, 58), (30, 70, 46), (44, 92, 62)])
    trunk: List[RGB] = field(default_factory=lambda: [
        (92, 70, 50), (104, 80, 58), (80, 62, 46), (112, 90, 68)])
    source: str = "built-in"


# name -> wall colour, glass colour, bay width (m), window width/height as cell fractions, kind
FACADE_STYLES: Dict[str, dict] = {
    "glass_blue":       dict(wall=(52, 70, 92),    glass=(96, 150, 196),  bay_w=3.0, win=(0.90, 0.84), kind="curtain"),
    "glass_dark":       dict(wall=(34, 40, 48),    glass=(58, 84, 104),   bay_w=3.0, win=(0.90, 0.84), kind="curtain"),
    "office_concrete":  dict(wall=(168, 166, 160), glass=(74, 98, 120),   bay_w=3.6, win=(1.00, 0.46), kind="ribbon"),
    "office_white":     dict(wall=(226, 224, 218), glass=(84, 112, 138),  bay_w=3.6, win=(0.72, 0.52), kind="punched"),
    "brick_red":        dict(wall=(150, 78, 62),   glass=(60, 76, 92),    bay_w=3.2, win=(0.40, 0.50), kind="punched"),
    "brick_brown":      dict(wall=(122, 88, 66),   glass=(58, 72, 86),    bay_w=3.2, win=(0.42, 0.50), kind="punched"),
    "stucco_cream":     dict(wall=(226, 212, 182), glass=(70, 88, 106),   bay_w=3.4, win=(0.38, 0.46), kind="punched"),
    "stucco_pastel":    dict(wall=(200, 214, 206), glass=(68, 86, 104),   bay_w=3.4, win=(0.38, 0.46), kind="punched"),
    "house_siding":     dict(wall=(232, 228, 216), glass=(72, 90, 108),   bay_w=3.6, win=(0.34, 0.42), kind="punched"),
    "industrial_metal": dict(wall=(142, 152, 158), glass=(70, 88, 100),   bay_w=5.0, win=(0.70, 0.22), kind="industrial"),
}
BAYS_PER_TILE = 4
FLOORS_PER_TILE = 4


def _noise_layer(shape, rng, sigma):
    return rng.normal(0.0, sigma, shape)


def facade_texture(style: str, px: int, lit_fraction: float, rng: np.random.Generator) -> Image.Image:
    """One tile = 4 bays x 4 floors. Image top is the top of the tile."""
    s = FACADE_STYLES[style]
    wall = np.array(s["wall"], dtype=np.float64)
    glass = np.array(s["glass"], dtype=np.float64)
    arr = np.ones((px, px, 3)) * wall
    cell = px // BAYS_PER_TILE
    kind = s["kind"]

    if kind == "industrial":           # corrugated sheet
        stripes = (np.sin(np.arange(px) * 2 * np.pi / max(4, px // 40)) * 7.0)[None, :, None]
        arr += stripes
    if kind in ("punched", "ribbon") and "brick" in style:
        arr[::max(2, px // 48), :, :] -= 14       # mortar lines
    arr += _noise_layer((px, px, 1), rng, 4.0)

    ww, wh = s["win"]
    for fl in range(FLOORS_PER_TILE):              # fl = 0 is the top floor in image space
        y_floor_bottom = (fl + 1) * cell
        if kind == "industrial" and fl != 0:
            continue
        for bay in range(BAYS_PER_TILE):
            x0 = bay * cell + int(cell * (1 - ww) / 2)
            x1 = bay * cell + int(cell * (1 + ww) / 2)
            sill = 0.10 if kind == "curtain" else (0.62 if kind == "industrial" else 0.30)
            y1 = y_floor_bottom - int(cell * sill)
            y0 = max(fl * cell + 1, y1 - int(cell * wh))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            tint = glass * rng.uniform(0.78, 1.18)
            if rng.random() < lit_fraction and kind != "industrial":
                tint = np.array([238, 212, 150]) * rng.uniform(0.8, 1.0)
            grad = np.linspace(1.16, 0.86, y1 - y0)[:, None, None]      # sky reflection, brighter on top
            arr[y0:y1, x0:x1, :] = tint * grad
            if kind in ("punched", "industrial"):                     # frame
                arr[y0:y1, x0:x0 + 1, :] = wall * 0.7
                arr[y0:y1, x1 - 1:x1, :] = wall * 0.7
                arr[y1 - 1:y1, x0:x1, :] = wall * 0.6
            if kind in ("curtain", "ribbon"):                           # mullion
                mid = (x0 + x1) // 2
                arr[y0:y1, mid:mid + 1, :] = wall * 0.8
        arr[y_floor_bottom - 2:y_floor_bottom, :, :] *= 0.86            # floor line
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def tile_size_m(style: str, floor_h: float) -> Tuple[float, float]:
    return FACADE_STYLES[style]["bay_w"] * BAYS_PER_TILE, floor_h * FLOORS_PER_TILE


# ------------------------------------------------------------------- swatches
SWATCH_GRID = 4


def swatch_uv(index: int, fu: float, fv: float) -> Tuple[float, float]:
    """UV inside swatch ``index`` of a 4x4 atlas; fu, fv in [0, 1]. v counts from the image bottom."""
    col, row = index % SWATCH_GRID, index // SWATCH_GRID
    pad = 0.12
    u = (col + pad + (1 - 2 * pad) * fu) / SWATCH_GRID
    v = (row + pad + (1 - 2 * pad) * fv) / SWATCH_GRID
    return u, v


def _atlas(colors: List[RGB], px: int, rng: np.random.Generator, gradient: bool, sigma: float) -> Image.Image:
    cell = px // SWATCH_GRID
    arr = np.zeros((px, px, 3))
    for i in range(SWATCH_GRID * SWATCH_GRID):
        col, row = i % SWATCH_GRID, i // SWATCH_GRID
        base = np.array(colors[i % len(colors)], dtype=np.float64)
        tile = np.ones((cell, cell, 3)) * base
        if gradient:                                       # darker at the bottom, lighter on top
            tile *= np.linspace(1.22, 0.72, cell)[:, None, None]
        tile += rng.normal(0, sigma, (cell, cell, 1))
        y0 = px - (row + 1) * cell                         # row 0 sits at the image bottom
        arr[y0:y0 + cell, col * cell:(col + 1) * cell] = tile
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def roof_atlas(palette: Palette, rng: np.random.Generator, px: int = 256) -> Image.Image:
    return _atlas(list(palette.roof_flat[:8]) + list(palette.roof_gable[:8]), px, rng, False, 7.0)


def vegetation_atlas(palette: Palette, rng: np.random.Generator, px: int = 256) -> Image.Image:
    colors = list(palette.tree_broadleaf[:8]) + list(palette.tree_conifer[:4]) + list(palette.trunk[:4])
    return _atlas(colors, px, rng, True, 9.0)


# --------------------------------------------------------------------- ground
def render_ground(world: World, params: TerrainParams, palette: Palette,
                  rng: np.random.Generator, underlay: Image.Image | None = None) -> Image.Image:
    """Top-down ground texture. Image top = north."""
    W = params.textures.ground_px
    size, half = world.size, world.half
    scale = W / size

    def px(pts):
        return [((x + half) * scale, (half - y) * scale) for x, y in pts]

    img = Image.new("RGB", (W, W), palette.asphalt)
    draw = ImageDraw.Draw(img)

    for poly in world.water:                       # grassy bank first, then the water itself
        if len(poly) >= 3:
            bank = Polygon(poly).buffer(5.5)
            if bank.geom_type == "Polygon":
                draw.polygon(px(bank.exterior.coords), fill=palette.park)
    for poly in world.water:
        if len(poly) >= 3:
            draw.polygon(px(poly), fill=palette.water)

    ground_rgb = {"paved": palette.paved, "grass": palette.grass, "dirt": palette.dirt}
    sidewalk = params.roads.sidewalk_width_m
    for block in world.blocks:
        if len(block.outline) < 3:
            continue
        draw.polygon(px(block.outline), fill=palette.sidewalk)
        for hole in block.holes:
            if len(hole) >= 3:
                draw.polygon(px(hole), fill=palette.asphalt)
        prof = params.zones.get(block.zone)
        kind = "grass" if block.zone == "park" else (prof.ground if prof else "paved")
        fill = palette.park if block.zone == "park" else ground_rgb.get(kind, palette.paved)
        inner = Polygon(block.outline).buffer(-sidewalk)
        for part in ([inner] if inner.geom_type == "Polygon" else getattr(inner, "geoms", [])):
            if part.is_empty:
                continue
            draw.polygon(px(part.exterior.coords), fill=fill)
            if block.zone == "park":                     # a couple of footpaths
                minx, miny, maxx, maxy = part.bounds
                for _ in range(2):
                    a = (rng.uniform(minx, maxx), miny) if rng.random() < 0.5 else (minx, rng.uniform(miny, maxy))
                    b = (rng.uniform(minx, maxx), maxy) if rng.random() < 0.5 else (maxx, rng.uniform(miny, maxy))
                    path = LineString([a, b]).intersection(part)
                    for seg in ([path] if path.geom_type == "LineString" else getattr(path, "geoms", [])):
                        if seg.geom_type == "LineString" and seg.length > 4:
                            draw.line(px(seg.coords), fill=palette.sidewalk, width=max(2, int(2.2 * scale)))

    for lot in world.lots:
        prof = params.zones.get(lot.zone)
        base = np.array(ground_rgb.get(prof.ground if prof else "paved", palette.paved), dtype=np.float64)
        if not lot.built and prof is not None and prof.ground == "paved":
            base = np.array(palette.asphalt, dtype=np.float64) * 1.25      # parking lot
        shade = tuple(int(c) for c in np.clip(base * rng.uniform(0.9, 1.1), 0, 255))
        if len(lot.outline) >= 3:
            draw.polygon(px(lot.outline), fill=shade)

    if params.textures.lane_markings:
        dash, gap = 4.0, 6.0
        for r in world.roads:
            if r.kind == "street":
                continue
            line = LineString(r.pts)
            d = 0.0
            while d < line.length:
                a, b = line.interpolate(d), line.interpolate(min(d + dash, line.length))
                draw.line(px([(a.x, a.y), (b.x, b.y)]), fill=palette.lane, width=max(1, int(0.35 * scale)))
                d += dash + gap

    arr = np.asarray(img, dtype=np.float64)
    low = Image.fromarray((rng.random((48, 48)) * 255).astype(np.uint8)).resize((W, W), Image.BICUBIC)
    arr *= (0.9 + 0.2 * np.asarray(low, dtype=np.float64) / 255.0)[..., None]
    arr += rng.normal(0, 3.5, (W, W, 1))
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(0.6))

    alpha = float(np.clip(params.satellite.ground_underlay, 0.0, 1.0))
    if underlay is not None and alpha > 0:
        sat = underlay.convert("RGB").resize((W, W), Image.BICUBIC)
        out = Image.blend(out, sat, alpha)
    return out
