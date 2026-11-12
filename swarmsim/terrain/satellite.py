"""Open satellite imagery as a colour and vegetation reference.

Default source: Sentinel-2 cloudless 2016 by EOX (CC BY 4.0), roughly 10 m per
pixel. That is far too coarse to texture a single facade, and satellites look
straight down anyway, so the image is used for what it can actually tell us:

  * the colour palette of roofs, bare ground, vegetation and water
  * how green the area is (drives tree density)
  * optionally, a faint underlay blended into the ground texture

Any other image works too: set ``terrain.satellite.image_path`` to a local
file that covers the area (for example a Copernicus Browser export or an
OpenAerialMap tile).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from .params import SatelliteParams
from .textures import Palette


# -------------------------------------------------------------------- download
def _tile_xy(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def fetch_mosaic(bbox: List[float], zoom: int, tile_url: str, tile_px: int = 256) -> Image.Image:
    """Download XYZ (web-mercator) tiles covering bbox = [south, west, north, east] and crop."""
    import requests
    s, w, n, e = bbox
    x0, y0 = _tile_xy(n, w, zoom)
    x1, y1 = _tile_xy(s, e, zoom)
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1), int(y1)
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * tile_px, (ty1 - ty0 + 1) * tile_px))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            url = tile_url.format(z=zoom, x=tx, y=ty)
            resp = requests.get(url, timeout=30, headers={"User-Agent": "swarmsim-terrain/0.1"})
            resp.raise_for_status()
            tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
            mosaic.paste(tile, ((tx - tx0) * tile_px, (ty - ty0) * tile_px))
    crop = (int((x0 - tx0) * tile_px), int((y0 - ty0) * tile_px),
            max(int((x0 - tx0) * tile_px) + 2, int((x1 - tx0) * tile_px)),
            max(int((y0 - ty0) * tile_px) + 2, int((y1 - ty0) * tile_px)))
    return mosaic.crop(crop)


def load_or_fetch(p: SatelliteParams, default_bbox: List[float]) -> Image.Image:
    if p.image_path:
        return Image.open(p.image_path).convert("RGB")
    bbox = p.bbox or default_bbox
    key = hashlib.md5(json.dumps([bbox, p.zoom, p.tile_url]).encode()).hexdigest()[:12]
    cache = os.path.join(p.cache_dir, f"sat_{key}.png")
    if os.path.exists(cache):
        return Image.open(cache).convert("RGB")
    img = fetch_mosaic(bbox, p.zoom, p.tile_url)
    os.makedirs(p.cache_dir, exist_ok=True)
    img.save(cache)
    return img


# -------------------------------------------------------------------- analysis
def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 20):
    centres = x[rng.choice(len(x), k, replace=False)].astype(np.float64)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - centres[None]) ** 2).sum(-1)
        labels = d.argmin(1)
        for j in range(k):
            if np.any(labels == j):
                centres[j] = x[labels == j].mean(0)
    return centres, np.bincount(labels, minlength=k) / len(x)


def classify(rgb: np.ndarray) -> np.ndarray:
    """Per-colour class: 0 built/bare, 1 vegetation, 2 water."""
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    total = r + g + b + 1e-6
    exg = (2 * g - r - b) / total                    # excess-green index
    out = np.zeros(rgb.shape[:-1], dtype=int)
    out[exg > 0.06] = 1
    out[(b > r * 1.12) & (b >= g * 0.98) & (total < 420)] = 2
    return out


def vegetation_fraction(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB").resize((128, 128)))
    return float((classify(arr) == 1).mean())


def _spread(colors: List[np.ndarray], n: int, rng: np.random.Generator) -> List[Tuple[int, int, int]]:
    out = []
    for i in range(n):
        base = colors[i % len(colors)] * rng.uniform(0.88, 1.12)
        out.append(tuple(int(c) for c in np.clip(base, 0, 255)))
    return out


def palette_from_image(img: Image.Image, k: int, rng: np.random.Generator, base: Palette | None = None) -> Palette:
    """Re-colour the default palette with clusters found in the image."""
    pal = base or Palette()
    arr = np.asarray(img.convert("RGB").resize((96, 96))).reshape(-1, 3).astype(np.float64)
    centres, share = _kmeans(arr, min(k, len(arr)), rng)
    cls = classify(centres)
    veg = [c for c, t in zip(centres, cls) if t == 1]
    water = [c for c, t in zip(centres, cls) if t == 2]
    built = sorted([(s_, c) for c, t, s_ in zip(centres, cls, share) if t == 0], key=lambda q: -q[0])
    built_c = [c for _, c in built]
    if veg:
        v = np.mean(veg, axis=0)
        pal.grass = tuple(int(c) for c in np.clip(v * 1.12, 0, 255))
        pal.park = tuple(int(c) for c in np.clip(v, 0, 255))
        pal.tree_broadleaf = _spread([v * 0.78, v * 0.9, v * 0.7], 8, rng)
        pal.tree_conifer = _spread([v * 0.55, v * 0.62], 4, rng)
    if water:
        pal.water = tuple(int(c) for c in np.mean(water, axis=0))
    if built_c:
        greyness = lambda c: -float(np.ptp(c))        # noqa: E731
        greys = sorted(built_c, key=greyness)
        pal.roof_flat = _spread(greys[:max(1, len(greys) // 2 + 1)], 8, rng)
        warm = sorted(built_c, key=lambda c: -(c[0] - c[2]))
        pal.roof_gable = _spread(warm[:max(1, len(warm) // 2)], 8, rng)
        darkest = min(built_c, key=lambda c: c.sum())
        pal.asphalt = tuple(int(c) for c in np.clip(darkest * 0.8, 30, 110))
        pal.dirt = tuple(int(c) for c in np.clip(np.mean(warm[:2], axis=0), 0, 255))
    pal.source = "satellite"
    return pal
