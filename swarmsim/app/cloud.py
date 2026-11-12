"""Coloured point clouds of a city, sampled from its meshes and textures.

Points are scattered at random over every triangle (evenly by area), take the
colour of the texture at that spot, and have a little light-and-shade baked in
so walls facing the light read differently from walls facing away.

Each kind (ground, buildings, trees) has its own density in points per square
metre; on big maps all densities are scaled down to stay within ``BUDGET``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .meshes import _obj

BUDGET = 1_800_000                     # points for the whole city at most
DENSITY = {"ground": 2.5, "building": 9.0, "tree": 5.0}
LIGHT = np.array([0.36, -0.48, 0.8]) / np.linalg.norm([0.36, -0.48, 0.8])
POINT = np.dtype([("p", "<f4", 3), ("c", "u1", 4)])     # 16 bytes per point


def _kind(entry: dict) -> str:
    k = entry.get("kind", "")
    return k if k in DENSITY else "building"


def _texture(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _sample(tris: np.ndarray, count: int, tex: np.ndarray, rng) -> np.ndarray:
    """``count`` points on the triangles (n, 3, 5: x, y, z, u, v)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b[:, :3] - a[:, :3], c[:, :3] - a[:, :3]), axis=1)
    cum = np.cumsum(area)
    k = np.minimum(np.searchsorted(cum, rng.random(count) * cum[-1]), len(tris) - 1)
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)                 # uniform barycentric weights
    pts = w[:, 0, None] * a[k] + w[:, 1, None] * b[k] + w[:, 2, None] * c[k]
    h, wd = tex.shape[:2]
    u, v = np.mod(pts[:, 3], 1.0), np.mod(pts[:, 4], 1.0)             # textures repeat
    col = tex[((1.0 - v) * (h - 1)).astype(int), (u * (wd - 1)).astype(int)].astype(float)
    n = np.cross(b[k, :3] - a[k, :3], c[k, :3] - a[k, :3])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    shade = 0.72 + 0.40 * np.abs(n @ LIGHT)
    out = np.empty(count, POINT)
    out["p"] = pts[:, :3]
    out["c"][:, :3] = np.clip(col * shade[:, None], 0, 255).astype(np.uint8)
    out["c"][:, 3] = 255
    return out


@lru_cache(maxsize=4)
def _build(city_dir: str, stamp: float) -> dict:
    d = Path(city_dir)
    manifest = json.loads((d / "scene.json").read_text())
    meshes = []
    for e in manifest["meshes"]:
        t = _obj(d / e["file"]).reshape(-1, 3, 5).astype(np.float64)
        area = 0.5 * np.linalg.norm(np.cross(t[:, 1, :3] - t[:, 0, :3], t[:, 2, :3] - t[:, 0, :3]), axis=1)
        keep = area > 1e-6
        meshes.append((_kind(e), t[keep], float(area[keep].sum()), d / e["texture"]))
    want = sum(a * DENSITY[k] for k, _, a, _ in meshes)
    scale = min(1.0, BUDGET / max(want, 1.0))
    rng = np.random.default_rng(0)
    textures, parts = {}, {k: [] for k in DENSITY}
    for kind, tris, area, tex_path in meshes:
        count = int(area * DENSITY[kind] * scale)
        if count <= 0 or not len(tris):
            continue
        if tex_path not in textures:
            textures[tex_path] = _texture(tex_path)
        parts[kind].append(_sample(tris, count, textures[tex_path], rng))
    out = {}
    for kind, chunks in parts.items():
        if chunks:
            pts = np.concatenate(chunks)
            out[kind] = {"data": pts.tobytes(), "count": len(pts),
                         "spacing": float(1.0 / np.sqrt(DENSITY[kind] * scale))}
    return out


def city_cloud(city_dir: str | Path) -> dict:
    """{kind: {"data": bytes, "count": n, "spacing": metres between neighbouring points}}"""
    d = Path(city_dir)
    return _build(str(d.resolve()), (d / "scene.json").stat().st_mtime)
