"""Data model of a generated city. Pure data: no rendering, no physics."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np


@dataclass
class Road:
    pts: List[List[float]]
    width: float
    kind: str = "street"            # street | arterial | avenue


@dataclass
class Building:
    footprint: List[List[float]]    # exterior ring, counter-clockwise, no closing point
    height: float                   # wall height above ground_z
    ground_z: float = 0.0           # where the walls visibly start
    base_z: float = 0.0             # bottom of the mesh (slightly below ground on slopes)
    zone: str = "midrise"
    style: str = "office_concrete"
    roof: str = "flat"              # flat | gable
    roof_color: int = 0
    stacked: bool = False           # True for a tower sitting on a podium
    holes: List[List[List[float]]] = field(default_factory=list)

    @property
    def top_z(self) -> float:
        return self.ground_z + self.height


@dataclass
class Tree:
    x: float
    y: float
    z: float
    height: float
    canopy_r: float
    kind: str = "broadleaf"         # broadleaf | conifer
    color: int = 0


@dataclass
class Block:
    outline: List[List[float]]
    zone: str
    holes: List[List[List[float]]] = field(default_factory=list)


@dataclass
class Lot:
    outline: List[List[float]]
    zone: str
    built: bool


class World:
    """Everything the other modules need to know about the map."""

    def __init__(self, size: float, heightmap: np.ndarray):
        self.size = float(size)
        self.heightmap = np.asarray(heightmap, dtype=np.float32)   # [iy, ix]
        self.roads: List[Road] = []
        self.blocks: List[Block] = []
        self.lots: List[Lot] = []
        self.buildings: List[Building] = []
        self.trees: List[Tree] = []
        self.water: List[List[List[float]]] = []
        self.centre = [0.0, 0.0]
        self.meta: dict = {}
        self.obstacle_height: np.ndarray | None = None   # top of buildings/trees above z=0, 2 m grid
        self.obstacle_res = 2.0

    # ------------------------------------------------------------------ queries
    @property
    def half(self) -> float:
        return self.size / 2.0

    def height_at(self, x, y):
        """Ground elevation (bilinear). Accepts scalars or arrays."""
        n = self.heightmap.shape[0]
        fx = (np.asarray(x, dtype=np.float64) + self.half) / self.size * (n - 1)
        fy = (np.asarray(y, dtype=np.float64) + self.half) / self.size * (n - 1)
        fx = np.clip(fx, 0, n - 1 - 1e-9)
        fy = np.clip(fy, 0, n - 1 - 1e-9)
        ix = fx.astype(int)
        iy = fy.astype(int)
        tx = fx - ix
        ty = fy - iy
        h = self.heightmap
        out = (h[iy, ix] * (1 - tx) * (1 - ty) + h[iy, ix + 1] * tx * (1 - ty)
               + h[iy + 1, ix] * (1 - tx) * ty + h[iy + 1, ix + 1] * tx * ty)
        return float(out) if np.ndim(out) == 0 else out

    def obstacle_top_at(self, x, y):
        """Absolute z of the highest building/tree at (x, y); ground height if none."""
        g = self.height_at(x, y)
        if self.obstacle_height is None:
            return g
        n = self.obstacle_height.shape[0]
        ix = np.clip(((np.asarray(x) + self.half) / self.obstacle_res).astype(int), 0, n - 1)
        iy = np.clip(((np.asarray(y) + self.half) / self.obstacle_res).astype(int), 0, n - 1)
        top = np.maximum(self.obstacle_height[iy, ix], g)
        return float(top) if np.ndim(top) == 0 else top

    def is_free(self, x: float, y: float, z: float, margin: float = 3.0, clearance: float | None = None) -> bool:
        """True if the point is inside the map and clear of buildings and trees."""
        if clearance is not None:
            margin = clearance
        if abs(x) > self.half - margin or abs(y) > self.half - margin:
            return False
        offs = (-margin, 0.0, margin)
        tops = [self.obstacle_top_at(x + dx, y + dy) for dx in offs for dy in offs]
        return z > max(tops) + margin

    def max_obstacle_z(self) -> float:
        tops = [b.top_z for b in self.buildings] + [t.z + t.height for t in self.trees]
        return max(tops) if tops else float(self.heightmap.max())

    # short names used by the physics and RL modules
    ground_z = height_at
    obstacle_z = obstacle_top_at

    @property
    def max_height(self) -> float:
        return self.max_obstacle_z()

    # ---------------------------------------------------------------------- io
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "heightmap.npy"), self.heightmap)
        if self.obstacle_height is not None:
            np.save(os.path.join(out_dir, "obstacle_height.npy"), self.obstacle_height)
        data = {
            "size": self.size,
            "centre": self.centre,
            "obstacle_res": self.obstacle_res,
            "meta": self.meta,
            "roads": [asdict(r) for r in self.roads],
            "blocks": [asdict(b) for b in self.blocks],
            "lots": [asdict(lot) for lot in self.lots],
            "buildings": [asdict(b) for b in self.buildings],
            "trees": [asdict(t) for t in self.trees],
            "water": self.water,
        }
        with open(os.path.join(out_dir, "world.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, out_dir: str) -> "World":
        with open(os.path.join(out_dir, "world.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        world = cls(data["size"], np.load(os.path.join(out_dir, "heightmap.npy")))
        world.centre = data.get("centre", [0.0, 0.0])
        world.meta = data.get("meta", {})
        world.obstacle_res = data.get("obstacle_res", 2.0)
        world.roads = [Road(**r) for r in data["roads"]]
        world.blocks = [Block(**b) for b in data["blocks"]]
        world.lots = [Lot(**lot) for lot in data.get("lots", [])]
        world.buildings = [Building(**b) for b in data["buildings"]]
        world.trees = [Tree(**t) for t in data["trees"]]
        world.water = data.get("water", [])
        oh = os.path.join(out_dir, "obstacle_height.npy")
        if os.path.exists(oh):
            world.obstacle_height = np.load(oh)
        return world
