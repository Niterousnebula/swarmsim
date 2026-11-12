"""Wind field: mean wind with height profile, turbulence, and building shelter."""
from __future__ import annotations

import numpy as np

from .config import WindConfig


class WindField:
    """``at(points)`` returns the air velocity (N, 3) in the world frame.

    * mean wind follows a power law with height above ground,
    * gusts are a sum of slowly drifting spatial waves whose amplitudes follow
      Ornstein-Uhlenbeck processes (smooth in space and in time),
    * points in the lee of a taller obstacle see a reduced wind.
    """
    N_MODES = 6

    def __init__(self, cfg: WindConfig, city=None, seed: int = 0):
        self.cfg, self.city = cfg, city
        self.reset(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c, K = self.cfg, self.N_MODES
        a = np.radians(c.direction_deg)
        self.dir = np.array([np.cos(a), np.sin(a), 0.0])
        ang = self.rng.uniform(0, 2 * np.pi, K)
        wavelength = c.gust_length_scale * self.rng.uniform(0.6, 1.8, K)
        self.k = (2 * np.pi / wavelength)[:, None] * np.stack(
            [np.cos(ang), np.sin(ang), self.rng.uniform(-0.3, 0.3, K)], 1)
        self.phase = self.rng.uniform(0, 2 * np.pi, K)
        d = self.rng.standard_normal((K, 3))
        d[:, 2] *= c.vertical_gust_ratio
        self.gust_dir = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.amp = self.rng.standard_normal(K)
        self.t = 0.0

    def step(self, dt: float):
        c = self.cfg
        theta = dt / max(c.gust_time_scale, 1e-3)
        self.amp += -theta * self.amp + np.sqrt(2 * theta) * self.rng.standard_normal(self.N_MODES)
        self.phase += dt * 0.3 * c.mean_speed * np.linalg.norm(self.k[:, :2], axis=1)   # gusts drift downwind
        self.t += dt

    def at(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, float))
        c = self.cfg
        ground = self.city.ground_z(pts[:, 0], pts[:, 1]) if self.city is not None else 0.0
        agl = np.maximum(pts[:, 2] - ground, 0.5)
        speed = c.mean_speed * np.minimum((agl / c.ref_height) ** c.shear_exponent, 2.0)
        w = speed[:, None] * self.dir[None]
        if c.gust_intensity > 0:
            wave = np.cos(pts @ self.k.T - self.phase[None]) * self.amp[None]          # (N, K)
            w = w + c.gust_intensity * np.sqrt(2.0 / self.N_MODES) * (wave @ self.gust_dir)
        if self.city is not None and c.building_shelter > 0:
            sheltered = np.zeros(len(pts), bool)
            for dist in (6.0, 15.0, 30.0):                                             # look upwind
                up = pts[:, :2] - dist * self.dir[None, :2]
                sheltered |= self.city.obstacle_z(up[:, 0], up[:, 1]) > pts[:, 2] + 0.2 * dist
            w = w * np.where(sheltered, 1.0 - c.building_shelter, 1.0)[:, None]
        return w
