"""Tiny value-noise helpers (no external dependency)."""
from __future__ import annotations

import numpy as np


class ValueNoise:
    """Smooth, periodic 2-D value noise in [0, 1]."""

    def __init__(self, rng: np.random.Generator, n: int = 64):
        self.n = n
        self.grid = rng.random((n, n))

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        tx = x - x0
        ty = y - y0
        tx = tx * tx * (3 - 2 * tx)
        ty = ty * ty * (3 - 2 * ty)
        n = self.n
        g = self.grid
        a = g[y0 % n, x0 % n]
        b = g[y0 % n, (x0 + 1) % n]
        c = g[(y0 + 1) % n, x0 % n]
        d = g[(y0 + 1) % n, (x0 + 1) % n]
        return a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty

    def fbm(self, x, y, octaves: int = 3, gain: float = 0.5):
        """Fractal sum, still roughly in [0, 1]."""
        total = 0.0
        amp = 1.0
        norm = 0.0
        freq = 1.0
        for _ in range(max(1, octaves)):
            total = total + amp * self(np.asarray(x) * freq, np.asarray(y) * freq)
            norm += amp
            amp *= gain
            freq *= 2.0
        return total / norm
