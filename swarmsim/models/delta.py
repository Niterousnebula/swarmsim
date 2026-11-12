"""A simple delta flying-wing UAV, built as a low-poly mesh straight from the spec.

No CAD package is needed: the shape is a handful of triangles scaled by the
wing span. Body axes as in the simulator: x forward, y left, z up, metres.

    nose ──►  /\
             /  \        swept leading edges, small tip fins,
            / ▲  \       a raised centre ridge and a pusher propeller
           /__|___\      disc at the trailing edge
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# Planform and thickness as fractions of the span (a typical small foam flying wing).
NOSE_X = 0.24
TIP_LE_X = -0.10
TIP_TE_X = -0.18
ROOT_TE_X = -0.15
RIDGE_X = 0.02
RIDGE_TOP, RIDGE_BOTTOM = 0.050, -0.025
ROOT_TOP, ROOT_BOTTOM = 0.016, -0.008
FIN_HEIGHT = 0.085
PROP_RADIUS = 0.07


def planform(span: float) -> dict:
    """Key dimensions (m) for collision shapes: root chord, its centre, and the tip box."""
    b = span
    return {"root_chord": (NOSE_X - ROOT_TE_X) * b, "root_centre_x": 0.5 * (NOSE_X + ROOT_TE_X) * b,
            "tip_chord": (TIP_LE_X - TIP_TE_X) * b, "wing_centre_x": 0.5 * (TIP_LE_X + ROOT_TE_X) * b,
            "thickness": (RIDGE_TOP - RIDGE_BOTTOM) * b}


def triangles(span: float) -> np.ndarray:
    """(n, 3, 3) triangle corners in metres."""
    b, h = span, span / 2
    N = (NOSE_X * b, 0, 0)
    tris = []
    for s in (1, -1):                                        # left half, then the mirror image
        T1, T2 = (TIP_LE_X * b, s * h, 0), (TIP_TE_X * b, s * h, 0)
        for C, R in (((RIDGE_X * b, 0, RIDGE_TOP * b), (ROOT_TE_X * b, 0, ROOT_TOP * b)),
                     ((RIDGE_X * b, 0, RIDGE_BOTTOM * b), (ROOT_TE_X * b, 0, ROOT_BOTTOM * b))):
            tris += [(N, T1, C), (C, T1, T2), (C, T2, R)]
        tris.append(((ROOT_TE_X * b, 0, ROOT_TOP * b), T2, (ROOT_TE_X * b, 0, ROOT_BOTTOM * b)))
        F = (TIP_TE_X * b - 0.03 * b, s * h, FIN_HEIGHT * b)  # tip fin
        tris.append(((TIP_LE_X * b + 0.02 * b, s * h, 0), T2, F))
    cx, r, k = ROOT_TE_X * b - 0.01 * b, PROP_RADIUS * b, 16  # pusher propeller disc
    ring = [(cx, r * np.cos(2 * np.pi * j / k), r * np.sin(2 * np.pi * j / k)) for j in range(k)]
    tris += [((cx, 0, 0), ring[j], ring[(j + 1) % k]) for j in range(k)]
    return np.asarray(tris, float)


def write_stl(span: float, path: str | Path, scale: float = 1000.0) -> Path:
    """Binary STL in millimetres (the URDFs load meshes with scale 0.001)."""
    t = triangles(span) * scale
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    out = bytearray(b"swarmsim delta flying wing".ljust(80, b" "))
    out += struct.pack("<I", len(t))
    for normal, tri in zip(n, t):
        out += struct.pack("<12fH", *normal, *tri.reshape(-1), 0)
    path = Path(path)
    if not path.exists() or path.read_bytes() != bytes(out):
        path.write_bytes(bytes(out))
    return path
