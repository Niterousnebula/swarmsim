"""Small batched rotation helpers (quaternions are PyBullet order: x, y, z, w)."""
from __future__ import annotations

import numpy as np


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N, 4) -> (N, 3, 3) rotation matrices (body -> world)."""
    q = np.atleast_2d(q)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def yaw_to_quat(yaw: float, pitch_up: float = 0.0) -> list:
    """Quaternion for heading ``yaw`` (rad, CCW from +x) and nose-up pitch."""
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(-pitch_up / 2), np.sin(-pitch_up / 2)   # FLU: +y rotation is nose down
    # q = qz(yaw) * qy(pitch)
    return [float(-sy * sp), float(cy * sp), float(sy * cp), float(cy * cp)]


def euler_from_mat(R: np.ndarray):
    """Roll (right wing down +), nose-up pitch, yaw (CCW +) from (N,3,3)."""
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch_up = np.arcsin(np.clip(R[:, 2, 0], -1, 1))
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return roll, pitch_up, yaw


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def normalize(v, axis=-1, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def cross(a, b):
    """Row-wise cross product of (N, 3) arrays (much faster than np.cross for small N)."""
    return np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                     a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], axis=1)
