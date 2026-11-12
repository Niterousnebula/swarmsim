"""Scripted ground defence platform and its projectiles.

The platform is deliberately simple game logic: pick the nearest visible
drone, turn towards it (optionally leading the target), and fire at a fixed
rate while aligned and in range.  Projectiles are point masses with gravity,
drag and wind; a pass within ``hit_radius`` of a drone counts as a hit.
"""
from __future__ import annotations

import numpy as np
import pybullet as pb

from ..models.specs import DefenceSpec
from .mathutil import wrap


class Projectiles:
    def __init__(self, capacity: int = 256):
        self.pos = np.zeros((capacity, 3)); self.vel = np.zeros((capacity, 3))
        self.age = np.zeros(capacity); self.active = np.zeros(capacity, bool)
        self.prev = np.zeros((capacity, 3))

    def clear(self):
        self.active[:] = False

    def fire(self, pos, vel):
        free = np.flatnonzero(~self.active)
        if len(free):
            i = free[0]
            self.pos[i] = pos; self.prev[i] = pos; self.vel[i] = vel; self.age[i] = 0; self.active[i] = True

    def step(self, dt, gravity, drag_k, lifetime, wind_fn):
        a = np.flatnonzero(self.active)
        if not len(a):
            return a
        v_rel = self.vel[a] - wind_fn(self.pos[a])
        acc = -drag_k * np.linalg.norm(v_rel, axis=1, keepdims=True) * v_rel
        acc[:, 2] -= gravity
        self.prev[a] = self.pos[a]
        self.vel[a] += acc * dt
        self.pos[a] += self.vel[a] * dt
        self.age[a] += dt
        self.active[a[self.age[a] > lifetime]] = False
        return np.flatnonzero(self.active)


class DefencePlatform:
    def __init__(self, client: int, urdf: str, spec: DefenceSpec, pos, rng):
        self.c, self.spec, self.rng = client, spec, rng
        self.base = np.asarray(pos, float)
        self.body = pb.loadURDF(urdf, self.base.tolist(), useFixedBase=True, physicsClientId=client)
        self.muzzle = self.base + np.array([0, 0, spec.muzzle_height])
        self.reset()

    def reset(self, pos=None):
        if pos is not None:
            self.base = np.asarray(pos, float)
            self.muzzle = self.base + np.array([0, 0, self.spec.muzzle_height])
        self.az = float(self.rng.uniform(-np.pi, np.pi))
        self.el = np.radians(25.0)
        self.cooldown = 0.0; self.track_time = 0.0; self.target = -1
        self.ammo = self.spec.ammo; self.shots = 0
        self._show()

    def _show(self):
        q = pb.getQuaternionFromEuler([0, 0, self.az])
        pb.resetBasePositionAndOrientation(self.body, self.base.tolist(), q, physicsClientId=self.c)

    def aim_dir(self):
        return np.array([np.cos(self.el) * np.cos(self.az), np.cos(self.el) * np.sin(self.az), np.sin(self.el)])

    # ------------------------------------------------------------------
    def _aim_point(self, p, v, gravity):
        """Where to point so a projectile meets a target at p moving with v."""
        s = self.spec
        if not s.lead_target:
            return p
        rel = p - self.muzzle
        a = v @ v - s.projectile_speed ** 2
        b = 2 * rel @ v
        c = rel @ rel
        disc = b * b - 4 * a * c
        if disc < 0 or abs(a) < 1e-9:
            return p
        t = (-b - np.sqrt(disc)) / (2 * a)
        if t <= 0:
            return p
        return p + v * t + np.array([0, 0, 0.5 * gravity * t * t])       # lead + drop compensation

    def step(self, dt, drone_pos, drone_vel, drone_alive, visible, gravity, projectiles: Projectiles):
        """``visible[i]``: line of sight from the muzzle to drone i (computed by the world)."""
        s = self.spec
        self.cooldown = max(0.0, self.cooldown - dt)
        rel = drone_pos - self.muzzle
        dist = np.linalg.norm(rel, axis=1)
        ok = drone_alive & visible & (dist < s.detection_range) & (dist > s.min_range)
        if not ok.any():
            self.target, self.track_time = -1, 0.0
            return False
        tgt = int(np.flatnonzero(ok)[np.argmin(dist[ok])])
        self.track_time = self.track_time + dt if tgt == self.target else 0.0
        self.target = tgt

        aim = self._aim_point(drone_pos[tgt], drone_vel[tgt], gravity) - self.muzzle
        az_d = np.arctan2(aim[1], aim[0])
        el_raw = np.arctan2(aim[2], np.hypot(aim[0], aim[1]))
        el_d = np.clip(el_raw, np.radians(s.elevation_min_deg), np.radians(s.elevation_max_deg))
        reachable = abs(el_raw - el_d) < 1e-6          # targets outside the elevation limits cannot be engaged
        step = np.radians(s.slew_rate_deg) * dt
        self.az = float(wrap(self.az + np.clip(wrap(az_d - self.az), -step, step)))
        self.el = float(self.el + np.clip(el_d - self.el, -step, step))
        self._show()

        aligned = (abs(wrap(az_d - self.az)) < np.radians(s.aim_tolerance_deg)
                   and abs(el_d - self.el) < np.radians(s.aim_tolerance_deg))
        if (aligned and reachable and self.cooldown <= 0 and self.ammo > 0 and dist[tgt] <= s.range_m
                and self.track_time >= s.reaction_time and s.fire_rate_hz > 0):
            d = self.aim_dir() + np.radians(s.dispersion_deg) * self.rng.standard_normal(3)
            d /= np.linalg.norm(d)
            projectiles.fire(self.muzzle + d * s.barrel_length, d * s.projectile_speed)
            self.cooldown = 1.0 / s.fire_rate_hz
            self.ammo -= 1; self.shots += 1
            return True
        return False
