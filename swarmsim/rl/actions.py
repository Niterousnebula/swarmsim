"""Action sets: how a policy output becomes a flight command.

Discrete mode exposes named primitives (hover, forward, up, yaw_left ...).
Which primitives exist, and how fast they move the drone, are hyperparameters
(``RLConfig.discrete_actions``, ``move_speed``, ``climb_speed``, ``yaw_rate``).

Continuous mode exposes a 4-vector in [-1, 1]: forward, left, up, yaw rate.

Directions are relative to the drone's heading, so "forward" is always where
the nose points.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .config import ALL_DISCRETE_ACTIONS, RLConfig

# name -> (forward, left, up, yaw) unit command
PRIMITIVES = {
    "hover": (0, 0, 0, 0), "forward": (1, 0, 0, 0), "back": (-1, 0, 0, 0),
    "left": (0, 1, 0, 0), "right": (0, -1, 0, 0), "up": (0, 0, 1, 0), "down": (0, 0, -1, 0),
    "yaw_left": (0, 0, 0, 1), "yaw_right": (0, 0, 0, -1),
}


class ActionSet:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg
        self.discrete = cfg.action_mode == "discrete"
        if self.discrete:
            unknown = set(cfg.discrete_actions) - set(ALL_DISCRETE_ACTIONS)
            if unknown:
                raise ValueError(f"Unknown discrete action(s) {sorted(unknown)}; choose from {ALL_DISCRETE_ACTIONS}")
            self.names = tuple(cfg.discrete_actions)
            self.table = np.array([PRIMITIVES[n] for n in self.names], float)
            self.space = spaces.Discrete(len(self.names))
        elif cfg.action_mode == "continuous":
            self.names = ("forward", "left", "up", "yaw")
            self.space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        else:
            raise ValueError("action_mode must be 'discrete' or 'continuous'")

    def index(self, name: str) -> int:
        return self.names.index(name)

    def unit(self, action) -> np.ndarray:
        """Policy output -> (forward, left, up, yaw) in [-1, 1]."""
        if self.discrete:
            return self.table[int(action)]
        return np.clip(np.asarray(action, float).reshape(4), -1, 1)

    def apply(self, world, i: int, action):
        """Send the command for drone ``i`` to the physics world."""
        self.command(world, i, self.unit(action))

    def command(self, world, i: int, unit):
        """Fly drone ``i`` with a (forward, left, up, yaw) command in [-1, 1]. Unlike a
        discrete action this can combine axes, which is what manual piloting needs."""
        c = self.cfg
        f, l, u, yw = np.clip(np.asarray(unit, float).reshape(4), -1, 1)
        if world.kind(i) == "quad":
            yaw = np.arctan2(world.R[i, 1, 0], world.R[i, 0, 0])
            cy, sy = np.cos(yaw), np.sin(yaw)
            v = np.array([cy * f - sy * l, sy * f + cy * l, 0.0]) * c.move_speed
            v[2] = u * c.climb_speed
            world.command_velocity(i, v, yw * c.yaw_rate)
        else:
            # Fixed-wing aircraft cannot hover or fly sideways: left/right and yaw become
            # turns, forward/back change airspeed, "hover" means fly straight and level.
            spec = world.models.fixedwing
            j = i - world.n_quads
            speed = float(np.clip(world.wings.cmd_speed[j] + f * c.fw_speed_step / max(c.policy_hz, 1e-6),
                                  spec.min_speed, spec.max_speed))
            climb = u * (spec.max_climb if u > 0 else spec.max_descent)   # full up = steepest climb
            world.command_fixedwing(i, turn_rate=(l + 0.5 * yw) * c.fw_turn_rate,
                                    climb_rate=climb, airspeed=speed)
