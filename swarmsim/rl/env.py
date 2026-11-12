"""PettingZoo parallel environment (the standard multi-agent RL interface)."""
from __future__ import annotations

import functools

import numpy as np
from pettingzoo import ParallelEnv

from ..config import SimConfig, load_config
from .core import SwarmCore


class SwarmEnv(ParallelEnv):
    """Each drone is an agent ("drone_0", "drone_1", ...).

    A drone leaves ``env.agents`` when it reaches its goal, crashes, is hit,
    leaves the map or runs out of time.  The episode ends when no drone is left.
    """
    metadata = {"name": "swarmsim_v0", "render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: SimConfig | str | None = None, render_mode: str | None = None,
                 seed: int = 0, world_index: int = 0):
        if not isinstance(cfg, SimConfig):
            cfg = load_config(cfg)
        self.cfg, self.render_mode = cfg, render_mode
        self.core = SwarmCore(cfg, seed=seed, gui=(render_mode == "human"), world_index=world_index)
        self.possible_agents = [f"drone_{i}" for i in range(self.core.n)]
        self._index = {a: i for i, a in enumerate(self.possible_agents)}
        self.agents = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.core.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.core.action_space

    def reset(self, seed=None, options=None):
        obs = self.core.reset(seed)
        self.agents = list(self.possible_agents)
        return ({a: obs[i] for a, i in self._index.items()}, {a: {} for a in self.agents})

    def step(self, actions):
        acts = [actions.get(a, 0) for a in self.possible_agents]
        obs, rew, term, trunc, outcome = self.core.step(acts)
        live = list(self.agents)
        out = ({a: obs[self._index[a]] for a in live},
               {a: float(rew[self._index[a]]) for a in live},
               {a: bool(term[self._index[a]]) for a in live},
               {a: bool(trunc[self._index[a]]) for a in live},
               {a: ({"outcome": outcome[self._index[a]]} if outcome[self._index[a]] else {}) for a in live})
        self.agents = [a for a in live if not (term[self._index[a]] or trunc[self._index[a]])]
        return out

    def render(self):
        if self.render_mode == "rgb_array":
            return self.core.world.render()
        return None

    def close(self):
        self.core.close()
