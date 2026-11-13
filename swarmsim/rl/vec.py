"""Stable-Baselines3 adapter: every drone in every world is one "environment"
of a VecEnv, so a single shared policy is trained on all of them (parameter
sharing).  Finished drones respawn immediately, which keeps the batch full."""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from ..config import SimConfig
from .core import SwarmCore


class SwarmVecEnv(VecEnv):
    def __init__(self, cfg: SimConfig, n_worlds: int = 1, seed: int = 0):
        self.cfg = cfg
        rl = cfg.rl
        self.pool = int(rl.train_cities)              # 0 = classic behaviour: simulator k always uses city k
        self.swapping = self.pool > n_worlds          # more cities than simulators: rotate through the pool
        self._swap_rng = np.random.default_rng(seed + 77)
        self._since = [0] * n_worlds
        first = [(rl.city_pool_offset + (int(self._swap_rng.integers(self.pool)) if self.swapping else k % self.pool))
                 if self.pool else k for k in range(n_worlds)]
        self.cores = [SwarmCore(cfg, seed=seed + 1000 * k, world_index=first[k]) for k in range(n_worlds)]
        self.cities_seen = set(first)
        c = self.cores[0]
        self.per_world = c.n
        super().__init__(n_worlds * c.n, c.observation_space, c.action_space)
        self._actions = None
        self._ep_ret = np.zeros(self.num_envs); self._ep_len = np.zeros(self.num_envs, int)
        self.outcomes = deque(maxlen=400)          # recent episode outcomes for logging

    def reset(self):
        self._ep_ret[:] = 0; self._ep_len[:] = 0
        return np.concatenate([c.reset() for c in self.cores]).astype(np.float32)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        n = self.per_world
        all_obs, all_rew, all_done, infos = [], [], [], []
        for k, core in enumerate(self.cores):
            obs, rew, term, trunc, outcome = core.step(self._actions[k * n:(k + 1) * n])
            done = term | trunc
            winfo = [{} for _ in range(n)]
            for i in np.flatnonzero(done):
                g = k * n + i
                winfo[i] = {"terminal_observation": obs[i].copy(), "TimeLimit.truncated": bool(trunc[i]),
                            "outcome": outcome[i],
                            "episode": {"r": float(self._ep_ret[g] + rew[i]), "l": int(self._ep_len[g] + 1)}}
                self.outcomes.append(outcome[i])
                core.respawn(i)
            if done.any():
                fresh = core.observe()
                obs[done] = fresh[done]
            self._since[k] += n
            if self.swapping and self._since[k] >= self.cfg.rl.city_swap_steps:
                for i in np.flatnonzero(~done):       # cut the running episodes short (counts as a time limit)
                    g = k * n + i
                    winfo[i] = {"terminal_observation": obs[i].copy(), "TimeLimit.truncated": True,
                                "episode": {"r": float(self._ep_ret[g] + rew[i]), "l": int(self._ep_len[g] + 1)}}
                done = np.ones(n, bool)
                obs = self.swap_city(k)
            self._ep_ret[k * n:(k + 1) * n] += rew; self._ep_len[k * n:(k + 1) * n] += 1
            self._ep_ret[k * n:(k + 1) * n][done] = 0; self._ep_len[k * n:(k + 1) * n][done] = 0
            all_obs.append(obs); all_rew.append(rew); all_done.append(done); infos += winfo
        return (np.concatenate(all_obs).astype(np.float32), np.concatenate(all_rew).astype(np.float32),
                np.concatenate(all_done), infos)

    def swap_city(self, k: int) -> np.ndarray:
        """Move simulator ``k`` to another city of the pool; returns the first observations there."""
        index = self.cfg.rl.city_pool_offset + int(self._swap_rng.integers(self.pool))
        self.cores[k].close()
        self.cores[k] = SwarmCore(self.cfg, seed=int(self._swap_rng.integers(1 << 30)), world_index=index)
        self.cities_seen.add(index)
        self._since[k] = 0
        return self.cores[k].reset()

    def close(self):
        for c in self.cores:
            c.close()

    # --- minimal VecEnv plumbing ---------------------------------------------
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.cores[0], attr_name, None)] * len(self._idx(indices))

    def set_attr(self, attr_name, value, indices=None):
        for c in self.cores:
            setattr(c, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.cores[0], method_name)(*args, **kwargs)] * len(self._idx(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._idx(indices))

    def _idx(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        return [indices] if isinstance(indices, int) else list(indices)
