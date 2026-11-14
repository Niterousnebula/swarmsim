"""Environment logic shared by the PettingZoo env and the training wrapper:
spawning, observations, rewards and episode bookkeeping."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from gymnasium import spaces

from ..config import SimConfig
from ..physics import World
from ..terrain import generate_and_export
from .actions import ActionSet
from .wingpilot import WingPilot


def randomized_terrain(base, index: int):
    """City number ``index`` of the training distribution: same map size as ``base``, but its own
    street pattern, block size, density, building mix, relief and river. Deterministic per index."""
    import copy
    t = copy.deepcopy(base)
    rng = np.random.default_rng([int(base.seed), int(index)])
    t.seed = int(base.seed) + int(index)
    t.roads.pattern = str(rng.choice(["grid", "organic", "radial"]))
    t.roads.block_size_m = float(rng.uniform(70.0, 125.0))
    t.roads.rotation_deg = float(rng.uniform(0.0, 90.0))
    t.zoning.downtown_radius_frac = float(rng.choice([0.0, 0.0, 0.15, 0.3]))
    t.zoning.midrise_radius_frac = float(max(t.zoning.downtown_radius_frac + 0.1, rng.uniform(0.2, 0.7)))
    t.zoning.park_fraction = float(rng.uniform(0.03, 0.15))
    t.relief.amplitude_m = float(rng.uniform(0.0, 8.0))
    t.water.river = bool(rng.random() < 0.3)
    t.textures.ground_px = min(int(t.textures.ground_px), 512)      # training cities: small files
    return t


def ensure_city(cfg: SimConfig, seed_offset: int = 0) -> Path:
    """Return a city bundle for this config, building (and caching) it if needed."""
    if cfg.rl.city_dir:
        return Path(cfg.rl.city_dir)
    tcfg = replace(cfg.terrain, seed=cfg.terrain.seed + seed_offset)
    if cfg.rl.randomize_cities:
        tcfg = randomized_terrain(cfg.terrain, seed_offset)
    key = hashlib.md5(json.dumps(asdict(tcfg), sort_keys=True, default=str).encode()).hexdigest()[:10]
    out = Path(cfg.rl.cache_dir) / f"city_{key}"
    if not (out / "scene.json").exists():
        generate_and_export(tcfg, str(out), verbose=False)
    return out


class SwarmCore:
    """One simulated world with ``n`` drones.

    ``step(actions)`` advances by one policy step and returns per-drone arrays.
    Drones that finish (goal, crash, hit, out of bounds, time-out) stay parked
    until ``respawn(i)`` or ``reset()`` is called.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0, gui: bool = False, world_index: int = 0):
        self.cfg, self.rl = cfg, cfg.rl
        rl = self.rl
        self.n = rl.n_quads + rl.n_fixedwing
        self.rng = np.random.default_rng(seed)
        self.world = World(ensure_city(cfg, world_index), cfg.models, cfg.physics,
                           n_quads=rl.n_quads, n_fixedwing=rl.n_fixedwing,
                           defence_positions=[(0, 0)] * rl.n_defence, gui=gui, seed=seed)
        self.city = self.world.city
        self.actions = ActionSet(rl)
        self.substeps = max(1, int(round(cfg.physics.physics_hz / rl.policy_hz)))
        self.on_substep = None            # optional callback after every physics step (live viewer)
        self.max_steps = int(rl.max_episode_s * rl.policy_hz)

        # ---- observation layout --------------------------------------------
        parts = [("velocity", 3), ("attitude", 4), ("altitude", 1), ("goal", 4),
                 ("rays", rl.n_rays + 2), ("neighbors", 7 * rl.k_neighbors)]
        if rl.observe_defence:
            parts.append(("defence", 4))
        if rl.observe_projectiles:
            parts.append(("projectile", 7))
        parts.append(("type", 1))
        self.obs_layout, k = {}, 0
        for name, size in parts:
            self.obs_layout[name] = slice(k, k + size); k += size
        self.obs_dim = k
        self.observation_space = spaces.Box(-5.0, 5.0, (self.obs_dim,), np.float32)
        self.action_space = self.actions.space
        ang = np.linspace(0, 2 * np.pi, rl.n_rays, endpoint=False)
        self._ray_dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros(rl.n_rays)], 1)

        # ---- per-drone state --------------------------------------------------
        self.goal = np.zeros((self.n, 3))
        self.rays = np.full((self.n, rl.n_rays + 2), rl.ray_range)   # last range-finder readings (m)
        self.reflex_count = 0
        self.prev_dist = np.zeros(self.n)
        self.steps = np.zeros(self.n, int)
        self.active = np.zeros(self.n, bool)
        self.min_ray = np.full(self.n, rl.ray_range)
        self.visited = None
        self.t_relocate = 0.0
        self.manual = {}      # drone index -> (forward, left, up, yaw) flown instead of the policy's action
        self.start_at = {}    # drone index -> fixed start point [x, y, z]; others start at random points
        self.goal_at = {}     # drone index -> fixed goal point [x, y, z]; others get random goals
        self.defence_at = {}  # platform index -> fixed position [x, y] or [x, y, z]; others placed at random
        self.stats = {"goal": 0, "crash": 0, "hit": 0, "out_of_bounds": 0, "timeout": 0}
        if rl.fw_mode not in ("search", "policy"):
            raise ValueError("rl.fw_mode must be 'search' or 'policy'")
        self.wingpilot = WingPilot(self) if rl.fw_mode == "search" and rl.n_fixedwing else None
        # Planes must fly above the tallest roof. On maps whose towers reach the flight ceiling
        # they may go up to 30 m above their cruise band instead of ending as out of bounds.
        self.is_wing = np.array([self.world.kind(i) == "fixedwing" for i in range(self.n)])
        self.wing_top = self.city.max_height + max(rl.fw_altitude) + 30.0

    # ------------------------------------------------------------------ spawn
    def _sample_point(self, agl, min_sep=6.0, far_from=None, min_dist=0.0, high=False, max_dist=0.0, near=0.0):
        city, half = self.city, self.city.half
        margin = 20.0
        for _ in range(300):
            x, y = self.rng.uniform(-half + margin, half - margin, 2)
            if near > 0.0 and self.world.defences:          # somewhere around one of the platforms
                d = self.world.defences[int(self.rng.integers(len(self.world.defences)))]
                r, a = near * np.sqrt(self.rng.random()), self.rng.uniform(0.0, 2.0 * np.pi)
                x = float(np.clip(d.muzzle[0] + r * np.cos(a), -half + margin, half - margin))
                y = float(np.clip(d.muzzle[1] + r * np.sin(a), -half + margin, half - margin))
            if high:      # fixed-wing: above every roof
                z = city.max_height + self.rng.uniform(*self.rl.fw_altitude)
            else:
                z = float(city.ground_z(x, y)) + self.rng.uniform(*agl)
            p = np.array([x, y, z])
            if not city.is_free(x, y, z, clearance=3.0):
                continue
            if self.rl.clear_of_defence and any(
                    np.hypot(x - d.muzzle[0], y - d.muzzle[1]) < 1.1 * self.cfg.models.defence.range_m
                    for d in self.world.defences):
                continue
            if far_from is not None and np.linalg.norm(p[:2] - far_from[:2]) < min_dist:
                continue
            if far_from is not None and max_dist > min_dist and np.linalg.norm(p[:2] - far_from[:2]) > max_dist:
                continue
            others = self.world.pos[self.world.alive]
            if len(others) and np.min(np.linalg.norm(others - p, axis=1)) < min_sep:
                continue
            return p
        return np.array([0.0, 0.0, city.max_height + max(self.rl.fw_altitude) + 4.0 * self.rng.random()])

    def open_ground(self, x: float, y: float) -> bool:
        """True if a defence platform fits at (x, y): no building or tree within 2.5 m."""
        city = self.city
        xs = np.array([x, x + 2.5, x - 2.5, x, x]); ys = np.array([y, y, y, y + 2.5, y - 2.5])
        return bool(np.all(city.obstacle_z(xs, ys) - city.ground_z(xs, ys) < 0.3))

    def place_defence(self, k: int):
        """Put platform ``k`` at its fixed position ([x, y] on the ground, or [x, y, z],
        for example on a roof), or at a random open spot on the ground."""
        half = self.city.half
        if k in self.defence_at:
            x, y, *z = self.defence_at[k]
        else:
            z = []
            for _ in range(500):
                x, y = self.rng.uniform(-0.6 * half, 0.6 * half, 2)
                if self.open_ground(x, y):
                    break
        self.world.defences[k].reset([x, y, float(z[0]) if z else float(self.city.ground_z(x, y))])

    def _place_defences(self):
        for k in range(len(self.world.defences)):
            self.place_defence(k)
        self.world.projectiles.clear()
        self.t_relocate = self.world.time

    def respawn(self, i: int):
        rl, w = self.rl, self.world
        if w.alive[i]:
            w.kill(i)
        wing = w.kind(i) == "fixedwing"
        start = (np.array(self.start_at[i], float) if i in self.start_at
                 else self._sample_point(rl.spawn_agl, high=wing, near=0.0 if wing else rl.spawn_near_defence_m))
        dist = min(rl.min_goal_distance, 0.6 * self.city.size)
        if rl.task == "coverage":
            goal = start
        elif i in self.goal_at:
            goal = np.array(self.goal_at[i], float)
        elif rl.shared_goal and self.active.any():
            goal = self.goal[np.flatnonzero(self.active)[0]]
        else:
            goal = self._sample_point(rl.goal_agl, min_sep=0.0, far_from=start, min_dist=dist, high=False,
                                      max_dist=rl.max_goal_distance)
        self.goal[i] = goal
        yaw = (np.arctan2(goal[1] - start[1], goal[0] - start[0]) + self.rng.uniform(-1.0, 1.0)
               if rl.task == "navigate" else self.rng.uniform(-np.pi, np.pi))
        if wing and self.wingpilot:
            yaw = self.wingpilot.spawn_yaw(start)
        w.spawn(i, start, yaw)
        self.steps[i] = 0
        self.active[i] = True
        if wing and self.wingpilot:
            self.wingpilot.spawned(i)
        self.prev_dist[i] = np.linalg.norm(self._target(i) - start)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        w = self.world
        for i in range(self.n):
            if w.alive[i]:
                w.kill(i)
        self.active[:] = False
        if self.wingpilot:
            self.wingpilot.reset()
        self._place_defences()
        if self.rl.task == "coverage":
            m = int(np.ceil(self.city.size / self.rl.coverage_cell_m))
            self.visited = np.zeros((m, m), bool)
        for i in range(self.n):
            self.respawn(i)
        w.wind.reset(int(self.rng.integers(1 << 30)))
        return self.observe()

    # ------------------------------------------------------------- task logic
    def _cell(self, p):
        m = self.visited.shape[0]
        c = np.clip(((p[:2] + self.city.half) / self.rl.coverage_cell_m).astype(int), 0, m - 1)
        return int(c[1]), int(c[0])

    def _target(self, i: int) -> np.ndarray:
        """Point the drone is currently trying to reach."""
        if self.rl.task != "coverage":
            return self.goal[i]
        p = self.world.pos[i]
        todo = np.argwhere(~self.visited)
        if not len(todo):
            return p.copy()
        centres = (todo[:, ::-1] + 0.5) * self.rl.coverage_cell_m - self.city.half
        k = np.argmin(np.linalg.norm(centres - p[:2], axis=1))
        return np.array([centres[k, 0], centres[k, 1], p[2]])

    def _reflex(self, i: int, unit) -> np.ndarray:
        """Obstacle braking, as on a real flight controller: if the range finder shows something in
        the direction the command would move the drone, drop the horizontal part of the command and
        climb instead (or just stop, when there is no room above). Also refuses to descend onto a
        surface. The policy still chooses everything else, including where to turn."""
        rl, w = self.rl, self.world
        f, l, u, yw = np.asarray(unit, float)
        rays, nr = self.rays[i], rl.n_rays
        changed = False
        if abs(f) > 0.05 or abs(l) > 0.05:
            k = int(round(np.arctan2(l, f) / (2 * np.pi / nr))) % nr          # ray nearest the direction of travel
            ahead = min(rays[k], rays[(k - 1) % nr], rays[(k + 1) % nr])
            speed = float(np.hypot(w.vel[i, 0], w.vel[i, 1]))
            brake = max(rl.reflex_min_m, speed * rl.reflex_time_s)
            if ahead >= brake:                 # a second beam 2 m lower, so roofs are cleared with a margin
                yaw = np.arctan2(w.R[i, 1, 0], w.R[i, 0, 0]) + np.arctan2(l, f)
                low = w.raycast((w.pos[i] - [0.0, 0.0, 2.0])[None], np.array([[np.cos(yaw), np.sin(yaw), 0.0]]), brake)
                ahead = min(ahead, float(low[0]))
            if ahead < brake:
                agl = w.pos[i, 2] - self.city.ground_z(w.pos[i, 0], w.pos[i, 1])
                f = l = 0.0
                u = 1.0 if (rays[nr + 1] > 4.0 and agl < rl.ceiling_agl - 15.0) else max(u, 0.0)
                changed = True
        if u < 0.0 and rays[nr] < 3.0:
            u, changed = 0.0, True
        self.reflex_count += int(changed)
        return np.array([f, l, u, yw])

    # ------------------------------------------------------------------- step
    def step(self, actions):
        rl, w = self.rl, self.world
        for i in range(self.n):
            if not self.active[i]:
                continue
            if i in self.manual:
                self.actions.command(w, i, self.manual[i])
            elif self.wingpilot and w.kind(i) == "fixedwing":
                self.wingpilot.command(i)             # search-and-home autopilot, not the policy
            elif rl.collision_reflex and w.kind(i) == "quad":
                self.actions.command(w, i, self._reflex(i, self.actions.unit(actions[i])))
            else:
                self.actions.apply(w, i, actions[i])
        crashed = np.zeros(self.n, bool); hit = np.zeros(self.n, bool); oob = np.zeros(self.n, bool)
        for _ in range(self.substeps):
            w.step()
            if self.on_substep is not None:
                self.on_substep()
            agl = w.pos[:, 2] - self.city.ground_z(w.pos[:, 0], w.pos[:, 1])
            ev_h = w.hit & w.alive
            ev_c = w.crashed & w.alive & ~ev_h
            too_high = (agl > rl.ceiling_agl) & ~(self.is_wing & (w.pos[:, 2] <= self.wing_top))
            ev_o = (w.out_of_bounds | too_high) & w.alive & ~ev_h & ~ev_c
            hit |= ev_h; crashed |= ev_c; oob |= ev_o
            for i in np.flatnonzero(ev_h | ev_c | ev_o):
                w.kill(i)
        if self.wingpilot:
            self.wingpilot.update()

        reward = np.zeros(self.n)
        terminated = np.zeros(self.n, bool); truncated = np.zeros(self.n, bool)
        outcome = [None] * self.n
        obs = self.observe()                      # also refreshes self.min_ray
        for i in np.flatnonzero(self.active):
            self.steps[i] += 1
            r = rl.r_time
            if hit[i]:
                r += rl.r_hit; terminated[i] = True; outcome[i] = "hit"
            elif crashed[i]:
                r += rl.r_crash; terminated[i] = True; outcome[i] = "crash"
            elif oob[i]:
                r += rl.r_out_of_bounds; terminated[i] = True; outcome[i] = "out_of_bounds"
            else:
                target = self._target(i)
                d = float(np.linalg.norm(target - w.pos[i]))
                if rl.task == "coverage":
                    cell = self._cell(w.pos[i])
                    if not self.visited[cell]:
                        self.visited[cell] = True
                        r += rl.r_cell
                        reward += rl.team_share * rl.r_cell * self.active
                        reward[i] -= rl.team_share * rl.r_cell
                        d = float(np.linalg.norm(self._target(i) - w.pos[i]))
                    else:
                        r += rl.r_progress * (self.prev_dist[i] - d)
                    if self.visited.all():
                        terminated[i] = True; outcome[i] = "goal"
                else:
                    r += rl.r_progress * (self.prev_dist[i] - d)
                    if w.kind(i) == "fixedwing":      # planes: close sideways (they cannot hover) and at its height
                        to = target - w.pos[i]
                        reached = np.hypot(to[0], to[1]) < 3.0 * rl.goal_radius and abs(to[2]) < rl.goal_radius
                    else:
                        reached = d < rl.goal_radius
                    if reached:
                        r += rl.r_goal; terminated[i] = True; outcome[i] = "goal"
                        reward += rl.team_share * rl.r_goal * self.active
                        reward[i] -= rl.team_share * rl.r_goal
                self.prev_dist[i] = d
                if self.min_ray[i] < rl.proximity_m:
                    r += rl.r_proximity * (1.0 - self.min_ray[i] / rl.proximity_m)
                if not terminated[i] and self.steps[i] >= self.max_steps:
                    truncated[i] = True; outcome[i] = "timeout"
            reward[i] += r
            if outcome[i]:
                self.stats[outcome[i]] += 1
                self.active[i] = False
                if w.alive[i]:
                    w.kill(i)
        if self.world.defences and w.time - self.t_relocate > rl.defence_relocate_s:
            self._place_defences()
        return obs, reward, terminated, truncated, outcome

    # ------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        rl, w, L = self.rl, self.world, self.obs_layout
        obs = np.zeros((self.n, self.obs_dim), np.float32)
        idx = np.flatnonzero(w.alive & self.active)
        if not len(idx):
            return obs
        pos, vel, R = w.pos[idx], w.vel[idx], w.R[idx]
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)

        def to_heading(v):                    # world vector -> heading frame (x forward, y left)
            return np.stack([cy * v[:, 0] + sy * v[:, 1], -sy * v[:, 0] + cy * v[:, 1], v[:, 2]], 1)

        vmax = max(self.cfg.models.quad.max_speed, 1.0)
        obs[idx, L["velocity"]] = to_heading(vel) / vmax
        roll = np.arctan2(R[:, 2, 1], R[:, 2, 2]); pitch = np.arcsin(np.clip(R[:, 2, 0], -1, 1))
        obs[idx, L["attitude"]] = np.stack([roll, pitch, sy, cy], 1)
        agl = pos[:, 2] - self.city.ground_z(pos[:, 0], pos[:, 1])
        obs[idx, L["altitude"]] = (agl / 100.0)[:, None]

        tgt = np.stack([self._target(i) for i in idx]) - pos
        dist = np.linalg.norm(tgt, axis=1)
        obs[idx, L["goal"]] = np.concatenate(
            [to_heading(tgt) / np.maximum(dist, 1e-6)[:, None], (dist / self.city.size)[:, None]], 1)

        # range finder: horizontal fan in the heading frame + straight down + straight up
        nr = rl.n_rays
        dirs = np.empty((len(idx), nr + 2, 3))
        dirs[:, :nr, 0] = cy[:, None] * self._ray_dirs[None, :, 0] - sy[:, None] * self._ray_dirs[None, :, 1]
        dirs[:, :nr, 1] = sy[:, None] * self._ray_dirs[None, :, 0] + cy[:, None] * self._ray_dirs[None, :, 1]
        dirs[:, :nr, 2] = 0.0
        dirs[:, nr] = (0, 0, -1); dirs[:, nr + 1] = (0, 0, 1)
        origins = np.repeat(pos, nr + 2, axis=0)
        d = w.raycast(origins, dirs.reshape(-1, 3), rl.ray_range).reshape(len(idx), nr + 2)
        obs[idx, L["rays"]] = d / rl.ray_range
        self.rays[:] = rl.ray_range
        self.rays[idx] = d
        self.min_ray[:] = rl.ray_range
        self.min_ray[idx] = d.min(1)

        if rl.k_neighbors and len(idx) > 1:
            rel = pos[None] - pos[:, None]                                  # [i, j] = j relative to i
            dd = np.linalg.norm(rel, axis=2) + np.eye(len(idx)) * 1e9
            order = np.argsort(dd, axis=1)[:, :rl.k_neighbors]
            nb = np.zeros((len(idx), rl.k_neighbors, 7), np.float32)
            for a in range(len(idx)):
                for s, j in enumerate(order[a]):
                    if dd[a, j] > rl.neighbor_range:
                        continue
                    rp, rv = rel[a, j], vel[j] - vel[a]
                    nb[a, s] = [(cy[a] * rp[0] + sy[a] * rp[1]) / rl.neighbor_range,
                                (-sy[a] * rp[0] + cy[a] * rp[1]) / rl.neighbor_range,
                                rp[2] / rl.neighbor_range,
                                (cy[a] * rv[0] + sy[a] * rv[1]) / vmax,
                                (-sy[a] * rv[0] + cy[a] * rv[1]) / vmax, rv[2] / vmax, 1.0]
            obs[idx, L["neighbors"]] = nb.reshape(len(idx), -1)

        if rl.observe_defence and w.defences:
            spec = self.cfg.models.defence
            mz = np.stack([dfc.muzzle for dfc in w.defences])
            rel = mz[None] - pos[:, None]
            dd = np.linalg.norm(rel, axis=2)
            k = dd.argmin(1)
            near = rel[np.arange(len(idx)), k]
            obs[idx, L["defence"]] = np.concatenate(
                [to_heading(near) / spec.detection_range,
                 (dd[np.arange(len(idx)), k] < spec.range_m).astype(float)[:, None]], 1)

        if rl.observe_projectiles:
            act = np.flatnonzero(w.projectiles.active)
            if len(act):
                rel = w.projectiles.pos[act][None] - pos[:, None]
                dd = np.linalg.norm(rel, axis=2)
                k = dd.argmin(1); ar = np.arange(len(idx))
                seen = dd[ar, k] < 80.0
                rp = to_heading(rel[ar, k]) / 80.0
                rv = to_heading(w.projectiles.vel[act][k] - vel) / 100.0
                obs[idx, L["projectile"]] = np.concatenate([rp, rv, np.ones((len(idx), 1))], 1) * seen[:, None]

        obs[idx, L["type"]] = np.array([[1.0 if w.kind(i) == "fixedwing" else 0.0] for i in idx])
        return np.clip(obs, -5.0, 5.0)

    def info(self) -> dict:
        """Static description handed to policy plugins."""
        return {"obs_dim": self.obs_dim, "obs_layout": self.obs_layout,
                "action_mode": self.rl.action_mode, "action_names": self.actions.names,
                "n_agents": self.n, "map_size": self.city.size, "n_rays": self.rl.n_rays, "ray_range": self.rl.ray_range,
                "policy_hz": self.rl.policy_hz, "ceiling_agl": self.rl.ceiling_agl}

    def close(self):
        self.world.close()
