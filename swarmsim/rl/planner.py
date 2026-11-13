"""Classical baseline: A* route planning around buildings and defence platforms.

This is the yardstick a learned policy has to beat. It is an *oracle*: unlike
a learned policy, which only sees its own observation vector, the planner reads
the full map and the true position of every defence platform from the
simulator. It needs no training.

    python main.py demo --model planner
    python scripts/benchmark.py --policies planner scripted sb3:runs/stage1_navigate/model.zip

How it works, per quadcopter:
  1. The map is cut into square cells. A cell is blocked when a building or
     tree reaches the drone's cruise height.
  2. Cells inside a platform's firing range *and* in its line of sight cost
     ``threat_cost`` times more to cross, so the route goes around the firing
     zone, or behind buildings, whenever a detour exists.
  3. A* finds the cheapest route; a small controller follows it with the same
     action primitives a learned policy uses.

Set ``threat_cost=1`` to get a planner that ignores the defence (shortest
route only), which isolates how much the threat-avoidance itself is worth.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .actions import PRIMITIVES
from .plugin import PolicyPlugin, register

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def astar(cost: np.ndarray, start: tuple, goal: tuple):
    """Cheapest 8-connected path on a grid. ``cost`` is the price per metre-cell of entering
    a cell; ``np.inf`` marks a blocked cell. Returns a list of (row, col) or None."""
    n_r, n_c = cost.shape
    if not (np.isfinite(cost[start]) and np.isfinite(cost[goal])):
        return None
    cmin = float(cost[np.isfinite(cost)].min())
    best = {start: 0.0}
    came = {}
    heap = [(0.0, start)]
    closed = set()
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        g0 = best[cur]
        r, c = cur
        for dr, dc, step in NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < n_r and 0 <= cc < n_c):
                continue
            w = cost[rr, cc]
            if not np.isfinite(w):
                continue
            if dr and dc and not (np.isfinite(cost[r, cc]) and np.isfinite(cost[rr, c])):
                continue                                   # do not cut a blocked corner
            g = g0 + step * w
            nxt = (rr, cc)
            if g < best.get(nxt, np.inf):
                best[nxt] = g
                came[nxt] = cur
                h = cmin * math.hypot(goal[0] - rr, goal[1] - cc)
                heapq.heappush(heap, (g + h, nxt))
    return None


@register
class PlannerPolicy(PolicyPlugin):
    name = "planner"

    def __init__(self, info, cell_m: float = 4.0, threat_cost: float = 25.0, threat_margin_m: float = 8.0,
                 clearance_m: float = 4.0, lookahead_m: float = 10.0, cruise_percentile: float = 99.5,
                 cruise_cap_agl: float = 60.0, **kw):
        super().__init__(info)
        self.cell, self.threat_cost, self.threat_margin = float(cell_m), float(threat_cost), float(threat_margin_m)
        self.clearance, self.lookahead = float(clearance_m), float(lookahead_m)
        self.cruise_percentile, self.cruise_cap = float(cruise_percentile), float(cruise_cap_agl)
        self.core = None
        self.reset()

    # ---------------------------------------------------------------- set-up
    def bind(self, core) -> None:
        """Give the planner the simulator it plans in (called by the runner)."""
        self.core = core
        city = core.city
        self.n_cells = int(math.ceil(city.size / self.cell))
        centres = (np.arange(self.n_cells) + 0.5) * self.cell - city.half
        self.gx, self.gy = np.meshgrid(centres, centres)            # [row = y, col = x]
        self.ground = np.asarray(city.ground_z(self.gx, self.gy), float)
        top = np.asarray(city.obstacle_z(self.gx, self.gy), float)
        pad = np.pad(top, 1, mode="edge")                           # widen every obstacle by one cell
        self.top = np.max([pad[1 + a:1 + a + self.n_cells, 1 + b:1 + b + self.n_cells]
                           for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)
        # cruise above nearly every roof and tree, so the route depends on where the goal and the
        # platforms are (which a drone can sense) and not on obstacles it cannot see from afar
        over = float(np.percentile(self.top - self.ground, self.cruise_percentile)) + self.clearance + 1.0
        self.cruise_agl = min(over, self.cruise_cap)
        self.reset()

    def reset(self) -> None:
        self.paths = {}            # drone -> dict(points, z, goal, key)
        self._threat_cache = {}

    def _cell_of(self, x, y):
        half = self.core.city.half
        return (int(np.clip((y + half) / self.cell, 0, self.n_cells - 1)),
                int(np.clip((x + half) / self.cell, 0, self.n_cells - 1)))

    # ---------------------------------------------------------------- threat
    def _defence_key(self):
        return tuple(np.round(np.concatenate([d.muzzle for d in self.core.world.defences]), 1)) \
            if self.core.world.defences else ()

    def _threat(self, z: float) -> np.ndarray:
        """Boolean grid: cells a platform can both reach and see at flight height ``z``."""
        key = (self._defence_key(), round(z))
        if key not in self._threat_cache:
            w = self.core.world
            mask = np.zeros((self.n_cells, self.n_cells), bool)
            reach = self.core.cfg.models.defence.range_m + self.threat_margin
            for d in w.defences:
                near = np.argwhere(np.hypot(self.gx - d.muzzle[0], self.gy - d.muzzle[1]) < reach)
                if not len(near):
                    continue
                seen = np.zeros(len(near), bool)
                for dz in (0.0, 3.0):                      # a little above too: cover that only just hides a drone is no cover
                    pts = np.column_stack([self.gx[near[:, 0], near[:, 1]], self.gy[near[:, 0], near[:, 1]],
                                           np.full(len(near), z + dz)])
                    seen |= np.asarray(w.line_of_sight(d.muzzle, pts), bool)
                mask[near[seen, 0], near[seen, 1]] = True
            if len(self._threat_cache) > 64:
                self._threat_cache = {}
            self._threat_cache[key] = mask
        return self._threat_cache[key]

    # ------------------------------------------------------------------ plan
    def _plan(self, i: int):
        core = self.core
        pos, goal = core.world.pos[i], core.goal[i]
        s, g = self._cell_of(pos[0], pos[1]), self._cell_of(goal[0], goal[1])
        ceiling = float(np.max(self.ground)) + core.rl.ceiling_agl - 15.0
        z = max(float(pos[2]), float(goal[2]), float(self.core.city.ground_z(pos[0], pos[1])) + self.cruise_agl)
        z = min(z, ceiling)
        path = None
        while path is None:
            cost = np.where(self.top + self.clearance > z, np.inf, 1.0)
            cost[self._threat(z) & np.isfinite(cost)] = self.threat_cost
            for cell in (s, g):                                     # the drone is where it is: never block its own cell
                if not np.isfinite(cost[cell]):
                    cost[cell] = 1.0
            path = astar(cost, s, g)
            if path is None:
                if z + 8.0 > ceiling:
                    break
                z += 8.0                                            # boxed in at this height: try flying higher
        if path is None:                                            # last resort: straight line
            pts = np.array([pos[:2], goal[:2]], float)
        else:
            path = self._shortcut(path, cost)
            pts = np.array([[self.gx[r, c], self.gy[r, c]] for r, c in path], float)
            pts = self._densify(pts)
            pts[0], pts[-1] = pos[:2], goal[:2]
        self.paths[i] = {"pts": pts, "z": z, "goal": goal.copy(), "key": self._defence_key(), "k": 0}

    @staticmethod
    def _line_ok(cost, a, b) -> bool:
        """True if the straight line a -> b crosses no blocked cell and no cell dearer than its cheaper
        end (so a leg that ends just inside a firing zone is never extended across the zone)."""
        limit = min(cost[a], cost[b])
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
        rr = np.round(np.linspace(a[0], b[0], n)).astype(int)
        cc = np.round(np.linspace(a[1], b[1], n)).astype(int)
        return bool(np.all(cost[rr, cc] <= limit))

    def _shortcut(self, path, cost):
        """Replace the staircase of grid cells by a few straight legs (fewer turns, faster flight)."""
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._line_ok(cost, path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out

    def _densify(self, pts):
        """A point every cell length along the legs, so 'nearest point' and look-ahead work."""
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / self.cell)))
            out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
        return np.array(out)

    def _needs_plan(self, i: int) -> bool:
        p = self.paths.get(i)
        if p is None or p["key"] != self._defence_key():
            return True
        if np.linalg.norm(p["goal"] - self.core.goal[i]) > 1.0:
            return True
        pos = self.core.world.pos[i]
        return float(np.min(np.linalg.norm(p["pts"][p["k"]:p["k"] + 8] - pos[:2], axis=1))) > 3.0 * self.cell

    # ---------------------------------------------------------------- follow
    def _command(self, i: int, o: np.ndarray):
        """(forward, left, up, yaw) in [-1, 1] and the single primitive that matters most right now."""
        core, L = self.core, self.info["obs_layout"]
        w = core.world
        pos, goal, p = w.pos[i], core.goal[i], self.paths[i]
        pts = p["pts"]
        k = p["k"] + int(np.argmin(np.linalg.norm(pts[p["k"]:p["k"] + 8] - pos[:2], axis=1)))
        p["k"] = k
        j = k
        while j < len(pts) - 1 and np.linalg.norm(pts[j] - pos[:2]) < self.lookahead:
            j += 1
        aim = pts[j]
        yaw = math.atan2(w.R[i, 1, 0], w.R[i, 0, 0])
        to_goal = float(np.hypot(goal[0] - pos[0], goal[1] - pos[1]))
        final = to_goal < 2.5 * self.lookahead
        want_z = float(goal[2]) if final else p["z"]
        dz = want_z - float(pos[2])
        err = (math.atan2(aim[1] - pos[1], aim[0] - pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        if to_goal < 3.0:                                           # over the goal: only height is left
            err = 0.0
        rays = o[L["rays"]] * self.info["ray_range"]
        nr = self.info["n_rays"]
        front = float(min(rays[0], rays[1], rays[nr - 1]))
        turn = "yaw_left" if err > 0 else "yaw_right"
        if rays[nr] < 4.0 or (front < 6.0 and not final):           # reflex: something too close below or ahead
            name = "up"
        elif dz > 4.0:
            name = "up"
        elif abs(err) > 0.35:
            name = turn
        elif to_goal < 3.0:
            name = "up" if dz > 1.0 else ("down" if dz < -1.0 else "hover")
        elif dz < -3.0 and (final or abs(err) < 0.1):
            name = "down"
        elif abs(err) > 0.15:
            name = turn
        else:
            name = "forward"
        unit = (max(0.0, math.cos(err)) if abs(err) < 0.6 and to_goal > 3.0 else 0.0, 0.0,
                float(np.clip(dz / 3.0, -1, 1)) if name != "up" else 1.0, float(np.clip(err / 0.4, -1, 1)))
        return unit, name

    def act(self, obs):
        if self.core is None:
            raise RuntimeError("The planner policy needs the simulator: call policy.bind(core) first "
                               "(python main.py demo and scripts/benchmark.py do this for you).")
        core, names = self.core, list(self.info["action_names"])
        discrete = self.info["action_mode"] == "discrete"
        out = []
        for i in range(core.n):
            if not (core.active[i] and core.world.alive[i]) or core.world.kind(i) == "fixedwing":
                out.append(names.index("hover") if discrete and "hover" in names else (0 if discrete else PRIMITIVES["hover"]))
                continue
            if self._needs_plan(i):
                self._plan(i)
            unit, name = self._command(i, obs[i])
            if discrete:
                out.append(next(names.index(c) for c in (name, "forward", "hover", names[0]) if c in names))
            else:
                out.append(unit)
        return np.array(out) if discrete else np.array(out, np.float32)


@register
class ShortestPathPolicy(PlannerPolicy):
    """The same planner with the threat cost switched off: avoids buildings, ignores the defence."""
    name = "planner_blind"

    def __init__(self, info, **kw):
        kw["threat_cost"] = 1.0
        super().__init__(info, **kw)
