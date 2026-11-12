"""Search-and-home autopilot for fixed-wing drones (``rl.fw_mode: search``).

Search: the map is split into rings of equal area around the middle, one per
plane. The rings are rounded squares (shaped like the map, so the outer ring
reaches into the corners). Each plane loops round within its own ring on a slow
spiral that moves in (or out) by a little less than its sensor's swath per lap,
so together the planes cover the map without gaps or double work.

Find: a plane's sensor sees any target within ``rl.fw_detect_range`` metres
(horizontally), its own or another plane's. A sighting is shared with the
swarm: every plane whose target it is stops searching and flies straight to it.

Arrive: a plane flies to its target at its safe height (above the tallest roof).
Close to it, it circles the target and spirals down to the target's height, then
flies in. It only descends where its circle is clear of buildings and trees; if
the target sits too low among tall buildings, it circles at the lowest safe
height instead (and never reaches it).

Safety: a plane that is about to leave the map turns hard back inwards, and when
two planes converge at about the same height the higher-numbered one climbs over.

The policy does not fly planes in this mode; their commands come from here.
"""
from __future__ import annotations

import numpy as np

OVERLAP = 0.85          # spiral pitch per lap, as a share of the sensor's swath (2 x detect range)
EDGE_M = 35.0           # outermost circle stays this far inside the map edge
LOOKAHEAD_M = 15.0      # how sharply a plane steers back onto its circle
K_HOME = 0.9           # turn rate (rad/s) per radian off the target's bearing
K_TRACK = 0.6           # turn rate (rad/s) per radian off the search loop's direction
SAME_TARGET_M = 2.0     # goals closer than this are the same target
SHAPE_P = 8.0           # rings are |x|^p + |y|^p = r^p: 2 would be circles, larger is squarer
FENCE_S, FENCE_M = 3.0, 30.0   # turn back if the next FENCE_S seconds would end within FENCE_M of the edge
                               # (generous: wind high up can carry a turning plane a long way)
SEPARATE_M, SEPARATE_DZ = 30.0, 5.0   # planes this close sideways and vertically make room


def level(xy):
    """Which ring a point is on: its 'radius' in the rounded-square sense."""
    x, y = abs(float(xy[0])), abs(float(xy[1]))
    m = max(x, y, 1e-6)
    return m * ((x / m) ** SHAPE_P + (y / m) ** SHAPE_P) ** (1.0 / SHAPE_P)


def along(xy) -> float:
    """Direction (radians) of the anticlockwise loop through this point."""
    x, y = float(xy[0]), float(xy[1])
    m = max(abs(x), abs(y), 1e-6)
    gx = np.sign(x) * (abs(x) / m) ** (SHAPE_P - 1)          # gradient of the level, up to scale
    gy = np.sign(y) * (abs(y) / m) ** (SHAPE_P - 1)
    return float(np.arctan2(gx, -gy))                         # gradient turned a quarter left

ORBIT = 1.5            # radius of the descent circle round a target, in turn radii
CLEAR_M = 6.0           # descend no lower than this above anything under or inside the circle
STACK_M = 5.0           # planes waiting above a target they cannot reach circle this far apart in height
FINAL_APPROACH_M = 40.0
SEARCH, HOME = 0, 1


def orbit_centre(city, goal, radius: float) -> np.ndarray:
    """Where to circle while descending: round the target, moved inwards if the circle
    would otherwise reach the map edge."""
    lim = city.half - EDGE_M - radius
    return np.clip(np.asarray(goal[:2], float), -lim, lim)


def approach_floor(city, goal, radius: float) -> float:
    """Lowest height (absolute z) a plane can circle down to near ``goal`` and then fly in
    from: clear of everything within the circle, between it and the target, and a margin."""
    c = orbit_centre(city, goal, radius)
    r = radius + 30.0 + float(np.hypot(*(np.asarray(goal[:2]) - c)))
    g = np.arange(-r, r + 0.1, 4.0)
    x, y = np.meshgrid(g, g)
    inside = np.hypot(x, y) <= r
    tops = city.obstacle_z(c[0] + x[inside], c[1] + y[inside])
    return float(np.max(tops)) + CLEAR_M


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class WingPilot:
    def __init__(self, core):
        self.core = core
        n = core.n
        self.state = np.full(n, SEARCH)
        self.cruise_z = np.zeros(n)
        self.radius = np.zeros(n)                  # current search circle
        self.outward = np.ones(n, bool)
        self.found: list[np.ndarray] = []          # targets seen this episode, shared by the swarm
        self.newly_found: list[int] = []           # planes that switched to HOME since the last pop
        spec, g = core.cfg.models.fixedwing, core.cfg.physics.gravity
        turn_r = spec.cruise_speed ** 2 / (g * np.tan(np.radians(spec.max_bank_deg)))
        self.turn_r = turn_r
        self.orbit_r = ORBIT * turn_r
        self._floors: dict[tuple, float] = {}
        self.home_queue: dict[tuple, list[int]] = {}
        half = core.city.half
        r_min = max(1.5 * turn_r, 0.15 * half)
        r_max = max(r_min + 10.0, half - EDGE_M)
        self.wings = [i for i in range(n) if core.world.kind(i) == "fixedwing"]
        k = len(self.wings)
        edges = np.sqrt(r_min ** 2 + (r_max ** 2 - r_min ** 2) * np.arange(k + 1) / max(k, 1))  # equal-area rings
        self.band = {i: (float(edges[j]), float(edges[j + 1])) for j, i in enumerate(self.wings)}

    # ----------------------------------------------------------------- state
    def is_wing(self, i: int) -> bool:
        return i in self.band

    def reset(self):
        self.found.clear()
        self.newly_found.clear()
        self.home_queue.clear()
        self.state[:] = SEARCH

    def spawn_yaw(self, start) -> float:
        """Start already flying round the loop (anticlockwise, like every plane)."""
        return along(start[:2])

    def spawned(self, i: int):
        core = self.core
        lo, hi = self.band[i]
        self.cruise_z[i] = core.world.pos[i, 2]
        self.radius[i] = float(np.clip(level(core.world.pos[i, :2]), lo, hi))   # nearest point of its ring
        self.outward[i] = self.radius[i] < 0.5 * (lo + hi)
        self.state[i] = HOME if self._known(core.goal[i]) else SEARCH

    def _known(self, goal) -> bool:
        return any(np.linalg.norm(goal - f) < SAME_TARGET_M for f in self.found)

    def update(self):
        """After a step: does any plane's sensor see a target that is still being searched for?"""
        core, w = self.core, self.core.world
        eyes = [i for i in self.wings if core.active[i] and w.alive[i]]
        if not eyes:
            return
        seen_from = w.pos[eyes, :2]
        for j in self.wings:
            if not (core.active[j] and self.state[j] == SEARCH):
                continue
            if np.min(np.hypot(*(seen_from - core.goal[j, :2]).T)) <= core.rl.fw_detect_range:
                target = core.goal[j].copy()
                self.found.append(target)

                members = [
                    k for k in self.wings
                    if core.active[k]
                    and self.state[k] == SEARCH
                    and np.linalg.norm(core.goal[k] - target) < SAME_TARGET_M
                ]

                members.sort(
                    key=lambda k: np.hypot(
                        core.world.pos[k, 0] - target[0],
                        core.world.pos[k, 1] - target[1]
                    )
                )

                self.home_queue[tuple(np.round(target, 1))] = members

                for k in members:
                    self.state[k] = HOME
                    self.newly_found.append(k)
        del self.newly_found[:-64]                     # only the app reads these; keep the list short

    def pop_found(self) -> list[int]:
        out, self.newly_found = self.newly_found, []
        return out

    def floor(self, goal) -> float:
        key = tuple(np.round(goal, 1))
        if key not in self._floors:
            self._floors[key] = approach_floor(self.core.city, goal, self.orbit_r)
        return self._floors[key]

    def modes(self) -> list[str]:
        """'' for quadcopters; 'search', 'home', or 'blocked' (target too low among buildings)."""
        out = []
        for i in range(self.core.n):
            if not self.is_wing(i):
                out.append("")
            elif self.state[i] != HOME:
                out.append("search")
            else:
                out.append("home")
        return out

    # --------------------------------------------------------------- command
    def home_leader(self, i: int) -> bool:
        goal = self.core.goal[i]
        key = tuple(np.round(goal, 1))
        queue = self.home_queue.get(key, [])

        for k in queue:
            if self.core.active[k] and self.core.world.alive[k]:
                return k == i

        return False
    def command(self, i: int):
        core, w = self.core, self.core.world
        spec = core.cfg.models.fixedwing
        pos, vel = w.pos[i], w.vel[i]
        course = np.arctan2(vel[1], vel[0])
        speed = float(np.hypot(vel[0], vel[1]))
        if self.state[i] == HOME and self.home_leader(i):
            goal = core.goal[i]
            to = goal - pos
            dist = float(np.hypot(to[0], to[1]))
            err = wrap(np.arctan2(to[1], to[0]) - course)
            want_z = float(goal[2])
            safe_z = max(self.cruise_z[i], goal[2])
            R = self.orbit_r
            if dist < FINAL_APPROACH_M:
                # close enough: stop orbiting and fly directly to the goal
                turn = K_HOME * err
                z_cmd = goal[2]

            elif dist > 5.0 * R:
                turn, z_cmd = K_HOME * err, safe_z

            elif abs(pos[2] - want_z) > 3.0:
                # circle the target (anticlockwise) and spiral down once on the circle
                rel = pos[:2] - orbit_centre(core.city, goal, R)
                d = max(float(np.hypot(*rel)), 1.0)
                want = np.arctan2(rel[1], rel[0]) + np.pi / 2 + np.arctan((d - R) / LOOKAHEAD_M)
                turn = max(speed, 1.0) / R + K_TRACK * wrap(want - course)
                # descend only on the circle; pushed well off it (wind, another plane), climb back up
                z_cmd = want_z if abs(d - R) < 30.0 else safe_z
            else:                                             # down at its height: fly in
                turn = 0.0 if (dist < 2.0 * self.turn_r and abs(err) > np.pi / 2) else K_HOME * err
                z_cmd = goal[2]
            climb = float(np.clip(0.8 * (z_cmd - pos[2]), -spec.max_descent, spec.max_climb))
            airspeed = spec.cruise_speed      # slowing down in a hard turn would stall the plane
        else:
            lo, hi = self.band[i]
            pitch = OVERLAP * 2.0 * core.rl.fw_detect_range
            if hi - lo <= pitch:                       # the ring is narrower than the sensor's swath
                r = 0.5 * (lo + hi)
            else:
                step = pitch * max(speed, 1.0) / core.rl.policy_hz / (2 * np.pi * self.radius[i])
                r = self.radius[i] + (step if self.outward[i] else -step)
                if r >= hi or r <= lo:                 # turn the spiral round at the ring's edge
                    self.outward[i] = not self.outward[i]
                    r = float(np.clip(r, lo, hi))
            self.radius[i] = r
            here = along(pos[:2])
            ahead = along(pos[:2] + 5.0 * np.array([np.cos(here), np.sin(here)]))
            bend = wrap(ahead - here) / 5.0                                # how sharply the loop curves here
            want = here + np.arctan((level(pos[:2]) - r) / LOOKAHEAD_M)   # steer back onto the loop
            turn = max(speed, 1.0) * bend + K_TRACK * wrap(want - course)
            climb = 0.4 * (self.cruise_z[i] - pos[2])
            airspeed = spec.cruise_speed
        ahead = pos[:2] + FENCE_S * vel[:2]
        if np.max(np.abs(ahead)) > core.city.half - FENCE_M:              # geofence: turn back inwards
            turn = 10.0 * np.sign(wrap(np.arctan2(-pos[1], -pos[0]) - course) or 1.0)
        for j in self.wings:                                                # make room for another plane
            if j == i or not (core.active[j] and w.alive[j]):
                continue
            rel, closing = w.pos[j] - pos, float(np.dot(w.pos[j] - pos, w.vel[j] - vel))
            if np.hypot(rel[0], rel[1]) < SEPARATE_M and abs(rel[2]) < SEPARATE_DZ and closing < 0 and i > j:
                climb = spec.max_climb
        w.command_fixedwing(i, turn_rate=float(turn), climb_rate=float(climb), airspeed=float(airspeed))
