"""Live 3-D viewer used by ``--gui``.

Adds what the bare PyBullet window lacks: a camera that follows a drone, mouse
and keyboard camera control, labels on every drone / goal / defence platform,
a status line, buttons, and smooth real-time playback.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pybullet as pb

HELP = """
+------------------------- swarmsim live view --------------------------+
|  WHAT YOU SEE                                                         |
|    D0, D1 ...   drones (blue = quadcopter, orange = fixed-wing)       |
|    G0, G1 ...   the goal of each drone (green ball)                   |
|    DEFENCE      the defence platform; red ring = its firing range     |
|    yellow balls projectiles                                           |
|                                                                       |
|  CAMERA                                                               |
|    N / B        follow next / previous drone                          |
|    O            overview of the whole map                             |
|    F            free camera (stops following)                         |
|    drag mouse   rotate          arrow keys   rotate / tilt            |
|    Z / X        zoom in / out   mouse wheel  fine zoom                |
|                                                                       |
|  PLAYBACK                                                             |
|    Space        pause / resume                                        |
|    R            skip to the next episode                              |
|    Q            quit                                                  |
|    The panel on the right has the same actions as buttons and a       |
|    speed slider. Ignore the 'Explorer' list on the left.              |
+-----------------------------------------------------------------------+
"""

BLUE, ORANGE, GREEN, RED, WHITE = [0.15, 0.45, 1.0], [1.0, 0.5, 0.05], [0.1, 0.8, 0.2], [0.9, 0.1, 0.1], [1, 1, 1]
HIDDEN = [0.0, 0.0, -600.0]
FOLLOW, OVERVIEW, FREE = "follow", "overview", "free"
LABELS = {"goal": "reached its goal", "hit": "was hit", "crash": "crashed",
          "out_of_bounds": "left the map", "timeout": "ran out of time"}


class Viewer:
    def __init__(self, core, policy_name: str = "", episodes: int = 1, verbose: bool = True, panel: bool = True):
        self.core, self.w, self.c = core, core.world, core.world.c
        self.policy_name, self.episodes, self.verbose = policy_name, episodes, verbose
        c = self.c
        for flag, value in ((pb.COV_ENABLE_GUI, int(panel)), (pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0),
                            (pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0), (pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0),
                            (pb.COV_ENABLE_MOUSE_PICKING, 0), (pb.COV_ENABLE_KEYBOARD_SHORTCUTS, 0),
                            (pb.COV_ENABLE_SHADOWS, 0)):
            pb.configureDebugVisualizer(flag, value, physicsClientId=c)

        # side-panel controls (a "button" is a parameter whose min is above its max)
        self.p_speed, self.buttons = None, {}
        if panel:
            self.p_speed = pb.addUserDebugParameter("Speed (x real time)", 0.25, 8.0, 1.0, physicsClientId=c)
            names = [("next", "Follow next drone [N]"), ("overview", "Overview [O]"), ("free", "Free camera [F]"),
                     ("pause", "Pause / resume [Space]"), ("skip", "Next episode [R]")]
            self.buttons = {k: pb.addUserDebugParameter(label, 1, 0, 0, physicsClientId=c) for k, label in names}
        self.clicks = {k: 0.0 for k in self.buttons}
        # invisible body the status text is attached to; it is moved with the camera target
        # (attached text only tracks a body that has a visual shape, so it gets a 1 cm dot)
        dot = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 0, 1], physicsClientId=c)
        self.anchor = pb.createMultiBody(0, -1, dot, [0, 0, 0], physicsClientId=c)

        w = self.w
        w.enable_projectile_visuals(radius=0.45)
        goal_vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=1.2, rgbaColor=GREEN + [0.9], physicsClientId=c)
        self.goal_bodies = [pb.createMultiBody(0, -1, goal_vis, HIDDEN, physicsClientId=c) for _ in range(core.n)]
        self.goal_labels = [-1] * core.n
        self.goal_shown = np.full((core.n, 3), np.nan)
        for i, body in enumerate(w.body_ids):              # labels ride along with the drones
            col = ORANGE if w.kind(i) == "fixedwing" else BLUE
            pb.addUserDebugText(f"D{i}", [0, 0, 0.7], textColorRGB=col, textSize=1.5,
                                parentObjectUniqueId=body, physicsClientId=c)
            pb.addUserDebugLine([0, 0, 0.15], [0, 0, 0.65], col, 2, parentObjectUniqueId=body, physicsClientId=c)
        r = core.cfg.models.defence.range_m
        for plat in w.defences:                             # range ring rides along with the platform
            pb.addUserDebugText("DEFENCE", [0, 0, 5.0], textColorRGB=RED, textSize=1.5,
                                parentObjectUniqueId=plat.body, physicsClientId=c)
            ang = np.linspace(0, 2 * math.pi, 49)
            for a0, a1 in zip(ang[:-1], ang[1:]):
                pb.addUserDebugLine([r * math.cos(a0), r * math.sin(a0), 2.0], [r * math.cos(a1), r * math.sin(a1), 2.0],
                                    RED, 2, parentObjectUniqueId=plat.body, physicsClientId=c)

        self.mode, self.follow = FOLLOW, 0
        self.yaw, self.pitch, self.dist = 0.0, -22.0, 12.0
        self.target = np.zeros(3)
        self.paused = self.skip = self.quit = False
        self.speed = 1.0
        self.hud = [-1, -1, -1]
        self.goal_line = -1
        self._pending = None               # camera values sent to the window but not yet confirmed by it
        self._drag = False
        self._last_mouse = None
        self._last_input = self._last_hud = self._last_buttons = 0.0
        self._input_gap, self._hud_gap = 1 / 60, 0.33     # stretched automatically on slow machines
        self._hud_state = [None, None, None]               # what each status line currently shows
        self._due = time.perf_counter()
        self.counts = {k: 0 for k in LABELS}
        self.episode, self.sim_t, self.banner = 0, 0.0, ""
        if verbose:
            print(HELP)

    # ------------------------------------------------------------- episodes
    def new_episode(self, ep: int):
        self.episode, self.sim_t, self.banner = ep, 0.0, ""
        self.counts = {k: 0 for k in LABELS}
        self.skip = False
        self._sync_goals(force=True)
        live = self._live()
        self._follow(live[0] if live else 0)
        self._due = time.perf_counter()
        if self.verbose:
            print(f"\n=== episode {ep + 1} of {self.episodes}: {self.core.n} drones, policy '{self.policy_name}' ===")

    def after_step(self, outcome):
        """Call after every policy step with the per-drone outcome list."""
        for i, out in enumerate(outcome):
            if out:
                self.counts[out] += 1
                if self.verbose:
                    print(f"  [{self.sim_t:5.1f} s] drone D{i} {LABELS[out]}")
        self._sync_goals()

    def end_episode(self, hold_s: float = 3.0):
        c = self.counts
        self.banner = f"EPISODE OVER: {c['goal']} at goal, {c['hit']} hit, {c['crash']} crashed"
        if c['timeout'] + c['out_of_bounds']:
            self.banner += f", {c['timeout'] + c['out_of_bounds']} other"
        if self.verbose:
            print("  " + self.banner)
        if self.mode == FOLLOW:
            self._overview()
        end = time.perf_counter() + hold_s
        while time.perf_counter() < end and not (self.quit or self.skip):
            self.tick(force_hud=True)
            time.sleep(0.02)

    # --------------------------------------------------------------- timing
    def on_substep(self):
        """Called by the environment after every physics step: paces real time, moves the camera."""
        self.sim_t += self.w.dt
        self.tick()
        self._due += self.w.dt / max(self.speed, 1e-3)
        now = time.perf_counter()
        if self._due - now > 0:
            time.sleep(self._due - now)
        elif now - self._due > 0.25:          # fell behind (slow machine): do not try to catch up
            self._due = now

    def wait_if_paused(self):
        while self.paused and not (self.quit or self.skip):
            self.tick(force_hud=True)
            time.sleep(0.02)
            self._due = time.perf_counter()

    # ---------------------------------------------------------------- input
    def _live(self):
        return [i for i in range(self.core.n) if self.core.active[i] and self.w.alive[i]]

    def _follow(self, i: int):
        self.mode, self.follow = FOLLOW, i
        fwd = self.w.R[i][:, 0]
        self.yaw = math.degrees(math.atan2(fwd[1], fwd[0])) - 90.0     # start behind the drone
        self.pitch = -20.0
        self.dist = 34.0 if self.w.kind(i) == "fixedwing" else 11.0
        self.target = np.array(self.w.pos[i], dtype=float)
        self._pending = (self.yaw, self.pitch, self.dist)
        self._apply_camera()

    def _overview(self):
        self.mode = OVERVIEW
        city = self.w.city
        self.pitch, self.dist = -52.0, 0.95 * city.size
        self.target = np.array([0.0, 0.0, float(city.ground_z(0.0, 0.0))])
        self._pending = (self.yaw, self.pitch, self.dist)
        self._apply_camera()

    def _step_follow(self, direction: int):
        live = self._live()
        if not live:
            return
        if self.mode == FOLLOW and self.follow in live:
            self._follow(live[(live.index(self.follow) + direction) % len(live)])
        else:
            self._follow(live[0])

    def command(self, name: str):
        """One-shot actions; used by keys, buttons and tests."""
        if name == "next":
            self._step_follow(+1)
        elif name == "prev":
            self._step_follow(-1)
        elif name == "overview":
            self._overview()
        elif name == "free":
            self.mode = FREE
        elif name == "pause":
            self.paused = not self.paused
            if self.verbose:
                print("  paused" if self.paused else "  resumed")
        elif name == "skip":
            self.skip = True
        elif name == "quit":
            self.quit = True

    def rotate(self, d_yaw: float = 0.0, d_pitch: float = 0.0, zoom: float = 1.0):
        self.yaw += d_yaw
        self.pitch = float(np.clip(self.pitch + d_pitch, -89.0, 20.0))
        self.dist = float(np.clip(self.dist * zoom, 2.0, 3.0 * self.w.city.size))
        self._pending = (self.yaw, self.pitch, self.dist)

    def _poll(self) -> bool:
        """Read keys, mouse and buttons. Returns True if the user moved the camera."""
        c = self.c
        keys = pb.getKeyboardEvents(physicsClientId=c)
        hit = lambda ch: bool(keys.get(ord(ch), 0) & pb.KEY_WAS_TRIGGERED)       # noqa: E731
        held = lambda code: bool(keys.get(code, 0) & pb.KEY_IS_DOWN)             # noqa: E731
        for ch, name in (("n", "next"), ("b", "prev"), ("o", "overview"), ("f", "free"),
                         (" ", "pause"), ("r", "skip"), ("q", "quit")):
            if hit(ch):
                self.command(name)
        moved = False
        step = 1.6
        for code, dy, dp in ((pb.B3G_LEFT_ARROW, -step, 0), (pb.B3G_RIGHT_ARROW, step, 0),
                             (pb.B3G_UP_ARROW, 0, -step * 0.6), (pb.B3G_DOWN_ARROW, 0, step * 0.6)):
            if held(code):
                self.rotate(dy, dp)
                moved = True
        if held(ord("z")):
            self.rotate(zoom=0.965)
            moved = True
        if held(ord("x")):
            self.rotate(zoom=1.035)
            moved = True

        native = held(pb.B3G_CONTROL) or held(pb.B3G_ALT)     # PyBullet's own Ctrl/Alt-drag is active
        for etype, mx, my, button, state in pb.getMouseEvents(physicsClientId=c):
            if etype == 2 and button == 0:
                self._drag = bool(state & (pb.KEY_IS_DOWN | pb.KEY_WAS_TRIGGERED)) and not state & pb.KEY_WAS_RELEASED
                self._last_mouse = (mx, my)
            elif etype == 1:
                if self._drag and not native and self._last_mouse is not None:
                    self.rotate(-(mx - self._last_mouse[0]) * 0.3, -(my - self._last_mouse[1]) * 0.25)
                    moved = True
                self._last_mouse = (mx, my)

        now = time.perf_counter()
        if self.buttons and now - self._last_buttons >= 0.1:      # panel widgets: 10 times a second is plenty
            self._last_buttons = now
            for name, pid in self.buttons.items():
                n = pb.readUserDebugParameter(pid, physicsClientId=c)
                if n != self.clicks[name]:
                    self.clicks[name] = n
                    self.command(name)
            self.speed = float(pb.readUserDebugParameter(self.p_speed, physicsClientId=c))
        elif not self.buttons:                                              # no panel: speed on the - and = keys
            if hit("-"):
                self.speed = max(0.25, self.speed / 2)
            if hit("="):
                self.speed = min(8.0, self.speed * 2)
        return moved

    # --------------------------------------------------------------- camera
    def _apply_camera(self):
        pb.resetDebugVisualizerCamera(self.dist, self.yaw, self.pitch, self.target.tolist(), physicsClientId=self.c)
        pb.resetBasePositionAndOrientation(self.anchor, self.target.tolist(), [0, 0, 0, 1], physicsClientId=self.c)

    def _read_back(self, cam):
        """Adopt PyBullet's own wheel zoom / Ctrl-drag, but never a stale value from before our last change
        (the window applies camera commands a frame or two late)."""
        seen = (cam[8], cam[9], cam[10])
        if self._pending is not None:
            if all(abs(a - b) < 1e-2 * max(1.0, abs(b)) for a, b in zip(seen, self._pending)):
                self._pending = None
            return
        self.yaw, self.pitch, self.dist = seen

    def tick(self, force_hud: bool = False):
        now = time.perf_counter()
        if now - self._last_input >= self._input_gap:
            self._last_input = now
            cam = pb.getDebugVisualizerCamera(physicsClientId=self.c)
            self._read_back(cam)
            mode_before = (self.mode, self.follow)
            self._poll()
            if self.mode == FOLLOW:
                live = self._live()
                if self.follow in live:
                    self.target += (self.w.pos[self.follow] - self.target) * 0.3
                    self._apply_camera()
                elif live:
                    self._follow(live[0])
                else:
                    self._overview()
            elif self.mode == OVERVIEW:
                self._apply_camera()
            else:                                          # free: keep PyBullet's own target (Ctrl-drag pans)
                self.target = np.array(cam[11], dtype=float)
                self._apply_camera()
            if (self.mode, self.follow) != mode_before:
                force_hud = True
            cost = time.perf_counter() - now               # never spend more than ~a quarter of the time here
            self._input_gap = float(np.clip(4.0 * cost, 1 / 60, 0.2))
        if force_hud or now - self._last_hud >= self._hud_gap:
            t0 = time.perf_counter()
            self._last_hud = t0
            self._draw_hud()
            self._hud_gap = float(np.clip(8.0 * (time.perf_counter() - t0), 0.33, 2.0))

    # ------------------------------------------------------------------ HUD
    def _sync_goals(self, force: bool = False):
        core, c = self.core, self.c
        for i in range(core.n):
            on = bool(core.active[i] and self.w.alive[i])
            pos = np.array(core.goal[i] if on else HIDDEN, dtype=float)
            if force or not np.allclose(pos, self.goal_shown[i], atol=0.05, equal_nan=False):
                self.goal_shown[i] = pos
                pb.resetBasePositionAndOrientation(self.goal_bodies[i], pos.tolist(), [0, 0, 0, 1], physicsClientId=c)
                self.goal_labels[i] = pb.addUserDebugText(
                    f"G{i}", (pos + [0, 0, 1.8]).tolist(), textColorRGB=GREEN, textSize=1.3,
                    replaceItemUniqueId=self.goal_labels[i], physicsClientId=c)

    def status_lines(self):
        core, w, c = self.core, self.w, self.counts
        swarm = (f"t {int(self.sim_t)} s   flying {len(self._live())}   at goal {c['goal']}   hit {c['hit']}   "
                 f"crashed {c['crash']}" + ("   PAUSED" if self.paused else (f"   speed x{self.speed:.2g}" if self.speed != 1 else "")))
        if self.banner:
            return self.banner, "next episode starts in a moment", swarm
        if self.mode == FOLLOW and self.follow in self._live():
            i = self.follow
            agl = w.pos[i, 2] - w.city.ground_z(w.pos[i, 0], w.pos[i, 1])
            return (f"FOLLOWING D{i} ({w.kind(i)})   [N] next   [O] overview",
                    f"goal G{i} in {np.linalg.norm(core.goal[i] - w.pos[i]):.0f} m   speed {np.linalg.norm(w.vel[i]):.1f} m/s"
                    f"   height {agl:.0f} m", swarm)
        if self.mode == OVERVIEW:
            return "OVERVIEW   [N] follow a drone   [F] free camera", "D = drone   G = its goal   red ring = defence range", swarm
        return "FREE CAMERA   [N] follow a drone   [O] overview", "drag / arrows rotate   Z X zoom   Ctrl+middle-drag pans", swarm

    def _draw_hud(self):
        """Three status lines in the upper left of the view, attached to the camera anchor."""
        c, w = self.c, self.w
        cam = pb.getDebugVisualizerCamera(physicsClientId=c)
        fwd, up = np.array(cam[5], dtype=float), np.array(cam[4], dtype=float)
        right = np.cross(fwd, up)
        right /= max(np.linalg.norm(right), 1e-9)
        up /= max(np.linalg.norm(up), 1e-9)
        dist = self.dist
        colors = ([0.0, 0.0, 0.0], [0.05, 0.1, 0.35], [0.05, 0.1, 0.35])
        view = (round(cam[8], 1), round(cam[9], 1), round(dist, 2))
        for k, text in enumerate(self.status_lines()):
            if self._hud_state[k] == (text, view):         # redrawing text is the slow part: only when it changed
                continue
            self._hud_state[k] = (text, view)
            offset = -right * 0.36 * dist + up * (0.43 - 0.05 * k) * dist
            self.hud[k] = pb.addUserDebugText(text, offset.tolist(), textColorRGB=colors[k], textSize=1.3,
                                              parentObjectUniqueId=self.anchor,
                                              replaceItemUniqueId=self.hud[k], physicsClientId=c)
        if self.mode == FOLLOW and self.follow in self._live():
            i = self.follow
            self.goal_line = pb.addUserDebugLine(w.pos[i].tolist(), self.core.goal[i].tolist(), GREEN, 2,
                                                 replaceItemUniqueId=self.goal_line, physicsClientId=c)
        elif self.goal_line >= 0:
            pb.removeUserDebugItem(self.goal_line, physicsClientId=c)
            self.goal_line = -1
