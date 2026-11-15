"""The live simulation behind the app: one thread that owns the PyBullet world,
flies the chosen policy in real time and publishes a snapshot after every step."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..config import load_config
from . import scenario
from ..rl.wingpilot import SHAPE_P
from .cloud import city_cloud
from .trainer import BUILTIN, DEFAULT_CONFIG, RUNS

RESPAWN_DELAY_S = 1.5         # a finished drone stays away this long (simulated time) before it starts again
PILOT_TIMEOUT_S = 1.0         # manual input older than this is dropped (browser tab closed or frozen)
OUTCOMES = ("goal", "hit", "crash", "timeout", "out_of_bounds")
SETUP_KEYS = ("city", "n_quads", "n_fixedwing", "n_defence")


class Sim(threading.Thread):
    def __init__(self, model: str = "stage1_navigate"):
        super().__init__(daemon=True, name="swarmsim-sim")
        self.cond = threading.Condition()
        self.snapshot = json.dumps({"seq": 0, "status": "loading", "message": "Starting the simulator"})
        self.seq = 0
        self.scene: dict | None = None            # static description of what is loaded (see _describe)
        self.assets: dict[str, tuple[Path, float]] = {}   # asset id -> (file, scale) the server may hand out
        self.cmds: queue.Queue = queue.Queue()
        self.core = self.policy = self.obs = None
        self.model = model
        self.setup: dict = {}                     # what the person changed: city, n_quads, n_fixedwing, n_defence
        self.city = None                          # city folder that is loaded
        self.starts: dict[int, list] = {}         # fixed points, kept across model changes on the same city
        self.goals: dict[int, list] = {}
        self.defence_xy: dict[int, list] = {}
        self.status, self.message = "loading", "Starting the simulator"
        self.paused, self.speed, self.seed = False, 1.0, 100
        self.pilot: int | None = None
        self.pilot_cmd, self.pilot_time = (0.0, 0.0, 0.0, 0.0), 0.0
        self.stats = dict.fromkeys(OUTCOMES, 0)
        self.events: list[dict] = []
        self.event_id = 0
        self.respawn_at: dict[int, float] = {}
        self.loads = 0
        self.run_id = os.urandom(4).hex()          # in asset URLs, so a browser never reuses files cached by an earlier run
        self.stopping = False

    # ------------------------------------------------------------- from the server
    def send(self, cmd: dict):
        """Fire and forget."""
        if cmd.get("type") == "keys":             # latest value wins: no need to queue
            c = [float(np.clip(v, -1, 1)) for v in (list(cmd.get("cmd", [])) + [0, 0, 0, 0])[:4]]
            self.pilot_cmd, self.pilot_time = tuple(c), time.time()
        else:
            self.cmds.put(cmd)

    def call(self, cmd: dict, timeout: float = 300.0) -> dict:
        """Run a command on the simulation thread and wait for its answer."""
        reply: queue.Queue = queue.Queue(maxsize=1)
        self.cmds.put({**cmd, "_reply": reply})
        try:
            return reply.get(timeout=timeout)
        except queue.Empty:
            return {"error": "The simulator did not answer in time."}

    # -------------------------------------------------------------------- thread
    def run(self):
        self._load()
        while not self.stopping:
            t0 = time.perf_counter()
            try:
                changed = self._commands()
                if self.core is not None and self.status == "running" and not self.paused:
                    self._step()
                    self._publish()
                    dt = 1.0 / (self.core.rl.policy_hz * self.speed)
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                    continue
                if changed:
                    self._publish()
            except Exception as exc:              # keep the app alive and show what went wrong
                traceback.print_exc()
                self.status, self.message = "error", f"{type(exc).__name__}: {exc}"
                self._publish()
            time.sleep(0.03)
        if self.core is not None:
            self.core.close()

    def _commands(self) -> bool:
        changed = False
        while True:
            try:
                cmd = self.cmds.get_nowait()
            except queue.Empty:
                return changed
            changed = True
            reply = cmd.pop("_reply", None)
            try:
                out = self._command(cmd) or {}
            except ValueError as exc:              # a message for the person at the page
                out = {"error": str(exc)}
            except Exception as exc:
                traceback.print_exc()
                out = {"error": f"{type(exc).__name__}: {exc}"}
            if reply is not None:
                reply.put({"ok": True, **out} if "error" not in out else out)

    def _command(self, cmd: dict) -> dict | None:
        kind = cmd.get("type")
        if kind == "load":
            return self._reload(model=str(cmd.get("model", self.model)))
        if kind == "setup":
            if cmd.get("defaults"):
                return self._reload(setup={})
            new = {"city": scenario.check_city(str(cmd.get("city"))),
                   **scenario.check_counts(cmd.get("n_quads"), cmd.get("n_fixedwing"), cmd.get("n_defence"))}
            return self._reload(setup=new)
        if kind == "generate":
            keys = ("n_quads", "n_fixedwing", "n_defence")
            counts = scenario.check_counts(*(cmd[k] for k in keys)) if all(k in cmd for k in keys) else {}
            self.status, self.message = "loading", "Building a new city"
            self._publish()
            try:
                city = scenario.generate_city(str(cmd.get("style")), str(cmd.get("pattern")),
                                              cmd.get("size", 0), cmd.get("seed", 0))
            except Exception:
                self.status = "running" if self.core is not None else "error"
                raise
            return {**self._reload(setup={**self.setup, **counts, "city": city.as_posix()}), "city": city.as_posix()}
        if kind == "place":
            return self._place(cmd)
        if kind == "unplace":
            return self._unplace(cmd)
        if kind == "pause":
            self.paused = bool(cmd.get("value", not self.paused))
        elif kind == "speed":
            self.speed = float(np.clip(float(cmd.get("value", 1.0)), 0.25, 8.0))
        elif kind == "reset" and self.core is not None:
            self.seed += 1
            self._reset()
        elif kind == "pilot":
            d = cmd.get("drone")
            ok = self.core is not None and isinstance(d, int) and 0 <= d < self.core.n
            self.pilot = d if ok else None
            self.pilot_cmd = (0.0, 0.0, 0.0, 0.0)
        return None

    # ---------------------------------------------------------------------- load
    def _reload(self, model: str | None = None, setup: dict | None = None) -> dict:
        """Load a different model or scenario; on failure go back to what worked."""
        old = (self.model, self.setup)
        if model is not None:
            self.model = model
        if setup is not None:
            self.setup = setup
        error = self._load()
        if error:
            self.model, self.setup = old
            self._load()
            raise ValueError(error)
        return {}

    def _load(self) -> str | None:
        self.status, self.message = "loading", f"Loading {self.model}"
        self._publish()
        if self.core is not None:
            self.core.close()
            self.core = None
        try:
            from ..rl.core import SwarmCore, ensure_city
            from ..rl.plugin import load_policy
            if self.model in BUILTIN:
                config, spec = DEFAULT_CONFIG, self.model
            else:
                run = RUNS / self.model
                if not (run / "model.zip").exists():
                    raise FileNotFoundError(f"No trained model in runs/{self.model}")
                config, spec = run / "config.yaml", f"sb3:{(run / 'model.zip').as_posix()}"
            cfg = load_config(config)
            model_city = Path(ensure_city(cfg)).as_posix()
            s = self.setup
            cfg.rl = replace(cfg.rl, **{k: s[k] for k in ("n_quads", "n_fixedwing", "n_defence") if k in s},
                             city_dir=s.get("city", model_city), shared_goal=True,   # one goal for every drone
                             collision_reflex=True)        # obstacle braking, as in the paper's results
            city = cfg.rl.city_dir
            if city != self.city:                 # fixed points belong to one map
                self.starts.clear(); self.goals.clear(); self.defence_xy.clear()
            self.city = city
            self.core = SwarmCore(cfg, seed=self.seed)
            self.policy = load_policy(spec, self.core.info())
            if hasattr(self.policy, "bind"):
                self.policy.bind(self.core)
            n, nd = self.core.n, len(self.core.world.defences)
            for d, size in ((self.starts, n), (self.goals, n), (self.defence_xy, nd)):
                for k in [k for k in d if k >= size]:
                    del d[k]
            self.core.start_at, self.core.goal_at, self.core.defence_at = self.starts, self.goals, self.defence_xy
            self.pilot = None
            self.loads += 1
            self._describe(model_city)
            self._reset()
            self.status, self.message = "running", ""
            error = None
        except Exception as exc:
            traceback.print_exc()
            if self.core is not None:
                self.core.close()
                self.core = None
            self.status, self.message = "error", f"Could not load {self.model}: {exc}"
            error = self.message
        self._publish()
        return error

    def _describe(self, model_city: str):
        """Static facts about the loaded world; the page fetches them once per load."""
        core, cfg = self.core, self.core.cfg
        city_dir = Path(self.city)
        assets = {}

        def asset(path, scale: float = 1.0) -> str:
            key = f"a{len(assets)}"
            assets[key] = (path, scale)
            return f"/api/asset/{self.asset_tag()}/{key}"

        clouds = [{"kind": k, "url": asset(("cloud", self.city, k)), "count": v["count"], "spacing": v["spacing"]}
                  for k, v in city_cloud(city_dir).items()]
        models = Path(cfg.models.assets_dir)
        d = cfg.models.defence
        self.assets = assets
        self.scene = {
            "id": self.loads, "model": self.model, "size": float(core.city.size),
            "city": self.city, "model_city": model_city, "setup": dict(self.setup),
            "clouds": clouds,
            "vehicles": {k: asset(models / f"{f}.stl", 0.001)
                         for k, f in (("quad", "quadcopter"), ("wing", "fixedwing"), ("defence", "defence"))},
            "n": core.n, "kinds": [core.world.kind(i) for i in range(core.n)],
            "sizes": {"quad": 2 * (cfg.models.quad.arm_length + cfg.models.quad.prop_radius),
                      "wing": cfg.models.fixedwing.span},
            "n_quads": cfg.rl.n_quads, "n_fixedwing": cfg.rl.n_fixedwing,
            "n_defence": len(core.world.defences), "defence_range": d.range_m,
            "goal_radius": cfg.rl.goal_radius, "task": cfg.rl.task, "policy_hz": cfg.rl.policy_hz,
            "fw_mode": cfg.rl.fw_mode, "fw_detect_range": cfg.rl.fw_detect_range, "loop_shape": SHAPE_P,
            "move_speed": cfg.rl.move_speed, "max_episode_s": cfg.rl.max_episode_s,
            "ceiling_agl": cfg.rl.ceiling_agl,
        }

    def asset_tag(self) -> str:
        return f"{self.run_id}-{self.loads}"

    def _reset(self):
        self.obs = self.core.reset(self.seed)
        self.policy.reset()
        self.respawn_at.clear()
        self.stats = dict.fromkeys(OUTCOMES, 0)
        self.events = []

    # ----------------------------------------------------------------- placement
    def probe(self, cmd: dict) -> dict:
        """What a click at this camera ray would place (called from the web server's
        threads while the mouse moves; reads only map data, never PyBullet)."""
        core = self.core
        if core is None or self.status != "running":
            return {"ok": False, "reason": "Loading"}
        try:
            out = scenario.resolve(core, cmd)
        except ValueError as exc:
            hit = scenario.cast(core.city, cmd.get("eye"), cmd.get("dir"))
            return {"ok": False, "reason": str(exc),
                    "point": None if hit is None else np.round(hit, 2).tolist()}
        return {"ok": True, **out, "points": {str(k): v for k, v in out["points"].items()},
                "grounds": {str(k): v for k, v in out["grounds"].items()}}

    def _place(self, cmd: dict) -> dict:
        if self.core is None:
            raise ValueError("Nothing is loaded yet.")
        core, w = self.core, self.core.world
        out = scenario.resolve(core, cmd)
        points = out["points"]
        if out["what"] == "defence":
            for k, p in points.items():
                self.defence_xy[k] = p
                core.place_defence(k)
        elif out["what"] == "start":
            self.starts.update(points)
            self._respawn(list(points))
        else:
            self.goals.update(points)
            self._set_goal(points)
        return {"note": out["note"], "label": out["label"]}

    def _set_goal(self, points: dict):
        """New goal(s) for these drones, straight away. Planes have not seen it yet: they search again."""
        core, w = self.core, self.core.world
        for i, p in points.items():
            if core.active[i]:
                core.goal[i] = p
                core.prev_dist[i] = float(np.linalg.norm(core._target(i) - w.pos[i]))
        if core.wingpilot:
            core.wingpilot.reset()

    def _unplace(self, cmd: dict) -> dict:
        if self.core is None:
            raise ValueError("Nothing is loaded yet.")
        what, index = cmd.get("what"), cmd.get("index")
        if what == "defence":
            ks = range(len(self.core.world.defences)) if index == "all" else [index]
            for k in ks:
                if isinstance(k, int) and 0 <= k < len(self.core.world.defences):
                    self.defence_xy.pop(k, None)
                    self.core.place_defence(k)
            return {}
        if what == "goal":                             # one new random goal, shared by every drone
            self.goals.clear()
            core = self.core
            g = core._sample_point(core.rl.goal_agl, min_sep=0.0)
            self._set_goal({i: g.copy() for i in range(core.n)})
            return {}
        drones = scenario.drones_of(self.core, index)
        for i in drones:
            self.starts.pop(i, None)
            if what != "start":
                self.goals.pop(i, None)
        self._respawn(drones)
        return {}

    def _respawn(self, drones: list[int]):
        """Start these drones again now, all out of the way first so they cannot meet."""
        w = self.core.world
        for i in drones:
            if w.alive[i]:
                w.kill(i)
            self.respawn_at.pop(i, None)
        for i in drones:
            self.core.respawn(i)
        self.obs = self.core.observe()

    # ---------------------------------------------------------------------- step
    def _step(self):
        core, w = self.core, self.core.world
        fresh = time.time() - self.pilot_time < PILOT_TIMEOUT_S
        core.manual = {self.pilot: self.pilot_cmd if fresh else (0.0, 0.0, 0.0, 0.0)} if self.pilot is not None else {}
        before = w.pos.copy()
        obs, _, _, _, outcome = core.step(self.policy.act(self.obs))
        for i, o in enumerate(outcome):
            if o:
                self.stats[o] += 1
                self.event_id += 1
                self.events.append({"id": self.event_id, "drone": i, "outcome": o,
                                    "pos": np.round(before[i], 2).tolist()})
                self.respawn_at[i] = w.time + RESPAWN_DELAY_S
        if core.wingpilot:
            for i in core.wingpilot.pop_found():       # a plane's target was spotted: it heads there now
                self.event_id += 1
                self.events.append({"id": self.event_id, "drone": i, "outcome": "found",
                                    "pos": np.round(w.pos[i], 2).tolist()})
        self.events = self.events[-24:]
        due = [i for i, t in self.respawn_at.items() if w.time >= t]
        for i in due:
            core.respawn(i)
            del self.respawn_at[i]
        self.obs = core.observe() if due else obs

    # ------------------------------------------------------------------- publish
    def _publish(self):
        snap = {"status": self.status, "message": self.message, "model": self.model,
                "paused": self.paused, "speed": self.speed, "pilot": self.pilot,
                "scene": self.scene["id"] if self.scene and self.core is not None else None}
        core = self.core
        if core is not None and self.status == "running":
            w = core.world
            on = (w.alive & core.active).astype(float)
            agl = w.pos[:, 2] - core.city.ground_z(w.pos[:, 0], w.pos[:, 1])
            dist = np.linalg.norm(np.stack([core._target(i) for i in range(core.n)]) - w.pos, axis=1)
            rows = np.concatenate([w.pos, w.R.reshape(core.n, 9), on[:, None],
                                   np.linalg.norm(w.vel, axis=1)[:, None], agl[:, None], dist[:, None]], 1)
            act = np.flatnonzero(w.projectiles.active)
            snap.update(
                time=round(float(w.time), 2),
                drones=np.round(rows, 3).tolist(),
                goals=np.round(core.goal, 2).tolist(),
                defences=[[round(float(v), 3) for v in (*d.base, d.az, d.el)] + [int(d.target), int(d.ammo)]
                          for d in w.defences],
                projectiles=np.round(np.concatenate([w.projectiles.pos[act], w.projectiles.vel[act]], 1), 2).tolist(),
                fixed={"starts": self.starts, "goals": self.goals, "defences": self.defence_xy},
                wings=({"modes": core.wingpilot.modes(), "loop": np.round(core.wingpilot.radius, 1).tolist()}
                       if core.wingpilot else None),
                stats=self.stats, events=self.events)
        with self.cond:
            self.seq += 1
            snap["seq"] = self.seq
            self.snapshot = json.dumps(snap, separators=(",", ":"))
            self.cond.notify_all()
