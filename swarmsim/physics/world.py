"""The simulation world: loads a city bundle and the vehicle models into
PyBullet and advances everything by one physics step at a time."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as pb

from ..models import ModelsConfig, ensure_models
from ..terrain import World as CityMap
from .config import PhysicsConfig
from .defence import DefencePlatform, Projectiles
from .vehicles import FixedWingFleet, QuadFleet
from .wind import WindField


class World:
    """Typical use::

        world = World(city_dir, models_cfg, physics_cfg, n_quads=6, n_fixedwing=2)
        world.spawn(0, [0, 0, 30])
        world.command_velocity(0, [3, 0, 0])
        world.step()

    Drones have one global index: quadcopters first, then fixed-wings.
    """

    def __init__(self, city_dir: str | Path, models: ModelsConfig | None = None,
                 cfg: PhysicsConfig | None = None, n_quads: int = 4, n_fixedwing: int = 0,
                 defence_positions=(), gui: bool = False, seed: int | None = None):
        self.cfg = cfg or PhysicsConfig()
        self.models = models or ModelsConfig()
        self.dt = 1.0 / self.cfg.physics_hz
        self.gui = gui
        self.rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        self.c = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setGravity(0, 0, -self.cfg.gravity, physicsClientId=self.c)
        pb.setTimeStep(self.dt, physicsClientId=self.c)
        if gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=self.c)

        self.city = CityMap.load(city_dir)
        self.static_bodies = self._load_city(Path(city_dir))
        assets = ensure_models(self.models)
        g, rho = self.cfg.gravity, self.cfg.air_density
        self.quads = QuadFleet(self.c, str(assets / "quadcopter.urdf"), n_quads, self.models.quad, g, rho)
        self.wings = FixedWingFleet(self.c, str(assets / "fixedwing.urdf"), n_fixedwing, self.models.fixedwing, g, rho)
        self.n_quads, self.n = n_quads, n_quads + n_fixedwing
        self.body_ids = self.quads.bodies + self.wings.bodies
        self._body_index = {b: i for i, b in enumerate(self.body_ids)}
        if not self.cfg.drone_collisions:
            for b in self.body_ids:      # collide with the world (group 1) but not with each other
                pb.setCollisionFilterGroupMask(b, -1, 2, 1, physicsClientId=self.c)

        self.defences = [DefencePlatform(self.c, str(assets / "defence.urdf"), self.models.defence,
                                         self._on_ground(p), self.rng) for p in defence_positions]
        self.projectiles = Projectiles()
        self._proj_vis = []
        self.wind = WindField(self.cfg.wind, self.city, seed=int(self.rng.integers(1 << 30)))
        self.crashed = np.zeros(self.n, bool)      # touched something this step
        self.hit = np.zeros(self.n, bool)          # hit by a projectile this step
        self.time = 0.0
        self._tick = 0
        self._wind_cache = [None, None]
        self.out_of_bounds = np.zeros(self.n, bool)
        self._refresh()

    # ------------------------------------------------------------------ setup
    def _load_city(self, d: Path):
        """Load every mesh listed in the terrain module's scene.json."""
        import json
        manifest = json.loads((d / "scene.json").read_text())
        bodies, textures = [], {}
        for e in manifest["meshes"]:
            mesh = str(d / e["file"])
            col = -1
            if e.get("collision", True):
                col = pb.createCollisionShape(pb.GEOM_MESH, fileName=mesh,
                                              flags=pb.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.c)
            vis = pb.createVisualShape(pb.GEOM_MESH, fileName=mesh, physicsClientId=self.c)
            b = pb.createMultiBody(0, col, vis, physicsClientId=self.c)
            if e["texture"] not in textures:
                textures[e["texture"]] = pb.loadTexture(str(d / e["texture"]), physicsClientId=self.c)
            pb.changeVisualShape(b, -1, textureUniqueId=textures[e["texture"]], rgbaColor=[1, 1, 1, 1],
                                 specularColor=[0, 0, 0], physicsClientId=self.c)
            bodies.append(b)
        return bodies

    def _on_ground(self, p):
        x, y = float(p[0]), float(p[1])
        return [x, y, float(self.city.ground_z(x, y))]

    def close(self):
        if pb.isConnected(self.c):
            pb.disconnect(self.c)

    # --------------------------------------------------------------- vehicles
    def _fleet(self, i: int):
        return (self.quads, i) if i < self.n_quads else (self.wings, i - self.n_quads)

    def kind(self, i: int) -> str:
        return "quad" if i < self.n_quads else "fixedwing"

    def spawn(self, i: int, pos, yaw: float = 0.0):
        fleet, j = self._fleet(i)
        fleet.spawn(j, pos, yaw)
        self._wind_cache = [None, None]
        self._refresh()

    def kill(self, i: int):
        fleet, j = self._fleet(i)
        fleet.kill(j)
        self._refresh()

    def command_velocity(self, i: int, vel, yaw_rate: float = 0.0):
        """Quadcopter: world-frame velocity + yaw rate.  Fixed-wing: the velocity is
        converted to turn rate / climb rate / airspeed."""
        fleet, j = self._fleet(i)
        if fleet is self.quads:
            fleet.set_command(j, vel, yaw_rate)
        else:
            v = np.asarray(vel, float)
            speed = float(np.hypot(v[0], v[1]))
            heading = np.arctan2(self.vel[i, 1], self.vel[i, 0])
            err = (np.arctan2(v[1], v[0]) - heading + np.pi) % (2 * np.pi) - np.pi if speed > 0.5 else 0.0
            fleet.set_command(j, turn_rate=1.2 * err + yaw_rate, climb_rate=v[2],
                              airspeed=speed if speed > 0.5 else None)

    def command_fixedwing(self, i: int, turn_rate: float, climb_rate: float, airspeed: float | None = None):
        self.wings.set_command(i - self.n_quads, turn_rate, climb_rate, airspeed)

    def _refresh(self):
        self.pos = np.concatenate([self.quads.pos, self.wings.pos])
        self.vel = np.concatenate([self.quads.vel, self.wings.vel])
        self.R = np.concatenate([self.quads.R, self.wings.R])
        self.alive = np.concatenate([self.quads.alive, self.wings.alive])
        self.radius = np.concatenate([np.full(self.quads.n, self.quads.radius),
                                      np.full(self.wings.n, self.wings.radius)])

    # ------------------------------------------------------------------- step
    def step(self):
        """Advance the simulation by one physics step (1 / physics_hz seconds)."""
        dt = self.dt
        self.wind.step(dt)
        self._tick += 1
        for k, fleet in enumerate((self.quads, self.wings)):
            if fleet.n:
                fleet.read_state()
                if self._tick % 4 == 1 or self._wind_cache[k] is None:     # wind changes slowly: sample at physics_hz / 4
                    self._wind_cache[k] = self.wind.at(fleet.pos)
                fleet.step(dt, self._wind_cache[k])
        self._refresh()

        if self.defences:
            self._step_defences(dt)
        pb.stepSimulation(physicsClientId=self.c)
        self.time += dt

        # collisions of drones with anything
        self.crashed[:] = False
        for i in np.flatnonzero(self.alive):
            if pb.getContactPoints(bodyA=self.body_ids[i], physicsClientId=self.c):
                self.crashed[i] = True
        half = self.city.half
        out = (np.abs(self.pos[:, 0]) > half) | (np.abs(self.pos[:, 1]) > half)
        self.out_of_bounds = out & self.alive

    def _step_defences(self, dt):
        self.hit[:] = False
        spec = self.models.defence
        for d in self.defences:
            visible = self.line_of_sight(d.muzzle, self.pos)
            d.step(dt, self.pos, self.vel, self.alive, visible, self.cfg.gravity, self.projectiles)
        act = self.projectiles.step(dt, self.cfg.gravity, spec.projectile_drag,
                                    spec.projectile_lifetime, self.wind.at)
        if len(act):
            P0, P1 = self.projectiles.prev[act], self.projectiles.pos[act]
            # hits on drones: closest approach of each projectile segment to each drone
            alive = np.flatnonzero(self.alive)
            if len(alive):
                seg = P1 - P0
                L2 = np.maximum((seg ** 2).sum(1), 1e-9)
                rel = self.pos[alive][None] - P0[:, None]                       # (P, D, 3)
                t = np.clip((rel * seg[:, None]).sum(-1) / L2[:, None], 0, 1)
                dmin = np.linalg.norm(rel - t[..., None] * seg[:, None], axis=-1)
                near = dmin < spec.hit_radius
                self.hit[alive[near.any(0)]] = True
                self.projectiles.active[act[near.any(1)]] = False
            # hits on terrain / buildings
            res = pb.rayTestBatch(P0.tolist(), P1.tolist(), physicsClientId=self.c)
            for k, r in enumerate(res):
                if r[0] in self.static_bodies:
                    self.projectiles.active[act[k]] = False
        if self.gui or self._proj_vis:
            self._draw_projectiles()

    # ---------------------------------------------------------------- sensing
    def line_of_sight(self, origin, targets) -> np.ndarray:
        targets = np.atleast_2d(targets)
        if not len(targets):
            return np.zeros(0, bool)
        res = pb.rayTestBatch([list(origin)] * len(targets), targets.tolist(), physicsClientId=self.c)
        return np.array([r[0] not in self.static_bodies for r in res])

    def raycast(self, origins: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
        """Distance to the first obstacle along each ray (== max_range if nothing is hit).
        Rays start slightly outside the drone so it does not see itself."""
        o = np.asarray(origins, float); dvec = np.asarray(directions, float)
        start = o + 0.45 * dvec
        res = pb.rayTestBatch(start.tolist(), (o + max_range * dvec).tolist(), physicsClientId=self.c)
        frac = np.array([r[2] for r in res])
        return 0.45 + frac * (max_range - 0.45)

    # -------------------------------------------------------------- rendering
    def enable_projectile_visuals(self, n: int = 24, radius: float = 0.25):
        vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius, rgbaColor=[1.0, 0.75, 0.1, 1],
                                   physicsClientId=self.c)
        self._proj_vis = [pb.createMultiBody(0, -1, vis, [0, 0, -500], physicsClientId=self.c)
                          for _ in range(n)]

    def _draw_projectiles(self):
        if not self._proj_vis:
            self.enable_projectile_visuals()
        act = np.flatnonzero(self.projectiles.active)
        for k, b in enumerate(self._proj_vis):
            p = self.projectiles.pos[act[k]].tolist() if k < len(act) else [0, 0, -500]
            pb.resetBasePositionAndOrientation(b, p, [0, 0, 0, 1], physicsClientId=self.c)

    def render(self, width=640, height=480, eye=None, target=None, fov=60.0) -> np.ndarray:
        """Software-rendered RGB image (works headless)."""
        if eye is None:
            h = self.city.half
            eye, target = [-1.1 * h, -1.3 * h, 0.9 * h], [0, 0, 10]
        vm = pb.computeViewMatrix(list(eye), list(target), [0, 0, 1])
        pm = pb.computeProjectionMatrixFOV(fov, width / height, 0.3, 4 * self.city.size)
        img = pb.getCameraImage(width, height, vm, pm, renderer=pb.ER_TINY_RENDERER,
                                lightDirection=[0.4, -0.6, 1.0], shadow=1, physicsClientId=self.c)
        return np.reshape(img[2], (height, width, 4))[:, :, :3].astype(np.uint8)


PhysicsWorld = World      # alias, to tell it apart from swarmsim.terrain.World
