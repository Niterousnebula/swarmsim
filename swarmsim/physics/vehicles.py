"""Flight dynamics of the two drone types, batched over the whole fleet.

PyBullet integrates the rigid bodies; this file computes the forces and
torques that act on them every physics step (propulsion, aerodynamics, wind)
and runs the low-level flight controllers that turn high-level commands
(velocity / turn-rate set-points) into those forces.

Body frame for both types: x forward, y left, z up.
"""
from __future__ import annotations

import numpy as np
import pybullet as pb

from ..models.specs import FixedWingSpec, QuadSpec
from .mathutil import cross, euler_from_mat, normalize, quat_to_mat, wrap, yaw_to_quat

GRAVEYARD_Z = -200.0


class Fleet:
    """State shared by both vehicle types (arrays of length n)."""
    kind = "base"

    def __init__(self, client: int, urdf: str, n: int, radius: float):
        self.c, self.n, self.radius = client, n, radius
        self.bodies = []
        for i in range(n):
            b = pb.loadURDF(urdf, [3.0 * i, 0, GRAVEYARD_Z - 5.0 * i], physicsClientId=client)
            pb.changeDynamics(b, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=client)
            self.bodies.append(b)
        self.pos = np.zeros((n, 3)); self.vel = np.zeros((n, 3))
        self.R = np.tile(np.eye(3), (n, 1, 1)); self.omega_b = np.zeros((n, 3))
        self.alive = np.zeros(n, bool)

    def read_state(self):
        quat = np.empty((self.n, 4)); w = np.empty((self.n, 3))
        for i, b in enumerate(self.bodies):
            p, q = pb.getBasePositionAndOrientation(b, physicsClientId=self.c)
            v, om = pb.getBaseVelocity(b, physicsClientId=self.c)
            self.pos[i] = p; quat[i] = q; self.vel[i] = v; w[i] = om
        self.R = quat_to_mat(quat)
        self.omega_b = np.einsum("nji,nj->ni", self.R, w)

    def _place(self, i, pos, quat, vel):
        pb.resetBasePositionAndOrientation(self.bodies[i], list(pos), quat, physicsClientId=self.c)
        pb.resetBaseVelocity(self.bodies[i], list(vel), [0, 0, 0], physicsClientId=self.c)
        self.pos[i] = pos; self.vel[i] = vel; self.omega_b[i] = 0
        self.R[i] = quat_to_mat(np.array([quat]))[0]

    def kill(self, i: int):
        """Remove a drone from play (parked far below the map)."""
        self.alive[i] = False
        self._place(i, [3.0 * i, 0, GRAVEYARD_Z - 5.0 * i - (50 if self.kind == "fixedwing" else 0)],
                    [0, 0, 0, 1], [0, 0, 0])

    def _hold_dead(self):
        for i in np.flatnonzero(~self.alive):
            pb.resetBaseVelocity(self.bodies[i], [0, 0, 0], [0, 0, 0], physicsClientId=self.c)
            pb.applyExternalForce(self.bodies[i], -1, [0, 0, self._weight], [0, 0, 0], pb.LINK_FRAME,
                                  physicsClientId=self.c)

    def _apply(self, F_world: np.ndarray, tau_world: np.ndarray):
        for i in np.flatnonzero(self.alive):
            b = self.bodies[i]
            pb.applyExternalForce(b, -1, F_world[i].tolist(), self.pos[i].tolist(), pb.WORLD_FRAME,
                                  physicsClientId=self.c)
            pb.applyExternalTorque(b, -1, tau_world[i].tolist(), pb.WORLD_FRAME, physicsClientId=self.c)


# ---------------------------------------------------------------------------
class QuadFleet(Fleet):
    """Quadcopters with four individually saturating, lagging rotors.

    Command per drone: desired world-frame velocity (vx, vy, vz) and yaw rate.
    A geometric attitude controller tracks it.
    """
    kind = "quad"

    def __init__(self, client, urdf, n, spec: QuadSpec, gravity: float, rho: float):
        super().__init__(client, urdf, n, spec.collision_radius)
        self.spec, self.g, self.rho = spec, gravity, rho
        self._weight = spec.mass * gravity
        self.J = np.array(spec.inertia)
        self.f_max = spec.thrust_to_weight * spec.mass * gravity / 4.0
        l = spec.arm_length / np.sqrt(2)
        # rotor order: front-left, rear-left, rear-right, front-right; spin alternates
        xs = np.array([l, -l, -l, l]); ys = np.array([l, l, -l, -l]); spin = np.array([1, -1, 1, -1])
        self.alloc = np.stack([np.ones(4), ys, -xs, spin * spec.torque_to_thrust])   # [T, tx, ty, tz] = alloc @ f
        self.alloc_inv = np.linalg.inv(self.alloc)
        self.motor = np.zeros((n, 4))                 # current rotor thrusts (N)
        self.cmd_vel = np.zeros((n, 3)); self.cmd_yaw_rate = np.zeros(n)
        self.yaw_sp = np.zeros(n)
        # controller gains (natural frequency / damping of the attitude loop)
        # Yaw is the weak axis of a quadcopter (it only has rotor reaction torque), so its
        # loop is much softer than roll/pitch and it is the first thing given up on saturation.
        wn_xy, wn_z, zeta = 12.0, 2.5, 0.9
        self.kR = np.array([wn_xy ** 2, wn_xy ** 2, wn_z ** 2]) * self.J
        self.kW = 2 * zeta * np.array([wn_xy, wn_xy, wn_z]) * self.J
        self.yaw_torque_max = 0.6 * spec.torque_to_thrust * self._weight
        self.kv = np.array([2.5, 2.5, 4.0])

    def spawn(self, i: int, pos, yaw: float = 0.0):
        self._place(i, pos, yaw_to_quat(yaw), [0, 0, 0])
        self.alive[i] = True
        self.motor[i] = self._weight / 4.0
        self.cmd_vel[i] = 0; self.cmd_yaw_rate[i] = 0; self.yaw_sp[i] = yaw
        self.pos[i] = pos; self.vel[i] = 0

    def set_command(self, i: int, vel, yaw_rate: float = 0.0):
        s = self.spec
        v = np.asarray(vel, float).copy()
        h = np.hypot(v[0], v[1])
        if h > s.max_speed:
            v[:2] *= s.max_speed / h
        v[2] = np.clip(v[2], -s.max_climb, s.max_climb)
        self.cmd_vel[i] = v
        self.cmd_yaw_rate[i] = np.clip(yaw_rate, -s.max_yaw_rate, s.max_yaw_rate)

    def step(self, dt: float, wind: np.ndarray):
        """Compute and apply forces for one physics step (call after read_state)."""
        s, m, g = self.spec, self.spec.mass, self.g
        yaw = np.arctan2(self.R[:, 1, 0], self.R[:, 0, 0])
        # yaw-rate command -> pure rate control; zero command -> hold the heading reached
        self.yaw_sp = np.where(np.abs(self.cmd_yaw_rate) > 1e-3, yaw, self.yaw_sp)
        self.yaw_sp = wrap(yaw + np.clip(wrap(self.yaw_sp - yaw), -0.6, 0.6))

        # ---- outer loop: velocity -> desired acceleration ------------------
        acc = self.kv * (self.cmd_vel - self.vel)
        a_h_max = g * np.tan(np.radians(s.max_tilt_deg))
        h = np.linalg.norm(acc[:, :2], axis=1)
        acc[:, :2] *= np.minimum(1.0, a_h_max / np.maximum(h, 1e-9))[:, None]
        acc[:, 2] = np.clip(acc[:, 2], -0.6 * g, 1.0 * g)
        v_rel = self.vel - wind
        drag = -0.5 * self.rho * s.drag_area * np.linalg.norm(v_rel, axis=1, keepdims=True) * v_rel
        F_des = m * (acc + np.array([0, 0, g])) - drag          # feed-forward: lean into the wind

        # ---- attitude target and geometric attitude controller -------------
        b3 = self.R[:, :, 2]
        thrust = np.clip(np.einsum("ni,ni->n", F_des, b3), 0.0, 4 * self.f_max)
        b3d = normalize(F_des)
        b1c = np.stack([np.cos(self.yaw_sp), np.sin(self.yaw_sp), np.zeros(self.n)], 1)
        b2d = normalize(cross(b3d, b1c))
        b1d = cross(b2d, b3d)
        Rd = np.stack([b1d, b2d, b3d], axis=2)
        E = np.einsum("nji,njk->nik", Rd, self.R) - np.einsum("nji,njk->nik", self.R, Rd)
        eR = 0.5 * np.stack([E[:, 2, 1], E[:, 0, 2], E[:, 1, 0]], 1)
        w_des = np.zeros((self.n, 3)); w_des[:, 2] = self.cmd_yaw_rate
        tau = -self.kR * eR - self.kW * (self.omega_b - w_des) + cross(self.omega_b, self.J * self.omega_b)

        # ---- motor mixing, saturation and lag -------------------------------
        tau[:, 2] = np.clip(tau[:, 2], -self.yaw_torque_max, self.yaw_torque_max)
        wrench = np.concatenate([thrust[:, None], tau[:, :2], np.zeros((self.n, 1))], 1)
        f0 = wrench @ self.alloc_inv.T                        # thrust + roll/pitch first ...
        fy = tau[:, 2:3] * self.alloc_inv[:, 3][None]          # ... then as much yaw as still fits
        with np.errstate(divide="ignore", invalid="ignore"):
            room = np.where(fy > 0, (self.f_max - f0) / fy, np.where(fy < 0, -f0 / fy, 1.0))
        scale = np.clip(np.nanmin(room, axis=1), 0.0, 1.0)[:, None]
        f_cmd = np.clip(f0 + scale * fy, 0.0, self.f_max)
        self.motor += (f_cmd - self.motor) * min(1.0, dt / s.motor_time_constant)
        actual = self.motor @ self.alloc.T                    # [T, tx, ty, tz] really produced

        F = b3 * actual[:, :1] + drag
        tau_w = np.einsum("nij,nj->ni", self.R, actual[:, 1:])
        self._apply(F, tau_w)
        self._hold_dead()


# ---------------------------------------------------------------------------
class FixedWingFleet(Fleet):
    """Fixed-wing aircraft: lift/drag/side-force from the airflow, propeller thrust,
    and a rate-command autopilot.

    Command per aircraft: turn rate (rad/s, + = left), climb rate (m/s), airspeed (m/s).
    Forces are physical (the aircraft stalls and sinks if it gets slow).  Rotational
    dynamics are simplified: the autopilot commands body rates, with authority that
    scales with dynamic pressure plus weathercock stability terms.
    """
    kind = "fixedwing"

    def __init__(self, client, urdf, n, spec: FixedWingSpec, gravity: float, rho: float):
        super().__init__(client, urdf, n, spec.collision_radius)
        self.spec, self.g, self.rho = spec, gravity, rho
        self._weight = spec.mass * gravity
        self.J = np.array(spec.inertia)
        self.k_ind = 1.0 / (np.pi * spec.oswald * spec.aspect_ratio)
        self.cmd_turn = np.zeros(n); self.cmd_climb = np.zeros(n)
        self.cmd_speed = np.full(n, spec.cruise_speed)
        self.throttle = np.full(n, 0.5)
        self.airspeed = np.full(n, spec.cruise_speed)
        self.alpha = np.zeros(n)
        self.int_gamma = np.zeros(n)                  # integrator of the flight-path loop

    def spawn(self, i: int, pos, yaw: float = 0.0, speed: float | None = None):
        v = speed or self.spec.cruise_speed
        vel = [v * np.cos(yaw), v * np.sin(yaw), 0.0]
        self._place(i, pos, yaw_to_quat(yaw, pitch_up=0.02), vel)
        self.alive[i] = True
        self.cmd_turn[i] = 0; self.cmd_climb[i] = 0; self.cmd_speed[i] = v; self.throttle[i] = 0.5
        self.int_gamma[i] = 0.0; self.airspeed[i] = v
        self.pos[i] = pos; self.vel[i] = vel

    def set_command(self, i: int, turn_rate: float, climb_rate: float, airspeed: float | None = None):
        s = self.spec
        max_turn = self.g * np.tan(np.radians(s.max_bank_deg)) / max(self.airspeed[i], s.min_speed)
        self.cmd_turn[i] = np.clip(turn_rate, -max_turn, max_turn)
        self.cmd_climb[i] = np.clip(climb_rate, -s.max_descent, s.max_climb)
        if airspeed is not None:
            self.cmd_speed[i] = np.clip(airspeed, s.min_speed, s.max_speed)

    def step(self, dt: float, wind: np.ndarray):
        s, g = self.spec, self.g
        S, b, c = s.wing_area, s.span, s.chord
        xb, yb = self.R[:, :, 0], self.R[:, :, 1]

        # ---- air data -------------------------------------------------------
        v_air = self.vel - wind
        V = np.maximum(np.linalg.norm(v_air, axis=1), 0.1)
        v_hat = v_air / V[:, None]
        vb = np.einsum("nji,nj->ni", self.R, v_air)           # airflow in body axes
        alpha = np.arctan2(-vb[:, 2], np.maximum(vb[:, 0], 0.1))
        beta = np.arcsin(np.clip(vb[:, 1] / V, -1, 1))
        qbar = 0.5 * self.rho * V ** 2
        self.airspeed, self.alpha = V, alpha

        # ---- aerodynamic forces --------------------------------------------
        alpha_stall = (s.cl_max - s.cl0) / s.cl_alpha
        cl = np.clip(s.cl0 + s.cl_alpha * alpha, -0.8 * s.cl_max, s.cl_max)
        post = np.clip((alpha - alpha_stall) / 0.25, 0.0, 1.0)             # lift collapses past the stall
        cl = cl * (1.0 - 0.5 * post)
        cd = s.cd0 + self.k_ind * cl ** 2 + 1.2 * np.sin(alpha) ** 2 * (post + (alpha < -0.25))
        lift_dir = normalize(cross(v_hat, yb))
        F = (qbar * S * cl)[:, None] * lift_dir - (qbar * S * cd)[:, None] * v_hat \
            - (qbar * S * s.side_force_coeff * beta)[:, None] * yb

        # ---- autopilot -------------------------------------------------------
        roll, pitch_up, _ = euler_from_mat(self.R)
        gamma = np.arcsin(
         np.clip(v_air[:, 2] / np.maximum(V, 1.0), -1, 1)
            )  # flight-path angle (nose up = positive)
        # commanded flight-path angle, limited by the spec. The climb angle eases out as
        # airspeed falls towards min_speed; the descent angle is a fixed envelope limit.
        fade = s.envelope_fade_speed
        up = np.radians(s.max_climb_deg) * np.clip((V - s.min_speed) / fade, 0.0, 1.0)
        down = np.radians(s.max_descent_deg)
        climb = np.clip(self.cmd_climb, -s.max_descent, s.max_climb)
        gamma_cmd = np.clip(np.arcsin(np.clip(climb / np.maximum(V, 1.0), -1.0, 1.0)), -down, up)
        roll_cmd = np.clip(np.arctan(-self.cmd_turn * V / g), -np.radians(s.max_bank_deg), np.radians(s.max_bank_deg))
        turn_now = -g * np.tan(np.clip(roll, -1.2, 1.2)) / V              # turn rate the current bank produces
        w_cmd = self.R[:, 2, :] * turn_now[:, None]                        # R^T [0, 0, turn]
        w_cmd[:, 0] += 12.0 * (roll_cmd - roll)
        self.int_gamma = np.clip(self.int_gamma + (gamma_cmd - gamma) * dt, -0.3, 0.3) * self.alive
        pitch_target = np.clip(gamma_cmd + 0.03, -np.radians(s.max_pitch_down_deg), np.radians(s.max_pitch_deg))

        pitch_err = (
            4.5 * (gamma_cmd - gamma)
            + 1.5 * self.int_gamma
            + 14.5 * (pitch_target - pitch_up)
        )
        w_cmd[:, 1] += -1.0 * pitch_err                                          # +y rotation is nose down
        authority = np.clip((V / s.min_speed) ** 2, 0.0, 1.0)
        tau = self.J * np.array([8.0, 10.0, 4.0]) * (w_cmd - self.omega_b) * authority[:, None]
        alpha_trim = (self._weight / np.maximum(qbar * S, 1.0) - s.cl0) / s.cl_alpha   # level-flight incidence
        tau[:, 1] += qbar * S * c * 0.5 * np.clip(alpha - np.clip(alpha_trim, -0.1, alpha_stall), -0.6, 0.6)  # pitch stiffness
        tau[:, 2] += qbar * S * b * 0.12 * beta                                # weathercock (yaw into wind)
        tau += cross(self.omega_b, self.J * self.omega_b)

        # ---- propulsion ------------------------------------------------------
        avail = s.max_thrust * np.clip(1.0 - V / (2.2 * s.max_speed), 0.2, 1.0)
        q_cmd = 0.5 * self.rho * self.cmd_speed ** 2 * S
        cl_need = self._weight / (q_cmd * np.cos(np.clip(roll, -1.1, 1.1)))
        drag_ff = q_cmd * (s.cd0 + self.k_ind * cl_need ** 2)                 # drag at the commanded speed
        thr_cmd = (drag_ff + self._weight * np.sin(gamma_cmd)) / avail + 0.15 * (self.cmd_speed - V)
        self.throttle += (np.clip(thr_cmd, 0.0, 1.0) - self.throttle) * min(1.0, dt / 0.3)
        thrust = self.throttle * avail
        F += thrust[:, None] * xb

        self._apply(F, np.einsum("nij,nj->ni", self.R, tau))
        self._hold_dead()
