"""Performance and geometry parameters of the three platforms.

Geometry numbers describe the prebuilt meshes and set the collision shapes;
mass / aero / performance numbers drive the physics module.  Everything can be overridden from YAML.
The values are generic, round numbers for a simulation - they do not describe
any real product.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuadSpec:
    # geometry (m)
    arm_length: float = 0.22            # centre to motor axis
    body_size: tuple = (0.16, 0.10, 0.045)
    prop_radius: float = 0.10
    # mass properties
    mass: float = 1.0                   # kg
    inertia: tuple = (0.012, 0.012, 0.022)   # Ixx, Iyy, Izz  kg m^2
    # propulsion
    thrust_to_weight: float = 2.2       # all four motors at full throttle / weight
    motor_time_constant: float = 0.05   # s, first-order motor lag
    torque_to_thrust: float = 0.016     # m, yaw reaction torque per N of rotor thrust
    # aerodynamics
    drag_area: float = 0.05             # Cd * A  (m^2)
    # flight envelope used by the on-board controller
    max_speed: float = 10.0             # m/s horizontal
    max_climb: float = 4.0              # m/s vertical
    max_yaw_rate: float = 1.5           # rad/s
    max_tilt_deg: float = 35.0
    collision_radius: float = 0.32


@dataclass
class FixedWingSpec:
    # geometry (m)
    span: float = 1.8
    chord: float = 0.25
    fuselage_length: float = 1.1
    # mass properties
    mass: float = 3.0
    inertia: tuple = (0.25, 0.35, 0.55)
    # aerodynamics (lift curve with a soft stall, parabolic drag polar)
    cl0: float = 0.30
    cl_alpha: float = 5.0               # per rad
    cl_max: float = 1.3
    cd0: float = 0.035
    oswald: float = 0.8
    side_force_coeff: float = 0.6       # per rad of sideslip
    # propulsion
    max_thrust: float = 45.0            # N (thrust/weight ~1.5, like a sport RC plane: enough for a steep climb)
    # flight envelope used by the autopilot
    cruise_speed: float = 18.0
    min_speed: float = 12.0
    max_speed: float = 48.0
    max_bank_deg: float = 85.0
    # climb limits
    max_climb: float = 26.0             # m/s, fastest climb (60 deg at cruise speed is ~15.6 m/s)
    max_climb_deg: float = 80.0         # steepest climb angle; eased off as airspeed nears min_speed
    max_pitch_deg: float = 70.0         # nose-up attitude limit (climb angle + angle of attack)
    # descent limits
    max_descent: float = 26.0           # m/s, fastest sink rate the autopilot will command
    max_descent_deg: float = 80.0       # fixed steepest commanded descent angle
    max_pitch_down_deg: float = 70.0    # nose-down attitude limit
    # how far above min_speed (m/s) the climb-angle limit starts to ease off
    envelope_fade_speed: float = 4.0
    collision_radius: float = 0.9

    @property
    def wing_area(self) -> float:
        return self.span * self.chord

    @property
    def aspect_ratio(self) -> float:
        return self.span / self.chord


@dataclass
class DefenceSpec:
    """Scripted ground platform.  Abstract game-style parameters."""
    # geometry (m)
    base_size: tuple = (2.4, 2.4, 0.7)
    turret_radius: float = 0.75
    barrel_length: float = 1.8
    muzzle_height: float = 2.0
    # sensing
    detection_range: float = 160.0      # m, needs line of sight
    reaction_time: float = 0.6          # s between first detection and first shot
    # engagement
    range_m: float = 120.0              # max engagement distance
    min_range: float = 8.0
    fire_rate_hz: float = 3.0           # projectiles per second
    projectile_speed: float = 110.0     # m/s at the muzzle
    projectile_drag: float = 0.0015     # 1/m, deceleration = k * |v| * v
    projectile_lifetime: float = 3.0    # s
    hit_radius: float = 0.7             # m, proximity radius counted as a hit
    dispersion_deg: float = 1.2         # random aiming error (1 sigma)
    ammo: int = 400
    # turret kinematics
    slew_rate_deg: float = 70.0         # deg/s in azimuth and elevation
    elevation_min_deg: float = 3.0
    elevation_max_deg: float = 85.0
    aim_tolerance_deg: float = 2.5      # fires only when aligned within this
    lead_target: bool = True            # aim ahead of moving targets


@dataclass
class ModelsConfig:
    assets_dir: str = "assets/models"
    quad: QuadSpec = field(default_factory=QuadSpec)
    fixedwing: FixedWingSpec = field(default_factory=FixedWingSpec)
    defence: DefenceSpec = field(default_factory=DefenceSpec)
