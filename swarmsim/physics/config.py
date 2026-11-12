"""Hyperparameters of the physics module."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WindConfig:
    mean_speed: float = 3.0            # m/s at the reference height
    direction_deg: float = 30.0        # direction the wind blows TOWARDS (0 = +x/east, 90 = +y/north)
    ref_height: float = 10.0           # m above ground
    shear_exponent: float = 0.25       # power-law wind profile (urban terrain ~0.25-0.4)
    gust_intensity: float = 1.0        # m/s, standard deviation of turbulence
    gust_time_scale: float = 4.0       # s, how quickly gusts change
    gust_length_scale: float = 60.0    # m, size of gust cells
    vertical_gust_ratio: float = 0.5   # vertical turbulence relative to horizontal
    building_shelter: float = 0.6      # wind reduction in the lee of obstacles (0 = none, 1 = calm)


@dataclass
class PhysicsConfig:
    gravity: float = 9.81              # m/s^2
    air_density: float = 1.225         # kg/m^3
    physics_hz: int = 120              # integration rate
    wind: WindConfig = field(default_factory=WindConfig)
    drone_collisions: bool = True      # drones can collide with each other
    seed: int = 0
