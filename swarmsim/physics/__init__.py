"""Module 3 - physics: PyBullet rigid bodies + flight dynamics + wind + projectiles."""
from .config import PhysicsConfig, WindConfig
from .wind import WindField
from .world import PhysicsWorld, World

__all__ = ["PhysicsConfig", "WindConfig", "WindField", "World", "PhysicsWorld"]
