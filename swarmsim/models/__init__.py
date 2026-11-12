"""Module 2 - vehicle models (prebuilt meshes, URDFs) and platform specifications."""
from .assets import ensure_models, write_urdfs
from .specs import DefenceSpec, FixedWingSpec, ModelsConfig, QuadSpec

__all__ = ["ensure_models", "write_urdfs", "ModelsConfig", "QuadSpec", "FixedWingSpec", "DefenceSpec"]
