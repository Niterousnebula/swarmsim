"""One YAML file describes a whole experiment.

Every module owns a dataclass of hyperparameters; ``SimConfig`` groups them:

    terrain  -> swarmsim.terrain.params.TerrainParams
    models   -> swarmsim.models.specs.ModelsConfig
    physics  -> swarmsim.physics.config.PhysicsConfig
    rl       -> swarmsim.rl.config.RLConfig

A YAML file only lists what it changes. Unknown keys raise an error, so a typo
never silently falls back to a default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models.specs import ModelsConfig
from .physics.config import PhysicsConfig
from .rl.config import RLConfig
from .terrain.params import TerrainParams
from .utils.cfg import apply_overrides, deep_merge, load_yaml, parse_set_flags, to_dict  # noqa: F401


@dataclass
class SimConfig:
    terrain: TerrainParams = field(default_factory=TerrainParams)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    rl: RLConfig = field(default_factory=RLConfig)


def load_config(path: str | Path | None = None, overrides: dict | None = None,
                set_flags: list[str] | None = None) -> SimConfig:
    """Defaults <- YAML file <- ``overrides`` dict <- ``--set a.b=c`` flags."""
    raw = load_yaml(str(path) if path else None)
    raw = deep_merge(raw, overrides or {})
    raw = deep_merge(raw, parse_set_flags(set_flags))
    return apply_overrides(SimConfig(), raw)
