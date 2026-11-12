"""Module 4 - reinforcement learning: environment, trainer, policy plugins."""
from .config import RLConfig, TrainConfig

_LAZY = {"SwarmCore": "core", "SwarmEnv": "env", "PolicyPlugin": "plugin", "PLUGINS": "plugin",
         "load_policy": "plugin", "register": "plugin"}
__all__ = ["RLConfig", "TrainConfig", *_LAZY]


def __getattr__(name):          # heavy imports (PyBullet, PettingZoo) only when first used
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
