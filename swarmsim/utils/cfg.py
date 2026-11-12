"""Small config helpers shared by every module.

All hyperparameters live in plain dataclasses. A YAML file (or ``--set a.b=1``
flags on the command line) only needs to list the values it wants to change;
everything else keeps its default. Unknown keys raise an error so that a typo
in a hyperparameter name never passes silently.
"""
from __future__ import annotations

import dataclasses
import typing
from typing import Any

import yaml


def _value_type_of_dict(owner: Any, field_name: str):
    hints = typing.get_type_hints(type(owner))
    args = typing.get_args(hints.get(field_name, None))
    return args[1] if len(args) == 2 else None


def _coerce(current: Any, new: Any) -> Any:
    if isinstance(current, bool):
        return bool(new)
    if isinstance(current, float) and isinstance(new, int):
        return float(new)
    if isinstance(current, tuple) and isinstance(new, list):
        return tuple(new)
    return new


def apply_overrides(obj: Any, data: dict | None, path: str = "") -> Any:
    """Recursively merge ``data`` into the dataclass instance ``obj``."""
    if not data:
        return obj
    valid = {f.name for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(
                f"Unknown config key '{path}{key}'. Valid keys here: {sorted(valid)}"
            )
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            apply_overrides(current, value, f"{path}{key}.")
        elif isinstance(current, dict) and isinstance(value, dict):
            vtype = _value_type_of_dict(obj, key)
            for sub_key, sub_val in value.items():
                existing = current.get(sub_key)
                if dataclasses.is_dataclass(existing) and isinstance(sub_val, dict):
                    apply_overrides(existing, sub_val, f"{path}{key}.{sub_key}.")
                elif vtype is not None and dataclasses.is_dataclass(vtype) and isinstance(sub_val, dict):
                    current[sub_key] = apply_overrides(vtype(), sub_val, f"{path}{key}.{sub_key}.")
                else:
                    current[sub_key] = sub_val
        else:
            setattr(obj, key, _coerce(current, value))
    return obj


def parse_set_flags(pairs: list[str] | None) -> dict:
    """Turn ``["terrain.seed=3", "physics.wind.speed_mps=8"]`` into a nested dict."""
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got '{pair}'")
        key, raw = pair.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def to_dict(obj: Any) -> dict:
    return dataclasses.asdict(obj)


def load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(obj: Any, path: str) -> None:
    data = to_dict(obj) if dataclasses.is_dataclass(obj) else obj
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
