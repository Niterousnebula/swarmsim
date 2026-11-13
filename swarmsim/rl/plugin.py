"""Policy plugins: anything that maps observations to actions.

A plugin receives the observation matrix of all drones (n, obs_dim) and returns
one action per drone.  Trained models, scripted baselines and your own code
all use the same three-method interface, so the runner (``run.py``) and the
simulator never need to know where the decisions come from.

    spec strings understood by ``load_policy``:
        "sb3:runs/default/model.zip"       Stable-Baselines3 model
        "torchscript:policy.pt"            TorchScript module  obs -> action / logits
        "onnx:policy.onnx"                 ONNX graph (onnxruntime), same contract
        "scripted"                         rule-based baseline (no training)
        "planner"                          A* route planner that avoids defence platforms (oracle baseline)
        "planner_blind"                    the same planner ignoring the defence
        "random"                           uniformly random actions
        "mypkg.mymodule:MyPolicy"          any importable PolicyPlugin subclass
"""
from __future__ import annotations

import importlib

import numpy as np


class PolicyPlugin:
    name = "base"

    def __init__(self, info: dict, **kwargs):
        """``info`` describes the environment (see ``SwarmCore.info``)."""
        self.info = info

    def reset(self) -> None:
        """Called at the start of every episode."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (n_drones, obs_dim) -> actions: (n_drones,) ints or (n_drones, 4) floats."""
        raise NotImplementedError


class RandomPolicy(PolicyPlugin):
    name = "random"

    def __init__(self, info, seed: int = 0, **kw):
        super().__init__(info)
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        if self.info["action_mode"] == "discrete":
            return self.rng.integers(len(self.info["action_names"]), size=len(obs))
        return self.rng.uniform(-1, 1, (len(obs), 4)).astype(np.float32)


class ScriptedPolicy(PolicyPlugin):
    """Hand-written baseline: turn towards the goal, fly forward, climb over
    whatever blocks the way, descend once above the goal.  Useful as a sanity
    check and as a yardstick for trained policies.  It reads only the
    observation vector, exactly like a learned policy."""
    name = "scripted"

    def act(self, obs):
        from .actions import PRIMITIVES
        L, info = self.info["obs_layout"], self.info
        nr, names = info["n_rays"], list(info["action_names"])
        discrete = info["action_mode"] == "discrete"
        out = []
        for o in obs:
            goal, rays = o[L["goal"]], o[L["rays"]] * info["ray_range"]
            agl = o[L["altitude"]][0] * 100.0
            dist = goal[3] * info["map_size"]
            horiz, dz = dist * float(np.hypot(goal[0], goal[1])), dist * goal[2]
            bearing = float(np.arctan2(goal[1], goal[0])) if horiz > 3.0 else 0.0
            front = min(rays[0], rays[1], rays[nr - 1])
            k = int(np.argmin(rays[:nr]))
            turn = "yaw_left" if bearing > 0 else "yaw_right"
            if rays[nr] < 5.0:                                   # too low over something
                want = "up"
            elif rays[k] < 3.5:                                  # something very close: back away from it
                away = 2 * np.pi * k / nr + np.pi
                want = ("forward", "left", "back", "right")[int(np.round(away / (np.pi / 2))) % 4]
            elif abs(bearing) > 0.45:
                want = turn
            elif front < 14.0 and horiz > 6.0:                   # blocked: go over (or around at the ceiling)
                want = "up" if agl < info["ceiling_agl"] - 15.0 else "left"
            elif horiz < 6.0 and abs(dz) > 1.5:                  # above/below the goal
                want = "up" if dz > 0 else "down"
            elif dz > 4.0:
                want = "up"
            elif abs(bearing) > 0.12:
                want = turn
            else:
                want = "forward"
            if discrete:
                out.append(next(names.index(c) for c in (want, "forward", "hover", names[0]) if c in names))
            else:
                out.append(PRIMITIVES[want])
        return np.array(out) if discrete else np.array(out, np.float32)


class SB3Policy(PolicyPlugin):
    name = "sb3"

    def __init__(self, info, path: str, deterministic: bool = True, device: str = "cpu", **kw):
        super().__init__(info)
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device=device)
        self.deterministic = deterministic
        want = self.model.observation_space.shape[0]
        if want != info["obs_dim"]:
            raise ValueError(f"Model expects {want} observations but the environment provides "
                             f"{info['obs_dim']}. Run it with the config it was trained with.")

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


class TorchScriptPolicy(PolicyPlugin):
    name = "torchscript"

    def __init__(self, info, path: str, **kw):
        super().__init__(info)
        import torch
        self.torch = torch
        self.module = torch.jit.load(path, map_location="cpu").eval()

    def act(self, obs):
        with self.torch.no_grad():
            out = self.module(self.torch.as_tensor(obs, dtype=self.torch.float32)).numpy()
        if self.info["action_mode"] == "discrete":
            return out.argmax(1) if out.ndim == 2 else out.astype(int)
        return np.clip(out, -1, 1)


class OnnxPolicy(PolicyPlugin):
    """Runs an exported network with onnxruntime (no PyTorch needed at run time)."""
    name = "onnx"

    def __init__(self, info, path: str, **kw):
        super().__init__(info)
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def act(self, obs):
        out = self.session.run(None, {self.input: np.asarray(obs, np.float32)})[0]
        if self.info["action_mode"] == "discrete":
            return out.argmax(1) if out.ndim == 2 else out.astype(int)
        return np.clip(out, -1, 1)


PLUGINS = {c.name: c for c in (RandomPolicy, ScriptedPolicy, SB3Policy, TorchScriptPolicy, OnnxPolicy)}


def register(cls):
    """Decorator: make your own PolicyPlugin available by name."""
    PLUGINS[cls.name] = cls
    return cls


def load_policy(spec: str, info: dict, **kwargs) -> PolicyPlugin:
    name, _, arg = spec.partition(":")
    if name.startswith("planner"):
        from . import planner  # noqa: F401  (registers the A* baselines)
    if name in PLUGINS:
        return PLUGINS[name](info, path=arg, **kwargs) if arg else PLUGINS[name](info, **kwargs)
    if arg:                                   # "module.path:ClassName"
        cls = getattr(importlib.import_module(name), arg)
        return cls(info, **kwargs)
    raise ValueError(f"Unknown policy '{spec}'. Built-in: {sorted(PLUGINS)}")


def _actor_network(sb3_path: str):
    """The actor of a trained SB3 PPO model as a plain module: obs -> logits / mean action."""
    import torch
    from stable_baselines3 import PPO

    model = PPO.load(sb3_path, device="cpu")
    policy = model.policy

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.extractor = policy.mlp_extractor.policy_net
            self.head = policy.action_net

        def forward(self, obs):
            return self.head(self.extractor(obs))

    return Net().eval(), torch.zeros(1, model.observation_space.shape[0])


def export_torchscript(sb3_path: str, out_path: str) -> str:
    """Standalone TorchScript file that runs without Stable-Baselines3."""
    import torch
    net, example = _actor_network(sb3_path)
    torch.jit.trace(net, example).save(out_path)
    return out_path


def export_onnx(sb3_path: str, out_path: str) -> str:
    """Standalone ONNX file (needs the ``onnx`` and ``onnxscript`` packages to export)."""
    import torch
    net, example = _actor_network(sb3_path)
    torch.onnx.export(net, (example,), out_path, input_names=["obs"], output_names=["out"],
                      dynamic_axes={"obs": {0: "n"}, "out": {0: "n"}})
    return out_path
