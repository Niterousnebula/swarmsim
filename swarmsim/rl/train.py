"""Train a shared policy for the swarm with PPO (Stable-Baselines3).

    python -m swarmsim.rl.train --config configs/default.yaml
    python -m swarmsim.rl.train --config configs/default.yaml --timesteps 300000 --run-dir runs/test

Curriculum (recommended): learn to navigate first, then add the defence platform.

    python -m swarmsim.rl.train --config configs/quick_demo.yaml --set rl.n_defence=0 --run-dir runs/stage1
    python -m swarmsim.rl.train --config configs/quick_demo.yaml --init runs/stage1/model.zip --run-dir runs/stage2
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from ..config import SimConfig, load_config, to_dict


def train(cfg: SimConfig, verbose: int = 1, init_model: str | None = None) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecMonitor

    from .vec import SwarmVecEnv

    t = cfg.rl.train
    if cfg.rl.fw_mode == "search" and cfg.rl.n_fixedwing:
        raise ValueError("Fixed-wings in rl.fw_mode=search are flown by their own autopilot, not the policy. "
                         "Train with rl.n_fixedwing=0, or set rl.fw_mode=policy to train them.")
    run_dir = Path(t.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_plain(to_dict(cfg)), sort_keys=False))
    stop_flag = run_dir / "stop"         # create this file to end training early; the model is still saved
    stop_flag.unlink(missing_ok=True)

    raw = SwarmVecEnv(cfg, n_worlds=t.n_worlds, seed=t.seed)
    env = VecMonitor(raw)

    class Progress(BaseCallback):
        """Writes one CSV row per rollout: reward and how episodes ended."""

        def _on_training_start(self):
            self.t0 = time.time()
            self.file = open(run_dir / "progress.csv", "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(["timesteps", "seconds", "mean_return", "mean_length", "goal",
                                  "crash", "hit", "out_of_bounds", "timeout"])
            self.next_ckpt = t.checkpoint_every

        def _on_step(self):
            return not stop_flag.exists()

        def _on_rollout_end(self):
            eps = list(self.model.ep_info_buffer)
            counts = Counter(raw.outcomes); n = max(sum(counts.values()), 1)
            row = [self.num_timesteps, round(time.time() - self.t0, 1),
                   round(float(np.mean([e["r"] for e in eps])), 3) if eps else "",
                   round(float(np.mean([e["l"] for e in eps])), 1) if eps else ""]
            row += [round(counts[k] / n, 3) for k in ("goal", "crash", "hit", "out_of_bounds", "timeout")]
            self.writer.writerow(row); self.file.flush()
            for k in ("goal", "crash", "hit", "out_of_bounds", "timeout"):
                self.logger.record(f"outcome/{k}", counts[k] / n)
            if t.checkpoint_every and self.num_timesteps >= self.next_ckpt:
                self.model.save(run_dir / f"checkpoint_{self.num_timesteps}")
                self.next_ckpt += t.checkpoint_every

        def _on_training_end(self):
            self.file.close()

    if t.algo.lower() != "ppo":
        raise ValueError("Only 'ppo' is wired up in this prototype (see README for adding others).")
    if init_model:                       # continue from an earlier model (same observation/action layout)
        model = PPO.load(init_model, env=env, device=t.device, learning_rate=t.learning_rate,
                         ent_coef=t.ent_coef, n_steps=t.n_steps, batch_size=t.batch_size, n_epochs=t.n_epochs)
        model.verbose = verbose
    else:
        model = PPO("MlpPolicy", env, n_steps=t.n_steps, batch_size=t.batch_size, n_epochs=t.n_epochs,
                    learning_rate=t.learning_rate, gamma=t.gamma, gae_lambda=t.gae_lambda,
                    ent_coef=t.ent_coef, clip_range=t.clip_range, seed=t.seed, device=t.device,
                    policy_kwargs={"net_arch": list(t.net_arch)}, verbose=verbose)
    model.learn(total_timesteps=t.total_timesteps, callback=Progress())
    out = run_dir / "model.zip"
    model.save(out)
    stop_flag.unlink(missing_ok=True)
    env.close()
    return out


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--n-worlds", type=int, default=None)
    ap.add_argument("--init", default=None, help="start from this saved model instead of from scratch")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config key, e.g. --set rl.n_defence=0 (repeatable)")
    args = ap.parse_args()
    cfg = load_config(args.config, set_flags=args.set)
    over = {k: v for k, v in (("total_timesteps", args.timesteps), ("run_dir", args.run_dir),
                              ("n_worlds", args.n_worlds)) if v is not None}
    cfg.rl.train = replace(cfg.rl.train, **over)
    print("saved", train(cfg, init_model=args.init))


if __name__ == "__main__":
    main()
