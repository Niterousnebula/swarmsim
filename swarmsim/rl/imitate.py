"""Imitation learning: teach a network to fly like the A* planner (DAgger).

    python -m swarmsim.rl.imitate --config configs/bench_defence.yaml --name bc_planner

The planner (``planner.py``) solves the task but is an oracle: it reads the map
and the defence positions from the simulator. This script distils its behaviour
into the same kind of network PPO trains, which only sees the drone's own
observation vector. The result is an ordinary model (``runs/<name>/model.zip``)
that can be flown, benchmarked, and used as the starting point for PPO
(``python main.py train --init <name> ...``).

Method (DAgger, Ross et al. 2011): round 0 records the planner flying. In later
rounds the *network* flies, and the planner only labels what it would have done
in each situation the network got itself into. Training on the growing data set
teaches the network to recover from its own mistakes, which plain copying of
expert flights does not.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml

from ..config import SimConfig, load_config
from ..utils.cfg import to_dict
from .plugin import load_policy


def imitate(cfg: SimConfig, run_dir: str | Path, rounds: int = 8, samples_per_round: int = 30000,
            epochs: int = 6, batch_size: int = 2048, lr: float = 1e-3, expert: str = "planner",
            seed: int = 0, verbose: bool = True) -> Path:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    from .train import _plain
    from .vec import SwarmVecEnv

    t = cfg.rl.train
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_plain(to_dict(cfg)), sort_keys=False))
    raw = SwarmVecEnv(cfg, n_worlds=t.n_worlds, seed=seed)
    model = PPO("MlpPolicy", VecMonitor(raw), n_steps=t.n_steps, batch_size=t.batch_size, n_epochs=t.n_epochs,
                learning_rate=t.learning_rate, gamma=t.gamma, gae_lambda=t.gae_lambda, ent_coef=t.ent_coef,
                clip_range=t.clip_range, seed=seed, device="cpu", policy_kwargs={"net_arch": list(t.net_arch)})
    policy = model.policy
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    cores = raw.cores
    experts = []
    for core in cores:
        e = load_policy(expert, core.info())
        e.bind(core)
        experts.append(e)

    data_obs, data_act = [], []
    log = open(run_dir / "imitation.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["round", "expert_share", "samples_total", "loss", "agreement", "goal", "hit", "crash", "timeout", "seconds"])
    t0, ep_seed = time.time(), seed * 100000

    for rnd in range(rounds):
        beta = 1.0 if rnd == 0 else 0.0            # who flies: the planner in round 0, the network afterwards
        got, counts = 0, {"goal": 0, "hit": 0, "crash": 0, "timeout": 0, "out_of_bounds": 0}
        policy.set_training_mode(False)
        while got < samples_per_round:
            for core, exp in zip(cores, experts):
                ep_seed += 1
                obs = core.reset(ep_seed); exp.reset()
                while core.active.any():
                    want = np.asarray(exp.act(obs))                       # the planner's choice in this situation
                    live = core.active & core.world.alive
                    data_obs.append(obs[live].copy()); data_act.append(want[live].copy())
                    got += int(live.sum())
                    if beta >= 1.0:
                        act = want
                    else:
                        with torch.no_grad():
                            act, _ = model.predict(obs, deterministic=False)
                    obs, _, _, _, outcome = core.step(act)
                    for o in outcome:
                        if o:
                            counts[o] += 1
        X = torch.as_tensor(np.concatenate(data_obs), dtype=torch.float32)
        Y = torch.as_tensor(np.concatenate(data_act))
        policy.set_training_mode(True)
        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for k in range(0, len(X), batch_size):
                idx = perm[k:k + batch_size]
                _, logp, _ = policy.evaluate_actions(X[idx], Y[idx])
                loss = -logp.mean()
                optim.zero_grad(); loss.backward(); optim.step()
        policy.set_training_mode(False)
        with torch.no_grad():
            sample = torch.randperm(len(X))[:20000]
            pred, _ = model.predict(X[sample].numpy(), deterministic=True)
            agree = float((torch.as_tensor(pred) == Y[sample]).float().mean()) if Y.dim() == 1 else float("nan")
        n = max(sum(counts.values()), 1)
        row = [rnd, beta, len(X), round(float(loss.detach()), 4), round(agree, 3)] + \
              [round(counts[k] / n, 3) for k in ("goal", "hit", "crash", "timeout")] + [round(time.time() - t0, 1)]
        writer.writerow(row); log.flush()
        if verbose:
            who = "planner" if beta >= 1.0 else "network"
            print(f"round {rnd}: {who} flew, goal {row[5]:.0%} hit {row[6]:.0%} crash {row[7]:.0%} | "
                  f"{len(X)} samples, agreement with planner {agree:.0%}", flush=True)
        model.save(run_dir / "model.zip")
    log.close()
    raw.close()
    return run_dir / "model.zip"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/bench_defence.yaml")
    ap.add_argument("--name", default="bc_planner", help="saved as runs/<name>/model.zip")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=30000, help="drone-steps collected per round")
    ap.add_argument("--expert", default="planner")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, set_flags=args.set)
    out = imitate(cfg, Path("runs") / args.name, rounds=args.rounds, samples_per_round=args.samples,
                  expert=args.expert, seed=args.seed)
    print(f"saved {out}\nBenchmark it with: python scripts/benchmark.py --policies planner sb3:{out.as_posix()}")


if __name__ == "__main__":
    main()
