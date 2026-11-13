#!/usr/bin/env python3
"""Generalisation experiment: does training on more generated cities help on cities never seen?

    python scripts/experiment_generalisation.py --quick        # 5-minute smoke test of the whole pipeline
    python scripts/experiment_generalisation.py                # the real thing (many hours, resumable)
    python scripts/experiment_generalisation.py --summary      # just print the table from finished runs

For every pool size N (default 1, 4, 16, 64) and every seed it
  1. trains a network to imitate the A* planner on N generated cities   (swarmsim.rl.imitate)
  2. fine-tunes it with PPO on the same N cities                        (swarmsim.rl.train)
  3. tests it on  a) cities it trained on,
                  b) generated cities it has never seen,
                  c) real cities from OpenStreetMap (worlds/delhi_cp, worlds/nsut ...).
The planner and scripted baselines are tested once on the same episodes.

Safe to interrupt (Ctrl+C) and start again: finished stages are skipped. Everything is written
under --out (default runs/gen) and results/gen. To split the work between two computers, give each
a different --seeds list and merge the results/gen folders afterwards.
"""
import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmsim.config import load_config            # noqa: E402
from swarmsim.rl.core import ensure_city           # noqa: E402
from swarmsim.rl.run import run_policy             # noqa: E402

OUTCOMES = ("goal", "hit", "crash", "timeout", "out_of_bounds")
TEST_OFFSET = 100000                               # generated test cities: indices no training pool ever uses
MAX_POOL = 64


def pool_offset(n: int, seed: int) -> int:
    """Which cities a run trains on. Pools of different seeds are disjoint (when they fit), and
    every pool lies inside the first 64 city indices so at most 64 training cities are ever built."""
    return (seed * n) % max(MAX_POOL - n + 1, 1) if n < MAX_POOL else 0


def evaluate(cfg, spec, city_dirs, episodes, seed):
    """Fly ``episodes`` episodes in each city; returns summed outcome counts."""
    total = {k: 0 for k in OUTCOMES}
    for j, city in enumerate(city_dirs):
        c = load_config(None)                       # fresh copy, then graft the experiment settings in
        c.terrain, c.models, c.physics = cfg.terrain, cfg.models, cfg.physics
        c.rl = replace(cfg.rl, city_dir=Path(city).as_posix())
        r = run_policy(c, spec, episodes=episodes, seed=seed + 1000 * j, verbose=False)
        for k in OUTCOMES:
            total[k] += r["counts"][k]
    return total


def rate(counts, key):
    return counts[key] / max(sum(counts.values()), 1)


def summarise(res_dir: Path, sets):
    rows = []
    base = json.loads((res_dir / "baselines.json").read_text()) if (res_dir / "baselines.json").exists() else {}
    runs = sorted(res_dir.glob("N*_s*.json"))
    by_n = {}
    for f in runs:
        r = json.loads(f.read_text())
        by_n.setdefault(r["n_cities"], []).append(r)
    print(f"\n{'method':<22}{'runs':>5}  " + "  ".join(f"{s + ' goal':>14} {s + ' hit':>13}" for s in sets))
    def line(name, n_runs, cell):
        print(f"{name:<22}{n_runs:>5}  " + "  ".join(cell(s) for s in sets))
    for name, per_set in base.items():
        line(name, 1, lambda s: f"{100 * rate(per_set[s], 'goal'):>13.1f}% {100 * rate(per_set[s], 'hit'):>12.1f}%"
             if s in per_set else f"{'-':>14} {'-':>13}")
        rows.append({"method": name, "runs": 1, **{f"{s}_{k}": round(rate(per_set[s], k), 4)
                                                    for s in sets if s in per_set for k in ("goal", "hit", "crash", "timeout")}})
    for n in sorted(by_n):
        rs = by_n[n]
        def stat(s, key):
            v = [rate(r["eval"][s], key) for r in rs if s in r["eval"]]
            if not v:
                return None
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
            return m, sd
        def cell(s):
            g, h = stat(s, "goal"), stat(s, "hit")
            return f"{'-':>14} {'-':>13}" if g is None else f"{100 * g[0]:>8.1f}±{100 * g[1]:<4.1f}% {100 * h[0]:>7.1f}±{100 * h[1]:<4.1f}%"
        line(f"learned, N = {n} cities", len(rs), cell)
        row = {"method": f"learned_N{n}", "runs": len(rs)}
        for s in sets:
            for key in ("goal", "hit", "crash", "timeout"):
                st = stat(s, key)
                if st:
                    row[f"{s}_{key}"], row[f"{s}_{key}_sd"] = round(st[0], 4), round(st[1], 4)
        rows.append(row)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("method", "runs"), k))
        with open(res_dir / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"\n(mean ± standard deviation over runs; table saved to {res_dir / 'summary.csv'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/gen_experiment.yaml")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 16, 64], help="training pool sizes N")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="independent repetitions")
    ap.add_argument("--bc-rounds", type=int, default=10)
    ap.add_argument("--bc-samples", type=int, default=30000)
    ap.add_argument("--ppo-steps", type=int, default=500000)
    ap.add_argument("--test-cities", type=int, default=10, help="unseen generated cities")
    ap.add_argument("--episodes", type=int, default=5, help="episodes per test city")
    ap.add_argument("--real", nargs="*", default=["worlds/delhi_cp", "worlds/nsut"], help="real-city folders")
    ap.add_argument("--real-episodes", type=int, default=15)
    ap.add_argument("--baselines", nargs="*", default=["planner", "planner_blind", "scripted"])
    ap.add_argument("--out", default="runs/gen")
    ap.add_argument("--results", default="results/gen")
    ap.add_argument("--quick", action="store_true", help="tiny budgets, to check that everything runs")
    ap.add_argument("--summary", action="store_true", help="only print the table of finished runs")
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.bc_rounds, args.bc_samples = [1, 3], [0], 2, 1500
        args.ppo_steps, args.test_cities, args.episodes, args.real_episodes = 4096, 2, 1, 1
        args.out, args.results = "runs/gen_quick", "results/gen_quick"

    out, res = Path(args.out), Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    real = [p for p in args.real if (Path(p) / "scene.json").exists()]
    for p in set(args.real) - set(real):
        print(f"note: real map '{p}' not found, skipping it")
    sets = ["train", "unseen"] + [Path(p).name for p in real]
    if args.summary:
        return summarise(res, sets)

    from swarmsim.rl.imitate import imitate
    from swarmsim.rl.train import train

    cfg = load_config(args.config)
    t0 = time.time()
    print(f"building {args.test_cities} test cities ...", flush=True)
    test_dirs = [ensure_city(cfg, TEST_OFFSET + j) for j in range(args.test_cities)]
    eval_sets = {"unseen": (test_dirs, args.episodes)}
    for p in real:
        eval_sets[Path(p).name] = ([p], args.real_episodes)

    # ---- baselines, once
    bfile = res / "baselines.json"
    base = json.loads(bfile.read_text()) if bfile.exists() else {}
    for spec in args.baselines:
        if spec in base:
            continue
        print(f"baseline {spec} ...", flush=True)
        base[spec] = {name: evaluate(cfg, spec, dirs, eps, seed=5000) for name, (dirs, eps) in eval_sets.items()}
        bfile.write_text(json.dumps(base, indent=1))

    # ---- learned policies
    todo = [(n, s) for s in args.seeds for n in args.sizes]
    for done_runs, (n, seed) in enumerate(todo):
        tag = f"N{n}_s{seed}"
        rfile = res / f"{tag}.json"
        if rfile.exists():
            continue
        t1 = time.time()
        run_cfg = load_config(args.config, {"rl": {"train_cities": n, "city_pool_offset": pool_offset(n, seed),
                                                   "train": {"seed": seed, "total_timesteps": args.ppo_steps,
                                                             "run_dir": (out / tag).as_posix()}}})
        bc_model = out / f"{tag}_bc" / "model.zip"
        if not (out / f"{tag}_bc" / "done").exists():
            print(f"[{tag}] imitation on {n} cit{'y' if n == 1 else 'ies'} ...", flush=True)
            imitate(run_cfg, out / f"{tag}_bc", rounds=args.bc_rounds, samples_per_round=args.bc_samples,
                    seed=seed, verbose=False)
            (out / f"{tag}_bc" / "done").write_text("ok")
        model = out / tag / "model.zip"
        if not (out / tag / "done").exists():
            print(f"[{tag}] PPO fine-tuning, {args.ppo_steps:,} steps ...", flush=True)
            train(run_cfg, verbose=0, init_model=str(bc_model))
            (out / tag / "done").write_text("ok")
        print(f"[{tag}] testing ...", flush=True)
        train_dirs = [ensure_city(run_cfg, run_cfg.rl.city_pool_offset + j) for j in range(min(n, 4))]
        sets_now = {"train": (train_dirs, max(1, args.episodes * 2 // len(train_dirs))), **eval_sets}
        result = {"n_cities": n, "seed": seed, "pool_offset": run_cfg.rl.city_pool_offset,
                  "bc_rounds": args.bc_rounds, "bc_samples": args.bc_samples, "ppo_steps": args.ppo_steps,
                  "eval": {name: evaluate(cfg, f"sb3:{model.as_posix()}", dirs, eps, seed=5000)
                           for name, (dirs, eps) in sets_now.items()},
                  "minutes": round((time.time() - t1) / 60, 1)}
        rfile.write_text(json.dumps(result, indent=1))
        left = sum(not (res / f"N{a}_s{b}.json").exists() for a, b in todo)
        print(f"[{tag}] done in {result['minutes']} min: unseen goal {100 * rate(result['eval']['unseen'], 'goal'):.0f}% "
              f"hit {100 * rate(result['eval']['unseen'], 'hit'):.0f}%  |  {left} runs left, "
              f"{(time.time() - t0) / 3600:.1f} h so far", flush=True)
    summarise(res, sets)


if __name__ == "__main__":
    main()
