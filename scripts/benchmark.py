#!/usr/bin/env python3
"""Compare policies on one fixed set of episodes.

    python scripts/benchmark.py
    python scripts/benchmark.py --policies planner scripted sb3:runs/stage1_navigate/model.zip --episodes 50
    python scripts/benchmark.py --worlds worlds/delhi_cp worlds/nsut --set rl.max_episode_s=240

Every policy flies the same episodes (same seeds, so the same starts, goals and
platform positions). The table reports the share of drones per outcome with a
95% Wilson confidence interval on the goal and hit rates; the full numbers are
saved as JSON under results/.

Policies: planner, planner_blind, scripted, random, or sb3:<path to model.zip>.
A trained model needs the observation layout it was trained with, which the
default benchmark config shares with the bundled stage1/stage2 models.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmsim.config import load_config        # noqa: E402
from swarmsim.rl.run import run_policy         # noqa: E402

OUTCOMES = ("goal", "hit", "crash", "timeout", "out_of_bounds")


def wilson(k: int, n: int, z: float = 1.96):
    """95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/bench_defence.yaml")
    ap.add_argument("--policies", nargs="+", default=["planner", "planner_blind", "scripted",
                                                      "sb3:runs/stage1_navigate/model.zip"])
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1000, help="first episode seed (keep fixed to compare runs)")
    ap.add_argument("--worlds", nargs="*", default=[None], help="city folders to test on (default: the config's own map)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=None, help="JSON file for the results (default: results/benchmark_<time>.json)")
    args = ap.parse_args()

    rows = []
    for world in args.worlds:
        flags = list(args.set) + ([f"rl.city_dir={world}"] if world else [])
        cfg = load_config(args.config, set_flags=flags)
        label = Path(world).name if world else "config map"
        for spec in args.policies:
            t0 = time.time()
            r = run_policy(cfg, spec, episodes=args.episodes, seed=args.seed, verbose=False)
            n = sum(r["counts"].values())
            row = {"world": label, "policy": spec, "drones": n, "episodes": args.episodes, "seed": args.seed,
                   "counts": r["counts"], "mean_return": round(r["mean_return"], 2),
                   "goal_ci": wilson(r["counts"]["goal"], n), "hit_ci": wilson(r["counts"]["hit"], n),
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"  done: {label} / {spec}  ({row['seconds']} s)", flush=True)

    name = lambda s: Path(s.split(":", 1)[1]).parent.name if s.startswith("sb3:") else s   # noqa: E731
    pct = lambda k, n: f"{100 * k / max(n, 1):5.1f}%"                                      # noqa: E731
    ci = lambda c: f"[{100 * c[0]:.0f}-{100 * c[1]:.0f}]"                                  # noqa: E731
    print(f"\n{'map':<16}{'policy':<18}{'drones':>7}  {'goal':>6} {'95% CI':<9} {'hit':>6} {'95% CI':<9} "
          f"{'crash':>6} {'timeout':>8} {'other':>6}")
    for r in rows:
        c, n = r["counts"], r["drones"]
        print(f"{r['world']:<16}{name(r['policy']):<18}{n:>7}  {pct(c['goal'], n)} {ci(r['goal_ci']):<9} "
              f"{pct(c['hit'], n)} {ci(r['hit_ci']):<9} {pct(c['crash'], n)} {pct(c['timeout'], n):>8} "
              f"{pct(c['out_of_bounds'], n)}")
    out = Path(args.out) if args.out else Path("results") / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": args.config, "set": args.set, "rows": rows}, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
