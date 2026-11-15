"""Record flights for the replay website (replay/).

The replay site is a static page: no Python, no physics, nothing to keep running.
This script flies every chosen model in every chosen city, with the same starts,
goals and defence positions for every model, and writes what the page needs:

    replay/data/index.json                     models, cities, episodes and their outcomes
    replay/data/vehicles/{quad,wing,defence}.bin   vehicle meshes (float32 x, y, z, u, v)
    replay/data/cities/<city>/<kind>.bin       city point clouds, packed (see pack_cloud)
    replay/data/flights/<model>/<city>/<ep>.json   one episode, ten frames a second

    python scripts/record_replays.py                       # the default set (about 20-40 minutes)
    python scripts/record_replays.py --episodes 1 --quick  # a fast check

The settings are the paper's evaluation settings (configs/gen_experiment.yaml,
collision reflex on), so the flights shown are the flights that were measured.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swarmsim.app.cloud import city_cloud, DENSITY      # noqa: E402
from swarmsim.app.meshes import vertex_buffer           # noqa: E402
from swarmsim.config import load_config                 # noqa: E402
from swarmsim.rl.core import SwarmCore, ensure_city     # noqa: E402
from swarmsim.rl.plugin import load_policy              # noqa: E402

OUTCOMES = ("goal", "hit", "crash", "timeout", "out_of_bounds")
TEST_OFFSET = 100000          # the paper's unseen test cities (never used for training)

# id: (label, short label, description, policy spec)
MODELS = {
    "best": ("Best model (DAgger + PPO)", "Best model",
             "Trained on one city by imitating the A* planner (DAgger), then fine-tuned with PPO. "
             "The best trained model in the paper: 83.6% of drones reached the goal over all test maps.",
             "sb3:runs/gen/N1_s2/model.zip"),
    "n64": ("DAgger + PPO, 64 cities", "64-city model",
            "The same training on 64 generated cities. In the paper: 81.8% of drones reached the goal and 8.0% were hit on unseen cities.",
            "sb3:runs/gen/N64_s1/model.zip"),
    "planner": ("A* planner (threat-aware)", "A* threat-aware",
                "Full-map A* search that pays a high cost to pass through the defence range. The teacher used for imitation.",
                "planner"),
    "planner_blind": ("A* planner (threat-blind)", "A* threat-blind",
                      "The same A* search with no threat cost: shortest safe path through buildings only.",
                      "planner_blind"),
    "scripted": ("Scripted (fly straight)", "Scripted",
                 "Climbs and flies straight at the goal. A baseline with no planning.",
                 "scripted"),
}

# id: (label, description, folder or test-city index)
CITIES = {
    "delhi_cp": ("Connaught Place, New Delhi", "Real map from OpenStreetMap.", "worlds/delhi_cp"),
    "nsut": ("NSUT campus, Delhi", "Real map from OpenStreetMap.", "worlds/nsut"),
    "test1": ("Test city 1", "Generated city never seen in training.", 0),
    "test2": ("Test city 2", "Generated city never seen in training.", 1),
    "test3": ("Test city 3", "Generated city never seen in training.", 2),
}

CLOUD_BUDGET = 700_000        # points per city on the website (the app uses up to 1.8 million)


# ------------------------------------------------------------------ static data
def pack_cloud(pts: np.ndarray, out: Path) -> dict:
    """16-byte points (float32 x, y, z + RGBA) -> 9 bytes: int16 x, y, z in centimetres
    from an origin, then RGB. The page unpacks them back into the renderer's format."""
    p = pts["p"].astype(np.float64)
    lo = p.min(0)
    span = p.max(0) - lo
    step = max(0.02, float(span.max()) / 65000.0)          # 2 cm, or coarser on huge maps
    q = np.round((p - lo) / step).astype(np.int64) - 32768
    rec = np.empty(len(pts), np.dtype([("q", "<i2", 3), ("c", "u1", 3)]))
    rec["q"] = np.clip(q, -32768, 32767)
    rec["c"] = pts["c"][:, :3]
    out.write_bytes(rec.tobytes())
    return {"origin": [round(float(v), 4) for v in lo], "step": step}


def export_city(city_dir: Path, out: Path) -> list:
    from swarmsim.app import cloud
    old = cloud.BUDGET
    cloud.BUDGET = CLOUD_BUDGET
    cloud._build.cache_clear()
    try:
        clouds = city_cloud(city_dir)
    finally:
        cloud.BUDGET = old
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for kind, v in clouds.items():
        pts = np.frombuffer(v["data"], cloud.POINT)
        info = pack_cloud(pts, out / f"{kind}.bin")
        meta.append({"kind": kind, "count": v["count"], "spacing": v["spacing"], **info})
    return meta


def export_vehicles(cfg, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    models = Path(cfg.models.assets_dir)
    for key, name in (("quad", "quadcopter"), ("wing", "fixedwing"), ("defence", "defence")):
        (out / f"{key}.bin").write_bytes(vertex_buffer(models / f"{name}.stl", 0.001))


# --------------------------------------------------------------------- flights
def quat(R: np.ndarray) -> list:
    """Rotation matrix -> unit quaternion (x, y, z, w)."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = [(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s, 0.25 / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return [round(float(v), 4) for v in q]


def frame(core, t0: float = 0.0) -> dict:
    w = core.world
    on = w.alive & core.active
    agl = w.pos[:, 2] - core.city.ground_z(w.pos[:, 0], w.pos[:, 1])
    drones = []
    for i in range(core.n):
        if not on[i]:
            drones.append(None)
            continue
        dist = float(np.linalg.norm(core._target(i) - w.pos[i]))
        drones.append([*(round(float(v), 2) for v in w.pos[i]), *quat(w.R[i]),
                       round(float(np.linalg.norm(w.vel[i])), 1), round(float(agl[i]), 1), round(dist, 1)])
    act = np.flatnonzero(w.projectiles.active)
    return {
        "t": round(float(w.time - t0), 2),
        "d": drones,
        "f": [[round(float(v), 3) for v in (d.az, d.el)] + [int(d.target), int(d.ammo)] for d in w.defences],
        "p": np.round(np.concatenate([w.projectiles.pos[act], w.projectiles.vel[act]], 1), 2).tolist(),
    }


def fly(core, policy, seed: int) -> dict:
    obs = core.reset(seed)
    policy.reset()
    w = core.world
    t0 = float(w.time)                       # the world clock keeps running across episodes
    frames, events = [frame(core, t0)], []
    counts = dict.fromkeys(OUTCOMES, 0)
    while core.active.any():
        before = w.pos.copy()
        obs, _, _, _, outcome = core.step(policy.act(obs))
        for i, o in enumerate(outcome):
            if o:
                counts[o] += 1
                events.append({"t": round(float(w.time) - t0, 2), "drone": i, "outcome": o,
                               "pos": [round(float(v), 2) for v in before[i]]})
        frames.append(frame(core, t0))
    return {
        "seed": seed,
        "duration": frames[-1]["t"],
        "starts": [d[:3] if d else None for d in frames[0]["d"]],
        "goals": np.round(core.goal, 2).tolist(),
        "defences": [[round(float(v), 2) for v in d.base] for d in w.defences],
        "shots": int(sum(d.shots for d in w.defences)),
        "counts": counts,
        "events": events,
        "frames": frames,
    }


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/gen_experiment.yaml")
    ap.add_argument("--out", default="replay/data")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--cities", nargs="+", default=list(CITIES))
    ap.add_argument("--episodes", type=int, default=3, help="episodes per model and city")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--quick", action="store_true", help="cap episodes at 20 s (a check, not real data)")
    args = ap.parse_args()

    out = ROOT / args.out
    cfg = load_config(ROOT / args.config)
    rl = replace(cfg.rl, collision_reflex=True, **({"max_episode_s": 20} if args.quick else {}))
    export_vehicles(cfg, out / "vehicles")

    index = {"made": time.strftime("%Y-%m-%d"), "policy_hz": cfg.rl.policy_hz,
             "settings": {"drones": rl.n_quads, "defences": rl.n_defence, "max_episode_s": rl.max_episode_s,
                          "goal_radius": rl.goal_radius, "defence_range": cfg.models.defence.range_m},
             "sizes": {"quad": 2 * (cfg.models.quad.arm_length + cfg.models.quad.prop_radius),
                       "wing": cfg.models.fixedwing.span},
             "models": [], "cities": [], "flights": {}}
    for mid in args.models:
        label, short, desc, spec = MODELS[mid]
        kind = "trained" if spec.startswith("sb3:") else "planner"
        index["models"].append({"id": mid, "label": label, "short": short, "desc": desc, "kind": kind})

    for j, cid in enumerate(args.cities):
        label, desc, where = CITIES[cid]
        city_dir = Path(where) if isinstance(where, str) else ensure_city(cfg, TEST_OFFSET + where)
        city_dir = (ROOT / city_dir) if not city_dir.is_absolute() else city_dir
        print(f"[{cid}] {label}: {city_dir}", flush=True)
        clouds = export_city(city_dir, out / "cities" / cid)

        c = load_config(None)
        c.terrain, c.models, c.physics = cfg.terrain, cfg.models, cfg.physics
        c.rl = replace(rl, city_dir=city_dir.as_posix())
        core = SwarmCore(c, seed=args.seed)
        index["cities"].append({"id": cid, "label": label, "desc": desc, "size": float(core.city.size),
                                "clouds": clouds, "kinds": [core.world.kind(i) for i in range(core.n)]})
        for mid in args.models:
            spec = MODELS[mid][3]
            if spec.startswith("sb3:"):
                spec = "sb3:" + (ROOT / spec[4:]).as_posix()
            policy = load_policy(spec, core.info())
            if hasattr(policy, "bind"):
                policy.bind(core)
            runs = []
            for ep in range(args.episodes):
                t0 = time.time()
                seed = args.seed + 1000 * j + ep             # the same scenario for every model
                rec = fly(core, policy, seed)
                rec.update(model=mid, city=cid, episode=ep + 1)
                path = out / "flights" / mid / cid / f"{ep + 1}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rec, separators=(",", ":")))
                runs.append({"episode": ep + 1, "file": path.relative_to(out).as_posix(),
                             "duration": rec["duration"], "counts": rec["counts"], "shots": rec["shots"]})
                print(f"  {mid:14s} ep {ep + 1}: {rec['counts']}  {rec['duration']:.0f} s sim, "
                      f"{time.time() - t0:.0f} s real, {path.stat().st_size / 1e3:.0f} kB", flush=True)
            index["flights"].setdefault(mid, {})[cid] = runs
        core.close()
        (out / "index.json").write_text(json.dumps(index, indent=1))  # usable after every city
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"done: {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
