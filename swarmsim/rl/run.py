"""Run a policy plugin in the simulator: evaluation numbers, live view, GIF.

    python -m swarmsim.rl.run --config configs/default.yaml --policy sb3:runs/default/model.zip
    python -m swarmsim.rl.run --policy scripted --episodes 5 --record flight.gif
    python -m swarmsim.rl.run --policy sb3:runs/default/model.zip --gui
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..config import SimConfig, load_config
from .core import SwarmCore
from .plugin import load_policy


def run_policy(cfg: SimConfig, policy_spec: str, episodes: int = 5, seed: int = 100, gui: bool = False,
               record: str | None = None, realtime: bool = False, frame_every: int = 2,
               verbose: bool = True) -> dict:
    core = SwarmCore(cfg, seed=seed, gui=gui)
    policy = load_policy(policy_spec, core.info())
    frames, totals = [], {"goal": 0, "crash": 0, "hit": 0, "out_of_bounds": 0, "timeout": 0}
    returns, shots = [], 0
    if record:
        core.world.enable_projectile_visuals(radius=0.8)
    for ep in range(episodes):
        obs = core.reset(seed + ep)
        policy.reset()
        if gui and ep == 0:
            _gui_camera(core)
        ep_ret = np.zeros(core.n); step = 0
        while core.active.any():
            t0 = time.time()
            actions = policy.act(obs)
            obs, rew, term, trunc, outcome = core.step(actions)
            ep_ret += rew
            for o in outcome:
                if o:
                    totals[o] += 1
            if record and ep == 0 and step % frame_every == 0:
                frames.append(_frame(core))
            step += 1
            if gui:
                _update_markers(core)
            if gui or realtime:
                time.sleep(max(0.0, 1.0 / cfg.rl.policy_hz - (time.time() - t0)))
        returns.append(float(ep_ret.mean()))
        shots += sum(d.shots for d in core.world.defences)
        if verbose:
            print(f"episode {ep + 1}/{episodes}: mean return {returns[-1]:.1f}")
    core.close()
    n = max(sum(totals.values()), 1)
    result = {"policy": policy_spec, "episodes": episodes, "drones_per_episode": core.n,
              "mean_return": float(np.mean(returns)),
              "rates": {k: round(v / n, 3) for k, v in totals.items()}, "counts": totals,
              "defence_shots": shots}
    if record and frames:
        from PIL import Image
        if record.lower().endswith(".mp4"):
            import imageio
            imageio.mimsave(record, frames, fps=cfg.rl.policy_hz / frame_every, macro_block_size=None)
        else:
            imgs = [Image.fromarray(f).quantize(96) for f in frames]
            imgs[0].save(record, save_all=True, append_images=imgs[1:], loop=0, optimize=True,
                         duration=int(1000 * frame_every / cfg.rl.policy_hz))
        result["recording"] = record
    return result


def _update_markers(core):
    """Drones are far too small to see from an overview camera, so recordings add
    visual-only markers: blue = quadcopter, orange = fixed-wing, green = goal,
    red = defence platform, yellow = projectile. Markers have no collision shape."""
    import pybullet as pb
    w = core.world
    hidden = [0.0, 0.0, -500.0]

    def ball(radius, rgba):
        vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=w.c)
        return pb.createMultiBody(0, -1, vis, hidden, physicsClientId=w.c)

    if not hasattr(core, "_markers"):
        core._markers = {
            "drone": [ball(1.7, [0.1, 0.4, 1.0, 1] if w.kind(i) == "quad" else [1.0, 0.5, 0.05, 1]) for i in range(core.n)],
            "goal": [ball(2.0, [0.1, 0.95, 0.25, 1]) for _ in range(core.n)],
            "defence": [ball(3.0, [0.9, 0.1, 0.1, 1]) for _ in w.defences],
        }
    m = core._markers
    for i in range(core.n):
        on = bool(w.alive[i] and core.active[i])
        pb.resetBasePositionAndOrientation(m["drone"][i], (w.pos[i] + [0, 0, 2.0]).tolist() if on else hidden,
                                           [0, 0, 0, 1], physicsClientId=w.c)
        pb.resetBasePositionAndOrientation(m["goal"][i], core.goal[i].tolist() if on else hidden,
                                           [0, 0, 0, 1], physicsClientId=w.c)
    for b, d in zip(m["defence"], w.defences):
        pb.resetBasePositionAndOrientation(b, (d.muzzle + [0, 0, 5.0]).tolist(), [0, 0, 0, 1], physicsClientId=w.c)


def _gui_camera(core):
    """Start the 3-D window on an overview of the whole map (the default view sits
    one metre from the origin). Set once, so the user can orbit and zoom afterwards."""
    import pybullet as pb
    w = core.world
    pb.resetDebugVisualizerCamera(cameraDistance=0.9 * w.city.size, cameraYaw=35.0, cameraPitch=-40.0,
                                  cameraTargetPosition=[0.0, 0.0, 10.0], physicsClientId=w.c)


def _frame(core, width=480, height=360):
    """Chase-style overview: look at the swarm centroid from above and behind."""
    _update_markers(core)
    w = core.world
    alive = w.alive
    centre = w.pos[alive].mean(0) if alive.any() else np.array([0, 0, 20.0])
    spread = float(np.linalg.norm(w.pos[alive] - centre, axis=1).max()) if alive.any() else 50.0
    dist = float(np.clip(1.6 * spread + 60.0, 80.0, 1.2 * w.city.size))
    eye = centre + np.array([-0.55 * dist, -0.65 * dist, 0.6 * dist])
    return w.render(width, height, eye=eye, target=centre)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default="scripted")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--gui", action="store_true", help="open the PyBullet 3-D window")
    ap.add_argument("--record", default=None, help="write a .gif or .mp4 (needs imageio-ffmpeg) of the first episode")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config key, e.g. --set rl.move_speed=8 (repeatable)")
    args = ap.parse_args()
    cfg = load_config(args.config, set_flags=args.set)
    print(json.dumps(run_policy(cfg, args.policy, args.episodes, args.seed, args.gui, args.record), indent=2))


if __name__ == "__main__":
    main()
