"""One entry point for swarmsim.

    python main.py                                  open the control app: live 3-D view, piloting, training
    python main.py app --model stage2               ... starting with the model trained against the defence platform
    python main.py demo --model stage2              fly a model in PyBullet's own window instead
    python main.py demo --model scripted --no-gui --episodes 5
    python main.py demo --record flight.gif         save the first episode instead of opening a window
    python main.py train --name my_stage1 --timesteps 1000000
    python main.py train --name my_stage2 --defence --init my_stage1
    python main.py demo --model my_stage2           watch a model you trained
    python main.py test                             run the test suite

It can be started with any Python: if the packages are not importable it restarts
itself with the project's own environment in .venv.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
COMMANDS = ("app", "demo", "train", "test")
BUNDLED = {"stage1": "runs/stage1_navigate", "stage2": "runs/stage2_defence"}
BUILTIN_POLICIES = ("scripted", "random")
DEFAULT_CONFIG = "configs/quick_demo.yaml"


def _ensure_environment():
    """Restart inside .venv when this interpreter does not have the dependencies."""
    if importlib.util.find_spec("pybullet") and importlib.util.find_spec("stable_baselines3"):
        return
    if VENV_PYTHON.exists() and not os.environ.get("SWARMSIM_REEXEC"):
        env = dict(os.environ, SWARMSIM_REEXEC="1")
        raise SystemExit(subprocess.call([str(VENV_PYTHON), str(ROOT / "main.py"), *sys.argv[1:]], env=env))
    raise SystemExit("swarmsim's dependencies are not installed for this Python. See SETUP.md.")


def _run_dir(name: str) -> Path:
    """'stage1', a folder name inside runs/, or a path to a run folder."""
    for candidate in (Path(BUNDLED.get(name, name)), Path("runs") / name):
        if (candidate / "model.zip").exists():
            return candidate
    raise SystemExit(f"No trained model found for '{name}'. Expected one of {sorted(BUNDLED)}, "
                     f"{' / '.join(BUILTIN_POLICIES)}, or a folder that contains model.zip.")


def cmd_app(args):
    from swarmsim.app import serve

    name = args.model if args.model in BUILTIN_POLICIES else _run_dir(args.model).name
    serve(name, port=args.port, open_browser=not args.no_browser)


def cmd_demo(args):
    from swarmsim.config import load_config
    from swarmsim.rl.run import run_policy

    if args.model in BUILTIN_POLICIES:
        config, policy = args.config or DEFAULT_CONFIG, args.model
    else:
        run = _run_dir(args.model)                 # a model must run with the config it was trained with
        config, policy = args.config or run / "config.yaml", f"sb3:{(run / 'model.zip').as_posix()}"
    cfg = load_config(config, set_flags=args.set)
    gui = not (args.no_gui or args.record)
    if gui:
        print("Opening the 3-D window. Click inside it once, then use the keys listed below.")
    result = run_policy(cfg, policy, episodes=args.episodes, seed=args.seed, gui=gui, record=args.record)
    print(json.dumps(result, indent=2))


def cmd_train(args):
    from dataclasses import replace

    from swarmsim.config import load_config
    from swarmsim.rl.train import train

    sets = list(args.set)
    if not args.defence:
        sets.insert(0, "rl.n_defence=0")           # stage 1: learn to navigate before facing the platform
    cfg = load_config(args.config or DEFAULT_CONFIG, set_flags=sets)
    over = {"run_dir": (Path("runs") / args.name).as_posix()}
    if args.timesteps is not None:
        over["total_timesteps"] = args.timesteps
    cfg.rl.train = replace(cfg.rl.train, **over)
    init = str(_run_dir(args.init) / "model.zip") if args.init else None
    print(f"Training {cfg.rl.train.total_timesteps:,} steps into {over['run_dir']} "
          f"(progress in {over['run_dir']}/progress.csv)")
    print("saved", train(cfg, init_model=init))
    print(f"Watch it with: python main.py demo --model {args.name}")


def cmd_test(args):
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "tests", "-q", *args.pytest_args]))


def main():
    _ensure_environment()
    os.chdir(ROOT)                                 # configs, cache/ and runs/ are relative to the project
    sys.path.insert(0, str(ROOT))

    argv = sys.argv[1:]
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv = ["app", *argv]                      # `python main.py` on its own opens the control app

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    set_help = "override any config key, e.g. --set physics.wind.mean_speed=8 (repeatable)"

    app = sub.add_parser("app", help="control app in the browser: 3-D view, piloting, training (default)")
    app.add_argument("--model", default="stage1",
                     help="what to fly first: stage1, stage2, scripted, random, or a run folder (default: stage1)")
    app.add_argument("--port", type=int, default=8765)
    app.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    app.set_defaults(func=cmd_app)

    demo = sub.add_parser("demo", help="fly a policy in PyBullet's window and print how the drones did")
    demo.add_argument("--model", default="stage1",
                      help="stage1, stage2, scripted, random, or the name/path of a run folder (default: stage1)")
    demo.add_argument("--episodes", type=int, default=2)
    demo.add_argument("--seed", type=int, default=100)
    demo.add_argument("--no-gui", action="store_true", help="no window, just the numbers (much faster)")
    demo.add_argument("--record", default=None, metavar="FILE", help="write the first episode to a .gif (no window)")
    demo.add_argument("--config", default=None, help="config file; defaults to the one saved with the model")
    demo.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help=set_help)
    demo.set_defaults(func=cmd_demo)

    tr = sub.add_parser("train", help="train a PPO policy into runs/<name>")
    tr.add_argument("--name", required=True, help="run name; the model is saved in runs/<name>")
    tr.add_argument("--timesteps", type=int, default=None, help="default: the value in the config file")
    tr.add_argument("--defence", action="store_true", help="train with the defence platform switched on")
    tr.add_argument("--init", default=None, metavar="MODEL", help="continue from stage1, stage2 or another run")
    tr.add_argument("--config", default=None, help=f"default: {DEFAULT_CONFIG}")
    tr.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help=set_help)
    tr.set_defaults(func=cmd_train)

    te = sub.add_parser("test", help="run the test suite")
    te.add_argument("pytest_args", nargs="*", help="extra arguments passed to pytest")
    te.set_defaults(func=cmd_test)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
