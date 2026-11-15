"""Runs training as a separate process so the 3-D view stays live, and reports
its progress from the run's progress.csv."""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

DEFAULT_CONFIG = "configs/quick_demo.yaml"
RUNS = Path("runs")
BUILTIN = ("planner", "planner_blind", "scripted", "random")
BUILTIN_LABELS = {      # (short label for the menu, longer description shown on hover)
    "planner": ("A* planner (oracle)", "Plans a route with full knowledge of the map and the turret"),
    "planner_blind": ("A* planner, threat-blind", "Same planner, but it ignores the turret"),
    "scripted": ("Scripted baseline", "Hand-written rules, no training"),
    "random": ("Random actions", "Picks a random action every step"),
}
# The strongest models from the generalisation experiment (results/gen_reflex), best first.
FEATURED = {
    "gen/N1_s2": ("Best model (DAgger + PPO)", "Best overall in the paper: trained on 1 city, seed 2"),
    "gen/N64_s1": ("DAgger + PPO, 64 cities", "Trained on 64 generated cities, seed 1"),
    "bc_ppo": ("DAgger + PPO, suburb map", "Trained on the single suburb map of the benchmark"),
}


def list_models() -> list[dict]:
    """Everything that can be flown: the best trained models, the planners and baselines,
    then every other trained run in runs/ (newest first)."""
    out = [{"name": n, "kind": "featured", "label": label, "desc": desc}
           for n, (label, desc) in FEATURED.items() if (RUNS / n / "model.zip").exists()]
    out += [{"name": n, "kind": "builtin", "label": BUILTIN_LABELS[n][0], "desc": BUILTIN_LABELS[n][1]}
            for n in BUILTIN]
    runs = [d for d in RUNS.iterdir() if (d / "model.zip").exists()] if RUNS.exists() else []
    runs.sort(key=lambda d: (d / "model.zip").stat().st_mtime, reverse=True)
    out += [{"name": d.name, "kind": "trained", "label": d.name} for d in runs if d.name not in FEATURED]
    return out


class Trainer:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.log = None
        self.info: dict = {}

    # ------------------------------------------------------------------ control
    def start(self, name: str, timesteps: int, defence: bool, init: str | None) -> dict:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                raise ValueError("A training run is already in progress.")
            name = (name or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", name):
                raise ValueError("Name the run with letters, digits, - or _ (no spaces).")
            run_dir = RUNS / name
            if run_dir.exists() and any(run_dir.iterdir()):
                raise ValueError(f"runs/{name} already exists. Pick another name.")
            timesteps = int(timesteps)
            if not 1_000 <= timesteps <= 50_000_000:
                raise ValueError("Timesteps must be between 1,000 and 50,000,000.")
            config = DEFAULT_CONFIG
            cmd = [sys.executable, "-m", "swarmsim.rl.train", "--run-dir", run_dir.as_posix(),
                   "--timesteps", str(timesteps), "--set", f"rl.n_defence={1 if defence else 0}"]
            if init:
                src = RUNS / init
                if not (src / "model.zip").exists():
                    raise ValueError(f"No model in runs/{init} to start from.")
                config = (src / "config.yaml").as_posix()       # same observation and action layout
                cmd += ["--init", (src / "model.zip").as_posix()]
            cmd += ["--config", config]
            run_dir.mkdir(parents=True, exist_ok=True)
            self.log = open(run_dir / "train.log", "w")
            # Own process group: Ctrl+C in the app's terminal must not kill training before it saves.
            group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                     else {"start_new_session": True})
            self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, **group)
            self.info = {"name": name, "total": timesteps, "defence": bool(defence), "init": init,
                         "started": time.time(), "stopping": False, "ended": None}
        return self.status()

    def stop(self) -> dict:
        """Ask the trainer to finish the current step and save the model."""
        with self.lock:
            if self.proc and self.proc.poll() is None:
                (RUNS / self.info["name"] / "stop").touch()
                self.info["stopping"] = True
        return self.status()

    def shutdown(self, wait_s: float = 60.0):
        """The app is closing: let a running training save its model first."""
        with self.lock:
            proc, name = self.proc, self.info.get("name")
        if not proc or proc.poll() is not None:
            return
        print(f"Saving training run runs/{name} before quitting (Ctrl+C again to drop it)...")
        (RUNS / name / "stop").touch()
        try:
            proc.wait(timeout=wait_s)
            print("Saved." if (RUNS / name / "model.zip").exists() else "Training ended without saving.")
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            proc.terminate()

    # ------------------------------------------------------------------- status
    def status(self) -> dict:
        with self.lock:
            if not self.proc:
                return {"state": "idle"}
            info, code = dict(self.info), self.proc.poll()
            run_dir = RUNS / info["name"]
            if code is not None and self.info["ended"] is None:
                self.info["ended"] = info["ended"] = time.time()
                if self.log:
                    self.log.close()
        rows = _progress(run_dir / "progress.csv")
        done = rows[-1][0] if rows else 0
        if code is None:
            state = "stopping" if info["stopping"] else ("running" if rows else "starting")
        elif code == 0 and (run_dir / "model.zip").exists():
            state = "done"
        else:
            state = "failed"
        out = {"state": state, "name": info["name"], "total": info["total"], "timesteps": done,
               "defence": info["defence"], "init": info["init"],
               "elapsed": round((info["ended"] or time.time()) - info["started"], 1),
               "rows": _thin(rows, 240)}
        if state == "failed":
            out["error"] = _tail(run_dir / "train.log")
        return out


def _progress(path: Path) -> list[list[float]]:
    """[timesteps, mean_return, goal, crash, hit, out_of_bounds, timeout] per rollout."""
    rows = []
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append([int(r["timesteps"]), float(r["mean_return"]) if r["mean_return"] else None]
                                + [float(r[k]) for k in ("goal", "crash", "hit", "out_of_bounds", "timeout")])
                except (ValueError, KeyError, TypeError):
                    continue                              # a row that is still being written
    except OSError:
        pass
    return rows


def _thin(rows: list, limit: int) -> list:
    if len(rows) <= limit:
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit - 1)] + [rows[-1]]


def _tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").strip().splitlines()[-lines:])
    except OSError:
        return "Training stopped unexpectedly."
