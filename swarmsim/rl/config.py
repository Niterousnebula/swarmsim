"""Hyperparameters of the RL module (environment, actions, rewards, training)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

ALL_DISCRETE_ACTIONS = ("hover", "forward", "back", "left", "right", "up", "down", "yaw_left", "yaw_right")


@dataclass
class TrainConfig:
    algo: str = "ppo"
    total_timesteps: int = 1_000_000     # agent-steps (all drones share one policy)
    n_worlds: int = 2                    # simulations run side by side (different city seeds)
    n_steps: int = 256                   # rollout length per drone
    batch_size: int = 1024
    n_epochs: int = 6
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    net_arch: tuple = (128, 128)
    device: str = "auto"
    run_dir: str = "runs/default"
    checkpoint_every: int = 100_000
    seed: int = 0


@dataclass
class RLConfig:
    # --- swarm composition --------------------------------------------------
    n_quads: int = 6
    n_fixedwing: int = 0
    n_defence: int = 1

    # --- world --------------------------------------------------------------
    city_dir: Optional[str] = None       # use a pre-built city bundle; None = build from `terrain`
    cache_dir: str = "cache"

    # --- episode ------------------------------------------------------------
    task: str = "navigate"               # "navigate" (reach a goal) or "coverage" (visit every cell)
    policy_hz: float = 10.0              # how often the policy picks an action
    max_episode_s: float = 90.0
    spawn_agl: tuple = (12.0, 35.0)      # start height above ground (quadcopters)
    goal_agl: tuple = (10.0, 35.0)
    fw_altitude: tuple = (3.0, 8.0)      # fixed-wing start and goal height above the tallest roof or tree on the map
    fw_mode: str = "search"              # fixed-wings: "search" = circle the map, fly to the target once seen
                                         # (see wingpilot.py); "policy" = flown by the policy like quadcopters
    fw_detect_range: float = 60.0        # m: a searching plane sees its target this far away (horizontally)
    min_goal_distance: float = 120.0
    goal_radius: float = 8.0
    shared_goal: bool = False            # all drones fly to the same goal
    ceiling_agl: float = 130.0           # flying above this ends the episode
    coverage_cell_m: float = 50.0
    defence_relocate_s: float = 120.0    # the platform moves to a new random spot this often (training)

    # --- actions ------------------------------------------------------------
    action_mode: str = "discrete"        # "discrete" or "continuous"
    discrete_actions: tuple = ALL_DISCRETE_ACTIONS
    move_speed: float = 6.0              # m/s for forward/back/left/right
    climb_speed: float = 2.5             # m/s for up/down
    yaw_rate: float = 0.9                # rad/s for yaw_left/yaw_right
    fw_turn_rate: float = 0.30           # rad/s for fixed-wing left/right
    fw_speed_step: float = 3.0           # m/s added/removed by forward/back (fixed-wing)

    # --- observations -------------------------------------------------------
    n_rays: int = 12                     # horizontal range-finder rays
    ray_range: float = 40.0
    k_neighbors: int = 3
    neighbor_range: float = 60.0
    observe_defence: bool = True
    observe_projectiles: bool = True

    # --- rewards ------------------------------------------------------------
    r_progress: float = 0.1              # per metre closer to the goal
    r_goal: float = 20.0
    r_crash: float = -20.0
    r_hit: float = -20.0
    r_out_of_bounds: float = -10.0
    r_time: float = -0.02                # per policy step
    r_proximity: float = -0.1            # scaled penalty when an obstacle is closer than `proximity_m`
    proximity_m: float = 4.0
    r_cell: float = 2.0                  # coverage: reward for a newly visited cell
    team_share: float = 0.0              # fraction of each drone's goal/cell reward given to all others

    train: TrainConfig = field(default_factory=TrainConfig)
