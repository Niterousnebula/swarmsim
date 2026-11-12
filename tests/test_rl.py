import numpy as np
import pytest

from swarmsim.config import load_config
from swarmsim.rl import SwarmCore, SwarmEnv, load_policy
from swarmsim.rl.plugin import PolicyPlugin, register

SMALL = {"terrain": {"size_m": 250, "seed": 9, "textures": {"ground_px": 512}}}


def cfg(tmp_path, **rl):
    return load_config(None, {**SMALL, "rl": {"cache_dir": str(tmp_path / "cache"), "n_quads": 3,
                                              "max_episode_s": 6, **rl}})


def test_pettingzoo_api(tmp_path):
    from pettingzoo.test import parallel_api_test
    env = SwarmEnv(cfg(tmp_path))
    parallel_api_test(env, num_cycles=80)
    env.close()


def test_action_hyperparameters(tmp_path):
    c = cfg(tmp_path, discrete_actions=["hover", "up", "forward"], climb_speed=3.0, move_speed=4.0, n_defence=0)
    c.physics.wind.mean_speed = 0.0; c.physics.wind.gust_intensity = 0.0
    core = SwarmCore(c)
    assert core.action_space.n == 3
    core.reset(0)
    w = core.world
    for i in range(3):
        w.spawn(i, [20.0 * i, 0, w.city.max_height + 20], yaw=0.0)
    for _ in range(25):
        core.step([core.actions.index("hover"), core.actions.index("up"), core.actions.index("forward")])
    assert np.linalg.norm(w.vel[0]) < 0.2
    assert w.vel[1, 2] == pytest.approx(3.0, abs=0.2)
    assert w.vel[2, 0] == pytest.approx(4.0, abs=0.3)
    core.close()
    with pytest.raises(ValueError):
        SwarmCore(cfg(tmp_path, discrete_actions=["hover", "teleport"]))


def test_continuous_actions_and_reward_for_progress(tmp_path):
    c = cfg(tmp_path, action_mode="continuous", n_defence=0)
    core = SwarmCore(c)
    core.reset(1)
    w = core.world
    start = np.array([0.0, 0.0, w.city.max_height + 20]); w.spawn(0, start, yaw=0.0)
    core.goal[0] = start + [80.0, 0, 0]; core.prev_dist[0] = 80.0
    total = 0.0
    for _ in range(30):
        _, rew, *_ = core.step(np.array([[1, 0, 0, 0]] * 3, np.float32))
        total += rew[0]
    assert w.pos[0, 0] > 10.0 and total > 0.5        # flying towards the goal is rewarded
    core.close()


def test_plugins(tmp_path):
    c = cfg(tmp_path)
    core = SwarmCore(c)
    obs = core.reset(0)
    for spec in ("random", "scripted"):
        a = load_policy(spec, core.info()).act(obs)
        assert a.shape == (3,) and all(core.action_space.contains(int(x)) for x in a)

    @register
    class AlwaysUp(PolicyPlugin):
        name = "always_up"

        def act(self, obs):
            return np.full(len(obs), self.info["action_names"].index("up"))

    assert (load_policy("always_up", core.info()).act(obs) == core.actions.index("up")).all()
    core.close()


def test_train_save_load_run(tmp_path):
    from swarmsim.rl.plugin import export_torchscript
    from swarmsim.rl.run import run_policy
    from swarmsim.rl.train import train
    c = cfg(tmp_path)
    c.rl.train.total_timesteps = 600; c.rl.train.n_steps = 64; c.rl.train.batch_size = 96
    c.rl.train.n_worlds = 1; c.rl.train.run_dir = str(tmp_path / "run"); c.rl.train.device = "cpu"
    model = train(c, verbose=0)
    assert model.exists() and (tmp_path / "run" / "progress.csv").exists()
    res = run_policy(c, f"sb3:{model}", episodes=1, verbose=False)
    assert sum(res["counts"].values()) == 3
    ts = export_torchscript(str(model), str(tmp_path / "policy.pt"))
    res2 = run_policy(c, f"torchscript:{ts}", episodes=1, verbose=False)
    assert res2["counts"] == res["counts"]            # same network, same decisions
    pytest.importorskip("onnx"); pytest.importorskip("onnxruntime")
    from swarmsim.rl.plugin import export_onnx
    ox = export_onnx(str(model), str(tmp_path / "policy.onnx"))
    res3 = run_policy(c, f"onnx:{ox}", episodes=1, verbose=False)
    assert res3["counts"] == res["counts"]
