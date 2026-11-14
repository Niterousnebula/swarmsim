import numpy as np

from swarmsim.config import load_config
from swarmsim.rl.core import SwarmCore
from swarmsim.rl.planner import astar
from swarmsim.rl.plugin import load_policy
from swarmsim.rl.run import run_policy

SMALL = {"terrain": {"size_m": 250, "seed": 9, "textures": {"ground_px": 512}},
         "models": {"defence": {"range_m": 50.0, "detection_range": 80.0}},
         "rl": {"n_quads": 4, "n_defence": 1, "min_goal_distance": 80, "max_episode_s": 60, "clear_of_defence": True}}


def test_astar_goes_around_walls_and_expensive_cells():
    cost = np.ones((9, 9))
    cost[4, 1:8] = np.inf                                   # a wall with gaps at both ends
    path = astar(cost, (0, 4), (8, 4))
    assert path[0] == (0, 4) and path[-1] == (8, 4)
    assert all(np.isfinite(cost[c]) for c in path) and any(c[1] in (0, 8) for c in path)
    cost[4, 0] = cost[4, 8] = np.inf
    assert astar(cost, (0, 4), (8, 4)) is None              # sealed off
    cost = np.ones((9, 9)); cost[2:7, 2:7] = 25.0           # a costly zone in the middle
    path = astar(cost, (4, 0), (4, 8))
    assert all(cost[c] == 1.0 for c in path)                # cheaper to walk around it


def test_clear_of_defence_keeps_starts_and_goals_out_of_range():
    core = SwarmCore(load_config(None, SMALL), seed=3)
    for seed in range(5):
        core.reset(seed)
        mz = core.world.defences[0].muzzle
        for pts in (core.world.pos, core.goal):
            assert np.all(np.hypot(pts[:, 0] - mz[0], pts[:, 1] - mz[1]) >= 50.0)
    core.close()


def test_planner_route_avoids_the_firing_zone():
    core = SwarmCore(load_config(None, SMALL), seed=5)
    aware, blind = load_policy("planner", core.info()), load_policy("planner_blind", core.info())
    aware.bind(core); blind.bind(core)
    inside = {"planner": 0, "planner_blind": 0}
    for seed in range(4):
        obs = core.reset(seed)
        mz = core.world.defences[0].muzzle
        for name, pol in (("planner", aware), ("planner_blind", blind)):
            pol.reset(); pol.act(obs)
            for p in pol.paths.values():
                d = np.hypot(p["pts"][:, 0] - mz[0], p["pts"][:, 1] - mz[1])
                seen = core.world.line_of_sight(mz, np.column_stack([p["pts"], np.full(len(d), p["z"])]))
                inside[name] += int(np.sum((d < 50.0) & seen))
    core.close()
    assert inside["planner"] == 0 and inside["planner_blind"] > 0


def test_planner_flies_to_the_goal():
    res = run_policy(load_config(None, SMALL), "planner", episodes=2, seed=11, verbose=False)
    assert res["counts"]["goal"] >= 6 and res["counts"]["hit"] <= 1


def test_imitation_produces_a_loadable_model(tmp_path):
    from swarmsim.rl.imitate import imitate
    cfg = load_config(None, {**SMALL, "rl": {**SMALL["rl"], "max_episode_s": 8, "train": {"n_worlds": 1}}})
    model = imitate(cfg, tmp_path / "bc", rounds=2, samples_per_round=150, epochs=1, verbose=False)
    assert model.exists() and (tmp_path / "bc" / "imitation.csv").read_text().count("\n") == 3
    res = run_policy(cfg, f"sb3:{model}", episodes=1, seed=1, verbose=False)
    assert sum(res["counts"].values()) == 4


def test_city_pool_gives_each_index_its_own_style_and_rotates(tmp_path):
    from swarmsim.rl.core import randomized_terrain
    from swarmsim.rl.vec import SwarmVecEnv
    base = load_config(None, SMALL).terrain
    a, b, again = randomized_terrain(base, 1), randomized_terrain(base, 2), randomized_terrain(base, 1)
    assert a == again and a != b and a.size_m == base.size_m          # deterministic, different, same map size
    cfg = load_config(None, {**SMALL, "terrain": {"size_m": 200, "seed": 9, "textures": {"ground_px": 256}},
                             "rl": {**SMALL["rl"], "train_cities": 3, "randomize_cities": True, "city_swap_steps": 40,
                                    "cache_dir": str(tmp_path), "train": {"n_worlds": 1}}})
    env = SwarmVecEnv(cfg, n_worlds=1, seed=0)
    obs = env.reset()
    swaps = 0
    for _ in range(40):
        env.step_async(np.zeros(env.num_envs, int))
        obs, rew, done, info = env.step_wait()
        swaps += int(done.all())
    env.close()
    assert swaps >= 2 and len(env.cities_seen) >= 2 and obs.shape == (4, env.observation_space.shape[0])


def test_goal_distance_cap():
    cfg = load_config(None, {**SMALL, "rl": {**SMALL["rl"], "min_goal_distance": 60, "max_goal_distance": 90}})
    core = SwarmCore(cfg, seed=2)
    for seed in range(4):
        core.reset(seed)
        d = np.hypot(*(core.goal[:, :2] - core.world.pos[:, :2]).T)
        assert np.all(d >= 59.0) and np.all(d <= 91.0)
    core.close()


def test_collision_reflex_stops_a_drone_flying_at_a_wall():
    """A drone told to fly forward into the tallest building crashes without the reflex and
    climbs over it with the reflex."""
    from shapely.geometry import Polygon
    results = {}
    for reflex in (False, True):
        cfg = load_config(None, {**SMALL, "rl": {**SMALL["rl"], "n_quads": 1, "n_defence": 0, "max_episode_s": 60,
                                                 "collision_reflex": reflex}})
        core = SwarmCore(cfg, seed=4)
        core.reset(0)
        b = max((q for q in core.city.buildings if not q.stacked), key=lambda q: q.height)
        minx, miny, maxx, maxy = Polygon(b.footprint).bounds
        start = np.array([minx - 12.0, (miny + maxy) / 2, b.ground_z + 0.5 * b.height])
        core.world.spawn(0, start)                               # 12 m west of the wall, facing east
        core.observe()
        forward, outcome, top = core.actions.index("forward"), None, start[2]
        for _ in range(200):
            _, _, _, _, out = core.step(np.array([forward]))
            top = max(top, float(core.world.pos[0, 2]))
            if out[0] or core.world.pos[0, 0] > maxx + 3.0:
                outcome = out[0]
                break
        results[reflex] = (outcome, top - start[2], core.reflex_count)
        core.close()
    assert results[False][0] == "crash" and results[False][2] == 0
    assert results[True][0] != "crash" and results[True][1] > 0.4 * 6.0 and results[True][2] > 0
