# Running the generalisation experiment

Written for the repository owner. Everything here lives on the `research` branch; `main` is
untouched apart from the 3-D viewer and the two real maps.

**Question:** does training on more procedurally generated cities make a swarm policy work
better in cities it has never seen, including real ones?

## Where things stand

- **Works:** an A\* planner that routes around the defence (about 99% goals, no hits), and a
  learned policy, imitation of that planner plus PPO fine-tuning, that is hit 6.5% of the time
  where the scripted baseline and plain PPO are hit 18 to 20% (300 m suburb, one platform of
  60 m range, 200 drone flights per method, one training seed).
- **Not done:** this experiment. It is built and smoke-tested, but the full grid has never
  been run.
- **Background reading:** `swarmsim/rl/planner.py`, `swarmsim/rl/imitate.py` and
  `scripts/benchmark.py` each start with a short explanation.

## Setup

In your existing clone:

```bash
git fetch
git checkout research
git pull
python -m pytest tests -q         # expect: 27 passed
```

If `scripts/experiment_generalisation.py` is missing after the pull, the last step has not
been pushed yet. Ask for `swarmsim_research_step4.zip`, unpack it over the repository root
(it only adds or replaces nine files), run the tests, and commit.

Your existing `.venv` is enough: the branch needs no new packages and no GPU. Allow about
400 MB of disk for generated cities and models.

On a new computer instead:

```bash
git clone https://github.com/Niterousnebula/swarmsim
cd swarmsim
git checkout research
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux / macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest tests -q
```

## Step 1: check that everything runs (about 5 minutes)

```bash
python scripts/experiment_generalisation.py --quick
```

It should finish with a small table. The numbers mean nothing at this budget; it only proves
the pipeline works on your machine. You can delete `runs/gen_quick` and `results/gen_quick` afterwards.

## Step 2: the experiment

```bash
python scripts/experiment_generalisation.py
```

What it does, for pool sizes N = 1, 4, 16, 64 and seeds 0, 1, 2 (12 runs):

1. trains a network to imitate the A\* planner on N generated cities (DAgger, 10 rounds),
2. fine-tunes it with PPO for 500,000 steps on the same cities,
3. tests it on its training cities, on 10 unseen generated cities, and on the real maps
   `worlds/delhi_cp` and `worlds/nsut`.

The planner and scripted baselines are tested once on the same episodes.

- **It can be interrupted.** Press Ctrl+C and run the same command again later: finished stages
  are skipped.
- **Progress** is printed after every run, with the number of runs left.
- **Keep the computer awake** (disable sleep) while it runs.
- **Time:** see "Timings" below. Expect roughly a day on a laptop.

To split the work between two computers, run different seeds on each and copy the
`results/gen/*.json` files into one folder afterwards:

```bash
python scripts/experiment_generalisation.py --seeds 0 1      # computer A
python scripts/experiment_generalisation.py --seeds 2 3 4    # computer B
```

## Step 3: results

```bash
python scripts/experiment_generalisation.py --summary
```

prints the table (mean ± standard deviation over seeds) and writes `results/gen/summary.csv`.
Per-run details are in `results/gen/N<cities>_s<seed>.json`, the baselines in
`results/gen/baselines.json`, training curves in `runs/gen/<run>/progress.csv`.

Commit the whole `results/gen` folder to the `research` branch. The trained models
(`runs/gen/*/model.zip`) are ignored by git and only needed to fly or re-test them:

```bash
git add results/gen
git commit -m "Generalisation experiment results"
git push
```

To watch a trained model afterwards:

```bash
python main.py demo --model runs/gen/N16_s0
```

## Timings

Measured on a 2-core Linux machine without GPU: one run (imitation + PPO + testing) took about
28 minutes, the baselines about 10 minutes. The Windows laptop used so far was roughly three
times slower on the same benchmarks, which would put the default 12 runs at 16 to 18 hours.
A desktop with fast cores will be quicker; the simulation is single-threaded, so core speed
matters and core count does not.

## How the test scenario is built

- One defence platform with a 60 m firing range, placed on open ground.
- Quadcopters start within 200 m of it but outside the firing zone; each goal is 100 to 250 m
  from its start, also outside the zone. This keeps the task equally hard on a 300 m generated
  city and on a 1 km real map.
- Generated cities differ in street pattern, block size, density, building heights, relief and
  river, chosen at random per city index (`rl.randomize_cities`).
- On the real maps the defence matters less than on generated ones, because tall, dense
  buildings block its line of sight: a planner that ignores it is hit about 5% of the time
  there, against about 25% on generated cities (small samples, expect these to move).

## What to expect

If generalisation works the way it does in other procedurally generated benchmarks, the
"unseen" and real-city columns should improve as N grows, while the "train" column stays high
for every N (a policy trained on one city memorises it). A flat or reversed trend is also a
result worth reporting; do not rerun with other settings until the first table is in.

## Known limits (read before interpreting the table)

- **Weakened defence.** The platform has a 60 m range, not the default 120 m, because at 120 m
  on a 300 m map there is almost no safe route.
- **Varied cities are hard.** The one full-budget trial so far (16 cities, an earlier version of
  the scenario) reached 91% goals on its training cities and 73.5% on unseen ones, with 11.5%
  crashes into tall buildings.
- **Real maps test navigation more than avoidance**, for the line-of-sight reason given above.
- **The planner is an oracle.** It reads the map and the platform position from the simulator;
  treat it as an upper bound.
- **Three seeds is the minimum.** Use `--seeds 0 1 2 3 4` if the time is available.

## Useful options

| Option                               | Meaning                                        |
| ------------------------------------ | ---------------------------------------------- |
| `--sizes 1 4 16 64`                  | pool sizes to train                            |
| `--seeds 0 1 2`                      | repetitions (5 is better for a paper)          |
| `--ppo-steps 500000`                 | PPO fine-tuning budget per run                 |
| `--bc-rounds 10 --bc-samples 30000`  | imitation budget per run                       |
| `--test-cities 10 --episodes 5`      | size of the unseen test set                    |
| `--real worlds/delhi_cp worlds/nsut` | real maps to test on (add more folders freely) |

Settings of the scenario itself (defence range, number of drones, trip length) are in
`configs/gen_experiment.yaml`. Change them only before the first run, never between runs.
