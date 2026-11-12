# Setting up swarmsim on your own computer

Works on Windows, macOS and Linux. About 15 minutes, most of it downloads.
No GPU needed: the policy network is small and the physics runs on the CPU.

## 1. Install Miniforge (once)

Download and install Miniforge from https://conda-forge.org/download/ .
It gives you Python and the `conda` command without touching any Python you
already have.

Why conda and not plain `pip`: the physics engine (PyBullet) has to be compiled
when installed with pip on Windows, which needs Microsoft's C++ build tools.
The conda package is ready-made.

## 2. Unzip the project

Unzip `swarmsim_with_maps.zip` anywhere, for example `Documents/swarmsim`.
Open a terminal **in that folder** (Windows: open "Miniforge Prompt", then
`cd` to the folder). Every command below is run from there.

## 3. Create the environment and install

```bash
conda create -n swarmsim python=3.11 -y
conda activate swarmsim
conda install -c conda-forge pybullet -y
pip install -r requirements.txt
```

Linux only, optional: `pip` downloads the large GPU build of PyTorch by
default. To get the small CPU build instead, run this before the last line:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Each time you open a new terminal, run `conda activate swarmsim` first.

## 4. Check that it works (1 minute)

```bash
pytest tests -q
```

Expected: `20 passed`. (The ONNX part of one test is skipped automatically if
the optional ONNX packages are not installed.)

Then fly the bundled trained model once:

```bash
python -m swarmsim.rl.run --config runs/stage1_navigate/config.yaml --policy sb3:runs/stage1_navigate/model.zip --episodes 2
```

It prints how many drones reached their goal. Add `--gui` to watch in a 3-D
window.

Or open the control app, which shows the flight in 3-D in your browser and has
buttons for flying models, piloting a drone yourself and starting training:

```bash
python main.py
```

## 5. The maps

The maps are already in the zip, so nothing has to be generated or downloaded.

| Folder | What it is |
|--------|------------|
| `cache/city_002262c9eb`, `cache/city_cf69f2c11f` | the two 300 m training maps of `configs/quick_demo.yaml` |
| `cache/city_a64c3360bd`, `cache/city_950ae68d7d` | the two 800 m training maps of `configs/default.yaml` |
| `worlds/city_suburb`, `worlds/city_dense_core` | two extra showcase cities |

Training picks up the `cache/` maps automatically, as long as you run from the
project folder and leave the `terrain:` section of the config unchanged. If
you do change a terrain setting, a new map is generated on the spot (a few
seconds) and stored in `cache/` next to the others.

To train on one specific map, for example the suburb:

```bash
python -m swarmsim.rl.train --config configs/quick_demo.yaml --set rl.city_dir=worlds/city_suburb --run-dir runs/suburb
```

Each map folder has a `preview_map.png` you can open to see it.

## 6. Train

Recommended order: learn to navigate first, then switch the defence platform on.

```bash
# stage 1: navigation only
python -m swarmsim.rl.train --config configs/quick_demo.yaml --set rl.n_defence=0 --timesteps 1000000 --run-dir runs/my_stage1

# stage 2: continue from stage 1 with the defence platform
python -m swarmsim.rl.train --config configs/quick_demo.yaml --init runs/my_stage1/model.zip --timesteps 2000000 --run-dir runs/my_stage2
```

To continue from the models already in the zip instead of starting over, use
`--init runs/stage1_navigate/model.zip` (or `runs/stage2_defence/model.zip`).

What to expect:

* Speed was about 1,000 steps per second on a 2-core machine, so 1,000,000
  steps is roughly 15 to 20 minutes. Training runs in one process, so a
  faster single core helps more than many cores.
* Progress is written to `runs/<name>/progress.csv` after every update:
  mean reward and the share of drones that reached the goal, crashed, were
  hit, or timed out. Open it in Excel while training runs.
* Stage 1 reached about 70% goals after 600,000 steps in our test. Stage 2
  had not improved after 300,000 steps, so it needs much longer, or an easier
  platform, for example:
  `--set models.defence.range_m=70 --set models.defence.detection_range=100 --set models.defence.dispersion_deg=2.5`
* Stop early with Ctrl+C only if you do not need the model: it is saved at
  the end (and every `rl.train.checkpoint_every` steps if you set that above 0).

For the bigger 800 m city use `--config configs/default.yaml`.

## 7. Evaluate and send the result back

```bash
python -m swarmsim.rl.run --config runs/my_stage2/config.yaml --policy sb3:runs/my_stage2/model.zip --episodes 10
python -m swarmsim.rl.run --config runs/my_stage2/config.yaml --policy sb3:runs/my_stage2/model.zip --episodes 1 --record flight.gif
```

Compare against the hand-written baseline with `--policy scripted`.

To share a trained model, zip the whole `runs/my_stage2` folder. It holds
`model.zip`, `config.yaml` (needed to run the model) and `progress.csv`.

## If something goes wrong

| Message | Fix |
|---------|-----|
| `Microsoft Visual C++ 14.0 or greater is required` | PyBullet was installed with pip. Run `conda install -c conda-forge pybullet -y` and repeat `pip install -r requirements.txt`. |
| `ModuleNotFoundError: No module named 'swarmsim'` | The terminal is not in the project folder. `cd` into it. |
| `conda: command not found` | Use the "Miniforge Prompt" (Windows) or open a new terminal after installing Miniforge. |
| `Model expects N observations but the environment provides M` | Run the model with the `config.yaml` saved in its own run folder. |
| `--gui` shows nothing or fails | Needs a normal desktop session. It does not work over plain SSH. Use `--record flight.gif` instead. |
| `Unknown config key ...` | A setting name is misspelled. The message lists the valid names. |

More detail on every module and setting is in `README.md`.
