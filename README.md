# swarmsim

A modular drone-swarm simulation environment in Python. Four modules that only
talk to each other through files and plain data, so each can be replaced or
extended on its own.

| # | Module | What it does | Entry point |
|---|--------|--------------|-------------|
| 1 | `swarmsim.terrain` | Procedural city: roads, zoned blocks, textured buildings, trees, river, relief. OSM and satellite imagery as reference data. | `scripts/generate_city.py` |
| 2 | `swarmsim.models` | Meshes of a quadcopter, a delta flying-wing drone and a ground air-defence platform, plus URDFs and a spec sheet. | `assets/models` |
| 3 | `swarmsim.physics` | PyBullet world: gravity, air density, wind with gusts and building shelter, rotor and wing aerodynamics, ballistic projectiles. | `swarmsim.physics.World` |
| 4 | `swarmsim.rl` | PettingZoo environment, PPO trainer, and a policy plugin that loads a trained model and flies it. | `python -m swarmsim.rl.train` / `.run` |

The drones are the learning agents. The defence platform is scripted.

## Install

```bash
pip install -r requirements.txt        # Python 3.10+
pytest tests                           # 20 tests, about 30 s
```

## Control app

```bash
python main.py
```

opens a page in your browser with a live 3-D view of the simulation and buttons
for everything else. It needs no extra packages and no internet connection.

| Part | What it does |
|------|--------------|
| 3-D view | The city as a coloured point cloud on black (every point takes the colour of the building, road or tree it sits on), with the drones, defence platforms, trails, goals, range rings and projectiles drawn solid. Motion is drawn on simulation time, so it stays smooth at any speed. |
| Moving around | `W A S D` move across the map, `Space` / `Ctrl` up and down, `Q E` or the arrow keys turn and tilt, the scroll wheel zooms. Dragging orbits or pans, depending on the tool picked in the toolbar (the right mouse button does the other one). |
| Follow | Click a drone (or press `1`-`9`; Shift+click adds more drones to the pick), then `F` keeps the camera on it. |
| Fixed-wings | Planes circle the map until they spot their target, then fly to it ("Target found" flashes). Select a plane to see its search loop (orange) and what its sensor covers (blue). |
| Pilot | `Enter` hands you the selected drone: `W S` forward and back, `A D` sideways, `Q E` turn, `Space` / `Ctrl` climb and descend. The policy keeps flying the others. `Esc` gives it back. |
| Keys menu | The see-through panel in the corner lists the keys for the current mode and lights up the ones you hold. `H` hides it. |
| Simulation | Choose what flies (any run in `runs/`, the scripted baseline, random actions), pause, change speed, start a new episode. |
| Scenario: World | Pick the city (the bundled ones, the training maps, or **New city**: style, street layout, 300 / 500 / 800 m and a seed, built in a few seconds into `worlds/`), and the number of quadcopters (0-24), fixed-wing drones (0-12) and air defences (0-6). **Apply** rebuilds the world; **Model defaults** goes back to what the model was trained with. The settings stay when you switch models. |
| Scenario: Positions | While placing, a see-through preview follows the mouse: green with a short description where a click would work, red with the reason where it would not. **Place** an air defence on open ground or a flat roof (pitched roofs, roof edges, trees and water are refused); the next defence still at a random spot follows straight on. Every drone shares **one goal** (planes reach it by flying over it): **Set goal** and click, or **Random** for a new random one. For starts, click a drone button to pick it (Shift+click picks several, **All** picks every drone), set the height above ground, then **Set start** and click the map. With one drone picked and **Then go on to the next drone** ticked, each click sets the next drone's start, so you can set them up one after another; with several picked, one click starts them all, spread 8 m apart. When the starts are done and no goal is set yet, it asks for the goal. Points over buildings are raised to clear them; planes start above the tallest roof. Esc or a right-click stops placing. **Random** goes back to random points. Set points are used on every restart until you change the city. |
| Training | Name a run, choose the number of steps and what to start from, press Start. Progress and a chart of the outcome rates update while it runs. "Stop and save" ends early and keeps the model. The finished run appears in the list of things to fly. |

The page is served by `swarmsim/app` on `127.0.0.1:8765` (`--port` changes it)
and is only reachable from your own computer. Training runs as a separate
process, so the view stays live. `python main.py demo`, `train` and `test` are
the same things without the page; `python main.py --help` lists them.

## Quick start

```bash
# 1. a city (2 s) plus 3-D preview images
python scripts/generate_city.py --out worlds/demo --preview

# 2 + 3 + 4. the vehicle models ship in assets/models; fly the rule-based baseline, record a GIF
python -m swarmsim.rl.run --config configs/quick_demo.yaml --policy scripted --episodes 3 --record flight.gif

# train, then run the trained model through the plugin
python -m swarmsim.rl.train --config configs/quick_demo.yaml --set rl.n_defence=0 --run-dir runs/stage1
python -m swarmsim.rl.train --config configs/quick_demo.yaml --init runs/stage1/model.zip --run-dir runs/stage2
python -m swarmsim.rl.run   --config configs/quick_demo.yaml --policy sb3:runs/stage2/model.zip --gui
```

To clean up, `python reset.py` lists and then deletes the runs you trained, `cache/` (maps built
from configs and downloaded map data) and the cities generated in the app (`worlds/custom_*`).
It keeps the bundled models and shipped maps. Use `--dry-run` to only list, `--runs` or `--maps`
for one kind, `--include-bundled` to also delete `runs/stage1_navigate` and `runs/stage2_defence`,
and `--yes` to skip the question.

One YAML file describes an experiment (`configs/default.yaml` lists the main
keys). Every script also takes `--set key=value`, for example
`--set terrain.roads.pattern=radial --set physics.wind.mean_speed=8`. A
misspelled key is an error, never a silent default.

## Module 1: terrain

Pipeline: road network, land pieces between roads, zoning, lot subdivision,
building footprints and heights, trees, heightmap, textures, meshes.

Output folder (a "scene"):

```
scene.json            which meshes and textures to load (read by the physics module)
world.json            roads, blocks, lots, buildings, trees as plain data
heightmap.npy         ground elevation grid
obstacle_height.npy   top of buildings/trees on a 2 m grid (spawn checks, wind shelter)
meshes/*.obj          ground, walls per facade style, roofs, trees
textures/*.png        ground map, one facade texture per style, roof and vegetation swatches
preview_map.png       top-down picture
```

Main hyperparameters (all in `swarmsim/terrain/params.py`):

| Group | Keys |
|-------|------|
| map | `seed`, `size_m`, `floor_height_m` |
| `roads` | `pattern` (grid / organic / radial), `block_size_m`, `block_jitter`, `rotation_deg`, `arterial_every`, widths, `removal_prob`, `diagonal_avenues` |
| `zoning` | `downtown_radius_frac`, `midrise_radius_frac`, `park_fraction`, `industrial_fraction` |
| `zones.<name>` | lot size, setback, coverage, height distribution (median, sigma, min, max), `tower_prob`, `l_shape_prob`, roof type, facade styles, yard trees |
| `trees` | on/off, street spacing, park density, height range, conifer share |
| `relief`, `water` | hill amplitude and wavelength, river on/off and width |
| `textures` | resolutions, lit-window share, lane markings |

Presets: `configs/city_suburb.yaml`, `configs/city_dense_core.yaml`.

### OpenStreetMap (`terrain.osm.mode`)

| Mode | Result |
|------|--------|
| `off` | Pure procedural with the built-in zone profiles. |
| `profile` | Downloads a real area, measures block size, street widths, grid angle, footprint sizes, building heights, park and water share, then generates a new city with those statistics. |
| `roads` | Real road network, everything else generated. |
| `full` | Real roads, footprints, parks, water and mapped trees. Missing heights are sampled. |

```bash
python scripts/generate_city.py --out worlds/delhi_like \
  --set osm.mode=profile --set "osm.bbox=[28.626,77.212,28.636,77.224]" --set satellite.enabled=true
```

Downloads are cached in `cache/`. To work offline, save an Overpass API JSON
export and point `osm.json_path` at it.

### Satellite imagery (`terrain.satellite`)

Default source is Sentinel-2 cloudless 2016 by EOX (CC BY 4.0), about 10 m per
pixel. At that resolution, and looking straight down, a satellite image cannot
texture a facade. It is used for what it does show:

* the colour palette of roofs, bare ground, vegetation and water,
* how green the area is, which scales tree density,
* optionally a faint underlay in the ground texture (`ground_underlay`).

Facades are generated procedurally (ten styles with windows, lit windows,
brick courses, curtain walls). To use real facade photos, replace
`textures/facade_<style>.png` in a scene folder; one tile is 4 bays by 4 floors.
Any local image can replace the download through `satellite.image_path`.

Credits to keep when publishing maps: "Map data (c) OpenStreetMap contributors,
ODbL 1.0" and "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH
(Contains modified Copernicus Sentinel data 2016 & 2017)". Both are written
into `scene.json` when used.

## Module 2: models

The quadcopter and the defence platform ship as ready meshes in `assets/models`
(`<name>.stl` is what the simulator loads, `<name>.step` the CAD source). The
fixed-wing is a simple delta flying wing (swept wing, tip fins, pusher
propeller) that `swarmsim/models/delta.py` writes as `fixedwing.stl` from
`fixedwing.span`, so its size always follows the config. `swarmsim/models/assets.py`
writes `<name>.urdf` (mass, inertia, simple collision shapes) and a shared
`specs.yaml` from the values in `specs.py` every time a world is created.

Performance features live in the same spec classes and are read by the physics
module:

* **Quadcopter**: mass, inertia, thrust-to-weight, motor lag, drag area, speed / climb / tilt limits.
* **Fixed-wing**: lift curve with stall, drag polar, max thrust, cruise / min / max speed, bank limit, and separate climb and descent limits (`max_climb` m/s and `max_climb_deg`, default 60°; `max_descent` m/s and `max_descent_deg`). Full climb is eased off as airspeed nears `min_speed`. In manual piloting, `Space` gives the steepest climb.
* **Defence platform**: `detection_range`, `range_m`, `min_range`, `fire_rate_hz`, `projectile_speed`, `projectile_drag`, `hit_radius`, `dispersion_deg`, `ammo`, `slew_rate_deg`, elevation limits, `reaction_time`, `lead_target`.

All numbers are generic round values for a simulation. They do not describe a
real product.

## Module 3: physics

`swarmsim.physics.World(scene_dir, models, physics_cfg, n_quads, n_fixedwing, defence_positions)`

* **Gravity, air density, time step**: `physics.gravity`, `air_density`, `physics_hz`.
* **Wind** (`physics.wind`): mean wind with a power-law height profile, gusts that are smooth in space and time, and reduced wind in the lee of buildings.
* **Quadcopter**: four rotors with saturation and lag, a geometric attitude controller, aerodynamic drag. Commanded with a velocity and a yaw rate.
* **Fixed-wing**: lift, drag and side force from the airflow, propeller thrust, stall. Commanded with turn rate, climb rate and airspeed. Rotation is simplified: an autopilot commands body rates rather than modelling control surfaces.
* **Defence platform**: picks the nearest drone it can see, slews at a limited rate, fires at a fixed rate when aligned and in range. Projectiles are point masses with gravity, drag and wind.
* **Collisions**: drones against terrain, buildings, trees and each other.

## Module 4: RL

`SwarmEnv` is a PettingZoo `ParallelEnv`: one agent per drone, identical
observation and action spaces, so one policy can be shared by the swarm.

**Actions** (`rl.action_mode`): `discrete` gives named primitives, `continuous`
gives a 4-vector (forward, left, up, yaw). The hyperparameters:

| Key | Meaning |
|-----|---------|
| `discrete_actions` | which primitives exist: `hover, forward, back, left, right, up, down, yaw_left, yaw_right` |
| `move_speed`, `climb_speed`, `yaw_rate` | how fast each primitive moves a quadcopter |
| `fw_turn_rate`, `fw_speed_step` | what left/right and forward/back mean for a fixed-wing |
| `policy_hz` | decisions per second |

**Observations**: own velocity and attitude, height above ground, direction and
distance to the goal, range-finder rays, nearest neighbours, nearest defence
platform, nearest projectile, vehicle type. Each block can be sized or switched
off (`n_rays`, `ray_range`, `k_neighbors`, `observe_defence`, `observe_projectiles`).

**Tasks**: `navigate` (reach a goal point) and `coverage` (visit every map cell).

**Fixed-wing drones** (`rl.fw_mode`): with `search` (the default) the policy
flies only the quadcopters, and the planes are flown by `swarmsim/rl/wingpilot.py`.
The map is split into rounded-square rings of equal area, one per plane; each
plane loops round its ring on a slow spiral that covers it without gaps. A
plane sees any target within `rl.fw_detect_range` metres (default 60), and a
sighting is shared: the plane whose target it is flies straight to it. Planes
turn back before the map edge, and when two converge at the same height the
higher-numbered one climbs over. `rl.fw_altitude` sets their height above the
tallest roof (default 3-8 m). With `policy` the trained model flies the planes
like quadcopters. Training refuses `search` with planes in the swarm, since the
policy would not be flying them.

In a test with six planes and no defences, they reached 99% of targets on the
300 m map (median 5 s to spot, 14 s to arrive) and 75% on an 800 m map within
the 90 s limit. On big maps raise `max_episode_s` or `fw_detect_range`.
Reward weights are the `r_*` keys.

**Policy plugin** (`swarmsim/rl/plugin.py`): anything that maps the observation
matrix to actions.

```
sb3:runs/x/model.zip      Stable-Baselines3 model
torchscript:policy.pt     exported network
onnx:policy.onnx          exported network, runs with onnxruntime only
scripted                  rule-based baseline
random
mypkg.module:MyPolicy     your own PolicyPlugin subclass
```

```python
from swarmsim.rl.plugin import PolicyPlugin, register

@register
class MyPolicy(PolicyPlugin):
    name = "mine"
    def act(self, obs):                 # (n_drones, obs_dim) -> (n_drones,)
        return [self.info["action_names"].index("forward")] * len(obs)
```

A model must be run with the config it was trained with (same observation and
action layout). The trainer saves that config next to the model.

## What was and was not verified

Verified here: the test suite, the three road patterns, quadcopter
and fixed-wing flight, wind, the defence platform, PettingZoo API conformance,
train / save / load / run through all three model formats.

Not verified here: live downloads from the Overpass API and the EOX tile
server. The build machine had no access to either host. The OSM parser,
profile fitting and all three OSM modes are tested with data written in the
Overpass JSON format, and the satellite analysis with local images. Expect to
check the tile URL template (`satellite.tile_url`) on first use.

## Test training run

`configs/quick_demo.yaml` (300 m suburb, 8 quadcopters), PPO on 2 CPU cores at
roughly 1,000 agent-steps per second. Evaluation: 6 episodes on unseen seeds,
48 drone-episodes per row (the few not listed left the map).

| Scenario | Policy | Reached goal | Hit | Crashed | Timed out |
|----------|--------|--------------|-----|---------|-----------|
| no defence | PPO stage 1 (600k steps, about 9 min) | 71% | - | 13% | 15% |
| no defence | scripted baseline | 83% | - | 2% | 15% |
| no defence | random | 0% | - | 6% | 94% |
| defence on | PPO stage 2 (+300k steps) | 40% | 44% | 4% | 8% |
| defence on | PPO stage 1, never saw the platform | 35% | 50% | 4% | 6% |
| defence on | scripted baseline | 46% | 48% | 0% | 6% |
| defence on | random | 0% | 31% | 0% | 69% |

Reading: navigation is learned from scratch and gets close to the hand-written
baseline. The three "defence on" rows for real policies are within noise of
each other at this sample size, so 300k steps were not enough for the policy to
learn anything useful about the platform. It covers most of a 300 m map
(120 m engagement range). Train longer, use a larger map, or lower `range_m`,
`fire_rate_hz` or raise `dispersion_deg` to make the task learnable sooner.

Both models are in `runs/`. `progress.csv` for stage 1 covers its last 450k
steps. `docs/flight_trained.mp4` shows the stage-2 model; recordings add
coloured markers because the drones are too small to see from the overview
camera (blue quadcopter, orange fixed-wing, green goal, red platform, yellow
projectile).

## Known limits

* Rendering without `--gui` uses PyBullet's software renderer: fine for previews and short GIFs, slow for video.
* Building facades are procedural. See the satellite section for the reason and for how to swap in photos.
* The fixed-wing has no control-surface model.
* PPO is the only trainer wired up. The environment is standard PettingZoo, so other libraries plug in directly.
