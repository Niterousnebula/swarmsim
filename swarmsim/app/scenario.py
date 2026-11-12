"""Scenario settings for the app: which city, how many drones and platforms,
new cities from a few presets, and the geometry of placing things by clicking."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

CITY_ROOTS = ("worlds", "cache")
LIMITS = {"n_quads": (0, 24), "n_fixedwing": (0, 12), "n_defence": (0, 6)}
MAX_DRONES = 24
HEIGHT_RANGE = (5.0, 120.0)            # m above ground for clicked start and goal points
EDGE_M = 6.0                           # keep clicked points this far inside the map edge
SPREAD_M = 8.0                         # spacing of start points when one click places every drone

# Presets for new cities. Road layout, size and seed are chosen separately.
STYLES = {
    "suburb": {"label": "Suburb", "terrain": {
        "roads": {"block_size_m": 110, "arterial_every": 4, "removal_prob": 0.2, "diagonal_avenues": 0},
        "zoning": {"downtown_radius_frac": 0.0, "midrise_radius_frac": 0.15, "park_fraction": 0.12,
                   "industrial_fraction": 0.05},
        "trees": {"street_tree_prob": 0.8, "park_density_per_ha": 120},
        "relief": {"amplitude_m": 8}}},
    "mixed": {"label": "Mixed town", "terrain": {
        "relief": {"amplitude_m": 5}}},
    "dense": {"label": "Dense high-rise core", "terrain": {
        "roads": {"block_size_m": 85, "diagonal_avenues": 0},
        "zoning": {"downtown_radius_frac": 0.42, "midrise_radius_frac": 0.75, "park_fraction": 0.05},
        "zones": {"downtown": {"height_median_m": 70, "height_max_m": 220, "tower_prob": 0.6},
                  "midrise": {"height_median_m": 26}},
        "relief": {"amplitude_m": 3}}},
}
PATTERNS = ("grid", "organic", "radial")
SIZES = (300, 500, 800)


# ------------------------------------------------------------------- cities
def _label(d: Path) -> str:
    name = d.name
    if d.parent.name == "cache":
        return f"Training map {name.removeprefix('city_')[:4]}"
    if name.startswith("custom_"):
        style, pattern, size, seed = (name.split("_") + ["", "", "", ""])[1:5]
        return f"{STYLES.get(style, {}).get('label', style)}, {pattern} #{seed}"
    return name.removeprefix("city_").replace("_", " ").capitalize()


def list_cities() -> list[dict]:
    out = []
    for root in CITY_ROOTS:
        for scene in sorted(Path(root).glob("*/scene.json")):
            d = scene.parent
            try:
                m = json.loads(scene.read_text())
            except (OSError, ValueError):
                continue
            out.append({"path": d.as_posix(), "label": _label(d), "size": float(m.get("size", 0)),
                        "buildings": int(m.get("counts", {}).get("buildings", 0))})
    return out


def check_city(path: str) -> str:
    known = {c["path"] for c in list_cities()}
    if path not in known:
        raise ValueError(f"Unknown city '{path}'.")
    return path


def generate_city(style: str, pattern: str, size: int, seed: int) -> Path:
    """Build a new city into worlds/ (or reuse it if these settings were built before)."""
    from ..config import load_config
    from ..terrain import generate_and_export
    if style not in STYLES:
        raise ValueError(f"Unknown city style '{style}'.")
    if pattern not in PATTERNS:
        raise ValueError(f"Road layout must be one of {', '.join(PATTERNS)}.")
    size, seed = int(size), int(seed)
    if size not in SIZES:
        raise ValueError(f"Size must be one of {', '.join(map(str, SIZES))} m.")
    if not 0 <= seed <= 99_999:
        raise ValueError("Seed must be between 0 and 99999.")
    terrain = copy.deepcopy(STYLES[style]["terrain"])
    terrain.update(seed=seed, size_m=size)
    terrain.setdefault("roads", {})["pattern"] = pattern
    if size <= 400:
        terrain["water"] = {"river": False}          # a river would take up most of a small map
    out = Path("worlds") / f"custom_{style}_{pattern}_{size}_{seed}"
    if not (out / "scene.json").exists():
        generate_and_export(load_config(None, overrides={"terrain": terrain}).terrain, str(out), verbose=False)
    return out


def check_counts(n_quads, n_fixedwing, n_defence) -> dict:
    vals = {"n_quads": n_quads, "n_fixedwing": n_fixedwing, "n_defence": n_defence}
    out = {}
    for k, v in vals.items():
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{k} must be a whole number.") from None
        lo, hi = LIMITS[k]
        if not lo <= v <= hi:
            names = {"n_quads": "Quadcopters", "n_fixedwing": "Fixed-wing drones", "n_defence": "Air defences"}
            raise ValueError(f"{names[k]} must be between {lo} and {hi}.")
        out[k] = v
    if not 1 <= out["n_quads"] + out["n_fixedwing"] <= MAX_DRONES:
        raise ValueError(f"Fly between 1 and {MAX_DRONES} drones in total.")
    return out




# ---------------------------------------------------------------- placement
# Clicks are resolved here, on the map data (no PyBullet), so the live preview
# while the mouse moves and the final placement use exactly the same rules.

def _index(city):
    """Building footprints and tree positions, built once per map."""
    idx = getattr(city, "_app_index", None)
    if idx is None:
        from shapely import STRtree
        from shapely.geometry import Polygon
        polys = [Polygon(b.footprint, b.holes or None) for b in city.buildings]
        trees = (np.array([[t.x, t.y, t.canopy_r] for t in city.trees], float) if city.trees
                 else np.zeros((0, 3)))
        idx = {"polys": polys, "tree": STRtree(polys) if polys else None, "trees": trees,
               "top": float(np.max(city.obstacle_height)) if city.obstacle_height is not None
               else float(np.max(city.heightmap))}
        city._app_index = idx
    return idx


def in_water(city, x: float, y: float) -> bool:
    polys = getattr(city, "_app_water", None)
    if polys is None:
        from shapely.geometry import Polygon
        polys = [Polygon(p).buffer(2.0) for p in city.water if len(p) >= 3]
        city._app_water = polys
    from shapely.geometry import Point
    pt = Point(x, y)
    return any(p.contains(pt) for p in polys)


def building_at(city, x: float, y: float, near: float = 0.0):
    """The tallest building whose footprint contains (x, y), or lies within ``near`` m of it."""
    from shapely.geometry import Point
    idx = _index(city)
    if idx["tree"] is None:
        return None, None
    pt = Point(x, y)
    hits = idx["tree"].query(pt.buffer(near) if near else pt, predicate="intersects")
    best = max(hits, key=lambda k: city.buildings[k].top_z, default=None)
    return (None, None) if best is None else (city.buildings[best], idx["polys"][best])


def cast(city, eye, direction) -> np.ndarray | None:
    """First point where a ray from the camera meets the ground, a roof, a wall or a tree."""
    eye = np.asarray(eye, float).reshape(3)
    d = np.asarray(direction, float).reshape(3)
    d = d / max(np.linalg.norm(d), 1e-9)
    half, top = city.half, _index(city)["top"] + 1.0
    t0, t1 = 0.0, 1e6
    for a in (0, 1):                                    # where the ray is above the map square
        if abs(d[a]) < 1e-9:
            if abs(eye[a]) > half:
                return None
            continue
        ta, tb = sorted(((-half - eye[a]) / d[a], (half - eye[a]) / d[a]))
        t0, t1 = max(t0, ta), min(t1, tb)
    if d[2] < 0:
        t0 = max(t0, (eye[2] - top) / -d[2])            # skip the empty sky above everything
    elif eye[2] > top:
        return None
    if t1 <= t0:
        return None
    t = np.arange(t0, t1, 0.5)
    p = eye + t[:, None] * d
    under = p[:, 2] <= city.obstacle_top_at(p[:, 0], p[:, 1])
    if not under.any():
        return None
    k = int(np.argmax(under))
    if k == 0:
        return p[0]
    lo, hi = t[k - 1], t[k]
    for _ in range(12):                                 # narrow down the crossing
        mid = 0.5 * (lo + hi)
        q = eye + mid * d
        if q[2] <= city.obstacle_top_at(q[0], q[1]):
            hi = mid
        else:
            lo = mid
    return eye + hi * d


def surface(city, x: float, y: float):
    """What is at (x, y): 'water', 'ground', 'roof' (with its building and footprint), or 'tree'."""
    ground, top = float(city.ground_z(x, y)), float(city.obstacle_z(x, y))
    if top - ground < 0.3:
        return ("water" if in_water(city, x, y) else "ground"), None, None
    b, poly = building_at(city, x, y)
    if b is not None:
        return "roof", b, poly
    trees = _index(city)["trees"]
    if len(trees) and np.any(np.hypot(trees[:, 0] - x, trees[:, 1] - y) < trees[:, 2] + 1.0):
        return "tree", None, None
    b, poly = building_at(city, x, y, near=2.5)        # the side of a building
    return ("roof", b, poly) if b is not None else ("tree", None, None)


def defence_spot(core, x: float, y: float) -> tuple[list[float], str]:
    """Where a platform clicked at (x, y) would stand and a short description, or ValueError."""
    city = core.city
    if max(abs(x), abs(y)) > city.half - EDGE_M:
        raise ValueError("Pick a point inside the map.")
    kind, b, poly = surface(city, x, y)
    if kind == "water":
        raise ValueError("Not in the water. Pick land or a flat roof.")
    if kind == "tree":
        raise ValueError("Not on a tree. Pick open ground or a flat roof.")
    if kind == "ground":
        if not core.open_ground(x, y):
            raise ValueError("Too close to a building or tree.")
        return [round(x, 2), round(y, 2), round(float(city.ground_z(x, y)), 2)], "Ground"
    from shapely.geometry import Point
    if b.roof != "flat":
        raise ValueError("That roof is pitched. Pick a flat roof.")
    if not poly.buffer(-2.5).contains(Point(x, y)):
        raise ValueError("Too close to the roof edge.")
    xs = np.array([x, x + 2.5, x - 2.5, x, x]); ys = np.array([y, y, y, y + 2.5, y - 2.5])
    if np.any(city.obstacle_z(xs, ys) > b.top_z + 0.5):
        raise ValueError("Something taller is in the way on this roof.")
    up = round(b.top_z - float(city.ground_z(x, y)))
    return [round(x, 2), round(y, 2), round(b.top_z, 2)], f"Flat roof, {up} m up"


def flight_point(core, x: float, y: float, height: float, fixedwing: bool) -> tuple[list[float], str]:
    """A point to start from or fly to above (x, y): ``height`` m above the ground, raised
    above any building or tree there. Fixed-wing drones start above the tallest roof (pass
    ``fixedwing=False`` for goals: every drone shares one goal, and planes fly over it)."""
    city = core.city
    half = city.half - EDGE_M
    if abs(x) > half or abs(y) > half:
        raise ValueError("Pick a point inside the map.")
    ground = float(city.ground_z(x, y))
    z, note = ground + height, ""
    lowest = city.max_height + float(min(core.rl.fw_altitude))
    if fixedwing and z < lowest:
        z, note = lowest, "fixed-wing drones fly above the tallest roof"
    if not city.is_free(x, y, z, clearance=3.0):
        offs = (-3.0, 0.0, 3.0)
        top = max(float(city.obstacle_z(x + dx, y + dy)) for dx in offs for dy in offs)
        z, note = top + 5.0, f"raised to {round(top + 5.0 - ground)} m to clear what is below"
    if z - ground > core.rl.ceiling_agl - 5.0 and not (fixedwing and z <= core.wing_top - 5.0):
        raise ValueError("That is above the flight ceiling. Pick a lower spot.")
    return [round(x, 2), round(y, 2), round(z, 2)], note


def spread(core, x: float, y: float, height: float, drones: list[int]) -> tuple[dict, str]:
    """Start points for several drones around (x, y), SPREAD_M apart, nearest first."""
    k = int(np.ceil(np.sqrt(len(drones)))) + 3
    grid = [(a, b) for a in range(-k, k + 1) for b in range(-k, k + 1)]
    grid.sort(key=lambda g: (g[0] ** 2 + g[1] ** 2, g))
    out, notes = {}, set()
    for a, b in grid:
        if len(out) == len(drones):
            break
        i = drones[len(out)]
        try:
            p, note = flight_point(core, x + a * SPREAD_M, y + b * SPREAD_M, height,
                                   core.world.kind(i) == "fixedwing")
        except ValueError:
            continue
        out[i] = p
        if note:
            notes.add(note)
    if len(out) < len(drones):
        raise ValueError("Not enough room there for every drone. Pick a more open spot.")
    return out, "; ".join(sorted(notes))


def drones_of(core, index) -> list[int]:
    """"all", one drone number, or a list of them."""
    if index == "all":
        return list(range(core.n))
    picked = index if isinstance(index, list) else [index]
    if picked and all(isinstance(i, int) and 0 <= i < core.n for i in picked):
        return sorted(set(picked))
    raise ValueError("Pick a drone first.")


def resolve(core, cmd: dict) -> dict:
    """Turn a click (a camera ray) into what would be placed, or raise ValueError with the reason.

    Returns {"what", "index", "points": {i: [x, y, z]}, "grounds": {i: ground z}, "label", "note", "ground"}."""
    what, index = cmd.get("what"), cmd.get("index")
    hit = cast(core.city, cmd.get("eye"), cmd.get("dir"))
    if hit is None:
        raise ValueError("Point at the city.")
    x, y = float(hit[0]), float(hit[1])
    out = {"what": what, "index": index, "ground": round(float(core.city.ground_z(x, y)), 2), "note": ""}
    if what == "defence":
        if not (isinstance(index, int) and 0 <= index < len(core.world.defences)):
            raise ValueError("There is no such air defence.")
        p, out["label"] = defence_spot(core, x, y)
        out["points"], out["grounds"] = {index: p}, {index: out["ground"]}
        return out
    if what not in ("start", "goal"):
        raise ValueError("Choose what to place.")
    height = float(np.clip(float(cmd.get("height", 25.0)), *HEIGHT_RANGE))
    drones = drones_of(core, index)
    if what == "start" and len(drones) > 1:
        out["points"], out["note"] = spread(core, x, y, height, drones)
    else:
        points, notes = {}, set()
        for i in drones:
            lift = what == "start" and core.world.kind(i) == "fixedwing"
            points[i], note = flight_point(core, x, y, height, lift)
            notes.add(note)
        out["points"], out["note"] = points, "; ".join(sorted(n for n in notes if n))
    out["grounds"] = {i: round(float(core.city.ground_z(q[0], q[1])), 2) for i, q in out["points"].items()}
    if what == "goal" and core.wingpilot:                # can planes get down to it?
        q = next(iter(out["points"].values()))
        lowest = core.wingpilot.floor(np.asarray(q, float))
        if lowest > q[2] + 3.0:
            note = f"planes can't get that low here (buildings around it): they will circle {round(lowest - out['ground'])} m up"
            out["note"] = "; ".join(n for n in (out["note"], note) if n)
    first = next(iter(out["points"].values()))
    out["label"] = f"{round(first[2] - out['ground'])} m above ground"
    return out
