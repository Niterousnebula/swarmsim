#!/usr/bin/env python3
"""Module 1 on its own: generate a city scene (meshes, textures, map data).

    python scripts/generate_city.py --out worlds/demo
    python scripts/generate_city.py --config configs/city_suburb.yaml --out worlds/suburb --preview
    python scripts/generate_city.py --set seed=3 --set roads.pattern=radial --set size_m=600 --out worlds/radial

    # real-world reference (needs internet): measure an OSM area and build a new city with its proportions
    python scripts/generate_city.py --set osm.mode=profile --set "osm.bbox=[28.626,77.212,28.636,77.224]" \\
        --set satellite.enabled=true --out worlds/delhi_like

--set takes any key of swarmsim/terrain/params.py (dots for nesting).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmsim.config import load_config                      # noqa: E402
from swarmsim.terrain import generate_and_export             # noqa: E402
from swarmsim.utils.cfg import apply_overrides, parse_set_flags   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="experiment YAML (its 'terrain' section is used)")
    ap.add_argument("--out", default="worlds/demo")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--preview", action="store_true", help="also render 3-D preview images (slow software renderer)")
    args = ap.parse_args()

    params = load_config(args.config).terrain
    apply_overrides(params, parse_set_flags(args.set))
    world = generate_and_export(params, args.out)
    for line in world.meta.get("attribution", []):
        print("credit:", line)
    if args.preview:
        from PIL import Image
        from swarmsim.physics import World as Sim
        sim = Sim(args.out, n_quads=0)
        h = world.half
        cx, cy = world.centre
        Image.fromarray(sim.render(1280, 800)).save(Path(args.out) / "preview_3d.png")
        Image.fromarray(sim.render(1280, 800, eye=[cx + 0.45 * h, cy - 0.5 * h, world.height_at(cx, cy) + 55],
                                   target=[cx, cy, world.height_at(cx, cy) + 25])).save(Path(args.out) / "preview_close.png")
        sim.close()
        print("wrote preview_3d.png, preview_close.png")


if __name__ == "__main__":
    main()
