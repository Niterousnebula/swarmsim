"""Write the URDFs and the specs file that go with the prebuilt meshes in ``assets/models``.

The quadcopter and defence meshes (``<name>.stl``) ship with the project. The
fixed-wing is a simple delta flying wing generated from its span (``delta.py``).
The URDFs carry mass, inertia and collision shapes, which come from the config,
so they are rewritten whenever a world is created.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from . import delta
from .specs import ModelsConfig

MODEL_NAMES = ("quadcopter", "fixedwing", "defence")

URDF = """<?xml version="1.0"?>
<robot name="{name}">
  <material name="body"><color rgba="{r} {g} {b} 1"/></material>
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry><mesh filename="{name}.stl" scale="0.001 0.001 0.001"/></geometry>
      <material name="body"/>
    </visual>
{collisions}  </link>
</robot>
"""


def _col(shape: str, xyz=(0, 0, 0)) -> str:
    return (f'    <collision>\n      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}"/>\n'
            f'      <geometry>{shape}</geometry>\n    </collision>\n')


def write_urdfs(cfg: ModelsConfig, out: Path) -> dict:
    q, f, d = cfg.quad, cfg.fixedwing, cfg.defence
    w = delta.planform(f.span)
    reach = q.arm_length + q.prop_radius
    (out / "quadcopter.urdf").write_text(URDF.format(
        name="quadcopter", r=0.16, g=0.17, b=0.2, mass=q.mass,
        ixx=q.inertia[0], iyy=q.inertia[1], izz=q.inertia[2],
        collisions=_col(f'<cylinder radius="{reach:.3f}" length="0.10"/>')))
    (out / "fixedwing.urdf").write_text(URDF.format(
        name="fixedwing", r=0.85, g=0.86, b=0.88, mass=f.mass,
        ixx=f.inertia[0], iyy=f.inertia[1], izz=f.inertia[2],
        collisions=_col(f'<box size="{w["root_chord"]:.3f} {0.3 * f.span:.3f} {w["thickness"]:.3f}"/>',
                        (round(w["root_centre_x"], 3), 0, 0))                     # centre of the delta
        + _col(f'<box size="{2 * w["tip_chord"]:.3f} {f.span:.3f} 0.03"/>',
               (round(w["wing_centre_x"], 3), 0, 0))))                              # swept outer wing
    bx, by, bz = d.base_size
    (out / "defence.urdf").write_text(URDF.format(
        name="defence", r=0.33, g=0.38, b=0.30, mass=0, ixx=0, iyy=0, izz=0,
        collisions=_col(f'<box size="{bx} {by} {bz}"/>', (0, 0, bz / 2))
        + _col(f'<cylinder radius="{d.turret_radius}" length="{max(d.muzzle_height - bz, 0.4):.3f}"/>',
               (0, 0, bz + max(d.muzzle_height - bz, 0.4) / 2))))
    return {n: f"{n}.urdf" for n in MODEL_NAMES}


def ensure_models(cfg: ModelsConfig) -> Path:
    """Return the assets folder with URDFs and specs.yaml in sync with ``cfg``."""
    out = Path(cfg.assets_dir)
    if out.is_dir():
        delta.write_stl(cfg.fixedwing.span, out / "fixedwing.stl")
    missing = [f"{n}.stl" for n in MODEL_NAMES if not (out / f"{n}.stl").exists()]
    if missing:
        raise FileNotFoundError(f"Model mesh(es) {missing} not found in '{out}'. They ship with the project "
                                f"in assets/models; run from the project folder or set models.assets_dir.")
    write_urdfs(cfg, out)
    specs = {"quad": asdict(cfg.quad), "fixedwing": asdict(cfg.fixedwing), "defence": asdict(cfg.defence)}
    (out / "specs.yaml").write_text(yaml.safe_dump(
        {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
         for k, v in specs.items()}, sort_keys=False))
    return out
