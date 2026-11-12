"""Hyperparameters of the procedural city generator.

Units are metres unless the name says otherwise. The map is a square centred
on the origin: x points east, y points north, z points up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RoadParams:
    pattern: str = "grid"            # grid | organic | radial
    block_size_m: float = 95.0       # mean spacing between parallel roads
    block_jitter: float = 0.18       # 0 = perfectly regular, 0.4 = very uneven
    rotation_deg: float = 12.0       # rotation of the whole street grid
    arterial_every: int = 3          # every n-th road is a wide arterial
    arterial_width_m: float = 18.0
    street_width_m: float = 9.0
    removal_prob: float = 0.10       # chance a minor street segment is missing (merges blocks)
    diagonal_avenues: int = 1        # straight avenues cutting across the pattern
    avenue_width_m: float = 20.0
    organic_warp_m: float = 38.0     # how far 'organic' bends the grid
    radial_spokes: int = 12          # 'radial' pattern only
    sidewalk_width_m: float = 2.5


@dataclass
class ZoneProfile:
    """How lots are cut and built inside one land-use zone."""
    lot_min_m: float = 22.0
    lot_max_m: float = 34.0
    setback_m: float = 2.0           # gap between lot edge and building
    coverage: float = 0.85           # share of the remaining lot the footprint fills
    build_prob: float = 0.95         # chance a street-facing lot gets a building
    interior_build_prob: float = 0.35  # same for lots with no street frontage
    height_median_m: float = 18.0    # heights are log-normal around the median
    height_sigma: float = 0.35
    height_min_m: float = 6.0
    height_max_m: float = 60.0
    centre_boost: float = 0.0        # extra height towards the city centre (0..2)
    tower_prob: float = 0.0          # chance of podium + set-back tower
    l_shape_prob: float = 0.0
    roof: str = "flat"               # flat | gable | mixed
    styles: List[str] = field(default_factory=lambda: ["office_concrete"])
    ground: str = "paved"            # paved | grass | dirt
    yard_tree_prob: float = 0.0


def default_zones() -> Dict[str, ZoneProfile]:
    return {
        "downtown": ZoneProfile(
            lot_min_m=30, lot_max_m=46, setback_m=1.5, coverage=0.92,
            height_median_m=52, height_sigma=0.45, height_min_m=18, height_max_m=170,
            centre_boost=1.0, tower_prob=0.45, l_shape_prob=0.1,
            styles=["glass_blue", "glass_dark", "office_concrete", "office_white"],
            ground="paved"),
        "midrise": ZoneProfile(
            lot_min_m=22, lot_max_m=36, setback_m=2.0, coverage=0.86,
            height_median_m=19, height_sigma=0.32, height_min_m=9, height_max_m=45,
            l_shape_prob=0.25,
            styles=["brick_red", "brick_brown", "stucco_cream", "office_concrete", "stucco_pastel"],
            ground="paved", yard_tree_prob=0.1),
        "residential": ZoneProfile(
            lot_min_m=16, lot_max_m=24, setback_m=3.5, coverage=0.55,
            build_prob=0.92, interior_build_prob=0.15,
            height_median_m=6.5, height_sigma=0.22, height_min_m=3.5, height_max_m=12,
            roof="mixed", styles=["house_siding", "stucco_cream", "brick_red", "stucco_pastel"],
            ground="grass", yard_tree_prob=0.55),
        "industrial": ZoneProfile(
            lot_min_m=40, lot_max_m=70, setback_m=5.0, coverage=0.72,
            build_prob=0.85, interior_build_prob=0.5,
            height_median_m=9, height_sigma=0.25, height_min_m=6, height_max_m=18,
            styles=["industrial_metal", "office_concrete"],
            ground="dirt", yard_tree_prob=0.05),
        "park": ZoneProfile(build_prob=0.0, interior_build_prob=0.0, ground="grass"),
    }


@dataclass
class ZoningParams:
    downtown_radius_frac: float = 0.26   # of half the map size
    midrise_radius_frac: float = 0.50
    park_fraction: float = 0.08          # share of blocks turned into parks
    industrial_fraction: float = 0.12    # share of outer blocks that are industrial
    centre_offset_frac: float = 0.12     # how far the centre may sit from the map middle


@dataclass
class TreeParams:
    enabled: bool = True
    street_spacing_m: float = 16.0
    street_tree_prob: float = 0.55
    park_density_per_ha: float = 85.0
    height_min_m: float = 5.0
    height_max_m: float = 15.0
    canopy_ratio: float = 0.34           # canopy radius / tree height
    conifer_fraction: float = 0.2
    max_trees: int = 4000


@dataclass
class ReliefParams:
    amplitude_m: float = 7.0             # 0 = flat
    wavelength_m: float = 380.0
    octaves: int = 3
    grid_n: int = 97                     # ground mesh / heightmap resolution


@dataclass
class WaterParams:
    river: bool = True
    width_m: float = 30.0
    meander: float = 0.16                # sideways swing as a fraction of map size


@dataclass
class TextureParams:
    ground_px: int = 2048
    facade_px: int = 256
    window_lit_fraction: float = 0.18
    lane_markings: bool = True


@dataclass
class OSMParams:
    """OpenStreetMap as a reference dataset.

    mode:
      off      - pure procedural, built-in zone profiles
      profile  - read a real area, measure its statistics (block size, heights,
                 footprints, road widths, tree cover) and generate a *new* city
                 that follows them
      roads    - keep the real road network, generate everything else
      full     - real roads, building footprints, parks and water; missing
                 heights and trees are filled in procedurally
    """
    mode: str = "off"
    bbox: List[float] = field(default_factory=lambda: [28.6260, 77.2120, 28.6360, 77.2240])  # south, west, north, east
    json_path: str = ""                  # use a saved Overpass JSON instead of downloading
    cache_dir: str = "cache/osm"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    timeout_s: int = 90


@dataclass
class SatelliteParams:
    """Open satellite imagery used as a colour / vegetation reference."""
    enabled: bool = False
    image_path: str = ""                 # local image covering the bbox (skips download)
    tile_url: str = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2016_3857/default/g/{z}/{y}/{x}.jpg"
    attribution: str = ("Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH "
                        "(Contains modified Copernicus Sentinel data 2016 & 2017)")
    zoom: int = 14
    bbox: List[float] = field(default_factory=list)   # defaults to osm.bbox
    cache_dir: str = "cache/sat"
    palette_k: int = 8
    ground_underlay: float = 0.0         # 0..1 blend of the satellite image into the ground texture
    drive_tree_density: bool = True      # scale tree counts by measured vegetation cover


@dataclass
class TerrainParams:
    seed: int = 7
    size_m: float = 800.0
    floor_height_m: float = 3.2
    roads: RoadParams = field(default_factory=RoadParams)
    zoning: ZoningParams = field(default_factory=ZoningParams)
    zones: Dict[str, ZoneProfile] = field(default_factory=default_zones)
    trees: TreeParams = field(default_factory=TreeParams)
    relief: ReliefParams = field(default_factory=ReliefParams)
    water: WaterParams = field(default_factory=WaterParams)
    textures: TextureParams = field(default_factory=TextureParams)
    osm: OSMParams = field(default_factory=OSMParams)
    satellite: SatelliteParams = field(default_factory=SatelliteParams)
