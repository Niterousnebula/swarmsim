import numpy as np
import pytest

from swarmsim.models import ModelsConfig
from swarmsim.physics import PhysicsConfig, WindConfig, WindField, World
from swarmsim.physics.defence import Projectiles
from swarmsim.terrain import TerrainParams, World as CityMap, generate_and_export

CALM = WindConfig(mean_speed=0.0, gust_intensity=0.0)


@pytest.fixture(scope="module")
def city(tmp_path_factory):
    d = tmp_path_factory.mktemp("city")
    p = TerrainParams(seed=4, size_m=300)
    p.textures.ground_px = 512
    generate_and_export(p, str(d), verbose=False)
    return d


def run(w, seconds):
    for _ in range(int(seconds * w.cfg.physics_hz)):
        w.step()


def test_quad_hover_and_velocity_tracking(city):
    w = World(city, n_quads=2, cfg=PhysicsConfig(wind=CALM))
    z = w.city.max_height + 20
    w.spawn(0, [0, 0, z]); w.spawn(1, [15, 0, z])
    run(w, 2.0)
    assert np.linalg.norm(w.pos[0] - [0, 0, z]) < 0.05
    w.command_velocity(0, [5, 0, 0]); w.command_velocity(1, [0, -3, 2], yaw_rate=0.8)
    run(w, 4.0)
    assert np.allclose(w.vel[0], [5, 0, 0], atol=0.15)
    assert np.allclose(w.vel[1], [0, -3, 2], atol=0.25)
    w.close()


def test_gravity_and_wind_are_real(city):
    # stronger gravity -> motors must work harder to hover
    thrust = {}
    for g in (9.81, 3.71):
        w = World(city, n_quads=1, cfg=PhysicsConfig(gravity=g, wind=CALM))
        w.spawn(0, [0, 0, w.city.max_height + 20]); run(w, 2.0)
        thrust[g] = w.quads.motor[0].sum()
        assert abs(w.vel[0, 2]) < 0.05
        w.close()
    assert thrust[9.81] == pytest.approx(9.81, rel=0.03) and thrust[3.71] == pytest.approx(3.71, rel=0.03)
    # a hovering quad must lean into a steady wind
    w = World(city, n_quads=1, cfg=PhysicsConfig(wind=WindConfig(mean_speed=6.0, gust_intensity=0.0)))
    w.spawn(0, [0, 0, w.city.max_height + 30]); run(w, 4.0)
    tilt = np.degrees(np.arccos(w.R[0, 2, 2]))
    assert tilt > 5.0 and np.linalg.norm(w.vel[0]) < 0.3
    w.close()


def test_wind_profile_and_shelter(city):
    from shapely.geometry import Polygon
    c = CityMap.load(city)
    wf = WindField(WindConfig(mean_speed=5.0, direction_deg=0.0, gust_intensity=0.0, building_shelter=0.6), c)
    b = max((q for q in c.buildings if not q.stacked), key=lambda q: q.height)
    minx, miny, maxx, maxy = Polygon(b.footprint).bounds
    cy = Polygon(b.footprint).centroid.y
    high = wf.at(np.array([[maxx, cy, b.top_z + 60.0]]))[0]
    lee = wf.at(np.array([[maxx + 4.0, cy, b.ground_z + 3.0]]))[0]   # just downwind of the wall
    assert high[0] > 5.0                       # faster aloft
    assert lee[0] < 0.5 * 5.0                  # sheltered behind the building
    gusty = WindField(WindConfig(mean_speed=0.0, gust_intensity=2.0), None, seed=1)
    samples = []
    for _ in range(4000):
        gusty.step(0.05); samples.append(gusty.at(np.zeros((1, 3)))[0])
    assert 0.8 < np.std(np.array(samples)[:, :2]) < 3.5


def test_fixedwing_flies_turns_and_climbs(city):
    w = World(city, n_quads=0, n_fixedwing=1, cfg=PhysicsConfig(wind=CALM))
    z = w.city.max_height + 40
    w.spawn(0, [-100, -100, z], yaw=np.pi / 4)
    run(w, 5.0)
    assert abs(w.pos[0, 2] - z) < 2.0 and abs(w.wings.airspeed[0] - 18.0) < 0.8
    w.command_fixedwing(0, turn_rate=0.25, climb_rate=2.0)
    h0 = np.arctan2(w.vel[0, 1], w.vel[0, 0]); z0 = w.pos[0, 2]
    run(w, 4.0)
    turned = (np.arctan2(w.vel[0, 1], w.vel[0, 0]) - h0 + np.pi) % (2 * np.pi) - np.pi
    assert 0.6 < turned < 1.2                 # ~0.25 rad/s to the left
    assert w.pos[0, 2] - z0 > 4.0
    w.close()


def test_projectile_is_ballistic():
    p = Projectiles()
    p.fire([0, 0, 0], [50.0, 0, 50.0])
    t, dt = 0.0, 1 / 240
    while p.pos[0, 2] >= 0:
        p.step(dt, 9.81, 0.0, 100.0, lambda x: np.zeros_like(x)); t += dt
    assert t == pytest.approx(2 * 50 / 9.81, abs=0.03)          # time of flight
    assert p.pos[0, 0] == pytest.approx(50 * t, rel=0.01)        # range without drag
    q = Projectiles(); q.fire([0, 0, 0], [50.0, 0, 50.0])
    while q.pos[0, 2] >= 0:
        q.step(dt, 9.81, 0.002, 100.0, lambda x: np.zeros_like(x))
    assert q.pos[0, 0] < 0.9 * p.pos[0, 0]                       # drag shortens the shot


def _open_spot(c):
    """A ground point with nothing built or planted within a few metres."""
    for r in np.arange(0.0, 120.0, 4.0):
        for a in np.arange(0.0, 2 * np.pi, 0.4):
            x, y = r * np.cos(a), r * np.sin(a)
            xs, ys = x + np.array([0, 6, -6, 0, 0, 12, 20, 28]), y + np.array([0, 0, 0, 6, -6, 0, 0, 0])
            if np.all(c.obstacle_z(xs, ys) - c.ground_z(xs, ys) < 0.3):
                return x, y
    raise RuntimeError("no open spot")


def test_defence_respects_range_and_fire_rate(city):
    m = ModelsConfig()
    m.defence.range_m = 60.0; m.defence.detection_range = 200.0; m.defence.fire_rate_hz = 2.0
    m.defence.dispersion_deg = 0.3                               # tight aim so the hit check is not a coin toss
    spot = _open_spot(CityMap.load(str(city)))
    w = World(city, m, PhysicsConfig(wind=CALM), n_quads=1, defence_positions=[spot])
    d = w.defences[0]
    w.spawn(0, d.muzzle + [28.0, 0, 95.0]); run(w, 4.0)
    assert d.shots == 0 and d.target == 0                        # tracked but out of range
    w.spawn(0, d.muzzle + [0, 0, 40.0]); run(w, 3.0)
    assert d.shots == 0                                          # straight overhead: above the elevation limit
    w.spawn(0, d.muzzle + [28.0, 0, 28.0]); run(w, 0.3); n0 = d.shots; hit = False
    for _ in range(int(3.0 * 120)):
        w.step(); hit |= bool(w.hit[0])
    assert 4 <= d.shots - n0 <= 7                                # ~2 per second for 3 s
    assert hit                                                   # a hovering target in range gets hit
    w.close()
