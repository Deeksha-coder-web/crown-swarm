import numpy as np
import pytest
from scipy.spatial.distance import pdist

from crownswarm import CONTROLLERS, TEST_PLANS, TRAIN_PLANS, EpisodeConfig, get_plan, run_episode
from crownswarm import controllers as C
from crownswarm import world
from crownswarm.floorplans import spawn
from crownswarm.metrics import coverage_ceiling, covered_mask


def test_train_and_test_plans_are_disjoint_and_valid():
    assert not set(TRAIN_PLANS) & set(TEST_PLANS)
    for name in TRAIN_PLANS + TEST_PLANS + ("open",):
        p = get_plan(name)
        assert p.reach[p.cell(*p.entrance)]
        assert len(p.cover_pts) > 100


def test_spawn_is_packed_at_entrance_without_overlap():
    p = get_plan("office")
    pos = spawn(p, 80, 0.3, np.random.default_rng(0))
    assert pdist(pos).min() >= 0.6
    assert all(world.clearance_at(p.clear, p.cs, x, y) >= 0.3 for x, y in pos)
    assert np.hypot(*(pos - p.entrance).T).max() < 8


def test_line_of_sight_is_blocked_by_walls():
    p = get_plan("office")
    # corridor to corridor: clear; corridor to a room straight through the wall: blocked
    assert world.line_of_sight(p.free, p.cs, 2, 8, 20, 8)
    assert not world.line_of_sight(p.free, p.cs, 5, 8, 5, 12)


def test_occlusion_hides_agents_behind_a_closer_one():
    pos = np.array([[5.0, 8.0], [6.0, 8.0], [7.0, 8.0]])
    heading = np.zeros(3)
    ptr = np.array([0, 2, 2, 2])
    nbr = np.array([1, 2])
    dist = np.array([1.0, 2.0])
    los = np.array([True, True])
    with_occ = world.sector_signal(pos, heading, ptr, nbr, dist, los, 0.3, 3.0, True)
    without = world.sector_signal(pos, heading, ptr, nbr, dist, los, 0.3, 3.0, False)
    assert with_occ[0, 0] == pytest.approx((1 - 1 / 3) ** 2)
    assert without[0, 0] == pytest.approx((1 - 1 / 3) ** 2 + (1 - 2 / 3) ** 2)


@pytest.mark.parametrize("name", list(CONTROLLERS))
def test_agents_never_enter_walls_or_overlap(name):
    out = run_episode(EpisodeConfig(controller=name, plan="rubble", steps=300, record_every=10,
                                    fail_frac=0.2, sway=0.05, seed=3))
    p = get_plan("rubble")
    for pos in (f["pos"] for f in out["frames"]):
        assert min(world.clearance_at(p.clear, p.cs, x, y) for x, y in pos) >= 0.3 - 1e-9
        assert pdist(pos).min() > 0.6 - 0.05
    assert out["n_final"] < 60


def test_only_lloyd_sees_privileged_state(monkeypatch):
    seen = {}
    for name, cls in CONTROLLERS.items():
        orig = cls.act

        def spy(self, s, st, rng, _orig=orig, _name=name):
            seen[_name] = s.pos is not None or s.plan is not None
            return _orig(self, s, st, rng)
        monkeypatch.setattr(cls, "act", spy)
        run_episode(EpisodeConfig(controller=name, steps=2))
    assert seen == {n: n == "Lloyd" for n in CONTROLLERS}


def test_param_encoding_roundtrip():
    for cls in CONTROLLERS.values():
        if cls.space:
            p = {q.name: q.default for q in cls.space}
            back = cls.decode(cls.encode(p))
            assert back == pytest.approx(p)


def test_b_total_stop_rule_deadlocks_but_quadrant_rule_spreads():
    kw = dict(controller="B", plan="hall", steps=400, seed=0)
    total = run_episode(EpisodeConfig(flags={"stop_on": "total"}, **kw))
    quad = run_episode(EpisodeConfig(**kw))
    assert quad["coverage"] > total["coverage"] + 0.1


def test_coverage_ceiling_bounds_actual_coverage():
    p = get_plan("apartment")
    pos = spawn(p, 60, 0.3, np.random.default_rng(0))
    assert covered_mask(pos, p, 2.0).mean() <= coverage_ceiling(p, 60) + 1e-9
