"""branch_mlp's teacher EMA momentum schedules, against a frozen baseline."""


from cmct.train import ema_momentum_at
from tests.conftest import load_fixture


def test_dacs_schedule_matches_the_frozen_baseline():
    fx = load_fixture("ema_schedule.json")
    got = [ema_momentum_at(t, fx["momentum"], "dacs", fx["hard_copy_iters"])
           for t in range(len(fx["dacs"]))]
    assert got == fx["dacs"]


def test_hard_copy_schedule_matches_the_frozen_baseline():
    fx = load_fixture("ema_schedule.json")
    got = [ema_momentum_at(t, fx["momentum"], "hard_copy", fx["hard_copy_iters"])
           for t in range(len(fx["hard_copy"]))]
    assert got == fx["hard_copy"]


def test_dacs_first_update_is_a_hard_copy():
    assert ema_momentum_at(0, 0.99, "dacs", 100) == 0.0
