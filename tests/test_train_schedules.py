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


def test_evaluate_reports_none_for_a_branch_that_is_switched_off():
    """`evaluate` is the one place both `enabled` switches converge, and a
    missing branch has to come back as None rather than 0.0 -- a fabricated
    0.00% in the eval line would read as a collapsed model."""
    import torch

    from cmct.evaluate import evaluate

    class FakeLoraTeacher:
        def __call__(self, image):
            logits = torch.zeros(image.size(0), 3)
            logits[:, 1] = 1.0            # always predicts class 1
            return logits, None

    loader = [{"img": torch.zeros(4, 3, 2, 2), "label": torch.tensor([1, 1, 0, 1])}]

    acc_lora, acc_mlp, acc_ens = evaluate(
        FakeLoraTeacher(), None, None, loader, torch.device("cpu")
    )
    assert acc_lora == 75.0
    assert acc_mlp is None
    assert acc_ens is None
