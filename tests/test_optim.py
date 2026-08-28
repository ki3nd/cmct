"""Optimizer construction and LR schedules."""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from cmct.config.schema import OptimConfig
from cmct.engine import build_lr_scheduler, build_optimizer, lr_at

INV = OptimConfig(lr=3e-6, scheduler="inv", gamma=3e-4, decay=0.75, nesterov=True)


def two_groups(multiplier=1000.0):
    encoder, head = nn.Linear(4, 4), nn.Linear(4, 4)
    return [
        {"params": list(encoder.parameters()), "lr": 1.0},
        {"params": list(head.parameters()), "lr": multiplier},
    ]


# --- lr_at -------------------------------------------------------------------

@pytest.mark.parametrize("step", [0, 1, 7, 1000, 5000, 9999])
def test_inv_matches_the_formula(step):
    expected = INV.lr * (1.0 + INV.gamma * step) ** (-INV.decay)
    assert lr_at(step, INV) == pytest.approx(expected)


def test_inv_starts_at_lr_and_decreases():
    assert lr_at(0, INV) == pytest.approx(INV.lr)
    values = [lr_at(s, INV) for s in range(0, 10_000, 100)]
    assert values == sorted(values, reverse=True)


def test_none_keeps_lr_constant():
    cfg = OptimConfig(lr=1e-3, scheduler="none")
    assert {lr_at(s, cfg) for s in (0, 10, 10_000)} == {1e-3}


def test_cosine_needs_total_steps_and_ends_at_zero():
    cfg = OptimConfig(lr=1e-3, scheduler="cosine")
    with pytest.raises(ValueError, match="needs total_steps"):
        lr_at(0, cfg)
    assert lr_at(0, cfg, 100) == pytest.approx(1e-3)
    assert lr_at(100, cfg, 100) == pytest.approx(0.0, abs=1e-12)
    assert lr_at(50, cfg, 100) == pytest.approx(1e-3 * 0.5 * (1 + math.cos(math.pi / 2)))


def test_lr_at_rejects_bad_input():
    with pytest.raises(ValueError, match="step must be >= 0"):
        lr_at(-1, INV)
    with pytest.raises(ValueError, match="unknown scheduler"):
        lr_at(0, OptimConfig(lr=1.0, scheduler="linear"))  # type: ignore[arg-type]


# --- optimizer ---------------------------------------------------------------

def test_sgd_carries_the_configured_hyperparameters():
    optimizer = build_optimizer(two_groups(), INV)
    assert isinstance(optimizer, torch.optim.SGD)
    for group in optimizer.param_groups:
        assert group["momentum"] == INV.momentum
        assert group["weight_decay"] == INV.weight_decay
        assert group["nesterov"] is True


def test_nesterov_is_off_unless_configured():
    optimizer = build_optimizer(two_groups(), OptimConfig(lr=1e-3))
    assert all(g["nesterov"] is False for g in optimizer.param_groups)


# --- the two together --------------------------------------------------------

def test_group_lr_is_multiplier_times_schedule_at_every_step():
    """The group lr holds the MULTIPLIER and the schedule holds the real LR, so
    the head's effective LR must be `multiplier` times the encoder's throughout."""
    optimizer = build_optimizer(two_groups(multiplier=1000.0), INV)
    scheduler = build_lr_scheduler(optimizer, INV, total_steps=10_000)

    for step in range(6):
        encoder_lr, head_lr = (g["lr"] for g in optimizer.param_groups)
        assert encoder_lr == pytest.approx(lr_at(step, INV))
        assert head_lr == pytest.approx(1000.0 * lr_at(step, INV))
        scheduler.step()


def test_constructor_lr_is_overridden_by_the_group_lr():
    """`lr=` on the optimizer has no effect once every group sets its own -- a
    detail that makes the starting LR look like cfg.lr when it is really
    cfg.lr * multiplier for the head."""
    optimizer = build_optimizer(two_groups(multiplier=1000.0), INV)
    build_lr_scheduler(optimizer, INV, total_steps=10_000)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1000.0 * INV.lr)
    assert optimizer.param_groups[1]["lr"] != pytest.approx(INV.lr)


def test_scheduler_step_count_drives_the_schedule():
    optimizer = build_optimizer(two_groups(multiplier=1.0), INV)
    scheduler = build_lr_scheduler(optimizer, INV, total_steps=10_000)
    for _ in range(100):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(lr_at(100, INV))


# --- warmup then cosine ------------------------------------------------------

WARMUP_COSINE = OptimConfig(lr=0.0035, warmup_lr=0.001, scheduler="warmup_cosine")
TOTAL, WARMUP = 1000, 50


def warmup_lr_at(step):
    return lr_at(step, WARMUP_COSINE, total_steps=TOTAL, warmup_steps=WARMUP)


def test_warmup_holds_a_flat_learning_rate():
    for step in range(WARMUP):
        assert warmup_lr_at(step) == pytest.approx(0.001), step


def test_the_boundary_step_still_uses_the_warmup_rate():
    """The schedule this reproduces overwrites the group's lr on every warmup step
    and steps the cosine only afterwards, so the first post-warmup iteration
    trains at the warmup rate and the cosine value first applies one step later.
    A schedule that jumps at the boundary is a different schedule."""
    assert warmup_lr_at(WARMUP) == pytest.approx(0.001)
    assert warmup_lr_at(WARMUP + 1) != pytest.approx(0.001)


def test_cosine_runs_over_the_post_warmup_horizon():
    """T_max is total_steps - warmup_steps, the cosine clock starts at 1 on the
    step after warmup, and its AMPLITUDE is warmup_lr -- see
    test_the_cosine_amplitude_is_the_warmup_rate_not_lr."""
    horizon = TOTAL - WARMUP
    for step, clock in ((51, 1), (100, 50), (999, 949)):
        expected = 0.001 * 0.5 * (1 + math.cos(math.pi * clock / horizon))
        assert warmup_lr_at(step) == pytest.approx(expected), step


def test_the_cosine_amplitude_is_the_warmup_rate_not_lr():
    """A branch's `lr` has NO effect on its learning rate once warmup_steps > 0.

    CosineAnnealingLR.get_lr() is recursive -- it scales the group's CURRENT lr
    rather than reading base_lrs -- and the reference's loop overwrites that lr
    with warmup_lr on every warmup step. So the cosine continues from warmup_lr
    and `lr` survives only as an unused base_lrs entry. Measured against the real
    scheduler at T=950: step 51 is 9.999973e-04, not 3.499990e-03.
    """
    assert warmup_lr_at(51) == pytest.approx(9.999973e-04, rel=1e-5)
    assert warmup_lr_at(51) != pytest.approx(3.499990e-03, rel=1e-3)

    doubled_lr = OptimConfig(lr=0.007, warmup_lr=0.001,
                             scheduler="warmup_cosine")
    for step in (0, WARMUP, 51, 500, 999):
        assert lr_at(step, doubled_lr, TOTAL, WARMUP) == \
            pytest.approx(warmup_lr_at(step)), step


def test_cosine_reaches_zero_at_the_end_of_the_horizon():
    assert warmup_lr_at(WARMUP + (TOTAL - WARMUP)) == pytest.approx(0.0, abs=1e-12)


def test_cosine_is_clamped_past_the_horizon():
    end = warmup_lr_at(TOTAL)
    assert warmup_lr_at(TOTAL + 500) == pytest.approx(end)


def test_the_rate_decreases_monotonically_after_warmup():
    values = [warmup_lr_at(s) for s in range(WARMUP + 1, TOTAL + 1)]
    assert values == sorted(values, reverse=True)


def test_the_rate_does_not_step_up_at_the_boundary():
    """The cosine starts from warmup_lr, so the rate decays continuously from the
    warmup value -- it never jumps up to `lr`."""
    assert warmup_lr_at(WARMUP + 1) < 0.001
    assert warmup_lr_at(WARMUP + 1) == pytest.approx(0.001, rel=1e-5)


def test_no_warmup_means_no_flat_phase():
    cfg = OptimConfig(lr=0.0035, warmup_lr=0.001, scheduler="warmup_cosine")
    assert lr_at(0, cfg, total_steps=100, warmup_steps=0) == pytest.approx(0.0035)


def test_warmup_cosine_without_warmup_lr_uses_lr():
    cfg = OptimConfig(lr=0.0035, scheduler="warmup_cosine")
    assert lr_at(0, cfg, total_steps=100, warmup_steps=10) == pytest.approx(0.0035)


def test_warmup_cosine_needs_total_steps():
    with pytest.raises(ValueError, match="needs total_steps"):
        lr_at(0, WARMUP_COSINE, warmup_steps=WARMUP)


def test_warmup_longer_than_the_run_raises():
    with pytest.raises(ValueError, match="warmup_steps"):
        lr_at(0, WARMUP_COSINE, total_steps=40, warmup_steps=50)


def test_the_schedule_drives_the_optimizer_group():
    """End to end through LambdaLR, the way the training script uses it."""
    encoder = nn.Linear(4, 4)
    optimizer = build_optimizer([{"params": list(encoder.parameters()), "lr": 1.0}],
                                WARMUP_COSINE)
    scheduler = build_lr_scheduler(optimizer, WARMUP_COSINE, total_steps=TOTAL,
                                   warmup_steps=WARMUP)
    for step in range(WARMUP + 10):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(warmup_lr_at(step)), step
        scheduler.step()


def test_existing_schedules_are_unaffected_by_the_new_argument():
    for cfg in (INV, OptimConfig(lr=1e-3, scheduler="none"),
                OptimConfig(lr=1e-3, scheduler="cosine")):
        assert lr_at(7, cfg, total_steps=100) == \
            lr_at(7, cfg, total_steps=100, warmup_steps=50)


def test_matches_the_reference_loop_step_by_step():
    """Simulates the reference's own loop -- a CosineAnnealingLR over
    total - warmup steps, the group's lr overwritten on every warmup step, and
    scheduler.step() called only after warmup -- and requires the same learning
    rate at every one of the 1000 steps.

    This is what catches an off-by-one at the boundary, which is the only place
    the two implementations could plausibly disagree.
    """
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=WARMUP_COSINE.lr)
    horizon = max(TOTAL - WARMUP, 1)
    reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=horizon
    )

    for step in range(TOTAL):
        in_warmup = step < WARMUP
        if in_warmup:
            for group in optimizer.param_groups:
                group["lr"] = WARMUP_COSINE.warmup_lr
        # the rate this step actually trains at, read before the scheduler moves
        reference_lr = optimizer.param_groups[0]["lr"]
        assert reference_lr == pytest.approx(warmup_lr_at(step)), step
        if not in_warmup:
            reference_scheduler.step()
