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
