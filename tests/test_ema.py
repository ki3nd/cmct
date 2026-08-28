"""EMA update mechanics and momentum schedules."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from cmct.config.schema import EmaConfig
from cmct.engine.ema import ema_update, momentum_at


def pair(dtype=torch.float32):
    torch.manual_seed(0)
    student = nn.Linear(8, 4)
    teacher = nn.Linear(8, 4)
    with torch.no_grad():
        teacher.weight.copy_(student.weight * 1.05)
        teacher.bias.copy_(student.bias * 1.05)
    return teacher.to(dtype), student.to(dtype)


# --- schedules ---------------------------------------------------------------

@pytest.mark.parametrize("schedule", ["ramp", "const", "hard_copy_then_jump"])
def test_every_schedule_starts_at_zero(schedule):
    """The teacher is allocated holding the student's initialization; the first
    update must replace it outright rather than blend it in. No schedule opts
    out."""
    cfg = EmaConfig(momentum=0.996, schedule=schedule)
    assert momentum_at(0, cfg) == 0.0


def test_ramp_is_monotone_and_capped():
    cfg = EmaConfig(momentum=0.99, schedule="ramp")
    values = [momentum_at(t, cfg) for t in range(500)]
    assert values == sorted(values)
    assert max(values) <= cfg.momentum
    assert momentum_at(10_000, cfg) == pytest.approx(cfg.momentum)


def test_hard_copy_then_jump():
    cfg = EmaConfig(momentum=0.99, schedule="hard_copy_then_jump", warmup_iters=3)
    assert [momentum_at(t, cfg) for t in range(5)] == [0.0, 0.0, 0.0, 0.99, 0.99]


def test_const_applies_from_step_one():
    cfg = EmaConfig(momentum=0.996, schedule="const")
    assert momentum_at(1, cfg) == 0.996
    assert momentum_at(999, cfg) == 0.996


def test_a_high_constant_momentum_would_retain_the_initialization():
    """Why step 0 is forced to 0.0: at momentum 0.996 the teacher still carries
    0.996 ** 1000 = 1.8% of whatever it started from after 1000 steps."""
    assert 0.996 ** 1000 > 0.01


def test_negative_step_raises():
    with pytest.raises(ValueError, match="step must be >= 0"):
        momentum_at(-1, EmaConfig())


# --- update ------------------------------------------------------------------

def test_momentum_zero_is_a_hard_copy():
    teacher, student = pair()
    ema_update(teacher, student, 0.0)
    assert torch.equal(teacher.weight, student.weight)
    assert torch.equal(teacher.bias, student.bias)


def test_momentum_one_leaves_teacher_untouched():
    teacher, student = pair()
    before = teacher.weight.clone()
    ema_update(teacher, student, 1.0)
    assert torch.equal(teacher.weight, before)


def test_update_is_the_exact_convex_combination():
    teacher, student = pair()
    before = teacher.weight.clone()
    ema_update(teacher, student, 0.9)
    expected = 0.9 * before + 0.1 * student.weight
    assert torch.allclose(teacher.weight, expected, atol=0, rtol=1e-6)


def test_momentum_outside_unit_interval_raises():
    teacher, student = pair()
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="momentum must be within"):
            ema_update(teacher, student, bad)


def test_non_float_buffers_are_hard_copied_not_blended():
    """num_batches_tracked is an integer counter; a fractional momentum has no
    meaning for it."""
    teacher, student = nn.BatchNorm1d(4), nn.BatchNorm1d(4)
    for _ in range(7):
        student(torch.randn(3, 4))
    assert int(student.num_batches_tracked) == 7
    assert int(teacher.num_batches_tracked) == 0

    ema_update(teacher, student, 0.99)
    assert int(teacher.num_batches_tracked) == 7


def test_float_buffers_are_blended():
    teacher, student = nn.BatchNorm1d(4), nn.BatchNorm1d(4)
    for _ in range(7):
        student(torch.randn(3, 4) * 5 + 2)
    before = teacher.running_mean.clone()
    ema_update(teacher, student, 0.5)
    assert torch.allclose(teacher.running_mean, 0.5 * before + 0.5 * student.running_mean)


def test_fp16_teacher_raises():
    teacher, student = pair(torch.float16)
    with pytest.raises(TypeError, match="must be float32"):
        ema_update(teacher, student, 0.99)


def test_fp16_ema_freezes_while_fp32_converges():
    """The reason the fp32 requirement exists. In fp16 the per-step nudge rounds
    to no change and the teacher stops moving; in fp32 it closes the gap."""
    def run(dtype):
        torch.manual_seed(0)
        student = (torch.randn(4096) * 0.02).to(dtype)
        teacher = (student.float() * 1.05).to(dtype)
        start = (teacher.float() - student.float()).abs().mean().item()
        froze_at = None
        for step in range(1000):
            before = teacher.clone()
            teacher.mul_(0.99).add_(student, alpha=0.01)
            if froze_at is None and torch.equal(before, teacher):
                froze_at = step
        end = (teacher.float() - student.float()).abs().mean().item()
        return start / end, froze_at

    fp16_ratio, fp16_froze = run(torch.float16)
    fp32_ratio, fp32_froze = run(torch.float32)

    assert fp16_froze is not None and fp16_froze < 200
    assert fp16_ratio < 5
    assert fp32_ratio > 100
    assert fp32_froze is None or fp32_froze > 500
