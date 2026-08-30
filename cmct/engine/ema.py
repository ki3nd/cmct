"""EMA teacher updates and momentum schedules.

The teacher is the EMA accumulator, so it must be fp32. Measured on 4096
elements with the teacher 5% away from a stationary student and momentum 0.99:

    fp16   |teacher - student|: 7.87e-04 -> 5.06e-04   (froze at step 97)
    fp32   |teacher - student|: 8.07e-04 -> 8.33e-08

In fp16 it is not the value that underflows, it is the UPDATE. The per-step
nudge is (1 - momentum) * (student - teacher); once that falls below fp16's
representable increment at the teacher's magnitude (relative eps 9.8e-4),
`add_` rounds to no change at all and the teacher sits at its old value
forever. With momentum 0.99 that happens after roughly 100 steps -- exactly
when the ramp finishes. Hence the fp32 check below.
"""
from __future__ import annotations

import torch
from torch import nn

from cmct.config.schema import EmaConfig


def momentum_at(step: int, cfg: EmaConfig) -> float:
    """Momentum for a teacher update at `step` (the branch's own step, from 0).

    **Step 0 is 0.0 under every schedule.** The teacher module is allocated as a
    structural copy of the student, which means it starts holding the student's
    INITIALIZATION -- including a classifier head initialized to near-zero noise.
    Blending that in would leave the initialization inside the teacher for as long
    as the momentum is high: at 0.996 it still carries 2% of it after 1000 steps.
    So the first update always replaces the teacher outright, and no schedule may
    opt out. Since the update runs after optimizer.step(), what it copies is the
    student's weights after one real training step, never the initialization.

    "ramp": min(step / (step + 1), momentum). Rises continuously to `momentum`
    over roughly momentum / (1 - momentum) steps (99 steps at 0.99), with no
    warmup-length hyperparameter and no discontinuous jump.

    "hard_copy_then_jump": 0 for `warmup_iters` steps, then a jump to `momentum`.
    Kept for ablation only.

    "const": `momentum` from step 1 on.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if step == 0:
        return 0.0
    if cfg.schedule == "ramp":
        return min(step / (step + 1), cfg.momentum)
    if cfg.schedule == "hard_copy_then_jump":
        return 0.0 if step < cfg.warmup_iters else cfg.momentum
    if cfg.schedule == "const":
        return cfg.momentum
    raise ValueError(f"unknown ema schedule {cfg.schedule!r}")


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """Move `teacher` toward `student` by (1 - momentum), in place.

    Walks state_dict, so it covers parameters AND buffers over the whole module.
    The teacher is a structural copy of the student, and going through
    state_dict keeps it one.

    A non-floating buffer (BatchNorm's num_batches_tracked, for instance) is
    hard-copied rather than blended: a fractional momentum has no meaning for an
    integer counter.

    state_dict() returns references to the live tensors, so the in-place mul_ and
    add_ below are the update -- no load_state_dict is needed.
    """
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"momentum must be within [0, 1], got {momentum}")

    teacher_state = teacher.state_dict()
    for key, value in student.state_dict().items():
        target = teacher_state[key]
        if not torch.is_floating_point(value):
            target.copy_(value)
            continue
        if target.dtype is not torch.float32:
            raise TypeError(
                f"teacher tensor '{key}' is {target.dtype}, but an EMA teacher must be "
                f"float32: in float16 the update rounds to zero and the teacher freezes "
                f"(see this module's docstring)"
            )
        target.mul_(momentum).add_(value.to(torch.float32), alpha=1.0 - momentum)
