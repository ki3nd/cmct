"""Loss weight schedules."""
from __future__ import annotations

import math


def sigmoid_ramp(step: int, max_iter: int, gamma: float = 1.0) -> float:
    """2 / (1 + exp(-gamma * p)) - 1, where p = min(step / max_iter, 1).

    Note the range: with gamma = 1.0 this runs from 0.0 to **0.4621**, not to
    1.0, because 2 / (1 + e^-1) - 1 = 0.4621. A term weighted by
    `lambda1 * ramp` therefore peaks at 0.4621 * lambda1, and reading it as a
    0-to-1 ramp overstates the final weight by more than a factor of two.

    gamma is 1.0 in every configuration this reproduces: the schedule it comes
    from accepts gamma but is never constructed with one.

    `step` is passed in rather than counted internally. The original held a
    `curr_iter` field, read at the start of the loss and advanced at the end,
    which made "read the ramp before the forward pass" a rule callers had to
    know. There is no ordering to get wrong here.
    """
    if max_iter <= 0:
        raise ValueError(f"max_iter must be > 0, got {max_iter}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    p = min(step / max_iter, 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
