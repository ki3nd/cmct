"""Optimizer and LR schedule construction.

The "inv" schedule reproduces the one branch 2 comes from, and the way it is
wired is easy to misread. Each param group carries a MULTIPLIER as its lr (1.0
for the encoder, `param_group_multipliers[name]` for the head), and the actual
learning rate lives in the LambdaLR lambda:

    lr_group(x) = multiplier * lr * (1 + gamma * x) ** -decay

So the `lr=` passed to the optimizer constructor has no effect: every group sets
its own. Reading the code quickly suggests the starting LR is `cfg.lr`; it is
really `cfg.lr` for the encoder and `cfg.lr * multiplier` for the head.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cmct.config.schema import OptimConfig


def lr_at(step: int, cfg: OptimConfig, total_steps: int | None = None) -> float:
    """The schedule's multiplicative factor at `step`, excluding the group's own
    multiplier. Exposed separately so it can be checked without an optimizer."""
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if cfg.scheduler == "none":
        return cfg.lr
    if cfg.scheduler == "inv":
        return cfg.lr * (1.0 + cfg.gamma * float(step)) ** (-cfg.decay)
    if cfg.scheduler == "cosine":
        if total_steps is None or total_steps <= 0:
            raise ValueError("scheduler 'cosine' needs total_steps > 0")
        progress = min(step / total_steps, 1.0)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


def build_optimizer(param_groups: list[dict], cfg: OptimConfig) -> Optimizer:
    """SGD, because that is what both branches use. There is no optimizer-name
    knob: adding one would mean a config field selecting between an optimizer in
    use and one that is not."""
    return torch.optim.SGD(
        param_groups, lr=cfg.lr, momentum=cfg.momentum,
        weight_decay=cfg.weight_decay, nesterov=cfg.nesterov,
    )


def build_lr_scheduler(optimizer: Optimizer, cfg: OptimConfig,
                       total_steps: int | None = None) -> LRScheduler:
    """LambdaLR multiplies each group's initial lr -- its multiplier -- by
    lr_at(step)."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_at(step, cfg, total_steps)
    )
