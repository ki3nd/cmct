"""Accuracy and loss over a finite loader."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    accuracy: float
    """Percent."""
    loss: float
    """Mean cross-entropy, without label smoothing -- matching the criterion the
    evaluation this reproduces uses."""
    correct: int
    total: int


@torch.no_grad()
def evaluate(logits_fn: Callable[[Tensor], Tensor], loader: DataLoader,
             device: str) -> EvalResult:
    """Score whatever `logits_fn` computes.

    Takes a function rather than a model so the same code scores the student
    (`model.logits`) and the teacher (`model.teacher_logits`) without knowing
    anything about either.
    """
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch in loader:
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        logits = logits_fn(images)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.numel()
    if total == 0:
        raise ValueError("evaluate: loader produced no samples")
    return EvalResult(
        accuracy=100.0 * correct / total,
        loss=loss_sum / total,
        correct=correct,
        total=total,
    )
