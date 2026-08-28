"""Confidence-masked cross-entropy against a reference distribution's argmax."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

EPSILON = 1e-8


def pseudo_label_ce(logits: Tensor, reference: Tensor, threshold: float,
                    reduce: Literal["mask", "ratio"] = "mask") -> Tensor:
    """Cross-entropy against argmax(reference), keeping only confident samples.

    "mask" averages over the samples that clear `threshold` -- it divides by the
    COUNT that pass, so the magnitude is independent of how few do. That keeps the
    term from fading out when the reference is unsure, at the cost of high
    variance when only a handful pass.

    "ratio" takes cross-entropy over the whole batch, giving every sample its
    argmax pseudo-label, and scales by the FRACTION that pass. Smoother and lower
    variance, but the term shrinks in proportion.

    Both are exactly 0 when nothing passes. The reference is detached: it is a
    target, never a path for gradients.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")
    if reduce not in ("mask", "ratio"):
        raise ValueError(f"reduce must be 'mask' or 'ratio', got {reduce!r}")

    reference = reference.detach()
    confidence, labels = torch.max(reference, dim=-1)
    mask = confidence.ge(threshold).to(logits.dtype)

    if reduce == "ratio":
        return F.cross_entropy(logits, labels) * mask.mean()
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return (per_sample * mask).sum() / (mask.sum() + EPSILON)


def pass_fraction(reference: Tensor, threshold: float) -> float:
    """Share of the batch clearing `threshold`.

    Worth logging on its own: it is what makes the "mask" reduction noisy, and it
    is invisible in the loss value.
    """
    return float(reference.detach().max(dim=-1).values.ge(threshold).float().mean())
