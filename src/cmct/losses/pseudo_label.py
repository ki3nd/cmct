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

    Both are exactly 0 when nothing passes -- at ANY logits dtype; see the mask
    below. The reference is detached: it is a target, never a path for gradients.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")
    if reduce not in ("mask", "ratio"):
        raise ValueError(f"reduce must be 'mask' or 'ratio', got {reduce!r}")

    reference = reference.detach()
    confidence, labels = torch.max(reference, dim=-1)
    # float32, NOT logits.dtype. Under fp16 logits the epsilon below underflows
    # (fp16: 0.0 + 1e-8 == 0.0), so a batch where nothing clears the threshold
    # divides 0.0 by 0.0 and returns NaN instead of 0 -- which then propagates
    # through backward into every LoRA factor and the teacher, permanently. The
    # reference builds the mask with .float() for the same reason
    # (trainers/da/phpl_momentum.py:779). fp16 per-sample losses promote to
    # float32 against it, so the reduction below is fp32 either way.
    mask = confidence.ge(threshold).float()

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
