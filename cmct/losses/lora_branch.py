"""Branch 1's loss: source cross-entropy, a confidence-masked pseudo-label term,
and MK-MMD between source and target features.

The pseudo-label reference is passed in rather than chosen here, because it
switches during training: the frozen zero-shot CLIP through warmup, the EMA
teacher afterwards. `reference_name` is carried through only so a log can say
which one produced the number.

Every reported component is already multiplied by its weight, so they sum to the
total. The implementation this replaces printed one term unweighted while adding
a weighted one to the total, and its printed components did not add up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from cmct.losses.mmd import mk_mmd
from cmct.losses.pseudo_label import pass_fraction, pseudo_label_ce


@dataclass
class LoraBranchOutput:
    total: Tensor
    source_ce: Tensor
    pseudo_label: Tensor
    mmd: Tensor
    mask_ratio: float
    """Share of the target batch clearing the confidence threshold. Invisible in
    the loss value, and it is what makes the "mask" reduction noisy."""
    reference: str


class LoraBranchLoss:
    def __init__(self, threshold: float = 0.85,
                 reduce: Literal["mask", "ratio"] = "mask",
                 mmd_weight: float = 1.0) -> None:
        self.threshold = threshold
        self.reduce = reduce
        self.mmd_weight = mmd_weight

    def __call__(self, *, source_logits: Tensor, source_label: Tensor,
                 target_logits: Tensor, reference_probabilities: Tensor,
                 source_features: Tensor | None = None,
                 target_features: Tensor | None = None,
                 reference_name: str = "teacher") -> LoraBranchOutput:
        """Keyword-only: these are same-shaped tensors, so a positional swap
        would be a silent error rather than a type error."""
        source_ce = F.cross_entropy(source_logits, source_label)
        pseudo = pseudo_label_ce(target_logits, reference_probabilities,
                                 self.threshold, self.reduce)

        if self.mmd_weight > 0:
            if source_features is None or target_features is None:
                raise ValueError(
                    "mmd_weight > 0 needs both source_features and target_features"
                )
            mmd = mk_mmd(source_features, target_features)
        else:
            mmd = torch.zeros((), device=source_logits.device,
                              dtype=source_logits.dtype)

        return LoraBranchOutput(
            # `mmd` is reported RAW and weighted only here, like the other three
            # terms. It used to be reported pre-multiplied, so the four logged
            # numbers were not on one scale and `mmd` did not compare with the
            # reference's, which prints all four raw and applies the weights only
            # when summing.
            total=source_ce + pseudo + self.mmd_weight * mmd,
            source_ce=source_ce, pseudo_label=pseudo, mmd=mmd,
            mask_ratio=pass_fraction(reference_probabilities, self.threshold),
            reference=reference_name,
        )
