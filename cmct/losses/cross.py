"""The cross-teaching term: one branch's teacher supervising the other's student.

The reference computes both directions with the SAME function
(train_mfa_v2.py:684), so they are one function here too. What differs between
the directions is only which reference distribution is passed in -- and that
difference matters more than it looks: branch 1's teacher emits
`logit_scale * cosine` with logit_scale around 100, which saturates softmax, so
most of its samples clear the threshold. Branch 2's head is linear and
label-smoothed, so few of its samples do. The same 0.85 therefore selects very
different fractions in the two directions, which is why `mask_ratio` is reported
per direction rather than once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor

from cmct.losses.pseudo_label import pass_fraction, pseudo_label_ce

MODES = ("mask",)
"""Implemented forms of the cross loss.

The reference also has "gini" for branch 2 (CMKD's task/distill pair with the
other branch's teacher as the reference, ramped, no threshold). It is out of
scope, and naming it raises rather than quietly running "mask": a cross term
that silently changed form would be invisible in the loss value.
"""


@dataclass
class CrossOutput:
    value: Tensor
    """UNWEIGHTED. The caller multiplies by cross_weight, so this is the same
    number the reference logs as loss_u1_cross / loss2_cross."""
    mask_ratio: float
    """Fraction of the target batch clearing the threshold. Invisible in the
    loss value: with "mask", a batch where two samples pass and a batch where
    twenty pass both report the mean over those that did."""


def cross_loss(*, target_logits: Tensor, reference_probabilities: Tensor,
               threshold: float = 0.85,
               reduce: Literal["mask", "ratio"] = "mask",
               mode: str = "mask", branch: str = "") -> CrossOutput:
    """Masked cross-entropy on the other branch's teacher's argmax.

    Keyword-only: `target_logits` and `reference_probabilities` have the same
    shape, so a positional swap would train the student to predict itself
    without raising anything.
    """
    if mode not in MODES:
        where = f"branches[{branch}].cross_mode" if branch else "cross_mode"
        raise ValueError(
            f"{where}: {mode!r} is not implemented; available: {list(MODES)}"
        )
    return CrossOutput(
        value=pseudo_label_ce(target_logits, reference_probabilities,
                              threshold, reduce),
        mask_ratio=pass_fraction(reference_probabilities, threshold),
    )
