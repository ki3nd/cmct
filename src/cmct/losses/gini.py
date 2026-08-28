"""Class-balanced Gini impurity and the calibration coefficient it is weighted by."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def gini_impurity(pred: Tensor, coe: Tensor | float = 1.0) -> Tensor:
    """Class-balanced Gini impurity of a batch of predicted distributions.

    Two properties that are easy to lose and hard to notice:

    - The outer reduction is a SUM over the batch, not a mean, so the loss scales
      with batch size. Changing the batch size changes the effective weight of
      every term built on this.
    - `sum_dim` is the class-wise sum over the batch, detached, so each class's
      contribution is divided by how often it appears. That makes this a
      class-BALANCED objective rather than plain entropy minimisation.

    Args:
        pred: [B, C] probabilities, already softmaxed.
        coe: per-sample weight [B], or a scalar.
    """
    sum_dim = torch.sum(pred, dim=0).unsqueeze(dim=0).detach()
    return torch.sum(coe * (1 - torch.sum(pred**2 / sum_dim, dim=-1)))


def calibrated_coefficient(pred: Tensor, reference: Tensor) -> Tensor:
    """exp(-KL(reference || pred)), detached, in (0, 1].

    The direction matters and is easy to invert. F.kl_div(input, target) computes
    sum target * (log target - input), so passing log(pred) as input and
    `reference` as target gives KL(reference || pred).

    A coefficient near 1 means the two distributions agree. Callers use it to
    weight an agreement term, and (1 - coe) to weight a disagreement term.

    Detached: this is a weight, never a path for gradients.

    Args:
        pred: [B, C] probabilities.
        reference: [B, C] probabilities.
    """
    distance = F.kl_div(pred.log(), reference, reduction="none").sum(-1)
    return torch.exp(-distance).detach()
