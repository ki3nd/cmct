"""Branch 2's self-training loss: source CE plus the CMKD transfer terms.

Four components, summed:

    clf      CrossEntropy(label_smoothing) on the head's SOURCE logits
    task     lambda1 * ramp * gini(target_pred, coe)
    distill  lambda1 * ramp * gini(0.5 * (target_pred + cosine_pred), 1 - coe)
    reg      lambda2 * CE(source cosine logits) + lambda3 * ramp * gini(cosine_pred)

The reference for the self-consistency terms is the student's own LIVE cosine
branch, not a teacher. That is the design, not an omission: this loss trains a
learned head against the frozen-text cosine branch of the same encoder, and the
teacher exists for evaluation and for teaching other branches.

`reg` is what keeps the cosine branch from drifting. `task` and `distill` reach
it only through `coe`, which is detached, so without `reg` nothing would train
it.

Note that the two cross-entropies differ: `clf` carries label smoothing, the one
inside `reg` does not.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from cmct.losses.gini import calibrated_coefficient, gini_impurity
from cmct.losses.schedules import sigmoid_ramp


@dataclass
class CmkdOutput:
    total: Tensor
    clf: Tensor
    task: Tensor
    distill: Tensor
    reg: Tensor
    ramp: float
    """The schedule's value at this step, reported so a log can show where the
    transfer terms currently sit rather than only their product."""


class CmkdLoss:
    def __init__(self, max_iter: int, lambda1: float = 0.25, lambda2: float = 0.1,
                 lambda3: float = 0.025, label_smoothing: float = 0.1,
                 gamma: float = 1.0) -> None:
        """`max_iter` is the branch's own total number of steps: the ramp reaches
        its ceiling there. Upstream hardcoded 10000, which is also
        total_macro_steps * steps_per_macro for the configuration reproduced
        here."""
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")
        self.max_iter = max_iter
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.label_smoothing = label_smoothing
        self.gamma = gamma

    def __call__(self, *, source_logits: Tensor, source_label: Tensor,
                 source_cosine_logits: Tensor, target_logits: Tensor,
                 target_cosine_logits: Tensor, step: int,
                 self_reference_logits: Tensor | None = None) -> CmkdOutput:
        """Keyword-only on purpose: these are same-shaped tensors, so swapping
        two of them positionally would be a silent error rather than a type
        error."""
        ramp = sigmoid_ramp(step, self.max_iter, self.gamma)

        clf = F.cross_entropy(source_logits, source_label,
                              label_smoothing=self.label_smoothing)

        target_pred = F.softmax(target_logits, dim=1)
        # The STUDENT's live cosine branch. `reg` uses this one and only this
        # one, so it keeps a gradient into the visual encoder.
        cosine_pred = F.softmax(target_cosine_logits, dim=-1)

        # The self-reference for `coe` and `mix` alone. `self_reference_logits`
        # is the teacher's cosine branch when the caller supplies it
        # (--s2-self-from-teacher); the reference swaps ONLY these two, never
        # `reg` (vlpuda_pure/models/cmkd.py:42-45 against :54). Passing the
        # teacher for all three looks harmless and is not: the teacher runs
        # under no_grad, so `reg` would contribute exactly zero gradient and the
        # target domain would stop reaching branch 2's encoder through the
        # cosine branch at all -- while the logged number stayed plausible.
        reference_pred = (cosine_pred if self_reference_logits is None
                          else F.softmax(self_reference_logits, dim=-1))
        coe = calibrated_coefficient(target_pred, reference_pred)
        mixed = 0.5 * (target_pred + reference_pred.detach())

        weight = self.lambda1 * ramp
        task = weight * gini_impurity(target_pred, coe)
        distill = weight * gini_impurity(mixed, 1 - coe)
        reg = (
            self.lambda2 * F.cross_entropy(source_cosine_logits, source_label)
            + self.lambda3 * ramp * gini_impurity(cosine_pred)
        )

        return CmkdOutput(
            total=clf + task + distill + reg,
            clf=clf, task=task, distill=distill, reg=reg, ramp=ramp,
        )

    @classmethod
    def from_branch_config(cls, branch, max_iter: int) -> CmkdLoss:
        """Read this branch type's private knobs out of `extra`."""
        extra = dict(branch.extra)
        unknown = sorted(set(extra) - {"lambda1", "lambda2", "lambda3",
                                       "label_smoothing", "gamma"})
        if unknown:
            raise ValueError(
                f"branches[{branch.name}].extra: unknown key(s) {unknown} for branch "
                f"type '{branch.type}'"
            )
        return cls(max_iter=max_iter, **extra)
