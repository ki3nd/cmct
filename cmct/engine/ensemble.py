"""Scoring several models, and optionally their combination, in one pass.

One pass matters: the models can only be compared, and combined, if each saw the
same image in the same order. Two separate passes over a shuffled loader would
compare them on different data; two passes over a deterministic one would just
cost twice as much.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from cmct.engine.evaluator import EvalResult

MODES = ("off", "mean_prob", "mean_logit")


def _combine(logits: list[Tensor], mode: str) -> Tensor:
    """Return combined LOGITS -- log-probabilities for "mean_prob", so that the
    caller's cross-entropy stays a cross-entropy in both modes.

    Averaging probabilities and averaging logits are different operations, not
    two spellings of one. Branch 1's logits carry `logit_scale` around 100 and
    branch 2's come from a linear head, so a logit average is dominated by
    branch 1 while a probability average is not.
    """
    if mode == "mean_logit":
        return torch.stack(logits).mean(dim=0)
    probabilities = torch.stack([F.softmax(x, dim=-1) for x in logits]).mean(dim=0)
    return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()


@torch.no_grad()
def evaluate_ensemble(logits_fns: dict[str, Callable[[Tensor], Tensor]],
                      loader: DataLoader, device: str,
                      mode: Literal["off", "mean_prob", "mean_logit"] = "off",
                      ) -> dict[str, EvalResult]:
    """Score every model in `logits_fns`, plus their combination unless `mode`
    is "off".

    The returned dict has one entry per key of `logits_fns`, and an "ensemble"
    entry only when `mode` is not "off" -- absent rather than present-and-empty,
    so a caller cannot report a number that was never computed.
    """
    if mode not in MODES:
        raise ValueError(f"cotrain.ensemble must be one of {list(MODES)}, got {mode!r}")
    if not logits_fns:
        raise ValueError("evaluate_ensemble: no models to score")
    if "ensemble" in logits_fns:
        raise ValueError("'ensemble' is reserved; name the branch something else")

    names = list(logits_fns)
    keys = names if mode == "off" else [*names, "ensemble"]
    correct = dict.fromkeys(keys, 0)
    loss_sum = dict.fromkeys(keys, 0.0)
    total = 0

    for batch in loader:
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        produced = [logits_fns[name](images) for name in names]
        scored = dict(zip(names, produced, strict=True))
        if mode != "off":
            scored["ensemble"] = _combine(produced, mode)
        for key, logits in scored.items():
            loss_sum[key] += float(F.cross_entropy(logits, labels, reduction="sum"))
            correct[key] += int((logits.argmax(dim=1) == labels).sum())
        total += labels.numel()

    if total == 0:
        raise ValueError("evaluate_ensemble: loader produced no samples")
    return {
        key: EvalResult(accuracy=100.0 * correct[key] / total,
                        loss=loss_sum[key] / total,
                        correct=correct[key], total=total)
        for key in keys
    }
