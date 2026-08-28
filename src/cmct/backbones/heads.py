"""The learned classification head."""
from __future__ import annotations

from torch import nn


def _init_weights(module: nn.Module) -> None:
    """Linear starts near zero (std 1e-3), so the head begins with almost no
    opinion. BatchNorm's bias is frozen: it stays in the optimizer's param group
    but never receives a gradient."""
    name = type(module).__name__
    if "Linear" in name:
        nn.init.normal_(module.weight, std=0.001)
    elif "BatchNorm" in name:
        module.bias.requires_grad_(False)
        if module.affine:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


class ClassifierHead(nn.Sequential):
    """BatchNorm1d -> LayerNorm -> Linear(bias=False).

    The BatchNorm1d is the only place in this branch where train/eval mode
    changes the head's output, since it carries running statistics.
    """

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__(
            nn.BatchNorm1d(feature_dim),
            nn.LayerNorm(feature_dim, eps=1e-6),
            nn.Linear(feature_dim, num_classes, bias=False),
        )
        self.apply(_init_weights)
