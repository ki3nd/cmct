"""Branch 2's model: a CLIP encoder with a learned head, plus its EMA teacher.

The teacher is one object covering BOTH halves -- an EMA encoder and an EMA
head -- built together and updated together with a single momentum. Half a
teacher is not a teacher: the head is trained on the LIVE encoder's features, so
a smoothed encoder paired with an instantaneously-tracking head (or the reverse)
is two things from two different moments.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from cmct.backbones.clip.model import CLIP
from cmct.backbones.encoder import ClipEncoder
from cmct.backbones.heads import ClassifierHead
from cmct.engine.ema import ema_update

TEMPLATE = "an image of a {}"
"""The template branch 2 comes from. Its hardcoded Office-Home prompt list is
exactly `"an image of a " + directory_name.replace("_", " ").lower()` for all 65
classes, in the same sorted order -- verified class by class against
models/backbone.py, 0 of 65 differing."""


def _freeze(module: nn.Module) -> nn.Module:
    """Put a module in eval mode and detach it from autograd."""
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module


class VlpModel(nn.Module):
    """Student encoder + head, and their EMA teacher.

    Requires an fp32 CLIP model. In the codebase this replaces, branch 2 was
    always fp32 and there was no way to make it fp16; fp16 is rejected here
    rather than supported, because an fp16 EMA teacher freezes after ~100 steps
    (see cmct.engine.ema). Branch 1 may still be fp16 -- its EMA touches only
    fp32 LoRA factors.
    """

    def __init__(self, clip_model: CLIP, classnames: list[str], num_classes: int,
                 template: str = TEMPLATE) -> None:
        super().__init__()
        dtype = clip_model.visual.conv1.weight.dtype
        if dtype is not torch.float32:
            raise TypeError(
                f"branch 'vlp_clip' requires an fp32 CLIP model, got {dtype}. Set this "
                f"branch's backbone.dtype to fp32: an fp16 EMA teacher stops updating "
                f"after roughly 100 steps (see cmct.engine.ema)"
            )
        if len(classnames) != num_classes:
            raise ValueError(
                f"got {len(classnames)} classnames but num_classes={num_classes}"
            )

        self.encoder = ClipEncoder(clip_model, classnames, template)
        self.head = ClassifierHead(self.encoder.feature_dim, num_classes)

        # .float() is a no-op while the student is fp32, but it states the
        # invariant here instead of leaving it to depend on the check above.
        self.teacher_encoder = _freeze(copy.deepcopy(self.encoder).float())
        self.teacher_head = _freeze(copy.deepcopy(self.head).float())
        self.teacher_updates = 0
        """How many EMA updates the teacher has received. Zero means it still
        holds the student's initialization; see ema_update."""

    # --- student ------------------------------------------------------------

    def features(self, images: Tensor) -> Tensor:
        self.encoder.freeze_batchnorm()
        return self.encoder.features(images)

    def logits(self, images: Tensor) -> Tensor:
        return self.head(self.features(images))

    def cosine_logits(self, images: Tensor) -> Tensor:
        return self.encoder.cosine_logits(self.features(images))

    # --- teacher ------------------------------------------------------------

    @torch.no_grad()
    def teacher_logits(self, images: Tensor) -> Tensor:
        """EMA head on EMA features. This is what other branches read, and what
        evaluation scores."""
        return self.teacher_head(self.teacher_encoder.features(images))

    @torch.no_grad()
    def teacher_cosine_logits(self, images: Tensor) -> Tensor:
        return self.teacher_encoder(images)

    def ema_update(self, momentum: float) -> None:
        """Both halves, one momentum, no way to call them apart.

        The FIRST update ignores `momentum` and replaces the teacher outright.
        Until then the teacher holds the student's initialization -- including a
        head initialized to near-zero noise -- and blending that in leaves it
        there for a long time: at momentum 0.996 the teacher still carries 1.8%
        of it after 1000 steps. Called after optimizer.step(), this first update
        therefore starts the teacher from the student's weights after one real
        training step.

        The code this replaces got the same outcome from three things happening
        to line up -- a step counter starting at 0, a momentum formula that is 0
        there, and the update being placed after optimizer.step(). Changing any
        one of them would have kept the initialization, silently. Here it does
        not depend on the caller.
        """
        if self.teacher_updates == 0:
            momentum = 0.0
        ema_update(self.teacher_encoder, self.encoder, momentum)
        ema_update(self.teacher_head, self.head, momentum)
        self.teacher_updates += 1

    @property
    def teacher_is_initialized(self) -> bool:
        """False until the first ema_update. While False, the teacher's outputs
        come from the student's initialization, not from anything trained."""
        return self.teacher_updates > 0

    # --- plumbing -----------------------------------------------------------

    def param_groups(self, lr: float, head_multiplier: float) -> list[dict]:
        """Two groups: the visual tower, and the head at a multiple of the LR.

        Frozen by omission: the whole text tower and logit_scale.

        The head group includes BatchNorm's bias even though its requires_grad is
        False. The optimizer skips it (its grad stays None), and keeping it here
        matches the param groups this replaces.
        """
        return [
            {"params": list(self.encoder.visual_parameters()), "lr": lr},
            {"params": list(self.head.parameters()), "lr": head_multiplier * lr},
        ]

    def train(self, mode: bool = True) -> VlpModel:
        """Switch the student only; the teachers stay in eval.

        nn.Module.train() recurses into every submodule, so the inherited version
        would flip the teachers into train mode -- and a teacher head in train
        mode makes its BatchNorm1d use batch statistics and update its running
        stats from the student's batches. The code this replaces avoided that by
        never calling train() on the model at all, and setting the student's two
        submodules directly at every call site. Overriding here means a call site
        cannot get it wrong.
        """
        self.encoder.train(mode)
        self.head.train(mode)
        self.teacher_encoder.eval()
        self.teacher_head.eval()
        self.training = mode
        return self

    def retokenize(self, classnames: list[str]) -> None:
        """Rebuild prompts on the student AND the teacher: text_features is not
        in state_dict, so an EMA update never carries the change across."""
        self.encoder.retokenize(classnames)
        self.teacher_encoder.retokenize(classnames)
