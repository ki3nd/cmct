"""Branch 1's model: a LoRA CLIP with a cosine head, its EMA teacher, and a
frozen zero-shot reference.

Text features are recomputed on every forward. LoRA is applied to the text tower,
so 774,144 of the 1,935,360 trainable parameters at the reference configuration
live there; a cached text embedding carries no graph, and those parameters would
sit in the optimizer receiving no gradient at all -- silently. The implementation
this comes from calls its text encoder inside forward for the same reason.

The teacher is a full deepcopy, backbone included, so train() on one side cannot
reach the other. Only LoRA parameters are EMA'd: they are float32, and a float16
EMA at this momentum stops moving within ~100 steps.

The teacher stays in eval permanently, so its LoRA dropout is off and its
pseudo-labels are deterministic. The student trains with dropout active on both
towers, which is why two forward calls in one step produce different text
embeddings -- and why the text embedding cannot be reused across them.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator

import torch
from torch import Tensor, nn

from cmct.backbones.clip import tokenize
from cmct.backbones.clip.model import CLIP
from cmct.backbones.encoder import ClipEncoder
from cmct.backbones.lora import (
    apply_lora,
    freeze_except_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
)

TEMPLATE = "a photo of a {}."
"""Branch 1's prompt template, with the trailing period. Branch 2 uses
"an image of a {}" without one; the two token sequences differ, so the branches
see different text embeddings. That is deliberate -- each keeps its own."""


class LoraModel(nn.Module):
    def __init__(self, clip_model: CLIP, classnames: list[str], *, rank: int,
                 alpha: float, dropout: float, params: tuple[str, ...],
                 rank_ramp: list[int] | None, positions: str,
                 encoders: tuple[str, ...], param_dtype: torch.dtype,
                 template: str = TEMPLATE) -> None:
        super().__init__()
        self.template = template
        self.classnames = list(classnames)
        self.num_classes = len(classnames)

        clip_dtype = clip_model.visual.conv1.weight.dtype
        if clip_dtype is not param_dtype:
            raise TypeError(
                f"param_dtype is {param_dtype} but the CLIP handed in is "
                f"{clip_dtype}. These come from one config field "
                f"(branches[].backbone.dtype) through two separate paths -- "
                f"load_clip() for the whole model and param_dtype for the LoRA "
                f"layers -- so a mismatch means one of them was not updated. "
                f"Left alone it fails mid-forward with a bare dtype error."
            )

        self.student = clip_model
        apply_lora(self.student, rank=rank, alpha=alpha, dropout=dropout,
                   params=params, rank_ramp=rank_ramp, positions=positions,
                   encoders=encoders, param_dtype=param_dtype)
        freeze_except_lora(self.student)

        prompts = [template.format(name) for name in self.classnames]
        self.register_buffer("tokenized_prompts", tokenize(prompts), persistent=False)

        # A full copy, so the two never share a module and train()/eval() on one
        # cannot reach the other.
        self.teacher = copy.deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()
        self.teacher_updates = 0

        self.zero_shot: ClipEncoder | None = None
        self.train(True)

    # --- text side ----------------------------------------------------------

    def text_features(self, model: nn.Module | None = None) -> Tensor:
        """Recomputed every call, never cached: the text tower carries LoRA."""
        target = self.student if model is None else model
        prompts = self.tokenized_prompts.to(target.logit_scale.device)
        features = target.encode_text(prompts)
        return features / features.norm(dim=-1, keepdim=True)

    # --- student ------------------------------------------------------------

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        text = self.text_features()
        raw = self.student.encode_image(images.type(text.dtype))
        normalized = raw / raw.norm(dim=-1, keepdim=True)
        logits = self.student.logit_scale.exp() * normalized @ text.t()
        return logits, normalized

    def logits(self, images: Tensor) -> Tensor:
        return self.forward(images)[0]

    def features(self, images: Tensor, normalize: bool = True) -> Tensor:
        """Two behaviours behind one flag would be one place to pass the wrong
        thing without a type error, so callers name it."""
        raw = self.student.encode_image(images.type(self.student.dtype))
        return raw / raw.norm(dim=-1, keepdim=True) if normalize else raw

    # --- teacher ------------------------------------------------------------

    @torch.no_grad()
    def teacher_logits(self, images: Tensor) -> Tensor:
        text = self.text_features(self.teacher)
        raw = self.teacher.encode_image(images.type(text.dtype))
        normalized = raw / raw.norm(dim=-1, keepdim=True)
        return self.teacher.logit_scale.exp() * normalized @ text.t()

    def ema_update(self, momentum: float) -> None:
        """LoRA parameters only.

        The first update replaces them outright, whatever momentum is passed:
        until then the teacher holds the student's initialization, and at a high
        momentum that would linger for thousands of steps.
        """
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be within [0, 1], got {momentum}")
        if self.teacher_updates == 0:
            momentum = 0.0

        teacher_state = self.teacher.state_dict()
        with torch.no_grad():
            for name, param in lora_parameters(self.student):
                target = teacher_state[name]
                if target.dtype is not torch.float32:
                    raise TypeError(
                        f"teacher tensor '{name}' is {target.dtype}, but an EMA "
                        f"teacher must be float32: in float16 the update rounds "
                        f"to zero and the teacher freezes"
                    )
                target.mul_(momentum).add_(param.detach().to(torch.float32),
                                           alpha=1.0 - momentum)
        self.teacher_updates += 1

    @property
    def teacher_is_initialized(self) -> bool:
        return self.teacher_updates > 0

    # --- zero-shot reference ------------------------------------------------

    def attach_zero_shot(self, clip_model: CLIP) -> None:
        """A CLIP with no LoRA, used as the self-reference during warmup.

        Its text tower is frozen, so ClipEncoder's cached text features are
        correct here -- unlike the student's. It is built from the same class
        names and the same template, so its prompts are the student's; a test
        asserts the two token tensors are equal rather than this code forcing it.

        `clip_model` must be a FRESH CLIP, not a copy of the student: apply_lora
        mutates in place, so a deepcopy of the model handed to __init__ carries
        the student's LoRA and would make this a second trainable branch rather
        than a zero-shot reference. Rejected outright, because the resulting
        pseudo-labels would look plausible and be wrong.
        """
        already_adapted = [name for name, _ in lora_parameters(clip_model)]
        if already_adapted:
            raise ValueError(
                f"attach_zero_shot needs a CLIP with no LoRA, but this one has "
                f"{len(already_adapted)} LoRA parameters (first: "
                f"{already_adapted[0]}). Load a fresh checkpoint rather than "
                f"copying the student."
            )
        encoder = ClipEncoder(clip_model, self.classnames, self.template)
        for param in encoder.parameters():
            param.requires_grad_(False)
        encoder.eval()
        self.zero_shot = encoder

    @property
    def has_zero_shot(self) -> bool:
        return self.zero_shot is not None

    @torch.no_grad()
    def zero_shot_logits(self, images: Tensor) -> Tensor:
        if self.zero_shot is None:
            raise RuntimeError(
                "the zero-shot reference is not attached (or was released); call "
                "attach_zero_shot first"
            )
        return self.zero_shot(images)

    def release_zero_shot(self) -> None:
        """Free it once warmup ends -- it is a whole extra CLIP on the device and
        nothing reads it afterwards."""
        self.zero_shot = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- plumbing -----------------------------------------------------------

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for _, param in lora_parameters(self.student):
            yield param

    def param_groups(self, lr: float) -> list[dict]:
        """One group: every LoRA factor, in both towers, at one learning rate.
        The reference uses a single group too -- no per-tower multiplier."""
        return [{"params": list(self.trainable_parameters()), "lr": lr}]

    def train(self, mode: bool = True) -> LoraModel:
        """Set the student explicitly and hold the teacher in eval.

        The reference never called train() at all: its LoRA modules happened to
        be in train mode because freshly constructed modules default to it, while
        the CLIP around them arrived in eval. That worked, but only until someone
        called eval() once.
        """
        self.student.train(mode)
        self.teacher.eval()
        if self.zero_shot is not None:
            self.zero_shot.eval()
        self.training = mode
        return self

    def lora_state_dict(self) -> dict[str, Tensor]:
        return lora_state_dict(self.student)

    def load_lora_state_dict(self, state: dict[str, Tensor],
                             strict: bool = True) -> None:
        load_lora_state_dict(self.student, state, strict=strict)
