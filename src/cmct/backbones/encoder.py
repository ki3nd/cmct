"""CLIP wrapped as a feature extractor with a cosine-similarity head.

Two outputs from one forward pass, and the difference is load-bearing:

    features(x)          raw image features, NOT normalized
    cosine_logits(feat)  logit_scale * normalize(feat) @ text_features.T

A learned classifier head consumes the raw features; the cosine branch
normalizes internally. Feeding normalized features to the classifier head would
change the input scale its BatchNorm1d sees.
"""
from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from cmct.backbones.clip import tokenize
from cmct.backbones.clip.model import CLIP


class ClipEncoder(nn.Module):
    """`text` and `text_features` are NON-PERSISTENT buffers, so they move with
    the module but never appear in state_dict. Three consequences, all
    deliberate:

    - loading a checkpoint does not restore them; they are rebuilt from
      classnames
    - the EMA update walks state_dict, so it never touches them
    - changing classnames after construction works, but must be done on every
      copy of this module (see `retokenize`)

    Non-persistent buffers rather than plain attributes because `nn.Module.to()`
    moves only parameters and buffers: as plain attributes they stayed on the CPU
    while the weights went to the GPU, and the first cosine-similarity matmul
    raised a device mismatch. The codebase this comes from has the same
    non-buffer design and solves it by pinning to CUDA at construction
    (`clip.load(device="cuda")`, `tokenize(...).cuda()`,
    `encode_text().detach().cuda()`), which also makes it unable to run on CPU.

    `text_features` is a snapshot computed once here, not recomputed per
    forward. That is valid only because the text tower is frozen: it is not in
    any optimizer param group (see VlpModel.param_groups). Putting the text
    tower into an optimizer would change its weights while this snapshot
    silently kept its value from construction time.
    """

    def __init__(self, clip_model: CLIP, classnames: list[str], template: str) -> None:
        super().__init__()
        self.model = clip_model
        self.template = template
        self.register_buffer("text", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("text_features", torch.empty(0), persistent=False)
        self.retokenize(classnames)

    @property
    def feature_dim(self) -> int:
        """Read off the model rather than looked up by architecture name, so it
        cannot disagree with the checkpoint. 512 for ViT-B/16 and RN101, 1024
        for RN50."""
        return self.model.visual.output_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.model.visual.conv1.weight.dtype if hasattr(self.model.visual, "conv1") \
            else next(self.model.visual.parameters()).dtype

    @torch.no_grad()
    def retokenize(self, classnames: list[str]) -> None:
        """Rebuild the prompts and their cached embeddings.

        Must be called on each copy of this module separately: `text` and
        `text_features` are not in state_dict, so neither deepcopy-then-mutate
        nor an EMA update propagates a change made to one copy.

        Assigning to a registered buffer keeps it a buffer, so the tensors stay
        movable by `.to()` after every retokenize, not just the first.
        """
        device = next(self.model.parameters()).device
        self.text = tokenize([self.template.format(c) for c in classnames]).to(device)
        features = self.model.encode_text(self.text).detach()
        self.text_features = features / features.norm(dim=1, keepdim=True)

    def features(self, images: Tensor) -> Tensor:
        """Raw image features, unnormalized."""
        return self.model.encode_image(images)

    def cosine_logits(self, features: Tensor) -> Tensor:
        normalized = features / features.norm(dim=1, keepdim=True)
        return self.model.logit_scale.exp() * normalized @ self.text_features.t()

    def forward(self, images: Tensor) -> Tensor:
        return self.cosine_logits(self.features(images))

    def visual_parameters(self) -> Iterator[nn.Parameter]:
        """The only part of CLIP this branch trains. The text tower and
        logit_scale are deliberately absent."""
        return self.model.visual.parameters()

    def freeze_batchnorm(self) -> None:
        """Put every BatchNorm into eval mode.

        A no-op on ViT, which has none. It starts mattering the moment the
        backbone is a ResNet, so it runs unconditionally rather than being
        skipped for the architecture in use today.
        """
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
