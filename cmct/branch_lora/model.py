"""CLIP model wrappers for the LoRA branch.

`load_clip_to_cpu` downloads and builds the CLIP checkpoint; `Simple_TextEncoder`
runs CLIP's transformer over tokenized prompts; `LoraCLIP` wraps the image and
text encoders into a single cosine-similarity classifier; `FrozenTeacherCLIP`
is the EMA teacher variant that must stay in eval mode unconditionally.
"""

import torch
import torch.nn as nn

from cmct.clip import clip


def load_clip_to_cpu(backbone_name: str, backbone_path: str) -> nn.Module:
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, backbone_path)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class Simple_TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class LoraCLIP(nn.Module):
    """CLIP with a hand-written text encoder, returning logits and image features.

    `normalize_feat` controls ONLY the second (image_features) return value --
    logits always use the L2-normalized feature internally (that's what
    cosine-similarity classification requires), regardless of this flag.
    Default True is what this branch wants: its cosine-similarity design feeds
    the normalized feature to MK-MMD.
    """

    def __init__(self, classnames, clip_model, template: str):
        super().__init__()
        self.text_encoder = Simple_TextEncoder(clip_model)

        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        prompt_prefix = template
        prompts = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts = clip.tokenize(prompts)

    def forward(self, image, normalize_feat=True):
        text_features = self.text_encoder(self.tokenized_prompts.to(self.logit_scale.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features_norm @ text_features.t()

        return logits, (image_features_norm if normalize_feat else image_features)


class FrozenTeacherCLIP(LoraCLIP):
    """teacher_now never receives gradients (its LoRA is written directly via
    EMA, never via an optimizer step) and must always run deterministically,
    always recomputing LoRA live from the current
    lora_A/lora_B rather than a stale merged snapshot (see LinearLoRA.train()/
    lora_train() in lora/layers.py: calling .train(mode) on a LinearLoRA
    either bakes the current LoRA delta into the frozen weight once and locks
    out future updates, or re-enables dropout -- both wrong for a teacher).

    Overriding train() here to unconditionally force every submodule's
    `.training = False`, regardless of who calls .train()/.eval() or what
    mode is requested, makes this hold no matter what -- independent of
    tracking every call site (set_model_mode fires at the start of
    every epoch and after every test()) or any call site added later."""

    def train(self, mode=True):
        for m in self.modules():
            m.training = False
        return self
