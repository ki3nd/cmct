"""Build a CLIP model from a local checkpoint.

`dtype` is required and has no default. In the two codebases this project
replaces, that single choice was the only difference between their otherwise
byte-identical copies of `model.py` -- one called `convert_weights` (fp16), the
other had the call commented out (fp32). Here the caller has to say which.
"""
from typing import Literal

import torch

from cmct.backbones.clip.model import CLIP, build_model


def load_state_dict(checkpoint: str) -> dict:
    """Read a CLIP checkpoint that may be either a TorchScript archive (what
    OpenAI ships, e.g. ViT-B-16.pt) or a plain state_dict."""
    try:
        return torch.jit.load(checkpoint, map_location="cpu").eval().state_dict()
    except RuntimeError:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return obj.state_dict() if hasattr(obj, "state_dict") else obj


def load_clip(checkpoint: str, dtype: Literal["fp16", "fp32"]) -> CLIP:
    """Instantiate CLIP from `checkpoint` at the requested precision.

    "fp16" applies `convert_weights` (upstream behavior); "fp32" leaves every
    parameter in float32.
    """
    if dtype not in ("fp16", "fp32"):
        raise ValueError(f"dtype must be 'fp16' or 'fp32', got {dtype!r}")

    state_dict = load_state_dict(checkpoint)
    for key in ("input_resolution", "context_length", "vocab_size"):
        state_dict.pop(key, None)

    model = build_model(state_dict)          # build_model calls convert_weights
    if dtype == "fp32":
        model.float()
    return model.eval()
