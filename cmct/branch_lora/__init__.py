"""Branch 1: CLIP with depth-ramped LoRA, plus its EMA teacher."""

from .ema import copy_lora_params, ema_update_lora_params
from .model import FrozenTeacherCLIP, LoraCLIP, load_clip_to_cpu

__all__ = [
    "FrozenTeacherCLIP",
    "LoraCLIP",
    "copy_lora_params",
    "ema_update_lora_params",
    "load_clip_to_cpu",
]
