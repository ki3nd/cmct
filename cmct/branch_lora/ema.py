"""EMA (Mean-Teacher) updates over LoRA parameters.

The momentum is a single float rather than a per-key function: nothing in
this codebase varies momentum by depth, so a uniform momentum is all that
is needed.
"""

import torch


def _lora_param_items(model):
    return [(k, v) for k, v in model.state_dict().items() if "lora_" in k]


@torch.no_grad()
def copy_lora_params(src_model, dst_model) -> None:
    dst_state = dst_model.state_dict()
    for k, v in _lora_param_items(src_model):
        dst_state[k].copy_(v)


@torch.no_grad()
def ema_update_lora_params(ema_model, src_model, momentum: float) -> None:
    """LoRA A/B are fp32 (see lora/layers.py's _half_except_lora), so this
    plain in-place EMA doesn't underflow the way it would on fp16 leaves."""
    ema_state = ema_model.state_dict()
    for k, v in _lora_param_items(src_model):
        ema_state[k].mul_(momentum).add_(v, alpha=1.0 - momentum)
