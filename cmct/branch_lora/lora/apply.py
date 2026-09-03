"""LoRA injection for CLIP's ViT encoders."""

import torch
import torch.nn as nn

from .layers import PlainMultiheadAttentionLoRA

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def apply_lora(model, *, backbone_name, position, params, r, alpha, dropout, rank_ramp):
    """Replace the attention modules of both CLIP encoders with LoRA versions.

    Returns the injected PlainMultiheadAttentionLoRA layers, text encoder first
    then vision encoder, in block order.
    """
    list_lora_layers = []
    _apply_text_lora(
        model, list_lora_layers,
        position=position, params=params, r=r, alpha=alpha,
        dropout=dropout, rank_ramp=rank_ramp,
    )
    _apply_vit_lora(
        model, list_lora_layers,
        backbone_name=backbone_name, position=position, params=params, r=r,
        alpha=alpha, dropout=dropout, rank_ramp=rank_ramp,
    )
    return list_lora_layers


def compute_rank(block_idx: int, r: int, rank_ramp: list) -> int:
    """LoRA rank grows with depth. A block clearing rank_ramp[i] uses r * 2**(i+1).

    With r=2 and rank_ramp=[2, 4, 6, 8, 10] this gives, for blocks 0..11:
    [2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64].
    """
    if not rank_ramp:
        return r

    ramp = rank_ramp
    
    for i, ramp_start_index in reversed(list(enumerate(ramp))):
        if block_idx >= ramp_start_index:
            rank_multiplier = 2**(i + 1)  
            return r * rank_multiplier
    return r

def _apply_text_lora(model, list_lora_layers, *, position, params, r, alpha, dropout, rank_ramp):
    indices = INDEX_POSITIONS_TEXT.get(position)
    if indices is None:
        raise KeyError(f"Unknown text position: {position}")
    text_encoder = model.text_encoder.transformer
    for i, block in enumerate(text_encoder.resblocks):
        if i not in indices:
            continue
        rank = compute_rank(i, r, rank_ramp)
        for name, submodule in block.named_children():
            if isinstance(submodule, nn.MultiheadAttention):
                new_multi_head_lora = PlainMultiheadAttentionLoRA(
                    submodule,
                    enable_lora={*params},
                    r=rank,
                    lora_alpha=alpha,
                    dropout_rate=dropout
                )
                setattr(block, name, new_multi_head_lora)
                list_lora_layers.append(new_multi_head_lora)

def _apply_vit_lora(model, list_lora_layers, *, backbone_name, position, params, r, alpha, dropout, rank_ramp):
    backbone = backbone_name
    positions = INDEX_POSITIONS_VISION.get(backbone, {})
    indices = positions.get(position)
    if indices is None:
        raise KeyError(f"Unknown vision position '{position}' for backbone '{backbone}'")
    vision_encoder = model.image_encoder.transformer
    for i, block in enumerate(vision_encoder.resblocks):
        if i not in indices:
            continue
        rank = compute_rank(i, r, rank_ramp)
        for name, submodule in block.named_children():
            if isinstance(submodule, nn.MultiheadAttention):
                new_multi_head_lora = PlainMultiheadAttentionLoRA(
                    submodule,
                    enable_lora={*params},
                    r=rank,
                    lora_alpha=alpha,
                    dropout_rate=dropout
                )
                setattr(block, name, new_multi_head_lora)
                list_lora_layers.append(new_multi_head_lora)


def save_lora(list_lora_layers, save_dir, filename, *, r, alpha, params):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        for name, param in layer.state_dict().items():
            if 'lora' in name:
                layer_weights[name] = param.detach().cpu()
        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': r,
        'alpha': alpha,
        'encoder': 'both',
        'params': params,
        'position': 'all'
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }
    save_path = f'{save_dir}/{filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')
