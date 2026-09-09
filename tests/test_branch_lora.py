
import torch

from cmct.branch_lora.lora.apply import compute_rank
from tests.conftest import load_fixture


def test_rank_ramp_matches_the_frozen_baseline():
    fx = load_fixture("rank_ramp.json")
    got = [compute_rank(i, fx["r"], fx["ramp"]) for i in range(12)]
    assert got == fx["expected"]


def test_rank_ramp_disabled_is_flat():
    assert [compute_rank(i, 2, []) for i in range(12)] == [2] * 12


def test_ema_update_is_convex_combination():
    from cmct.branch_lora.ema import ema_update_lora_params

    class M(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.full((2, 2), value))

    ema, src = M(0.0), M(1.0)
    ema_update_lora_params(ema, src, 0.99)
    assert torch.allclose(ema.lora_A, torch.full((2, 2), 0.01))


def test_text_lora_can_be_switched_off(clip_weights):
    """With lora.text false, every injected layer belongs to the vision tower.

    Guards a failure that is otherwise silent: leaving text LoRA in would keep
    training the text tower while the design says the prompt is the only
    text-side adaptation, and nothing would crash.
    """
    import os
    import torch

    from cmct.branch_lora.model import LoraCLIP, load_clip_to_cpu
    from cmct.branch_lora.lora.apply import apply_lora

    kwargs = dict(backbone_name="ViT-B/16", position="all", params=["q", "k", "v"],
                  r=2, alpha=1, dropout=0.25, rank_ramp=[2, 4, 6, 8, 10])

    # Derive the directory from the fixture's resolved path so the fixture's
    # skip-guard actually gates what we load.
    weights_dir = os.path.dirname(clip_weights)

    # LoraCLIP's CURRENT signature -- template positional, no prompt arguments.
    # Task 3 changes it to keywords and updates this one call.
    clip_model = load_clip_to_cpu("ViT-B/16", weights_dir)
    model = LoraCLIP(["dog", "cat"], clip_model, "a photo of a {}.")
    layers_vision_only = apply_lora(model, **kwargs, text=False)

    clip_model2 = load_clip_to_cpu("ViT-B/16", weights_dir)
    model2 = LoraCLIP(["dog", "cat"], clip_model2, "a photo of a {}.")
    layers_both = apply_lora(model2, **kwargs, text=True)

    # Both towers are 12 blocks, so injecting into one gives exactly half.
    assert len(layers_both) == 2 * len(layers_vision_only)

    # The load-bearing assertion: no injected layer sits in the text tower.
    text_layer_ids = {id(m) for block in model.text_encoder.transformer.resblocks
                      for m in block.children()}
    assert not any(id(layer) in text_layer_ids for layer in layers_vision_only)
    assert not any("lora_" in n for n, _ in model.text_encoder.named_parameters())
