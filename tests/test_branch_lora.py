
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


def test_text_lora_can_be_switched_off(clip_model_factory):
    """With lora.text false, every injected layer belongs to the vision tower.

    Guards a failure that is otherwise silent: leaving text LoRA in would keep
    training the text tower while the design says the prompt is the only
    text-side adaptation, and nothing would crash.
    """
    from cmct.branch_lora.model import LoraCLIP
    from cmct.branch_lora.lora.apply import apply_lora

    kwargs = dict(backbone_name="ViT-B/16", position="all", params=["q", "k", "v"],
                  r=2, alpha=1, dropout=0.25, rank_ramp=[2, 4, 6, 8, 10])

    # Two independent model instances: apply_lora mutates a model's submodules
    # in place, so this needs a fresh build per model rather than a shared one.
    clip_model = clip_model_factory()
    model = LoraCLIP(["dog", "cat"], clip_model, template="a photo of a {}.", n_ctx=4, learnable=True)
    layers_vision_only = apply_lora(model, **kwargs, text=False)

    clip_model2 = clip_model_factory()
    model2 = LoraCLIP(["dog", "cat"], clip_model2, template="a photo of a {}.", n_ctx=4, learnable=True)
    layers_both = apply_lora(model2, **kwargs, text=True)

    # Both towers are 12 blocks, so injecting into one gives exactly half.
    assert len(layers_both) == 2 * len(layers_vision_only)

    # The load-bearing assertion: no injected layer sits in the text tower.
    text_layer_ids = {id(m) for block in model.text_encoder.transformer.resblocks
                      for m in block.children()}
    assert not any(id(layer) in text_layer_ids for layer in layers_vision_only)
    assert not any("lora_" in n for n, _ in model.text_encoder.named_parameters())


def test_the_teacher_copy_covers_the_prompt_context():
    """Guards a silent divergence: the copy filters on "lora_", which the
    context's name does not match, so without this the teacher would keep its
    initial context forever while the student's moved -- two different text
    embeddings, no error.
    """
    import torch
    import torch.nn as nn

    from cmct.branch_lora.ema import copy_lora_params, ema_update_lora_params

    class Fake(nn.Module):
        def __init__(self, ctx, lora):
            super().__init__()
            self.prompt_learner = nn.Module()
            self.prompt_learner.ctx = nn.Parameter(torch.full((4, 8), ctx))
            self.lora_A = nn.Parameter(torch.full((2, 8), lora))

    student, teacher = Fake(1.0, 3.0), Fake(0.0, 0.0)
    copy_lora_params(student, teacher)
    assert torch.allclose(teacher.prompt_learner.ctx, torch.full((4, 8), 1.0))
    assert torch.allclose(teacher.lora_A, torch.full((2, 8), 3.0))

    # momentum 0.0 is the EMA-off setting the shipped config uses: a hard copy.
    student2 = Fake(5.0, 7.0)
    ema_update_lora_params(teacher, student2, 0.0)
    assert torch.allclose(teacher.prompt_learner.ctx, torch.full((4, 8), 5.0))
    assert torch.allclose(teacher.lora_A, torch.full((2, 8), 7.0))
