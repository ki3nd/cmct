
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


def test_lora_clip_tokenizes_the_prompt_list_verbatim(clip_weights):
    """Both branches must tokenize IDENTICAL strings, so branch_mlp sees only the
    text tower's LoRA adaptation when it reads these embeddings -- never a
    prompt change. This branch used to build its prompts from a
    "a photo of a {}." template, which differed from branch_mlp's list on
    officehome; the guard is that the list now goes through verbatim.
    """
    import os

    import torch

    from cmct.branch_lora.model import LoraCLIP, load_clip_to_cpu
    from cmct.clip import clip

    prompts = ["an image of a alarm clock", "an image of a fan"]
    clip_model = load_clip_to_cpu("ViT-B/16", os.path.dirname(clip_weights)).float()
    model = LoraCLIP(prompts, clip_model)

    assert torch.equal(model.tokenized_prompts, clip.tokenize(prompts))

    with torch.no_grad():
        feats = model.text_features()
    assert feats.shape == (2, clip_model.visual.output_dim)
    torch.testing.assert_close(
        feats.norm(dim=-1), torch.ones(2), rtol=1e-4, atol=1e-4
    )
