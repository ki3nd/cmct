"""LoRA layer and injection: rank schedule, SVD init, forward, EMA, save/load."""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from cmct.backbones.lora import (
    LoRALinear,
    MultiheadAttentionLoRA,
    apply_lora,
    freeze_except_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    rank_at,
    scaling_for,
)
from cmct.engine.ema import ema_update

RAMP = [2, 4, 6, 8, 10]


def mha(dim=64, heads=4, dtype=torch.float32):
    torch.manual_seed(0)
    return nn.MultiheadAttention(dim, heads).to(dtype)


# --- rank and scaling --------------------------------------------------------

def test_ascending_rank_schedule():
    assert [rank_at(i, 2, RAMP) for i in range(12)] == \
        [2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64]


def test_no_ramp_means_one_rank_everywhere():
    assert {rank_at(i, 4, None) for i in range(12)} == {4}
    assert {rank_at(i, 4, []) for i in range(12)} == {4}


def test_scaling_is_alpha_over_sqrt_rank():
    """Not alpha / rank -- the two differ by sqrt(rank), a factor of 8 at 64."""
    for rank in (2, 4, 8, 16, 32, 64):
        assert scaling_for(rank, 1.0) == pytest.approx(1.0 / math.sqrt(rank))
        if rank > 1:
            assert scaling_for(rank, 1.0) != pytest.approx(1.0 / rank)


def test_rank_and_scaling_reject_bad_input():
    with pytest.raises(ValueError, match="rank must be > 0"):
        rank_at(0, 0, RAMP)
    with pytest.raises(ValueError, match="block_index must be >= 0"):
        rank_at(-1, 2, RAMP)
    with pytest.raises(ValueError, match="rank must be > 0"):
        scaling_for(0, 1.0)


# --- SVD init ----------------------------------------------------------------

@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 1e-6), (torch.float16, 5e-4)])
def test_frozen_plus_delta_reconstructs_the_original(dtype, tolerance):
    """The residual is written back, so the sum returns the pretrained weight --
    to the precision of the frozen tensor. float16 lands near 2e-4, which is why
    equivalence tests cannot demand exact equality."""
    torch.manual_seed(0)
    linear = nn.Linear(64, 64)
    original = linear.weight.detach().clone().to(torch.float32)
    layer = LoRALinear(linear, rank=8, alpha=1.0, dropout=0.0, param_dtype=dtype)

    reconstructed = layer.weight.detach().to(torch.float32) + layer.delta().detach()
    error = (reconstructed - original).norm() / original.norm()
    assert float(error) < tolerance


def test_delta_is_not_zero_at_init():
    """Unlike standard LoRA, which zeroes B so the delta starts at zero."""
    layer = LoRALinear(nn.Linear(64, 64), rank=8, alpha=1.0, dropout=0.0)
    assert float(layer.delta().detach().norm()) > 0.0


def test_residual_is_written_back():
    torch.manual_seed(0)
    linear = nn.Linear(64, 64)
    original = linear.weight.detach().clone()
    layer = LoRALinear(linear, rank=8, alpha=1.0, dropout=0.0)
    assert not torch.allclose(layer.weight.detach(), original)


def test_init_takes_the_principal_components():
    """The top-r singular values, not the smallest r."""
    torch.manual_seed(0)
    linear = nn.Linear(64, 64)
    original = linear.weight.detach().clone().to(torch.float32)
    rank = 6
    layer = LoRALinear(linear, rank=rank, alpha=1.0, dropout=0.0)

    original_sv = torch.linalg.svdvals(original)
    delta_sv = torch.linalg.svdvals(layer.delta().detach())[:rank]
    assert torch.allclose(delta_sv, original_sv[:rank], rtol=1e-4)
    assert not torch.allclose(delta_sv, original_sv[-rank:].flip(0), rtol=1e-2)


def test_lora_factors_are_always_float32():
    for dtype in (torch.float16, torch.float32):
        layer = LoRALinear(nn.Linear(32, 32), rank=4, alpha=1.0, dropout=0.0,
                           param_dtype=dtype)
        assert layer.lora_A.dtype is torch.float32
        assert layer.lora_B.dtype is torch.float32
        assert layer.weight.dtype is dtype


def test_frozen_weight_takes_no_gradient():
    layer = LoRALinear(nn.Linear(32, 32), rank=4, alpha=1.0, dropout=0.0)
    assert layer.weight.requires_grad is False
    assert layer.lora_A.requires_grad and layer.lora_B.requires_grad


# --- forward -----------------------------------------------------------------

def test_forward_matches_the_formula_without_dropout():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(32, 16), rank=4, alpha=1.0, dropout=0.0).eval()
    x = torch.randn(5, 32)
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias) \
        + torch.nn.functional.linear(x, layer.delta())
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_dropout_is_active_only_in_train_mode():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(32, 32), rank=4, alpha=1.0, dropout=0.25)
    x = torch.randn(8, 32)

    layer.train()
    torch.manual_seed(1)
    first = layer(x)
    torch.manual_seed(2)
    second = layer(x)
    assert not torch.allclose(first, second)

    layer.eval()
    assert torch.allclose(layer(x), layer(x))


def test_dropout_touches_only_the_lora_branch():
    """With A zeroed the delta vanishes, so train and eval must agree -- proving
    the frozen path is computed before dropout."""
    layer = LoRALinear(nn.Linear(32, 32), rank=4, alpha=1.0, dropout=0.9)
    with torch.no_grad():
        layer.lora_A.zero_()
    x = torch.randn(8, 32)
    layer.train()
    trained = layer(x)
    layer.eval()
    assert torch.allclose(trained, layer(x), atol=1e-6)


def test_train_and_eval_never_touch_the_frozen_weight():
    """There is no merge/unmerge: the implementation this replaces folded the
    delta into the weight on eval() and locked out later updates."""
    layer = LoRALinear(nn.Linear(32, 32), rank=4, alpha=1.0, dropout=0.25)
    before = layer.weight.detach().clone()
    layer.eval()
    layer.train()
    layer.eval()
    assert torch.equal(layer.weight.detach(), before)


def test_gradient_reaches_the_factors_in_float32():
    layer = LoRALinear(nn.Linear(32, 32), rank=4, alpha=1.0, dropout=0.0,
                       param_dtype=torch.float16)
    layer(torch.randn(4, 32, dtype=torch.float16)).sum().backward()
    for param in (layer.lora_A, layer.lora_B):
        assert param.grad is not None
        assert param.grad.dtype is torch.float32
        assert torch.any(param.grad != 0)


# --- attention wrapper -------------------------------------------------------

def test_wrapper_matches_the_original_attention_output():
    """At init the delta reconstructs the weight, so the wrapper must reproduce
    nn.MultiheadAttention's output."""
    module = mha(dim=64, heads=4)
    torch.manual_seed(0)
    x = torch.randn(7, 3, 64)
    with torch.no_grad():
        reference, _ = module(x, x, x, need_weights=False)
    wrapper = MultiheadAttentionLoRA(module, rank=8, alpha=1.0, dropout=0.0).eval()
    with torch.no_grad():
        got, _ = wrapper(x, x, x)
    assert torch.allclose(got, reference, atol=1e-5), float((got - reference).abs().max())


def test_wrapper_honours_an_attention_mask():
    module = mha(dim=32, heads=4)
    torch.manual_seed(0)
    x = torch.randn(5, 2, 32)
    mask = torch.full((5, 5), float("-inf")).triu(1)
    with torch.no_grad():
        reference, _ = module(x, x, x, need_weights=False, attn_mask=mask)
    wrapper = MultiheadAttentionLoRA(module, rank=4, alpha=1.0, dropout=0.0).eval()
    with torch.no_grad():
        got, _ = wrapper(x, x, x, attn_mask=mask)
    assert torch.allclose(got, reference, atol=1e-5)


def test_only_named_projections_get_lora():
    wrapper = MultiheadAttentionLoRA(mha(), rank=4, alpha=1.0, dropout=0.0,
                                     params=("q", "k", "v"))
    assert isinstance(wrapper.q_proj, LoRALinear)
    assert isinstance(wrapper.k_proj, LoRALinear)
    assert isinstance(wrapper.v_proj, LoRALinear)
    assert not isinstance(wrapper.out_proj, LoRALinear)
    names = {n for n, _ in lora_parameters(wrapper)}
    assert not any("out_proj" in n for n in names)
    assert len(names) == 6


def test_unknown_projection_name_raises():
    with pytest.raises(ValueError, match="unknown LoRA targets"):
        MultiheadAttentionLoRA(mha(), rank=4, alpha=1.0, dropout=0.0,
                               params=("q", "z"))


# --- EMA: must keep updating -------------------------------------------------

def two_lora_copies(dtype=torch.float32):
    import copy
    student = LoRALinear(nn.Linear(64, 64), rank=8, alpha=1.0, dropout=0.0)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        for param in (teacher.lora_A, teacher.lora_B):
            param.mul_(1.05)
        if dtype is torch.float16:
            student.lora_A.data = student.lora_A.data.half()
            student.lora_B.data = student.lora_B.data.half()
            teacher.lora_A.data = teacher.lora_A.data.half()
            teacher.lora_B.data = teacher.lora_B.data.half()
    return teacher, student


def lora_gap(teacher, student):
    return sum(float((t.float() - s.float()).abs().sum())
               for t, s in zip((teacher.lora_A, teacher.lora_B),
                               (student.lora_A, student.lora_B), strict=True))


@pytest.mark.parametrize("momentum", [0.99, 0.996])
def test_ema_tracks_the_theoretical_rate_in_float32(momentum):
    """The reason lora_A and lora_B are float32.

    In exact arithmetic the gap after n steps is start * momentum ** n. float32
    stays within an order of magnitude of that over 1000 steps -- float16 ends
    four orders away: 0.64 of the original gap against a theoretical 4.3e-5. It does eventually stop
    moving -- but only once the gap is a negligible fraction of where it started,
    which is the difference from float16 (see the next test: stalls at step ~97
    with 64% of the gap still open).
    """
    steps = 1000
    teacher, student = two_lora_copies()
    start = lora_gap(teacher, student)
    froze_at, gap_at_freeze = None, None
    for step in range(steps):
        before = (teacher.lora_A.detach().clone(), teacher.lora_B.detach().clone())
        ema_update(teacher, student, momentum)
        if froze_at is None and all(
            torch.equal(b, a.detach()) for b, a in
            zip(before, (teacher.lora_A, teacher.lora_B), strict=True)
        ):
            froze_at, gap_at_freeze = step, lora_gap(teacher, student)
    end = lora_gap(teacher, student)

    assert end / start < 10 * momentum ** steps
    if froze_at is not None:
        assert gap_at_freeze / start < 1e-3, (
            f"stopped moving at step {froze_at} with "
            f"{100 * gap_at_freeze / start:.1f}% of the gap still open"
        )


def test_float16_factors_would_freeze():
    """Pins why float32 is not optional. Run directly on tensors, since
    ema_update refuses a float16 teacher outright."""
    torch.manual_seed(0)
    student = (torch.randn(512) * 0.02).half()
    teacher = (student.float() * 1.05).half()
    froze_at = None
    for step in range(1000):
        before = teacher.clone()
        teacher.mul_(0.996).add_(student, alpha=0.004)
        if froze_at is None and torch.equal(before, teacher):
            froze_at = step
    assert froze_at is not None and froze_at < 200


def test_ema_refuses_a_non_float32_teacher():
    teacher, student = two_lora_copies(torch.float16)
    with pytest.raises(TypeError, match="must be float32"):
        ema_update(teacher, student, 0.996)


def test_lora_parameters_selects_only_the_float32_factors():
    wrapper = MultiheadAttentionLoRA(mha().half(), rank=4, alpha=1.0, dropout=0.0,
                                     param_dtype=torch.float16)
    selected = dict(lora_parameters(wrapper))
    assert selected and all(p.dtype is torch.float32 for p in selected.values())
    others = {n: p for n, p in wrapper.named_parameters() if "lora_" not in n}
    assert others and all(p.dtype is torch.float16 for p in others.values())


# --- save / load -------------------------------------------------------------

def test_state_dict_holds_only_lora_keys():
    wrapper = MultiheadAttentionLoRA(mha(), rank=4, alpha=1.0, dropout=0.0)
    state = lora_state_dict(wrapper)
    assert state and all("lora_" in k for k in state)
    assert not any("weight" == k.split(".")[-1] for k in state)


def test_load_restores_exactly():
    wrapper = MultiheadAttentionLoRA(mha(), rank=4, alpha=1.0, dropout=0.0)
    saved = lora_state_dict(wrapper)
    frozen_before = wrapper.q_proj.weight.detach().clone()
    with torch.no_grad():
        wrapper.q_proj.lora_A.add_(1.0)
    load_lora_state_dict(wrapper, saved)
    for key, value in lora_state_dict(wrapper).items():
        assert torch.equal(value, saved[key]), key
    assert torch.equal(wrapper.q_proj.weight.detach(), frozen_before)


def test_load_rejects_a_mismatched_key_set():
    wrapper = MultiheadAttentionLoRA(mha(), rank=4, alpha=1.0, dropout=0.0)
    saved = lora_state_dict(wrapper)
    saved.pop(next(iter(saved)))
    with pytest.raises(ValueError, match="LoRA state mismatch"):
        load_lora_state_dict(wrapper, saved)


def test_load_rejects_a_different_rank():
    small = MultiheadAttentionLoRA(mha(), rank=4, alpha=1.0, dropout=0.0)
    large = MultiheadAttentionLoRA(mha(), rank=8, alpha=1.0, dropout=0.0)
    with pytest.raises(ValueError, match="different rank"):
        load_lora_state_dict(large, lora_state_dict(small), strict=False)


# --- injection into a CLIP ---------------------------------------------------

def test_apply_lora_replaces_the_selected_blocks(tiny_clip):
    created = apply_lora(tiny_clip, rank=2, alpha=1.0, dropout=0.25,
                         params=("q", "k", "v"), rank_ramp=None, positions="all",
                         encoders=("text", "vision"))
    assert len(created) == 2                      # tiny_clip has 1 block per encoder
    for encoder in (tiny_clip.transformer, tiny_clip.visual.transformer):
        assert isinstance(encoder.resblocks[0].attn, MultiheadAttentionLoRA)


def test_apply_lora_forward_is_unchanged_at_init(tiny_clip):
    import copy
    reference = copy.deepcopy(tiny_clip)
    torch.manual_seed(0)
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = reference.encode_image(images)
    apply_lora(tiny_clip, rank=4, alpha=1.0, dropout=0.0, params=("q", "k", "v"),
               rank_ramp=None, positions="all", encoders=("text", "vision"))
    with torch.no_grad():
        after = tiny_clip.encode_image(images)
    error = (after - before).norm() / before.norm()
    assert float(error) < 1e-4, float(error)


def test_freeze_except_lora(tiny_clip):
    apply_lora(tiny_clip, rank=2, alpha=1.0, dropout=0.0, params=("q", "k", "v"),
               rank_ramp=None, positions="all", encoders=("text", "vision"))
    freeze_except_lora(tiny_clip)
    trainable = {n for n, p in tiny_clip.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in n for n in trainable)


def test_apply_lora_rejects_an_unknown_position(tiny_clip):
    with pytest.raises(ValueError, match="unknown positions"):
        apply_lora(tiny_clip, rank=2, alpha=1.0, dropout=0.0, positions="everything")


def test_applying_twice_raises(tiny_clip):
    apply_lora(tiny_clip, rank=2, alpha=1.0, dropout=0.0, positions="all",
               encoders=("vision",))
    with pytest.raises(TypeError, match="already wrapped"):
        apply_lora(tiny_clip, rank=2, alpha=1.0, dropout=0.0, positions="all",
                   encoders=("vision",))
