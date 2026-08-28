"""Parity against old-cmct/loralib, imported and executed.

The reference layer is constructed on the same nn.MultiheadAttention our wrapper
receives, so this compares the SVD initialization itself rather than a
re-derivation of it.

lora_A and lora_B are NOT compared directly: torch.linalg.svd does not fix the
sign of its factors, so two runs can produce A, B and -A, -B with an identical
product. scaling * B @ A and the frozen residual are sign-invariant, and they are
what the forward pass uses.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from cmct.backbones.lora import MultiheadAttentionLoRA, rank_at, scaling_for

OLD_CMCT = Path("/home/pc1175/DA-Research/old-cmct")
SHADOWED = ("loralib", "utils")

pytestmark = pytest.mark.skipif(
    not (OLD_CMCT / "loralib" / "layers.py").is_file(),
    reason=f"no reference implementation at {OLD_CMCT}",
)


def import_reference_layer():
    """Import old-cmct's PlainMultiheadAttentionLoRA in an isolated sys.path and
    sys.modules scope, so its `loralib` and `utils` packages do not stay behind
    and shadow anything later in the session."""
    def shadowed(name: str) -> bool:
        return name in SHADOWED or name.split(".", 1)[0] in SHADOWED

    saved_modules = {k: v for k, v in sys.modules.items() if shadowed(k)}
    for key in saved_modules:
        del sys.modules[key]
    saved_path = sys.path[:]
    sys.path.insert(0, str(OLD_CMCT))
    try:
        return importlib.import_module("loralib.layers").PlainMultiheadAttentionLoRA
    finally:
        sys.path[:] = saved_path
        for key in [k for k in sys.modules if shadowed(k)]:
            del sys.modules[key]
        sys.modules.update(saved_modules)


def paired_attention(dim: int, heads: int, seed: int = 0):
    """Two identical fp16 attention modules -- the reference mutates the one it
    wraps, so each side needs its own."""
    torch.manual_seed(seed)
    reference = nn.MultiheadAttention(dim, heads).half()
    torch.manual_seed(seed)
    ours = nn.MultiheadAttention(dim, heads).half()
    assert torch.equal(reference.in_proj_weight, ours.in_proj_weight)
    return reference, ours


@pytest.mark.parametrize("rank", [2, 4, 8, 16, 32, 64])
def test_delta_and_residual_match_the_reference(rank):
    Reference = import_reference_layer()
    ref_mha, our_mha = paired_attention(dim=512, heads=8)

    reference = Reference(ref_mha, enable_lora={"q", "k", "v"}, r=rank,
                          lora_alpha=1, dropout_rate=0.25)
    ours = MultiheadAttentionLoRA(our_mha, rank=rank, alpha=1.0, dropout=0.25,
                                  params=("q", "k", "v"),
                                  param_dtype=torch.float16)

    for name in ("q", "k", "v"):
        ref_proj = getattr(reference, f"{name}_proj")
        our_proj = getattr(ours, f"{name}_proj")

        assert ref_proj.scaling == pytest.approx(our_proj.scaling), name

        ref_delta = (ref_proj.scaling
                     * (ref_proj.w_lora_B.detach().float()
                        @ ref_proj.w_lora_A.detach().float()))
        our_delta = our_proj.delta().detach().float()
        assert torch.allclose(ref_delta, our_delta, atol=1e-4), (
            name, float((ref_delta - our_delta).abs().max())
        )

        assert torch.allclose(ref_proj.weight.detach().float(),
                              our_proj.weight.detach().float(), atol=1e-4), name


def test_out_proj_is_untouched_by_both():
    Reference = import_reference_layer()
    ref_mha, our_mha = paired_attention(dim=256, heads=8)
    reference = Reference(ref_mha, enable_lora={"q", "k", "v"}, r=8,
                          lora_alpha=1, dropout_rate=0.25)
    ours = MultiheadAttentionLoRA(our_mha, rank=8, alpha=1.0, dropout=0.25,
                                  params=("q", "k", "v"),
                                  param_dtype=torch.float16)
    assert not any("proj.w_lora" in n for n, _ in reference.named_parameters()
                   if n.startswith("proj."))
    assert torch.allclose(reference.proj.weight.detach().float(),
                          ours.out_proj.weight.detach().float(), atol=1e-4)


def test_trainable_parameter_count_matches():
    """Counted over all 12 blocks of both towers at the real dimensions, with the
    ascending rank schedule."""
    Reference = import_reference_layer()
    total_reference = 0
    total_ours = 0
    for dim in (512, 768):                      # text tower, vision tower
        for block in range(12):
            rank = rank_at(block, 2, [2, 4, 6, 8, 10])
            ref_mha, our_mha = paired_attention(dim=dim, heads=8, seed=block)
            reference = Reference(ref_mha, enable_lora={"q", "k", "v"}, r=rank,
                                  lora_alpha=1, dropout_rate=0.25)
            ours = MultiheadAttentionLoRA(our_mha, rank=rank, alpha=1.0,
                                          dropout=0.25, params=("q", "k", "v"),
                                          param_dtype=torch.float16)
            total_reference += sum(p.numel() for n, p in reference.named_parameters()
                                   if "lora_" in n)
            total_ours += sum(p.numel() for n, p in ours.named_parameters()
                              if "lora_" in n)
    assert total_ours == total_reference
    assert total_ours == 1_935_360


def test_scaling_formula_matches_the_reference():
    Reference = import_reference_layer()
    for rank in (2, 8, 64):
        ref_mha, _ = paired_attention(dim=128, heads=8)
        reference = Reference(ref_mha, enable_lora={"q"}, r=rank, lora_alpha=1,
                              dropout_rate=0.0)
        assert reference.q_proj.scaling == pytest.approx(scaling_for(rank, 1.0))


def test_forward_matches_the_reference_in_eval_mode():
    """Both sides in eval so LoRA dropout is off and the comparison is
    deterministic."""
    Reference = import_reference_layer()
    ref_mha, our_mha = paired_attention(dim=512, heads=8)
    """Both sides in eval so LoRA dropout is off and the comparison is
    deterministic.

    Bounded by float16's relative epsilon rather than by the weight-level
    tolerance: this runs a whole attention block -- three projections, a softmax,
    two matmuls, an out projection -- in float16, so rounding accumulates across
    all of them. The weight-level parity tests above are what would catch an
    algorithm difference, and they hold at 1e-4.
    """
    Reference = import_reference_layer()
    ref_mha, our_mha = paired_attention(dim=512, heads=8)
    reference = Reference(ref_mha, enable_lora={"q", "k", "v"}, r=8,
                          lora_alpha=1, dropout_rate=0.25)
    # The reference's train() override (see loralib/layers.py) does not return
    # self, so `.eval()` cannot be chained the way it can on a plain nn.Module --
    # it still flips `self.training` in place, but the expression evaluates to
    # None. Call it as a statement and keep the reference object itself.
    reference.eval()
    ours = MultiheadAttentionLoRA(our_mha, rank=8, alpha=1.0, dropout=0.25,
                                  params=("q", "k", "v"),
                                  param_dtype=torch.float16).eval()
    torch.manual_seed(0)
    x = torch.randn(9, 2, 512).half()
    with torch.no_grad():
        expected, _ = reference(x, x, x, need_weights=False)
        got, _ = ours(x, x, x)
    error = (got.float() - expected.float()).norm() / expected.float().norm()
    assert float(error) < torch.finfo(torch.float16).eps, float(error)


def test_forward_matches_the_unwrapped_attention_in_float32():
    """float32 counterpart to the float16 forward test above, where rounding
    noise cannot hide a real divergence.

    The reference's `PlainMultiheadAttentionLoRA.__init__` hardcodes `.half()`
    on its q/k/v/out projections (old-cmct/loralib/layers.py:505-508), so it
    cannot be constructed in float32 -- there is no reference to compare
    against here. Instead this checks the property directly on our own
    implementation: the SVD split reconstructs the wrapped weight exactly in
    float32 (`frozen + scaling * B @ A == W_orig`, up to float32 rounding), so
    our wrapper's output must match the *unwrapped* nn.MultiheadAttention's
    output just as tightly.
    """
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(512, 8)
    torch.manual_seed(0)
    same_mha = nn.MultiheadAttention(512, 8)
    assert torch.equal(mha.in_proj_weight, same_mha.in_proj_weight)

    ours = MultiheadAttentionLoRA(same_mha, rank=8, alpha=1.0, dropout=0.0,
                                  params=("q", "k", "v"),
                                  param_dtype=torch.float32).eval()
    mha.eval()
    torch.manual_seed(0)
    x = torch.randn(9, 2, 512)
    with torch.no_grad():
        expected, _ = mha(x, x, x, need_weights=False)
        got, _ = ours(x, x, x)
    error = (got - expected).norm() / expected.norm()
    assert float(error) < 1e-6, float(error)
