"""Confidence-masked pseudo-label cross-entropy."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cmct.losses import pass_fraction, pseudo_label_ce

CLASSES = 4

MIXED = torch.tensor([
    [0.90, 0.05, 0.03, 0.02],      # passes 0.85
    [0.40, 0.30, 0.20, 0.10],      # fails
    [0.95, 0.03, 0.01, 0.01],      # passes
    [0.50, 0.50, 0.00, 0.00],      # fails
])


def sharp(labels, temperature=0.05):
    return F.softmax(F.one_hot(labels, CLASSES).float() / temperature, dim=-1)


def test_mask_mode_averages_over_the_samples_that_pass():
    """Divides by the count that clear the threshold, not by the batch size."""
    torch.manual_seed(0)
    logits = torch.randn(4, CLASSES)
    got = pseudo_label_ce(logits, MIXED, threshold=0.85, reduce="mask")
    per_sample = F.cross_entropy(logits, MIXED.argmax(dim=-1), reduction="none")
    expected = (per_sample[0] + per_sample[2]) / 2
    assert torch.allclose(got, expected, rtol=1e-6)


def test_ratio_mode_scales_the_whole_batch_by_the_pass_fraction():
    torch.manual_seed(0)
    logits = torch.randn(4, CLASSES)
    got = pseudo_label_ce(logits, MIXED, threshold=0.85, reduce="ratio")
    expected = F.cross_entropy(logits, MIXED.argmax(dim=-1)) * 0.5
    assert torch.allclose(got, expected, rtol=1e-6)


def test_the_two_modes_differ_on_a_partially_confident_batch():
    torch.manual_seed(0)
    logits = torch.randn(4, CLASSES)
    masked = pseudo_label_ce(logits, MIXED, 0.85, "mask")
    ratio = pseudo_label_ce(logits, MIXED, 0.85, "ratio")
    assert not torch.allclose(masked, ratio)


@pytest.mark.parametrize("reduce", ["mask", "ratio"])
def test_zero_when_nothing_passes(reduce):
    logits = torch.randn(3, CLASSES)
    reference = torch.full((3, CLASSES), 1.0 / CLASSES)
    assert float(pseudo_label_ce(logits, reference, 0.85, reduce)) == 0.0


@pytest.mark.parametrize("reduce", ["mask", "ratio"])
def test_all_pass_makes_the_two_modes_agree(reduce):
    torch.manual_seed(0)
    logits = torch.randn(6, CLASSES)
    reference = sharp(torch.arange(6) % CLASSES)
    got = pseudo_label_ce(logits, reference, 0.85, reduce)
    expected = F.cross_entropy(logits, reference.argmax(dim=-1))
    assert torch.allclose(got, expected, rtol=1e-5)


def test_mask_mode_magnitude_is_independent_of_the_pass_count():
    """The property that makes "mask" noisy rather than weak: with one sample
    passing, the term equals that sample's own cross-entropy -- it does not get
    divided by the batch size. "ratio" on the same batch is 16x smaller."""
    torch.manual_seed(0)
    logits = torch.randn(16, CLASSES)
    labels = torch.zeros(16, dtype=torch.long)
    reference = sharp(labels)
    reference[1:] = 1.0 / CLASSES                 # only sample 0 clears 0.85

    masked = pseudo_label_ce(logits, reference, 0.85, "mask")
    ratio = pseudo_label_ce(logits, reference, 0.85, "ratio")
    single = F.cross_entropy(logits[:1], labels[:1])

    assert torch.allclose(masked, single, rtol=1e-5)
    assert torch.allclose(ratio, F.cross_entropy(logits, labels) / 16, rtol=1e-5)
    assert float(masked) > 5 * float(ratio)


def test_threshold_is_inclusive():
    """ge, not gt: a sample exactly at the threshold passes."""
    logits = torch.randn(1, CLASSES)
    reference = torch.tensor([[0.85, 0.05, 0.05, 0.05]])
    assert float(pseudo_label_ce(logits, reference, 0.85, "ratio")) > 0.0
    assert float(pseudo_label_ce(logits, reference, 0.8500001, "ratio")) == 0.0


def test_reference_is_not_a_gradient_path():
    logits = torch.randn(4, CLASSES, requires_grad=True)
    reference = sharp(torch.zeros(4, dtype=torch.long)).requires_grad_(True)
    pseudo_label_ce(logits, reference, 0.85, "mask").backward()
    assert logits.grad is not None and torch.any(logits.grad != 0)
    assert reference.grad is None


def test_unknown_reduce_raises():
    with pytest.raises(ValueError, match="reduce must be"):
        pseudo_label_ce(torch.randn(2, CLASSES),
                        torch.full((2, CLASSES), 0.25), 0.85, "average")


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_threshold_out_of_range_raises(threshold):
    with pytest.raises(ValueError, match="threshold must be within"):
        pseudo_label_ce(torch.randn(2, CLASSES),
                        torch.full((2, CLASSES), 0.25), threshold, "mask")


# --- pass_fraction -----------------------------------------------------------

def test_pass_fraction_counts_the_confident_samples():
    assert pass_fraction(MIXED, 0.85) == pytest.approx(0.5)
    assert pass_fraction(MIXED, 0.99) == pytest.approx(0.0)
    assert pass_fraction(MIXED, 0.4) == pytest.approx(1.0)


def test_pass_fraction_takes_no_gradient():
    reference = MIXED.clone().requires_grad_(True)
    assert isinstance(pass_fraction(reference, 0.85), float)
    assert reference.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduce", ["mask", "ratio"])
def test_zero_not_nan_when_nothing_passes_at_any_dtype(dtype, reduce):
    """The epsilon guard has to survive the dtype the branch actually runs at.

    Built with .to(logits.dtype), the mask is fp16 under an fp16 backbone, and
    fp16 cannot represent 1e-8: `0.0 + 1e-8` rounds to 0.0, so the guarded
    division is 0.0 / 0.0 = NaN. One target batch with nothing over the
    threshold then poisons every LoRA factor and the teacher through backward.
    A float32 mask, which is what the reference uses, cannot underflow.
    """
    reference = torch.full((8, 5), 0.2)          # max prob 0.2, threshold 0.85
    logits = torch.randn(8, 5, dtype=dtype)

    loss = pseudo_label_ce(logits, reference, threshold=0.85, reduce=reduce)

    assert not torch.isnan(loss), f"NaN at {dtype} with reduce={reduce!r}"
    assert float(loss) == 0.0


def test_the_masked_reduction_is_float32_under_an_fp16_backbone():
    """Reducing in fp16 also costs precision when the mask is NOT empty: the
    reference's .float() mask promotes the per-sample losses, so the sum and the
    division happen at float32 regardless of the backbone."""
    reference = torch.zeros(64, 4)
    reference[:, 0] = 1.0                        # every sample passes
    logits = torch.randn(64, 4, dtype=torch.float16) * 8.0

    loss = pseudo_label_ce(logits, reference, threshold=0.85, reduce="mask")

    assert loss.dtype is torch.float32

    # The per-sample losses are still fp16 -- that is the backbone's precision,
    # not this function's business. What the float32 mask buys is that summing 64
    # of them and dividing does not happen in fp16.
    per_sample = F.cross_entropy(logits, reference.argmax(-1), reduction="none")
    assert per_sample.dtype is torch.float16
    assert float(loss) == pytest.approx(float(per_sample.float().mean()), rel=1e-6)
    assert float(loss) != pytest.approx(float(per_sample.mean()), rel=1e-9), \
        "fp16 and fp32 reductions agree here, so this test proves nothing"
