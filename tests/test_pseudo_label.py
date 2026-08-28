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
