"""The cross-teaching term, checked against old-cmct's own source."""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cmct.losses.cross import MODES, cross_loss

REFERENCE = Path("/home/pc1175/DA-Research/old-cmct/train_mfa_v2.py")


def reference_pseudo_label_loss(threshold: float):
    """The reference's own `_pseudo_label_loss`, executed from its source.

    It is a closure inside main() -- it captures `confi` and cannot be imported.
    Extracting its source and exec'ing it runs the reference's literal bytes, so
    this test tracks the reference rather than a transcription of it. Copying the
    body into this file would only compare our code against itself.
    """
    text = REFERENCE.read_text()
    match = re.search(r"\n(    def _pseudo_label_loss\(.*?\n)(?=\n    #)", text, re.S)
    assert match, "could not locate _pseudo_label_loss in the reference"
    namespace = {"torch": torch, "F": F, "confi": threshold}
    exec(textwrap.dedent(match.group(1)), namespace)
    return namespace["_pseudo_label_loss"]


def sample(n=16, k=7, seed=0, sharpness=6.0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, k, generator=generator)
    reference = F.softmax(torch.randn(n, k, generator=generator) * sharpness, dim=-1)
    return logits, reference


# --- parity ------------------------------------------------------------------

@pytest.mark.parametrize("threshold", [0.85, 0.5, 0.99])
def test_matches_the_reference_bit_for_bit(threshold):
    """Against the reference called the way both cross sites call it: with its
    `mode` left at the default (train_mfa_v2.py:861 omits the argument, :924
    passes a flag whose default is "mask")."""
    logits, reference = sample()
    theirs = reference_pseudo_label_loss(threshold)(logits, reference)
    ours = cross_loss(target_logits=logits, reference_probabilities=reference,
                      threshold=threshold)
    assert float(ours.value) == float(theirs)


def test_the_reduction_is_the_masked_one_not_the_ratio_one():
    """The reference's other reduction, which nothing selects. If our value
    matched it too, the parity test above would prove nothing about which one we
    compute."""
    logits, reference = sample()
    ratio = reference_pseudo_label_loss(0.85)(logits, reference, "ratio")
    ours = cross_loss(target_logits=logits, reference_probabilities=reference)
    assert abs(float(ours.value) - float(ratio)) > 1e-3


# --- reporting ---------------------------------------------------------------

def test_mask_ratio_is_the_fraction_clearing_the_threshold():
    logits, reference = sample()
    expected = float(reference.max(-1).values.ge(0.85).float().mean())
    out = cross_loss(target_logits=logits, reference_probabilities=reference)
    assert out.mask_ratio == pytest.approx(expected)


def test_the_value_is_unweighted():
    """The caller multiplies by cross_weight, so this number is the one the
    reference logs as loss_u1_cross -- not the one it adds to the total."""
    logits, reference = sample()
    out = cross_loss(target_logits=logits, reference_probabilities=reference)
    direct = (F.cross_entropy(logits, reference.argmax(-1), reduction="none")
              * reference.max(-1).values.ge(0.85).float()).sum() \
        / (reference.max(-1).values.ge(0.85).float().sum() + 1e-8)
    assert float(out.value) == pytest.approx(float(direct))


def test_nothing_passing_gives_zero_not_nan():
    logits, _ = sample()
    flat = torch.full((16, 7), 1.0 / 7)
    out = cross_loss(target_logits=logits.half(), reference_probabilities=flat)
    assert not torch.isnan(out.value)
    assert float(out.value) == 0.0
    assert out.mask_ratio == 0.0


# --- guards ------------------------------------------------------------------

def test_gini_raises_rather_than_silently_running_mask():
    """A cross term that quietly changed form would be invisible: the loss value
    stays plausible and only the numbers at the end of the run differ."""
    logits, reference = sample()
    with pytest.raises(ValueError) as caught:
        cross_loss(target_logits=logits, reference_probabilities=reference,
                   mode="gini", branch="vlp")
    message = str(caught.value)
    assert "gini" in message
    assert "branches[vlp].cross_mode" in message
    assert "mask" in message


def test_only_mask_is_implemented():
    assert MODES == ("mask",)
