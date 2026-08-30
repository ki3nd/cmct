"""Branch 1's loss: source CE + pseudo-label CE + MK-MMD."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cmct.losses import LoraBranchLoss, mk_mmd, pseudo_label_ce

CLASSES = 5
BATCH = 8


def batch(seed=0, confident=True):
    torch.manual_seed(seed)
    reference = F.one_hot(torch.arange(BATCH) % CLASSES, CLASSES).float()
    reference = F.softmax(reference / (0.05 if confident else 5.0), dim=-1)
    return dict(
        source_logits=torch.randn(BATCH, CLASSES),
        source_label=torch.randint(0, CLASSES, (BATCH,)),
        target_logits=torch.randn(BATCH, CLASSES),
        reference_probabilities=reference,
        source_features=torch.randn(BATCH, 16),
        target_features=torch.randn(BATCH, 16) + 0.5,
    )


def test_total_is_the_sum_of_the_reported_components():
    out = LoraBranchLoss()(**batch())          # mmd_weight defaults to 1.0
    assert torch.allclose(out.total,
                          out.source_ce + out.pseudo_label + out.mmd)


def test_components_are_reported_raw_and_weighted_only_in_the_total():
    """All four terms on one scale, and each comparable with the reference's own
    printed number -- it prints x, self, cross and mmd raw and applies the
    weights only when summing (phpl/train_mfa_v2.py:562)."""
    data = batch()
    out = LoraBranchLoss(mmd_weight=0.25)(**data)
    # RAW, like the other three components. The weight appears only in `total`.
    assert torch.allclose(out.mmd, mk_mmd(data["source_features"],
                                          data["target_features"]))
    assert torch.allclose(out.total,
                          out.source_ce + out.pseudo_label + 0.25 * out.mmd)


def test_mmd_weight_scales_only_the_mmd_term():
    one = LoraBranchLoss(mmd_weight=1.0)(**batch())
    half = LoraBranchLoss(mmd_weight=0.5)(**batch())
    # The reported term does not move -- it is raw. What moves is the total.
    assert torch.allclose(half.mmd, one.mmd)
    assert torch.allclose(half.source_ce, one.source_ce)
    assert torch.allclose(half.pseudo_label, one.pseudo_label)
    assert torch.allclose(one.total - half.total, 0.5 * one.mmd)


def test_mmd_weight_zero_skips_the_term_entirely():
    data = batch()
    out = LoraBranchLoss(mmd_weight=0.0)(**data)
    assert float(out.mmd) == 0.0
    assert torch.allclose(out.total, out.source_ce + out.pseudo_label)


def test_missing_features_are_allowed_when_mmd_is_off():
    data = batch()
    data.pop("source_features")
    data.pop("target_features")
    out = LoraBranchLoss(mmd_weight=0.0)(**data)
    assert float(out.mmd) == 0.0


@pytest.mark.parametrize("missing", ["source_features", "target_features"])
def test_missing_features_raise_when_mmd_is_on(missing):
    data = batch()
    data.pop(missing)
    with pytest.raises(ValueError, match="mmd_weight > 0 needs"):
        LoraBranchLoss(mmd_weight=1.0)(**data)


def test_source_ce_has_no_label_smoothing():
    """Branch 2's classifier CE carries label_smoothing=0.1; branch 1's does
    not."""
    data = batch()
    out = LoraBranchLoss()(**data)
    assert torch.allclose(out.source_ce,
                          F.cross_entropy(data["source_logits"],
                                          data["source_label"]))
    assert not torch.allclose(
        out.source_ce,
        F.cross_entropy(data["source_logits"], data["source_label"],
                        label_smoothing=0.1)
    )


@pytest.mark.parametrize("reduce", ["mask", "ratio"])
def test_pseudo_label_term_matches_the_standalone_function(reduce):
    data = batch()
    out = LoraBranchLoss(threshold=0.85, reduce=reduce)(**data)
    assert torch.allclose(out.pseudo_label,
                          pseudo_label_ce(data["target_logits"],
                                          data["reference_probabilities"],
                                          0.85, reduce))


def test_mask_ratio_is_reported():
    confident = LoraBranchLoss()(**batch(confident=True))
    unsure = LoraBranchLoss()(**batch(confident=False))
    assert confident.mask_ratio == 1.0
    assert unsure.mask_ratio == 0.0


def test_reference_name_is_carried_through():
    """Branch 1 switches its reference once, from the frozen zero-shot CLIP to
    the EMA teacher. The loss does not choose; it only records which was used."""
    for name in ("zero_shot", "teacher"):
        assert LoraBranchLoss()(**batch(), reference_name=name).reference == name


def test_gradient_reaches_the_student_only():
    data = batch()
    data["target_logits"].requires_grad_(True)
    data["reference_probabilities"].requires_grad_(True)
    LoraBranchLoss()(**data).total.backward()
    assert torch.any(data["target_logits"].grad != 0)
    assert data["reference_probabilities"].grad is None


def test_gradient_reaches_both_feature_sets_through_mmd():
    data = batch()
    data["source_features"].requires_grad_(True)
    data["target_features"].requires_grad_(True)
    LoraBranchLoss(mmd_weight=1.0)(**data).mmd.backward()
    for key in ("source_features", "target_features"):
        assert torch.any(data[key].grad != 0), key


def test_no_hidden_state():
    loss = LoraBranchLoss()
    first = loss(**batch())
    second = loss(**batch())
    assert torch.equal(first.total, second.total)
    assert first.mask_ratio == second.mask_ratio


def test_positional_arguments_are_rejected():
    """Six same-shaped tensors: a positional swap would be silent, so there are
    no positional arguments to swap."""
    data = batch()
    with pytest.raises(TypeError):
        LoraBranchLoss()(data["source_logits"], data["source_label"])
