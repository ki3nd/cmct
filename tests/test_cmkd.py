"""Branch 2's self-loss: component behaviour, and equality with the original.

Everything here runs on hand-made tensors; no CLIP is needed.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cmct.losses import CmkdLoss, calibrated_coefficient, gini_impurity, sigmoid_ramp

CLASSES = 5
BATCH = 8


def probs(rows=BATCH, cols=CLASSES, seed=0):
    torch.manual_seed(seed)
    return F.softmax(torch.randn(rows, cols), dim=-1)


# --- sigmoid_ramp -------------------------------------------------------------

def test_ramp_starts_at_zero():
    assert sigmoid_ramp(0, 1000) == 0.0


def test_ramp_ceiling_is_not_one():
    """2 / (1 + e^-1) - 1 = 0.4621. Reading this as a 0-to-1 ramp overstates the
    final weight of every term built on it by more than 2x."""
    assert sigmoid_ramp(1000, 1000) == pytest.approx(2 / (1 + math.exp(-1)) - 1)
    assert sigmoid_ramp(1000, 1000) == pytest.approx(0.4621, abs=1e-4)


def test_ramp_is_monotone():
    values = [sigmoid_ramp(s, 1000) for s in range(0, 1001, 10)]
    assert values == sorted(values)


def test_ramp_is_clamped_past_max_iter():
    """The original clamped curr_iter inside step(); the same holds here."""
    assert sigmoid_ramp(5000, 1000) == sigmoid_ramp(1000, 1000)


def test_ramp_rejects_bad_arguments():
    with pytest.raises(ValueError, match="max_iter must be > 0"):
        sigmoid_ramp(0, 0)
    with pytest.raises(ValueError, match="step must be >= 0"):
        sigmoid_ramp(-1, 10)


# --- gini_impurity -----------------------------------------------------------

def test_gini_matches_the_formula():
    pred = probs()
    sum_dim = pred.sum(dim=0).unsqueeze(0)
    expected = torch.sum(1 - torch.sum(pred**2 / sum_dim, dim=-1))
    assert torch.allclose(gini_impurity(pred), expected)


@pytest.mark.parametrize("rows", [1, 4, 8, 32])
def test_gini_reduction_is_a_sum(rows):
    """For `rows` identical distributions, sum_dim = rows * p, so each sample's
    term is 1 - 1/rows and a SUM gives exactly rows - 1. A mean would give
    (rows - 1) / rows, so this pins the reduction."""
    pred = probs(rows=1).repeat(rows, 1)
    assert float(gini_impurity(pred)) == pytest.approx(rows - 1, abs=1e-5)


def test_gini_applies_a_per_sample_weight():
    pred = probs()
    coe = torch.linspace(0.1, 1.0, BATCH)
    sum_dim = pred.sum(dim=0).unsqueeze(0)
    expected = torch.sum(coe * (1 - torch.sum(pred**2 / sum_dim, dim=-1)))
    assert torch.allclose(gini_impurity(pred, coe), expected)
    assert not torch.allclose(gini_impurity(pred, coe), gini_impurity(pred))


def test_gini_detaches_the_class_wise_sum():
    """sum_dim is detached, so its gradient path is cut. Compare against a
    hand-written version that detaches, and against one that does not."""
    pred = probs().requires_grad_(True)
    gini_impurity(pred).backward()
    ours = pred.grad.clone()

    pred2 = probs().requires_grad_(True)
    sum_dim = pred2.sum(dim=0).unsqueeze(0).detach()
    torch.sum(1 - torch.sum(pred2**2 / sum_dim, dim=-1)).backward()
    assert torch.allclose(ours, pred2.grad)

    pred3 = probs().requires_grad_(True)
    sum_dim_attached = pred3.sum(dim=0).unsqueeze(0)
    torch.sum(1 - torch.sum(pred3**2 / sum_dim_attached, dim=-1)).backward()
    assert not torch.allclose(ours, pred3.grad)


# --- calibrated_coefficient --------------------------------------------------

def test_coefficient_is_one_when_the_two_agree():
    pred = probs()
    assert torch.allclose(calibrated_coefficient(pred, pred),
                          torch.ones(BATCH), atol=1e-6)


def test_coefficient_falls_as_they_disagree():
    pred = probs(seed=0)
    near = F.softmax(pred.log() + 0.05 * torch.randn_like(pred), dim=-1)
    far = probs(seed=1)
    assert float(calibrated_coefficient(pred, near).mean()) > \
        float(calibrated_coefficient(pred, far).mean())


def test_coefficient_kl_direction():
    """exp(-KL(reference || pred)), not the reverse. F.kl_div(input, target)
    computes sum target * (log target - input)."""
    pred, reference = probs(seed=0), probs(seed=1)
    expected = torch.exp(-(reference * (reference.log() - pred.log())).sum(-1))
    assert torch.allclose(calibrated_coefficient(pred, reference), expected, atol=1e-6)

    reversed_direction = torch.exp(-(pred * (pred.log() - reference.log())).sum(-1))
    assert not torch.allclose(calibrated_coefficient(pred, reference),
                              reversed_direction, atol=1e-4)


def test_coefficient_is_detached():
    pred = probs().requires_grad_(True)
    assert calibrated_coefficient(pred, probs(seed=1)).requires_grad is False


# --- CmkdLoss ----------------------------------------------------------------

def batch(seed=0):
    torch.manual_seed(seed)
    return dict(
        source_logits=torch.randn(BATCH, CLASSES),
        source_label=torch.randint(0, CLASSES, (BATCH,)),
        source_cosine_logits=torch.randn(BATCH, CLASSES),
        target_logits=torch.randn(BATCH, CLASSES),
        target_cosine_logits=torch.randn(BATCH, CLASSES),
    )


def test_total_is_the_sum_of_its_parts():
    out = CmkdLoss(max_iter=100)(**batch(), step=50)
    assert torch.allclose(out.total, out.clf + out.task + out.distill + out.reg)


def test_at_step_zero_the_ramped_terms_vanish():
    out = CmkdLoss(max_iter=100)(**batch(), step=0)
    assert out.ramp == 0.0
    assert float(out.task) == 0.0
    assert float(out.distill) == 0.0
    data = batch()
    expected_reg = 0.1 * F.cross_entropy(data["source_cosine_logits"], data["source_label"])
    assert torch.allclose(out.reg, expected_reg)


def test_lambda1_scales_only_the_transfer_terms():
    a = CmkdLoss(max_iter=100, lambda1=0.25)(**batch(), step=50)
    b = CmkdLoss(max_iter=100, lambda1=0.50)(**batch(), step=50)
    assert torch.allclose(b.task, 2 * a.task)
    assert torch.allclose(b.distill, 2 * a.distill)
    assert torch.allclose(b.clf, a.clf)
    assert torch.allclose(b.reg, a.reg)


def test_label_smoothing_applies_to_clf_only():
    data = batch()
    out = CmkdLoss(max_iter=100, label_smoothing=0.1)(**data, step=0)
    smoothed = F.cross_entropy(data["source_logits"], data["source_label"],
                               label_smoothing=0.1)
    plain = F.cross_entropy(data["source_cosine_logits"], data["source_label"])
    assert torch.allclose(out.clf, smoothed)
    assert torch.allclose(out.reg, 0.1 * plain)
    assert not torch.allclose(out.clf, F.cross_entropy(data["source_logits"],
                                                       data["source_label"]))


def test_gradient_reaches_the_cosine_branch_only_through_reg():
    """coe is detached and `mixed` detaches its reference half, so the transfer
    terms train the head alone. Only reg trains the cosine branch -- which is why
    removing reg would leave it untrained."""
    loss = CmkdLoss(max_iter=100)

    data = batch()
    data["target_logits"].requires_grad_(True)
    data["target_cosine_logits"].requires_grad_(True)
    out = loss(**data, step=50)
    (out.task + out.distill).backward()
    assert torch.any(data["target_logits"].grad != 0)
    assert data["target_cosine_logits"].grad is None

    data = batch()
    data["target_cosine_logits"].requires_grad_(True)
    loss(**data, step=50).reg.backward()
    assert torch.any(data["target_cosine_logits"].grad != 0)


def test_no_hidden_state():
    """Calling twice at the same step gives the same numbers; the ramp comes from
    the argument, not from an internal counter."""
    loss = CmkdLoss(max_iter=100)
    first = loss(**batch(), step=7)
    second = loss(**batch(), step=7)
    assert torch.equal(first.total, second.total)
    assert first.ramp == second.ramp


def test_from_branch_config_rejects_unknown_extra_keys():
    from cmct.config.schema import BackboneConfig, BranchConfig, OptimConfig

    branch = BranchConfig(
        name="vlp", type="vlp_clip",
        backbone=BackboneConfig(checkpoint="x", dtype="fp32"),
        optim=OptimConfig(lr=1e-6),
        extra={"lambda1": 0.25, "mmd_weight": 1.0},
    )
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['mmd_weight'\]"):
        CmkdLoss.from_branch_config(branch, max_iter=100)


def test_from_branch_config_reads_the_real_experiment_config():
    from pathlib import Path

    from cmct.config import load_experiment

    cfg = load_experiment(
        Path(__file__).resolve().parents[1] / "configs" / "experiment"
        / "cmct_officehome_a2c.yaml"
    )
    vlp = [b for b in cfg.branches if b.type == "vlp_clip"][0]
    max_iter = cfg.cotrain.total_macro_steps * vlp.steps_per_macro
    loss = CmkdLoss.from_branch_config(vlp, max_iter=max_iter)
    assert max_iter == 10_000
    assert (loss.lambda1, loss.lambda2, loss.lambda3) == (0.25, 0.1, 0.025)
    assert loss.label_smoothing == 0.1


# --- equality with the original ----------------------------------------------

class OriginalLambdaSheduler(nn.Module):
    """Verbatim from vlpuda_pure/utils/tools.py."""

    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        return 2.0 / (1.0 + math.exp(-self.gamma * p)) - 1

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)


class OriginalCMKD(nn.Module):
    """Verbatim from vlpuda_pure/models/cmkd.py, minus the label_set branch and
    the never-called calibrated_coefficient1."""

    def __init__(self, lambda1, lambda2, lambda3, max_iter):
        super().__init__()
        self.lamb = OriginalLambdaSheduler(max_iter=max_iter)
        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3

    def calibrated_coefficient(self, pred, pred_pretrained):
        distance = F.kl_div(pred.log(), pred_pretrained, reduction="none").sum(-1)
        return torch.exp(-distance).detach()

    def gini_impurity(self, pred, coe=1.0):
        sum_dim = torch.sum(pred, dim=0).unsqueeze(dim=0).detach()
        return torch.sum(coe * (1 - torch.sum(pred**2 / sum_dim, dim=-1)))

    def regularization_term(self, target_pred_clip, source_logit_clip, source_label, lamb):
        return self.lambda2 * F.cross_entropy(source_logit_clip, source_label) + \
            self.lambda3 * lamb * self.gini_impurity(target_pred_clip)

    def forward(self, target_logit, target_logit_clip, source_logit_clip, source_label):
        target_pred = F.softmax(target_logit, dim=1)
        target_pred_clip = F.softmax(target_logit_clip, dim=-1)
        coe = self.calibrated_coefficient(target_pred, target_pred_clip)
        target_pred_mix = 0.5 * (target_pred + target_pred_clip.detach())
        lamb = self.lamb.lamb()
        task_loss = self.lambda1 * lamb * self.gini_impurity(target_pred, coe)
        distill_loss = self.lambda1 * lamb * self.gini_impurity(target_pred_mix, 1 - coe)
        reg_loss = self.regularization_term(target_pred_clip, source_logit_clip,
                                            source_label, lamb)
        self.lamb.step()
        return task_loss + distill_loss + reg_loss


def test_transfer_loss_equals_the_original_step_by_step():
    """The condition for calling this a reproduction: run both for 20 steps on
    the same tensors and require equality at every step."""
    max_iter = 40
    original = OriginalCMKD(0.25, 0.1, 0.025, max_iter)
    ours = CmkdLoss(max_iter=max_iter, lambda1=0.25, lambda2=0.1, lambda3=0.025)

    for step in range(20):
        data = batch(seed=step + 1)
        reference = original(
            data["target_logits"], data["target_cosine_logits"],
            data["source_cosine_logits"], data["source_label"],
        )
        out = ours(**data, step=step)
        transfer = out.task + out.distill + out.reg
        assert torch.allclose(transfer, reference, atol=0, rtol=1e-6), step


def test_clf_loss_equals_the_original():
    data = batch()
    original = nn.CrossEntropyLoss(label_smoothing=0.1)
    out = CmkdLoss(max_iter=100, label_smoothing=0.1)(**data, step=0)
    assert torch.allclose(out.clf, original(data["source_logits"], data["source_label"]))
