"""Equality against the real VLP-UDA code, imported and executed.

This is stronger than a copy of the original pasted into a test: it runs
`vlpuda_pure/models/cmkd.py` itself. Skipped when that tree is absent, so the
project does not depend on it -- it is a fidelity check, not a runtime dependency.

That file differs from upstream VLP-UDA (github.com/Wenlve-Zhou/VLP-UDA,
bf8f0494) by exactly one thing: an optional `self_ref_logit_clip` parameter whose
default of None reproduces the original arithmetic. `utils/tools.py`'s
LambdaSheduler is byte-identical to upstream. So running against this tree is
running against upstream for this code path.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cmct.losses import CmkdLoss

VLPUDA = Path("/home/pc1175/DA-Research/old-cmct/vlpuda_pure")
SHADOWED = ("models", "utils")
CLASSES = 5
BATCH = 8

pytestmark = pytest.mark.skipif(
    not (VLPUDA / "models" / "cmkd.py").is_file(),
    reason=f"no VLP-UDA tree at {VLPUDA}",
)


def import_original_cmkd():
    """Import vlpuda_pure's own CMKD in an isolated sys.path / sys.modules scope.

    Its internal imports are absolute (`from utils.tools import ...`), so the
    directory has to be on sys.path -- and `utils` and `models` must not be left
    behind in sys.modules afterwards, or a later test importing either name would
    get this tree instead.
    """
    def shadowed(name):
        return name in SHADOWED or name.split(".", 1)[0] in SHADOWED

    saved_modules = {k: v for k, v in sys.modules.items() if shadowed(k)}
    for key in saved_modules:
        del sys.modules[key]
    saved_path = sys.path[:]
    sys.path.insert(0, str(VLPUDA))
    try:
        return importlib.import_module("models.cmkd").CMKD
    finally:
        sys.path[:] = saved_path
        for key in [k for k in sys.modules if shadowed(k)]:
            del sys.modules[key]
        sys.modules.update(saved_modules)


def batch(seed):
    torch.manual_seed(seed)
    return dict(
        source_logits=torch.randn(BATCH, CLASSES),
        source_label=torch.randint(0, CLASSES, (BATCH,)),
        source_cosine_logits=torch.randn(BATCH, CLASSES),
        target_logits=torch.randn(BATCH, CLASSES),
        target_cosine_logits=torch.randn(BATCH, CLASSES),
    )


def original_args(max_iter, lambda1=0.25, lambda2=0.1, lambda3=0.025):
    return argparse.Namespace(max_iter=max_iter, lambda1=lambda1,
                              lambda2=lambda2, lambda3=lambda3)


@pytest.mark.parametrize("max_iter", [40, 10_000])
def test_transfer_loss_matches_the_real_cmkd_at_every_step(max_iter):
    original_cls = import_original_cmkd()
    original = original_cls(original_args(max_iter))
    ours = CmkdLoss(max_iter=max_iter, lambda1=0.25, lambda2=0.1, lambda3=0.025)

    for step in range(30):
        data = batch(step + 1)
        reference = original(
            data["target_logits"], data["target_cosine_logits"],
            data["source_cosine_logits"], data["source_label"],
        )
        out = ours(**data, step=step)
        transfer = out.task + out.distill + out.reg
        assert torch.allclose(transfer, reference, atol=0, rtol=1e-6), (
            f"step {step}: ours {float(transfer)} vs original {float(reference)}"
        )


def test_ramp_matches_the_real_lambda_scheduler():
    """Our ramp is read from a step argument; the original advances an internal
    counter inside forward(). Check they agree step for step, including the clamp
    past max_iter."""
    original_cls = import_original_cmkd()
    max_iter = 25
    original = original_cls(original_args(max_iter))
    ours = CmkdLoss(max_iter=max_iter)

    for step in range(40):
        expected = original.lamb.lamb()
        data = batch(step + 1)
        got = ours(**data, step=step).ramp
        assert got == pytest.approx(expected, abs=1e-12), step
        original.lamb.step()


@pytest.mark.parametrize("weights", [(0.25, 0.1, 0.025), (1.0, 0.5, 0.5), (0.1, 0.0, 1.0)])
def test_matches_under_other_weightings(weights):
    lambda1, lambda2, lambda3 = weights
    original_cls = import_original_cmkd()
    original = original_cls(original_args(50, lambda1, lambda2, lambda3))
    ours = CmkdLoss(max_iter=50, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

    for step in range(10):
        data = batch(step + 100)
        reference = original(
            data["target_logits"], data["target_cosine_logits"],
            data["source_cosine_logits"], data["source_label"],
        )
        out = ours(**data, step=step)
        assert torch.allclose(out.task + out.distill + out.reg, reference,
                              atol=0, rtol=1e-6), step


@pytest.mark.parametrize("step", [0, 1, 30, 99])
def test_gradients_match_the_real_cmkd(step):
    """Equal values are not enough -- the gradient into the head and into the
    cosine branch has to match too, or training would diverge from the original
    while every logged number looked right.

    The original reads its ramp from an internal counter, so that counter has to
    be positioned at `step` BEFORE the forward pass, not after.
    """
    original_cls = import_original_cmkd()
    keys = ("target_logits", "target_cosine_logits", "source_cosine_logits")

    def grads_from(which):
        data = batch(7)
        for key in keys:
            data[key].requires_grad_(True)
        if which == "original":
            original = original_cls(original_args(100))
            original.lamb.curr_iter = step
            loss = original(data["target_logits"], data["target_cosine_logits"],
                            data["source_cosine_logits"], data["source_label"])
        else:
            out = CmkdLoss(max_iter=100)(**data, step=step)
            loss = out.task + out.distill + out.reg
        loss.backward()
        return {key: data[key].grad.clone() for key in keys}

    reference, ours = grads_from("original"), grads_from("ours")
    for key in keys:
        assert torch.allclose(reference[key], ours[key], atol=0, rtol=1e-6), (key, step)

    if step > 0:
        assert torch.any(reference["target_logits"] != 0), (
            "the transfer terms should reach the head once the ramp is nonzero"
        )


def test_our_clf_loss_matches_the_real_transfernet_criterion():
    """TransferNet builds clf_loss as nn.CrossEntropyLoss(label_smoothing=...)
    on the head's source logits (make_model.py:47, :56)."""
    data = batch(3)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    out = CmkdLoss(max_iter=100, label_smoothing=0.1)(**data, step=0)
    assert torch.allclose(out.clf, criterion(data["source_logits"], data["source_label"]))
    assert not torch.allclose(
        out.clf, F.cross_entropy(data["source_logits"], data["source_label"])
    )
