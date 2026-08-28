"""Multi-kernel MMD, checked against the reference implementation."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

from cmct.losses import mk_mmd

OLD_CMCT = Path("/home/pc1175/DA-Research/old-cmct")
HAS_REFERENCE = (OLD_CMCT / "utils" / "MK_MMD.py").is_file()


def import_reference():
    """utils/MK_MMD.py imports only torch, so a path insert is enough -- but
    restore sys.modules so the name `utils` does not linger and shadow anything
    in a later test."""
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "utils"}
    for key in saved:
        del sys.modules[key]
    saved_path = sys.path[:]
    sys.path.insert(0, str(OLD_CMCT))
    try:
        return importlib.import_module("utils.MK_MMD").MK_MMD
    finally:
        sys.path[:] = saved_path
        for key in [k for k in sys.modules if k.split(".")[0] == "utils"]:
            del sys.modules[key]
        sys.modules.update(saved)


def features(n, d=64, seed=0, shift=0.0):
    torch.manual_seed(seed)
    return torch.randn(n, d) + shift


# --- properties --------------------------------------------------------------

def test_zero_for_identical_batches():
    x = features(16)
    assert float(mk_mmd(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_grows_with_the_shift_between_distributions():
    source = features(32, seed=0)
    near = features(32, seed=1, shift=0.1)
    far = features(32, seed=1, shift=3.0)
    assert float(mk_mmd(source, far)) > float(mk_mmd(source, near))


def test_is_symmetric():
    source, target = features(24, seed=0), features(24, seed=1, shift=0.5)
    assert float(mk_mmd(source, target)) == pytest.approx(
        float(mk_mmd(target, source)), rel=1e-6
    )


def test_gradient_flows_to_both_sides():
    source = features(16, seed=0).requires_grad_(True)
    target = features(16, seed=1, shift=1.0).requires_grad_(True)
    mk_mmd(source, target).backward()
    for tensor in (source, target):
        assert tensor.grad is not None and torch.any(tensor.grad != 0)


def test_bandwidth_is_detached():
    """The bandwidth is an estimate of scale, not something the gradient should
    move. Compare against a hand-built version that detaches it, and against one
    that does not."""
    def hand_built(source, target, detach):
        total = torch.cat([source, target], dim=0)
        n = total.size(0)
        distance = ((total.unsqueeze(0).expand(n, n, total.size(1))
                     - total.unsqueeze(1).expand(n, n, total.size(1))) ** 2).sum(2)
        raw = distance.detach() if detach else distance
        bandwidth = raw.sum() / (n**2 - n) / 2.0**2
        kernels = sum(torch.exp(-distance / (bandwidth * 2.0**i)) for i in range(5))
        k = source.size(0)
        return (kernels[:k, :k].mean() + kernels[k:, k:].mean()
                - kernels[:k, k:].mean() - kernels[k:, :k].mean())

    grads = []
    for fn in (mk_mmd, lambda s, t: hand_built(s, t, True),
               lambda s, t: hand_built(s, t, False)):
        source = features(12, seed=0).requires_grad_(True)
        fn(source, features(12, seed=1, shift=1.0)).backward()
        grads.append(source.grad.clone())
    assert torch.allclose(grads[0], grads[1], rtol=1e-5)
    assert not torch.allclose(grads[0], grads[2], rtol=1e-3)


def test_fix_sigma_overrides_the_estimate():
    source, target = features(16, seed=0), features(16, seed=1, shift=1.0)
    assert not torch.allclose(mk_mmd(source, target),
                              mk_mmd(source, target, fix_sigma=10.0))


def test_shape_errors():
    with pytest.raises(ValueError, match="expected 2-D"):
        mk_mmd(torch.randn(4), torch.randn(4, 8))
    with pytest.raises(ValueError, match="feature dimensions differ"):
        mk_mmd(torch.randn(4, 8), torch.randn(4, 16))


# --- parity with the reference ----------------------------------------------

@pytest.mark.skipif(not HAS_REFERENCE, reason="no reference implementation")
@pytest.mark.parametrize("shift", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("size", [(8, 8), (32, 32), (16, 24)])
def test_matches_the_reference(shift, size):
    reference = import_reference()
    n_source, n_target = size
    source = features(n_source, seed=0)
    target = features(n_target, seed=1, shift=shift)
    assert torch.allclose(mk_mmd(source, target), reference(source, target),
                          rtol=1e-6, atol=0)


@pytest.mark.skipif(not HAS_REFERENCE, reason="no reference implementation")
def test_gradients_match_the_reference():
    """Equal values are not enough: a wrong bandwidth or a missing detach can
    leave the value intact and the gradient different."""
    reference = import_reference()
    grads = []
    for fn in (reference, mk_mmd):
        source = features(16, seed=0).requires_grad_(True)
        target = features(16, seed=1, shift=1.0).requires_grad_(True)
        fn(source, target).backward()
        grads.append((source.grad.clone(), target.grad.clone()))
    for ours, theirs in zip(grads[1], grads[0], strict=True):
        assert torch.allclose(ours, theirs, rtol=1e-6, atol=0)


@pytest.mark.skipif(not HAS_REFERENCE, reason="no reference implementation")
def test_kernel_count_and_multiplier_match_the_reference():
    """Reference defaults are kernel_num=5, kernel_mul=2.0. Changing either must
    change the value, so a silent default drift cannot pass."""
    reference = import_reference()
    source, target = features(16, seed=0), features(16, seed=1, shift=1.0)
    baseline = reference(source, target)
    assert torch.allclose(mk_mmd(source, target, kernel_num=5, kernel_mul=2.0),
                          baseline, rtol=1e-6)
    assert not torch.allclose(mk_mmd(source, target, kernel_num=3), baseline,
                              rtol=1e-3)
    assert not torch.allclose(mk_mmd(source, target, kernel_mul=3.0), baseline,
                              rtol=1e-3)
