
import torch

from cmct.losses import DebiasTracker, diversity_loss, masked_cross_entropy, mk_mmd
from tests.conftest import load_fixture


def test_mk_mmd_matches_the_frozen_baseline():
    fx = load_fixture("mk_mmd.json")
    got = mk_mmd(torch.tensor(fx["source"]), torch.tensor(fx["target"])).item()
    assert abs(got - fx["expected"]) < 1e-6


def test_masked_cross_entropy_matches_the_frozen_baseline():
    fx = load_fixture("masked_ce.json")
    got = masked_cross_entropy(torch.tensor(fx["logits"]),
                               torch.tensor(fx["prob_ref"]),
                               fx["threshold"]).item()
    assert abs(got - fx["expected"]) < 1e-6


def test_masked_cross_entropy_is_zero_when_nothing_clears_the_threshold():
    logits = torch.zeros(4, 5)
    prob_ref = torch.full((4, 5), 0.2)  # max prob 0.2 < 0.85
    assert masked_cross_entropy(logits, prob_ref, 0.85).item() == 0.0


def test_debias_tracker_corrects_before_updating_qhat():
    # DebiasTracker.correct's contract:
    # correct `logits` using `qhat` as of BEFORE the call, THEN update
    # `qhat` from the raw, pre-correction prediction -- never the other
    # way round. This test fails if that order is swapped.
    torch.manual_seed(0)
    num_classes = 5
    tau, momentum = 0.5, 0.9
    logits = torch.randn(3, num_classes)

    tracker = DebiasTracker(num_classes, tau=tau, momentum=momentum, device="cpu")

    # The FIRST call must use the untouched, uniform initial qhat: its
    # output must equal `logits - tau * log(1 / num_classes)` EXACTLY. This
    # only holds if qhat has not yet been updated by the time the
    # correction is computed.
    uniform_qhat = torch.full((num_classes,), 1.0 / num_classes)
    expected_first = logits - tau * torch.log(uniform_qhat)
    first = tracker.correct(logits)
    assert torch.equal(first, expected_first)

    qhat_after_first = tracker.qhat.clone()
    # qhat must have moved, in the direction of the observed (raw,
    # pre-correction) class distribution, i.e. away from uniform.
    assert not torch.equal(qhat_after_first, uniform_qhat)
    prob_raw = torch.softmax(logits, dim=-1).mean(dim=0)
    expected_qhat = momentum * uniform_qhat + (1.0 - momentum) * prob_raw
    assert torch.allclose(qhat_after_first, expected_qhat)

    # A second call on the IDENTICAL logits must return something
    # different, because qhat moved between the two calls.
    second = tracker.correct(logits)
    assert not torch.equal(first, second)


def test_diversity_terms_punish_collapsing_onto_one_class():
    # The ONE property both terms exist for. `branch_lora` has no other
    # defence against predicting a single class for the whole batch, and
    # plain confidence maximisation would score these two IDENTICALLY --
    # both batches are equally confident. Removing the class-balancing
    # (the `/ column_mass` in balanced_gini, the marginal-entropy half of
    # information_maximization) makes this test fail.
    spread = torch.eye(3) * 20.0            # confident, one class each
    collapsed = torch.zeros(3, 3)
    collapsed[:, 0] = 20.0                  # equally confident, all one class

    for kind in ("im", "gini"):
        assert diversity_loss(collapsed, kind).item() > diversity_loss(spread, kind).item() + 0.5, kind


def test_diversity_terms_stay_finite_at_fp16():
    # branch_lora runs fp16 and its logit_scale (~100) drives softmax to
    # near-one-hot, so log(p) underflows unless the terms cast to fp32.
    logits = (torch.eye(4) * 100.0).half()
    for kind in ("im", "gini"):
        assert torch.isfinite(diversity_loss(logits, kind)), kind
