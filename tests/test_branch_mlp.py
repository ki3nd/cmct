import pytest

import torch

from cmct.branch_mlp.backbone import PROMPTS, prompts_for
from cmct.branch_mlp.loss import CMKD, LambdaScheduler
from tests.conftest import load_fixture

# cmct's dataset name -> the spelling the frozen prompt fixture is keyed by.
PROMPT_FIXTURE_KEYS = {
    "officehome": "office_home",
    "visda17": "visda",
    "digits": "digits",
    "office31": "office31",
    "domainnet": "domain_net",
    "imageclef": "image_clef",
}


def test_prompt_literals_match_the_frozen_baseline():
    """The prompt lists are not derived from the dataset -- nothing in the
    code makes them agree with the class names. If they ever drifted, the
    branch_mlp cosine logits would be silently mislabelled, so pin them
    against a frozen baseline (tests/fixtures/prompts.json; see
    tests/fixtures/README.md) rather than trusting the literals to stay in
    sync with themselves."""
    baseline = load_fixture("prompts.json")
    assert set(baseline) == set(PROMPT_FIXTURE_KEYS.values())
    for name, fixture_key in PROMPT_FIXTURE_KEYS.items():
        assert prompts_for(name) == baseline[fixture_key], name
    assert set(PROMPTS) == set(PROMPT_FIXTURE_KEYS)


def test_lambda_schedule_matches_the_frozen_baseline():
    fx = load_fixture("lambda_schedule.json")
    sched = LambdaScheduler(gamma=fx["gamma"], max_iter=fx["max_iter"])
    got = []
    for _ in range(len(fx["expected"])):
        got.append(sched.lamb())
        sched.step()
    assert got == fx["expected"]


def _cmkd_from_fixture(fx):
    return CMKD(lambdas=fx["lambdas"], lamb_gamma=fx["lamb_gamma"], max_iter=fx["max_iter"])


def _cmkd_inputs(fx):
    return (torch.tensor(fx["target_logit"]), torch.tensor(fx["target_logit_clip"]),
            torch.tensor(fx["source_logit_clip"]), torch.tensor(fx["source_label"]))


def test_cmkd_loss_matches_the_frozen_baseline_with_live_self_reference():
    fx = load_fixture("cmkd_loss.json")
    got = _cmkd_from_fixture(fx)(
        torch.tensor(fx["target_logit"]), torch.tensor(fx["target_logit_clip"]),
        torch.tensor(fx["source_logit_clip"]), torch.tensor(fx["source_label"]),
    ).item()
    assert abs(got - fx["expected_live_ref"]) < 1e-6


def test_cmkd_loss_matches_the_frozen_baseline_with_teacher_self_reference():
    fx = load_fixture("cmkd_loss.json")
    got = _cmkd_from_fixture(fx)(
        torch.tensor(fx["target_logit"]), torch.tensor(fx["target_logit_clip"]),
        torch.tensor(fx["source_logit_clip"]), torch.tensor(fx["source_label"]),
        self_ref_logit_clip=torch.tensor(fx["self_ref_logit_clip"]),
    ).item()
    assert abs(got - fx["expected_self_ref"]) < 1e-6


def test_teacher_ema_is_convex_and_hard_copies_integer_buffers():
    from cmct.branch_mlp.ema import ema_update_teacher

    class M(torch.nn.Module):
        def __init__(self, w, n):
            super().__init__()
            self.w = torch.nn.Parameter(torch.full((2,), w))
            self.register_buffer("num_batches_tracked", torch.tensor(n))

    teacher, student = M(0.0, 0), M(1.0, 7)
    ema_update_teacher(teacher, student, 0.99)
    assert torch.allclose(teacher.w, torch.full((2,), 0.01))
    assert teacher.num_batches_tracked.item() == 7


def test_cmkd_loss_sequence_matches_the_frozen_baseline_with_live_self_reference():
    """Past iteration 0 the scheduler's lamb is nonzero, so task_loss and
    distill_loss actually contribute. CMKD.forward steps its own scheduler,
    so repeated calls walk the schedule."""
    fx = load_fixture("cmkd_loss.json")
    loss_fn = _cmkd_from_fixture(fx)
    args = _cmkd_inputs(fx)
    got = [loss_fn(*args).item() for _ in range(fx["sequence_steps"])]
    for i, (g, e) in enumerate(zip(got, fx["expected_live_ref_sequence"])):
        assert abs(g - e) < 1e-6, (i, g, e)


def test_cmkd_loss_sequence_matches_the_frozen_baseline_with_teacher_self_reference():
    fx = load_fixture("cmkd_loss.json")
    loss_fn = _cmkd_from_fixture(fx)
    args = _cmkd_inputs(fx)
    self_ref = torch.tensor(fx["self_ref_logit_clip"])
    got = [loss_fn(*args, self_ref_logit_clip=self_ref).item()
           for _ in range(fx["sequence_steps"])]
    for i, (g, e) in enumerate(zip(got, fx["expected_self_ref_sequence"])):
        assert abs(g - e) < 1e-6, (i, g, e)


def test_self_reference_actually_changes_the_loss_once_lamb_is_nonzero():
    """The guard that can fail. At iteration 0 lamb == 0, so task_loss and
    distill_loss -- the only two terms reading self_ref_logit_clip -- vanish
    and both arms are trivially equal; an implementation that ignored
    self_ref_logit_clip entirely, or routed it into reg_loss, would still
    pass the iteration-0 tests. Past step 0 the two arms must diverge, and
    the frozen baseline fixture says by how much."""
    fx = load_fixture("cmkd_loss.json")
    live = fx["expected_live_ref_sequence"]
    self_ref = fx["expected_self_ref_sequence"]
    diverging = [i for i in range(fx["sequence_steps"]) if live[i] != self_ref[i]]
    assert diverging, "fixture itself never diverges -- recapture it"

    live_fn, self_fn = _cmkd_from_fixture(fx), _cmkd_from_fixture(fx)
    args = _cmkd_inputs(fx)
    self_ref_logit = torch.tensor(fx["self_ref_logit_clip"])
    for i in range(fx["sequence_steps"]):
        got_live = live_fn(*args).item()
        got_self = self_fn(*args, self_ref_logit_clip=self_ref_logit).item()
        if i in diverging:
            assert got_live != got_self, (
                f"step {i}: self-reference had no effect on the loss")
            assert abs((got_self - got_live) - (self_ref[i] - live[i])) < 1e-9, i


def test_the_target_forward_is_skipped_when_nothing_would_backprop(imagenet_weights):
    """Under the ImageNet source with branch_lora off there is no reference for
    CMKD and no cross-teaching, so every target-side loss is zero, nothing
    backward() traverses reaches the target forward's graph, and its saved
    tensors are never freed -- a whole backbone pass staying resident on top of
    branch_lora's peak, which is what put a T4 over its 14.5 GiB. The caller
    says so with need_target_logits=False; the pass must then not run."""
    from cmct.branch_mlp import TransferNet

    model = TransferNet(None, model_name="resnet50", source="imagenet", num_classes=5,
                        label_smoothing=0.1, lambdas=LAMBDAS, lamb_gamma=1.0, max_iter=100)
    model.base_network.train()
    model.classifier_layer.train()
    src, tgt = torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64)
    label = torch.randint(0, 5, (4,))

    clf_loss, transfer_loss, target_logits = model(src, tgt, label, need_target_logits=False)

    assert target_logits is None
    assert transfer_loss.item() == 0.0
    clf_loss.backward()  # the source pass is still the trainable one


LAMBDAS = {"task": 0.25, "source_ce": 0.0, "target_gini": 0.0}
"""What the ImageNet config carries: reg_loss's two weights must be zero there."""


def test_cmkd_without_cosine_logits_equals_the_full_loss_at_zero_reg_lambdas():
    """The ImageNet source has no cosine head, so CMKD runs with
    target_logit_clip/source_logit_clip as None and reg_loss dropped. That arm
    must differ from the full one in NOTHING but reg_loss -- an implementation
    that fell back to the (missing) cosine branch for the self-reference, or
    quietly rescaled coe/mix, would diverge here. Run past step 0 so lamb is
    nonzero and task_loss/distill_loss actually contribute."""
    fx = load_fixture("cmkd_loss.json")
    zero_reg = {"task": fx["lambdas"]["task"], "source_ce": 0.0, "target_gini": 0.0}
    full = CMKD(lambdas=zero_reg, lamb_gamma=fx["lamb_gamma"], max_iter=fx["max_iter"])
    bare = CMKD(lambdas=zero_reg, lamb_gamma=fx["lamb_gamma"], max_iter=fx["max_iter"])
    target_logit = torch.tensor(fx["target_logit"])
    self_ref = torch.tensor(fx["self_ref_logit_clip"])
    for i in range(fx["sequence_steps"]):
        expected = full(target_logit, torch.tensor(fx["target_logit_clip"]),
                        torch.tensor(fx["source_logit_clip"]), torch.tensor(fx["source_label"]),
                        self_ref_logit_clip=self_ref).item()
        got = bare(target_logit, None, None, None, self_ref_logit_clip=self_ref).item()
        assert abs(got - expected) < 1e-6, (i, got, expected)
    assert expected != 0.0, "fixture never leaves lamb == 0 -- recapture it"


def test_cmkd_without_cosine_logits_requires_a_self_reference():
    fx = load_fixture("cmkd_loss.json")
    loss_fn = _cmkd_from_fixture(fx)
    with pytest.raises(ValueError, match="self_ref_logit_clip"):
        loss_fn(torch.tensor(fx["target_logit"]), None, None, None)


def test_the_imagenet_transfer_loss_trains_the_branch_from_the_lora_reference(imagenet_weights):
    """The wiring the unit tests above cannot see: TransferNet must feed its
    OWN target logits and the caller's reference into CMKD and hand back a
    loss that backward() actually reaches the backbone through. A transfer_loss
    left disconnected from the graph, or built from the reference instead of
    the branch's own prediction, would leave base_network's gradients at None
    here while every other test still passed.

    Two calls because CMKD's lamb is 0 at step 0 -- task_loss and distill_loss,
    the only two terms left once reg_loss is dropped, vanish there.
    """
    from cmct.branch_mlp import TransferNet

    model = TransferNet(None, model_name="resnet50", source="imagenet", num_classes=5,
                        label_smoothing=0.1, lambdas=LAMBDAS, lamb_gamma=1.0, max_iter=100)
    model.base_network.train()
    model.classifier_layer.train()
    src, tgt = torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64)
    label = torch.randint(0, 5, (4,))
    self_ref = torch.randn(4, 5)

    model(src, tgt, label, self_ref_logit_clip=self_ref)
    _, transfer_loss, _ = model(src, tgt, label, self_ref_logit_clip=self_ref)

    assert transfer_loss.item() != 0.0
    transfer_loss.backward()
    last_conv = model.base_network.net.layer4[-1].conv3.weight
    assert last_conv.grad is not None and last_conv.grad.abs().sum().item() > 0
    assert self_ref.grad is None
