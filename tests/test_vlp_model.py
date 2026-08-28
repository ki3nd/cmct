"""Branch 2's model: structure, teacher construction, and update discipline."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from cmct.branches import VlpModel
from cmct.config.schema import EmaConfig
from cmct.engine.ema import momentum_at

CLASSNAMES = [f"class {i}" for i in range(5)]


IMAGE = 32
"""tiny_clip's input resolution."""


@pytest.fixture
def model(tiny_clip):
    return VlpModel(tiny_clip, CLASSNAMES, len(CLASSNAMES))


# --- structure ---------------------------------------------------------------

def test_real_checkpoint_properties(clip_fp32):
    """The two assertions that need the real trained checkpoint rather than the
    architecture alone."""
    model = VlpModel(clip_fp32, CLASSNAMES, len(CLASSNAMES))
    # inferred from the model, not looked up by architecture name
    assert model.encoder.feature_dim == 512
    # a trained logit_scale is ~100, so cosine logits arrive saturated. A freshly
    # initialised CLIP has exp(log(1/0.07)) = 14.3 instead, so this is a property
    # of the checkpoint, not of the class.
    assert 50 < float(model.encoder.model.logit_scale.exp().detach()) < 200


def test_feature_dim_follows_whatever_model_it_is_given(model):
    assert model.encoder.feature_dim == 32


def test_head_layout(model):
    assert [type(m) for m in model.head] == [nn.BatchNorm1d, nn.LayerNorm, nn.Linear]
    bn, ln, linear = model.head
    assert bn.num_features == 32
    assert ln.eps == 1e-6
    assert linear.bias is None
    assert linear.out_features == len(CLASSNAMES)


def test_head_init(model):
    bn, _, linear = model.head
    assert float(linear.weight.detach().std()) == pytest.approx(0.001, rel=0.2)
    assert bn.bias.requires_grad is False
    assert torch.all(bn.weight == 1.0)
    assert torch.all(bn.bias == 0.0)


def test_features_are_not_normalized(model):
    with torch.no_grad():
        features = model.features(torch.randn(2, 3, IMAGE, IMAGE))
    norms = features.norm(dim=1)
    assert features.shape == (2, 32)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_cosine_logits_carry_the_logit_scale(model):
    with torch.no_grad():
        features = model.features(torch.randn(2, 3, IMAGE, IMAGE))
        logits = model.encoder.cosine_logits(features)
        normalized = features / features.norm(dim=1, keepdim=True)
        raw = normalized @ model.encoder.text_features.t()
    scale = model.encoder.model.logit_scale.exp().detach()
    assert torch.allclose(logits, scale * raw, atol=1e-4)


def test_text_features_are_not_in_state_dict(model):
    keys = set(model.state_dict())
    assert not [k for k in keys if "text_features" in k or k.endswith(".text")]


def test_classname_count_must_match_num_classes(clip_fp32):
    with pytest.raises(ValueError, match="but num_classes"):
        VlpModel(clip_fp32, CLASSNAMES, 65)


def test_fp16_clip_is_rejected(tiny_clip):
    with pytest.raises(TypeError, match="requires an fp32 CLIP model"):
        VlpModel(tiny_clip.half(), CLASSNAMES, len(CLASSNAMES))


# --- teacher -----------------------------------------------------------------

def test_teacher_starts_identical_to_student(model):
    for teacher, student in ((model.teacher_encoder, model.encoder),
                             (model.teacher_head, model.head)):
        t, s = teacher.state_dict(), student.state_dict()
        assert set(t) == set(s)
        for key in s:
            assert torch.equal(t[key], s[key]), key


def test_teacher_params_need_no_grad(model):
    for module in (model.teacher_encoder, model.teacher_head):
        assert all(not p.requires_grad for p in module.parameters())


def test_teacher_is_fp32(model):
    for module in (model.teacher_encoder, model.teacher_head):
        for name, tensor in module.state_dict().items():
            if torch.is_floating_point(tensor):
                assert tensor.dtype is torch.float32, name


def test_model_train_cannot_flip_the_teacher(model):
    """nn.Module.train() recurses into every submodule; VlpModel overrides it so
    a call site cannot get this wrong."""
    model.train()
    assert model.encoder.training and model.head.training
    for module in (model.teacher_encoder, model.teacher_head):
        assert all(not m.training for m in module.modules())


def test_model_is_picklable_and_deepcopyable(model):
    """The guard is a class method, not a lambda bound onto the instance: a
    lambda would break torch.save(model) outright and would leave a deepcopy's
    guard pointing at the original module."""
    import copy as _copy
    import pickle

    pickle.loads(pickle.dumps(model))
    clone = _copy.deepcopy(model)
    clone.train()
    assert not clone.teacher_head.training
    assert clone.teacher_head is not model.teacher_head


def test_train_false_switches_the_student_off(model):
    model.train(True)
    assert model.head.training
    model.train(False)
    assert not model.head.training
    assert not model.teacher_head.training


def test_teacher_logits_use_the_ema_head_on_ema_features(model):
    images = torch.randn(2, 3, IMAGE, IMAGE)
    with torch.no_grad():
        expected = model.teacher_head(model.teacher_encoder.features(images))
        got = model.teacher_logits(images)
    assert torch.equal(got, expected)


def test_teacher_logits_carry_no_grad(model):
    logits = model.teacher_logits(torch.randn(1, 3, IMAGE, IMAGE))
    assert logits.requires_grad is False


# --- update ------------------------------------------------------------------

def test_ema_update_moves_both_halves(model):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.head.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        for p in model.encoder.visual_parameters():
            p.add_(torch.randn_like(p) * 0.01)

    head_before = model.teacher_head[2].weight.clone()
    visual_key = "model.visual.conv1.weight"
    encoder_before = model.teacher_encoder.state_dict()[visual_key].clone()

    model.ema_update(0.9)

    assert not torch.equal(model.teacher_head[2].weight, head_before)
    assert not torch.equal(model.teacher_encoder.state_dict()[visual_key], encoder_before)


def test_ema_update_at_momentum_zero_copies_the_student(model):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.head.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    model.ema_update(0.0)
    assert torch.equal(model.teacher_head[2].weight, model.head[2].weight)


@pytest.mark.parametrize("schedule", ["ramp", "const", "hard_copy_then_jump"])
def test_first_update_replaces_the_initialization(model, schedule):
    """momentum_at(0, ...) == 0 for every schedule, so the first update copies
    whatever the student is at that moment -- which, called after
    optimizer.step(), is the stepped weights and never the initialization."""
    cfg = EmaConfig(momentum=0.99, schedule=schedule)
    with torch.no_grad():
        model.head[2].weight.add_(1.0)
    model.ema_update(momentum_at(0, cfg))
    assert torch.equal(model.teacher_head[2].weight, model.head[2].weight)


def test_retokenize_updates_student_and_teacher(model):
    before_student = model.encoder.text_features.clone()
    before_teacher = model.teacher_encoder.text_features.clone()
    model.retokenize([f"a different {c}" for c in CLASSNAMES])
    assert not torch.allclose(model.encoder.text_features, before_student)
    assert not torch.allclose(model.teacher_encoder.text_features, before_teacher)
    assert torch.allclose(model.encoder.text_features, model.teacher_encoder.text_features)


def test_ema_update_does_not_propagate_text_features(model):
    """text_features is not in state_dict, so only retokenize() carries it."""
    model.encoder.retokenize([f"only student {c}" for c in CLASSNAMES])
    diverged = not torch.allclose(model.encoder.text_features,
                                  model.teacher_encoder.text_features)
    model.ema_update(0.5)
    assert diverged
    assert not torch.allclose(model.encoder.text_features,
                              model.teacher_encoder.text_features)


# --- param groups ------------------------------------------------------------

def test_param_groups_shape(model):
    groups = model.param_groups(lr=3e-6, head_multiplier=1000.0)
    assert len(groups) == 2
    assert groups[0]["lr"] == 3e-6
    assert groups[1]["lr"] == pytest.approx(3e-3)


def test_text_tower_and_logit_scale_are_frozen(model):
    tracked = {id(p) for g in model.param_groups(1.0, 1.0) for p in g["params"]}
    named = dict(model.encoder.model.named_parameters())
    for name in ("token_embedding.weight", "text_projection", "positional_embedding",
                 "ln_final.weight", "logit_scale"):
        assert id(named[name]) not in tracked, name
    for name, param in named.items():
        if name.startswith("visual."):
            assert id(param) in tracked, name


def test_head_group_keeps_the_frozen_batchnorm_bias(model):
    """Matches the param groups this replaces: the optimizer receives it and
    skips it, since its grad stays None."""
    head_group = model.param_groups(1.0, 1.0)[1]["params"]
    assert any(p is model.head[0].bias for p in head_group)
    assert model.head[0].bias.requires_grad is False


def test_teacher_params_are_absent_from_param_groups(model):
    tracked = {id(p) for g in model.param_groups(1.0, 1.0) for p in g["params"]}
    for module in (model.teacher_encoder, model.teacher_head):
        assert not any(id(p) in tracked for p in module.parameters())


def test_first_update_hard_copies_whatever_momentum_is_passed(model):
    """The guarantee does not depend on the caller passing the right momentum:
    the first update replaces the teacher even when asked for 0.996."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.head.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    assert model.teacher_is_initialized is False
    model.ema_update(0.996)
    assert model.teacher_is_initialized is True
    assert torch.equal(model.teacher_head[2].weight, model.head[2].weight)


def test_second_update_honours_the_momentum(model):
    model.ema_update(0.0)
    before = model.teacher_head[2].weight.clone()
    torch.manual_seed(1)
    with torch.no_grad():
        model.head[2].weight.add_(torch.randn_like(model.head[2].weight) * 0.1)
    model.ema_update(0.9)
    expected = 0.9 * before + 0.1 * model.head[2].weight
    assert torch.allclose(model.teacher_head[2].weight, expected, atol=0, rtol=1e-6)
    assert model.teacher_updates == 2


def test_teacher_updates_counts_every_call(model):
    for expected in range(1, 4):
        model.ema_update(0.99)
        assert model.teacher_updates == expected
