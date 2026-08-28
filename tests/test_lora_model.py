"""Branch 1's model: LoRA student, EMA teacher, frozen zero-shot reference."""
from __future__ import annotations

import copy

import pytest
import torch

from cmct.backbones.lora import LoRALinear, lora_parameters
from cmct.branches.lora_model import TEMPLATE, LoraModel

CLASSNAMES = ["alarm clock", "backpack", "computer mouse"]
IMAGE = 32          # tiny_clip's input resolution


def build(tiny_clip, **overrides):
    kwargs = dict(rank=2, alpha=1.0, dropout=0.25, params=("q", "k", "v"),
                  rank_ramp=None, positions="all", encoders=("text", "vision"),
                  param_dtype=torch.float32)
    kwargs.update(overrides)
    return LoraModel(tiny_clip, CLASSNAMES, **kwargs)


@pytest.fixture
def model(tiny_clip):
    return build(tiny_clip)


def fresh_clip():
    """A CLIP that has never seen apply_lora. The `tiny_clip` fixture handed to
    build() is mutated in place, so a copy of it is the student, not a reference."""
    from conftest import build_tiny_clip

    return build_tiny_clip()


def images(n=2):
    torch.manual_seed(0)
    return torch.randn(n, 3, IMAGE, IMAGE)


# --- structure ---------------------------------------------------------------

def test_only_lora_parameters_train(model):
    trainable = {n for n, p in model.student.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in n for n in trainable)
    assert trainable == {n for n, _ in lora_parameters(model.student)}


def test_lora_lands_in_both_towers(model):
    names = {n for n, _ in lora_parameters(model.student)}
    assert any(n.startswith("transformer.") for n in names), "no text-tower LoRA"
    assert any(n.startswith("visual.") for n in names), "no vision-tower LoRA"


def test_teacher_is_an_exact_copy_including_frozen_weights(model):
    """The reference builds its teacher from a second read of the checkpoint and
    syncs only the LoRA keys, so its frozen SVD residual matches the student's
    only because the SVD happens to be deterministic. A deepcopy does not rely on
    that, and this test states it."""
    student_state = model.student.state_dict()
    teacher_state = model.teacher.state_dict()
    assert set(student_state) == set(teacher_state)
    for key, value in student_state.items():
        assert torch.equal(value, teacher_state[key]), key


def test_teacher_takes_no_gradient(model):
    assert all(not p.requires_grad for p in model.teacher.parameters())


def test_param_groups_hold_only_lora(model):
    groups = model.param_groups(lr=0.0035)
    assert len(groups) == 1
    assert groups[0]["lr"] == 0.0035
    tracked = {id(p) for p in groups[0]["params"]}
    assert tracked == {id(p) for _, p in lora_parameters(model.student)}


# --- mode --------------------------------------------------------------------

def test_student_is_in_train_mode_right_after_construction(model):
    """Set explicitly. In the reference this held only because freshly built LoRA
    modules default to training=True while the surrounding CLIP arrived in eval,
    and nothing ever called .train()."""
    lora_layers = [m for m in model.student.modules() if isinstance(m, LoRALinear)]
    assert lora_layers
    assert all(m.training for m in lora_layers)


def test_teacher_stays_in_eval_through_train(model):
    model.train()
    assert all(not m.training for m in model.teacher.modules())
    model.train(False)
    assert all(not m.training for m in model.teacher.modules())


def test_teacher_logits_are_deterministic(model):
    """Dropout is off for the teacher, so pseudo-labels do not wobble between
    calls."""
    model.train()
    batch = images()
    assert torch.equal(model.teacher_logits(batch), model.teacher_logits(batch))


def test_teacher_logits_carry_no_grad(model):
    assert model.teacher_logits(images()).requires_grad is False


def test_train_false_makes_the_student_deterministic(model):
    batch = images()
    model.train(False)
    assert torch.allclose(model.logits(batch), model.logits(batch))


# --- text features are live --------------------------------------------------

def test_student_logits_differ_between_calls_in_train_mode(model):
    """LoRA dropout is active on both towers, so repeated calls differ. This is
    why the text embedding cannot be reused across calls within a step."""
    model.train()
    batch = images()
    torch.manual_seed(1)
    first = model.logits(batch)
    torch.manual_seed(2)
    second = model.logits(batch)
    assert not torch.allclose(first, second)


def test_text_tower_lora_changes_the_logits(model):
    """Proves the text embedding is recomputed rather than cached.

    Zeroing every text-tower LoRA factor removes the whole text-side LoRA
    contribution, so the logits must move. A small perturbation of a single
    factor is not enough to distinguish the two designs -- measured at 9.5e-07 on
    this model, which is indistinguishable from float noise; zeroing all six moves
    the logits by ~4.0.
    """
    model.train(False)
    batch = images()
    before = model.logits(batch).clone()
    text_lora = [p for n, p in lora_parameters(model.student)
                 if n.startswith("transformer.")]
    assert text_lora
    with torch.no_grad():
        for param in text_lora:
            param.zero_()
    delta = float((before - model.logits(batch)).detach().abs().max())
    assert delta > 1e-2, delta


def test_text_tower_lora_receives_a_gradient(model):
    """The consequence of computing text features live: 40% of the trainable
    parameters live in the text tower, and a cached embedding would leave them
    with no gradient path at all."""
    model.train(False)
    model.logits(images()).sum().backward()
    text_lora = {n: p for n, p in lora_parameters(model.student)
                 if n.startswith("transformer.")}
    assert text_lora
    for name, param in text_lora.items():
        assert param.grad is not None, name
        assert torch.any(param.grad != 0), name


def test_no_cached_text_features_field(model):
    assert not hasattr(model, "text_features_cache")
    keys = set(model.state_dict())
    assert not [k for k in keys if k.endswith(".text_features")]


# --- prompts -----------------------------------------------------------------

def test_template_has_the_trailing_period():
    assert TEMPLATE == "a photo of a {}."


def test_prompts_differ_from_branch_two():
    from cmct.branches.vlp_model import TEMPLATE as VLP_TEMPLATE

    ours = [TEMPLATE.format(c) for c in CLASSNAMES]
    theirs = [VLP_TEMPLATE.format(c) for c in CLASSNAMES]
    assert ours != theirs
    assert ours[0] == "a photo of a alarm clock."


def test_the_period_changes_the_tokenization():
    from cmct.backbones.clip import tokenize

    with_period = tokenize(["a photo of a alarm clock."])[0]
    without = tokenize(["a photo of a alarm clock"])[0]
    assert not torch.equal(with_period, without)
    assert int((with_period != 0).sum()) != int((without != 0).sum())


def test_tokenized_prompts_are_not_in_the_state_dict(model):
    """A non-persistent buffer: it moves with .to() but a checkpoint does not
    carry it, since it is rebuilt from the class names."""
    assert "tokenized_prompts" in model._buffers
    assert not [k for k in model.state_dict() if k.endswith("tokenized_prompts")]


# --- EMA ---------------------------------------------------------------------

def perturb_student_lora(model, scale=0.1, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for _, param in lora_parameters(model.student):
            param.add_(torch.randn_like(param) * scale)


def test_ema_touches_only_lora_parameters(model):
    frozen_before = {k: v.clone() for k, v in model.teacher.state_dict().items()
                     if "lora_" not in k}
    perturb_student_lora(model)
    model.ema_update(0.9)
    after = model.teacher.state_dict()
    for key, value in frozen_before.items():
        assert torch.equal(value, after[key]), key


def test_first_update_hard_copies_whatever_momentum_is_passed(model):
    perturb_student_lora(model)
    assert model.teacher_updates == 0
    assert model.teacher_is_initialized is False
    model.ema_update(0.996)
    teacher_state = model.teacher.state_dict()
    for name, param in lora_parameters(model.student):
        assert torch.equal(param.detach(), teacher_state[name]), name
    assert model.teacher_updates == 1
    assert model.teacher_is_initialized is True


def test_second_update_honours_the_momentum(model):
    model.ema_update(0.0)
    name, _ = next(iter(lora_parameters(model.student)))
    before = model.teacher.state_dict()[name].clone()
    perturb_student_lora(model, seed=1)
    model.ema_update(0.9)
    expected = 0.9 * before + 0.1 * dict(lora_parameters(model.student))[name].detach()
    assert torch.allclose(model.teacher.state_dict()[name], expected, rtol=1e-6)


def test_momentum_outside_the_unit_interval_raises(model):
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="momentum must be within"):
            model.ema_update(bad)


@pytest.mark.parametrize("momentum", [0.99, 0.996])
def test_ema_tracks_the_theoretical_rate(model, momentum):
    """The gap after n steps is start * momentum ** n in exact arithmetic. The
    factors are float32 so this holds within an order of magnitude; float16 would
    stall at step ~97 with most of the gap still open."""
    steps = 1000
    model.ema_update(0.0)                    # initialize
    perturb_student_lora(model, scale=0.5, seed=2)

    def gap():
        teacher_state = model.teacher.state_dict()
        return sum(float((p.detach() - teacher_state[n]).abs().sum())
                   for n, p in lora_parameters(model.student))

    start = gap()
    assert start > 0
    for _ in range(steps):
        model.ema_update(momentum)
    assert gap() / start < 10 * momentum ** steps


def test_ema_refuses_a_float16_teacher(model):
    """Mutated through named_parameters, not state_dict: state_dict returns
    param.detach(), so rebinding .data on the returned tensor leaves the
    parameter's dtype untouched. In-place mul_/add_ does reach it, which is why
    the EMA itself works through state_dict."""
    name, _ = next(iter(lora_parameters(model.student)))
    teacher_params = dict(model.teacher.named_parameters())
    teacher_params[name].data = teacher_params[name].data.half()
    assert dict(model.teacher.named_parameters())[name].dtype is torch.float16
    with pytest.raises(TypeError, match="must be float32"):
        model.ema_update(0.99)


def test_lora_factors_are_float32(model):
    for _, param in lora_parameters(model.student):
        assert param.dtype is torch.float32
    for name, tensor in model.teacher.state_dict().items():
        if "lora_" in name:
            assert tensor.dtype is torch.float32, name


# --- zero-shot reference -----------------------------------------------------

def test_zero_shot_differs_from_the_teacher(tiny_clip):
    model = build(tiny_clip)
    model.attach_zero_shot(fresh_clip())
    model.train(False)
    batch = images()
    assert not torch.allclose(model.zero_shot_logits(batch),
                              model.teacher_logits(batch))


def test_zero_shot_uses_the_same_prompts(tiny_clip):
    """Built from the same class names and template, so its prompt tensor is the
    student's. Asserted rather than forced by assignment."""
    model = build(tiny_clip)
    model.attach_zero_shot(fresh_clip())
    assert torch.equal(model.zero_shot.text, model.tokenized_prompts)


def test_zero_shot_has_no_lora(tiny_clip):
    model = build(tiny_clip)
    model.attach_zero_shot(fresh_clip())
    assert not list(lora_parameters(model.zero_shot))
    assert all(not p.requires_grad for p in model.zero_shot.parameters())


def test_release_zero_shot_then_using_it_raises(tiny_clip):
    model = build(tiny_clip)
    model.attach_zero_shot(fresh_clip())
    assert model.has_zero_shot
    model.release_zero_shot()
    assert not model.has_zero_shot
    with pytest.raises(RuntimeError, match="zero-shot reference"):
        model.zero_shot_logits(images())


def test_attach_zero_shot_rejects_a_model_that_already_has_lora(tiny_clip):
    """apply_lora mutates in place, so a copy of the model handed to __init__ is
    the student. Accepting it would give a second trainable branch dressed as a
    zero-shot reference, and its pseudo-labels would look plausible."""
    model = build(tiny_clip)
    with pytest.raises(ValueError, match="needs a CLIP with no LoRA"):
        model.attach_zero_shot(copy.deepcopy(tiny_clip))


def test_using_the_zero_shot_before_attaching_raises(model):
    assert not model.has_zero_shot
    with pytest.raises(RuntimeError, match="zero-shot reference"):
        model.zero_shot_logits(images())


# --- forward shapes ----------------------------------------------------------

def test_forward_returns_logits_and_normalized_features(model):
    model.train(False)
    logits, features = model(images(4))
    assert logits.shape == (4, len(CLASSNAMES))
    norms = features.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_features_can_be_returned_unnormalized(model):
    model.train(False)
    raw = model.features(images(4), normalize=False)
    norms = raw.norm(dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_logit_scale_is_applied(model):
    model.train(False)
    batch = images(3)
    logits, features = model(batch)
    text = model.text_features()
    scale = model.student.logit_scale.exp().detach()
    assert torch.allclose(logits, scale * features @ text.t(), atol=1e-4)


def test_classname_count_drives_the_output_width(tiny_clip):
    model = build(tiny_clip)
    assert model.num_classes == len(CLASSNAMES)
    model.train(False)
    assert model.logits(images(2)).shape[1] == len(CLASSNAMES)


# --- save / load -------------------------------------------------------------

def test_lora_state_round_trips(model):
    saved = model.lora_state_dict()
    assert saved and all("lora_" in k for k in saved)
    perturb_student_lora(model)
    model.load_lora_state_dict(saved)
    for key, value in model.lora_state_dict().items():
        assert torch.equal(value, saved[key]), key


def test_lora_state_excludes_the_frozen_weights(model):
    saved = model.lora_state_dict()
    assert not [k for k in saved if k.endswith(".weight") and "lora_" not in k]


def test_model_is_picklable_and_deepcopyable(model):
    import pickle

    pickle.loads(pickle.dumps(model))
    clone = copy.deepcopy(model)
    clone.train()
    assert all(not m.training for m in clone.teacher.modules())
    assert clone.teacher is not model.teacher


def test_param_dtype_must_match_the_clip_it_is_given(tiny_clip):
    """One config field reaches the model through two paths -- load_clip() for the
    whole CLIP and param_dtype for the LoRA layers. A mismatch used to surface as
    a bare `mat1 and mat2 must have the same dtype` mid-forward, naming neither
    the field nor the file."""
    with pytest.raises(TypeError, match="branches\\[\\].backbone.dtype"):
        build(tiny_clip, param_dtype=torch.float16)


def test_matching_fp16_is_accepted(tiny_clip):
    model = build(tiny_clip.half(), param_dtype=torch.float16)
    from cmct.backbones.lora import LoRALinear
    layer = next(m for m in model.student.modules() if isinstance(m, LoRALinear))
    assert layer.weight.dtype is torch.float16
    assert layer.lora_A.dtype is torch.float32
