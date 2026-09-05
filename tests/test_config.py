import textwrap

import pytest

from cmct.config import Config
from tests.conftest import read_text

CONFIG_PATH = "configs/officehome_a2c.yaml"


def test_shipped_config_parses_to_the_documented_effective_values():
    cfg = Config.from_yaml(CONFIG_PATH)
    assert cfg.seed == 42
    assert cfg.branch_lora.precision == "fp16"
    assert cfg.branch_lora.backbone.name == "ViT-B/16"
    assert cfg.branch_lora.backbone.path == "./assets"
    assert cfg.branch_mlp.backbone.name == "ViT-B/16"
    assert cfg.pseudo_label.threshold == 0.85
    assert cfg.branch_lora.lora.r == 2
    assert cfg.branch_lora.lora.rank_ramp == [2, 4, 6, 8, 10]
    assert cfg.branch_lora.warmup.lr == 0.006
    assert cfg.branch_lora.warmup.iters == 50
    assert cfg.branch_lora.ema_momentum == 0.99
    assert cfg.branch_mlp.warmup_iters == 500
    assert cfg.branch_mlp.self_reference == "own_teacher"
    assert cfg.branch_mlp.ema.schedule == "ramp"
    assert cfg.branch_mlp.ema.hard_copy_iters is None  # inert under "ramp", so absent
    assert cfg.train.iters == 1000
    assert cfg.train.mlp_steps_per_iter == 10
    assert cfg.data.batch_size.source == 32
    assert cfg.data.strong_aug is False
    assert cfg.pseudo_label.debias.enabled is False


def test_unknown_key_is_rejected_by_name(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH) + textwrap.dedent("""
        branch_mlp_typo:
          lr: 1.0
        """))
    with pytest.raises(ValueError, match="branch_mlp_typo"):
        Config.from_yaml(str(p))


def test_invalid_ema_schedule_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH).replace("schedule: ramp", "schedule: cosine"))
    with pytest.raises(ValueError, match="schedule"):
        Config.from_yaml(str(p))


def test_dassl_adapter_carries_batch_sizes_and_domains():
    from cmct.config import to_dassl_cfg

    dcfg = to_dassl_cfg(Config.from_yaml(CONFIG_PATH))
    assert dcfg.DATASET.NAME == "OfficeHome"
    assert dcfg.DATASET.SOURCE_DOMAINS == ["art"]
    assert dcfg.DATASET.TARGET_DOMAINS == ["clipart"]
    assert dcfg.DATALOADER.TRAIN_X.BATCH_SIZE == 32
    assert dcfg.DATALOADER.TRAIN_U.BATCH_SIZE == 32
    assert dcfg.DATALOADER.TEST.BATCH_SIZE == 128
    assert dcfg.DATALOADER.NUM_WORKERS == 8
    assert tuple(dcfg.INPUT.SIZE) == (224, 224)


def test_resolve_forces_cross_weight_to_zero_when_branch_lora_disabled(tmp_path):
    from cmct.config import resolve

    p = tmp_path / "disabled.yaml"
    p.write_text(read_text(CONFIG_PATH).replace("enabled: true", "enabled: false", 1))
    cfg = Config.from_yaml(str(p))
    assert cfg.branch_lora.enabled is False
    assert cfg.branch_mlp.cross_weight == 0.5  # unresolved value, unchanged by parsing

    resolved = resolve(cfg)
    assert resolved.branch_mlp.cross_weight == 0.0
    # Everything else about the resolved config is otherwise identical.
    assert resolved.branch_lora.enabled is False
    assert resolved.branch_mlp.lr == cfg.branch_mlp.lr


def test_resolve_is_a_no_op_when_branch_lora_is_enabled():
    from cmct.config import resolve

    cfg = Config.from_yaml(CONFIG_PATH)
    resolved = resolve(cfg)
    assert resolved == cfg


def test_unknown_key_at_a_nested_path_is_rejected_by_full_dotted_path(tmp_path):
    """_build recurses, so the nested arms need their own coverage: a typo two
    levels down must name where it is, not just that something is wrong."""
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH).replace(
        "  ema: {momentum: 0.99, schedule: ramp}",
        "  ema: {momentum: 0.99, schedule: ramp, momentom: 0.5}"))
    with pytest.raises(ValueError, match=r"branch_mlp\.ema\.momentom"):
        Config.from_yaml(str(p))


def test_missing_key_is_rejected_by_full_dotted_path(tmp_path):
    """A dropped key must fail loudly rather than inherit a dataclass default,
    since the whole point of this layer is that every effective value is
    written down in one file."""
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH).replace("  schedule: ramp\n", "")
                 .replace(", schedule: ramp", ""))
    with pytest.raises(ValueError, match=r"branch_mlp\.ema\.schedule"):
        Config.from_yaml(str(p))


def test_a_non_vit_backbone_is_rejected_for_the_lora_branch(tmp_path):
    """LoRA is injected into ViT attention blocks and the injection table covers
    ViT only, so a ResNet here has to fail at parse time rather than as a bare
    KeyError deep inside apply_lora."""
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH).replace(
        "    name: ViT-B/16          # must be a ViT", "    name: RN50          # must be a ViT"))
    with pytest.raises(ValueError, match=r"branch_lora\.backbone\.name"):
        Config.from_yaml(str(p))


def test_the_mlp_branch_accepts_a_resnet_backbone(tmp_path):
    """The MLP branch reads features through encode_image only, so any CLIP
    backbone is usable there -- including the ResNets, whose BatchNorm fix_bn
    then freezes."""
    p = tmp_path / "rn50.yaml"
    p.write_text(read_text(CONFIG_PATH).replace(
        "    name: ViT-B/16          # any CLIP backbone", "    name: RN50          # any CLIP backbone"))
    cfg = Config.from_yaml(str(p))
    assert cfg.branch_mlp.backbone.name == "RN50"
    assert cfg.branch_lora.backbone.name == "ViT-B/16"


def test_resolve_forces_cross_weight_to_zero_when_branch_mlp_disabled(tmp_path):
    """The mirror of the branch_lora case: with no CMKD teacher there is nothing
    for branch_lora to cross-teach from."""
    from cmct.config import resolve

    p = tmp_path / "lora_only.yaml"
    p.write_text(read_text(CONFIG_PATH).replace(
        "branch_mlp:\n  enabled: true", "branch_mlp:\n  enabled: false"))
    cfg = Config.from_yaml(str(p))
    assert cfg.branch_lora.cross_weight == 0.5  # unresolved value, unchanged by parsing

    resolved = resolve(cfg)
    assert resolved.branch_lora.cross_weight == 0.0
    assert resolved.branch_lora.enabled is True
    assert resolved.branch_mlp.cross_weight == 0.5  # the disabled branch is left alone


def test_disabling_both_branches_is_rejected(tmp_path):
    """Neither branch left to train is a config mistake, not a valid ablation:
    it would otherwise run the full macro-step loop doing nothing."""
    p = tmp_path / "neither.yaml"
    p.write_text(read_text(CONFIG_PATH).replace("enabled: true", "enabled: false"))
    with pytest.raises(ValueError, match="nothing left to train"):
        Config.from_yaml(str(p))


def test_hard_copy_iters_is_rejected_under_the_ramp_schedule(tmp_path):
    """`ema_momentum_at` never reads it under "ramp", so accepting it would let
    a number sit in the config file looking like it governs something."""
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH).replace(
        "schedule: ramp}", "schedule: ramp, hard_copy_iters: 100}"))
    with pytest.raises(ValueError, match="hard_copy_iters"):
        Config.from_yaml(str(p))


def test_hard_copy_iters_is_required_under_the_hard_copy_schedule(tmp_path):
    """The mirror: that schedule has no window without it."""
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH).replace("schedule: ramp}", "schedule: hard_copy}"))
    with pytest.raises(ValueError, match="hard_copy_iters"):
        Config.from_yaml(str(p))


def test_the_hard_copy_schedule_takes_its_own_window(tmp_path):
    p = tmp_path / "hard_copy.yaml"
    p.write_text(read_text(CONFIG_PATH).replace(
        "schedule: ramp}", "schedule: hard_copy, hard_copy_iters: 50}"))
    cfg = Config.from_yaml(str(p))
    assert cfg.branch_mlp.ema.schedule == "hard_copy"
    assert cfg.branch_mlp.ema.hard_copy_iters == 50


@pytest.mark.parametrize("source", ["own_clip", "own_teacher", "lora_teacher"])
def test_every_self_reference_source_parses(tmp_path, source):
    p = tmp_path / f"{source}.yaml"
    p.write_text(read_text(CONFIG_PATH).replace("self_reference: own_teacher", f"self_reference: {source}"))
    assert Config.from_yaml(str(p)).branch_mlp.self_reference == source


def test_an_unknown_self_reference_source_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH).replace("self_reference: own_teacher", "self_reference: frozen"))
    with pytest.raises(ValueError, match="self_reference"):
        Config.from_yaml(str(p))


def test_the_lora_teacher_source_needs_the_lora_branch(tmp_path):
    """train.py would call teacher_lora(...) on None otherwise."""
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH)
                 .replace("self_reference: own_teacher", "self_reference: lora_teacher")
                 .replace("enabled: true", "enabled: false", 1))
    with pytest.raises(ValueError, match="lora_teacher"):
        Config.from_yaml(str(p))
