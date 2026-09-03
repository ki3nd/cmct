import textwrap

import pytest

from cmct.config import Config
from tests.conftest import read_text

CONFIG_PATH = "configs/officehome_a2c.yaml"


def test_shipped_config_parses_to_the_documented_effective_values():
    cfg = Config.from_yaml(CONFIG_PATH)
    assert cfg.seed == 42
    assert cfg.branch_lora.precision == "fp16"
    assert cfg.pseudo_label.threshold == 0.85
    assert cfg.branch_lora.lora.r == 2
    assert cfg.branch_lora.lora.rank_ramp == [2, 4, 6, 8, 10]
    assert cfg.branch_lora.warmup.lr == 0.006
    assert cfg.branch_lora.warmup.iters == 50
    assert cfg.branch_lora.ema_momentum == 0.99
    assert cfg.branch_mlp.warmup_iters == 500
    assert cfg.branch_mlp.self_from_teacher is True
    assert cfg.branch_mlp.ema.schedule == "dacs"
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
    p.write_text(read_text(CONFIG_PATH).replace("schedule: dacs", "schedule: cosine"))
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
        "  ema: {momentum: 0.99, schedule: dacs, hard_copy_iters: 100}",
        "  ema: {momentum: 0.99, schedule: dacs, hard_copy_iters: 100, momentom: 0.5}"))
    with pytest.raises(ValueError, match=r"branch_mlp\.ema\.momentom"):
        Config.from_yaml(str(p))


def test_missing_key_is_rejected_by_full_dotted_path(tmp_path):
    """A dropped key must fail loudly rather than inherit a dataclass default,
    since the whole point of this layer is that every effective value is
    written down in one file."""
    p = tmp_path / "bad.yaml"
    p.write_text(read_text(CONFIG_PATH).replace("  schedule: dacs\n", "")
                 .replace("schedule: dacs, ", ""))
    with pytest.raises(ValueError, match=r"branch_mlp\.ema\.schedule"):
        Config.from_yaml(str(p))
