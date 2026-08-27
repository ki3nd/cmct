from pathlib import Path

import pytest
import yaml

from cmct.config import ConfigError, dump, load_experiment, load_resolved

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENT = CONFIGS / "experiment" / "cmct_officehome_a2c.yaml"


def write_experiment(tmp_path, mutate=None, dataset_name="officehome"):
    """Write a temporary experiment YAML, optionally mutated first. config_root
    still points at the real configs/, so dataset YAMLs are not duplicated."""
    raw = yaml.safe_load(EXPERIMENT.read_text())
    if mutate is not None:
        mutate(raw)
    raw["dataset"] = dataset_name
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def load_tmp(tmp_path, mutate=None, dataset_name="officehome"):
    return load_experiment(write_experiment(tmp_path, mutate, dataset_name),
                           config_root=CONFIGS)


def test_loads_shipped_experiment():
    cfg = load_experiment(EXPERIMENT)
    assert cfg.dataset.name == "officehome"
    assert cfg.dataset.num_classes == 65
    assert cfg.data.source_domains == ["art"]
    assert cfg.data.target_domains == ["clipart"]
    assert cfg.cotrain.cross_ref_refresh == "macro"
    assert [b.name for b in cfg.branches] == ["lora", "vlp"]


def test_dataset_yaml_resolved_relative_to_experiment_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_experiment(EXPERIMENT)
    assert cfg.dataset.dir == "office_home"


def test_per_branch_backbone_and_dtype():
    cfg = load_experiment(EXPERIMENT)
    lora, vlp = cfg.branches
    assert lora.backbone.dtype == "fp16"
    assert vlp.backbone.dtype == "fp32"
    assert lora.steps_per_macro == 1
    assert vlp.steps_per_macro == 10


def test_per_branch_defaults_are_independent():
    cfg = load_experiment(EXPERIMENT)
    lora, vlp = cfg.branches
    assert lora.optim.scheduler == "cosine"
    assert vlp.optim.scheduler == "inv"
    assert lora.ema.schedule == "const"
    assert vlp.ema.schedule == "dacs"
    assert vlp.optim.param_group_multipliers == {"classifier": 1000.0}


def test_extra_carries_branch_private_knobs():
    cfg = load_experiment(EXPERIMENT)
    lora, vlp = cfg.branches
    assert lora.extra["mmd_weight"] == 1.0
    assert vlp.extra["lambda1"] == 0.25
    assert "mmd_weight" not in vlp.extra


def test_unknown_top_level_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown top-level key"):
        load_tmp(tmp_path, lambda r: r.update(trainer="PHPL"))


def test_unknown_nested_key(tmp_path):
    with pytest.raises(ConfigError, match=r"data: unknown key\(s\) \['num_classes'\]"):
        load_tmp(tmp_path, lambda r: r["data"].update(num_classes=65))


def test_unknown_branch_key(tmp_path):
    with pytest.raises(ConfigError, match=r"branches\[0\]: unknown key"):
        load_tmp(tmp_path, lambda r: r["branches"][0].update(arch="RN101"))


def test_missing_required_field(tmp_path):
    with pytest.raises(ConfigError, match=r"data: missing required field\(s\) \['root'\]"):
        load_tmp(tmp_path, lambda r: r["data"].pop("root"))


def test_missing_cross_ref_refresh(tmp_path):
    with pytest.raises(ConfigError, match="cross_ref_refresh"):
        load_tmp(tmp_path, lambda r: r["cotrain"].pop("cross_ref_refresh"))


def test_missing_branch_dtype(tmp_path):
    with pytest.raises(ConfigError, match="dtype"):
        load_tmp(tmp_path, lambda r: r["branches"][0]["backbone"].pop("dtype"))


def test_bad_dtype(tmp_path):
    with pytest.raises(ConfigError, match=r"'fp8' is not a valid value"):
        load_tmp(tmp_path, lambda r: r["branches"][0]["backbone"].update(dtype="fp8"))


def test_bad_cross_ref_refresh(tmp_path):
    with pytest.raises(ConfigError, match="is not a valid value"):
        load_tmp(tmp_path, lambda r: r["cotrain"].update(cross_ref_refresh="epoch"))


def test_domain_not_in_dataset(tmp_path):
    with pytest.raises(ConfigError, match="not in dataset 'officehome'"):
        load_tmp(tmp_path, lambda r: r["data"].update(source_domains=["sketch"]))


def test_source_target_overlap(tmp_path):
    with pytest.raises(ConfigError, match="overlap"):
        load_tmp(tmp_path, lambda r: r["data"].update(target_domains=["art"]))


def test_empty_domain_list(tmp_path):
    with pytest.raises(ConfigError, match="must not be empty"):
        load_tmp(tmp_path, lambda r: r["data"].update(source_domains=[]))


def test_duplicate_branch_names(tmp_path):
    def mutate(r):
        r["branches"][1]["name"] = r["branches"][0]["name"]
    with pytest.raises(ConfigError, match="duplicate names"):
        load_tmp(tmp_path, mutate)


def test_no_branches(tmp_path):
    with pytest.raises(ConfigError, match="at least one branch is required"):
        load_tmp(tmp_path, lambda r: r.update(branches=[]))


def test_steps_per_macro_must_be_positive(tmp_path):
    with pytest.raises(ConfigError, match="steps_per_macro"):
        load_tmp(tmp_path, lambda r: r["branches"][0].update(steps_per_macro=0))


def test_threshold_out_of_range(tmp_path):
    with pytest.raises(ConfigError, match="threshold"):
        load_tmp(tmp_path, lambda r: r["branches"][0]["pseudo_label"].update(threshold=1.5))


def test_unknown_dataset_name(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_tmp(tmp_path, dataset_name="cifar10")


def test_wrong_type_raises(tmp_path):
    with pytest.raises(ConfigError, match="expected int"):
        load_tmp(tmp_path, lambda r: r["run"].update(seed="forty two"))


def test_int_accepted_where_float_expected(tmp_path):
    cfg = load_tmp(tmp_path, lambda r: r["branches"][0]["optim"].update(lr=1))
    assert cfg.branches[0].optim.lr == 1.0
    assert isinstance(cfg.branches[0].optim.lr, float)


def test_dump_then_load_resolved_round_trips(tmp_path):
    cfg = load_experiment(EXPERIMENT)
    out = dump(cfg, tmp_path)
    assert out.is_file()
    assert load_resolved(out) == cfg


def test_load_resolved_rejects_experiment_form():
    with pytest.raises(ConfigError, match="not a resolved config"):
        load_resolved(EXPERIMENT)


def test_pixel_stats_become_tuples():
    cfg = load_experiment(EXPERIMENT)
    assert isinstance(cfg.dataset.pixel_mean, tuple)
    assert len(cfg.dataset.pixel_std) == 3


def test_classname_overrides_present_but_off_by_default():
    cfg = load_experiment(EXPERIMENT)
    assert cfg.data.clarify_classnames is False
    assert cfg.dataset.classname_overrides["mouse"] == "computer mouse"


def test_every_dataset_yaml_loads(tmp_path):
    for path in sorted((CONFIGS / "dataset").glob("*.yaml")):
        name = path.stem
        spec = yaml.safe_load(path.read_text())

        def mutate(r, dom=spec["domains"]):
            r["data"]["source_domains"] = [dom[0]]
            r["data"]["target_domains"] = [dom[1]]

        cfg = load_experiment(
            write_experiment(tmp_path, mutate, dataset_name=name), config_root=CONFIGS
        )
        assert cfg.dataset.name == name
        assert cfg.dataset.num_classes == spec["num_classes"]
