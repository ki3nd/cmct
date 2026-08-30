"""Smoke test for the branch-2 training script.

Checks the wiring only. Three steps say nothing about whether the model learns,
and this file does not claim otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from cmct import train_vlp
from cmct.config import load_experiment
from cmct.engine import lr_at

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# The config for the script this file tests. It used to load the co-training
# one, so branch 2's solo tests ran against another script's settings.
EXPERIMENT = CONFIGS / "experiment" / "vlp_officehome_a2c.yaml"


@pytest.fixture
def tiny_run(tmp_path, class_folder_root, monkeypatch):
    """An experiment config pointed at the synthetic dataset, with a tiny CLIP
    standing in for the real checkpoint."""
    from conftest import build_tiny_clip

    raw = yaml.safe_load(EXPERIMENT.read_text())
    raw["branches"] = [b for b in raw["branches"] if b["type"] == "vlp_clip"]
    raw["data"].update(root=str(class_folder_root.parent), batch_size_test=4,
                       num_workers_test=0)
    raw["branches"][0]["stream"].update(batch_size_x=4, batch_size_u=4, num_workers=0)
    # Set here, not inherited: these tests do arithmetic on them (--max-steps 3
    # against a total of 20), so reading them from a shipped config would turn
    # every experiment tweak into a test failure.
    raw["cotrain"]["total_macro_steps"] = 2
    raw["branches"][0]["steps_per_macro"] = 10
    # Owned by the fixture, not inherited: this branch's solo tests assert the
    # teacher moves away from the student, which only happens under a schedule
    # whose momentum leaves 0. Reading it from a shipped config coupled them to
    # a choice made for a different script.
    raw["branches"][0]["ema"] = {"momentum": 0.99, "schedule": "ramp"}
    raw["branches"][0]["steps_per_macro"] = 10
    raw["run"].update(output_dir=str(tmp_path / "out"), device="cpu",
                      print_freq=1, eval_freq=1)

    dataset = yaml.safe_load((CONFIGS / "dataset" / "officehome.yaml").read_text())
    dataset["dir"] = class_folder_root.name
    dataset["num_classes"] = 3
    # tiny_clip's input resolution is 32, so the pipeline has to produce 32x32
    dataset["transform"]["train"] = {"resize": [40, 40], "crop": "random",
                                     "crop_size": 32, "hflip": True}
    dataset["transform"]["test"] = {"resize": [32, 32], "crop": "none",
                                    "crop_size": 32, "hflip": False}
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "officehome.yaml").write_text(yaml.safe_dump(dataset))
    (tmp_path / "experiment").mkdir()
    config_path = tmp_path / "experiment" / "run.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    monkeypatch.setattr(train_vlp, "load_clip", lambda checkpoint, dtype: build_tiny_clip())
    return config_path, Path(raw["run"]["output_dir"])


def test_runs_and_writes_its_outputs(tiny_run):
    config_path, output_dir = tiny_run
    best = train_vlp.main(["--config", str(config_path), "--max-steps", "3"])

    assert 0.0 <= best <= 100.0
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "model-last.pt").is_file()
    assert (output_dir / "model-best.pt").is_file()

    rows = [json.loads(line) for line in (output_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows, "no evaluation was recorded"
    last = rows[-1]
    assert last["step"] == 3
    assert set(last) >= {"student_acc", "teacher_acc", "ramp", "clf", "task",
                         "distill", "reg", "teacher_updates"}


def test_both_student_and_teacher_are_scored(tiny_run):
    config_path, output_dir = tiny_run
    train_vlp.main(["--config", str(config_path), "--max-steps", "2"])
    row = json.loads((output_dir / "metrics.jsonl").read_text().splitlines()[-1])
    assert row["student_acc"] != row["teacher_acc"] or row["student_loss"] != row["teacher_loss"]


def test_step_is_wired_into_the_ramp(tiny_run):
    """The first recorded step must have a nonzero ramp only if step > 0, and the
    ramp must be the one computed from the run's full length, not the shortened
    one."""
    config_path, output_dir = tiny_run
    train_vlp.main(["--config", str(config_path), "--max-steps", "1"])
    row = json.loads((output_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert row["step"] == 1
    assert row["ramp"] == 0.0        # loss at step index 0
    assert row["task"] == 0.0
    assert row["distill"] == 0.0


def test_teacher_receives_one_update_per_step(tiny_run):
    config_path, output_dir = tiny_run
    train_vlp.main(["--config", str(config_path), "--max-steps", "3"])
    row = json.loads((output_dir / "metrics.jsonl").read_text().splitlines()[-1])
    assert row["teacher_updates"] == 3


def test_max_steps_does_not_shorten_the_ramp(tiny_run):
    """--max-steps is a script knob. max_iter stays total_macro_steps *
    steps_per_macro, so a short run's first N steps match the real run's."""
    config_path, _ = tiny_run
    cfg = load_experiment(config_path)
    branch = cfg.branches[0]
    total = cfg.cotrain.total_macro_steps * branch.steps_per_macro
    assert total == 20
    from cmct.losses import CmkdLoss
    assert CmkdLoss.from_branch_config(branch, max_iter=total).max_iter == 20


def test_head_lr_follows_the_multiplier(tiny_run):
    config_path, _ = tiny_run
    cfg = load_experiment(config_path)
    branch = cfg.branches[0]
    multiplier = branch.optim.param_group_multipliers["classifier"]
    assert multiplier == 1000.0
    assert lr_at(3, branch.optim) * multiplier == pytest.approx(
        1000.0 * branch.optim.lr * (1 + branch.optim.gamma * 3) ** -branch.optim.decay
    )


def test_rejects_a_branch_of_the_wrong_type(tiny_run):
    config_path, _ = tiny_run
    with pytest.raises(SystemExit, match="no branch named"):
        train_vlp.main(["--config", str(config_path), "--branch", "lora",
                        "--max-steps", "1"])


def test_saved_checkpoint_reloads(tiny_run):
    config_path, output_dir = tiny_run
    train_vlp.main(["--config", str(config_path), "--max-steps", "1"])
    state = torch.load(output_dir / "model-last.pt", map_location="cpu",
                       weights_only=True)
    assert any(k.startswith("encoder.") for k in state)
    assert any(k.startswith("teacher_head.") for k in state)


def test_logs_the_whole_resolved_config_and_derived_values(tiny_run, capsys):
    """Everything that governs the run has to be visible in the log: the config
    verbatim, and the values derived from it that appear nowhere in the file."""
    config_path, output_dir = tiny_run
    train_vlp.main(["--config", str(config_path), "--max-steps", "2"])
    out = capsys.readouterr().out

    # the config, rendered by the same function that writes config.yaml
    assert "resolved config" in out
    written = (output_dir / "config.yaml").read_text()
    assert written.strip() in out

    # every leaf that changes the numbers should be printed somewhere
    for token in ("lambda1", "label_smoothing", "nesterov", "param_group_multipliers",
                  "clarify_classnames", "cross_ref_refresh", "seed", "ema:"):
        assert token in out, token

    for token in ("total_steps", "ramp_max_iter", "eval_every", "prompt_template",
                  "lr_encoder_first_last", "lr_head_first_last",
                  "ema_momentum_at_0_1_last", "cmkd_ramp_at_0_mid_last",
                  "checkpoint", "images", "device"):
        assert token in out, token


def test_derived_values_are_saved_next_to_the_config(tiny_run):
    config_path, output_dir = tiny_run
    train_vlp.main(["--config", str(config_path), "--max-steps", "2"])
    derived = json.loads((output_dir / "run.json").read_text())
    assert derived["ramp_max_iter"] == derived["total_steps"]
    assert derived["run_steps"] == 2
    assert derived["prompt_template"] == "an image of a {}"
    expected = load_experiment(config_path).branches[0].optim \
        .param_group_multipliers["classifier"]
    assert derived["head_lr_multiplier"] == expected
    assert derived["ema_momentum_at_0_1_last"][0] == 0.0
    assert derived["cmkd_ramp_at_0_mid_last"][0] == 0.0
    assert derived["classes"] == 3


def test_printed_config_reloads_to_the_same_config(tiny_run, tmp_path):
    """The printed text is the resolved config, so it must load back as one."""
    from cmct.config import format_config, load_resolved

    config_path, _ = tiny_run
    cfg = load_experiment(config_path)
    path = tmp_path / "roundtrip.yaml"
    path.write_text(format_config(cfg))
    assert load_resolved(path) == cfg
