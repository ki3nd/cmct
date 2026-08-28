"""Smoke test for the branch-1 training script.

Wiring only. Six steps say nothing about whether the model learns, and this file
does not claim otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from cmct import train_lora
from cmct.config import load_experiment
from cmct.engine import lr_at

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENT = CONFIGS / "experiment" / "lora_officehome_a2c.yaml"


@pytest.fixture
def tiny_run(tmp_path, class_folder_root, monkeypatch):
    from conftest import build_tiny_clip

    raw = yaml.safe_load(EXPERIMENT.read_text())
    raw["data"].update(root=str(class_folder_root.parent), batch_size_test=4,
                       num_workers_test=0)
    raw["branches"][0]["stream"].update(batch_size_x=4, batch_size_u=4,
                                        num_workers=0)
    # tiny_clip is fp32; the model rejects a param_dtype that disagrees with it.
    raw["branches"][0]["backbone"]["dtype"] = "fp32"
    raw["branches"][0]["warmup_steps"] = 2
    raw["cotrain"]["total_macro_steps"] = 6
    raw["run"].update(output_dir=str(tmp_path / "out"), device="cpu",
                      print_freq=1, eval_freq=3)

    dataset = yaml.safe_load((CONFIGS / "dataset" / "officehome.yaml").read_text())
    dataset["dir"] = class_folder_root.name
    dataset["num_classes"] = 3
    dataset["transform"]["train"] = {"resize": [40, 40], "crop": "random",
                                     "crop_size": 32, "hflip": True}
    dataset["transform"]["test"] = {"resize": [32, 32], "crop": "none",
                                    "crop_size": 32, "hflip": False}
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "officehome.yaml").write_text(yaml.safe_dump(dataset))
    (tmp_path / "experiment").mkdir()
    config_path = tmp_path / "experiment" / "run.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    monkeypatch.setattr(train_lora, "load_clip",
                        lambda checkpoint, dtype: build_tiny_clip())
    monkeypatch.setattr(train_lora, "resolve_checkpoint", lambda c: c)
    return config_path, Path(raw["run"]["output_dir"])


def rows(output_dir):
    return [json.loads(line)
            for line in (output_dir / "metrics.jsonl").read_text().splitlines()]


def test_runs_and_writes_its_outputs(tiny_run):
    config_path, output_dir = tiny_run
    best = train_lora.main(["--config", str(config_path)])

    assert 0.0 <= best <= 100.0
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "run.json").is_file()
    assert (output_dir / "model-last.pt").is_file()

    recorded = rows(output_dir)
    assert recorded
    assert set(recorded[-1]) >= {"step", "teacher_acc", "student_acc", "source_ce",
                                 "pseudo_label", "mmd", "mask_ratio", "reference",
                                 "teacher_updates", "lr"}


def test_best_follows_the_teacher(tiny_run):
    """Branch 1 evaluates the teacher; branch 2 evaluates its student. Opposite,
    and deliberate."""
    config_path, output_dir = tiny_run
    best = train_lora.main(["--config", str(config_path)])
    assert best == pytest.approx(max(r["teacher_acc"] for r in rows(output_dir)))


def test_both_teacher_and_student_are_scored(tiny_run):
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    row = rows(output_dir)[-1]
    assert row["teacher_acc"] != row["student_acc"] or \
        row["teacher_loss"] != row["student_loss"]


def test_the_reference_switches_after_warmup(tiny_run):
    """warmup_steps is 2 here, and evaluation lands on steps 3 and 6, so every
    recorded row should already be past the switch."""
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    recorded = rows(output_dir)
    assert all(r["reference"] == "teacher" for r in recorded), \
        [(r["step"], r["reference"]) for r in recorded]


def test_the_reference_is_the_zero_shot_model_during_warmup(tiny_run):
    """Evaluate inside the warmup window to see the other branch of the switch."""
    config_path, output_dir = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    raw["run"]["eval_freq"] = 1
    raw["branches"][0]["warmup_steps"] = 4
    config_path.write_text(yaml.safe_dump(raw))

    train_lora.main(["--config", str(config_path)])
    by_step = {r["step"]: r["reference"] for r in rows(output_dir)}
    assert by_step[1] == "zero_shot"
    assert by_step[4] == "zero_shot"
    assert by_step[5] == "teacher"


def test_the_zero_shot_reference_is_released(tiny_run):
    config_path, _ = tiny_run
    train_lora.main(["--config", str(config_path)])
    assert train_lora.LAST_MODEL is not None
    assert not train_lora.LAST_MODEL.has_zero_shot


def test_no_zero_shot_model_is_built_when_there_is_no_warmup(tiny_run):
    config_path, _ = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    raw["branches"][0]["warmup_steps"] = 0
    config_path.write_text(yaml.safe_dump(raw))
    train_lora.main(["--config", str(config_path)])
    assert not train_lora.LAST_MODEL.has_zero_shot


def test_teacher_receives_one_update_per_step(tiny_run):
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    row = rows(output_dir)[-1]
    assert row["teacher_updates"] == row["step"]


def test_only_lora_weights_are_saved(tiny_run):
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    state = torch.load(output_dir / "model-last.pt", map_location="cpu",
                       weights_only=True)
    assert state and all("lora_" in k for k in state)


def test_the_logged_lr_follows_the_schedule(tiny_run):
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    cfg = load_experiment(config_path)
    branch = cfg.branches[0]
    total = cfg.cotrain.total_macro_steps * branch.steps_per_macro
    for row in rows(output_dir):
        expected = lr_at(row["step"] - 1, branch.optim, total, branch.warmup_steps)
        assert row["lr"] == pytest.approx(expected), row["step"]


def test_max_steps_does_not_shorten_the_schedule(tiny_run):
    """--max-steps is a script knob. The LR schedule and the CMKD-style ramp span
    total_macro_steps regardless, so a short run's first N steps match the real
    run's."""
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path), "--max-steps", "3"])
    derived = json.loads((output_dir / "run.json").read_text())
    assert derived["run_steps"] == 3
    assert derived["total_steps"] == 6


def test_logs_the_config_and_derived_values(tiny_run, capsys):
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    out = capsys.readouterr().out
    assert (output_dir / "config.yaml").read_text().strip() in out
    for token in ("total_steps", "warmup_steps", "lr_first_last",
                  "config_lr_unused", "prompt_template", "trainable_parameters",
                  "images", "device"):
        assert token in out, token


def test_run_json_records_that_lr_does_not_govern(tiny_run):
    """`lr` is in the config but does not set the learning rate under
    warmup_cosine. run.json records both so a reader is not misled."""
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    derived = json.loads((output_dir / "run.json").read_text())
    assert derived["config_lr_unused"] == 0.0035
    assert derived["lr_first_last"][0] == pytest.approx(0.001)


def test_prompt_examples_are_normalized(tiny_run):
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    derived = json.loads((output_dir / "run.json").read_text())
    assert derived["prompt_template"] == "a photo of a {}."
    assert derived["prompt_examples"][0] == "a photo of a alarm clock."


def test_unknown_extra_key_is_rejected(tiny_run):
    config_path, _ = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    raw["branches"][0]["extra"]["lora_paramz"] = [1]
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit, match="unknown key"):
        train_lora.main(["--config", str(config_path)])


def test_missing_extra_key_is_rejected(tiny_run):
    config_path, _ = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    del raw["branches"][0]["extra"]["mmd_weight"]
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit, match="missing key"):
        train_lora.main(["--config", str(config_path)])


def test_rejects_a_branch_of_the_wrong_type(tiny_run):
    config_path, _ = tiny_run
    with pytest.raises(SystemExit, match="no branch named"):
        train_lora.main(["--config", str(config_path), "--branch", "vlp"])
