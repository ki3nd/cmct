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
from cmct.config.schema import EmaConfig
from cmct.engine import lr_at
from cmct.engine.evaluator import EvalResult

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
    """warmup_steps is 2, and evaluation lands on steps 2 (the boundary), 3 and
    6. Step 2 is the LAST warmup step, so it still reports the zero-shot
    reference; everything after it reports the teacher."""
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    recorded = rows(output_dir)
    seen = {r["step"]: r["reference"] for r in recorded}
    assert seen[2] == "zero_shot", seen
    assert all(ref == "teacher" for step, ref in seen.items() if step > 2), seen


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


def test_a_tie_keeps_the_EARLIER_of_two_equally_good_models(tiny_run, monkeypatch):
    """The save guard is strict and evaluated before `best` is updated, which is
    the reference's guard (train_mfa_v2.py uses `>`).

    `teacher.accuracy >= best` AFTER the update is not the always-true guard it
    looks like -- `best` is only raised when the accuracy improved, so the two
    agree on every case except a tie, where `>=` replaces the earlier of two
    equally good models with the later one. This test is the tie.
    """
    config_path, output_dir = tiny_run
    # three evaluations now (the warmup boundary, the cadence, the last step),
    # two calls each; every teacher score identical, so only the first can save
    scores = iter([70.0] * 6)
    monkeypatch.setattr(train_lora, "evaluate", lambda fn, loader, device:
                        EvalResult(next(scores), 0.0, 0, 1))

    best = train_lora.main(["--config", str(config_path)])

    assert best == pytest.approx(70.0)
    saved = torch.load(output_dir / "model-best.pt", weights_only=True)
    last = torch.load(output_dir / "model-last.pt", weights_only=True)
    assert any(not torch.equal(saved[k], last[k]) for k in saved), \
        "the tie rewrote model-best.pt with the later model"


def test_the_saved_checkpoint_is_the_teacher(tiny_run):
    """Evaluation scores the teacher and `best` follows it, so the checkpoint has
    to be the teacher's factors -- the reference saves lora1_t. Saving the
    student stores a model whose accuracy was never measured."""
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    model = train_lora.LAST_MODEL

    saved = torch.load(output_dir / "model-last.pt", weights_only=True)
    teacher = model.teacher_lora_state_dict()
    student = model.lora_state_dict()

    assert set(saved) == set(teacher)
    assert all(torch.equal(saved[k], teacher[k]) for k in saved)
    assert any(not torch.equal(student[k], teacher[k]) for k in saved), \
        "student and teacher are identical, so this test cannot tell them apart"


def test_the_warmup_boundary_is_always_evaluated(tiny_run):
    """The step where the pseudo-label reference switches from the frozen
    zero-shot model to the EMA teacher gets an evaluation of its own, off the
    cadence -- the reference does the same and tags it "end of s1 warmup"
    (train_mfa_v2.py:968). It is the number that says whether warmup was long
    enough. Here warmup ends at 2 and eval_freq is 3, so nothing else would
    evaluate there.
    """
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])

    recorded = rows(output_dir)
    steps = [r["step"] for r in recorded]
    assert 2 in steps, f"warmup boundary not evaluated; evaluated at {steps}"
    assert 2 % 3 != 0, "eval_freq would have covered it anyway; test proves nothing"

    boundary = next(r for r in recorded if r["step"] == 2)
    assert boundary["at_warmup_end"] is True
    assert all(r["at_warmup_end"] is False for r in recorded if r["step"] != 2)


def test_no_boundary_evaluation_when_there_is_no_warmup(tiny_run):
    config_path, output_dir = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    raw["branches"][0]["warmup_steps"] = 0
    config_path.write_text(yaml.safe_dump(raw))

    train_lora.main(["--config", str(config_path)])

    assert all(r["at_warmup_end"] is False for r in rows(output_dir))


def test_derived_evaluation_count_matches_what_happens(tiny_run):
    """`evaluations` is printed before the run as a sanity check, so it has to
    count the off-cadence boundary evaluation too."""
    config_path, output_dir = tiny_run
    train_lora.main(["--config", str(config_path)])
    declared = json.loads((output_dir / "run.json").read_text())["evaluations"]
    assert declared == len(rows(output_dir))


def test_const_momentum_applies_from_step_zero():
    """The reference applies its constant from step 0, so the teacher keeps the
    SVD initialization -- which reproduces zero-shot CLIP -- and decays away from
    it rather than discarding it. engine.ema.momentum_at returns 0.0 at step 0
    for branch 2's sake; this branch overrides that one point, only for "const",
    and leaves the shared function alone."""
    cfg = EmaConfig(momentum=0.99, schedule="const")
    assert [train_lora.ema_momentum(t, cfg) for t in range(3)] == [0.99, 0.99, 0.99]


def test_ramp_still_hard_copies_at_step_zero():
    """A ramp named in the config is a request for the hard copy its own formula
    gives, min(0 / 1, m) == 0. The override must not reach it."""
    cfg = EmaConfig(momentum=0.99, schedule="ramp")
    assert train_lora.ema_momentum(0, cfg) == 0.0
    assert train_lora.ema_momentum(1, cfg) == 0.5


def test_the_teacher_keeps_99_percent_of_its_initialization_after_one_step(tiny_run):
    """The point of the change, measured on the model rather than on the momentum:
    after one step at momentum 0.99 the teacher's LoRA factors must have moved
    exactly 1% of the way from their initial values to the student's stepped
    ones. Under a step-0 hard copy they move 100%."""
    config_path, _ = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    raw["cotrain"]["total_macro_steps"] = 1
    raw["branches"][0]["warmup_steps"] = 0
    config_path.write_text(yaml.safe_dump(raw))

    from cmct.branches.lora_model import LoraModel
    captured = {}
    original_init = LoraModel.__init__

    def spy(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["init"] = {k: v.clone()
                            for k, v in self.teacher_lora_state_dict().items()}

    LoraModel.__init__ = spy
    try:
        train_lora.main(["--config", str(config_path)])
    finally:
        LoraModel.__init__ = original_init

    model = train_lora.LAST_MODEL
    assert model.teacher_updates == 1
    init = captured["init"]
    teacher = model.teacher_lora_state_dict()
    student = model.lora_state_dict()

    span = {k: (student[k] - init[k]).norm().item() for k in init}
    active = [k for k in init if span[k] > 1e-8]
    assert active, "the student's factors did not move; this test cannot measure anything"
    for key in active:
        share = (teacher[key] - init[key]).norm().item() / span[key]
        assert share == pytest.approx(0.01, abs=2e-3), \
            f"{key}: teacher moved {share:.4f} of the way to the student, expected 0.01"
