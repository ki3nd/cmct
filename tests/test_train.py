"""Co-training: both branches, each teaching the other."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from cmct import train

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENT = CONFIGS / "experiment" / "cmct_officehome_a2c.yaml"


@pytest.fixture
def tiny_run(tmp_path, class_folder_root, monkeypatch):
    from conftest import build_tiny_clip

    raw = yaml.safe_load(EXPERIMENT.read_text())
    raw["data"].update(root=str(class_folder_root.parent), batch_size_test=4,
                       num_workers_test=0)
    for branch in raw["branches"]:
        branch["stream"].update(batch_size_x=4, batch_size_u=4, num_workers=0)
        branch["backbone"]["dtype"] = "fp32"      # tiny_clip is fp32
        # A random tiny CLIP never clears 0.85, so at the shipped threshold every
        # cross term would be exactly 0 and these tests would pass on a build
        # that never computed one.
        branch["pseudo_label"]["threshold"] = 0.0
    raw["branches"][0]["warmup_steps"] = 2        # branch 1, in macro-steps
    raw["branches"][1]["warmup_steps"] = 4        # branch 2, in micro-steps
    raw["branches"][1]["steps_per_macro"] = 2
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

    monkeypatch.setattr(train, "load_clip",
                        lambda checkpoint, dtype: build_tiny_clip())
    return config_path, Path(raw["run"]["output_dir"])


def rows(output_dir):
    return [json.loads(line)
            for line in (output_dir / "metrics.jsonl").read_text().splitlines()]


def edit(config_path, **cotrain):
    raw = yaml.safe_load(config_path.read_text())
    raw["cotrain"].update(cotrain)
    config_path.write_text(yaml.safe_dump(raw))
    return raw


# --- the nesting -------------------------------------------------------------

def test_branch_two_takes_steps_per_macro_steps_for_each_of_branch_ones(tiny_run):
    """The reference's --s2-per-s1. Branch 2 converges faster and is run denser;
    getting this wrong would silently train one branch ten times too little."""
    config_path, _ = tiny_run
    train.main(["--config", str(config_path)])
    state = train.LAST_STATE
    assert state["lora"]["model"].teacher_updates == 6
    assert state["vlp"]["model"].teacher_updates == 12


# --- cross-teaching ----------------------------------------------------------

def test_neither_branch_has_a_cross_term_during_its_own_warmup(tiny_run):
    """The two warmups end at different macro-steps -- branch 1 at 2, branch 2 at
    macro 2 as well but counted in its own 4 micro-steps -- so a single cadence
    cannot show both transitions. print_freq 1 records every macro-step."""
    config_path, output_dir = tiny_run
    train.main(["--config", str(config_path)])
    recorded = {r["macro"]: r for r in rows(output_dir)}

    assert recorded[1]["lora_cross"] == 0.0, "branch 1 taught during its warmup"
    assert recorded[2]["lora_cross"] == 0.0
    assert recorded[1]["vlp_cross"] == 0.0, "branch 2 taught during its warmup"
    assert recorded[2]["vlp_cross"] == 0.0


def test_both_branches_have_a_NON_ZERO_cross_term_after_warmup(tiny_run):
    """Not merely present -- non-zero. With the shipped 0.85 threshold on a random
    tiny CLIP nothing clears the mask and both terms are exactly 0.0, which a
    build that never computed a cross term would also produce."""
    config_path, output_dir = tiny_run
    train.main(["--config", str(config_path)])
    after = [r for r in rows(output_dir) if r["macro"] > 2]
    assert after
    assert all(r["lora_cross"] > 0.0 for r in after), [r["lora_cross"] for r in after]
    assert all(r["vlp_cross"] > 0.0 for r in after), [r["vlp_cross"] for r in after]
    assert all(r["lora_cross_mask"] == 1.0 for r in after)


def test_the_reference_switches_at_branch_ones_warmup_boundary(tiny_run):
    config_path, output_dir = tiny_run
    train.main(["--config", str(config_path)])
    seen = {r["macro"]: r["lora_reference"] for r in rows(output_dir)}
    assert seen[1] == "zero_shot" and seen[2] == "zero_shot"
    assert all(ref == "teacher" for macro, ref in seen.items() if macro > 2)


def test_gini_cross_mode_raises_rather_than_running_mask(tiny_run):
    config_path, _ = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    raw["branches"][1]["cross_mode"] = "gini"
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit) as caught:
        train.main(["--config", str(config_path)])
    assert "cross_mode" in str(caught.value) and "gini" in str(caught.value)


# --- staleness ---------------------------------------------------------------

@pytest.mark.parametrize("refresh,lag", [("macro", 1), ("micro", 0)])
def test_how_stale_branch_ones_cross_reference_is(tiny_run, monkeypatch, refresh, lag):
    """Branch 1 reads branch 2's teacher BEFORE branch 2 takes its micro-steps,
    so under "macro" its reference is a whole inner loop out of date. That falls
    out of the reference's nesting (train_mfa_v2.py:751-758) rather than from a
    decision, but it IS what the reference does.

    Measured, not read: branch 2's teacher is replaced by a stub whose output
    encodes how many EMA updates it has received, so the reference branch 1
    actually used names the version it came from.
    """
    from cmct.branches.vlp_model import VlpModel

    config_path, _ = tiny_run
    edit(config_path, cross_ref_refresh=refresh, total_macro_steps=5)
    steps_per_macro = yaml.safe_load(config_path.read_text())["branches"][1]["steps_per_macro"]

    real_ema = VlpModel.ema_update

    def counting_ema(self, momentum):
        real_ema(self, momentum)

    def versioned_logits(self, images):
        # class 0 gets a logit equal to the update count, so softmax's argmax and
        # magnitude both identify the version
        out = torch.zeros(images.shape[0], 3)
        out[:, 0] = float(self.teacher_updates)
        return out

    monkeypatch.setattr(VlpModel, "ema_update", counting_ema)
    monkeypatch.setattr(VlpModel, "teacher_logits", versioned_logits)

    seen = []
    real_cross = train.cross_loss

    def spy(**kwargs):
        if kwargs.get("branch") == "lora":
            probabilities = kwargs["reference_probabilities"]
            # invert softmax on the one non-zero logit to recover the version
            p0 = float(probabilities[0, 0])
            seen.append(round(torch.log(torch.tensor(p0 / (1 - p0) * 2)).item()))
        return real_cross(**kwargs)

    monkeypatch.setattr(train, "cross_loss", spy)
    train.main(["--config", str(config_path)])

    assert seen, "branch 1's cross term never ran"
    # branch 1's cross term starts at macro index 2 (warmup_steps = 2)
    for offset, version in enumerate(seen):
        macro = 2 + offset
        expected = (macro - lag + 1) * steps_per_macro if lag == 0 else macro * steps_per_macro
        assert version == expected, (
            f"macro {macro}: branch 1 saw teacher version {version}, expected "
            f"{expected} under cross_ref_refresh={refresh!r}"
        )


# --- reporting ---------------------------------------------------------------

def test_every_reported_model_gets_its_own_best(tiny_run, monkeypatch):
    """A single best following one chosen model would hide the case this run
    exists to detect: cross-teaching helping one branch and hurting the other.
    Branch 1 improves here while branch 2 degrades."""
    config_path, output_dir = tiny_run
    edit(config_path, ensemble="off", total_macro_steps=6)
    from cmct.engine.evaluator import EvalResult

    scores = iter([(10.0, 90.0), (50.0, 40.0)])

    def fake(logits_fns, loader, device, mode):
        a, b = next(scores)
        return {"lora": EvalResult(a, 0.0, 0, 1), "vlp": EvalResult(b, 0.0, 0, 1)}

    monkeypatch.setattr(train, "evaluate_ensemble", fake)
    best = train.main(["--config", str(config_path)])

    assert best == {"lora": 50.0, "vlp": 90.0}


def test_ensemble_is_off_by_default_and_absent_everywhere(tiny_run):
    config_path, output_dir = tiny_run
    best = train.main(["--config", str(config_path)])
    assert "ensemble" not in best
    assert all("ensemble_acc" not in r for r in rows(output_dir))
    assert "ensemble" not in json.loads((output_dir / "run.json").read_text())["reported"]


def test_turning_the_ensemble_on_reports_a_third_number(tiny_run):
    config_path, output_dir = tiny_run
    edit(config_path, ensemble="mean_prob")
    best = train.main(["--config", str(config_path)])
    assert set(best) == {"lora", "vlp", "ensemble"}
    assert any("ensemble_acc" in r for r in rows(output_dir))


# --- guards ------------------------------------------------------------------

def test_an_unknown_branch_type_raises(tiny_run):
    """Otherwise the run would train two branches and quietly ignore a third the
    config asked for."""
    config_path, _ = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    third = dict(raw["branches"][0])
    third["name"], third["type"] = "extra", "something_else"
    raw["branches"].append(third)
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit, match="something_else"):
        train.main(["--config", str(config_path)])


def test_a_missing_branch_type_raises(tiny_run):
    config_path, _ = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    raw["branches"] = [raw["branches"][0]]
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit, match="vlp_clip"):
        train.main(["--config", str(config_path)])


def test_the_shipped_config_loads_and_builds_both_branches(tiny_run):
    """The previous version of this file had never been loaded by any script and
    carried keys no model reads (`lora_encoder` for `lora_encoders`)."""
    config_path, _ = tiny_run
    train.main(["--config", str(config_path), "--max-macro-steps", "1"])
    assert train.LAST_STATE["lora"]["model"] is not None
    assert train.LAST_STATE["vlp"]["model"] is not None
