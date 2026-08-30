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

    def versioned_logits(self, images):
        # class 0 gets a logit equal to the update count, so softmax's argmax and
        # magnitude both identify the version
        out = torch.zeros(images.shape[0], 3)
        out[:, 0] = float(self.teacher_updates)
        return out

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

    scores = iter([(10.0, 90.0), (30.0, 70.0), (50.0, 40.0)])

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


# --- branch 2 must step exactly as its solo script does ----------------------

def test_branch_twos_head_sees_source_before_target(tiny_run, monkeypatch):
    """The head starts with a BatchNorm1d whose running stats are updated on
    every forward in train mode, so the ORDER of the two head calls decides which
    domain those buffers lean toward. They are EMA-copied into teacher_head,
    which evaluation scores AND which supplies branch 1's cross pseudo-labels.

    Reusing the target logits for the cross term made it tempting to compute them
    first; that reversed the order and moved running_mean about 5% toward the
    target domain over a run. Nothing else in the suite would have noticed -- no
    loss, gradient or accuracy assertion changes.
    """
    from cmct.backbones.heads import ClassifierHead

    config_path, _ = tiny_run
    # The source and target batches have different sizes, so which one reached
    # the head is readable from the tensor itself -- no bookkeeping to get wrong.
    raw = yaml.safe_load(config_path.read_text())
    # Sizes chosen to be unique: batch_size_test is 4, so reusing 4 here would
    # tag the teacher head's eval-time calls "source" as well.
    raw["branches"][1]["stream"]["batch_size_x"] = 5
    raw["branches"][1]["stream"]["batch_size_u"] = 6
    raw["branches"][1]["steps_per_macro"] = 2
    config_path.write_text(yaml.safe_dump(raw))

    seen: list[str] = []
    real_head = ClassifierHead.forward

    def spy_head(self, features):
        seen.append({5: "source", 6: "target"}.get(features.shape[0], "other"))
        return real_head(self, features)

    monkeypatch.setattr(ClassifierHead, "forward", spy_head)
    train.main(["--config", str(config_path), "--max-macro-steps", "1"])

    training_calls = [tag for tag in seen if tag in ("source", "target")]
    # The WHOLE prefix, both micro-steps -- checking only the first two calls
    # would miss an order that is right once and wrong afterwards.
    assert training_calls == ["source", "target"] * 2, training_calls


# --- the shipped config, not the fixture's overrides -------------------------

def test_the_shipped_config_pins_the_references_values():
    """The tiny_run fixture overwrites thresholds, warmups and steps_per_macro
    before the script ever reads them, so every test above runs on values that
    are not the shipped ones. This reads the file itself."""
    from cmct.config import load_experiment

    cfg = load_experiment(EXPERIMENT)
    lora = next(b for b in cfg.branches if b.type == "lora_clip")
    vlp = next(b for b in cfg.branches if b.type == "vlp_clip")

    assert cfg.cotrain.total_macro_steps == 1000
    assert cfg.cotrain.cross_ref_refresh == "macro"
    assert cfg.cotrain.ensemble == "off"

    assert (lora.steps_per_macro, lora.warmup_steps) == (1, 50)
    assert (vlp.steps_per_macro, vlp.warmup_steps) == (10, 500)
    for branch in (lora, vlp):
        assert branch.cross_weight == 0.5
        assert branch.cross_mode == "mask"
        assert branch.pseudo_label.threshold == 0.85
    assert lora.ema.momentum == 0.99 and lora.ema.schedule == "const"
    assert vlp.ema.schedule == "ramp"
    assert lora.optim.warmup_lr == 0.001 and lora.optim.grad_clip == 20.0

    # Not reachable from the branch-block equality test -- these live outside a
    # branch, and the solo configs deliberately differ (50/10 and 500/100), so
    # that test would not catch a regression here. eval_freq decides where `best`
    # is sampled.
    assert cfg.run.eval_freq == 200 and cfg.run.print_freq == 50
    assert cfg.run.seed == 42


def test_the_branch_blocks_match_the_single_branch_configs():
    """A co-training run and a solo run must differ only by the cross term, so
    every value outside the co-training knobs has to be the same file's."""
    import yaml

    cotrain = yaml.safe_load(EXPERIMENT.read_text())["branches"]
    solo = {}
    for name in ("lora_officehome_a2c", "vlp_officehome_a2c"):
        path = CONFIGS / "experiment" / f"{name}.yaml"
        for branch in yaml.safe_load(path.read_text())["branches"]:
            solo[branch["type"]] = branch

    cotrain_only = {"steps_per_macro", "warmup_steps", "cross_weight", "cross_mode"}
    for branch in cotrain:
        reference = solo[branch["type"]]
        for key, value in branch.items():
            if key in cotrain_only:
                continue
            assert value == reference[key], f"{branch['name']}.{key}"


# --- boundary evaluation and best checkpoints --------------------------------

def test_each_branchs_warmup_boundary_is_evaluated(tiny_run):
    """eval_freq is 3, so neither boundary sits on the cadence.

    Branch 2's warmup is set to 3 micro-steps against steps_per_macro 2, on
    purpose: its warmup then ends PART WAY through macro 2, which is the only
    case that distinguishes the reference's `>=` plus look-back from a naive
    `step2 == warmup_steps`. At the fixture's original 4/2 the boundary landed
    exactly on a macro edge and both forms fired at the same macro-step, so the
    test passed on either. The shipped 500/10 is likewise a clean multiple --
    this is about the formula surviving a warmup that is not.
    """
    config_path, output_dir = tiny_run
    raw = yaml.safe_load(config_path.read_text())
    raw["branches"][1]["warmup_steps"] = 3
    config_path.write_text(yaml.safe_dump(raw))

    train.main(["--config", str(config_path)])
    tagged = {r["macro"]: r["at_warmup_end"] for r in rows(output_dir)
              if r.get("at_warmup_end")}

    assert 2 in tagged, tagged
    assert any("lora" in tag for tag in tagged[2])
    assert any("vlp" in tag for tag in tagged[2]), (
        "branch 2's boundary was missed: step2 goes 2, 4 and never equals 3"
    )
    assert 2 % 3 != 0, "eval_freq would have covered it anyway"


def test_a_best_checkpoint_is_written_per_model(tiny_run, monkeypatch):
    config_path, output_dir = tiny_run
    from cmct.engine.evaluator import EvalResult

    scores = iter([(10.0, 90.0), (30.0, 70.0), (50.0, 40.0)])

    def fake(logits_fns, loader, device, mode):
        a, b = next(scores)
        return {"lora": EvalResult(a, 0.0, 0, 1), "vlp": EvalResult(b, 0.0, 0, 1)}

    written: list[tuple[str, int]] = []
    real_save = torch.save

    def counting_save(payload, path, *args, **kwargs):
        written.append((Path(path).name, len(payload)))
        return real_save(payload, path, *args, **kwargs)

    monkeypatch.setattr(train, "evaluate_ensemble", fake)
    monkeypatch.setattr(torch, "save", counting_save)
    train.main(["--config", str(config_path)])

    names = [name for name, _ in written]
    # lora improves at every evaluation (10 -> 30 -> 50), vlp only at the first
    # (90 -> 70 -> 40). Asserting the files merely EXIST passes on a build that
    # rewrites both at every evaluation, since both are written the first time.
    assert names.count("model-best-lora.pt") == 3
    assert names.count("model-best-vlp.pt") == 1
    assert names.count("model-last.pt") == 3

    # Each file holds only its own model: a combined snapshot would carry the
    # other branch's weights as of this branch's peak.
    assert dict(written)["model-best-lora.pt"] == 1
    assert dict(written)["model-best-vlp.pt"] == 1
    assert dict(written)["model-last.pt"] == 2
    assert set(torch.load(output_dir / "model-best-lora.pt",
                          weights_only=True)) == {"lora"}
    assert set(torch.load(output_dir / "model-best-vlp.pt",
                          weights_only=True)) == {"vlp"}


def test_branch_twos_cross_reference_comes_from_branch_ones_TEACHER(
        tiny_run, monkeypatch):
    """Swapping lora["teacher_logits"] for lora["model"].logits at the cross site
    keeps every logged number plausible and every other test green, while
    silently removing the EMA from one of the two teaching directions. The other
    direction is traced this way by the staleness test; this is its mirror.

    Only the TEACHER is stubbed, and with a sharp non-uniform distribution. The
    student is left alone -- patching it too would break branch 1's own backward,
    and a constant would be indistinguishable after softmax anyway.
    """
    from cmct.branches.lora_model import LoraModel

    config_path, _ = tiny_run
    marker = torch.tensor([12.0, 0.0, 0.0])
    monkeypatch.setattr(LoraModel, "teacher_logits",
                        lambda self, images: marker.repeat(images.shape[0], 1))

    seen = []
    real_cross = train.cross_loss

    def spy(**kwargs):
        if kwargs.get("branch") == "vlp":
            seen.append(kwargs["reference_probabilities"][0].clone())
        return real_cross(**kwargs)

    monkeypatch.setattr(train, "cross_loss", spy)
    train.main(["--config", str(config_path)])

    assert seen, "branch 2's cross term never ran"
    expected = torch.softmax(marker, dim=-1)
    for probabilities in seen:
        assert torch.allclose(probabilities, expected, atol=1e-5), (
            f"branch 2's cross reference was {probabilities.tolist()}, not the "
            f"teacher's {expected.tolist()} -- it came from another model"
        )


def test_branch_ones_scheduler_is_wired_with_its_warmup(tiny_run):
    """build_lr_scheduler takes `warmup` as an optional argument -- build_vlp
    deliberately omits it -- so dropping it in build_lora is a one-line slip no
    other test notices. It would run branch 1's cosine at amplitude `lr` from
    macro 0 instead of holding `warmup_lr` flat: a 3.5x learning-rate error
    through every warmup step, with the suite still green.

    The assertion has to compare against BOTH wirings, or it passes on either.
    """
    from cmct.engine import lr_at

    config_path, _ = tiny_run
    # Stop INSIDE the warmup. At the end of the run the cosine has decayed to
    # ~0 under either wiring, so the two become indistinguishable there.
    steps = 1
    train.main(["--config", str(config_path), "--max-macro-steps", str(steps)])
    state = train.LAST_STATE
    optim_cfg = state["lora"]["config"].optim
    warmup = state["lora"]["config"].warmup_steps
    total = state["lora"]["total_steps"]

    assert warmup > steps and optim_cfg.warmup_lr != optim_cfg.lr, \
        "this config cannot distinguish the two wirings"

    with_warmup = lr_at(steps, optim_cfg, total, warmup)
    without_warmup = lr_at(steps, optim_cfg, total, 0)
    assert with_warmup != pytest.approx(without_warmup), \
        "the two wirings agree here, so this test proves nothing"

    actual = state["lora"]["optimizer"].param_groups[0]["lr"]
    assert actual == pytest.approx(with_warmup, rel=1e-9), (
        f"lr after the run is {actual}; warmup-aware wiring gives {with_warmup}, "
        f"the slip gives {without_warmup}"
    )


def test_the_two_branches_treat_their_warmups_differently_for_the_LR():
    """Branch 1 holds a SEPARATE flat learning rate through its warmup; branch 2
    has no warmup learning rate at all and runs one continuous rule across its
    boundary. The reference gates only branch 1's scheduler
    (train_mfa_v2.py:940 `if not in_warmup1: sched1.step()`, against an ungated
    :867 for branch 2), and this is the consequence.

    Worth pinning because the two look like the same knob: both branches have
    `warmup_steps`, but it means "a different LR" on one and "when cross-teaching
    starts" on both.
    """
    from cmct.config import load_experiment
    from cmct.engine import lr_at

    cfg = load_experiment(EXPERIMENT)
    lora = next(b for b in cfg.branches if b.type == "lora_clip")
    vlp = next(b for b in cfg.branches if b.type == "vlp_clip")
    lora_total = cfg.cotrain.total_macro_steps * lora.steps_per_macro
    vlp_total = cfg.cotrain.total_macro_steps * vlp.steps_per_macro

    # Branch 1: flat at warmup_lr through the warmup AND its boundary step, then
    # decaying. `lr` never appears.
    flat = [lr_at(s, lora.optim, lora_total, lora.warmup_steps)
            for s in (0, lora.warmup_steps // 2, lora.warmup_steps)]
    assert flat == [pytest.approx(lora.optim.warmup_lr)] * 3, flat
    assert lora.optim.warmup_lr != lora.optim.lr
    after = lr_at(lora.warmup_steps + 1, lora.optim, lora_total, lora.warmup_steps)
    assert after < lora.optim.warmup_lr

    # Branch 2: no warmup LR, and no discontinuity at the boundary. The step
    # either side of it differs by the same amount as any other neighbouring
    # pair, so the boundary is invisible to the schedule.
    assert vlp.optim.warmup_lr is None
    around = [lr_at(s, vlp.optim, vlp_total)
              for s in (vlp.warmup_steps - 1, vlp.warmup_steps, vlp.warmup_steps + 1)]
    at_boundary = around[0] - around[1]
    after_boundary = around[1] - around[2]
    assert at_boundary == pytest.approx(after_boundary, rel=1e-2), around
    assert around[0] > around[1] > around[2], "branch 2's LR should decay throughout"
