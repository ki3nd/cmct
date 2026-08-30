"""Cross-model co-training: both branches, each teaching the other.

The loop is `old-cmct/train_mfa_v2.py`'s, and its shape is asymmetric on
purpose:

    for macro in range(total_macro_steps):
        fetch branch 1's batch
        cross reference for branch 1  <- branch 2's teacher, ONCE per macro
        for _ in range(branch2.steps_per_macro):
            cross reference for branch 2  <- branch 1's teacher, EVERY micro
            branch 2 steps
        branch 1 steps

Branch 2 takes `steps_per_macro` optimizer steps for every one of branch 1's.
A consequence, not a decision: branch 1 reads a teacher 2 that is stale by that
many micro-steps, while branch 2's reference is always current.
`cotrain.cross_ref_refresh` names it -- "macro" reproduces the reference,
"micro" recomputes branch 1's reference after the inner loop and is a deviation.

Each branch has its OWN warmup, counted in its own steps, and has no cross term
until its own warmup ends. The two warmups are separate knobs and are not
derived from one another.

Usage:
    python -m cmct.train --config configs/experiment/cmct_officehome_a2c.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cmct.backbones.clip import load_clip
from cmct.branches.lora_model import LoraModel
from cmct.branches.vlp_model import VlpModel
from cmct.config import BranchConfig, Config, dump, format_config, load_experiment
from cmct.data import BatchSource, build_split, build_test_loader
from cmct.engine import (
    build_lr_scheduler,
    build_optimizer,
    evaluate_ensemble,
    lr_at,
    momentum_at,
)
from cmct.losses import CmkdLoss, LoraBranchLoss, cross_loss
from cmct.train_lora import ema_momentum, read_extra, set_seed

LORA = "lora_clip"
VLP = "vlp_clip"

LAST_STATE: dict | None = None
"""Everything the most recent run built, so a test can inspect it afterwards.
Not part of what main() returns."""


def pick(cfg: Config, branch_type: str) -> BranchConfig:
    matching = [b for b in cfg.branches if b.type == branch_type]
    if not matching:
        raise SystemExit(
            f"co-training needs one branch of type {branch_type!r}; this config has "
            f"{[(b.name, b.type) for b in cfg.branches]}"
        )
    if len(matching) > 1:
        raise SystemExit(
            f"several branches of type {branch_type!r} ({[b.name for b in matching]}); "
            f"co-training pairs exactly one of each"
        )
    return matching[0]


def check_types(cfg: Config) -> None:
    """An unrecognized branch type would otherwise be silently skipped -- the run
    would train two branches and quietly ignore a third the config asked for."""
    unknown = [(b.name, b.type) for b in cfg.branches if b.type not in (LORA, VLP)]
    if unknown:
        raise SystemExit(
            f"unknown branch type(s) {unknown}; co-training knows {[LORA, VLP]}"
        )


def check_cross_mode(branch: BranchConfig) -> None:
    if branch.cross_mode != "mask":
        raise SystemExit(
            f"branches[{branch.name}].cross_mode: {branch.cross_mode!r} is not "
            f"implemented; only 'mask' is"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-macro-steps", type=int, default=None,
                        help="stop after this many MACRO steps. A script-level knob "
                             "for short runs; it shortens neither warmup nor any "
                             "schedule, so the first N steps of a short run match "
                             "the first N of the real one")
    parser.add_argument("--device", default=None, help="overrides run.device")
    return parser.parse_args(argv)


def build_lora(cfg: Config, branch: BranchConfig, split, device: str) -> dict:
    """Branch 1: LoRA on both towers, cosine classification, EMA teacher, MK-MMD."""
    extra = read_extra(branch)
    param_dtype = torch.float16 if branch.backbone.dtype == "fp16" else torch.float32
    model = LoraModel(
        load_clip(branch.backbone.checkpoint, branch.backbone.dtype),
        split.classnames,
        rank=extra["lora_rank"], alpha=extra["lora_alpha"],
        dropout=extra["lora_dropout"], params=tuple(extra["lora_params"]),
        rank_ramp=extra["lora_rank_ramp"], positions=extra["lora_positions"],
        encoders=tuple(extra["lora_encoders"]), param_dtype=param_dtype,
    ).to(device)

    warmup = branch.warmup_steps
    uses_zero_shot = warmup > 0 and extra["warmup_reference"] == "zero_shot"
    if uses_zero_shot:
        # A second read of the checkpoint, not a copy of the student: apply_lora
        # mutates in place and LoraModel rejects a model that already carries LoRA.
        model.attach_zero_shot(
            load_clip(branch.backbone.checkpoint, branch.backbone.dtype).to(device)
        )

    total = cfg.cotrain.total_macro_steps * branch.steps_per_macro
    optimizer = build_optimizer(model.param_groups(lr=1.0), branch.optim)
    return {
        "name": branch.name, "config": branch, "model": model, "extra": extra,
        "loss": LoraBranchLoss(
            threshold=branch.pseudo_label.threshold,
            reduce=branch.pseudo_label.self_reduce,
            mmd_weight=extra["mmd_weight"],
        ),
        "optimizer": optimizer,
        "scheduler": build_lr_scheduler(optimizer, branch.optim, total, warmup),
        "total_steps": total, "uses_zero_shot": uses_zero_shot,
        "teacher_logits": model.teacher_logits,
    }


def build_vlp(cfg: Config, branch: BranchConfig, split, device: str) -> dict:
    """Branch 2: full fine-tune with a learned head, CMKD self-training."""
    model = VlpModel(
        load_clip(branch.backbone.checkpoint, branch.backbone.dtype),
        split.classnames, split.num_classes,
    ).to(device)
    total = cfg.cotrain.total_macro_steps * branch.steps_per_macro
    multiplier = branch.optim.param_group_multipliers.get("classifier", 1.0)
    optimizer = build_optimizer(
        model.param_groups(lr=1.0, head_multiplier=multiplier), branch.optim)
    return {
        "name": branch.name, "config": branch, "model": model, "extra": {},
        "loss": CmkdLoss.from_branch_config(branch, max_iter=total),
        "optimizer": optimizer,
        "scheduler": build_lr_scheduler(optimizer, branch.optim, total),
        "total_steps": total, "head_multiplier": multiplier,
        "teacher_logits": model.teacher_logits,
    }


@torch.no_grad()
def lora_reference(lora: dict, images, in_warmup: bool):
    """Branch 1's SELF reference: the frozen zero-shot CLIP while in warmup under
    warmup_reference "zero_shot", the branch's own EMA teacher otherwise."""
    model = lora["model"]
    if in_warmup and lora["uses_zero_shot"]:
        return F.softmax(model.zero_shot_logits(images), dim=-1), "zero_shot"
    return F.softmax(model.teacher_logits(images), dim=-1), "teacher"


def main(argv: list[str] | None = None) -> dict[str, float]:
    """Returns the best accuracy of every model this run reports."""
    global LAST_STATE
    args = parse_args(argv)
    cfg = load_experiment(args.config)
    check_types(cfg)
    lora_cfg, vlp_cfg = pick(cfg, LORA), pick(cfg, VLP)
    for branch in (lora_cfg, vlp_cfg):
        check_cross_mode(branch)

    device = args.device or cfg.run.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"{device} is not available, falling back to cpu")
        device = "cpu"

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    set_seed(cfg.run.seed)

    macro_steps = cfg.cotrain.total_macro_steps
    run_macro = (macro_steps if args.max_macro_steps is None
                 else min(args.max_macro_steps, macro_steps))

    print("=" * 78)
    print(f"resolved config ({args.config})")
    print("=" * 78)
    print(format_config(cfg), end="")

    split = build_split(cfg.dataset, cfg.data)
    lora = build_lora(cfg, lora_cfg, split, device)
    vlp = build_vlp(cfg, vlp_cfg, split, device)
    streams = {b["name"]: BatchSource(split, cfg.dataset, b["config"], cfg.run.seed)
               for b in (lora, vlp)}
    test_loader = build_test_loader(split, cfg.dataset, cfg.data)

    ensemble_mode = cfg.cotrain.ensemble
    refresh = cfg.cotrain.cross_ref_refresh
    reported = [lora["name"], vlp["name"]]
    if ensemble_mode != "off":
        reported.append("ensemble")

    derived = {
        "device": device,
        "branches": f"{lora['name']} ({LORA}) + {vlp['name']} ({VLP})",
        "macro_steps": f"{run_macro} of {macro_steps}",
        "steps": f"{lora['name']} {run_macro * lora_cfg.steps_per_macro}"
                 f" / {vlp['name']} {run_macro * vlp_cfg.steps_per_macro}",
        "warmups": f"{lora['name']} {lora_cfg.warmup_steps} steps"
                   f" / {vlp['name']} {vlp_cfg.warmup_steps} steps",
        "cross_weights": f"{lora['name']} {lora_cfg.cross_weight}"
                         f" / {vlp['name']} {vlp_cfg.cross_weight}",
        "thresholds": f"{lora['name']} {lora_cfg.pseudo_label.threshold}"
                      f" / {vlp['name']} {vlp_cfg.pseudo_label.threshold}",
        "cross_ref_refresh": refresh,
        "ensemble": ensemble_mode,
        "reported": reported,
        "images": f"{len(split.train_x)} source / {len(split.train_u)} target "
                  f"/ {len(split.test)} test",
        "classes": split.num_classes,
        "eval_every": cfg.run.eval_freq,
        "print_every": cfg.run.print_freq,
        "output_dir": str(output_dir),
    }
    print("-" * 78)
    print("derived")
    print("-" * 78)
    width = max(len(k) for k in derived)
    for key, value in derived.items():
        print(f"  {key:<{width}}  {value}")
    print("-" * 78)
    (output_dir / "run.json").write_text(json.dumps(derived, indent=2) + "\n")

    # Every reported model gets its OWN best. A single best following one chosen
    # model would hide the case this run exists to detect: cross-teaching helping
    # one branch and hurting the other.
    best = dict.fromkeys(reported, 0.0)
    step2 = 0
    started = time.time()

    for macro in range(run_macro):
        in_warmup1 = macro < lora_cfg.warmup_steps
        lora["model"].train()
        batch1 = streams[lora["name"]].next().to(device)

        # Branch 1's cross reference, from branch 2's teacher. Computed HERE,
        # before branch 2 moves, so it is stale by steps_per_macro micro-steps --
        # which is what the reference does (train_mfa_v2.py:751-758).
        cross_for_1 = None
        if not in_warmup1 and refresh == "macro":
            with torch.no_grad():
                cross_for_1 = F.softmax(vlp["teacher_logits"](batch1.img_u), dim=-1)

        for _ in range(vlp_cfg.steps_per_macro):
            in_warmup2 = step2 < vlp_cfg.warmup_steps
            vlp["model"].train()
            batch2 = streams[vlp["name"]].next().to(device)

            vlp["optimizer"].zero_grad()
            source_features = vlp["model"].features(batch2.img_x)
            target_features = vlp["model"].features(batch2.img_u)
            target_logits2 = vlp["model"].head(target_features)
            out2 = vlp["loss"](
                source_logits=vlp["model"].head(source_features),
                source_label=batch2.label_x,
                source_cosine_logits=vlp["model"].encoder.cosine_logits(source_features),
                target_logits=target_logits2,
                target_cosine_logits=vlp["model"].encoder.cosine_logits(target_features),
                step=step2,
            )
            loss2 = out2.total
            cross2_value, cross2_mask = 0.0, 0.0
            if not in_warmup2:
                # Branch 2's cross reference, from branch 1's teacher, recomputed
                # every micro-step. Reuses target_logits2 rather than a second
                # forward: the reference calls that out as a real OOM contributor
                # and notes it also mutated the head's BatchNorm running stats an
                # extra time regardless of cross_weight (train_mfa_v2.py:824-833).
                with torch.no_grad():
                    cross_for_2 = F.softmax(
                        lora["teacher_logits"](batch2.img_u), dim=-1)
                cross2 = cross_loss(
                    target_logits=target_logits2,
                    reference_probabilities=cross_for_2,
                    threshold=vlp_cfg.pseudo_label.threshold,
                    mode=vlp_cfg.cross_mode, branch=vlp["name"],
                )
                loss2 = loss2 + vlp_cfg.cross_weight * cross2.value
                cross2_value = float(cross2.value.detach())
                cross2_mask = cross2.mask_ratio

            loss2.backward()
            if vlp_cfg.optim.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in vlp["optimizer"].param_groups
                     for p in g["params"] if p.requires_grad],
                    max_norm=vlp_cfg.optim.grad_clip,
                )
            vlp["optimizer"].step()
            vlp["model"].ema_update(momentum_at(step2, vlp_cfg.ema))
            vlp["scheduler"].step()
            step2 += 1

        # "micro": recompute after branch 2 has moved, so branch 1 reads a
        # current teacher. A deviation -- the reference cannot do this, its
        # reference is fixed before the inner loop.
        if not in_warmup1 and refresh == "micro":
            with torch.no_grad():
                cross_for_1 = F.softmax(vlp["teacher_logits"](batch1.img_u), dim=-1)

        reference, reference_name = lora_reference(lora, batch1.img_u, in_warmup1)
        lora["optimizer"].zero_grad()
        source_logits, source_features1 = lora["model"](batch1.img_x)
        target_features1 = (lora["model"].features(batch1.img_u)
                            if lora["extra"]["mmd_weight"] > 0 else None)
        target_logits1 = lora["model"].logits(batch1.student_img_u)
        out1 = lora["loss"](
            source_logits=source_logits, source_label=batch1.label_x,
            target_logits=target_logits1, reference_probabilities=reference,
            source_features=source_features1, target_features=target_features1,
            reference_name=reference_name,
        )
        loss1 = out1.total
        cross1_value, cross1_mask = 0.0, 0.0
        if cross_for_1 is not None:
            cross1 = cross_loss(
                target_logits=target_logits1, reference_probabilities=cross_for_1,
                threshold=lora_cfg.pseudo_label.threshold,
                mode=lora_cfg.cross_mode, branch=lora["name"],
            )
            loss1 = loss1 + lora_cfg.cross_weight * cross1.value
            cross1_value = float(cross1.value.detach())
            cross1_mask = cross1.mask_ratio

        loss1.backward()
        if lora_cfg.optim.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(lora["model"].trainable_parameters()),
                max_norm=lora_cfg.optim.grad_clip)
        lora["optimizer"].step()
        lora["scheduler"].step()
        lora["model"].ema_update(ema_momentum(macro, lora_cfg.ema))
        if lora["uses_zero_shot"] and in_warmup1 and macro + 1 == lora_cfg.warmup_steps:
            lora["model"].release_zero_shot()

        row = {
            "macro": macro + 1,
            f"{lora['name']}_loss": float(loss1.detach()),
            f"{lora['name']}_self": float(out1.pseudo_label.detach()),
            f"{lora['name']}_cross": cross1_value,
            f"{lora['name']}_cross_mask": cross1_mask,
            f"{lora['name']}_reference": reference_name,
            f"{vlp['name']}_loss": float(loss2.detach()),
            f"{vlp['name']}_cross": cross2_value,
            f"{vlp['name']}_cross_mask": cross2_mask,
        }
        if (macro + 1) % cfg.run.print_freq == 0:
            name1, name2 = lora["name"], vlp["name"]
            total1, self1 = row[f"{name1}_loss"], row[f"{name1}_self"]
            total2 = row[f"{name2}_loss"]
            lr1 = lr_at(macro, lora_cfg.optim, lora["total_steps"],
                        lora_cfg.warmup_steps)
            print(f"macro {macro + 1}/{run_macro}  "
                  f"{name1} {total1:.4f} (self {self1:.4f} "
                  f"cross {cross1_value:.4f} mask {cross1_mask:.2f} "
                  f"ref {reference_name})  "
                  f"{name2} {total2:.4f} "
                  f"(cross {cross2_value:.4f} mask {cross2_mask:.2f})  "
                  f"lr {lr1:.3e}")

        if (macro + 1) % cfg.run.eval_freq == 0 or (macro + 1) == run_macro:
            for entry in (lora, vlp):
                entry["model"].eval()
            scores = evaluate_ensemble(
                {lora["name"]: lora["teacher_logits"],
                 vlp["name"]: vlp["teacher_logits"]},
                test_loader, device, ensemble_mode,
            )
            for key, result in scores.items():
                best[key] = max(best[key], result.accuracy)
                row[f"{key}_acc"] = result.accuracy
                row[f"{key}_loss_eval"] = result.loss
                row[f"best_{key}"] = best[key]
            print("[eval] macro " + str(macro + 1) + "  " + "  ".join(
                f"{key} {scores[key].accuracy:.2f}% (best {best[key]:.2f}%)"
                for key in reported))
            torch.save({"lora": lora["model"].teacher_lora_state_dict(),
                        "vlp": vlp["model"].state_dict()},
                       output_dir / "model-last.pt")

        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    print(f"done in {time.time() - started:.1f}s. " + "  ".join(
        f"best {key} {best[key]:.2f}%" for key in reported))
    LAST_STATE = {"lora": lora, "vlp": vlp, "best": best, "reported": reported}
    return best


if __name__ == "__main__":
    main()
