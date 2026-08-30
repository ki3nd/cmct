"""Train branch 2 on its own.

No cross-teaching, no second branch: the point is to measure this branch in
isolation and compare it against the numbers of the codebase it comes from.

The loop shape is that codebase's, and the parts that matter are:

  - iteration-based, not epoch-based. Both training streams are infinite, so
    "epoch" is only an evaluation cadence.
  - total steps = total_macro_steps * steps_per_macro, and the CMKD ramp's
    max_iter is the SAME number, so the ramp spans exactly the run.
  - order within a step: zero_grad, forward, backward, optimizer.step(),
    EMA update, scheduler.step(). The EMA sits after the optimizer step, which
    is what makes its first update copy trained weights rather than the
    initialization.
  - evaluation scores the LIVE student head. The teacher is scored too and
    reported alongside, on the same deterministic loader, but `best` follows the
    student, because that is the number to compare.

Usage:
    python -m cmct.train_vlp --config configs/experiment/cmct_officehome_a2c.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cmct.backbones.clip import load_clip
from cmct.backbones.clip.download import resolve_checkpoint
from cmct.branches import VlpModel
from cmct.branches.vlp_model import TEMPLATE
from cmct.config import BranchConfig, Config, dump, format_config, load_experiment
from cmct.data import BatchSource, build_split, build_test_loader
from cmct.engine import (
    build_lr_scheduler,
    build_optimizer,
    evaluate,
    lr_at,
    momentum_at,
)
from cmct.losses import CmkdLoss
from cmct.losses.schedules import sigmoid_ramp

BRANCH_TYPE = "vlp_clip"


def set_seed(seed: int) -> None:
    """The four generators the original seeds, plus the two cuDNN flags it also
    sets (main.py:88-95). Without those a rerun varies even at a fixed seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_branch(cfg: Config, name: str | None) -> BranchConfig:
    if name is not None:
        matching = [b for b in cfg.branches if b.name == name]
        if not matching:
            raise SystemExit(
                f"no branch named {name!r}; have {[b.name for b in cfg.branches]}"
            )
        branch = matching[0]
        if branch.type != BRANCH_TYPE:
            raise SystemExit(f"branch {name!r} has type {branch.type!r}, not {BRANCH_TYPE!r}")
        return branch

    candidates = [b for b in cfg.branches if b.type == BRANCH_TYPE]
    if not candidates:
        raise SystemExit(f"no branch of type {BRANCH_TYPE!r} in this config")
    if len(candidates) > 1:
        raise SystemExit(
            f"several branches of type {BRANCH_TYPE!r} ({[b.name for b in candidates]}); "
            f"pass --branch"
        )
    return candidates[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--branch", default=None,
                        help=f"branch name; defaults to the only {BRANCH_TYPE} branch")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many steps. A script-level knob for short "
                             "runs; it does NOT shorten the CMKD ramp, so the first N "
                             "steps of a short run match the first N of the real one")
    parser.add_argument("--device", default=None, help="overrides run.device")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> float:
    """Returns the best student accuracy."""
    args = parse_args(argv)
    cfg = load_experiment(args.config)
    branch = pick_branch(cfg, args.branch)

    device = args.device or cfg.run.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"{device} is not available, falling back to cpu", flush=True)
        device = "cpu"

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg, output_dir)
    metrics_path = output_dir / "metrics.jsonl"

    set_seed(cfg.run.seed)

    total_steps = cfg.cotrain.total_macro_steps * branch.steps_per_macro
    run_steps = total_steps if args.max_steps is None else min(args.max_steps, total_steps)
    eval_every = cfg.run.eval_freq * branch.steps_per_macro
    print_every = cfg.run.print_freq * branch.steps_per_macro

    # Every setting that governs the run, verbatim -- the same text that goes to
    # config.yaml, so the log and the file cannot disagree.
    print("=" * 78, flush=True)
    print(f"resolved config ({args.config})", flush=True)
    print("=" * 78, flush=True)
    print(format_config(cfg), end="", flush=True)

    # the final step always evaluates, so it counts even when it is not on the cadence
    n_evals = len({*range(eval_every, run_steps + 1, eval_every), run_steps})
    multiplier = branch.optim.param_group_multipliers.get("classifier", 1.0)
    last = max(run_steps - 1, 0)
    checkpoint = resolve_checkpoint(branch.backbone.checkpoint) \
        if Path(branch.backbone.checkpoint).is_file() else branch.backbone.checkpoint

    split = build_split(cfg.dataset, cfg.data)

    # Values that govern the run but appear nowhere in the config, because they
    # are derived from it. These are the ones worth checking before a long run.
    derived = {
        "branch": f"{branch.name} ({branch.type})",
        "device": device,
        "checkpoint": str(checkpoint),
        "total_steps": total_steps,
        "run_steps": run_steps,
        "ramp_max_iter": total_steps,
        "eval_every": eval_every,
        "evaluations": n_evals,
        "print_every": print_every,
        "images": f"{len(split.train_x)} source / {len(split.train_u)} target "
                  f"/ {len(split.test)} test",
        "classes": split.num_classes,
        "prompt_template": TEMPLATE,
        "prompt_examples": [TEMPLATE.format(c) for c in split.classnames[:3]],
        "lr_encoder_first_last": [lr_at(0, branch.optim, total_steps),
                                  lr_at(last, branch.optim, total_steps)],
        "lr_head_first_last": [multiplier * lr_at(0, branch.optim, total_steps),
                               multiplier * lr_at(last, branch.optim, total_steps)],
        "head_lr_multiplier": multiplier,
        "ema_momentum_at_0_1_last": [momentum_at(0, branch.ema),
                                     momentum_at(1, branch.ema),
                                     momentum_at(last, branch.ema)],
        "cmkd_ramp_at_0_mid_last": [sigmoid_ramp(0, total_steps),
                                    sigmoid_ramp(total_steps // 2, total_steps),
                                    sigmoid_ramp(total_steps, total_steps)],
        "output_dir": str(output_dir),
    }
    print("-" * 78, flush=True)
    print("derived", flush=True)
    print("-" * 78, flush=True)
    width = max(len(k) for k in derived)
    for key, value in derived.items():
        print(f"  {key:<{width}}  {value}", flush=True)
    print("-" * 78, flush=True)
    (output_dir / "run.json").write_text(json.dumps(derived, indent=2) + "\n")

    stream = BatchSource(split, cfg.dataset, branch, cfg.run.seed)
    test_loader = build_test_loader(split, cfg.dataset, cfg.data)

    model = VlpModel(
        load_clip(branch.backbone.checkpoint, branch.backbone.dtype),
        split.classnames, split.num_classes,
    ).to(device)
    loss_fn = CmkdLoss.from_branch_config(branch, max_iter=total_steps)
    multiplier = branch.optim.param_group_multipliers.get("classifier", 1.0)
    optimizer = build_optimizer(model.param_groups(lr=1.0, head_multiplier=multiplier),
                                branch.optim)
    scheduler = build_lr_scheduler(optimizer, branch.optim, total_steps)

    best = 0.0
    started = time.time()
    for step in range(run_steps):
        model.train()
        batch = stream.next().to(device)

        optimizer.zero_grad()
        source_features = model.features(batch.img_x)
        target_features = model.features(batch.img_u)
        out = loss_fn(
            source_logits=model.head(source_features),
            source_label=batch.label_x,
            source_cosine_logits=model.encoder.cosine_logits(source_features),
            target_logits=model.head(target_features),
            target_cosine_logits=model.encoder.cosine_logits(target_features),
            step=step,
        )
        out.total.backward()
        if branch.optim.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad],
                max_norm=branch.optim.grad_clip,
            )
        optimizer.step()
        model.ema_update(momentum_at(step, branch.ema))
        scheduler.step()

        if (step + 1) % print_every == 0:
            print(f"step {step + 1}/{run_steps}  loss {float(out.total.detach()):.4f} "
                  f"(clf {float(out.clf.detach()):.4f} "
                  f"task {float(out.task.detach()):.4f} "
                  f"distill {float(out.distill.detach()):.4f} "
                  f"reg {float(out.reg.detach()):.4f}) "
                  f"ramp {out.ramp:.4f}  "
                  f"lr {lr_at(step + 1, branch.optim, total_steps):.3e}", flush=True)

        if (step + 1) % eval_every == 0 or (step + 1) == run_steps:
            model.eval()
            student = evaluate(model.logits, test_loader, device)
            teacher = evaluate(model.teacher_logits, test_loader, device)
            best = max(best, student.accuracy)
            print(f"[eval] step {step + 1}  student {student.accuracy:.2f}% "
                  f"(loss {student.loss:.4f})  teacher {teacher.accuracy:.2f}% "
                  f"(loss {teacher.loss:.4f})  best student {best:.2f}%", flush=True)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({
                    "step": step + 1,
                    "student_acc": student.accuracy, "student_loss": student.loss,
                    "teacher_acc": teacher.accuracy, "teacher_loss": teacher.loss,
                    "best_student_acc": best,
                    "loss": float(out.total.detach()), "clf": float(out.clf.detach()),
                    "task": float(out.task.detach()),
                    "distill": float(out.distill.detach()),
                    "reg": float(out.reg.detach()), "ramp": out.ramp,
                    "teacher_updates": model.teacher_updates,
                }) + "\n")
            if student.accuracy >= best:
                torch.save(model.state_dict(), output_dir / "model-best.pt")
            torch.save(model.state_dict(), output_dir / "model-last.pt")

    print(f"done in {time.time() - started:.1f}s. best student accuracy {best:.2f}%", flush=True)
    return best


if __name__ == "__main__":
    main()
