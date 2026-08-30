"""Train branch 1 on its own: CLIP with LoRA on both towers, cosine
classification, an EMA teacher, and MK-MMD.

No cross-teaching and no second branch. The loop shape is the reference's, and
the parts that matter are:

  - THREE separate forward passes per step. With no strong augmentation two of
    them see the same image, but LoRA dropout makes them differ, so collapsing
    them changes the numbers.
  - the pseudo-label reference switches once: the frozen zero-shot CLIP through
    the first `warmup_steps`, the EMA teacher afterwards. The zero-shot model is
    released at the boundary -- it is a whole extra CLIP and nothing reads it
    again.
  - the learning rate is flat at `warmup_lr` through the warmup AND through the
    boundary step, then decays on a cosine whose amplitude is `warmup_lr`, not
    `lr`. A branch's `lr` does not govern its learning rate here; see
    cmct.engine.optim.lr_at.
  - evaluation scores the TEACHER. Branch 2 scores its student; this is the
    opposite, and `best` follows the teacher because that is what the reference
    reports.

Usage:
    python -m cmct.train_lora --config configs/experiment/lora_officehome_a2c.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cmct.backbones.clip import load_clip
from cmct.backbones.clip.download import resolve_checkpoint
from cmct.branches.lora_model import TEMPLATE, LoraModel
from cmct.config import BranchConfig, Config, dump, format_config, load_experiment
from cmct.data import BatchSource, build_split, build_test_loader
from cmct.engine import (
    build_lr_scheduler,
    build_optimizer,
    evaluate,
    lr_at,
    momentum_at,
)
from cmct.losses import LoraBranchLoss

BRANCH_TYPE = "lora_clip"
EXTRA_KEYS = {
    "mmd_weight", "lora_rank", "lora_alpha", "lora_dropout", "lora_params",
    "lora_encoders", "lora_positions", "lora_rank_ramp", "warmup_reference",
}

WARMUP_REFERENCES = ("zero_shot", "teacher")
"""What produces pseudo-labels while the branch is in warmup.

"zero_shot": a frozen zero-shot CLIP, replaced by the teacher when warmup ends.
This is what the reference does, and the default in the shipped config.

"teacher": the EMA teacher from step 0, and no frozen model is built at all. The
teacher is a reasonable reference this early precisely because it is still mostly
its own initialization, which reproduces zero-shot CLIP -- ~60% of it at step 50
under a flat 0.99 -- while also carrying what the student has learned. An
extension, not parity: the reference implements only "zero_shot".
"""

LAST_MODEL: LoraModel | None = None
"""The model from the most recent run, so a test can inspect it afterwards. Not
part of what main() returns."""


def ema_momentum(step: int, cfg) -> float:
    """This branch's EMA momentum, which differs from engine.ema.momentum_at at
    exactly one point: step 0 under the "const" schedule.

    momentum_at returns 0.0 at step 0 under every schedule, so the teacher throws
    its initialization away and restarts from the stepped student. That is right
    for branch 2, whose head is randomly initialized, and wrong here. This
    branch's initialization is zero-shot CLIP: SoRA's SVD init puts the top-r
    principal components of each weight into the LoRA factors and subtracts the
    residual back out of the frozen weight, so `frozen + scaling * B @ A`
    reconstructs the original weight (measured: 1.5e-08 max difference, and the
    model's output matches zero-shot CLIP to 2.4e-07). The factors themselves are
    NOT zero -- |A|max and |B|max are around 0.6 -- so the usual LoRA argument
    that a zero-initialized delta makes the starting point irrelevant does not
    apply.

    The reference therefore applies its constant from step 0
    (train_mfa_v2.py:943 passes `lambda k: args.s1_ema_momentum`, and
    _ema_update_lora_params has no first-step branch), leaving the teacher 99%
    zero-shot after step 0 and ~60% after 50 steps. That is also what makes the
    end of warmup continuous: the pseudo-label source switches from the frozen
    zero-shot model to a teacher still mostly made of it.

    Only "const" is overridden. "ramp" is min(step / (step + 1), momentum), which
    is 0 at step 0 by its own formula, and a ramp asked for by name is a request
    for that hard copy.

    engine.ema.momentum_at is shared with branch 2 and is deliberately not
    touched; this wrapper keeps the change inside branch 1.
    """
    if step == 0 and cfg.schedule == "const":
        return cfg.momentum
    return momentum_at(step, cfg)


def set_seed(seed: int) -> None:
    """The four generators dassl's set_random_seed seeds -- which is all branch
    1's reference does (train_mfa_v2.py:517, dassl/utils/tools.py:73) -- plus two
    cuDNN flags it does NOT set.

    Without those two flags cuDNN picks convolution algorithms by benchmarking,
    so a rerun at a fixed seed still varies. That is a deliberate addition, not a
    reproduction: it costs some throughput and buys reruns that actually match.
    Branch 2's reference does set them; branch 1's does not.
    """
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
            raise SystemExit(
                f"branch {name!r} has type {branch.type!r}, not {BRANCH_TYPE!r}"
            )
        return branch

    candidates = [b for b in cfg.branches if b.type == BRANCH_TYPE]
    if not candidates:
        raise SystemExit(f"no branch of type {BRANCH_TYPE!r} in this config")
    if len(candidates) > 1:
        raise SystemExit(
            f"several branches of type {BRANCH_TYPE!r} "
            f"({[b.name for b in candidates]}); pass --branch"
        )
    return candidates[0]


def read_extra(branch: BranchConfig) -> dict:
    """`extra` is a free-form dict, so an unknown key would otherwise be ignored
    in silence -- exactly how a renamed knob stops taking effect."""
    unknown = sorted(set(branch.extra) - EXTRA_KEYS)
    if unknown:
        raise SystemExit(
            f"branches[{branch.name}].extra: unknown key(s) {unknown} for branch "
            f"type {BRANCH_TYPE!r}; valid: {sorted(EXTRA_KEYS)}"
        )
    missing = sorted(EXTRA_KEYS - set(branch.extra))
    if missing:
        raise SystemExit(
            f"branches[{branch.name}].extra: missing key(s) {missing}"
        )
    extra = dict(branch.extra)
    if extra["warmup_reference"] not in WARMUP_REFERENCES:
        raise SystemExit(
            f"branches[{branch.name}].extra.warmup_reference: "
            f"{extra['warmup_reference']!r} is not one of {list(WARMUP_REFERENCES)}"
        )
    return extra


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--branch", default=None,
                        help=f"branch name; defaults to the only {BRANCH_TYPE} branch")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many steps. A script-level knob for "
                             "short runs; it does NOT shorten the LR schedule or "
                             "the warmup, so the first N steps of a short run "
                             "match the first N of the real one")
    parser.add_argument("--device", default=None, help="overrides run.device")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> float:
    """Returns the best teacher accuracy."""
    global LAST_MODEL
    args = parse_args(argv)
    cfg = load_experiment(args.config)
    branch = pick_branch(cfg, args.branch)
    extra = read_extra(branch)

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
    run_steps = total_steps if args.max_steps is None else min(args.max_steps,
                                                              total_steps)
    warmup_steps = min(branch.warmup_steps, run_steps)
    eval_every = cfg.run.eval_freq * branch.steps_per_macro
    print_every = cfg.run.print_freq * branch.steps_per_macro

    print("=" * 78, flush=True)
    print(f"resolved config ({args.config})", flush=True)
    print("=" * 78, flush=True)
    print(format_config(cfg), end="", flush=True)

    split = build_split(cfg.dataset, cfg.data)
    param_dtype = (torch.float16 if branch.backbone.dtype == "fp16"
                   else torch.float32)
    model = LoraModel(
        load_clip(branch.backbone.checkpoint, branch.backbone.dtype),
        split.classnames,
        rank=extra["lora_rank"], alpha=extra["lora_alpha"],
        dropout=extra["lora_dropout"], params=tuple(extra["lora_params"]),
        rank_ramp=extra["lora_rank_ramp"], positions=extra["lora_positions"],
        encoders=tuple(extra["lora_encoders"]), param_dtype=param_dtype,
    ).to(device)
    uses_zero_shot = warmup_steps > 0 and extra["warmup_reference"] == "zero_shot"
    if uses_zero_shot:
        # A second read of the checkpoint, not a copy of the student: apply_lora
        # mutated the student in place, and LoraModel rejects a model that
        # already carries LoRA. Under warmup_reference "teacher" this is skipped
        # entirely -- it is a whole extra CLIP on the device that nothing reads.
        model.attach_zero_shot(
            load_clip(branch.backbone.checkpoint, branch.backbone.dtype).to(device)
        )

    trainable = sum(p.numel() for p in model.trainable_parameters())
    n_evals = len({*range(eval_every, run_steps + 1, eval_every), run_steps,
                   *([warmup_steps] if 0 < warmup_steps <= run_steps else [])})
    last = max(run_steps - 1, 0)
    checkpoint = (str(resolve_checkpoint(branch.backbone.checkpoint))
                  if Path(branch.backbone.checkpoint).is_file()
                  else branch.backbone.checkpoint)

    derived = {
        "branch": f"{branch.name} ({branch.type})",
        "device": device,
        "checkpoint": checkpoint,
        "total_steps": total_steps,
        "run_steps": run_steps,
        "warmup_steps": warmup_steps,
        "eval_every": eval_every,
        "evaluations": n_evals,
        "print_every": print_every,
        "images": f"{len(split.train_x)} source / {len(split.train_u)} target "
                  f"/ {len(split.test)} test",
        "classes": split.num_classes,
        "warmup_reference": extra["warmup_reference"],
        "prompt_template": TEMPLATE,
        "prompt_examples": [TEMPLATE.format(c) for c in split.classnames[:3]],
        "trainable_parameters": trainable,
        # `lr` does not govern: the cosine's amplitude is warmup_lr. See
        # cmct.engine.optim.lr_at.
        "lr_first_last": [lr_at(0, branch.optim, total_steps, branch.warmup_steps),
                          lr_at(last, branch.optim, total_steps,
                                branch.warmup_steps)],
        "config_lr_unused": branch.optim.lr,
        "ema_momentum_at_0_1_last": [ema_momentum(0, branch.ema),
                                     ema_momentum(1, branch.ema),
                                     ema_momentum(last, branch.ema)],
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
    loss_fn = LoraBranchLoss(threshold=branch.pseudo_label.threshold,
                             reduce=branch.pseudo_label.self_reduce,
                             mmd_weight=extra["mmd_weight"])
    optimizer = build_optimizer(model.param_groups(lr=1.0), branch.optim)
    scheduler = build_lr_scheduler(optimizer, branch.optim, total_steps,
                                   branch.warmup_steps)

    best = 0.0
    started = time.time()
    for step in range(run_steps):
        model.train()
        batch = stream.next().to(device)

        in_warmup = step < warmup_steps
        if in_warmup and uses_zero_shot:
            reference_logits = model.zero_shot_logits(batch.img_u)
            reference_name = "zero_shot"
        else:
            reference_logits = model.teacher_logits(batch.img_u)
            reference_name = "teacher"
        reference = F.softmax(reference_logits, dim=-1)

        optimizer.zero_grad()
        # Three forward passes, as the reference does. With strong_aug off the
        # second and third see the same image, but LoRA dropout makes them
        # differ, so they are not interchangeable.
        source_logits, source_features = model(batch.img_x)
        target_features = (model.features(batch.img_u)
                           if extra["mmd_weight"] > 0 else None)
        target_logits = model.logits(batch.student_img_u)

        out = loss_fn(
            source_logits=source_logits, source_label=batch.label_x,
            target_logits=target_logits, reference_probabilities=reference,
            source_features=source_features, target_features=target_features,
            reference_name=reference_name,
        )
        out.total.backward()
        if branch.optim.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()),
                                           max_norm=branch.optim.grad_clip)
        optimizer.step()
        scheduler.step()
        model.ema_update(ema_momentum(step, branch.ema))

        if uses_zero_shot and in_warmup and step + 1 == warmup_steps:
            model.release_zero_shot()

        current_lr = lr_at(step, branch.optim, total_steps, branch.warmup_steps)
        if (step + 1) % print_every == 0:
            print(f"step {step + 1}/{run_steps}  loss {float(out.total.detach()):.4f} "
                  f"(src {float(out.source_ce.detach()):.4f} "
                  f"pl {float(out.pseudo_label.detach()):.4f} "
                  f"mmd {float(out.mmd.detach()):.4f}) "
                  f"mask {out.mask_ratio:.2f} ref {out.reference} "
                  f"lr {current_lr:.3e}", flush=True)

        # The warmup boundary gets its own evaluation regardless of the cadence.
        # It is the one step where the branch changes what it learns from -- the
        # frozen zero-shot reference is replaced by the EMA teacher -- so the
        # accuracy on either side of it is the number that says whether warmup
        # was long enough. The reference evaluates here too, and tags the line
        # "end of s1 warmup" (train_mfa_v2.py:968, :974).
        at_warmup_end = warmup_steps > 0 and (step + 1) == warmup_steps
        if (step + 1) % eval_every == 0 or (step + 1) == run_steps or at_warmup_end:
            model.eval()
            teacher = evaluate(model.teacher_logits, test_loader, device)
            student = evaluate(model.logits, test_loader, device)
            improved = teacher.accuracy > best
            best = max(best, teacher.accuracy)
            tag = "  (end of warmup)" if at_warmup_end else ""
            print(f"[eval] step {step + 1}{tag}  teacher {teacher.accuracy:.2f}% "
                  f"(loss {teacher.loss:.4f})  student {student.accuracy:.2f}% "
                  f"(loss {student.loss:.4f})  best teacher {best:.2f}%", flush=True)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({
                    "step": step + 1,
                    "teacher_acc": teacher.accuracy, "teacher_loss": teacher.loss,
                    "student_acc": student.accuracy, "student_loss": student.loss,
                    "best_teacher_acc": best,
                    "loss": float(out.total.detach()),
                    "source_ce": float(out.source_ce.detach()),
                    "pseudo_label": float(out.pseudo_label.detach()),
                    "mmd": float(out.mmd.detach()),
                    "mask_ratio": out.mask_ratio,
                    "reference": out.reference,
                    "at_warmup_end": at_warmup_end,
                    "teacher_updates": model.teacher_updates,
                    "lr": current_lr,
                }) + "\n")
            # Only the LoRA factors: the frozen weights are reproducible from the
            # checkpoint, and this is what the reference's save_lora stores. The
            # TEACHER's factors, because the teacher is what `best` follows and
            # what evaluation scored -- saving the student would store a model
            # whose accuracy was never measured.
            #
            # `improved` is strict, and computed BEFORE `best` is updated, which
            # is the reference's guard. Comparing `teacher.accuracy >= best`
            # after the update is equivalent except on a TIE, where it replaces
            # the earlier of two equally good models with the later one.
            torch.save(model.teacher_lora_state_dict(), output_dir / "model-last.pt")
            if improved:
                torch.save(model.teacher_lora_state_dict(), output_dir / "model-best.pt")

    print(f"done in {time.time() - started:.1f}s. best teacher accuracy {best:.2f}%", flush=True)
    LAST_MODEL = model
    return best


if __name__ == "__main__":
    main()
