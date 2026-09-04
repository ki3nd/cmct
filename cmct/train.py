"""Cross-model co-training entry point.

Every value a run needs comes from `cmct.config.Config`, resolved from one
YAML file. The order of operations within a macro-step is the whole point of
this file: see the comments at each section below for why that order matters.

Two branches train side by side, each teaching the other:

  * branch_lora -- CLIP with depth-ramped LoRA, one update per macro-step,
    its own EMA teacher.
  * branch_mlp -- a learned head on CLIP features, trained with the CMKD
    self-training loss, `train.mlp_steps_per_iter` inner steps per
    macro-step, its own EMA teacher (backbone plus a separate head).

Usage:
    python -m cmct.train --config configs/officehome_a2c.yaml \\
        [--trace-out traces/cmct.json] [--trace-iters 20]
"""

import argparse
import copy
import dataclasses
import json
import os.path as osp

import torch
from torch.nn import functional as F

from cmct.branch_lora import (
    FrozenTeacherCLIP,
    LoraCLIP,
    copy_lora_params,
    ema_update_lora_params,
    load_clip_to_cpu,
)
from cmct.branch_lora.lora import apply_lora, save_lora
from cmct.branch_mlp import TransferNet, ema_update_teacher, prompts_for
from cmct.config import Config, resolve, to_dassl_cfg
from cmct.data import CyclingLoader, build_data_manager
from cmct.evaluate import evaluate
from cmct.losses import DebiasTracker, masked_cross_entropy, mk_mmd
from vendor.dassl.utils import mkdir_if_missing, set_random_seed

# The LoRA branch's prompt: every dataset this entry point supports maps to
# this one template. branch_mlp uses its own separate hardcoded prompt lists
# (cmct/branch_mlp/backbone.py) -- the two are deliberately different and must
# not be conflated.
LORA_PROMPT_TEMPLATE = "a photo of a {}."



def ema_momentum_at(step: int, momentum: float, schedule: str, hard_copy_iters: int) -> float:
    """Momentum shared by the CMKD teacher's backbone and its head.

    "dacs" ramps min(t / (t + 1), momentum) from t = 0, so the first update is a
    hard copy of the student's already-stepped weights and it approaches the
    target over roughly momentum / (1 - momentum) steps. "hard_copy" keeps
    momentum at 0 for hard_copy_iters steps, then jumps.

    Backbone and head always share this value: the head is trained against the
    live backbone's features, so a smoothed backbone paired with an
    instantaneously-tracking head would be inconsistent.
    """
    if schedule == "hard_copy":
        return 0.0 if step < hard_copy_iters else momentum
    return min(step / (step + 1), momentum)


def build_lora_pair(config: Config, classnames, device):
    """The LoRA student and its frozen EMA teacher.

    Supports ViT backbones only.
    """
    lora_kwargs = {
        "backbone_name": config.branch_lora.backbone.name,
        "position": config.branch_lora.lora.position,
        "params": config.branch_lora.lora.params,
        "r": config.branch_lora.lora.r,
        "alpha": config.branch_lora.lora.alpha,
        "dropout": config.branch_lora.lora.dropout,
        "rank_ramp": config.branch_lora.lora.rank_ramp,
    }

    clip_student = load_clip_to_cpu(config.branch_lora.backbone.name, config.branch_lora.backbone.path)
    clip_teacher = load_clip_to_cpu(config.branch_lora.backbone.name, config.branch_lora.backbone.path)
    if config.branch_lora.precision == "fp32":
        clip_student.float()
        clip_teacher.float()

    student = LoraCLIP(classnames, clip_student, LORA_PROMPT_TEMPLATE)
    teacher = FrozenTeacherCLIP(classnames, clip_teacher, LORA_PROMPT_TEMPLATE)
    lora_layers_student = apply_lora(student, **lora_kwargs)
    lora_layers_teacher = apply_lora(teacher, **lora_kwargs)

    for param in student.parameters():
        param.requires_grad_(False)
    for name, param in student.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
    for param in teacher.parameters():
        param.requires_grad_(False)

    copy_lora_params(student, teacher)
    student.to(device)
    teacher.to(device)
    teacher.eval()
    return student, teacher, lora_layers_student, lora_layers_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trace-out", type=str, default="",
                        help="Write a per-iteration loss trace to this JSON path.")
    parser.add_argument("--trace-iters", type=int, default=0,
                        help="Stop after this many macro-steps when tracing (0 = run to completion).")
    args = parser.parse_args()

    config = resolve(Config.from_yaml(args.config))
    lora_enabled = config.branch_lora.enabled
    mlp_enabled = config.branch_mlp.enabled
    # `resolve()` has already forced the surviving branch's cross_weight to 0.0
    # when the other one is off; these only report that consequence. Both being
    # off is rejected in config._validate, so exactly one of the two messages
    # can fire.
    if not lora_enabled:
        print("branch_lora.enabled is false: branch_mlp.cross_weight is 0.0 -- "
              "no LoRA teacher exists to cross-teach with")
    if not mlp_enabled:
        print("branch_mlp.enabled is false: branch_lora.cross_weight is 0.0 -- "
              "no CMKD teacher exists to cross-teach with. The LoRA branch still "
              "trains on source CE, self-distillation from its own EMA teacher "
              "(zero-shot CLIP during warmup) and MK-MMD.")
    print(f"Resolved config ({args.config}):")
    print(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True))

    cfg = to_dassl_cfg(config)
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
    mkdir_if_missing(config.output_dir)
    mkdir_if_missing(osp.join(config.output_dir, "teacher_lora"))
    if config.seed >= 0:
        set_random_seed(config.seed)

    # THE ORDER OF THE NEXT THREE BLOCKS IS LOAD-BEARING -- when both branches
    # are on. With branch_mlp.enabled false the second DataManager below is
    # never built, so the LoRA branch's own loaders draw DIFFERENT permutation
    # seeds than they would in a co-training run. A LoRA-only run is therefore
    # NOT bit-comparable against a co-training one; compare them statistically,
    # over seeds, not step by step.
    #
    # THE ORDER OF THE NEXT THREE BLOCKS IS LOAD-BEARING. The two DataManager
    # constructions are deliberately NOT adjacent, because
    # `CyclingLoader.__init__` calls `iter()` eagerly and a DataLoader's
    # RandomSampler has generator=None, so each wrap draws its permutation
    # seed from the global torch RNG. Collapsing this into one "build all the
    # loaders" call moves two of those four draws to before the block below,
    # which shifts every later draw.
    #
    # What that actually changes is NOT the LoRA weights: LoRALayer's
    # init_lora_param overwrites its nn.init.normal_ draw with a deterministic
    # SVD of the pretrained weight, so every lora_* tensor is identical
    # regardless of RNG state (measured). The real consumers downstream are
    # branch_mlp's sampler permutation stream and TransferNet.classifier_layer's
    # nn.init.normal_(std=0.001) head init. Verify against THOSE -- checking
    # the LoRA weights will show them unchanged and mislead you into thinking
    # this interleave is inert. See tests/test_data.py's RNG-order guard.
    print("Building the LoRA branch's data loader")
    dm_lora = build_data_manager(cfg, strong_aug=config.data.strong_aug)
    train_loader_x_lora = CyclingLoader(dm_lora.train_loader_x)
    train_loader_u_lora = CyclingLoader(dm_lora.train_loader_u)
    test_loader = dm_lora.test_loader
    classnames = dm_lora.dataset.classnames
    num_classes = dm_lora.num_classes

    # Only branch_mlp reads these prompts, so a LoRA-only run neither needs
    # them nor should be stopped by them.
    if mlp_enabled:
        # The CMKD branch prompts CLIP with a per-dataset list of literal strings
        # (branch_mlp/backbone.py), and nothing in the code makes that list agree
        # with the order dassl assigns labels in. If they ever diverged, that
        # branch's cosine logits would be silently mislabelled and every test would
        # still pass -- no unit test can catch it, because dassl sorts the CASED
        # directory names ("TV" < "Table") and a synthetic test tree cannot
        # reconstruct the real ones. So assert it here, against the dataset
        # actually on disk, before any GPU time is spent. This also catches a
        # partly-downloaded dataset, which would otherwise surface as a shape
        # mismatch deep inside CMKD.forward.
        prompts = prompts_for(config.data.name)
        if len(prompts) != num_classes:
            raise RuntimeError(
                f"branch_mlp has {len(prompts)} prompts for '{config.data.name}' but the "
                f"dataset at {config.data.root} yielded {num_classes} classes -- a partly "
                f"downloaded dataset, or the wrong data.name"
            )
        misaligned = [
            (i, p, c) for i, (p, c) in enumerate(zip(prompts, classnames))
            if not p.endswith(c.replace("_", " "))
        ]
        if misaligned:
            i, prompt, classname = misaligned[0]
            raise RuntimeError(
                f"branch_mlp's prompt list is out of order with dassl's classnames: "
                f"at index {i} the prompt is {prompt!r} but the class is {classname!r} "
                f"({len(misaligned)} of {num_classes} misaligned)"
            )

    if lora_enabled:
        print("Building the LoRA branch's student/teacher")
        student_lora, teacher_lora, _, lora_layers_teacher = build_lora_pair(
            config, classnames, device
        )
    else:
        student_lora, teacher_lora, lora_layers_teacher = None, None, None

    if mlp_enabled:
        print("Building the CMKD branch's data loader (dassl, same transform)")
        # A SEPARATE DataManager from the LoRA branch's: its own independent
        # shuffled stream.
        dm_mlp = build_data_manager(cfg, strong_aug=config.data.strong_aug)
        train_loader_x_mlp = CyclingLoader(dm_mlp.train_loader_x)
        train_loader_u_mlp = CyclingLoader(dm_mlp.train_loader_u)

        mlp_total_iters = config.train.iters * config.train.mlp_steps_per_iter

        print("Building the CMKD branch's TransferNet")
        model_mlp = TransferNet(
            prompts_for(config.data.name),
            model_name=config.branch_mlp.backbone.name,
            num_classes=num_classes,
            label_smoothing=config.branch_mlp.label_smoothing,
            lambdas=config.branch_mlp.lambdas,
            lamb_gamma=config.branch_mlp.lamb_gamma,
            max_iter=mlp_total_iters,
        ).to(device)
        # branch_mlp's classifier head needs its own teacher too, since it is a
        # freshly-initialized (near-random, small-std) module and the backbone's
        # EMA teacher alone would leave the head untracked. Hard-copied every
        # step for ema.hard_copy_iters under the "hard_copy" schedule, then a
        # real EMA after that.
        teacher_classifier = copy.deepcopy(model_mlp.classifier_layer).to(device)
        for param in teacher_classifier.parameters():
            param.requires_grad_(False)
        teacher_classifier.eval()
    else:
        model_mlp, teacher_classifier = None, None
        mlp_total_iters = 0

    if lora_enabled:
        # branch_lora.lr, passed here, never actually drives a step: every
        # macro-step during warmup writes `pg["lr"]` directly onto every param
        # group (see the warmup block below), and `CosineAnnealingLR` (built
        # below) is chainable -- it scales whatever LR is already on the param
        # group rather than recomputing from `base_lrs`. So the effective
        # schedule is warmup.lr held constant through warmup, then annealed to
        # 0 from there. The `lr` key set here is overwritten before it ever
        # takes a step.
        optim_lora = torch.optim.SGD(
            [p for p in student_lora.parameters() if p.requires_grad],
            lr=config.branch_lora.lr,
            momentum=config.branch_lora.momentum,
            weight_decay=config.branch_lora.weight_decay,
        )
    if mlp_enabled:
        # initial_lr=1.0 is baked into the param groups via get_parameters();
        # the LR value itself lives entirely in the LambdaLR below.
        optim_mlp = torch.optim.SGD(
            model_mlp.get_parameters(
                initial_lr=1.0, classifier_lr_mult=config.branch_mlp.classifier_lr_mult
            ),
            lr=config.branch_mlp.lr,
            momentum=config.branch_mlp.momentum,
            weight_decay=config.branch_mlp.weight_decay,
            nesterov=config.branch_mlp.nesterov,
        )
        sched_mlp = torch.optim.lr_scheduler.LambdaLR(
            optim_mlp,
            lr_lambda=lambda step: config.branch_mlp.lr
            * (1.0 + config.branch_mlp.lr_gamma * float(step)) ** (-config.branch_mlp.lr_decay),
        )

    if lora_enabled:
        lora_post_warmup = max(config.train.iters - config.branch_lora.warmup.iters, 1)
        sched_lora = torch.optim.lr_scheduler.CosineAnnealingLR(optim_lora, T_max=lora_post_warmup)

        print("Building the frozen zero-shot CLIP (used only during the LoRA branch's warmup)")
        clip_frozen = load_clip_to_cpu(config.branch_lora.backbone.name, config.branch_lora.backbone.path)
        if config.branch_lora.precision == "fp32":
            clip_frozen.float()
        teacher_frozen = FrozenTeacherCLIP(classnames, clip_frozen, LORA_PROMPT_TEMPLATE).to(device)
        for param in teacher_frozen.parameters():
            param.requires_grad_(False)
        teacher_frozen.eval()
    else:
        teacher_frozen = None

    confi = config.pseudo_label.threshold
    best_acc_lora, best_acc_mlp, best_acc_ens = 0.0, 0.0, 0.0

    # pseudo_label.debias: two INDEPENDENT trackers, one per branch. The LoRA
    # one covers teacher_frozen (the warmup self-reference) and teacher_lora
    # (the self-reference post-warmup, and the cross-reference to the CMKD
    # branch); the CMKD one covers the CMKD teacher's cosine branch, and only
    # when branch_mlp.self_from_teacher is set. Neither covers
    # teacher_classifier's output, which is not a CLIP prediction.
    use_debias = config.pseudo_label.debias.enabled
    debias_lora = DebiasTracker(
        num_classes, config.pseudo_label.debias.tau,
        config.pseudo_label.debias.momentum, device,
    ) if use_debias and lora_enabled else None
    debias_mlp = DebiasTracker(
        num_classes, config.pseudo_label.debias.tau,
        config.pseudo_label.debias.momentum, device,
    ) if use_debias and mlp_enabled else None

    trace = {"config": dataclasses.asdict(config), "iters": [], "evals": []}

    mlp_step_global = 0
    for macro in range(config.train.iters):
        # The LoRA branch's warmup runs on the macro-step cadence and the CMKD
        # branch's on the mlp_step_global cadence (mlp_steps_per_iter times
        # denser), tracked separately so each branch's cross-teaching onset can
        # be ablated independently -- they are NOT required to line up.
        if lora_enabled:
            in_warmup_lora = macro < config.branch_lora.warmup.iters

            if not in_warmup_lora and teacher_frozen is not None:
                # teacher_frozen is only ever read inside `if in_warmup_lora:`
                # blocks -- once the warmup ends it is dead weight (a whole
                # extra CLIP model) sitting on the GPU for the rest of the run.
                # Free it right here instead of holding it until the process
                # exits.
                del teacher_frozen, clip_frozen
                teacher_frozen = None
                torch.cuda.empty_cache()

            batch_x_lora = train_loader_x_lora.next()
            batch_u_lora = train_loader_u_lora.next()
            image_x_lora = batch_x_lora["img"].to(device)
            label_x_lora = batch_x_lora["label"].to(device)
            # weak = current/default view, used for every TEACHER-side
            # computation below (self-reference, cross-reference) -- unchanged.
            # strong = data.strong_aug's mildly harder view, used ONLY for the
            # LoRA student's OWN forward further down (feeding both the self
            # and the cross loss) -- free here, it just swaps which image
            # tensor the forward call gets, no extra pass.
            image_u_lora = batch_u_lora["img"].to(device)
            image_u_lora_strong = batch_u_lora["img2"].to(device) if config.data.strong_aug else image_u_lora

            with torch.no_grad():
                if in_warmup_lora:
                    logits_cross_for_lora, _ = teacher_frozen(image_u_lora)
                    if use_debias:
                        # teacher_frozen IS a genuine CLIP zero-shot prediction
                        # (the same tracker teacher_lora uses post-warmup, so
                        # it carries over seamlessly once teacher_lora takes
                        # over).
                        logits_cross_for_lora = debias_lora.correct(logits_cross_for_lora)
                    prob_cross_for_lora = F.softmax(logits_cross_for_lora, dim=-1)
                elif mlp_enabled:
                    # The CMKD teacher for the LoRA branch's cross term: EMA
                    # backbone (model_mlp.teacher_model) through its own EMA
                    # head (teacher_classifier) -- both are temporal-fusion
                    # EMAs, no live parameters involved on that side.
                    feat_cross_for_lora = model_mlp.teacher_model.forward_features(image_u_lora)
                    logits_cross_for_lora = teacher_classifier(feat_cross_for_lora)
                    prob_cross_for_lora = F.softmax(logits_cross_for_lora, dim=-1)
                else:
                    # No CMKD branch to cross-teach from. This is the ONLY arm
                    # that skips the teacher pass: with branch_mlp on but
                    # branch_lora.cross_weight at 0.0 the pass still runs each
                    # macro-step and its result is multiplied by zero.
                    prob_cross_for_lora = None

        if mlp_enabled:
            for _ in range(config.train.mlp_steps_per_iter):
                in_warmup_mlp = mlp_step_global < config.branch_mlp.warmup_iters
                batch_x_mlp = train_loader_x_mlp.next()
                batch_u_mlp = train_loader_u_mlp.next()
                data_x_mlp = batch_x_mlp["img"].to(device)
                label_x_mlp = batch_x_mlp["label"].to(device)
                data_u_mlp = batch_u_mlp["img"].to(device)
                # weak (data_u_mlp) feeds every TEACHER-side computation below
                # (self-reference, reg_loss, cross-reference to the LoRA teacher)
                # -- unchanged. strong feeds ONLY the classifier's own prediction
                # (own_pred_target_img below), which costs an EXTRA full-gradient
                # backbone forward pass here.
                data_u_mlp_strong = batch_u_mlp["img2"].to(device) if config.data.strong_aug else data_u_mlp

                # model_mlp.train() would ALSO flip teacher_model into train mode
                # (nn.Module.train() recurses into every submodule) -- it must stay
                # permanently eval (see TransferNet.__init__ and
                # ema_update_teacher). Set base_network/classifier_layer's mode
                # directly instead.
                model_mlp.base_network.train()
                model_mlp.classifier_layer.train()
                # TransferNet.forward() returns target_logits alongside clf_loss
                # (label_smoothing baked in) and transfer_loss, the full CMKD
                # self-training loss (task_loss + distill_loss + reg_loss) -- no
                # teacher/EMA involved in it by default (the real CMKD design),
                # UNLESS branch_mlp.self_from_teacher is set.
                self_ref_logit_clip = None
                if config.branch_mlp.self_from_teacher:
                    with torch.no_grad():
                        teacher_feat_u_mlp = model_mlp.teacher_model.forward_features(data_u_mlp)
                        self_ref_logit_clip = model_mlp.teacher_model.forward_head(teacher_feat_u_mlp).detach()
                        if use_debias:
                            # The CMKD teacher's cosine branch IS a genuine CLIP
                            # prediction (unlike teacher_classifier) -- its own
                            # tracker, separate from the LoRA branch's. Only
                            # reachable here, since this is the only place that
                            # cosine branch is computed OUTSIDE TransferNet's own
                            # forward().
                            self_ref_logit_clip = debias_mlp.correct(self_ref_logit_clip)
                clf_loss, transfer_loss, target_logits_mlp = model_mlp(
                    data_x_mlp, data_u_mlp, label_x_mlp,
                    self_ref_logit_clip=self_ref_logit_clip,
                    own_pred_target_img=(data_u_mlp_strong if config.data.strong_aug else None),
                )
                loss_mlp_self = clf_loss + transfer_loss

                if in_warmup_mlp or not lora_enabled:
                    loss_mlp_cross = torch.tensor(0.0, device=device)
                    loss_mlp = loss_mlp_self
                else:
                    # Cross-teaching is NOT part of CMKD -- added on top here.
                    # Reuses target_logits_mlp above instead of a separate
                    # model_mlp.predict(data_u_mlp) call: that used to cost a
                    # WHOLE EXTRA forward pass through the full CLIP backbone and
                    # mutated classifier_layer's BatchNorm1d running stats an extra
                    # time regardless of the cross weight's value. Neither concern
                    # applies anymore -- target_logits_mlp is free.
                    with torch.no_grad():
                        logits_lora_on_u_mlp, _ = teacher_lora(data_u_mlp)
                        if use_debias:
                            logits_lora_on_u_mlp = debias_lora.correct(logits_lora_on_u_mlp)
                        prob_cross_mlp = F.softmax(logits_lora_on_u_mlp, dim=-1)
                    loss_mlp_cross = masked_cross_entropy(target_logits_mlp, prob_cross_mlp, confi)
                    loss_mlp = loss_mlp_self + config.branch_mlp.cross_weight * loss_mlp_cross

                optim_mlp.zero_grad()
                loss_mlp.backward()
                optim_mlp.step()
                sched_mlp.step()

                # The EMA update runs AFTER optim_mlp.step(), and with the
                # PRE-increment `mlp_step_global` (incremented only at the bottom
                # of this loop) -- so under the "dacs" schedule, the very first
                # EMA update (step 0) is a hard copy of weights that have already
                # taken one optimizer step, not of the random init. Both matter:
                # reversing the order would make step 0's hard copy capture the
                # random init instead, and using the post-increment step would
                # shift every later momentum value by one step.
                #
                # Backbone and head EMA always share the SAME momentum this step --
                # giving the backbone a real momentum while the head hard-copies
                # (or vice versa) would leave the CMKD teacher as a smoothed
                # backbone paired with an instantaneously-tracking head -- a
                # mismatch, since the head is trained (via TransferNet.forward())
                # against the LIVE backbone's features, not the lagging EMA one.
                ema_momentum = ema_momentum_at(
                    mlp_step_global,
                    config.branch_mlp.ema.momentum,
                    config.branch_mlp.ema.schedule,
                    config.branch_mlp.ema.hard_copy_iters,
                )
                ema_update_teacher(model_mlp.teacher_model, model_mlp.base_network, ema_momentum)
                ema_update_teacher(teacher_classifier, model_mlp.classifier_layer, ema_momentum)
                mlp_step_global += 1

        if lora_enabled:
            logits_x_lora, feat_x_lora = student_lora(image_x_lora)
            loss_x_lora = F.cross_entropy(logits_x_lora, label_x_lora)
            # MK-MMD stays weak-vs-weak (the source has no strong view either)
            # -- comparing it against a strong-augmented target would confound
            # domain shift with augmentation-strength shift, which isn't what
            # MK-MMD is meant to measure. Cheap for a LoRA student, so just an
            # extra forward call rather than reusing the self/cross forward's
            # own feature.
            _, feat_u_lora_weak = student_lora(image_u_lora)
            loss_mmd_lora = mk_mmd(feat_x_lora, feat_u_lora_weak)
            # The self and cross losses use the STRONG view instead.
            logits_u_lora, _ = student_lora(image_u_lora_strong)

            if in_warmup_lora:
                loss_u_lora_self = masked_cross_entropy(logits_u_lora, prob_cross_for_lora, confi)
                loss_u_lora_cross = torch.tensor(0.0, device=device)
                loss_lora = loss_x_lora + loss_u_lora_self + config.branch_lora.mmd_weight * loss_mmd_lora
            else:
                with torch.no_grad():
                    logits_teacher_lora_self, _ = teacher_lora(image_u_lora)
                    if use_debias:
                        logits_teacher_lora_self = debias_lora.correct(logits_teacher_lora_self)
                    prob_self_lora = F.softmax(logits_teacher_lora_self, dim=-1)
                loss_u_lora_self = masked_cross_entropy(logits_u_lora, prob_self_lora, confi)
                loss_u_lora_cross = (
                    masked_cross_entropy(logits_u_lora, prob_cross_for_lora, confi)
                    if prob_cross_for_lora is not None
                    else torch.tensor(0.0, device=device)
                )
                loss_lora = (
                    loss_x_lora + loss_u_lora_self
                    + config.branch_lora.cross_weight * loss_u_lora_cross
                    + config.branch_lora.mmd_weight * loss_mmd_lora
                )

            if in_warmup_lora:
                for pg in optim_lora.param_groups:
                    pg["lr"] = config.branch_lora.warmup.lr

            optim_lora.zero_grad()
            loss_lora.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student_lora.parameters() if p.requires_grad],
                max_norm=config.branch_lora.grad_clip,
            )
            optim_lora.step()
            if not in_warmup_lora:
                sched_lora.step()

            ema_update_lora_params(teacher_lora, student_lora, config.branch_lora.ema_momentum)

        if (macro + 1) % config.train.print_freq == 0:
            header = f"macro [{macro + 1}/{config.train.iters}]"
            if mlp_enabled:
                header += f" (cmkd step {mlp_step_global}/{mlp_total_iters})"
            segments = []
            if lora_enabled:
                acc_x_lora = (logits_x_lora.argmax(-1) == label_x_lora).float().mean().item() * 100
                segments.append(
                    f"loss_lora {loss_lora.item():.4f} (x {loss_x_lora.item():.4f} "
                    f"self {loss_u_lora_self.item():.4f} cross {loss_u_lora_cross.item():.4f} "
                    f"mmd {loss_mmd_lora.item():.4f}) acc_x_lora {acc_x_lora:.2f}"
                )
            if mlp_enabled:
                with torch.no_grad():
                    acc_x_mlp = (torch.max(model_mlp.predict(data_x_mlp), 1)[1] == label_x_mlp).float().mean().item() * 100
                segments.append(
                    f"loss_mlp {loss_mlp.item():.4f} (clf {clf_loss.item():.4f} "
                    f"transfer {transfer_loss.item():.4f} cross {loss_mlp_cross.item():.4f}) "
                    f"acc_x_mlp {acc_x_mlp:.2f}"
                )
            print(f"{header} " + " | ".join(segments))

        # mlp_step_global has already been advanced past this macro-step's
        # inner iterations by here, so ">= warmup_iters" (not "==") catches the
        # macro-step the CMKD branch's own warmup ends in, even though it isn't
        # tied 1:1 to macro-steps like the LoRA branch's is.
        mlp_warmup_just_ended = mlp_enabled and (
            mlp_step_global >= config.branch_mlp.warmup_iters
            and mlp_step_global - config.train.mlp_steps_per_iter < config.branch_mlp.warmup_iters
        )
        lora_warmup_just_ended = lora_enabled and (macro + 1) == config.branch_lora.warmup.iters
        if (
            (macro + 1) % config.train.eval_freq == 0
            or (macro + 1) == config.train.iters
            or lora_warmup_just_ended
            or mlp_warmup_just_ended
        ):
            if mlp_enabled:
                model_mlp.eval()
            acc_lora, acc_mlp, acc_ens = evaluate(
                teacher_lora, model_mlp, teacher_classifier, test_loader, device
            )
            if args.trace_out:
                trace["evals"].append({"macro": macro + 1, "acc_lora": acc_lora,
                                       "acc_mlp": acc_mlp, "acc_ensemble": acc_ens})
            tags = []
            if lora_warmup_just_ended:
                tags.append("end of the LoRA branch's warmup")
            if mlp_warmup_just_ended:
                tags.append("end of the CMKD branch's warmup")
            tag = f" [{', '.join(tags)}]" if tags else ""
            lora_report = "n/a" if acc_lora is None else f"{acc_lora:.2f}%"
            mlp_report = "n/a" if acc_mlp is None else f"{acc_mlp:.2f}%"
            ens_report = "n/a" if acc_ens is None else f"{acc_ens:.2f}%"
            print(f"[eval] macro {macro + 1}{tag}: teacher_lora {lora_report} | "
                  f"teacher_mlp {mlp_report} | ensemble {ens_report}")

            if lora_enabled:
                save_lora(
                    lora_layers_teacher, osp.join(config.output_dir, "teacher_lora"),
                    filename="LoRA-last",
                    r=config.branch_lora.lora.r, alpha=config.branch_lora.lora.alpha,
                    params=config.branch_lora.lora.params,
                )
            if mlp_enabled:
                torch.save(model_mlp.state_dict(), osp.join(config.output_dir, "model_mlp-last.pt"))

            if lora_enabled and acc_lora > best_acc_lora:
                best_acc_lora = acc_lora
                save_lora(
                    lora_layers_teacher, osp.join(config.output_dir, "teacher_lora"),
                    filename="LoRA-best",
                    r=config.branch_lora.lora.r, alpha=config.branch_lora.lora.alpha,
                    params=config.branch_lora.lora.params,
                )
            if mlp_enabled and acc_mlp > best_acc_mlp:
                best_acc_mlp = acc_mlp
                torch.save(model_mlp.state_dict(), osp.join(config.output_dir, "model_mlp-best.pt"))
            if lora_enabled and mlp_enabled and acc_ens > best_acc_ens:
                best_acc_ens = acc_ens
                print(f"  new best ensemble ({best_acc_ens:.2f}%)")

        if args.trace_out:
            trace["iters"].append({
                "macro": macro + 1, "mlp_step": mlp_step_global,
                "loss_lora": loss_lora.item() if lora_enabled else None,
                "loss_source": loss_x_lora.item() if lora_enabled else None,
                "loss_self": loss_u_lora_self.item() if lora_enabled else None,
                "loss_cross": loss_u_lora_cross.item() if lora_enabled else None,
                "loss_mmd": loss_mmd_lora.item() if lora_enabled else None,
                "loss_mlp": loss_mlp.item() if mlp_enabled else None,
                "clf_loss": clf_loss.item() if mlp_enabled else None,
                "transfer_loss": transfer_loss.item() if mlp_enabled else None,
                "loss_mlp_cross": loss_mlp_cross.item() if mlp_enabled else None,
            })
            if args.trace_iters and macro + 1 >= args.trace_iters:
                break

    if args.trace_out:
        mkdir_if_missing(osp.dirname(args.trace_out) or ".")
        with open(args.trace_out, "w") as f:
            json.dump(trace, f, indent=2)
        print(f"trace written to {args.trace_out}")

    done = ["Done. best"]
    if lora_enabled:
        done.append(f"teacher_lora={best_acc_lora:.2f}%")
    if mlp_enabled:
        done.append(f"teacher_mlp={best_acc_mlp:.2f}%")
    if lora_enabled and mlp_enabled:
        done.append(f"ensemble={best_acc_ens:.2f}%")
    print(" ".join(done))


if __name__ == "__main__":
    main()
