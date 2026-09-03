# Design

This is a design document for someone extending `cmct` -- in particular,
someone replacing the CMKD loss. That is why `branch_mlp` is named for its
head (a CLIP backbone plus a linear classifier) rather than for the loss it
currently trains against: the loss is meant to be swappable, and the model
name should not have to change when it is swapped.

## Architecture

Two branches are trained side by side on the same source/target data, each
supplying pseudo-labels to the other through its own EMA teacher:

- **`branch_lora`** (`cmct/branch_lora/`): CLIP ViT-B/16 wrapped as a
  cosine-similarity classifier (`LoraCLIP`), with LoRA adapters injected into
  the attention layers of both the text and vision transformers
  (`cmct/branch_lora/lora/`). The rank of each adapter grows with block
  depth -- see "LoRA rank ramp" below.
- **`branch_mlp`** (`cmct/branch_mlp/`): a CLIP ViT-B/16 backbone
  (`ClipBackbone`) feeding a small linear task head (`TransferNet`), trained
  with the CMKD self-training loss (`cmct/branch_mlp/loss.py`).

Both branches read the same data through independent `DataManager`s (see
"Two independent DataManagers" below) and are trained together by
`cmct/train.py`, which owns the whole macro/micro step loop.

## Macro/micro step structure

`cmct/train.py`'s main loop runs `train.iters` **macro-steps**. Each
macro-step:

1. Runs `train.mlp_steps_per_iter` **inner steps** of `branch_mlp` (an
   optimizer step, a scheduler step, and an EMA update each).
2. Runs exactly ONE `branch_lora` update: a forward/backward/optimizer step
   over the LoRA student, followed by one EMA update of the LoRA teacher.

So `branch_mlp` trains at `mlp_steps_per_iter` times the update density of
`branch_lora`, tracked by its own counter `mlp_step_global` rather than the
macro-step counter -- the two branches' warmup schedules are NOT required to
line up (see below).

## Warmup phases

Each branch has its own warmup, on its own cadence, and each switches on a
different thing when it ends:

- **`branch_lora`'s warmup** runs for `branch_lora.warmup.iters` macro-steps.
  During warmup, the LoRA student's pseudo-label reference is a frozen
  zero-shot CLIP teacher (`teacher_frozen`) rather than the LoRA EMA teacher,
  and the optimizer's LR is pinned to `branch_lora.warmup.lr` every step
  (see "branch_lora.lr never drives a step" below). Cross-teaching from
  `branch_mlp` is also off during this warmup. Once it ends: the LoRA EMA
  teacher (`teacher_lora`) takes over as the self-reference, cross-teaching
  from `branch_mlp`'s EMA teacher switches on, the cosine-annealed LR
  schedule starts stepping, and the frozen zero-shot CLIP is freed (see
  "Frozen CLIP lifetime" below).
- **`branch_mlp`'s warmup** runs for `branch_mlp.warmup_iters` INNER steps
  (not macro-steps). While `mlp_step_global < branch_mlp.warmup_iters`,
  cross-teaching from `branch_lora`'s teacher is off and `branch_mlp` trains
  on its own CMKD loss alone. Once it ends, the cross-teaching term
  (`masked_cross_entropy` against `teacher_lora`'s prediction) switches on.

## Loss composition

**`branch_lora`**, per macro-step:

- `loss_x_lora`: source cross-entropy.
- `loss_mmd_lora`: MK-MMD between source features and a WEAK target view's
  features (see "MK-MMD uses the weak view" below).
- `loss_u_lora_self`: confidence-masked cross-entropy of the LoRA student's
  target prediction against its own teacher's (or, during warmup, the frozen
  zero-shot teacher's) prediction.
- `loss_u_lora_cross` (post-warmup only): confidence-masked cross-entropy
  against `branch_mlp`'s teacher's prediction, weighted by
  `branch_lora.cross_weight`.

**`branch_mlp`**, per inner step, via `CMKD.forward`:

- `clf_loss`: source cross-entropy (label-smoothed) through the task head.
- `transfer_loss` = `task_loss + distill_loss + reg_loss`, all defined in
  `cmct/branch_mlp/loss.py`:
  - `task_loss` / `distill_loss`: gini-impurity terms over the target
    prediction and its self-reference, weighted by a sigmoid-ramped `lamb`
    schedule (`LambdaScheduler`).
  - `reg_loss`: always uses the LIVE cosine branch's prediction, never the
    self-reference -- see "reg_loss asymmetry" below.
- `loss_mlp_cross` (post-warmup only, when `branch_lora.enabled`):
  confidence-masked cross-entropy against `branch_lora`'s teacher's
  prediction, weighted by `branch_mlp.cross_weight`. This reuses the target
  logits the forward pass already returned rather than taking a second
  forward pass -- see "Cross-teaching reuses the forward's own logits" below.

## EMA teachers

Both branches maintain an EMA teacher, updated every step the branch trains:

- `branch_lora`: `ema_update_lora_params` averages only the `lora_*`
  parameters (LoRA A/B matrices) between student and teacher, at a fixed
  momentum (`branch_lora.ema_momentum`).
- `branch_mlp`: `ema_update_teacher` averages the WHOLE backbone
  (`model_mlp.teacher_model`) AND a separate classifier-head teacher
  (`teacher_classifier`), at a momentum controlled by `branch_mlp.ema`
  (schedule `dacs` or `hard_copy`; see "EMA-after-step" and "Shared
  backbone/head momentum" below).

## Evaluation protocol

`cmct/evaluate.py`'s `evaluate` function feeds ONE shared dassl test loader
to both teachers (`teacher_lora`, and `model_mlp.teacher_model` through
`teacher_classifier`), so their ensemble average is computed over exactly the
same images each batch -- not two separately-shuffled streams. It returns
`(acc_lora, acc_mlp, acc_ensemble)`, with the first and third as `None` when
`branch_lora` is disabled (there is no LoRA teacher and no ensemble to
report).

## Load-bearing details

These are the details most likely to be broken by a well-intentioned
refactor, with the reasoning behind each.

### `branch_mlp`'s backbone must be fp32

`cmct/branch_mlp/backbone.py`'s `ClipBackbone.__init__` calls `model.float()`
right after loading. The classifier head built on top of it
(`TransferNet.classifier_layer`) is fp32, but `cmct.clip`'s `build_model`
downcasts the backbone to fp16 by default. Skipping the `.float()` call
doesn't fail quietly -- it fails on the very first forward, with
`LayerNorm`'s "expected scalar type Half but found Float" -- but the failure
is far from the cause: it looks like a `branch_mlp` bug, when the actual
cause is a default two layers down in the vendored CLIP.

### Two independent DataManagers

`cmct/train.py` builds a SEPARATE dassl `DataManager` for each branch. This
is deliberate: with one shared manager, both branches would see exactly the
same shuffled order of source and target examples every step, which is not
what "two independently trained branches" should mean. Evaluation is the
exception -- it uses ONE shared test loader (see "Evaluation protocol"
above), because the ensemble needs both teachers scoring the same images.

### Loader construction order shifts branch_mlp's randomness, not LoRA's

The order in which `cmct/train.py`'s `main()` builds the two `DataManager`s
and the LoRA student/teacher pair is load-bearing, and NOT because of the
LoRA weights.

`CyclingLoader.__init__` calls `iter(loader)` eagerly, and `iter()` on a
`DataLoader` whose `RandomSampler` has `generator=None` draws its permutation
seed from the global torch RNG. So `cmct/train.py` interleaves: the LoRA
branch's `DataManager` -> wrap its two loaders -> build the LoRA student and
teacher -> `branch_mlp`'s `DataManager` -> wrap its two loaders. Collapsing
this into "build both managers, then the LoRA pair" would move two RNG draws
earlier, shifting everything that draws from the RNG afterward.

What that shift does NOT change is the LoRA weights: `LoRALayer.init_lora_param`
overwrites its `nn.init.normal_` draw with a deterministic SVD of the
pretrained weight, so every `lora_*` tensor comes out identical regardless of
RNG state. What it DOES change is `branch_mlp`'s sampler permutation stream
and `TransferNet.classifier_layer`'s `nn.init.normal_(std=0.001)` head
initialization. `tests/test_data.py` guards the actual construction order
written in `cmct/train.py` with a static AST check, plus two tests that
demonstrate the RNG divergence directly.

### EMA-after-step

`branch_mlp`'s EMA update (`ema_update_teacher`, called twice: once for the
backbone, once for the classifier head) runs AFTER `optim_mlp.step()`, using
the PRE-increment `mlp_step_global` (the counter is only incremented at the
bottom of the inner-step loop). Under the `dacs` momentum schedule, this
combination is what makes the very first EMA update (`mlp_step_global == 0`)
a hard copy of weights that have already taken one optimizer step -- not a
copy of the random initialization. Reversing either half of this (EMA before
`.step()`, or using the post-increment counter) changes what that first
update captures.

### Shared backbone/head momentum

The backbone EMA and the classifier-head EMA always use the SAME momentum
value each step (`ema_momentum_at`, computed once per inner step and passed
to both `ema_update_teacher` calls). The head is trained (inside
`TransferNet.forward`) against the LIVE backbone's features, not the lagging
EMA backbone's -- so pairing a smoothed backbone with an instantaneously-
tracking head (or vice versa) would leave the two teachers inconsistent with
each other.

### `model_mlp.train()` must never be called

`nn.Module.train()` recurses into every submodule, including
`model_mlp.teacher_model` -- which must stay permanently in eval mode (it is
an EMA-only, never-backpropped copy; see `ema_update_teacher`). `cmct/train.py`
sets `model_mlp.base_network.train()` and `model_mlp.classifier_layer.train()`
individually instead of calling `model_mlp.train()`.

### MK-MMD uses the weak view

`branch_lora`'s MK-MMD term (`mk_mmd(feat_x_lora, feat_u_lora_weak)`) always
uses a separate forward pass on the WEAK (unaugmented) target view, even when
`data.strong_aug` is on and the rest of that macro-step uses the strong view.
This keeps domain-shift measurement from being confounded with
augmentation-strength shift, which is a different thing MK-MMD is not meant
to measure.

### Cross-teaching reuses the forward's own logits

`branch_mlp`'s cross-teaching term (`loss_mlp_cross`) reuses `target_logits`,
already returned by `TransferNet.forward`'s own call, rather than taking a
second forward pass through the backbone and classifier head. There is no
separate `model_mlp.predict(...)` call for this.

### reg_loss asymmetry

`CMKD.forward`'s `reg_loss` term always reads `target_pred_clip` (derived
from the LIVE cosine branch's `target_logit_clip`), never
`self_ref_logit_clip`. `task_loss` and `distill_loss` are the only two terms
that read the self-reference. This is deliberate: it keeps `reg_loss`
training the live branch regardless of which reference the self-consistency
terms are configured to use (`branch_mlp.self_from_teacher`).

### Frozen CLIP lifetime

The frozen zero-shot CLIP teacher (`teacher_frozen`, built only when
`branch_lora.enabled`) is read only inside `if in_warmup_lora:` blocks.
`cmct/train.py` frees it (`del teacher_frozen, clip_frozen` plus
`torch.cuda.empty_cache()`) the moment `branch_lora`'s warmup ends, rather
than holding an entire extra CLIP model on the GPU for the rest of the run.

### branch_lora.lr never drives a step

`branch_lora.lr` is passed to the LoRA optimizer's constructor, but nothing
downstream ever reads it again. Every macro-step during warmup writes
`pg["lr"] = branch_lora.warmup.lr` directly onto every optimizer param group.
`CosineAnnealingLR`, built once after the optimizer, is chainable: it scales
whatever LR is currently on the param group rather than recomputing from
`base_lrs`. So the effective LR schedule is `warmup.lr` held constant through
warmup, then annealed toward 0 from whatever `warmup.lr` was -- the `lr` key
set at construction time is overwritten before it ever takes effect.

### The prompt/classname assertion runs against the live dataset

`branch_mlp`'s class prompts (`cmct/branch_mlp/backbone.py`'s `PROMPTS`) are a
hardcoded literal, and nothing in the code makes that list agree with the
label order dassl assigns from the dataset's class directories. If they ever
diverged, `branch_mlp`'s cosine logits would be silently mislabelled. This
can't be caught by a synthetic unit test, because dassl sorts class
directory names with a case-sensitive ASCII sort (`TV` before `Table`) and a
synthetic test tree cannot reproduce the real dataset's cased directory
names. So `cmct/train.py` asserts prompt/classname alignment directly against
the live dataset at startup, before any GPU work, and fails with a specific
mismatch index rather than letting it surface later as a shape error deep
inside `CMKD.forward`.

## Extending this: swapping the CMKD loss

If you are replacing `CMKD` with a different self-training loss:

- `TransferNet.forward` (`cmct/branch_mlp/model.py`) is the integration
  point. It expects a loss module with a `forward(target_logit,
  target_logit_clip, source_logit_clip, source_label,
  self_ref_logit_clip=None)` signature returning a scalar.
- `branch_mlp`'s name does not need to change -- it names the head, not the
  loss.
- Use `scripts/compare_traces.py` to confirm the swap did not accidentally
  change other numbers: run the same config before and after with
  `--trace-out`, and diff the two traces. Any drift outside the loss values
  you intentionally changed indicates the swap touched something it
  shouldn't have.
