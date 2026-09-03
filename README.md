# cmct

Cross-model co-training for unsupervised domain adaptation on Office-Home,
driven by a single YAML config.

## Method

Two branches are co-trained on an unlabelled target domain, each supplying
the other with pseudo-labels through its EMA teacher:

- **`branch_lora`** -- CLIP ViT-B/16 with depth-ramped LoRA, trained with a
  source cross-entropy loss, an MK-MMD domain-alignment term, and
  confidence-masked pseudo-labels.
- **`branch_mlp`** -- a learned head on CLIP features, trained with the CMKD
  self-training loss.

Each branch's EMA teacher supplies pseudo-labels used to train the other
branch. Evaluation reports both teachers' accuracy and their ensemble.

See `docs/design.md` for the architecture in more detail.

## Running it

```bash
python -m cmct.train --config configs/officehome_a2c.yaml
```

Optional flags dump a per-iteration loss trace, useful for checking that a
change was not accidentally numeric (see `scripts/compare_traces.py`):

```bash
python -m cmct.train --config configs/officehome_a2c.yaml \
  --trace-out traces/run.json --trace-iters 60
```

## Dataset layout

Images must live under `<data.root>/office_home/<domain>/<Class_Name>/*.jpg`,
one directory per domain: `art`, `clipart`, `product`, `real_world`.

Class directories are discovered by listing and sorting them with a
case-sensitive ASCII sort. `branch_mlp`'s class-prompt list is hardcoded and
must line up with that exact order -- for example `TV` sorts before `Table`,
since uppercase letters precede lowercase in ASCII.

Two things guard this, and it is worth knowing exactly what each covers.
`tests/test_branch_mlp.py` checks the hardcoded prompt list against a frozen
baseline fixture, and `tests/test_dassl_dataset.py` checks that class
discovery really does sort case-sensitively. Neither compares the prompt list
against the dataset actually on disk -- a synthetic test tree cannot
reconstruct the real cased directory names. That comparison happens at run
time instead: `cmct/train.py` asserts, before any GPU work, that the prompt
list is the same length as the dataset's class list and lines up with it
element by element. A dataset whose classes change, or that is only partly
downloaded, fails there with a clear message rather than silently mislabelling
`branch_mlp`'s cosine head.

## Backbone weights

Each branch names its own backbone, and they are independent:

- `branch_lora.backbone.name` must be a ViT (`ViT-B/16`, `ViT-B/32`,
  `ViT-L/14`). LoRA is injected into the vision transformer's attention blocks
  and the injection table covers ViT only, so a ResNet here is rejected at
  parse time.
- `branch_mlp.backbone.name` accepts any CLIP backbone (`ViT-B/16`, `RN50`,
  `RN101`). That branch reads features through `encode_image` alone, and the
  classifier head's width is derived from the backbone rather than tabulated,
  so nothing else needs changing: 512 for ViT-B/16 and RN101, 1024 for RN50.

Choosing a ResNet for `branch_mlp` wakes a line that is inert on a ViT:
`fix_bn` puts every BatchNorm in the backbone into eval at the start of each
forward, so the backbone keeps its pretrained running statistics and never
adapts to the target domain. CLIP's RN50 visual tower has 55 BatchNorm modules;
ViT-B/16 has none. This is deliberate -- freezing BN is standard when
fine-tuning a pretrained backbone on a small, shifted target set -- and it
belongs to the fine-tuning setup rather than to the CMKD loss, so it should
survive a loss swap. Note also that `branch_mlp.lr` was tuned for ViT
fine-tuning; a ResNet wants its own value, so a first ResNet run being worse
does not by itself say the backbone is worse.

The two branches fetch CLIP ViT-B/16 independently, to two different places,
and a fresh run downloads it twice:

- `branch_lora` goes through `load_clip_to_cpu`, which calls
  `clip._download(url, backbone.path)`. With the shipped config that is
  `./assets`, so that directory must be writable.
- `branch_mlp` uses stock `clip.load(...)`, which downloads to
  `~/.cache/clip`.

Both copies are the same weights; nothing unifies the two locations. Budget
for it on a constrained machine: it is roughly 700 MB of download and disk
across the two locations.

## Settings that are not what they look like

- **The two branches run at different precisions, and neither is a choice.**
  `branch_lora.precision` (`fp16` or `fp32`) governs the LoRA branch only.
  `branch_mlp`'s backbone is always `fp32` and is not configurable: CLIP's
  `build_model` downcasts to fp16, and `branch_mlp/backbone.py` explicitly
  restores fp32 afterward. Forcing that backbone to stay fp16 crashes
  immediately -- its classifier head is fp32, so `LayerNorm` raises
  "expected scalar type Half but found Float" on the first forward.
- **`branch_lora.lr` looks like it sets the LoRA optimizer's learning rate,
  but it never actually drives a step.** Every macro-step during warmup
  writes `pg["lr"] = warmup.lr` directly onto every optimizer param group,
  and the `CosineAnnealingLR` scheduler built afterward is chainable -- it
  scales whatever LR is already on the param group rather than recomputing
  from `base_lrs`. So the effective schedule is `warmup.lr` held constant
  through warmup, then annealed to 0 from there; the `lr` key set at
  construction is overwritten before it ever takes a step.
- The LoRA rank is **not** uniform across blocks. `lora.rank_ramp: [2,4,6,8,10]`
  with `lora.r: 2` yields per-block ranks
  `[2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64]`.
- `pseudo_label.threshold` (`0.85`) is shared by both branches; it is not two
  independent thresholds that happen to match.
- `data.batch_size.target` is **inert**. dassl's `DATALOADER.TRAIN_U.SAME_AS_X`
  defaults to True, so the target loader always copies the source batch size.
  The config rejects a target value that differs from the source, rather than
  silently accepting a value it would go on to ignore.
- `INPUT.INTERPOLATION` and `INPUT.TRANSFORMS` in the dassl config produced by
  `to_dassl_cfg` are inert: the transforms that actually run are built
  directly in `cmct/data/transforms.py` and passed to `DataManager` as
  `custom_tfm_train`/`custom_tfm_test`, so `build_transform` (and its
  `INPUT.INTERPOLATION` reader) is never called. The transforms that run are
  BILINEAR.

## Load-bearing details

Some things in `cmct/train.py` are easy to "clean up" in a way that silently
changes results:

1. **`branch_mlp`'s backbone must stay fp32.** Its classifier head is fp32
   (`cmct/branch_mlp/backbone.py`); CLIP's own `build_model` downcasts to
   fp16, so the backbone widens itself back to fp32 explicitly with
   `model.float()` right after loading.
2. **Loader construction order.** `build_lora_pair` (the LoRA student/teacher
   pair) must be constructed *between* the two `DataManager`s' `CyclingLoader`
   wrapping calls, not before both or after both. `CyclingLoader.__init__`
   calls `iter()` eagerly, which draws from the global torch RNG. Moving both
   `DataManager`s' loader-wrapping together into one place shifts every
   subsequent draw from that RNG. This does NOT change the LoRA weights --
   `LoRALayer.init_lora_param` overwrites its random draw with a deterministic
   SVD of the pretrained weight, so those are identical regardless of RNG
   state. What it does shift is `branch_mlp`'s sampler permutation stream and
   its classifier head's `nn.init.normal_(std=0.001)` initialization.
   `tests/test_data.py` guards this ordering.
3. **EMA-after-step.** `branch_mlp`'s teacher EMA update runs *after*
   `optimizer.step()`, and with the pre-increment step index. Combined with
   the DACS EMA schedule, this is what makes the very first EMA update a hard
   copy of already-stepped (not random-init) student weights.
4. **`model_mlp.train()` must never be called.** `nn.Module.train()` recurses
   into every submodule, including the EMA teacher held inside `TransferNet`,
   which must stay permanently in eval mode. `base_network` and
   `classifier_layer` are switched to train mode individually instead.

## GPU requirement

`branch_mlp` calls `.cuda()` directly and therefore requires a GPU; it
ignores any `gpu` config value other than `0`. This is deliberate, not an
oversight.

## Checkpoints

Saved under `<output_dir>/`:

- `teacher_lora/LoRA-last.pt`, `teacher_lora/LoRA-best.pt`
- `model_mlp-last.pt`, `model_mlp-best.pt`

## Tests

```bash
python -m pytest tests -v
```

`tests/fixtures/` holds frozen numeric baselines used by several of these
tests -- see `tests/fixtures/README.md` for what each one guards. They must
not be regenerated from the code under test.

## Development

```bash
python -m pip install -e ".[dev]"
```

## Vendored / third-party code

`vendor/dassl` (trimmed), `cmct/clip`, and `cmct/branch_mlp` are vendored
third-party code; `cmct/branch_mlp` carries its own license as `NOTICE`.
