# Frozen test fixtures

Every file in this directory is a **frozen numeric baseline**, checked into
the repository deliberately. Each one captures the exact expected output of
some piece of the training code for a fixed set of inputs.

**Do not regenerate these files from the code under test.** Recomputing a
fixture by running the current implementation and saving its output would
make the corresponding test tautological -- it would only ever confirm that
the code agrees with itself, not that it produces the right numbers. If a
test using one of these fixtures starts failing, that is a signal to
investigate the code, not to refresh the fixture.

## What each file guards

- `cmkd_loss.json` -- `CMKD.forward` in `cmct/branch_mlp/loss.py`: single-call
  values and multi-step sequences for both the live-cosine-branch and
  teacher self-reference arms, used by `tests/test_branch_mlp.py`.
- `ema_schedule.json` -- `ema_momentum_at` in `cmct/train.py`: the "dacs" and
  "hard_copy" EMA momentum schedules, used by `tests/test_train_schedules.py`.
- `lambda_schedule.json` -- `LambdaScheduler.lamb` in
  `cmct/branch_mlp/loss.py`: the sigmoid ramp used to weight the CMKD task
  and distillation terms, used by `tests/test_branch_mlp.py`.
- `masked_ce.json` -- `masked_cross_entropy` in `cmct/losses.py`: confidence-
  masked cross-entropy against a pseudo-label, used by `tests/test_losses.py`.
- `mk_mmd.json` -- `mk_mmd` in `cmct/losses.py`: multi-kernel MMD between a
  source and target feature batch, used by `tests/test_losses.py`.
- `prompts.json` -- the per-dataset class-prompt literals in
  `cmct/branch_mlp/backbone.py`, used by `tests/test_branch_mlp.py` to catch
  any drift between the hardcoded prompt order and the dataset's class order.
- `rank_ramp.json` -- `compute_rank` in `cmct/branch_lora/lora/apply.py`: the
  depth-dependent LoRA rank ramp, used by `tests/test_branch_lora.py`.
