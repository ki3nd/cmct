"""Compare two per-iteration loss traces produced by `cmct.train --trace-out`.

This is how you check that a change you did not intend to be numeric was not
numeric: run the same config before and after, and diff the traces. Useful
when swapping out a piece of the training loop -- for example the CMKD loss
in `cmct/branch_mlp/loss.py` -- for something else that should behave
identically at the surrounding call sites.

Usage: python scripts/compare_traces.py traces/before.json traces/after.json
Exits non-zero on the first field that drifts beyond tolerance.
"""
import argparse
import json
import sys

LOSS_KEYS = ["loss_lora", "loss_source", "loss_self", "loss_cross", "loss_mmd",
             "loss_mlp", "clf_loss", "transfer_loss", "loss_mlp_cross"]



def _read_json(path):
    with open(path) as fh:
        return json.load(fh)

def close(a, b, rtol, atol):
    return abs(a - b) <= atol + rtol * abs(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    ref = _read_json(args.reference)
    cand = _read_json(args.candidate)

    if len(ref["iters"]) != len(cand["iters"]):
        sys.exit(f"iteration count differs: {len(ref['iters'])} vs {len(cand['iters'])}")
    if len(ref["evals"]) != len(cand["evals"]):
        sys.exit(f"eval count differs: {len(ref['evals'])} vs {len(cand['evals'])}")

    failures = []
    for r, c in zip(ref["iters"], cand["iters"]):
        for k in LOSS_KEYS:
            if not close(c[k], r[k], args.rtol, args.atol):
                failures.append(f"macro {r['macro']} {k}: reference {r[k]!r} candidate {c[k]!r}")
    for r, c in zip(ref["evals"], cand["evals"]):
        for k in ("acc_lora", "acc_mlp", "acc_ensemble"):
            if not close(c[k], r[k], args.rtol, args.atol):
                failures.append(f"eval @{r['macro']} {k}: reference {r[k]!r} candidate {c[k]!r}")

    if failures:
        print(f"{len(failures)} mismatch(es); first 20:")
        for line in failures[:20]:
            print("  " + line)
        sys.exit(1)
    print(f"traces match over {len(ref['iters'])} iterations and {len(ref['evals'])} evals")


if __name__ == "__main__":
    main()
