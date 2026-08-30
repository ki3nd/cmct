"""Scoring several models in one pass, and combining them."""
from __future__ import annotations

import pytest
import torch

from cmct.config.schema import CoTrainConfig
from cmct.engine import evaluate, evaluate_ensemble


class OneShotLoader:
    """Raises if iterated twice, so a second pass over the data is a failure
    rather than a silent doubling of cost."""

    def __init__(self, batches):
        self.batches = batches
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        if self.passes > 1:
            raise AssertionError("the loader was iterated more than once")
        return iter(self.batches)


def constant(logits):
    return lambda images: logits.repeat(images.shape[0], 1)


# Two models chosen so the two rules give OPPOSITE answers on label 0.
# A is confident about class 0 but with small logits. B is split almost evenly
# between classes 1 and 2 in probability -- it barely prefers 1 -- yet its logits
# are large. Averaging probabilities lets A's near-certainty win; averaging
# logits lets B's magnitude win. This is the branch-1-vs-branch-2 scale
# mismatch in miniature.
A = torch.tensor([[8.0, 0.0, 0.0]])
B = torch.tensor([[0.0, 10.0, 9.9]])


@pytest.fixture
def batches():
    return [{"img": torch.zeros(4, 3, 8, 8), "label": torch.zeros(4, dtype=torch.long)}]


def test_the_two_modes_are_not_two_spellings_of_one_thing(batches):
    """Branch 1's logits carry logit_scale around 100 while branch 2's come from
    a linear head. If these two modes agreed here, the choice would be untested
    and the scale difference invisible."""
    fns = {"a": constant(A), "b": constant(B)}
    by_prob = evaluate_ensemble(fns, batches, "cpu", "mean_prob")["ensemble"]
    by_logit = evaluate_ensemble(fns, batches, "cpu", "mean_logit")["ensemble"]
    assert by_prob.accuracy == 100.0, "mean_prob should follow A's near-certainty"
    assert by_logit.accuracy == 0.0, "mean_logit should be dominated by B's magnitude"


def test_mean_prob_is_the_argmax_of_the_averaged_probabilities(batches):
    fns = {"a": constant(A), "b": constant(B)}
    expected = (A.softmax(-1) + B.softmax(-1)).div(2).argmax(-1).item()
    result = evaluate_ensemble(fns, batches, "cpu", "mean_prob")["ensemble"]
    assert result.accuracy == (100.0 if expected == 0 else 0.0)


def test_off_produces_no_ensemble_entry_at_all(batches):
    """Absent, not present-and-empty: a caller cannot report a number that was
    never computed."""
    result = evaluate_ensemble({"a": constant(A), "b": constant(B)}, batches, "cpu", "off")
    assert set(result) == {"a", "b"}


def test_off_is_the_default_everywhere():
    assert CoTrainConfig(cross_ref_refresh="macro").ensemble == "off"
    result = evaluate_ensemble({"a": constant(A)},
                               [{"img": torch.zeros(2, 3, 8, 8),
                                 "label": torch.zeros(2, dtype=torch.long)}], "cpu")
    assert "ensemble" not in result


def test_one_pass_over_the_loader(batches):
    loader = OneShotLoader(batches)
    evaluate_ensemble({"a": constant(A), "b": constant(B)}, loader, "cpu", "mean_prob")
    assert loader.passes == 1


def test_per_model_numbers_match_evaluate_run_on_its_own(batches):
    """The per-branch numbers must not change because an ensemble was also
    computed."""
    alone = evaluate(constant(A), batches, "cpu")
    together = evaluate_ensemble({"a": constant(A), "b": constant(B)},
                                 batches, "cpu", "mean_prob")["a"]
    assert (together.accuracy, together.correct, together.total) == \
        (alone.accuracy, alone.correct, alone.total)
    assert together.loss == pytest.approx(alone.loss)


def test_an_unknown_mode_raises(batches):
    with pytest.raises(ValueError, match="cotrain.ensemble"):
        evaluate_ensemble({"a": constant(A)}, batches, "cpu", "mean")


def test_a_branch_named_ensemble_raises(batches):
    with pytest.raises(ValueError, match="reserved"):
        evaluate_ensemble({"ensemble": constant(A)}, batches, "cpu", "mean_prob")
