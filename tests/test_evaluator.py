"""Accuracy and loss over a loader."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cmct.engine import evaluate

CLASSES = 4


class Fixed(Dataset):
    """Images are index placeholders; the logits function ignores them."""

    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"img": torch.full((3, 2, 2), float(index)), "label": self.labels[index]}


def loader(labels, batch_size=3):
    return DataLoader(Fixed(labels), batch_size=batch_size)


def test_all_correct():
    labels = [0, 1, 2, 3, 0, 1, 2]
    result = evaluate(lambda x: F.one_hot(
        torch.tensor([labels[int(v[0, 0, 0])] for v in x]), CLASSES).float() * 20,
        loader(labels), "cpu")
    assert result.accuracy == 100.0
    assert result.correct == result.total == len(labels)
    assert result.loss < 1e-6


def test_all_wrong():
    labels = [0, 0, 0, 0, 0]
    result = evaluate(lambda x: F.one_hot(torch.full((x.shape[0],), 1), CLASSES).float() * 20,
                      loader(labels), "cpu")
    assert result.accuracy == 0.0
    assert result.correct == 0


def test_partial_and_short_final_batch():
    labels = [0, 0, 0, 0, 1, 1, 1]        # 7 samples, batch 3 -> 3 + 3 + 1
    result = evaluate(lambda x: F.one_hot(torch.zeros(x.shape[0], dtype=torch.long),
                                          CLASSES).float() * 20,
                      loader(labels, batch_size=3), "cpu")
    assert result.total == 7
    assert result.correct == 4
    assert result.accuracy == pytest.approx(100 * 4 / 7)


def test_loss_matches_mean_cross_entropy():
    labels = [0, 1, 2, 3]
    torch.manual_seed(0)
    logits = torch.randn(4, CLASSES)
    result = evaluate(lambda x: logits[int(x[0, 0, 0, 0]):int(x[0, 0, 0, 0]) + x.shape[0]],
                      loader(labels, batch_size=4), "cpu")
    expected = F.cross_entropy(logits, torch.tensor(labels))
    assert result.loss == pytest.approx(float(expected), rel=1e-6)


def test_two_functions_on_one_loader_give_two_results():
    """evaluate takes a function, so the same loader scores the student and the
    teacher without evaluate knowing anything about either."""
    labels = [0, 1, 2, 3]
    good = evaluate(lambda x: F.one_hot(
        torch.tensor([labels[int(v[0, 0, 0])] for v in x]), CLASSES).float() * 20,
        loader(labels), "cpu")
    bad = evaluate(lambda x: F.one_hot(torch.zeros(x.shape[0], dtype=torch.long),
                                       CLASSES).float() * 20, loader(labels), "cpu")
    assert good.accuracy == 100.0
    assert bad.accuracy == 25.0


def test_empty_loader_raises():
    with pytest.raises(ValueError, match="produced no samples"):
        evaluate(lambda x: x, loader([]), "cpu")
