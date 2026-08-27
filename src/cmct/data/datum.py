"""The two plain records the data layer speaks in."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Datum:
    """One image on disk."""

    path: str
    label: int
    domain: int
    classname: str


@dataclass
class DomainSplit:
    """A dataset resolved into the streams a UDA run needs.

    `test` is conventionally the same image set as `train_u` (transductive UDA:
    the target domain is evaluated on exactly the unlabeled data it adapted to).
    They stay separate lists so a dataset with an official held-out target split
    can say so without a special case.
    """

    train_x: list[Datum]
    """Labeled source images."""
    train_u: list[Datum]
    """Unlabeled target images. Labels are present in the records but read only
    by the evaluator, never during training."""
    test: list[Datum]
    classnames: list[str]
    """Indexed by label. The order defines the label mapping and must be stable
    across runs, since text-prompt embeddings are built in this order."""

    @property
    def num_classes(self) -> int:
        return len(self.classnames)
