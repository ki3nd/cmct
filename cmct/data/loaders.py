"""DataLoader construction and the infinite-stream wrapper.

Two properties this module exists to guarantee:

1. A stream's shuffle comes from the shared global torch RNG, not a per-stream
   generator -- the same mechanism dassl's own RandomSampler(data_source) uses
   (no generator= passed), matching train_mfa_v2.py's actual behaviour: every
   loader's shuffle, and every worker's augmentation seed, is drawn from ONE
   continuous stream seeded once via train_lora.set_seed / dassl's
   set_random_seed. Two branches therefore still shuffle differently (they
   draw from different points in that stream), but WHICH values a branch gets
   depends on how many draws happened before it was built -- exactly like the
   reference, and unlike a design keyed to a per-branch seed. This trades away
   order-independence on purpose, to close the gap the reference's real
   behaviour left open (see the git history for the per-stream sha256-seeded
   generator this replaces, which was independent of build order and of the
   reference alike).
2. A stream is torn down only on genuine exhaustion -- never per "epoch", never
   at an eval boundary. Rebuilding an iterator resets the shuffle and re-warms
   workers, which shows up later as an unexplained periodic loss artifact.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from cmct.config.schema import BranchConfig, DataConfig, DatasetSpec
from cmct.data.batch import Batch
from cmct.data.dataset import ImageListDataset
from cmct.data.datum import Datum, DomainSplit
from cmct.data.transforms import build_test, build_train


def _build_loader(items: list[Datum], transforms: Any, batch_size: int, num_workers: int,
                  train: bool) -> DataLoader:
    if not items:
        raise ValueError("empty loader: no images to iterate over")
    dataset = ImageListDataset(items, transforms)
    sampler: Any = RandomSampler(dataset) if train else SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        # No explicit generator, on either the sampler or the loader: both then
        # draw from the shared global RNG, matching dassl's own
        # build_data_loader (dassl/data/data_manager.py), which passes neither.
        # Train drops a short final batch; eval keeps every image.
        drop_last=train and len(items) >= batch_size,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


class InfiniteStream:
    """One persistent iterator over a DataLoader, re-created only when it is
    genuinely exhausted. next() never raises StopIteration."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._iterator = iter(loader)
        self.epochs = 0
        """How many times the underlying loader has been exhausted."""

    def next(self) -> dict[str, Any]:
        try:
            return next(self._iterator)
        except StopIteration:
            self.epochs += 1
            self._iterator = iter(self.loader)
            return next(self._iterator)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            yield self.next()


class BatchSource:
    """One branch's source+target stream pair, handing out Batch objects.

    Owns both loaders so "branch B's stream" is a single object, not two
    loaders a caller has to keep paired. Its shuffle comes from the global RNG
    at the point it is constructed (see this module's docstring) -- there is no
    per-branch seed to pass in.
    """

    def __init__(self, split: DomainSplit, dataset: DatasetSpec,
                 branch: BranchConfig) -> None:
        if branch.stream.strong_aug:
            raise NotImplementedError(
                f"branches[{branch.name}].stream.strong_aug: strong augmentation is not "
                f"implemented yet; set it back to false"
            )
        transforms = build_train(dataset)
        self.name = branch.name
        self.source = InfiniteStream(_build_loader(
            split.train_x, transforms, branch.stream.batch_size_x,
            branch.stream.num_workers, train=True,
        ))
        self.target = InfiniteStream(_build_loader(
            split.train_u, transforms, branch.stream.batch_size_u,
            branch.stream.num_workers, train=True,
        ))

    def next(self) -> Batch:
        source = self.source.next()
        target = self.target.next()
        return Batch(
            img_x=source["img"],
            label_x=source["label"],
            img_u=target["img"],
            img_u_strong=target.get("img2"),
        )


def build_test_loader(split: DomainSplit, dataset: DatasetSpec,
                      data: DataConfig) -> DataLoader:
    """ONE shared, deterministic test loader for every branch.

    Shared on purpose: teachers can only be compared to each other, and
    ensembled, if they were scored on the same image in the same order.
    """
    return _build_loader(
        split.test, build_test(dataset), data.batch_size_test,
        data.num_workers_test, train=False,
    )
