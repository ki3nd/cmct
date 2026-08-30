"""DataLoader construction and the infinite-stream wrapper.

Two properties this module exists to guarantee:

1. Each branch gets its own independently shuffled streams over the same
   underlying DomainSplit. Two branches consuming one iterator would see
   correlated batches, which quietly couples their gradients and removes the
   reason for having two of them. Independence comes from a per-stream
   torch.Generator seed, not from construction order, so a branch reproduces
   regardless of how many branches exist or in what order they are built.
2. A stream is torn down only on genuine exhaustion -- never per "epoch", never
   at an eval boundary. Rebuilding an iterator resets the shuffle and re-warms
   workers, which shows up later as an unexplained periodic loss artifact.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from cmct.config.schema import BranchConfig, DataConfig, DatasetSpec
from cmct.data.batch import Batch
from cmct.data.dataset import ImageListDataset
from cmct.data.datum import Datum, DomainSplit
from cmct.data.transforms import build_test, build_train

_ROLE_OFFSET = {"source": 0, "target": 1}


def stream_seed(base_seed: int, branch_name: str, role: str) -> int:
    """Deterministic per-stream seed.

    Derived from the run seed, the branch name and the stream's role, so the
    four training streams of a two-branch run all shuffle differently, every one
    of them reproduces on its own, and adding a branch leaves the others'
    streams untouched. Hashed with sha256 rather than hash() because hash() on a
    str is salted per process.
    """
    if role not in _ROLE_OFFSET:
        raise ValueError(f"role must be 'source' or 'target', got {role!r}")
    digest = hashlib.sha256(branch_name.encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "big") + _ROLE_OFFSET[role]) % (2**31)


def _build_loader(items: list[Datum], transforms: Any, batch_size: int, num_workers: int,
                  train: bool, seed: int | None) -> DataLoader:
    if not items:
        raise ValueError("empty loader: no images to iterate over")
    dataset = ImageListDataset(items, transforms)
    generator = None
    if train:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler: Any = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        # The SAME generator the sampler uses. Without it a rerun reshuffles
        # identically but AUGMENTS differently: DataLoader derives each worker's
        # base seed from `generator` when one is given and from the global RNG
        # when it is not, and the random crop and flip run inside the worker. So
        # `seed` used to fix the order of the images and nothing about the
        # images themselves.
        generator=generator if train else None,
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

    Owns both loaders so "branch B's stream" is a single object with a single
    derived seed, not two loaders a caller has to keep paired.
    """

    def __init__(self, split: DomainSplit, dataset: DatasetSpec,
                 branch: BranchConfig, base_seed: int) -> None:
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
            seed=stream_seed(base_seed, branch.name, "source"),
        ))
        self.target = InfiniteStream(_build_loader(
            split.train_u, transforms, branch.stream.batch_size_u,
            branch.stream.num_workers, train=True,
            seed=stream_seed(base_seed, branch.name, "target"),
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
        data.num_workers_test, train=False, seed=None,
    )
