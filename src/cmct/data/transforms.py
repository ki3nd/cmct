"""Transform pipelines, built from a DatasetSpec.

The order is fixed: Resize -> crop -> hflip -> ToTensor -> Normalize. Which
crop, and at what size, comes from the dataset's own TransformSpec, so a dataset
whose pipeline differs (VisDA) needs no branch in this code.
"""
from __future__ import annotations

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from torchvision.transforms.functional import InterpolationMode

from cmct.config.schema import CropSpec, DatasetSpec

_INTERPOLATION = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def _build(spec: CropSpec, dataset: DatasetSpec) -> Compose:
    steps: list = [
        Resize(list(spec.resize), interpolation=_INTERPOLATION[dataset.transform.interpolation])
    ]
    if spec.crop == "random":
        steps.append(RandomCrop(spec.crop_size))
    elif spec.crop == "center":
        steps.append(CenterCrop(spec.crop_size))
    if spec.hflip:
        steps.append(RandomHorizontalFlip())
    steps.append(ToTensor())
    steps.append(Normalize(mean=list(dataset.pixel_mean), std=list(dataset.pixel_std)))
    return Compose(steps)


def build_train(dataset: DatasetSpec) -> Compose:
    return _build(dataset.transform.train, dataset)


def build_test(dataset: DatasetSpec) -> Compose:
    return _build(dataset.transform.test, dataset)
