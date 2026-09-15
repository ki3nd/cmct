"""Image transforms used by the training and evaluation data pipelines."""

import torch
from torchvision.transforms import (
    ColorJitter,
    Compose,
    InterpolationMode,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


def build_transforms(image_size, pixel_mean, pixel_std, strong_aug: bool):
    """Train: Resize(256, 256) -> RandomCrop -> flip.
    Test: a direct Resize to the crop size, with NO CenterCrop.

    With strong_aug, the train transform becomes a list; dassl's DatasetWrapper
    turns a list into output["img"], output["img2"], ... The strong view is a
    harder crop plus light colour jitter -- deliberately mild, since aggressive
    augmentation (heavy color jitter plus RandAugment) is enough to push a
    CLIP-pretrained backbone off-distribution.
    """
    normalize = Normalize(mean=pixel_mean, std=pixel_std)
    crop_size = image_size

    tfm_train = Compose([
        Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
        RandomCrop(crop_size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    tfm_test = Compose([
        Resize([crop_size, crop_size], interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        normalize,
    ])

    if strong_aug:
        tfm_strong = Compose([
            RandomResizedCrop(crop_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            ToTensor(),
            normalize,
        ])
        tfm_train_list = [tfm_train, tfm_strong]
    else:
        tfm_train_list = tfm_train

    return tfm_train_list, tfm_test


def make_renormalizer(src_mean, src_std, dst_mean, dst_std, device):
    """A function converting an ALREADY-normalized batch from one
    normalization to another, exactly.

    This exists because the two branches can normalize differently while still
    having to cross-teach on THE SAME image. A second data loader cannot
    deliver that: the target batches are randomly cropped and flipped, so a
    second loader hands the teacher a different view of the image, not the same
    one differently normalized. Undoing one normalization and applying the
    other on the tensor itself is the only version that keeps the view fixed --
    and it is a single affine op, so it costs nothing next to a forward pass.

    Returns the identity when the two normalizations already agree, which is
    every run where both branches share a backbone source.
    """
    if list(src_mean) == list(dst_mean) and list(src_std) == list(dst_std):
        return lambda images: images

    def as_tensor(values):
        return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    # x_raw = x * src_std + src_mean, then (x_raw - dst_mean) / dst_std, folded
    # into one multiply-add so nothing is allocated per call beyond the result.
    scale = as_tensor(src_std) / as_tensor(dst_std)
    shift = (as_tensor(src_mean) - as_tensor(dst_mean)) / as_tensor(dst_std)
    return lambda images: images * scale + shift
