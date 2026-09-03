"""Image transforms used by the training and evaluation data pipelines."""

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
