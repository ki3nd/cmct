import torch

from cmct.data import make_renormalizer

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalize(raw, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return (raw - mean) / std


def test_renormalizing_matches_normalizing_the_raw_pixels_directly():
    # The contract: converting a CLIP-normalized batch must land exactly where
    # normalizing the SAME raw pixels with ImageNet statistics would -- that is
    # what lets a teacher read the other branch's batch at all.
    raw = torch.rand(4, 3, 8, 8)
    convert = make_renormalizer(CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                                torch.device("cpu"))
    got = convert(_normalize(raw, CLIP_MEAN, CLIP_STD))
    assert torch.allclose(got, _normalize(raw, IMAGENET_MEAN, IMAGENET_STD), atol=1e-5)


def test_matching_normalizations_give_back_the_same_tensor():
    images = torch.rand(2, 3, 4, 4)
    convert = make_renormalizer(CLIP_MEAN, CLIP_STD, CLIP_MEAN, CLIP_STD, torch.device("cpu"))
    assert convert(images) is images
