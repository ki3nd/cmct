import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    InterpolationMode,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from cmct.data import CyclingLoader
from cmct.data.transforms import build_transforms

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def test_cycling_loader_wraps_around_without_rebuilding_early():
    loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
    cycling = CyclingLoader(loader)
    seen = [cycling.next()[0].tolist() for _ in range(4)]
    assert seen == [[0, 1], [2, 3], [0, 1], [2, 3]]


def test_train_transform_is_a_single_callable_without_strong_aug():
    train, test = build_transforms(224, MEAN, STD, strong_aug=False)
    assert not isinstance(train, list)
    assert test(_img()).shape == (3, 224, 224)
    assert train(_img()).shape == (3, 224, 224)


def test_train_transform_is_a_two_view_list_with_strong_aug():
    train, _ = build_transforms(224, MEAN, STD, strong_aug=True)
    assert isinstance(train, list) and len(train) == 2


def test_train_transform_composition_matches_the_frozen_baseline():
    # Resize(256, 256, BILINEAR) -> RandomCrop(crop) -> RandomHorizontalFlip
    # -> ToTensor -> Normalize. No CenterCrop anywhere in this chain.
    train, _ = build_transforms(224, MEAN, STD, strong_aug=False)
    steps = train.transforms
    types = [type(t) for t in steps]
    assert types == [Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize]
    assert not any(isinstance(t, CenterCrop) for t in steps)

    resize, crop, _flip, _to_tensor, normalize = steps
    assert list(resize.size) == [256, 256]
    assert resize.interpolation == InterpolationMode.BILINEAR
    assert tuple(crop.size) == (224, 224)
    assert list(normalize.mean) == MEAN
    assert list(normalize.std) == STD


def test_test_transform_composition_matches_the_frozen_baseline():
    # A DIRECT Resize to the crop size, BILINEAR, no CenterCrop -> ToTensor
    # -> Normalize.
    _, test = build_transforms(224, MEAN, STD, strong_aug=False)
    steps = test.transforms
    types = [type(t) for t in steps]
    assert types == [Resize, ToTensor, Normalize]
    assert not any(isinstance(t, CenterCrop) for t in steps)

    resize, _to_tensor, normalize = steps
    assert list(resize.size) == [224, 224]
    assert resize.interpolation == InterpolationMode.BILINEAR
    assert list(normalize.mean) == MEAN
    assert list(normalize.std) == STD


def test_strong_transform_composition_matches_the_frozen_baseline():
    # RandomResizedCrop(crop, scale=(0.5, 1.0), BILINEAR) ->
    # RandomHorizontalFlip -> ColorJitter(brightness=0.2, contrast=0.2,
    # saturation=0.2, hue=0.0) -> ToTensor -> Normalize.
    train_list, _ = build_transforms(224, MEAN, STD, strong_aug=True)
    weak, strong = train_list
    types = [type(t) for t in strong.transforms]
    assert types == [RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize]

    rrc, _flip, jitter, _to_tensor, normalize = strong.transforms
    assert tuple(rrc.size) == (224, 224)
    assert tuple(rrc.scale) == (0.5, 1.0)
    assert rrc.interpolation == InterpolationMode.BILINEAR
    # hue=0.0 means "no hue jitter", which torchvision stores as None.
    assert jitter.brightness == (0.8, 1.2)
    assert jitter.contrast == (0.8, 1.2)
    assert jitter.saturation == (0.8, 1.2)
    assert jitter.hue is None
    assert list(normalize.mean) == MEAN
    assert list(normalize.std) == STD

    # The weak view (first element of the list) is unchanged from the
    # no-strong-aug train transform.
    assert [type(t) for t in weak.transforms] == [Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize]


def _img():
    from PIL import Image
    return Image.new("RGB", (300, 300))


# --- RNG-consumption order -------------------------------------------------
#
# These three guard a hazard no other test in this suite can see: the ORDER in
# which the global torch RNG is consumed while the pipeline is built.
# `CyclingLoader.__init__` calls `iter(loader)` eagerly, and `iter()` on a
# DataLoader whose RandomSampler has `generator=None` seeds its permutation
# from the global torch RNG. `cmct/train.py` therefore interleaves: the LoRA
# branch's DataManager -> wrap its two loaders -> build the LoRA pair (during
# which initialisation draws too) -> branch_mlp's DataManager -> wrap its two
# loaders. A "simplification" that builds both managers (or wraps all four
# loaders) together would silently change branch_mlp's sampler stream and its
# classifier head's initialisation.


def _office_home_tree(root):
    for domain in ("art", "clipart"):
        for name in ("Alarm_Clock", "Backpack"):
            d = root / "office_home" / domain / name
            d.mkdir(parents=True)
            (d / "a.jpg").write_bytes(b"")


def _synthetic_cfg(tmp_path):
    """A dassl cfg over a small synthetic Office-Home tree, as
    tests/test_dassl_dataset.py builds one. No image is ever decoded here: the
    loaders are constructed and iterated over zero times."""
    from vendor.dassl.config import get_cfg_default

    _office_home_tree(tmp_path)
    cfg = get_cfg_default()
    cfg.DATASET.NAME = "OfficeHome"
    cfg.DATASET.ROOT = str(tmp_path)
    cfg.DATASET.SOURCE_DOMAINS = ["art"]
    cfg.DATASET.TARGET_DOMAINS = ["clipart"]
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 2
    cfg.DATALOADER.TRAIN_U.BATCH_SIZE = 2
    cfg.DATALOADER.TEST.BATCH_SIZE = 2
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.PIXEL_MEAN = MEAN
    cfg.INPUT.PIXEL_STD = STD
    cfg.SEED = 42
    return cfg


def test_cycling_loader_construction_advances_the_global_torch_rng(tmp_path):
    """The reason loader-construction order matters, written down and checked.

    If this ever stops holding (a lazy CyclingLoader, or a sampler given its
    own generator), the interleaving in `cmct/train.py` stops being necessary
    -- but until then it is, and the two tests below depend on this fact.
    """
    from cmct.data import build_data_manager

    cfg = _synthetic_cfg(tmp_path)
    dm = build_data_manager(cfg)

    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    CyclingLoader(dm.train_loader_x)
    after = torch.get_rng_state().clone()

    assert not torch.equal(before, after)


def _post_build_rng_probe(cfg, *, collapsed):
    """A stand-in for whatever draws from the RNG after construction.

    Deliberately NOT the LoRA weights: `LoRALayer.init_lora_param` overwrites
    its `nn.init.normal_` draw with a deterministic SVD of the pretrained
    weight, so every `lora_*` tensor is identical regardless of RNG state. The
    draws that genuinely move are branch_mlp's sampler permutation stream and
    `TransferNet.classifier_layer`'s `nn.init.normal_(std=0.001)` head init.
    A plain `torch.randn` probe stands for those.

    `torch.randn(3)` stands in for that initialisation (the real thing is a
    CLIP model this test has no business loading).
    """
    from cmct.data import build_data_manager

    torch.manual_seed(1234)
    if collapsed:
        # The hazard: both managers built and all four loaders wrapped before
        # anything else draws.
        dm_lora = build_data_manager(cfg)
        dm_mlp = build_data_manager(cfg)
        for dm in (dm_lora, dm_mlp):
            CyclingLoader(dm.train_loader_x)
            CyclingLoader(dm.train_loader_u)
        return torch.randn(3)
    # cmct/train.py's actual order.
    dm_lora = build_data_manager(cfg)
    CyclingLoader(dm_lora.train_loader_x)
    CyclingLoader(dm_lora.train_loader_u)
    probe = torch.randn(3)
    dm_mlp = build_data_manager(cfg)
    CyclingLoader(dm_mlp.train_loader_x)
    CyclingLoader(dm_mlp.train_loader_u)
    return probe


def test_interleaved_and_collapsed_build_orders_diverge(tmp_path):
    """Collapsing the interleave is NOT inert: it changes what everything draws.

    With the interleaved order the post-construction draws happen after two
    CyclingLoaders; collapsed, they initialise after four -- a different point
    in the same RNG stream. branch_mlp's sampler is shifted the same way, from
    a state that never saw the intervening construction. Both branches would
    then run from different random state starting at macro-step 1, for a
    reason that has nothing to do with the training loop. This is why
    `cmct/train.py` interleaves the two `build_data_manager` calls around
    `build_lora_pair` and must not be "simplified" back into a single builder
    call.
    """
    cfg = _synthetic_cfg(tmp_path)
    interleaved = _post_build_rng_probe(cfg, collapsed=False)
    collapsed = _post_build_rng_probe(cfg, collapsed=True)
    assert not torch.equal(interleaved, collapsed)


def test_train_interleaves_the_lora_pair_between_the_two_data_managers():
    """`cmct/train.py`'s build order, checked statically.

    The two tests above establish WHY the order matters; this one pins the
    order actually written in `cmct/train.py`, so collapsing the interleave
    fails the suite instead of silently changing what every later draw sees.
    Expected sequence of RNG consumers in `main()`: the LoRA branch's manager,
    its two CyclingLoaders, the LoRA pair, branch_mlp's manager, its two
    CyclingLoaders.
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "cmct" / "train.py"
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")

    watched = {"build_data_manager", "CyclingLoader", "build_lora_pair"}
    # ast.walk yields breadth-first, so sort the matches back into source order.
    calls = [
        node.func.id
        for node in sorted(
            (n for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in watched),
            key=lambda n: (n.lineno, n.col_offset),
        )
    ]
    assert calls == [
        "build_data_manager",
        "CyclingLoader",
        "CyclingLoader",
        "build_lora_pair",
        "build_data_manager",
        "CyclingLoader",
        "CyclingLoader",
    ], f"cmct/train.py builds its RNG consumers in the wrong order: {calls}"
