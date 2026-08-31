"""Data layer: transforms match VLP-UDA, dassl's loader behaviour is preserved,
and each branch's streams are independent and reproducible."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from cmct.config import load_experiment
from cmct.config.schema import CropSpec, DatasetSpec, TransformSpec
from cmct.data import (
    BatchSource,
    DataError,
    ImageListDataset,
    InfiniteStream,
    build_split,
    build_test,
    build_test_loader,
    build_train,
)
from cmct.data.dataset import read_image

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENT = CONFIGS / "experiment" / "cmct_officehome_a2c.yaml"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def spec(name="officehome"):
    return load_experiment(EXPERIMENT).dataset if name == "officehome" else _spec_from_yaml(name)


def _spec_from_yaml(name):
    from cmct.config.loader import _build
    return _build(DatasetSpec, yaml.safe_load((CONFIGS / "dataset" / f"{name}.yaml").read_text()),
                  name)


def cfg_for(root: Path, dataset_name="officehome", **data_overrides):
    """Config pointed at a synthetic dataset root, with num_classes/dir adjusted
    to the fixture."""
    cfg = load_experiment(EXPERIMENT)
    cfg.dataset = dataclasses.replace(
        _spec_from_yaml(dataset_name), dir=root.name, num_classes=3,
    )
    cfg.data = dataclasses.replace(
        cfg.data, root=str(root.parent), source_domains=["art"], target_domains=["clipart"],
        **data_overrides,
    )
    return cfg


# --- transforms: must match VLP-UDA's own recipe -----------------------------

def test_train_transform_matches_vlpuda_recipe():
    steps = build_train(spec()).transforms
    assert [type(s) for s in steps] == [T.Resize, T.RandomCrop, T.RandomHorizontalFlip,
                                        T.ToTensor, T.Normalize]
    resize, crop, _, _, norm = steps
    assert resize.size == [256, 256]
    assert resize.interpolation is InterpolationMode.BILINEAR
    assert crop.size == (224, 224)
    assert tuple(norm.mean) == CLIP_MEAN
    assert tuple(norm.std) == CLIP_STD


def test_test_transform_matches_vlpuda_recipe():
    steps = build_test(spec()).transforms
    assert [type(s) for s in steps] == [T.Resize, T.ToTensor, T.Normalize]
    assert steps[0].size == [224, 224]
    assert steps[0].interpolation is InterpolationMode.BILINEAR


def test_interpolation_is_bilinear_not_bicubic():
    """vlpuda calls Resize([256,256]) with no interpolation argument, so the
    torchvision default (bilinear) is what actually ran -- despite the dassl
    trainer yaml saying INTERPOLATION: "bicubic", which never took effect."""
    for build in (build_train, build_test):
        assert build(spec()).transforms[0].interpolation is InterpolationMode.BILINEAR


def test_visda_pipeline_differs_from_the_others():
    visda = _spec_from_yaml("visda17")
    steps = build_train(visda).transforms
    assert [type(s) for s in steps] == [T.Resize, T.CenterCrop, T.RandomHorizontalFlip,
                                        T.ToTensor, T.Normalize]
    assert steps[0].size == [224, 224]


def test_visda_canonical_order_equals_original_order():
    """vlpuda's visda train pipeline is Resize -> hflip -> CenterCrop, while this
    code always emits Resize -> crop -> hflip. The two agree because CenterCrop
    at the resize size is a no-op; this pins that down instead of asserting it."""
    img = Image.new("RGB", (300, 260), color=(31, 63, 127))
    resize = T.Resize([224, 224], interpolation=InterpolationMode.BILINEAR)
    center = T.CenterCrop(224)
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])

    original = to_tensor(center(resize(img)))       # flip fixed off in both
    canonical = to_tensor(center(resize(img)))
    assert torch.equal(original, canonical)
    # and CenterCrop really is identity at that size
    assert torch.equal(to_tensor(resize(img)), to_tensor(center(resize(img))))


def test_crop_none_skips_the_crop_step():
    s = dataclasses.replace(
        spec(),
        transform=TransformSpec(train=CropSpec(resize=(256, 256), crop="none", hflip=False),
                                test=CropSpec(resize=(224, 224), crop="none")),
    )
    assert [type(x) for x in build_train(s).transforms] == [T.Resize, T.ToTensor, T.Normalize]


# --- reader ------------------------------------------------------------------

def test_class_folder_reader(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    assert split.classnames == ["alarm clock", "backpack", "mouse"]
    assert split.num_classes == 3
    assert len(split.train_x) == 3 * 4
    assert len(split.train_u) == 3 * 5
    assert split.test == split.train_u
    assert {d.label for d in split.train_x} == {0, 1, 2}


def test_class_folder_label_mapping_is_shared_across_domains(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    by_name = {(d.classname, d.label) for d in split.train_x + split.train_u}
    assert len({label for _, label in by_name}) == 3
    assert dict(by_name) == {"alarm_clock": 0, "backpack": 1, "mouse": 2}


def test_image_list_reader(image_list_root):
    cfg = cfg_for(image_list_root, dataset_name="visda17")
    cfg.data = dataclasses.replace(cfg.data, source_domains=["art"], target_domains=["clipart"])
    split = build_split(cfg.dataset, cfg.data)
    assert split.classnames == ["alarm clock", "backpack", "mouse"]
    assert len(split.train_x) == 3 * 4
    assert len(split.train_u) == 3 * 5
    assert all(Path(d.path).is_file() for d in split.train_x)


def test_num_classes_mismatch_raises(class_folder_root):
    cfg = cfg_for(class_folder_root)
    cfg.dataset = dataclasses.replace(cfg.dataset, num_classes=65)
    with pytest.raises(DataError, match="declares num_classes=65"):
        build_split(cfg.dataset, cfg.data)


def test_missing_domain_dir_raises(class_folder_root):
    cfg = cfg_for(class_folder_root)
    cfg.data = dataclasses.replace(cfg.data, source_domains=["product"])
    with pytest.raises(DataError, match="domain directory .* not found"):
        build_split(cfg.dataset, cfg.data)


def test_classnames_are_normalized_even_with_clarify_off(class_folder_root):
    """Underscores become spaces and names are lowercased whether or not the
    override flag is on. That transform is what reproduces the original's prompt
    list; gating it behind the flag put a literal underscore token into 9 of
    Office-Home's 65 prompts by default."""
    cfg = cfg_for(class_folder_root, clarify_classnames=False)
    split = build_split(cfg.dataset, cfg.data)
    assert split.classnames == ["alarm clock", "backpack", "mouse"]
    assert all("_" not in c and c == c.lower() for c in split.classnames)


def test_clarify_classnames_applies_overrides(class_folder_root):
    cfg = cfg_for(class_folder_root, clarify_classnames=True)
    split = build_split(cfg.dataset, cfg.data)
    assert split.classnames == ["alarm clock", "backpack", "computer mouse"]


# --- ImageListDataset: n views, one decode -----------------------------------

def test_two_views_come_from_one_disk_read(class_folder_root, monkeypatch):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)

    opens = []
    real_open = Image.open

    def counting_open(path, *a, **kw):
        opens.append(path)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(Image, "open", counting_open)
    weak, other = build_train(cfg.dataset), build_test(cfg.dataset)
    sample = ImageListDataset(split.train_x, [weak, other])[0]
    assert set(sample) == {"label", "index", "img", "img2"}
    assert len(opens) == 1


def test_single_transform_gives_only_img(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    sample = ImageListDataset(split.train_x, build_train(cfg.dataset))[0]
    assert set(sample) == {"label", "index", "img"}
    assert sample["img"].shape == (3, 224, 224)


def test_read_image_converts_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (8, 8), color=128).save(path)
    assert read_image(str(path)).mode == "RGB"


# --- loader: dassl behaviour --------------------------------------------------

def test_train_drops_short_final_batch_and_test_keeps_it(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    cfg.branches[0] = dataclasses.replace(
        cfg.branches[0],
        stream=dataclasses.replace(cfg.branches[0].stream, batch_size_x=5, batch_size_u=5,
                                   num_workers=0),
    )
    source = BatchSource(split, cfg.dataset, cfg.branches[0])
    assert len(split.train_x) == 12
    assert len(source.source.loader) == 2                    # 12 // 5, the short batch is dropped

    cfg.data = dataclasses.replace(cfg.data, batch_size_test=7, num_workers_test=0)
    test_loader = build_test_loader(split, cfg.dataset, cfg.data)
    assert len(test_loader) == 3                             # ceil(15 / 7), the short batch is kept


def test_batch_larger_than_dataset_disables_drop_last(class_folder_root):
    """dassl: drop_last = is_train and len(data_source) >= batch_size. A batch
    bigger than the dataset therefore keeps the single short batch rather than
    yielding an empty loader."""
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    branch = dataclasses.replace(
        cfg.branches[0],
        stream=dataclasses.replace(cfg.branches[0].stream, batch_size_x=999, batch_size_u=999,
                                   num_workers=0),
    )
    source = BatchSource(split, cfg.dataset, branch)
    assert len(source.source.loader) == 1
    assert source.source.loader.drop_last is False


def test_empty_loader_raises(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    split.train_x = []
    branch = dataclasses.replace(
        cfg.branches[0],
        stream=dataclasses.replace(cfg.branches[0].stream, num_workers=0),
    )
    with pytest.raises(ValueError, match="empty loader"):
        BatchSource(split, cfg.dataset, branch)


def test_test_loader_order_is_deterministic(class_folder_root):
    cfg = cfg_for(class_folder_root)
    cfg.data = dataclasses.replace(cfg.data, batch_size_test=4, num_workers_test=0)
    split = build_split(cfg.dataset, cfg.data)
    def order():
        loader = build_test_loader(split, cfg.dataset, cfg.data)
        return [i for batch in loader for i in batch["index"].tolist()]

    orders = [order(), order()]
    assert orders[0] == orders[1] == list(range(len(split.test)))


def test_strong_aug_raises_instead_of_silently_doing_nothing(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    cfg.branches[0] = dataclasses.replace(
        cfg.branches[0],
        stream=dataclasses.replace(cfg.branches[0].stream, strong_aug=True, num_workers=0),
    )
    with pytest.raises(NotImplementedError, match="strong_aug"):
        BatchSource(split, cfg.dataset, cfg.branches[0])


# --- streams: shuffle comes from the shared global RNG, like dassl's own -----
#
# There is no per-branch seed any more (see loaders.py's docstring): a stream's
# shuffle is whatever the global RNG produces at the point it is built, exactly
# like dassl's RandomSampler(data_source). These tests seed that global RNG
# explicitly (torch.manual_seed), the way train_lora.set_seed does once per
# real run, and check the property that actually holds now: two streams differ
# because they are built at different points in the stream, and rebuilding from
# the same point reproduces -- but WHICH point a branch lands on depends on how
# many draws happened first, i.e. on build order. That is a deliberate trade
# for matching train_mfa_v2.py's real behaviour, not an oversight.

def order_of(split, cfg, branch, take=12):
    src = BatchSource(split, cfg.dataset, branch)
    out = []
    while len(out) < take:
        out.extend(src.source.next()["index"].tolist())
    return out[:take]


def small_branch(cfg, index, name=None):
    branch = cfg.branches[index]
    return dataclasses.replace(
        branch, name=name or branch.name,
        stream=dataclasses.replace(branch.stream, batch_size_x=4, batch_size_u=4, num_workers=0),
    )


def test_two_branches_get_different_order(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    torch.manual_seed(0)
    a = order_of(split, cfg, small_branch(cfg, 0))
    b = order_of(split, cfg, small_branch(cfg, 1))
    assert a != b


def test_same_seed_and_build_order_reproduces(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    torch.manual_seed(0)
    first = order_of(split, cfg, small_branch(cfg, 0))
    torch.manual_seed(0)
    again = order_of(split, cfg, small_branch(cfg, 0))
    assert first == again


def test_build_order_changes_the_stream(class_folder_root):
    """Unlike a per-branch-seeded generator, a branch's shuffle now depends on
    how many global-RNG draws happened before it was built -- matching
    train_mfa_v2.py's dm1/dm2 (dassl.RandomSampler(data_source), no generator=),
    where the SAME branch built second gets a different shuffle than built
    first. This is the trade this design makes; see loaders.py's docstring."""
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    torch.manual_seed(0)
    built_first = order_of(split, cfg, small_branch(cfg, 0))
    torch.manual_seed(0)
    order_of(split, cfg, small_branch(cfg, 1))              # consumes RNG draws first
    built_second = order_of(split, cfg, small_branch(cfg, 0))
    assert built_first != built_second


def test_source_and_target_streams_differ(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    torch.manual_seed(0)
    src = BatchSource(split, cfg.dataset, small_branch(cfg, 0))
    source_order = src.source.next()["index"].tolist()
    target_order = src.target.next()["index"].tolist()
    assert source_order != target_order


def test_infinite_stream_never_exhausts(class_folder_root):
    cfg = cfg_for(class_folder_root)
    cfg.data = dataclasses.replace(cfg.data, batch_size_test=4, num_workers_test=0)
    split = build_split(cfg.dataset, cfg.data)
    loader = build_test_loader(split, cfg.dataset, cfg.data)
    stream = InfiniteStream(loader)
    n = len(loader)
    for _ in range(n * 3 + 1):
        assert "img" in stream.next()
    assert stream.epochs == 3


def test_batch_shape_and_strong_view_absent(class_folder_root):
    cfg = cfg_for(class_folder_root)
    split = build_split(cfg.dataset, cfg.data)
    src = BatchSource(split, cfg.dataset, small_branch(cfg, 0))
    batch = src.next()
    assert batch.img_x.shape == (4, 3, 224, 224)
    assert batch.label_x.shape == (4,)
    assert batch.img_u.shape == (4, 3, 224, 224)
    assert batch.img_u_strong is None
    assert batch.student_img_u is batch.img_u
