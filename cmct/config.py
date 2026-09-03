"""Run configuration: one YAML in, frozen dataclasses out.

Every value a run depends on is stated once, here, rather than assembled from
scattered defaults -- so nothing about a run's behavior has to be reconstructed
from more than one file.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, get_type_hints

import yaml

from cmct.losses import DIVERSITY_LOSSES
from vendor.dassl.config import get_cfg_default

# miniDomainNet is deliberately excluded: cmct/branch_mlp/backbone.py carries
# hardcoded prompt lists for officehome, office31, visda17 and domainnet only,
# so accepting "minidomainnet" here would only defer a KeyError to deep inside
# model construction. Rejecting it at config-parse time is strictly better.
LORA_BACKBONES = ("ViT-B/16", "ViT-B/32", "ViT-L/14")
"""Kept in step with branch_lora/lora/apply.py's INDEX_POSITIONS_VISION."""

MLP_BACKBONES = ("ViT-B/16", "RN50", "RN101")
"""Kept in step with branch_mlp/backbone.py's checkpoint map."""

DATASET_NAMES = {
    "officehome": "OfficeHome",
    "office31": "Office31",
    "visda17": "VisDA17",
    "domainnet": "DomainNet",
}

# INPUT.INTERPOLATION is set to "bicubic" here, overriding dassl's own default
# of "bilinear", to match the resolved dassl config this project expects. It is
# not exposed as a config knob because it never varies across the datasets
# this project supports -- and, as noted on `to_dassl_cfg` below, it is inert:
# the transforms that actually run are built directly and are BILINEAR.
_INTERPOLATION = "bicubic"


@dataclass(frozen=True)
class BatchSize:
    source: int
    target: int
    test: int


@dataclass(frozen=True)
class DataConfig:
    root: str
    name: str
    source: list[str]
    target: list[str]
    image_size: int
    pixel_mean: list[float]
    pixel_std: list[float]
    batch_size: BatchSize
    num_workers: int
    strong_aug: bool


@dataclass(frozen=True)
class TrainConfig:
    iters: int
    mlp_steps_per_iter: int
    print_freq: int
    eval_freq: int


@dataclass(frozen=True)
class LoraBackboneConfig:
    name: str
    """A CLIP checkpoint name. Must be one of the ViT backbones: LoRA is injected
    into the vision transformer's attention blocks, and the injection table only
    covers ViT (branch_lora/lora/apply.py's INDEX_POSITIONS_VISION)."""
    path: str
    """Where this branch downloads its checkpoint to."""


@dataclass(frozen=True)
class MlpBackboneConfig:
    name: str
    """A CLIP checkpoint name. Any CLIP backbone works here -- the branch reads
    features through `encode_image` and needs no architecture-specific plumbing,
    and the classifier head's width is derived from whatever the backbone emits.

    Choosing a ResNet has one consequence worth knowing: `fix_bn` in
    branch_mlp/model.py puts every BatchNorm in the backbone into eval at the
    start of each forward, so the backbone's BN keeps its pretrained running
    statistics and never adapts to the target domain. A ViT has no BatchNorm, so
    that line is inert there and live on a ResNet. It is deliberate -- freezing
    BN is standard when fine-tuning a pretrained backbone on a small, shifted
    target set -- and it belongs to the fine-tuning setup rather than to the
    CMKD loss, so it should survive a loss swap.

    This branch downloads through CLIP's own loader, which caches in
    ~/.cache/clip; it has no path setting of its own."""


@dataclass(frozen=True)
class LoraConfig:
    position: str
    params: list[str]
    r: int
    alpha: int
    dropout: float
    rank_ramp: list[int]


@dataclass(frozen=True)
class WarmupConfig:
    lr: float
    iters: int


@dataclass(frozen=True)
class DiversityConfig:
    """Diversity/confidence regulariser on `branch_lora`'s TARGET predictions.

    `branch_lora`'s four other terms (source CE, self and cross masked
    pseudo-label CE, MK-MMD) all push predictions to be sharper; none of them
    stops the branch collapsing onto a subset of classes. `weight: 0.0` leaves
    the branch exactly as it was, which is the shipped default so that `main`
    keeps reproducing the reference loss trace.
    """
    kind: str
    weight: float


@dataclass(frozen=True)
class BranchLoraConfig:
    enabled: bool
    backbone: LoraBackboneConfig
    # fp16 or fp32, and it governs THIS BRANCH ONLY. branch_mlp's backbone is
    # always fp32 and is not configurable -- its `.float()` call in
    # branch_mlp/backbone.py restores fp32 explicitly after cmct.clip's
    # build_model downcasts to fp16. The two branches therefore run at
    # DIFFERENT precisions, by design.
    precision: str
    lora: LoraConfig
    lr: float
    warmup: WarmupConfig
    momentum: float
    weight_decay: float
    grad_clip: float
    mmd_weight: float
    cross_weight: float
    ema_momentum: float
    diversity: DiversityConfig


@dataclass(frozen=True)
class Lambdas:
    task: float
    source_ce: float
    target_gini: float


@dataclass(frozen=True)
class TeacherEmaConfig:
    momentum: float
    schedule: str
    hard_copy_iters: int


@dataclass(frozen=True)
class BranchMlpConfig:
    backbone: MlpBackboneConfig
    lr: float
    classifier_lr_mult: float
    lr_gamma: float
    lr_decay: float
    momentum: float
    weight_decay: float
    nesterov: bool
    label_smoothing: float
    lambdas: Lambdas
    lamb_gamma: float
    warmup_iters: int
    cross_weight: float
    self_from_teacher: bool
    ema: TeacherEmaConfig


@dataclass(frozen=True)
class DebiasConfig:
    enabled: bool
    tau: float
    momentum: float


@dataclass(frozen=True)
class PseudoLabelConfig:
    threshold: float
    debias: DebiasConfig


@dataclass(frozen=True)
class Config:
    seed: int
    gpu: int
    output_dir: str
    data: DataConfig
    train: TrainConfig
    branch_lora: BranchLoraConfig
    branch_mlp: BranchMlpConfig
    pseudo_label: PseudoLabelConfig

    @staticmethod
    def from_yaml(path: str) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = _build(Config, raw, "")
        _validate(cfg)
        return cfg


def _build(cls, raw: Any, path: str):
    """Recursively instantiate `cls` from `raw`, rejecting keys with no field.

    A typo must fail here rather than silently leave a default in place -- that
    class of mistake is what the old merge chain made easy.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"{path or '<root>'}: expected a mapping, got {type(raw).__name__}")
    known = {f.name: f for f in fields(cls)}
    # `from __future__ import annotations` turns every dataclass field's
    # `.type` into an unevaluated string, so `is_dataclass(field.type)` would
    # always be False. Resolve real types once per class instead.
    hints = get_type_hints(cls)
    unknown = sorted(set(raw) - set(known))
    if unknown:
        where = f"{path}." if path else ""
        raise ValueError(f"unknown config key(s): {', '.join(where + k for k in unknown)}")
    missing = sorted(set(known) - set(raw))
    if missing:
        where = f"{path}." if path else ""
        raise ValueError(f"missing config key(s): {', '.join(where + k for k in missing)}")
    kwargs = {}
    for name in known:
        value = raw[name]
        child_path = f"{path}.{name}" if path else name
        field_type = hints[name]
        kwargs[name] = _build(field_type, value, child_path) if is_dataclass(field_type) else value
    return cls(**kwargs)


def _validate(cfg: Config) -> None:
    if cfg.branch_lora.backbone.name not in LORA_BACKBONES:
        # Without this the run dies much later, inside apply_lora, with a bare
        # KeyError about a "vision position".
        raise ValueError(
            f"branch_lora.backbone.name must be one of {sorted(LORA_BACKBONES)} "
            f"(LoRA is injected into ViT attention blocks), got "
            f"{cfg.branch_lora.backbone.name!r}"
        )
    if cfg.branch_mlp.backbone.name not in MLP_BACKBONES:
        raise ValueError(
            f"branch_mlp.backbone.name must be one of {sorted(MLP_BACKBONES)}, got "
            f"{cfg.branch_mlp.backbone.name!r}"
        )
    if cfg.branch_lora.diversity.kind not in DIVERSITY_LOSSES:
        raise ValueError(
            f"branch_lora.diversity.kind must be one of "
            f"{sorted(DIVERSITY_LOSSES)}, got {cfg.branch_lora.diversity.kind!r}"
        )
    if cfg.branch_lora.diversity.weight < 0.0:
        raise ValueError("branch_lora.diversity.weight must be non-negative")
    if cfg.branch_lora.precision not in ("fp16", "fp32"):
        raise ValueError(
            f"branch_lora.precision must be fp16 or fp32, got {cfg.branch_lora.precision!r}"
        )
    if cfg.branch_mlp.ema.schedule not in ("dacs", "hard_copy"):
        raise ValueError(
            f"branch_mlp.ema.schedule must be dacs or hard_copy, got {cfg.branch_mlp.ema.schedule!r}"
        )
    if cfg.data.name not in DATASET_NAMES:
        raise ValueError(f"data.name must be one of {sorted(DATASET_NAMES)}, got {cfg.data.name!r}")
    if cfg.train.iters <= 0:
        raise ValueError("train.iters must be positive")
    if not 0.0 <= cfg.pseudo_label.threshold <= 1.0:
        raise ValueError("pseudo_label.threshold must lie in [0, 1]")
    if cfg.data.batch_size.target != cfg.data.batch_size.source:
        # dassl's DATALOADER.TRAIN_U.SAME_AS_X defaults to True, and
        # to_dassl_cfg leaves it alone, so DataManager discards TRAIN_U's batch
        # size and copies TRAIN_X's. A different target value would be ignored
        # in silence, which is exactly what this config layer exists to stop:
        # rejecting a target that differs from the source, rather than
        # accepting a value it would go on to ignore.
        raise ValueError(
            f"data.batch_size.target ({cfg.data.batch_size.target}) must equal "
            f"data.batch_size.source ({cfg.data.batch_size.source}): dassl's "
            f"DATALOADER.TRAIN_U.SAME_AS_X makes the target value inert"
        )


def resolve(config: Config) -> Config:
    """Apply cross-field consequences after validation, returning a new frozen Config.

    Today's only consequence: with no LoRA teacher (branch_lora.enabled is
    False) there is nothing for branch_mlp to cross-teach from, so
    branch_mlp.cross_weight is forced to 0.0.
    """
    if not config.branch_lora.enabled and config.branch_mlp.cross_weight != 0.0:
        new_cmkd = dataclasses.replace(config.branch_mlp, cross_weight=0.0)
        config = dataclasses.replace(config, branch_mlp=new_cmkd)
    return config


def to_dassl_cfg(config: Config):
    """Build the yacs CfgNode that vendor.dassl's DataManager needs.

    Sets only what DataManager reads: DATASET.NAME/ROOT/SOURCE_DOMAINS/
    TARGET_DOMAINS, DATALOADER.TRAIN_X/TRAIN_U/TEST.BATCH_SIZE,
    DATALOADER.NUM_WORKERS, INPUT.SIZE/PIXEL_MEAN/PIXEL_STD/INTERPOLATION and
    SEED. Everything else keeps dassl's own default -- in particular
    INPUT.TRANSFORMS is never consulted because the custom transforms are
    passed to DataManager directly.

    Two of the keys set here are INERT in this pipeline. `INPUT.INTERPOLATION`
    would feed `build_transform`, but `DataManager` is always given
    `custom_tfm_train`/`custom_tfm_test` and so never calls it; its only other
    reader is `DatasetWrapper.to_tensor`, which is used only when
    `DATALOADER.RETURN_IMG0` is set, and it never is. The transforms that
    actually run are built in `cmct/data/transforms.py` and are BILINEAR, not
    bicubic. `INPUT.TRANSFORMS` is bypassed for the same reason and is
    deliberately left at dassl's empty default -- setting it to anything would
    have no effect on what actually runs.
    """
    cfg = get_cfg_default()

    cfg.DATASET.NAME = DATASET_NAMES[config.data.name]
    cfg.DATASET.ROOT = config.data.root
    cfg.DATASET.SOURCE_DOMAINS = list(config.data.source)
    cfg.DATASET.TARGET_DOMAINS = list(config.data.target)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = config.data.batch_size.source
    cfg.DATALOADER.TRAIN_U.BATCH_SIZE = config.data.batch_size.target
    cfg.DATALOADER.TEST.BATCH_SIZE = config.data.batch_size.test
    cfg.DATALOADER.NUM_WORKERS = config.data.num_workers

    cfg.INPUT.SIZE = (config.data.image_size, config.data.image_size)
    cfg.INPUT.PIXEL_MEAN = list(config.data.pixel_mean)
    cfg.INPUT.PIXEL_STD = list(config.data.pixel_std)
    cfg.INPUT.INTERPOLATION = _INTERPOLATION

    cfg.SEED = config.seed

    return cfg
