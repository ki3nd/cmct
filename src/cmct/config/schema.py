"""The config dataclass tree. A field with no default is required in YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CropSpec:
    """One stage of resize-then-crop. Applied in a fixed order:
    Resize(resize) -> crop -> hflip -> ToTensor -> Normalize."""

    resize: tuple[int, int]
    crop: Literal["random", "center", "none"] = "none"
    crop_size: int = 224
    hflip: bool = False


@dataclass
class TransformSpec:
    """Transform pipelines for one dataset. Shape is a dataset property, not an
    experiment knob: VisDA resizes straight to 224 and center-crops where the
    others resize to 256 and random-crop."""

    train: CropSpec
    test: CropSpec
    interpolation: Literal["bilinear", "bicubic"] = "bilinear"


@dataclass
class DatasetSpec:
    """Properties of a dataset, identical across experiments.

    Loaded from configs/dataset/<name>.yaml.
    """

    name: str
    dir: str
    domains: list[str]
    num_classes: int
    layout: Literal["class_folder", "image_list"]
    transform: TransformSpec
    pixel_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    pixel_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    classname_overrides: dict[str, str] = field(default_factory=dict)
    """Replacement class names, applied when data.clarify_classnames is on. Keys
    are the original class names, lowercased with underscores turned into
    spaces."""


@dataclass
class DataConfig:
    root: str
    source_domains: list[str]
    target_domains: list[str]
    batch_size_test: int = 128
    num_workers_test: int = 8
    clarify_classnames: bool = False
    """Apply DatasetSpec.classname_overrides. Shared by every branch: giving two
    branches two different sets of class names breaks the shared label space."""


@dataclass
class BackboneConfig:
    checkpoint: str
    """Path to a CLIP checkpoint. The architecture is inferred from the
    state_dict, so it is not declared here."""
    dtype: Literal["fp16", "fp32"]


@dataclass
class StreamConfig:
    """One branch's own data stream."""

    batch_size_x: int = 32
    batch_size_u: int = 32
    num_workers: int = 8
    strong_aug: bool = False
    """Produce an extra, harder-augmented target view for this branch. Its cost
    differs by branch type, so it is a per-branch choice. NOT IMPLEMENTED yet:
    setting it raises rather than silently doing nothing."""


@dataclass
class OptimConfig:
    lr: float
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = False
    grad_clip: float | None = None
    scheduler: Literal["cosine", "inv", "none"] = "cosine"
    gamma: float = 3e-4
    """Only read when scheduler == "inv": lr * (1 + gamma * t) ** -decay."""
    decay: float = 0.75
    """Only read when scheduler == "inv"."""
    warmup_lr: float | None = None
    """Flat LR held during the branch's warmup, or None to keep `lr`."""
    param_group_multipliers: dict[str, float] = field(default_factory=dict)
    """Per-param-group LR multiplier by group name, e.g. {"head": 1000.0}."""


@dataclass
class EmaConfig:
    momentum: float = 0.99
    schedule: Literal["const", "dacs", "hard_copy_then_jump"] = "dacs"
    warmup_iters: int = 100
    """Only read when schedule == "hard_copy_then_jump"."""


@dataclass
class PseudoLabelConfig:
    threshold: float = 0.85
    self_reduce: Literal["mask", "ratio"] = "mask"
    cross_reduce: Literal["mask", "ratio"] = "mask"
    """Separate from self_reduce because the two reference distributions sit in
    different mask-sparsity regimes."""


@dataclass
class DebiasConfig:
    enabled: bool = False
    tau: float = 0.5
    momentum: float = 0.99


@dataclass
class BranchConfig:
    name: str
    type: str
    """Branch-type registry key."""
    backbone: BackboneConfig
    optim: OptimConfig
    steps_per_macro: int = 1
    warmup_steps: int = 0
    """This branch's own steps before its cross term switches on."""
    cross_weight: float = 0.5
    cross_mode: str = "mask"
    """Form of the cross loss; valid values are defined by each branch type."""
    stream: StreamConfig = field(default_factory=StreamConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    debias: DebiasConfig = field(default_factory=DebiasConfig)
    extra: dict[str, Any] = field(default_factory=dict)
    """Branch-type-private settings, validated by that branch alone."""


@dataclass
class CoTrainConfig:
    cross_ref_refresh: Literal["macro", "micro"]
    total_macro_steps: int = 1000
    ensemble: Literal["mean_prob", "mean_logit"] = "mean_prob"


@dataclass
class RunConfig:
    output_dir: str
    seed: int = 42
    device: str = "cuda:0"
    print_freq: int = 50
    eval_freq: int = 200


@dataclass
class Config:
    dataset: DatasetSpec
    data: DataConfig
    cotrain: CoTrainConfig
    branches: list[BranchConfig]
    run: RunConfig
