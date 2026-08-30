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
    scheduler: Literal["cosine", "inv", "warmup_cosine", "none"] = "cosine"
    """"cosine": one cosine decay over total_steps. "inv": the formula below.
    "warmup_cosine": flat `warmup_lr` for `warmup_steps` AND for the boundary
    step itself, then a cosine decay over the remaining steps. "none": constant."""
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
    schedule: Literal["ramp", "const", "hard_copy_then_jump"] = "ramp"
    warmup_iters: int = 100
    """Only read when schedule == "hard_copy_then_jump"."""


@dataclass
class PseudoLabelConfig:
    threshold: float = 0.85
    self_reduce: Literal["mask", "ratio"] = "mask"


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
    ensemble: Literal["off", "mean_prob", "mean_logit"] = "off"
    """Whether to also score the two teachers combined, and how.

    "off" (default) computes no ensemble at all -- there is no ensemble number
    anywhere, rather than one computed and hidden. A deviation: the reference
    always reports all three (train_mfa_v2.py:266-289). Off here because the
    first question is whether cross-teaching helps each branch on its own, and
    an ensemble number mixes that with a second question.

    "mean_prob" averages probabilities, which is what the reference does.
    "mean_logit" averages logits and is NOT equivalent: branch 1's logits are
    `logit_scale * cosine` with logit_scale around 100 while branch 2's come
    from a linear head, so averaging logits weights branch 1 far more heavily.
    It exists to make that visible."""


@dataclass
class RunConfig:
    output_dir: str
    seed: int = 42
    device: str = "cuda:0"
    print_freq: int = 50
    eval_freq: int = 50
    """In macro-steps. With steps_per_macro = 10 that is every 500 branch steps,
    i.e. 20 evaluations over a 10000-step run -- the cadence of the training loop
    this reproduces (20 epochs x 500 iterations)."""


@dataclass
class Config:
    dataset: DatasetSpec
    data: DataConfig
    cotrain: CoTrainConfig
    branches: list[BranchConfig]
    run: RunConfig
