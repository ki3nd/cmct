"""Data pipeline: `CyclingLoader` and `build_data_manager`, which builds ONE
dassl `DataManager` with this project's transforms.

There is deliberately no whole-pipeline builder here. Both branches get their
own `DataManager`, and the two constructions must NOT sit adjacent to each
other in `cmct/train.py` -- the LoRA pair's initialization has to happen
BETWEEN them, because the order in which the global torch RNG is consumed is
part of what determines the run. `CyclingLoader.__init__` calls `iter(loader)`
eagerly, and `iter()` on a `DataLoader` whose `RandomSampler` has
`generator=None` draws its permutation seed from the global torch RNG.
Constructing both `DataManager`s (and wrapping all four of their loaders)
back-to-back would advance the RNG two draws further than intended before the
LoRA pair's own initialization runs.

Note what that does and does not change. The LoRA weights are NOT affected:
`LoRALayer.init_lora_param` overwrites its `nn.init.normal_` draw with a
deterministic SVD of the pretrained weight, so every `lora_*` tensor comes out
identical regardless of RNG state (measured). What DOES shift is the
`branch_mlp` sampler's permutation stream and `TransferNet.classifier_layer`'s
`nn.init.normal_(std=0.001)` head initialisation. So the interleaving lives at
the call site (`cmct/train.py`), which owns the ordering -- see
`tests/test_data.py`'s RNG-order guard.
"""

from vendor.dassl.data import DataManager

from .transforms import build_transforms


class CyclingLoader:
    """A DataLoader wrapped in ONE persistent iterator, re-created only when
    it's actually exhausted -- never torn down/rebuilt on any other schedule
    (in particular, never tied to "epoch" or eval boundaries).

    Constructing one consumes global torch RNG (see this module's docstring);
    it is deliberately NOT lazy -- the RNG draw has to happen at construction
    time, not at first use.
    """

    def __init__(self, loader):
        self.loader = loader
        self._it = iter(loader)

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)


def build_data_manager(cfg, *, strong_aug: bool = False):
    """Build ONE dassl `DataManager` with this project's transforms, passed as
    `custom_tfm_train` / `custom_tfm_test`. Called once per branch -- the LoRA
    branch's manager first, then the MLP branch's -- so each branch has an
    independent shuffled stream.

    `strong_aug` is not part of dassl's own `cfg` schema, so it is threaded
    through as an explicit parameter rather than read off `cfg`.
    """
    crop_size = cfg.INPUT.SIZE[0]
    tfm_train, tfm_test = build_transforms(
        crop_size, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD, strong_aug=strong_aug
    )
    return DataManager(cfg, custom_tfm_train=tfm_train, custom_tfm_test=tfm_test)
