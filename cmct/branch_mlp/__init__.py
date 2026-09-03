"""branch_mlp: a learned head on CLIP features, trained with the CMKD
self-training loss."""

from .backbone import prompts_for
from .ema import ema_update_teacher
from .model import TransferNet

__all__ = ["TransferNet", "ema_update_teacher", "prompts_for"]
