"""EMA teacher update for `branch_mlp`.

See NOTICE for this module's license terms.
"""

import torch


@torch.no_grad()
def ema_update_teacher(teacher_model, base_network, momentum=0.99):
    """EMA (temporal fusion) update for `TransferNet.teacher_model`, which is
    hard-copied from `base_network` at init and then updated every step by
    this function. It runs a generic parameter+buffer-wise EMA over the whole
    module (not just `.model.visual`) so it stays a structurally-consistent
    deepcopy of `base_network`; any non-floating buffer (e.g. BatchNorm's
    `num_batches_tracked`, if the backbone uses one) is hard-copied instead of
    EMA'd since a float momentum update doesn't apply to an integer
    counter."""
    teacher_state = teacher_model.state_dict()
    for k, v in base_network.state_dict().items():
        if not torch.is_floating_point(v):
            teacher_state[k].copy_(v)
        else:
            teacher_state[k].mul_(momentum).add_(v, alpha=1.0 - momentum)
