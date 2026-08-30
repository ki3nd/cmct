"""Multi-kernel maximum mean discrepancy between two feature sets.

The bandwidth is estimated from the combined batch's mean pairwise squared
distance and detached, so it is a fixed scale for the step rather than something
the gradient can move. Five Gaussian kernels then sit on a geometric ladder
around it.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _kernel_matrix(source: Tensor, target: Tensor, kernel_mul: float,
                   kernel_num: int, fix_sigma: float | None) -> Tensor:
    total = torch.cat([source, target], dim=0)
    n = total.size(0)
    n_samples = source.size(0) + target.size(0)

    left = total.unsqueeze(0).expand(n, n, total.size(1))
    right = total.unsqueeze(1).expand(n, n, total.size(1))
    distance = ((left - right) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = torch.as_tensor(fix_sigma, dtype=distance.dtype,
                                    device=distance.device)
    else:
        bandwidth = distance.detach().sum() / (n_samples**2 - n_samples)
    # Divided before the ladder is built, so the kernels straddle the estimate
    # rather than starting at it. Getting this order wrong shifts every kernel.
    bandwidth = bandwidth / kernel_mul ** (kernel_num // 2)

    return sum(torch.exp(-distance / (bandwidth * kernel_mul**i))
               for i in range(kernel_num))


def mk_mmd(source: Tensor, target: Tensor, kernel_mul: float = 2.0,
           kernel_num: int = 5, fix_sigma: float | None = None) -> Tensor:
    """MMD^2 estimate: mean(XX) + mean(YY) - mean(XY) - mean(YX)."""
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError(
            f"expected 2-D feature matrices, got {tuple(source.shape)} and "
            f"{tuple(target.shape)}"
        )
    if source.size(1) != target.size(1):
        raise ValueError(
            f"feature dimensions differ: {source.size(1)} and {target.size(1)}"
        )
    n_source = source.size(0)
    kernels = _kernel_matrix(source, target, kernel_mul, kernel_num, fix_sigma)
    xx = kernels[:n_source, :n_source]
    yy = kernels[n_source:, n_source:]
    xy = kernels[:n_source, n_source:]
    yx = kernels[n_source:, :n_source]
    return xx.mean() + yy.mean() - xy.mean() - yx.mean()
