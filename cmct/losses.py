"""Shared loss functions used by both branches."""

import torch
import torch.nn.functional as F


def _gaussian_kernel_matrix(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Multi-kernel Gaussian similarity matrix over the concatenated
    source/target features, used by `mk_mmd`."""
    n_samples = x.size(0) + y.size(0)
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-l2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)


def mk_mmd(source_features, target_features, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Multi-kernel Maximum Mean Discrepancy between two feature sets.

    source_features: (N, d) tensor
    target_features: (M, d) tensor
    """
    n_source = source_features.size(0)
    kernels = _gaussian_kernel_matrix(source_features, target_features, kernel_mul, kernel_num, fix_sigma)

    XX = kernels[:n_source, :n_source]
    YY = kernels[n_source:, n_source:]
    XY = kernels[:n_source, n_source:]
    YX = kernels[n_source:, :n_source]

    return torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)


def masked_cross_entropy(logits, prob_ref, threshold: float):
    """Cross-entropy against a pseudo-label derived from `prob_ref`, masked to
    only the examples where `prob_ref`'s top-class probability clears
    `threshold`."""
    max_probs, pseudo_label = torch.max(prob_ref, dim=-1)
    mask = max_probs.ge(threshold).float()
    epsilon = 1e-8
    return (F.cross_entropy(logits, pseudo_label, reduction="none") * mask).sum() / (mask.sum() + epsilon)


class DebiasTracker:
    """Debiases pseudo-label logits against a running class-marginal estimate
    `qhat`.

    The update ORDER in `correct` is load-bearing: correct `logits` using
    `qhat` as of BEFORE the call, THEN update `qhat` from the raw,
    pre-correction prediction -- never the other way round. Updating `qhat`
    first would let each correction see its own contribution, biasing the
    estimate toward whatever the model just predicted.
    """

    def __init__(self, num_classes, tau, momentum, device):
        self.tau = tau
        self.momentum = momentum
        self.qhat = torch.full((num_classes,), 1.0 / num_classes, device=device)

    def correct(self, logits):
        prob_raw = F.softmax(logits, dim=-1)
        corrected = logits - self.tau * torch.log(self.qhat)
        self.qhat.mul_(self.momentum).add_(prob_raw.mean(dim=0), alpha=1.0 - self.momentum)
        return corrected


def _probs_fp32(logits):
    """Softmax in float32.

    `branch_lora` runs at fp16 (see `branch_lora.precision`), and the entropy
    terms below take `log` of probabilities that a near-one-hot softmax drives
    to zero -- in fp16 those underflow to exactly 0 and the log to -inf. The
    other branch is already fp32, so this cast is a no-op there.
    """
    return F.softmax(logits.float(), dim=-1)


def information_maximization(logits):
    """DINE's IM term, with the sign flipped so it is MINIMISED.

    DINE writes `L = Lkd + Lml - Lim` with
    `Lim = h(E[f(x)]) - E[h(f(x))]`; this returns `-Lim` directly, so it adds
    to a loss with a positive weight like every other term here:

        E[h(f)]  -> minimised: each prediction becomes confident
        -h(E[f]) -> the mean prediction's entropy is MAXIMISED: the batch
                    spreads across classes instead of collapsing onto a few

    The second half is the one that matters for a branch whose logits are
    scaled by CLIP's `logit_scale` (~100): those predictions are already
    near-one-hot, so `E[h(f)]` sits near zero with a vanishing gradient while
    the marginal-entropy half still does real work. `balanced_gini` degrades
    more gracefully under that saturation.
    """
    probs = _probs_fp32(logits)
    epsilon = 1e-8
    mean_entropy = -(probs * torch.log(probs + epsilon)).sum(dim=-1).mean()
    marginal = probs.mean(dim=0)
    marginal_entropy = -(marginal * torch.log(marginal + epsilon)).sum()
    return mean_entropy - marginal_entropy


def balanced_gini(logits):
    """Class-balanced Gini impurity: minimised by confident AND diverse batches.

    Same functional form as `branch_mlp`'s CMKD `gini_impurity`, with one
    deliberate difference: this reduces by MEAN over the batch where CMKD
    reduces by SUM. `branch_lora`'s other terms (source CE, MK-MMD, the masked
    pseudo-label CE) are all batch means, so a sum here would make the
    configured weight depend on batch size. The two are therefore NOT
    numerically interchangeable -- do not compare their values across branches.

    Dividing each class's squared mass by that class's total mass across the
    batch (detached) is what rules out the degenerate answer: mapping every
    sample onto one class scores far worse than spreading them out.
    """
    probs = _probs_fp32(logits)
    epsilon = 1e-8
    column_mass = probs.sum(dim=0, keepdim=True).detach()
    return (1 - (probs ** 2 / (column_mass + epsilon)).sum(dim=-1)).mean()


DIVERSITY_LOSSES = {"im": information_maximization, "gini": balanced_gini}


def diversity_loss(logits, kind: str):
    """Dispatch to the diversity term named by `branch_lora.diversity.kind`."""
    return DIVERSITY_LOSSES[kind](logits)
