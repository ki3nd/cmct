"""CMKD, the cross-modal knowledge distillation self-training loss used by
`branch_mlp`.

See NOTICE for this module's license terms.
"""

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class LambdaScheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaScheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)


class CMKD(nn.Module):
    def __init__(self, lambdas, lamb_gamma, max_iter):
        """`lambdas` carries lambda1/lambda2/lambda3 as `task`/`source_ce`/
        `target_gini` (a mapping or any object with those attributes)."""
        super().__init__()
        self.lamb = LambdaScheduler(gamma=lamb_gamma, max_iter=max_iter)
        self.lambda1, self.lambda2, self.lambda3 = _unpack_lambdas(lambdas)

    def calibrated_coefficient(self, pred, pred_pretrained):
        distance = F.kl_div(pred.log(), pred_pretrained, reduction='none').sum(-1)
        coe = torch.exp(-distance).detach()
        return coe

    def gini_impurity(self,pred,coe=1.0):
        sum_dim = torch.sum(pred, dim=0).unsqueeze(dim=0).detach()
        return torch.sum(coe * (1 - torch.sum(pred ** 2 / sum_dim, dim=-1)))

    def regularization_term(self, target_pred_clip, source_logit_clip, source_label,lamb):
        return self.lambda2*F.cross_entropy(source_logit_clip, source_label) + \
            self.lambda3*lamb*self.gini_impurity(target_pred_clip)

    def forward(self, target_logit, target_logit_clip, source_logit_clip, source_label,
                self_ref_logit_clip=None):
        # self_ref_logit_clip: optional, SEPARATE reference for the self-
        # consistency terms (coe/mix) only -- reg_loss below always uses
        # target_pred_clip (from target_logit_clip), so it keeps training
        # the LIVE cosine branch regardless of what self-reference is used.
        # Defaults to target_logit_clip itself, i.e. the original behavior
        # (own live cosine branch as its own self-reference) when omitted.
        target_pred = F.softmax(target_logit, dim=1)
        target_pred_clip = F.softmax(target_logit_clip,dim=-1)
        self_ref_clip = target_logit_clip if self_ref_logit_clip is None else self_ref_logit_clip
        target_pred_self_ref = F.softmax(self_ref_clip, dim=-1)
        coe = self.calibrated_coefficient(target_pred, target_pred_self_ref)
        target_pred_mix = 0.5*(target_pred+target_pred_self_ref.detach())
        lamb = self.lamb.lamb()
        task_loss = self.lambda1 * lamb * self.gini_impurity(target_pred,coe)
        distill_loss = self.lambda1 * lamb *self.gini_impurity(target_pred_mix,1-coe)
        reg_loss = self.regularization_term(target_pred_clip, source_logit_clip, source_label,lamb)
        self.lamb.step()
        return task_loss + distill_loss + reg_loss


def _unpack_lambdas(lambdas):
    """Read lambda1/lambda2/lambda3 out of a mapping or a config dataclass."""
    if isinstance(lambdas, dict):
        return lambdas["task"], lambdas["source_ce"], lambdas["target_gini"]
    return lambdas.task, lambdas.source_ce, lambdas.target_gini
