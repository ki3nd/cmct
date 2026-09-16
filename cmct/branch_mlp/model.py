"""TransferNet: a pretrained backbone + a linear task head. This is
`branch_mlp`'s model -- named for its head, since the loss it trains against
is meant to be swappable; see `docs/design.md`.

CMKD runs under both backbone sources; what the source decides is where its
cross-modal reference comes from and how much of the loss survives:
  * "clip" -- `ClipBackbone` carries a cosine head, so the full CMKD loss runs
    (task + distill + reg) against the branch's own CLIP predictions.
  * "imagenet" -- `ImagenetBackbone` has no text encoder and so no cosine head.
    The reference is the LoRA branch's teacher instead, passed in by the
    caller, and `reg_loss` is dropped -- its two terms both read cosine logits
    this backbone cannot produce. What is left is task + distill. The caller
    adds a thresholded cross-teaching loss on top after warmup.

See NOTICE for this module's license terms.
"""

import copy

import torch
import torch.nn as nn

from .backbone import ClipBackbone, ImagenetBackbone
from .loss import CMKD


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
    elif classname.find('BatchNorm') != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
       m.eval()

class TransferNet(nn.Module):
    def __init__(self, prompts, *, model_name, source="clip", num_classes,
                 label_smoothing, lambdas, lamb_gamma, max_iter):
        super(TransferNet, self).__init__()
        # define the network
        # get the feature extractor and the pretrained head
        self.num_class = num_classes
        if source == "clip":
            self.base_network = ClipBackbone(prompts, model_name).cuda()
        else:
            self.base_network = ImagenetBackbone(model_name)
        self.teacher_model = copy.deepcopy(self.base_network)
        self.teacher_model.eval()

        # define the task head
        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.base_network.output_num),
            nn.LayerNorm(self.base_network.output_num, eps=1e-6),
            nn.Linear(self.base_network.output_num, self.num_class,bias=False))
        self.classifier_layer.apply(weights_init_classifier)

        # define the loss functions
        self.cmkd = CMKD(lambdas=lambdas, lamb_gamma=lamb_gamma, max_iter=max_iter)
        self.clf_loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, source, target_img, source_label, *,
                self_ref_logit_clip=None, own_pred_target_img=None,
                need_target_logits=True):
        # need_target_logits is read ONLY on the no-cosine-head path, and it is
        # about memory, not about compute. There, the target forward exists
        # solely to feed losses the CALLER applies (cross-teaching) plus CMKD
        # here; when branch_lora is off there is no reference for either, both
        # are zero, and nothing backward() traverses reaches this graph -- a
        # whole backbone pass would stay resident for the rest of the
        # macro-step. Returning None instead is what keeps that step's
        # footprint honest.
        #
        # The CLIP path ignores it: there `target_logits` feeds a loss computed
        # right here unconditionally, so it is never optional.
        if not self.base_network.has_cosine_head:
            # No cosine head, so no source/target cosine logits and no
            # reg_loss. CMKD still runs, against the reference the caller
            # supplies (the LoRA branch's teacher); without one there is no
            # target-side signal this branch can compute at all.
            #
            # fix_bn is deliberately NOT applied: it exists to hold a
            # CLIP-pretrained backbone's statistics still, and freezing an
            # ImageNet ResNet's BN to source-domain statistics is the opposite
            # of what this branch needs -- its BN should adapt to the target.
            source_logits = self.classifier_layer(self.base_network.forward_features(source))
            clf_loss = self.clf_loss(source_logits, source_label)
            zero = torch.zeros((), device=source_logits.device)
            if not need_target_logits:
                return clf_loss, zero, None
            # `target_img` is the weak view the reference was computed on;
            # `own_pred_target_img` the harder one the classifier predicts from
            # when data.strong_aug is set. Only one forward either way.
            own_img = target_img if own_pred_target_img is None else own_pred_target_img
            target_logits = self.classifier_layer(self.base_network.forward_features(own_img))
            if self_ref_logit_clip is None:
                return clf_loss, zero, target_logits
            transfer_loss = self.cmkd(target_logits, None, None, None,
                                      self_ref_logit_clip=self_ref_logit_clip)
            return clf_loss, transfer_loss, target_logits

        self.base_network.apply(fix_bn)
        source = self.base_network.forward_features(source)

        # calculate source classification loss Lclf
        source_logits = self.classifier_layer(source)
        clf_loss = self.clf_loss(source_logits, source_label)

        source_logits_clip = self.base_network.forward_head(source)
        target = self.base_network.forward_features(target_img)

        # calculate calibrated probability alignment loss Lcpa
        target_clip_logits = self.base_network.forward_head(target)
        # own_pred_target_img: the classifier's OWN prediction (fed into
        # self+cross loss) can come from a DIFFERENT (harder-augmented) view
        # than the one target_clip_logits/reg_loss above use -- an EXTRA full
        # backbone forward pass when provided (gradient-carrying, since this
        # needs to train the classifier). Defaults to None, i.e. a single-view
        # forward (own prediction from the SAME target as everything else
        # here).
        own_feat = target if own_pred_target_img is None else self.base_network.forward_features(own_pred_target_img)
        target_logits = self.classifier_layer(own_feat)

        # calculate calibrated gini impurity loss Lcgi
        transfer_loss = self.cmkd(target_logits, target_clip_logits, source_logits_clip, source_label,
                                   self_ref_logit_clip=self_ref_logit_clip)

        return clf_loss, transfer_loss, target_logits

    def get_parameters(self, initial_lr=1.0, classifier_lr_mult=1.0):
        params=[
            {'params': self.base_network.trainable_parameters(), 'lr': initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': classifier_lr_mult * initial_lr}
]
        return params

    def predict(self, x):
        features = self.base_network.forward_features(x)
        logit = self.classifier_layer(features)
        return logit
