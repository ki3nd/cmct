"""TransferNet: a pretrained backbone + a linear task head. This is
`branch_mlp`'s model -- named for its head, since the loss it trains against
is meant to be swappable; see `docs/design.md`.

Two backbone sources, and the source decides the loss:
  * "clip" -- `ClipBackbone`, which carries a cosine head, so the CMKD loss
    runs and supplies the branch's own target-side signal.
  * "imagenet" -- `ImagenetBackbone`, which has no text encoder and therefore
    no cosine head. CMKD cannot run at all there (its three cross-modal terms
    all read cosine logits), so this returns a zero transfer loss and the
    training loop supplies every target-side signal as thresholded
    pseudo-labels instead.

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
                 label_smoothing, lambdas=None, lamb_gamma=None, max_iter):
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
        self.cmkd = (
            CMKD(lambdas=lambdas, lamb_gamma=lamb_gamma, max_iter=max_iter)
            if self.base_network.has_cosine_head else None
        )
        self.clf_loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, source, target_img, source_label, *,
                self_ref_logit_clip=None, own_pred_target_img=None,
                need_target_logits=True):
        # need_target_logits is read ONLY on the no-cosine-head path, and it is
        # about memory, not about compute. There, the target forward exists
        # solely to feed the caller's pseudo-label losses; during that branch's
        # warmup those are both zero, so nothing backward() traverses reaches
        # this graph and its saved tensors are never freed -- a whole backbone
        # pass stays resident for the rest of the macro-step, on top of
        # branch_lora's own peak. Returning None instead is what keeps the
        # warmup step's footprint honest.
        #
        # The CMKD path ignores it: there `target_logits` feeds the loss
        # computed right here, so it is never optional.
        if self.cmkd is None:
            # No cosine head: source CE is the only loss computable here. Every
            # target-side signal (self and cross pseudo-labels) is applied by
            # the caller, so `target_img` is unused -- the single target pass
            # below is the classifier's own prediction, on the strong view when
            # there is one.
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
            own_img = target_img if own_pred_target_img is None else own_pred_target_img
            target_logits = self.classifier_layer(self.base_network.forward_features(own_img))
            return clf_loss, zero, target_logits

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
