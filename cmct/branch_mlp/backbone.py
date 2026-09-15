"""CLIP backbone for `branch_mlp`.

See NOTICE for this module's license terms.

The prompt list is passed in by `cmct/train.py`, which reads it from
`cmct/prompts.py` and hands the SAME list to branch_lora's text encoder.

This backbone pins itself to the current CUDA device via explicit `.cuda()`
calls; it has no CPU code path.
"""

import torch.nn as nn

from cmct import clip


class ClipBackbone(nn.Module):
    def __init__(self, prompts, model_name):
        super(ClipBackbone, self).__init__()
        # `model_name` is a CLIP checkpoint name, and any CLIP backbone works:
        # this branch only ever reads features through `encode_image`, so there
        # is no architecture-specific plumbing to keep in step.
        model, _preprocess = clip.load(model_name, device="cuda")
        # The classifier head's input width, taken from the backbone rather than
        # tabulated per checkpoint, so adding one needs no edit here. This is
        # `encode_image`'s output width -- the joint image-text embedding, after
        # the projection -- which is what `forward_features` returns and what
        # `forward_head` has to match against the text embedding. 512 for
        # ViT-B/16 and RN101, 1024 for RN50.
        self.output_num = model.visual.output_dim
        # This branch's backbone must be fp32, and getting that wrong is a
        # silent-until-it-crashes hazard. `cmct.clip`'s `build_model` downcasts
        # the model to fp16 (see `convert_weights` in `cmct/clip/model.py`),
        # but `classifier_layer` (in `cmct/branch_mlp/model.py`) is fp32, so
        # `LayerNorm` raises "expected scalar type Half but found Float" on
        # the first forward unless the backbone is explicitly widened back to
        # fp32 here.
        model = model.float()
        class_list = prompts
        self.model = model
        self.text = clip.tokenize(class_list).cuda()
        text_features = self.encode_text().detach().cuda()
        self.text_features = text_features / text_features.norm(dim=1, keepdim=True)

    def forward_features(self, x):
        feature = self.model.encode_image(x)
        return feature

    def encode_text(self):
        text_features = self.model.encode_text(self.text)
        return text_features

    def forward_head(self,image_features, return_text_logit=False):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        if return_text_logit:
            return logits_per_image,logits_per_text
        else:
            return logits_per_image

    def forward(self, x):
        image_features = self.forward_features(x)
        logits_per_image = self.forward_head(image_features)

        return logits_per_image
