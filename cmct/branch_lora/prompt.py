"""The learnable prompt context for branch_lora's text side.

CoOp-style: one context shared by every class, prepended to the class's own
fixed name tokens. This is the ONLY text-side adaptation when
`branch_lora.lora.text` is false -- the text transformer itself stays at frozen
CLIP weights.

The context is initialised from the token embeddings of `template`'s prefix, so
a context that never moves reproduces the hand-written template exactly. That
property is what makes `prompt.enabled: false` a usable ablation baseline, and
what the equivalence test in tests/test_branch_lora_prompt.py checks.
"""

import torch
import torch.nn as nn

from cmct.clip import clip


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, *, n_ctx, template, learnable):
        super().__init__()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # "a photo of a {}." -> "a photo of a"
        prefix_text = template.split("{}")[0].strip()
        prefix_ids = clip.tokenize(prefix_text)[0]
        # tokenize() wraps in SOT/EOT; strip both to get the prefix's own tokens.
        eot = int(prefix_ids.argmax())
        prefix_ids = prefix_ids[1:eot]

        with torch.no_grad():
            prefix_vectors = clip_model.token_embedding(prefix_ids).type(dtype)

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        take = min(n_ctx, prefix_vectors.shape[0])
        ctx_vectors[:take] = prefix_vectors[:take]
        self.ctx = nn.Parameter(ctx_vectors, requires_grad=learnable)

        # Each "X" is exactly one CLIP BPE token, so the placeholders occupy
        # positions 1..n_ctx and EOT lands where it would in the real prompt
        # (when n_ctx matches the prefix's token count). tokenized_prompts is
        # therefore a valid source for the EOT argmax even though the
        # embeddings fed to the transformer are not the ones it encodes.
        placeholder = " ".join(["X"] * n_ctx)
        prompts = [f"{placeholder} {name.replace('_', ' ')}." for name in classnames]
        tokenized_prompts = clip.tokenize(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])            # SOT
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])    # name, EOT, pad
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.n_ctx = n_ctx

    def forward(self):
        """The full embedding sequence, shape (n_cls, 77, ctx_dim)."""
        n_cls = self.token_prefix.shape[0]
        ctx = self.ctx.unsqueeze(0).expand(n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
