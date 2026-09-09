"""Guards for branch_lora's learnable prompt context.

The first test is the important one. `Simple_TextEncoder` reads its output
feature at `tokenized_prompts.argmax(dim=-1)`, the EOT position. Building the
embedding sequence by hand makes it possible to get that index wrong, and a
wrong index does NOT crash -- it silently reads a different token's features,
producing garbage text embeddings while the loss keeps looking healthy. Class
names tokenize to different lengths ("fan" is one token, "alarm clock" two), so
EOT sits at a different index for every class and an off-by-one is easy.

The guard: with the context at its initial value, the whole path must reproduce
exactly what the old hand-written-template path produced.
"""

import pytest
import torch

from cmct.branch_lora.model import LoraCLIP, Simple_TextEncoder, load_clip_to_cpu
from cmct.branch_lora.prompt import PromptLearner
from cmct.clip import clip

CLASSNAMES = ["alarm clock", "fan", "desk lamp", "postit notes"]
TEMPLATE = "a photo of a {}."


def test_ctx_at_init_reproduces_the_template_text_features(clip_weights):
    clip_model = load_clip_to_cpu("ViT-B/16", "./assets").float()

    # The old path: tokenize the literal template, embed, encode.
    encoder = Simple_TextEncoder(clip_model)
    ids = clip.tokenize([TEMPLATE.format(c) for c in CLASSNAMES])
    with torch.no_grad():
        embedded = clip_model.token_embedding(ids).type(encoder.dtype)
        expected = encoder(embedded, ids)
        expected = expected / expected.norm(dim=-1, keepdim=True)

    # The new path, through the actual production seam: LoraCLIP.text_features(),
    # not a hand-reconstructed call to text_encoder(). This is what would catch
    # text_features() passing the wrong (or a stale) id tensor to the encoder.
    model = LoraCLIP(CLASSNAMES, clip_model, template=TEMPLATE, n_ctx=4, learnable=True)
    with torch.no_grad():
        actual = model.text_features()

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_the_context_is_the_only_trainable_text_parameter(clip_weights):
    clip_model = load_clip_to_cpu("ViT-B/16", "./assets").float()
    model = LoraCLIP(CLASSNAMES, clip_model, template=TEMPLATE, n_ctx=4, learnable=True)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["prompt_learner.ctx"]
    assert model.prompt_learner.ctx.shape == (4, clip_model.ln_final.weight.shape[0])


def test_a_disabled_context_is_frozen(clip_weights):
    clip_model = load_clip_to_cpu("ViT-B/16", "./assets").float()
    model = LoraCLIP(CLASSNAMES, clip_model, template=TEMPLATE, n_ctx=4, learnable=False)
    assert model.prompt_learner.ctx.requires_grad is False


def test_n_ctx_mismatched_with_the_prefix_token_count_raises(clip_weights):
    """"a photo of a" is exactly 4 tokens. n_ctx=8 must not silently pad the
    context with untrained noise -- that would freeze noise into the prompt
    under prompt.enabled: false instead of reproducing the template."""
    clip_model = load_clip_to_cpu("ViT-B/16", "./assets").float()
    with pytest.raises(ValueError):
        PromptLearner(CLASSNAMES, clip_model, n_ctx=8, template=TEMPLATE, learnable=True)
