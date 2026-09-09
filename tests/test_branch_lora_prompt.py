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

from cmct.branch_lora.model import LoraCLIP, Simple_TextEncoder
from cmct.branch_lora.prompt import CTX_PARAM_NAME, PromptLearner
from cmct.clip import clip

CLASSNAMES = ["alarm clock", "fan", "desk lamp", "postit notes"]
TEMPLATE = "a photo of a {}."


def test_ctx_at_init_reproduces_the_template_text_features(clip_model):
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


def test_the_context_is_the_only_trainable_text_parameter(clip_model):
    model = LoraCLIP(CLASSNAMES, clip_model, template=TEMPLATE, n_ctx=4, learnable=True)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == [CTX_PARAM_NAME]
    assert model.prompt_learner.ctx.shape == (4, clip_model.ln_final.weight.shape[0])


def test_a_disabled_context_is_frozen(clip_model):
    model = LoraCLIP(CLASSNAMES, clip_model, template=TEMPLATE, n_ctx=4, learnable=False)
    assert model.prompt_learner.ctx.requires_grad is False


def test_n_ctx_mismatched_with_the_prefix_token_count_raises(clip_model):
    """"a photo of a" is exactly 4 tokens. n_ctx=8 must not silently pad the
    context with untrained noise -- that would freeze noise into the prompt
    under prompt.enabled: false instead of reproducing the template."""
    with pytest.raises(ValueError):
        PromptLearner(CLASSNAMES, clip_model, n_ctx=8, template=TEMPLATE, learnable=True)


def test_a_gradient_reaches_the_context(clip_model):
    """requires_grad is not the same as a gradient actually arriving: this
    fails if a future refactor detaches the expand/cat path in
    PromptLearner.forward or otherwise rebuilds the context in a way that
    severs it from the text encoder's output."""
    model = LoraCLIP(CLASSNAMES, clip_model, template=TEMPLATE, n_ctx=4, learnable=True)

    model.text_features().sum().backward()

    ctx_grad = model.prompt_learner.ctx.grad
    assert ctx_grad is not None
    assert ctx_grad.norm().item() > 0
