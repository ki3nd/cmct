"""LoRA for CLIP's attention projections.

Not standard LoRA. The initialization is an SVD split of the pretrained weight:
the top-r principal components become the trainable low-rank factor, and the
residual is written back into the frozen weight. So at step zero the delta is
NOT zero -- instead `frozen + scaling * B @ A` reconstructs the original weight,
and training moves the r directions the weight relies on most.

Three properties are load-bearing and each has a test:

1. `scaling = alpha / sqrt(rank)`, not the usual `alpha / rank`. Since rank
   varies with depth, so does scaling.
2. `lora_A` and `lora_B` are ALWAYS float32, whatever precision the rest of the
   network runs at. They are the tensors an optimizer and an EMA accumulate
   into, and in float16 an EMA at momentum 0.996 stops moving within ~100 steps
   (measured; see cmct.engine.ema).
3. Dropout applies only to the LoRA branch's input. The frozen path is computed
   before it, so the layer is `frozen(x) + scaling * (B @ A) @ dropout(x)`, not
   `(frozen + scaling * B @ A) @ dropout(x)`.

There is no merge/unmerge. The implementation this replaces folded the delta into
the frozen weight on `.eval()` and subtracted it on `.train()`, which meant
calling `.eval()` once baked the delta in and locked out every later update --
and forced its teacher to override `train()` to avoid exactly that. Here
`train()` and `eval()` only touch dropout.
"""
from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PARAM_NAMES = ("q", "k", "v")
"""The attention projections that can take a LoRA delta. out_proj never does:
the reference implementation's wrapper class defaults to including it, but both
of its trainer configs pass ('q', 'k', 'v'), and a class default is not a
configuration."""

POSITIONS: dict[str, list[int]] = {
    "all": list(range(12)),
    "top1": [11],
    "top3": [9, 10, 11],
    "bottom": [0, 1, 2, 3],
    "mid": [4, 5, 6, 7],
    "up": [8, 9, 10, 11],
    "half-up": [6, 7, 8, 9, 10, 11],
    "half-bottom": [0, 1, 2, 3, 4, 5],
}
"""One table for both encoders. The original kept two, differing only in that
one called the last block "top1" and the other "top"."""


def rank_at(block_index: int, rank: int, rank_ramp: list[int] | None) -> int:
    """Ascending rank: deeper blocks get a larger low-rank budget.

    A block whose index is >= rank_ramp[k] uses `rank * 2 ** (k + 1)`, scanning k
    from the end. With rank=2 and rank_ramp=[2, 4, 6, 8, 10] the 12 blocks get
    2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64.
    """
    if block_index < 0:
        raise ValueError(f"block_index must be >= 0, got {block_index}")
    if rank <= 0:
        raise ValueError(f"rank must be > 0, got {rank}")
    if not rank_ramp:
        return rank
    for k, start in reversed(list(enumerate(rank_ramp))):
        if block_index >= start:
            return rank * 2 ** (k + 1)
    return rank


def scaling_for(rank: int, alpha: float) -> float:
    """alpha / sqrt(rank).

    NOT alpha / rank: the implementation this reproduces has that line commented
    out in favour of the square root, and the two differ by sqrt(rank) -- a
    factor of 8 at rank 64.
    """
    if rank <= 0:
        raise ValueError(f"rank must be > 0, got {rank}")
    return alpha / math.sqrt(rank)


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank delta.

    `weight` holds the SVD residual and never receives a gradient. `lora_A` and
    `lora_B` hold the top-r components and are always float32.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float,
                 param_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = scaling_for(rank, alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        weight = linear.weight.detach().to(torch.float32)
        bias = None if linear.bias is None else linear.bias.detach()

        # Top-r principal components. The original exposes a `first_eigen` field
        # defaulting to True, i.e. principal rather than minor components, and
        # nothing ever sets it False.
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        u, s, vh = u[:, :rank], s[:rank], vh[:rank, :]
        # The scaling is divided out here and multiplied back in the forward, so
        # the factors themselves carry no scaling.
        sqrt_s = torch.diag(torch.sqrt(s / self.scaling))

        self.lora_A = nn.Parameter((sqrt_s @ vh).to(torch.float32))
        self.lora_B = nn.Parameter((u @ sqrt_s).to(torch.float32))

        residual = weight - self.scaling * (self.lora_B.detach() @ self.lora_A.detach())
        self.weight = nn.Parameter(residual.to(param_dtype), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.to(param_dtype),
                                                          requires_grad=False)

    def delta(self) -> Tensor:
        """scaling * B @ A, in float32. `weight + delta()` reconstructs the
        original weight to the precision of `param_dtype`."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: Tensor) -> Tensor:
        frozen = F.linear(x, self.weight, self.bias)
        lora_input = self.dropout(x) if self.dropout is not None else x
        # Cast the delta down to the activation dtype rather than letting the
        # float32 factors promote the whole network's activations.
        return frozen + F.linear(lora_input, self.delta().to(x.dtype))

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
                f"scaling={self.scaling:.4f}, dtype={self.weight.dtype}")


class MultiheadAttentionLoRA(nn.Module):
    """nn.MultiheadAttention with its fused projection split into q, k, v and
    out, so a LoRA delta can be attached per projection.

    Splitting is what makes per-projection LoRA possible at all: the original
    module keeps one `in_proj_weight` of shape [3 * embed_dim, embed_dim].
    """

    def __init__(self, mha: nn.MultiheadAttention, rank: int, alpha: float,
                 dropout: float, params: tuple[str, ...] = ("q", "k", "v"),
                 param_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        unknown = sorted(set(params) - set(PARAM_NAMES))
        if unknown:
            raise ValueError(f"unknown LoRA targets {unknown}; valid: {list(PARAM_NAMES)}")
        if not params:
            raise ValueError("params must name at least one projection")

        dim = mha.embed_dim
        self.embed_dim = dim
        self.num_heads = mha.num_heads
        self.head_dim = mha.head_dim
        self.batch_first = mha.batch_first

        has_bias = mha.in_proj_bias is not None
        weight = mha.in_proj_weight.detach()
        parts = {"q": weight[:dim], "k": weight[dim:2 * dim], "v": weight[2 * dim:]}
        biases = {"q": None, "k": None, "v": None}
        if has_bias:
            b = mha.in_proj_bias.detach()
            biases = {"q": b[:dim], "k": b[dim:2 * dim], "v": b[2 * dim:]}

        built: dict[str, nn.Module] = {}
        for name in ("q", "k", "v"):
            linear = nn.Linear(dim, dim, bias=has_bias)
            with torch.no_grad():
                linear.weight.copy_(parts[name])
                if has_bias:
                    linear.bias.copy_(biases[name])
            built[name] = self._maybe_lora(linear, name in params, rank, alpha,
                                           dropout, param_dtype)
        out = nn.Linear(dim, dim, bias=mha.out_proj.bias is not None)
        with torch.no_grad():
            out.weight.copy_(mha.out_proj.weight.detach())
            if out.bias is not None:
                out.bias.copy_(mha.out_proj.bias.detach())
        built["o"] = self._maybe_lora(out, False, rank, alpha, dropout, param_dtype)

        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (
            built["q"], built["k"], built["v"], built["o"]
        )

    @staticmethod
    def _maybe_lora(linear: nn.Linear, wanted: bool, rank: int, alpha: float,
                    dropout: float, param_dtype: torch.dtype) -> nn.Module:
        if wanted:
            return LoRALinear(linear, rank, alpha, dropout, param_dtype)
        linear.weight.requires_grad_(False)
        linear.weight.data = linear.weight.data.to(param_dtype)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)
            linear.bias.data = linear.bias.data.to(param_dtype)
        return linear

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                need_weights: bool = False, attn_mask: Tensor | None = None,
                key_padding_mask: Tensor | None = None
                ) -> tuple[Tensor, Tensor | None]:
        """Same signature and return shape as nn.MultiheadAttention, since it
        stands in for one inside CLIP's residual blocks. Sequence-first
        ([L, N, E]) unless the wrapped module was batch_first."""
        if self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))

        length, batch, _ = query.shape
        source = key.shape[0]

        def project(proj: nn.Module, x: Tensor, n: int) -> Tensor:
            """[L, N, E] -> [N, H, L, head_dim].

            FOUR dimensions, not the flattened [N*H, L, head_dim]. Both shapes
            compute the same attention, but scaled_dot_product_attention reads a
            3-D input as having no head dimension and falls back to its `math`
            backend, which MATERIALIZES the [N*H, L, L] attention matrix and
            saves it for backward. The 4-D layout lets it choose the flash or
            memory-efficient backend, which saves nothing of that size. On
            ViT-B/16 the difference is 9.4% of everything a visual forward
            retains -- and branch 1 holds three such forwards at once. The
            reference uses the 4-D layout (loralib/layers.py:651-653).
            """
            out = proj(x)
            return out.view(n, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        q = project(self.q_proj, query, length)
        k = project(self.k_proj, key, source)
        v = project(self.v_proj, value, source)

        mask = attn_mask
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask is not None and mask.dim() == 3:
            mask = mask.view(batch, self.num_heads, length, source)
        if key_padding_mask is not None:
            padding = key_padding_mask.view(batch, 1, 1, source)
            mask = padding if mask is None else mask + padding

        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = attended.permute(2, 0, 1, 3).contiguous().view(length, batch, self.embed_dim)
        output = self.out_proj(attended)

        if self.batch_first:
            output = output.transpose(0, 1)
        return output, None if not need_weights else torch.empty(0)


def _resblocks(model: nn.Module, encoder: str) -> nn.Sequential:
    if encoder == "vision":
        return model.visual.transformer.resblocks
    if encoder == "text":
        return model.transformer.resblocks
    raise ValueError(f"encoder must be 'vision' or 'text', got {encoder!r}")


def apply_lora(model: nn.Module, *, rank: int, alpha: float, dropout: float,
               params: tuple[str, ...] = ("q", "k", "v"),
               rank_ramp: list[int] | None = None, positions: str = "all",
               encoders: tuple[str, ...] = ("text", "vision"),
               param_dtype: torch.dtype = torch.float32
               ) -> list[MultiheadAttentionLoRA]:
    """Replace the attention module of the selected residual blocks, in place.

    Returns the modules it created, in the order visited.
    """
    if positions not in POSITIONS:
        raise ValueError(f"unknown positions {positions!r}; valid: {sorted(POSITIONS)}")
    indices = POSITIONS[positions]
    created: list[MultiheadAttentionLoRA] = []

    for encoder in encoders:
        blocks = _resblocks(model, encoder)
        for index, block in enumerate(blocks):
            if index not in indices:
                continue
            if not isinstance(block.attn, nn.MultiheadAttention):
                raise TypeError(
                    f"{encoder} block {index}: expected nn.MultiheadAttention, "
                    f"got {type(block.attn).__name__} -- already wrapped?"
                )
            replacement = MultiheadAttentionLoRA(
                block.attn, rank=rank_at(index, rank, rank_ramp), alpha=alpha,
                dropout=dropout, params=params, param_dtype=param_dtype,
            )
            block.attn = replacement
            created.append(replacement)

    if not created:
        raise ValueError(f"positions {positions!r} selected no blocks")
    return created


def lora_parameters(model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Every parameter whose name contains 'lora_'. These are the trainable ones,
    and the ones an EMA moves."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            yield name, param


def freeze_except_lora(model: nn.Module) -> None:
    """Only LoRA factors train. Everything else, including the SVD residual, is
    frozen."""
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state: dict[str, Tensor],
                         strict: bool = True) -> None:
    """Load only the LoRA factors, leaving the frozen weights untouched."""
    own = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    if strict:
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if missing or unexpected:
            raise ValueError(
                f"LoRA state mismatch: missing {missing}, unexpected {unexpected}"
            )
    with torch.no_grad():
        for key, value in state.items():
            if key not in own:
                continue
            if own[key].shape != value.shape:
                raise ValueError(
                    f"{key}: shape {tuple(value.shape)} does not match "
                    f"{tuple(own[key].shape)} -- different rank?"
                )
            own[key].copy_(value)
