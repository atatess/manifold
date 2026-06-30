"""MLX forward backend for Stable-DiffCoder.

mlx-lm's top-level ``Model.__call__`` / ``LlamaModel.__call__`` hardcode a causal
mask (via ``create_attention_mask``) and expose no seam for a custom one. The
per-layer ``Attention`` *does* accept a ``mask``, and ``mx.fast.scaled_dot_product_attention``
natively supports a 4D additive mask. So we replicate ``LlamaModel.__call__`` and
thread our block-causal mask down to every layer.

The weights themselves are vanilla Llama (the diffusion behavior lives entirely
in the sampling loop), so ``mlx_lm.load`` loads the 4-bit checkpoint unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

DEFAULT_REPO = "hunterbown/Stable-DiffCoder-8B-Instruct-mlx-4Bit"
MASK_TOKEN_ID = 5  # <[MASK_TOKEN]>; verified against the tokenizer

# Local checkout (repo_root/models/stable-diffcoder-4bit) preferred over the hub.
_LOCAL = Path(__file__).resolve().parents[2] / "models" / "stable-diffcoder-4bit"


def default_model_path() -> str:
    return str(_LOCAL) if _LOCAL.exists() else DEFAULT_REPO


def load_model(repo: Optional[str] = None):
    """Load the 4-bit MLX checkpoint. Returns ``(model, tokenizer)``."""
    from mlx_lm import load

    return load(repo or default_model_path())


def forward_with_mask(model, input_ids: mx.array, attn_mask: mx.array) -> mx.array:
    """Llama forward with a caller-supplied *additive* attention mask.

    ``input_ids``: ``[B, L]`` ints. ``attn_mask``: additive ``[L, L]`` (or
    broadcastable to ``[B, n_heads, L, L]``); 0 where allowed, large-negative
    where blocked. Returns logits ``[B, L, vocab]`` over all positions.
    """
    lm = model.model  # the inner LlamaModel
    h = lm.embed_tokens(input_ids)
    mask = attn_mask.astype(h.dtype)
    for layer in lm.layers:
        h = layer(h, mask, cache=None)
    h = lm.norm(h)
    if model.args.tie_word_embeddings:
        return lm.embed_tokens.as_linear(h)
    return model.lm_head(h)


def make_forward_fn(model) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Adapt an MLX model to the sampler's ``ForwardFn`` (numpy in/out).

    Note: returns full ``[L, vocab]`` logits in fp32. For a single function body
    L is small; slicing to the active block before host transfer is a later
    perf optimization.
    """

    def forward_fn(seq: np.ndarray, attn_mask: np.ndarray) -> np.ndarray:
        ids = mx.array(np.asarray(seq, dtype=np.int32)[None])  # [1, L]
        m = mx.array(np.asarray(attn_mask, dtype=np.float32))  # [L, L]
        logits = forward_with_mask(model, ids, m)  # [1, L, vocab], bf16
        logits = logits[0].astype(mx.float32)  # numpy has no bf16; cast in MLX first
        mx.eval(logits)
        return np.array(logits)

    return forward_fn
