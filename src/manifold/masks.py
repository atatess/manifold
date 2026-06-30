"""Block-causal attention mask for block diffusion.

Bidirectional *within* a block, causal *across* blocks — the pattern
Stable-DiffCoder trains/decodes with. Equivalent to the reference's
``kron(tril(num_blocks), ones(block_size))``.

Returned as an *additive* mask (0.0 where attention is allowed, ``neg``
where blocked). ``neg`` is a large finite negative rather than ``-inf``:
``-inf`` in a bf16 softmax can produce NaNs on MLX/MPS.
"""
from __future__ import annotations

import numpy as np

# Large finite negative; safe for bf16/fp16 softmax (avoids -inf -> NaN).
NEG: float = -1e9


def build_block_causal_mask(
    seq_len: int, block_size: int, neg: float = NEG
) -> np.ndarray:
    """Additive ``(seq_len, seq_len)`` float32 mask.

    Position ``i`` (block ``i // block_size``) may attend to position ``j``
    iff ``j``'s block <= ``i``'s block. Within a block, attention is full
    (bidirectional).
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    num_blocks = (seq_len + block_size - 1) // block_size
    block = np.tril(np.ones((num_blocks, num_blocks), dtype=bool))
    local = np.ones((block_size, block_size), dtype=bool)
    allowed = np.kron(block, local)[:seq_len, :seq_len]

    mask = np.where(allowed, np.float32(0.0), np.float32(neg))
    return mask.astype(np.float32)
