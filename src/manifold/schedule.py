"""Unmasking schedule for the denoising loop.

All math in fp32 (MLX/MPS have no float64). Tie-breaking is deterministic
(highest confidence first, then lowest index) so runs are reproducible and
the test suite is stable.
"""
from __future__ import annotations

import numpy as np


def num_transfer_per_step(mask_count: int, steps: int) -> list[int]:
    """How many masked tokens to commit at each step.

    Even split with the remainder front-loaded (commit slightly more early),
    matching the reference `get_num_transfer_tokens`. Sums to ``mask_count``.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    base, rem = divmod(mask_count, steps)
    return [base + 1 if i < rem else base for i in range(steps)]


def confidence_of_argmax(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy prediction + its softmax probability, per position.

    Returns ``(x0, conf)`` where ``x0[i]`` is argmax token at position ``i``
    and ``conf[i]`` is the softmax prob of that token (the "low_confidence"
    signal). Temperature-0 / greedy path.
    """
    logits = np.asarray(logits, dtype=np.float32)
    x0 = np.argmax(logits, axis=-1)
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    probs = e / e.sum(axis=-1, keepdims=True)
    conf = np.take_along_axis(probs, x0[..., None], axis=-1)[..., 0]
    return x0, conf.astype(np.float32)


def select_transfer(conf: np.ndarray, masked: np.ndarray, k: int) -> list[int]:
    """Pick the ``k`` highest-confidence positions among currently-masked ones.

    Ties break to the lower index. Returns highest-confidence-first. Never
    selects an unmasked position; returns fewer than ``k`` if fewer are masked.
    """
    if k <= 0:
        return []
    conf = np.asarray(conf, dtype=np.float32)
    masked = np.asarray(masked, dtype=bool)
    eligible = np.flatnonzero(masked).tolist()
    if not eligible:
        return []
    eligible.sort(key=lambda i: (-float(conf[i]), i))
    return eligible[:k]
