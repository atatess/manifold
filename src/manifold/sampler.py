"""The controllable block-diffusion denoising loop — the heart of Manifold.

Design goals:
- **Runtime-agnostic.** The forward pass is an injected dependency
  ``forward_fn(seq_ids, attn_mask) -> logits`` (numpy in/out). The MLX backend
  and a fake test model both satisfy it.
- **Pausable / inspectable / steerable.** After each forward, before committing,
  an ``on_step`` hook receives the full partial sequence and the block's logits,
  and may return an :class:`Intervention` to bias logits, forbid unmasking
  positions this step, or re-mask already-committed positions. Type-checker
  feedback (Phase 2) is one implementation of ``on_step`` — never a post-hoc lint.

The schedule mirrors Stable-DiffCoder's ``generate_block``: block-by-block,
left-to-right, ``block_length`` per block, ``steps_per_block`` denoising steps,
committing the highest-confidence masked positions each step. The forward is run
over the *whole* sequence each step (no KV cache); the block-causal mask makes
later blocks invisible to earlier ones, so this is correct, just not yet optimized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .masks import build_block_causal_mask
from .schedule import confidence_of_argmax, num_transfer_per_step, select_transfer

# (seq_ids[L], attn_mask[L, L]) -> logits[L, vocab]
ForwardFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class StepContext:
    """Snapshot handed to the ``on_step`` hook, *after* the forward, *before* commit."""

    step: int                       # global step index (0-based across all blocks)
    block: int                      # current block index (0-based)
    block_span: tuple[int, int]     # absolute [start, end) of the active block
    gen_span: tuple[int, int]       # absolute [start, end) of the whole edit region
    mask_id: int
    seq: np.ndarray                 # full current sequence; masked positions == mask_id
    block_logits: np.ndarray        # [block_len, vocab] logits for the active block
    x0: np.ndarray                  # [block_len] greedy predictions
    conf: np.ndarray                # [block_len] confidence (softmax prob of argmax)
    planned_k: int                  # tokens to be committed this step (before forbid)


@dataclass
class Intervention:
    """Returned by ``on_step`` to steer the *current* step (and beyond)."""

    # block-local index -> additive per-vocab bias applied to that position's logits
    logit_bias: Optional[dict[int, np.ndarray]] = None
    # absolute positions to set back to mask_id (un-commit); affects future steps
    remask: Optional[list[int]] = None
    # absolute positions to keep masked this step (deferred, not committed now)
    forbid_unmask: Optional[list[int]] = None


OnStep = Callable[[StepContext], Optional[Intervention]]


def denoise(
    forward_fn: ForwardFn,
    prompt_ids,
    gen_length: int,
    *,
    block_length: int = 4,
    steps_per_block: int = 4,
    mask_id: int = 5,
    on_step: Optional[OnStep] = None,
    mask_builder: Callable[[int, int], np.ndarray] = build_block_causal_mask,
) -> np.ndarray:
    """Denoise ``gen_length`` masked tokens appended after ``prompt_ids``.

    Returns the full token sequence (prompt + denoised region). The decoded edit
    is ``result[len(prompt_ids):]``.
    """
    prompt = [int(t) for t in prompt_ids]
    gen_start = len(prompt)
    gen_end = gen_start + gen_length
    seq = np.array(prompt + [mask_id] * gen_length, dtype=np.int64)
    seq_len = seq.shape[0]
    num_blocks = (gen_length + block_length - 1) // block_length

    global_step = 0
    for block in range(num_blocks):
        bstart = gen_start + block * block_length
        bend = min(bstart + block_length, gen_end)
        block_len = bend - bstart
        counts = num_transfer_per_step(block_len, steps_per_block)

        for k in counts:
            attn_mask = mask_builder(seq_len, block_length)
            logits = np.asarray(forward_fn(seq, attn_mask), dtype=np.float32)
            block_logits = np.array(logits[bstart:bend], dtype=np.float32)
            x0, conf = confidence_of_argmax(block_logits)

            interv = None
            if on_step is not None:
                interv = on_step(
                    StepContext(
                        step=global_step,
                        block=block,
                        block_span=(bstart, bend),
                        gen_span=(gen_start, gen_end),
                        mask_id=mask_id,
                        seq=seq,
                        block_logits=block_logits,
                        x0=x0,
                        conf=conf,
                        planned_k=k,
                    )
                )

            if interv is not None:
                if interv.logit_bias:
                    for local_idx, bias in interv.logit_bias.items():
                        block_logits[local_idx] = block_logits[local_idx] + np.asarray(
                            bias, dtype=np.float32
                        )
                    x0, conf = confidence_of_argmax(block_logits)
                if interv.remask:
                    for pos in interv.remask:
                        seq[pos] = mask_id

            masked_block = seq[bstart:bend] == mask_id
            if interv is not None and interv.forbid_unmask:
                for pos in interv.forbid_unmask:
                    local = pos - bstart
                    if 0 <= local < block_len:
                        masked_block[local] = False

            for local in select_transfer(conf, masked_block, k):
                seq[bstart + local] = x0[local]

            global_step += 1

    return seq
