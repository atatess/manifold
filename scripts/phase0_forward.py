"""Phase 0 — runnable forward pass.

Done when: one forward pass over a masked code snippet returns sane logits at
acceptable latency on the M2 Max.

This script:
  1. loads the 4-bit MLX checkpoint,
  2. verifies the mask token id (5) maps to <[MASK_TOKEN]>,
  3. builds a prompt + a masked region, runs ONE forward with our block-causal
     additive mask,
  4. checks the logits are finite (no NaN/Inf — the bf16 -inf gotcha),
  5. prints latency, peak memory, and the top predictions at masked positions.
"""
import time

import _bootstrap  # noqa: F401
import mlx.core as mx
import numpy as np

from manifold.masks import build_block_causal_mask
from manifold.model_mlx import MASK_TOKEN_ID, forward_with_mask, load_model


def peak_mem_gb() -> float:
    for fn in ("get_peak_memory",):
        if hasattr(mx, fn):
            return getattr(mx, fn)() / 1e9
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory() / 1e9
    return float("nan")


def main():
    print("Loading model (4-bit MLX)...")
    t0 = time.perf_counter()
    model, tokenizer = load_model()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s | weights peak {peak_mem_gb():.2f} GB")

    # 1. Verify the mask token id.
    decoded = tokenizer.decode([MASK_TOKEN_ID])
    print(f"\nmask id {MASK_TOKEN_ID} decodes to: {decoded!r}")

    # 2. Build prompt + masked region.
    prompt_text = "def is_even(n):\n    return "
    prompt_ids = tokenizer.encode(prompt_text)
    gen_length = 8  # one block-aligned masked region
    seq = np.array(list(prompt_ids) + [MASK_TOKEN_ID] * gen_length, dtype=np.int64)
    L = len(seq)
    block_length = 4
    mask = build_block_causal_mask(L, block_length)

    # 3. One forward (timed; mx is lazy so eval to force compute).
    ids_mx = mx.array(seq[None].astype(np.int32))
    mask_mx = mx.array(mask)
    forward_with_mask(model, ids_mx, mask_mx)  # warmup (kernel compile)
    mx.eval(model.parameters())
    t0 = time.perf_counter()
    logits = forward_with_mask(model, ids_mx, mask_mx)
    mx.eval(logits)
    dt = time.perf_counter() - t0

    logits_np = np.array(logits[0].astype(mx.float32))  # numpy has no bf16
    print(f"\nforward: seq_len={L}  logits shape={logits_np.shape}  dtype={logits.dtype}")
    print(f"  latency {dt * 1000:.0f} ms | peak {peak_mem_gb():.2f} GB")

    # 4. Finiteness (the bf16 -inf -> NaN gotcha).
    finite = np.isfinite(logits_np).all()
    print(f"  all finite: {finite}  (NaN={np.isnan(logits_np).any()} Inf={np.isinf(logits_np).any()})")

    # 5. Top predictions at masked positions.
    gen_start = len(prompt_ids)
    print("\ntop-5 predictions at masked positions:")
    for i in range(gen_length):
        row = logits_np[gen_start + i]
        top = np.argsort(-row)[:5]
        toks = [repr(tokenizer.decode([int(t)])) for t in top]
        print(f"  pos {i}: {', '.join(toks)}")

    print("\nPhase 0 check:", "PASS" if finite and logits_np.shape[0] == L else "FAIL")


if __name__ == "__main__":
    main()
