"""Phase 1 — controllable block-diffusion denoising on a single masked body.

Done when: we can run the loop end-to-end on the M2 Max, print the partial
output at each step, and demonstrate programmatic intervention between steps.

Usage:
  python scripts/phase1_denoise.py
  python scripts/phase1_denoise.py --gen-length 64 --instruction "Write is_prime(n)."
  python scripts/phase1_denoise.py --ban " True"   # demo: ban a token, show output changes
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np

from manifold.model_mlx import MASK_TOKEN_ID, load_model, make_forward_fn
from manifold.sampler import Intervention, denoise


def build_prompt(tokenizer, instruction: str):
    messages = [{"role": "user", "content": instruction}]
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return list(tokenizer.encode(text))
    except Exception:
        return list(tokenizer.encode(instruction + "\n"))


def render(seq, gen_start, tokenizer, mask_id):
    """Readable partial: committed code with masked holes dropped + a hole count."""
    gen = [int(t) for t in seq[gen_start:]]
    committed = [t for t in gen if t != mask_id]
    holes = sum(1 for t in gen if t == mask_id)
    text = tokenizer.decode(committed).replace("\n", "\\n")
    if len(text) > 90:
        text = text[:90] + "…"
    return holes, text


def run(forward_fn, prompt_ids, tokenizer, *, gen_length, block_length, steps_per_block,
        on_step=None, quiet=False):
    gen_start = len(prompt_ids)

    def printer(ctx):
        interv = on_step(ctx) if on_step else None
        if not quiet:
            holes, text = render(ctx.seq, gen_start, tokenizer, MASK_TOKEN_ID)
            print(f"  step {ctx.step:3d} | block {ctx.block:2d} | holes {holes:3d} | {text}")
        return interv

    out = denoise(
        forward_fn, prompt_ids, gen_length,
        block_length=block_length, steps_per_block=steps_per_block,
        mask_id=MASK_TOKEN_ID, on_step=printer,
    )
    return tokenizer.decode([int(t) for t in out[gen_start:] if int(t) != MASK_TOKEN_ID])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="Write a Python function is_even(n) that returns True if n is even.")
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=4)
    ap.add_argument("--steps-per-block", type=int, default=4)
    ap.add_argument("--ban", default=None, help="demo intervention: ban this token text")
    args = ap.parse_args()

    print("Loading model...")
    model, tokenizer = load_model()
    forward_fn = make_forward_fn(model)
    prompt_ids = build_prompt(tokenizer, args.instruction)
    print(f"prompt: {len(prompt_ids)} tokens | gen_length {args.gen_length} "
          f"| block {args.block_length} x {args.steps_per_block} steps\n")

    print("=== denoise (streaming partial output per step) ===")
    plain = run(forward_fn, prompt_ids, tokenizer,
                gen_length=args.gen_length, block_length=args.block_length,
                steps_per_block=args.steps_per_block)
    print("\nFINAL:\n" + plain)

    if args.ban:
        ban_ids = set(tokenizer.encode(args.ban))
        print(f"\n=== intervention demo: banning {args.ban!r} (token ids {sorted(ban_ids)}) ===")
        VOCAB = None

        def ban_hook(ctx):
            nonlocal VOCAB
            if VOCAB is None:
                VOCAB = ctx.block_logits.shape[-1]
            bias = np.zeros(VOCAB, dtype=np.float32)
            for t in ban_ids:
                if 0 <= t < VOCAB:
                    bias[t] = -1e9
            # apply the same ban to every position in the active block
            return Intervention(logit_bias={i: bias for i in range(ctx.block_logits.shape[0])})

        banned = run(forward_fn, prompt_ids, tokenizer,
                     gen_length=args.gen_length, block_length=args.block_length,
                     steps_per_block=args.steps_per_block, on_step=ban_hook, quiet=True)
        print("\nFINAL (with ban):\n" + banned)
        print(f"\nintervention changed output: {banned != plain}")


if __name__ == "__main__":
    main()
