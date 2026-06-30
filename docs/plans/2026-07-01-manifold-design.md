# Manifold — Design (2026-07-01)

A VS Code extension that does **whole-region code edits using a diffusion LLM running fully on-device**, with **type-checker diagnostics fed back into the denoising loop** so the model corrects its own type errors as it refines, and support for **editing multiple regions jointly**.

This document records the validated kickoff design. It supersedes nothing in the brief; it operationalizes it.

## Decisions (locked at kickoff)

| Question | Decision | Why |
|---|---|---|
| Forward-pass runtime | **MLX-first**, adapting mlx-lm PR #270 (Dream/DiffuCoder) + 4-bit weights; PyTorch-**CPU** oracle for parity checks | 4.6 GB vs ~16.5 GB; MLX SDPA natively takes a 4D additive mask; a same-family MLX diffusion loop already exists; the PyTorch-MPS "correctness" edge is undercut by MPS having no float64 |
| First target language | **Python + pyright** | Checker runs in the Python server *in the loop*; pyright `--outputjson` gives clean ranges; no Node sidecar on day one |
| Transport | **stdio JSON-RPC**, added at Phase 3 only | Local, no ports, airplane-mode-friendly; one warm process keeps the 8B model loaded. Phases 0–2 are a Python lib + CLI, no transport |

Model: `ByteDance-Seed/Stable-DiffCoder-8B-Instruct` (MIT, arxiv 2601.15892). 4-bit MLX weights: `hunterbown/Stable-DiffCoder-8B-Instruct-mlx-4Bit`.

## The model, precisely

Vanilla Llama (hidden 4096, 32 layers, 32 heads, 8 KV heads/GQA, vocab 155136, rope_theta 5e5, rms_norm_eps 1e-6, ctx 8192). The diffusion behavior is entirely in the **sampler**, which ships as a small MIT-licensed reference (`generate_block` in the bundled `generation_utils.py`; the original ByteDance repo packs the same logic into `modeling_stable_diffcoder.py`). The paper is training-only; that reference code is the spec.

- **Masked region = token id 5** (`<[MASK_TOKEN]>`). The edit region is literally filled with id 5; each step predicts and commits some of those positions.
- **Block-by-block, left-to-right**, `block_length=4`; N denoising steps per block.
- **Attention is block-causal**: bidirectional *within* a block, causal *across* blocks — `kron(tril(num_blocks), ones(block_size))`. NOT vanilla causal.
- **Per-step commit:** `num_transfer = mask_count // steps` (remainder front-loaded); each step commits the top-k masked positions ranked by **confidence = softmax prob of the argmax token**. `temperature=0` → greedy; `remasking='low_confidence'`.
- The reference **never un-commits** a token. Re-masking a committed-but-wrong span is **our addition** — and it is the mechanism type feedback uses.

## Architecture

Two processes (the second arrives at Phase 3):

1. **Model server (Python).** Loads the model, runs our custom denoising loop with step-level hooks, runs pyright between steps. The loop is **runtime-agnostic**; the forward pass is a swappable dependency.
2. **VS Code extension (TypeScript).** UI only: select region(s), type an instruction, stream the live denoise into a preview, accept/reject diff. No model logic.

### Package layout (`src/manifold/`)

- `masks.py` — pure: block-causal additive mask builder. *(TDD)*
- `schedule.py` — pure: per-step transfer counts + confidence-based unmask selection. *(TDD)*
- `sampler.py` — the controllable denoising loop. Depends on an **injected `forward_fn(token_ids, mask) -> logits`**, so it is testable with a fake model and runtime-agnostic. Emits per-step state and calls an `on_step` hook that can intervene. *(TDD via fake forward)*
- `model_mlx.py` — the MLX forward backend: loads the 4-bit weights via mlx-lm and threads our additive mask through a copied `LlamaModel.__call__` (mlx-lm's top-level call hardcodes causal; the per-layer `Attention` accepts a mask).
- `typecheck.py` — pyright integration (Phase 2). Maps diagnostic char ranges → token positions → re-mask / logit-bias decisions.
- `rpc.py` — stdio JSON-RPC server (Phase 3).
- `cli.py` — run a denoise on a masked snippet; stream partial output per step.

### The hook point (heart of the project)

The loop is:

```
x = prompt + [MASK]*gen_length              # masked region = id 5
for block in blocks(left→right):
    for step in range(steps_per_block):
        logits = forward_fn(x, block_causal_mask(len(x)))   # ← single forward
        # ─── PAUSE / INSPECT / INTERVENE (on_step hook) ───
        #   • read current partial x (with masks)
        #   • run pyright on the current partial decode (every k steps)
        #   • bias logits, force/forbid unmasking positions, or RE-MASK
        #     committed positions implicated in type errors
        x0, transfer = select_unmask(logits, x, k=num_transfer[step])
        x[transfer] = x0[transfer]          # commit
```

`on_step` receives `(step_ctx)` and may return interventions (logit deltas, a re-mask set, a forced-keep-masked set). Type feedback is one implementation of `on_step` — never a post-hoc lint pass.

## M2 Max gotchas (already cost-avoided)

- **No float64 on MLX/MPS.** The reference confidence/Gumbel math is float64 → do it in fp32.
- **`-inf` in bf16 softmax can NaN.** Build the additive mask with a large finite negative (`-1e9`) or a boolean mask.
- **Don't copy PR #270's logit shift.** Stable-DiffCoder defaults to `shift=False`; Dream shifts. Verify against the reference.
- mlx-lm internals (`create_attention_mask`) are causal-only — we own a copied `LlamaModel.__call__` rather than monkeypatching.

## Build phases & "done when"

- **Phase 0** — MLX forward over a masked snippet returns sane logits at acceptable latency.
- **Phase 1** — controllable block-diffusion loop on one masked function body: print partial output each step; demonstrate programmatic intervention between steps. *(If step control can't expose cleanly: STOP and report.)*
- **Phase 2** — pyright in the loop: a body the naive denoise leaves with a type error, the type-aware loop measurably fixes. The core demo.
- **Phase 3** — extension shell: select region, instruct, stream, accept/reject. stdio JSON-RPC.
- **Phase 4** (stretch) — multi-region / cross-file joint denoise.

## Validation strategy

The pure logic (masks, schedule, loop) is TDD'd with a fake forward. The MLX forward is validated for (a) sane logits and coherent unconstrained fills, and (b) optional numeric parity against a one-off PyTorch-CPU run of the original model on a tiny example (CPU avoids the float64/MPS problem). We have no published golden vectors, so coherent code generation is the primary functional signal.
