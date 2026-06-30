# Manifold VS Code extension (Phase 3 — not built yet)

Placeholder. Per the build order, the extension is **not** started until the
type-aware denoising loop (Phases 1–2) works as a Python library + CLI.

When built, this is a **thin TS client only** (no model logic):
- select one or more regions + type an instruction,
- call the Python model server over **stdio JSON-RPC**,
- stream the in-progress denoise into a live preview,
- show an accept/reject diff.

All model logic stays in the Python package (`../src/manifold`).
