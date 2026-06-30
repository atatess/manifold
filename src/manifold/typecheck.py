"""Type-checker-in-the-loop (Phase 2) — pyright integration.

NOT YET IMPLEMENTED. Planned shape, recorded so the loop's `on_step` contract
stays aligned with it:

- Run pyright on the current partial decode every k steps. Lowest-plumbing first
  cut: ``pyright --outputjson --stdin`` (or a temp file); optimize to a persistent
  ``pyright-langserver`` process if per-call latency dominates.
- Parse `generalDiagnostics[].range` (zero-based line/character start+end).
- Map each diagnostic char range -> token positions in the gen region.
- Return an `Intervention` (manifold.sampler.Intervention): re-mask the implicated
  committed tokens and/or down-bias the offending logits, so the next step
  denoises toward type-correctness.

This is the core thesis: type feedback goes INTO the loop, not as a post-hoc lint.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Diagnostic:
    line: int          # zero-based
    character: int      # zero-based
    end_line: int
    end_character: int
    severity: str       # "error" | "warning" | "information"
    message: str
    rule: str | None = None


def run_pyright(code: str) -> list[Diagnostic]:  # pragma: no cover - Phase 2
    raise NotImplementedError("Phase 2: wire pyright --outputjson into the loop")
