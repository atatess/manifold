"""`manifold` CLI entry point.

Phases 0-2 are exercised via the scripts in `scripts/` (phase0_forward.py,
phase1_denoise.py). This entry point currently just points there; it will grow a
`serve` subcommand (stdio JSON-RPC) at Phase 3.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(
        "manifold — local diffusion code editing\n"
        "  Phase 0:  python scripts/phase0_forward.py\n"
        "  Phase 1:  python scripts/phase1_denoise.py [--instruction ...] [--ban ...]\n"
        "  (Phase 3 'serve' over stdio JSON-RPC: not yet implemented)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
