"""stdio JSON-RPC server (Phase 3) — the bridge to the VS Code extension.

NOT YET IMPLEMENTED. Planned shape: VS Code spawns this as a subprocess; one
long-lived process keeps the 8B model warm. Line-delimited JSON-RPC over
stdin/stdout. A `denoise` request streams `step` notifications (the partial
decode) and ends with a `result`; the extension renders the live preview and a
final accept/reject diff. No network, no ports — airplane-mode friendly.
"""
from __future__ import annotations


def serve():  # pragma: no cover - Phase 3
    raise NotImplementedError("Phase 3: stdio JSON-RPC server")
