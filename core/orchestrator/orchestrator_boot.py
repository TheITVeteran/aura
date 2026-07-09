"""Compatibility shim for the active orchestrator boot mixin surface.

The canonical boot implementation lives in `core.orchestrator.boot`.
This file remains only for direct legacy imports that still expect the old
module path.
"""

from core.orchestrator.boot import OrchestratorBootMixin

__all__ = ["OrchestratorBootMixin"]
