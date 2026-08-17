"""Materialize synchronous services required by a foreground chat turn."""

from __future__ import annotations

import time
from typing import Any

from core.container import ServiceContainer


def _materialize_expression_path() -> dict[str, Any]:
    """Exercise the synchronous live-expression path used by the first turn."""

    from interface.routes.chat_desktop_repair import _build_aura_expression_frame

    started = time.perf_counter()
    frame = _build_aura_expression_frame("How are you doing right now?")
    contract = frame.get("contract") if isinstance(frame, dict) else None
    if contract is None:
        raise RuntimeError("foreground chat expression contract is unavailable")
    return {
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "contract_type": type(contract).__name__,
        "requires_live_grounding": bool(
            frame.get("requires_explicit_live_grounding", False)
        ),
    }


def materialize_foreground_chat_dependencies() -> dict[str, Any]:
    """Resolve the complete non-model foreground read path.

    This runs off the event loop during boot.  A successful receipt means the
    first user turn will not discover the skill catalog, open the intention
    ledger, or construct conversation-state readers while its deadline is
    already running.
    """

    cognitive_engine = ServiceContainer.get("cognitive_engine", default=None)
    if cognitive_engine is None or not callable(getattr(cognitive_engine, "think", None)):
        raise RuntimeError("foreground chat cognitive engine is unavailable")

    capability_engine = ServiceContainer.get("capability_engine", default=None)
    get_available_skills = getattr(capability_engine, "get_available_skills", None)
    if not callable(get_available_skills):
        raise RuntimeError("foreground chat capability catalog is unavailable")
    skill_names = tuple(sorted(str(name) for name in get_available_skills()))
    if not skill_names:
        raise RuntimeError("foreground chat capability catalog is empty")

    from core.agency.intention_loop import get_intention_loop
    from core.consciousness.temporal_finitude import get_temporal_finitude_model
    from core.conversation.unified_transcript import UnifiedTranscript
    from core.social.social_imagination import get_social_imagination

    transcript = UnifiedTranscript.get_instance()
    temporal_finitude = get_temporal_finitude_model()
    social_imagination = get_social_imagination()
    intention_loop = get_intention_loop()
    expression_path = _materialize_expression_path()

    return {
        "cognitive_engine": type(cognitive_engine).__name__,
        "capability_engine": type(capability_engine).__name__,
        "skill_count": len(skill_names),
        "transcript": type(transcript).__name__,
        "temporal_finitude": type(temporal_finitude).__name__,
        "social_imagination": type(social_imagination).__name__,
        "intention_loop": type(intention_loop).__name__,
        "expression_path": expression_path,
    }
