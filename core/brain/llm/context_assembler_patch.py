"""Compatibility aliases for the retired ContextAssembler monkey patch.

The canonical implementation lives in :mod:`core.brain.llm.context_assembler`.
This module remains only for older imports and must never replace class methods
at import time or during ``CognitiveEngine`` construction.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.ContextAssemblerPatch")


def _is_casual_interaction_v2(objective: str) -> bool:
    """Delegate legacy callers to the canonical interaction classifier."""
    from core.brain.llm.context_assembler import ContextAssembler

    return ContextAssembler._is_casual_interaction(objective)


def _build_aura_now_block(
    state: AuraState,
    objective: str,
    *,
    compact: bool = False,
) -> str:
    """Delegate legacy callers to the canonical state-grounding renderer."""
    from core.brain.llm.context_assembler import ContextAssembler

    return ContextAssembler._build_aura_now_prompt_block(
        state,
        objective,
        compact=compact,
    )


def _patched_build_system_prompt(state: AuraState) -> str:
    """Compatibility alias for the canonical system-prompt implementation."""
    from core.brain.llm.context_assembler import ContextAssembler

    return ContextAssembler.build_system_prompt(state)


def _patched_build_messages(
    state: AuraState,
    objective: str,
    max_tokens: int | None = None,
    *_args: object,
    **_kwargs: object,
) -> list[dict[str, str]]:
    """Compatibility alias for the canonical message assembler."""
    from core.brain.llm.context_assembler import ContextAssembler

    return ContextAssembler.build_messages(state, objective, max_tokens=max_tokens)


def patch_context_assembler() -> None:
    """Retained no-op for external callers from pre-consolidation releases."""
    logger.debug(
        "ContextAssemblerPatch is retired; canonical implementation already active"
    )
