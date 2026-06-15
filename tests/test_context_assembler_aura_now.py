from __future__ import annotations

from core.brain.llm.context_assembler import ContextAssembler
from core.state.aura_state import AuraState


def test_context_assembler_injects_state_grounded_aura_now_block() -> None:
    state = AuraState.default()
    state.cognition.current_objective = "What are you feeling right now?"

    block = ContextAssembler._build_aura_now_prompt_block(
        state,
        state.cognition.current_objective,
        compact=False,
    )

    assert "AURA NOW (STATE-GROUNDED)" in block
    assert "STATE-GROUNDED INTROSPECTION" in block
    assert "forbidden=proven phenomenal consciousness" in block
