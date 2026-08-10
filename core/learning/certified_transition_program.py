"""Bridge verified curriculum programs into the certified transition organ."""

from __future__ import annotations

from core.brain.llm.latent_cortex.typed_transition_executor import (
    CertifiedTransitionExecutor,
    CertifiedTransitionResult,
    TypedTransitionInput,
)
from core.learning.recurrence_curriculum import StructuredTransitionProgram


def execute_program_transition(
    program: StructuredTransitionProgram,
    *,
    transition_index: int,
    executor: CertifiedTransitionExecutor | None = None,
) -> CertifiedTransitionResult:
    if (
        not isinstance(program, StructuredTransitionProgram)
        or type(transition_index) is not int
        or not 0 <= transition_index < program.state_trace.depth
    ):
        raise ValueError("certified program transition is invalid")
    request = TypedTransitionInput(
        family=program.state_trace.family,
        depth=program.state_trace.depth,
        field_names=program.state_trace.field_names,
        state=program.state_trace.states[transition_index],
        action_field_names=program.action_field_names,
        action=program.actions[transition_index],
    )
    result = (executor or CertifiedTransitionExecutor()).execute(request)
    expected = program.state_trace.states[transition_index + 1]
    if result.next_state != expected:
        raise RuntimeError("certified executor and verified trace disagree")
    return result


__all__ = ["execute_program_transition"]
