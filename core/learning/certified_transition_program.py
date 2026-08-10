"""Bridge verified curriculum programs into the certified transition organ."""

from __future__ import annotations

from core.brain.llm.latent_cortex.typed_program_executor import (
    CERTIFIED_PROGRAM_EXECUTION_SCHEMA,
    CertifiedProgramExecution,
    execute_action_sequence,
    execute_compiled_action_program,
)
from core.brain.llm.latent_cortex.typed_transition_executor import (
    CertifiedTransitionExecutor,
    CertifiedTransitionResult,
    TypedTransitionInput,
)
from core.learning.recurrence_curriculum import StructuredTransitionProgram


def execute_program_student_rollin(
    program: StructuredTransitionProgram,
    *,
    executor: CertifiedTransitionExecutor | None = None,
) -> CertifiedProgramExecution:
    """Run a verified program autoregressively and check its private trace after."""

    if not isinstance(program, StructuredTransitionProgram):
        raise TypeError("certified program has the wrong type")
    execution = execute_action_sequence(
        family=program.state_trace.family,
        depth=program.state_trace.depth,
        field_names=program.state_trace.field_names,
        initial_state=program.state_trace.states[0],
        action_field_names=program.action_field_names,
        actions=program.actions,
        executor=executor,
    )
    if execution.states != program.state_trace.states:
        raise RuntimeError("student-roll-in execution and verified trace disagree")
    return execution


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


__all__ = [
    "CERTIFIED_PROGRAM_EXECUTION_SCHEMA",
    "CertifiedProgramExecution",
    "execute_action_sequence",
    "execute_compiled_action_program",
    "execute_program_student_rollin",
    "execute_program_transition",
]
