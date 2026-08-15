from __future__ import annotations

from core.learning.recurrence_curriculum import (
    RecurrenceTrainingTask,
    StructuredTransitionProgram,
    StructuredTransitionTrace,
)
from core.learning.transition_identifiability import (
    TRANSITION_IDENTIFIABILITY_SCHEMA,
    audit_public_transition_identifiability,
    public_transition_observations,
)


def _mathematics_task(*, seed: int, configuration: int, final_value: int) -> RecurrenceTrainingTask:
    trace = StructuredTransitionTrace(
        family="frontier_mathematics",
        depth=2,
        field_names=("pc", "count_lo", "count_hi", "witness_head", "done"),
        states=(
            (0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (2, final_value, 0, 0, 1),
        ),
    )
    program = StructuredTransitionProgram(
        state_trace=trace,
        action_field_names=("arg0", "arg1", "arg2", "arg3", "arg4", "arg5"),
        actions=(
            (configuration, 2, 0, 0, 1, 0),
            (0, 7, 0, 0, 0, 0),
        ),
    )
    return RecurrenceTrainingTask(
        prompt=f"public mathematics task {seed}",
        answer="answer",
        depth=2,
        family=trace.family,
        seed=seed,
        transition_trace=trace,
        transition_program=program,
    )


def test_public_prefix_resolves_a_non_markov_local_transition() -> None:
    tasks = (
        _mathematics_task(seed=1, configuration=3, final_value=1),
        _mathematics_task(seed=2, configuration=4, final_value=2),
    )

    report = audit_public_transition_identifiability(tasks, tasks)
    family = report["audit"]["families"]["frontier_mathematics"]

    assert report["schema"] == TRANSITION_IDENTIFIABILITY_SCHEMA
    assert family["state_current_action"]["ambiguous_keys"] == 1
    assert (
        family["state_current_action"]["empirical_deterministic_accuracy_ceiling"]
        < 1.0
    )
    assert family["state_full_public_prefix"]["ambiguous_keys"] == 0
    assert report["admission"]["admitted"] is True
    assert report["train_holdout_overlap"]["full_prefix_target_disagreements"] == 0
    assert len(report["report_sha256"]) == 64


def test_observation_projection_never_includes_a_future_action() -> None:
    task = _mathematics_task(seed=3, configuration=3, final_value=1)

    observations = public_transition_observations((task,))

    assert len(observations) == 2
    assert len(observations[0].action_prefix) == 1
    assert len(observations[1].action_prefix) == 2
    assert observations[0].action_prefix[-1] == observations[0].action
    assert observations[1].action_prefix[-1] == observations[1].action
    assert observations[0].action_prefix[0] != observations[1].action
