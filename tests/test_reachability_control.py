"""The control that decides whether an ablation may claim capability.

`tools/capability_ablation.py` used to hardcode
`tasks_solvable_without_component=True` under a comment reading "true by
construction". It was true only while the history fit inside the long_context
window. At history=40/window=12 the fact sat outside the window, the lesioned
arm scored 0.000 on every task because it could not SEE the answer, and the
scorecard still printed "capability: 1" — a mechanistic result wearing a
capability label.

These tests hold the control to the only standard that makes it worth having:
it must be able to come back negative. A check that cannot fail is not a check,
and this file exists because the thing it replaced could not fail.
"""

from __future__ import annotations

import pytest

from core.evaluation.lesion_inference import InferenceClass, LesionClaim
from tools.capability_ablation import (
    LONG_CONTEXT,
    battery,
    deterministic_responder,
    run_reachability_control,
)


def _claim(*, delta: float, solvable: bool) -> LesionClaim:
    return LesionClaim(
        condition="architecture_vs_long_context",
        subsystem="core.memory (context assembly)",
        metric_name="multi_turn_recall_success_rate",
        delta=delta,
        metric_has_other_producers=True,
        metric_is_task_success=True,
        tasks_solvable_without_component=solvable,
    )


def test_control_passes_when_the_answer_is_in_the_transcript() -> None:
    """The battery is gradeable: an unbudgeted reader solves it."""
    result = run_reachability_control(deterministic_responder, battery(40))

    assert result["success_rate"] > 0
    assert result["battery_is_gradeable"] is True
    assert result["tasks_solvable_without_component"] is True
    # The control is deliberately NOT an arm and must say so, or a reader will
    # compare its rate against the budgeted arms and call the gap an effect.
    assert result["under_budget_parity"] is False


def test_control_fails_when_nothing_can_reach_the_answer() -> None:
    """The failure path — the reason this control is worth running.

    A responder that never produces the answer key stands in for a battery
    whose answers are not recoverable from the transcript, or a grader that
    does not accept correct output. Either way no lesion over that battery
    means anything, and the control must say so rather than let a confident
    delta through.
    """

    def never_answers(_condition, _task, _turn, _history) -> str:
        return "I do not know."

    result = run_reachability_control(never_answers, battery(40))

    assert result["success_rate"] == 0.0
    assert result["battery_is_gradeable"] is False
    assert result["tasks_solvable_without_component"] is False


def test_control_counts_a_crashing_responder_as_failure() -> None:
    """A control that crashes is a failed control, not a skipped one."""

    def explodes(_condition, _task, _turn, _history) -> str:
        raise RuntimeError("model unavailable")

    tasks = battery(8)
    result = run_reachability_control(explodes, tasks)

    assert result["attempts"] == len(tasks)
    assert result["solved"] == 0
    assert result["battery_is_gradeable"] is False


def test_unreachable_battery_downgrades_the_claim_to_mechanistic() -> None:
    """The control's answer must actually move the licensed inference.

    This is the regression that matters: same +1.000 delta, same task-success
    metric, and the ONLY difference is whether the lesioned arm ever had a
    path to the answer. That difference has to change what may be claimed.
    """
    reachable = _claim(delta=1.0, solvable=True)
    unreachable = _claim(delta=1.0, solvable=False)

    assert reachable.inference_class is InferenceClass.CAPABILITY
    assert unreachable.inference_class is InferenceClass.MECHANISTIC
    assert reachable.supports_earns_its_cost is True
    assert unreachable.supports_earns_its_cost is False


def test_control_sees_the_whole_history_not_the_window() -> None:
    """The control must be unwindowed, or it just re-runs the lesioned arm.

    If the control inherited the 12-turn window it would score 0.000 on a
    40-turn history and permanently report the battery as ungradeable — the
    control would then be measuring the budget, which is the very thing the
    experiment varies.
    """
    tasks = battery(40)
    seen: list[int] = []

    def record_history_length(_condition, _task, _turn, history) -> str:
        seen.append(len(history))
        return "I do not know."

    run_reachability_control(record_history_length, tasks)

    assert seen, "control ran no tasks"
    assert min(seen) > 12, f"control was windowed: shortest history was {min(seen)} turns"


@pytest.mark.parametrize("history_turns", [8, 40])
def test_control_is_asked_of_the_lesioned_arm(history_turns: int) -> None:
    """The control probes the arm the claim is made against, not the treatment.

    Running it as `full_architecture` would ask whether Aura can solve the
    battery — which is the result, not the control — and would return 1.000
    even for a battery no plain reader could ever grade.
    """
    conditions: list[str] = []

    def record_condition(condition, _task, _turn, _history) -> str:
        conditions.append(condition)
        return "I do not know."

    run_reachability_control(record_condition, battery(history_turns))

    assert set(conditions) == {LONG_CONTEXT}
