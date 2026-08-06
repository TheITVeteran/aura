"""The revision-gate ablation must be able to report the gate as worthless.

The standing criticism this battery answers is that Aura's complexity has never
been shown to solve a problem a simpler system could not. A harness that can
only produce wins would not answer it — it would be the criticism, restated in
JSON. So the tests here spend most of their effort on the losing paths: a
battery where thinking twice always helps (the gate is dead weight), one where
it always hurts (the gate is dead weight the other way), and the tasks whose
improvement the verifier cannot see (the gate must decline to guess).

The one structural property worth more than any of the numbers: all three arms
are scored over the SAME two generations, so no result here can be explained by
one arm having been given more compute. That is asserted directly, because it
is the property that makes the comparison worth running at all.
"""

from __future__ import annotations

from core.evaluation.lesion_inference import InferenceClass
from tools.revision_ablation import (
    ALWAYS_REVISE,
    ARMS,
    GATED,
    SINGLE_PASS,
    RevisionTask,
    battery,
    choose,
    deterministic_responder,
    grade,
    run,
    verify,
)


def _counting_responder():
    calls: list[tuple[str, int]] = []

    def responder(task: RevisionTask, *, attempt: int, previous: str | None) -> str:
        calls.append((task.task_id, attempt))
        return deterministic_responder(task, attempt=attempt, previous=previous)

    return responder, calls


def test_every_arm_scores_the_same_two_generations() -> None:
    """The no-budget-confound property, asserted rather than described.

    If the harness generated per-arm, the arms would differ in sampling noise
    and — with a real model — in token spend, and any delta could be waved away
    as compute. Two generations per task, three arms scored over them.
    """
    tasks = battery()
    responder, calls = _counting_responder()

    ledger, _ = run(responder, tasks)

    assert len(calls) == 2 * len(tasks), "arms did not share generations"
    assert sorted({attempt for _, attempt in calls}) == [1, 2]
    for arm in ARMS:
        assert len(ledger.for_condition(arm)) == len(tasks)


def test_gate_blocks_regressions_the_naive_policy_accepts() -> None:
    """On tasks where pass 2 is worse, always_revise must lose and the gate must not."""
    tasks = [t for t in battery() if t.task_id.startswith("regresses_")]
    assert tasks, "battery lost its regression regime"

    ledger, _ = run(deterministic_responder, tasks)

    gated = ledger.summary(GATED)["success_rate"]
    naive = ledger.summary(ALWAYS_REVISE)["success_rate"]
    assert gated > naive, f"gate did not block regressions: {gated} vs {naive}"


def test_gate_captures_improvements_the_conservative_policy_misses() -> None:
    """On tasks where pass 2 is verifiably better, single_pass must lose."""
    tasks = [t for t in battery() if t.task_id.startswith("improves_")]
    assert tasks, "battery lost its improvement regime"

    ledger, _ = run(deterministic_responder, tasks)

    gated = ledger.summary(GATED)["success_rate"]
    conservative = ledger.summary(SINGLE_PASS)["success_rate"]
    assert gated > conservative, f"gate missed visible improvements: {gated} vs {conservative}"


def test_gate_declines_to_guess_when_the_verifier_cannot_see_the_improvement() -> None:
    """The honest limit, and it belongs in the reported number.

    Both answers satisfy every constraint the verifier checks; only the answer
    key separates them. A gate that adopted the challenger here would be
    guessing, and would score better on this battery while being worse in
    general — which is precisely the failure mode of tuning to a benchmark.
    """
    tasks = [t for t in battery() if t.task_id.startswith("invisible_improves_")]
    assert tasks, "battery lost its invisible-improvement regime"

    ledger, _ = run(deterministic_responder, tasks)

    gated = ledger.summary(GATED)["success_rate"]
    conservative = ledger.summary(SINGLE_PASS)["success_rate"]
    assert gated == conservative, "gate adopted a challenger it had no evidence for"


def test_verifier_is_not_the_answer_key() -> None:
    """A gate consulting the grader would be omniscient and the result circular.

    The verifier must accept at least one answer the grader rejects — that gap
    IS the difference between checking constraints and knowing the answer.
    """
    tasks = [t for t in battery() if t.task_id.startswith("invisible_improves_")]
    wrong_but_verifiable = 0
    for task in tasks:
        candidate = task.detail["first_answer"]
        if verify(candidate, task).ok and grade(candidate, task) == 0.0:
            wrong_but_verifiable += 1

    assert wrong_but_verifiable > 0, (
        "the verifier rejected every wrong answer, so it is functioning as the "
        "answer key and the ablation is circular"
    )


def test_harness_reports_the_gate_as_dead_weight_when_revision_always_helps() -> None:
    """Falsification path 1: if second passes are unconditionally good, the gate buys nothing."""

    def always_improves(task: RevisionTask, *, attempt: int, previous: str | None) -> str:
        return "0" if attempt == 1 else task.answer_key

    tasks = battery()
    ledger, _ = run(always_improves, tasks)

    gated = ledger.summary(GATED)["success_rate"]
    naive = ledger.summary(ALWAYS_REVISE)["success_rate"]
    assert naive >= gated, (
        "the harness credited the gate on a battery where a fixed policy is optimal; "
        "it cannot detect dead weight"
    )


def test_harness_reports_the_gate_as_dead_weight_when_revision_always_hurts() -> None:
    """Falsification path 2: if second passes are unconditionally bad, single_pass is optimal."""

    def always_regresses(task: RevisionTask, *, attempt: int, previous: str | None) -> str:
        return task.answer_key if attempt == 1 else "not a number"

    tasks = battery()
    ledger, _ = run(always_regresses, tasks)

    gated = ledger.summary(GATED)["success_rate"]
    conservative = ledger.summary(SINGLE_PASS)["success_rate"]
    assert conservative >= gated, (
        "the harness credited the gate where never revising is optimal"
    )


def test_crashing_responder_fails_every_arm_equally() -> None:
    """A crash must not silently advantage one arm's denominator."""

    def explodes(task: RevisionTask, *, attempt: int, previous: str | None) -> str:
        raise RuntimeError("model unavailable")

    tasks = battery()
    ledger, _ = run(explodes, tasks)

    for arm in ARMS:
        summary = ledger.summary(arm)
        assert summary["attempts"] == len(tasks)
        assert summary["success_rate"] == 0.0


def test_choose_reports_why_the_gate_decided() -> None:
    """A receipt naming the rule, so a surprising score can be traced to a decision."""
    task = battery()[0]
    _, why = choose(GATED, task.detail["first_answer"], task.answer_key, task)

    assert why.startswith("gate:"), why
    assert why != "gate:", "the gate returned no reason"


def test_scale_preserves_regime_proportions() -> None:
    """Scaling up to resolve an interval must not change what is being measured."""
    small = battery(1)
    large = battery(3)

    assert len(large) == 3 * len(small)
    for prefix in ("improves_", "regresses_", "stable_", "invisible_improves_"):
        n_small = len([t for t in small if t.task_id.startswith(prefix)])
        n_large = len([t for t in large if t.task_id.startswith(prefix)])
        assert n_large == 3 * n_small, f"{prefix} proportion drifted under scaling"


def test_battery_is_solvable_without_the_gate() -> None:
    """Otherwise the result is mechanistic, whatever the delta says.

    Same discipline as the retrieval ablation's reachability control: if the
    fixed policies could not solve any of these tasks, the gate would be the
    only path to an answer and beating them would establish wiring, not
    capability.
    """
    ledger, _ = run(deterministic_responder, battery())

    assert ledger.summary(SINGLE_PASS)["success_rate"] > 0.0
    assert ledger.summary(ALWAYS_REVISE)["success_rate"] > 0.0


def test_invisible_regime_keeps_the_gate_below_a_perfect_score() -> None:
    """The battery must not be winnable outright, or it is measuring itself.

    A gate that scored 1.000 here would mean every improvement was
    verifier-visible — an easier world than the one the gate ships into, and
    a number that would be quoted without its scope.
    """
    ledger, _ = run(deterministic_responder, battery())

    assert ledger.summary(GATED)["success_rate"] < 1.0


def test_inference_class_is_capability_not_mechanistic() -> None:
    """The licensed claim, derived from measured baseline performance."""
    from tools.revision_ablation import LesionClaim

    ledger, _ = run(deterministic_responder, battery())
    single = ledger.summary(SINGLE_PASS)["success_rate"]
    revise = ledger.summary(ALWAYS_REVISE)["success_rate"]

    claim = LesionClaim(
        condition="gated_revision_vs_fixed_policies",
        subsystem="core.brain.reasoning_revision_gate",
        metric_name="verifiable_task_success_rate",
        delta=ledger.summary(GATED)["success_rate"] - max(single, revise),
        metric_has_other_producers=True,
        metric_is_task_success=True,
        tasks_solvable_without_component=max(single, revise) > 0.0,
    )

    assert claim.inference_class is InferenceClass.CAPABILITY
