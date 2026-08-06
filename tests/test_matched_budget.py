"""tests/test_matched_budget.py — an unmatched comparison must not produce a number.

The load-bearing test in this file is
`test_the_retracted_agi_live_comparison_is_refused`. It reconstructs the arms of
`artifacts/current/agi_live/` from that bundle's retraction and asserts the
harness refuses to compute a verdict. That comparison was committed, reported
100% versus 16.67%, and stood for a month; the only thing that caught it was a
human getting suspicious about three structurally different baselines returning
an identical 0.1667. Suspicion is not a control.
"""

from __future__ import annotations

import pytest

from core.evaluation.matched_budget import (
    Attempt,
    AttemptLedger,
    ConditionBudget,
    UnmatchedBudgetsError,
    check_budget_parity,
    compare,
    equalise,
    require_budget_parity,
)


def _arm(name: str, **kwargs) -> ConditionBudget:
    defaults = dict(model_id="resident-32b", max_output_tokens=512, max_wall_clock_s=60.0)
    defaults.update(kwargs)
    return ConditionBudget(condition=name, **defaults)


# ── the defect this exists to prevent ────────────────────────────────────


def test_the_retracted_agi_live_comparison_is_refused():
    """The real arms, reconstructed. Five handicaps, one refusal."""
    baseline = ConditionBudget(
        condition="raw_llm",
        model_id="resident-32b",
        max_output_tokens=160,
        max_wall_clock_s=None,
        solver_available=False,
        memory_available=False,
    )
    full_aura = ConditionBudget(
        condition="full_aura",
        model_id="resident-32b",
        max_output_tokens=None,
        max_wall_clock_s=240.0,
        solver_available=True,
        memory_available=True,
        tools=frozenset({"web", "files"}),
    )

    report = check_budget_parity([baseline, full_aura])

    assert report.matched is False
    flagged = {violation.dimension for violation in report.violations}
    assert "max_output_tokens" in flagged, "the 160-token strangle was not caught"
    assert "solver_available" in flagged, "the solver asymmetry was not caught"
    assert {"max_wall_clock_s", "tools", "memory_available"} <= flagged


def test_a_void_comparison_yields_no_success_rate():
    """Refusal is the product. A caveated number is what the retraction was."""
    baseline = ConditionBudget(condition="baseline", model_id="m", max_output_tokens=160)
    treatment = ConditionBudget(condition="treatment", model_id="m", max_output_tokens=None)

    ledger = AttemptLedger()
    for index in range(12):
        ledger.record(Attempt(f"t{index}", "baseline", "no_answer"))
        ledger.record(Attempt(f"t{index}", "treatment", "success", 1.0))

    result = compare([baseline, treatment], ledger)

    assert result["verdict"] == "void"
    assert "success_rate" not in result, (
        "a void comparison still reported a success rate — the number is exactly "
        "what must not escape"
    )


# ── parity ───────────────────────────────────────────────────────────────


def test_matched_arms_compare():
    ledger = AttemptLedger()
    for index in range(5):
        ledger.record(Attempt(f"t{index}", "a", "success", 1.0))
        ledger.record(Attempt(f"t{index}", "b", "failure", 0.0))

    result = compare([_arm("a"), _arm("b")], ledger)

    assert result["verdict"] == "computed"
    assert result["success_rate"] == {"a": 1.0, "b": 0.0}


def test_a_declared_variable_is_not_a_violation():
    """An ablation MUST differ on the thing it ablates."""
    with_memory = _arm("full", memory_available=True, varied=frozenset({"memory_available"}))
    without = _arm("lesioned", memory_available=False, varied=frozenset({"memory_available"}))

    assert check_budget_parity([with_memory, without]).matched is True


def test_one_arm_cannot_declare_the_variable_alone():
    """Otherwise any handicap is legalised by the arm that benefits from it."""
    generous = _arm("full", solver_available=True, varied=frozenset({"solver_available"}))
    starved = _arm("baseline", solver_available=False)

    report = check_budget_parity([generous, starved])

    assert report.matched is False
    assert report.violations[0].dimension == "solver_available"


def test_a_handicap_that_favours_the_baseline_is_still_a_violation():
    """"We were generous to the control" is still an uncontrolled variable.

    It flatters a different conclusion, which is not the same as being sound.
    """
    generous_baseline = _arm("baseline", max_output_tokens=4096)
    treatment = _arm("treatment", max_output_tokens=512)

    assert check_budget_parity([generous_baseline, treatment]).matched is False


def test_unbounded_does_not_match_bounded():
    """`None` is a value. This is the exact shape of the original defect."""
    assert check_budget_parity(
        [_arm("a", max_output_tokens=None), _arm("b", max_output_tokens=160)]
    ).matched is False


def test_require_budget_parity_raises_rather_than_returning():
    with pytest.raises(UnmatchedBudgetsError) as excinfo:
        require_budget_parity([_arm("a", max_output_tokens=160), _arm("b")])
    assert "comparison void" in str(excinfo.value)


def test_varying_an_unknown_dimension_is_rejected():
    """A typo in `varied` would silently exempt nothing and look like it did."""
    with pytest.raises(ValueError, match="unknown budget dimension"):
        _arm("a", varied=frozenset({"max_tokens"}))


def test_parity_needs_two_arms():
    with pytest.raises(ValueError, match="at least two conditions"):
        check_budget_parity([_arm("only")])


def test_duplicate_condition_names_are_rejected():
    """Two arms with one name silently collapse into whichever was written last."""
    with pytest.raises(ValueError, match="duplicate condition names"):
        check_budget_parity([_arm("a"), _arm("a")])


# ── equalise ─────────────────────────────────────────────────────────────


def test_equalise_tightens_every_arm_to_the_stingiest():
    baseline = ConditionBudget(condition="baseline", model_id="m", max_output_tokens=160)
    full = ConditionBudget(
        condition="full",
        model_id="m",
        max_output_tokens=None,
        max_wall_clock_s=240.0,
        solver_available=True,
        tools=frozenset({"web"}),
    )

    equalised = equalise([baseline, full])

    assert check_budget_parity(equalised).matched is True
    assert all(arm.max_output_tokens == 160 for arm in equalised)
    assert all(arm.solver_available is False for arm in equalised)
    assert all(arm.tools == frozenset() for arm in equalised)


def test_equalise_leaves_the_declared_variable_alone():
    """Equalising the independent variable would erase the experiment."""
    varied = frozenset({"memory_available"})
    equalised = equalise(
        [
            _arm("full", memory_available=True, varied=varied),
            _arm("lesioned", memory_available=False, varied=varied),
        ]
    )
    assert {arm.memory_available for arm in equalised} == {True, False}


# ── honest denominators ──────────────────────────────────────────────────


def test_the_denominator_is_attempts_not_graded_runs():
    """A crash is an attempt. Dropping it is how a broken system reports 100%."""
    ledger = AttemptLedger()
    ledger.record(Attempt("t1", "aura", "success", 1.0))
    ledger.record(Attempt("t2", "aura", "crash"))
    ledger.record(Attempt("t3", "aura", "timeout"))

    summary = ledger.summary("aura")

    assert summary["attempts"] == 3
    assert summary["success_rate"] == pytest.approx(1 / 3)


def test_a_fallback_success_is_counted_but_not_counted_clean():
    """The architecture succeeding and a simpler lane rescuing it are not the same.

    This is the measurement the fallback ladder makes hard to see: a visible
    success can mean the principal model failed, the cognitive pipeline failed,
    and a reflex model answered.
    """
    ledger = AttemptLedger()
    ledger.record(Attempt("t1", "aura", "success", 1.0))
    ledger.record(Attempt("t2", "aura", "success", 1.0, fell_back=True, lane="reflex"))
    ledger.record(Attempt("t3", "aura", "success", 1.0, retries=2))
    ledger.record(Attempt("t4", "aura", "success", 1.0, human_intervention=True))

    summary = ledger.summary("aura")

    assert summary["success_rate"] == 1.0
    assert summary["clean_success_rate"] == 0.25, (
        "three of four successes needed a fallback, a retry or a human, and the "
        "report showed a flat 100%"
    )
    assert summary["fell_back"] == 1
    assert summary["needed_human"] == 1
    assert summary["total_retries"] == 2
    assert summary["lanes"] == ["reflex"]


def test_a_condition_with_no_attempts_voids_the_comparison():
    """Zero attempts is not a score of zero; it is an unmeasured condition."""
    ledger = AttemptLedger()
    ledger.record(Attempt("t1", "a", "success", 1.0))

    result = compare([_arm("a"), _arm("b")], ledger)

    assert result["verdict"] == "void"
    assert "not a condition that was measured" in result["reason"]


def test_budgets_reject_nonsense():
    with pytest.raises(ValueError, match="max_output_tokens"):
        _arm("a", max_output_tokens=0)
    with pytest.raises(ValueError, match="max_wall_clock_s"):
        _arm("a", max_wall_clock_s=-1.0)
    with pytest.raises(ValueError, match="max_retries"):
        _arm("a", max_retries=-1)
