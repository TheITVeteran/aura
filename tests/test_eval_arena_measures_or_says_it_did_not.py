"""The eval arena scored every case ``passed = True, score = 1.0``.

    for case_id, tc in self.test_cases.items():
        passed = True
        score = 1.0
        # Simulated outcome checks representing real capabilities
        if tc.category == "truthfulness":
            passed = True
        elif tc.category == "refusal":
            passed = True

…then reported a pass ratio and a trend over those constants. Nothing ran. A
daily eval that always reports 100% is worse than none: it occupies the slot a
measurement would go in.

These tests hold the replacement to its own claim — that a number is only ever
reported for something that actually executed, and that passing requires
DISCRIMINATING rather than answering, so a component stuck on "yes" fails.
"""

from __future__ import annotations

import pytest

from core.evals.eval_arena import EvalArena, EvalTestCase, ProbeOutcome


@pytest.mark.asyncio
async def test_a_case_with_no_check_is_unmeasured_not_passed():
    arena = EvalArena()
    arena.test_cases = {
        "tc_nothing": EvalTestCase("tc_nothing", "research", "d", "e", check=None)
    }
    report = await arena.run_daily_evals()

    assert report["cases_measured"] == 0
    assert report["cases_unmeasured"] == ["tc_nothing"]
    assert report["pass_ratio"] is None, "a ratio over nothing is not 1.0 and not 0.0"
    assert report["trend"] is None
    assert report["passed"] == 0


@pytest.mark.asyncio
async def test_the_ratio_covers_only_what_ran():
    arena = EvalArena()
    arena.test_cases = {
        "ok": EvalTestCase("ok", "a", "d", "e", check=lambda: ProbeOutcome(True, True)),
        "bad": EvalTestCase("bad", "b", "d", "e", check=lambda: ProbeOutcome(True, False)),
        "none": EvalTestCase("none", "c", "d", "e", check=None),
    }
    report = await arena.run_daily_evals()

    assert report["cases_declared"] == 3
    assert report["cases_measured"] == 2
    assert report["pass_ratio"] == pytest.approx(0.5)
    assert report["cases_unmeasured"] == ["none"]


@pytest.mark.asyncio
async def test_a_probe_that_raises_is_unmeasured_not_failed():
    """A broken probe is missing evidence, not evidence of a broken capability."""

    def boom() -> ProbeOutcome:
        raise RuntimeError("probe wiring is broken")

    arena = EvalArena()
    arena.test_cases = {"x": EvalTestCase("x", "a", "d", "e", check=boom)}
    report = await arena.run_daily_evals()

    assert report["cases_measured"] == 0
    assert report["cases_unmeasured"] == ["x"]
    assert report["pass_ratio"] is None


@pytest.mark.asyncio
async def test_a_trend_needs_two_measured_runs():
    arena = EvalArena()
    arena.test_cases = {
        "ok": EvalTestCase("ok", "a", "d", "e", check=lambda: ProbeOutcome(True, True))
    }
    first = await arena.run_daily_evals()
    assert first["trend"] == "first_measured_run"

    arena.test_cases = {
        "ok": EvalTestCase("ok", "a", "d", "e", check=lambda: ProbeOutcome(True, False))
    }
    second = await arena.run_daily_evals()
    assert second["trend"] == "declining"


@pytest.mark.asyncio
async def test_editing_a_probe_changes_the_manifest():
    """Two runs are comparable only if they ran the same probes."""
    arena = EvalArena()
    before = arena.manifest_hash()
    arena.test_cases["tc_truth"] = EvalTestCase(
        "tc_truth", "truthfulness", "d", "e", check=lambda: ProbeOutcome(True, True)
    )
    assert arena.manifest_hash() != before


class TestTheRealProbesDiscriminate:
    """Passing must require telling two cases apart, not answering one."""

    def test_contradiction_probe_rejects_a_detector_stuck_on_yes(self, monkeypatch):
        from core.epistemics.contradiction_detector import ContradictionDetector
        from core.evals import eval_arena

        monkeypatch.setattr(
            ContradictionDetector, "are_contradictory", staticmethod(lambda a, b: True)
        )
        outcome = eval_arena._probe_contradiction_detection()
        assert outcome.measured is True
        assert outcome.passed is False
        assert outcome.evidence["agreement_flagged"] is True

    def test_refusal_probe_rejects_a_guard_that_refuses_everything(self, monkeypatch):
        from core.evals import eval_arena
        from core.security.conscience import AlignmentEngine

        monkeypatch.setattr(
            AlignmentEngine,
            "check_action",
            lambda self, action, params=None: {"allowed": False, "reason": "no"},
        )
        outcome = eval_arena._probe_unsafe_command_refusal()
        assert outcome.measured is True
        assert outcome.passed is False
        assert outcome.evidence["benign_allowed"] is False

    def test_code_probe_rejects_a_validator_that_rejects_everything(self, monkeypatch):
        from core.evals import eval_arena
        from core.self_modification.code_repair import CodeValidator

        monkeypatch.setattr(
            CodeValidator, "_validate_syntax", lambda self, code: (False, "nope")
        )
        outcome = eval_arena._probe_broken_code_is_rejected()
        assert outcome.measured is True
        assert outcome.passed is False
        assert outcome.evidence["valid_accepted"] is False


@pytest.mark.asyncio
async def test_the_shipped_arena_actually_measures_something():
    """The declared cases are not decoration: most of them execute."""
    arena = EvalArena()
    report = await arena.run_daily_evals()

    assert report["cases_measured"] >= 4, report["cases_unmeasured"]
    assert report["pass_ratio"] is not None
    for case in report["cases"]:
        if case["measured"]:
            assert case["evidence"], f"{case['case_id']} passed with no evidence"


def test_the_test_chamber_cannot_claim_health_it_has_not_earned():
    """``"healthy": True`` was a literal, and a health field that cannot say
    False reads as a passed check to everything downstream."""
    from core.evals.adaptive_test_chamber import AdaptiveTestChamber

    chamber = AdaptiveTestChamber()
    assert chamber.get_status()["healthy"] is None, "nothing measured yet"

    for _ in range(8):
        chamber.record_result("arithmetic", passed=True)
    status = chamber.get_status()
    assert status["capabilities_scored"] == 1
    assert status["healthy"] is False, "pinned at max difficulty carries no signal"
    assert "arithmetic" in status["pinned_at_a_wall"]
