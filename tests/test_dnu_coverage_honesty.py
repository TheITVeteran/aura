"""A benchmark artifact must not report a pass it did not earn.

The authoritative artifacts/agi_live/DNU_AGI_PROOF.json reported:

    total_tasks: 1
    overall_pass_rate: 1.0
    tier: 2 "Emergent (Capped)"
    category_thresholds_passed: True
    passed: True

...while its own unsupported_claims listed all six categories as below minimum,
five of them with ZERO tasks. FINAL_VERDICT.txt said "DNU AGI NOT PROVEN", so
the runner knew. The JSON — the thing tooling parses — said passed: True.

The mechanism was the same vacuous-check pattern found elsewhere in this pass:
category_thresholds_passed iterated scorecard["categories"] looking for
"transfer" and checking its pass_rate. A category with zero attempted tasks
never appears in that dict, so the loop checked nothing and the flag stayed
True. A check that cannot fail on the data that should fail it is not a check.

Nothing here fabricates a harness. HLE / GPQA / SWE-bench / ARC-AGI / GAIA were
not run and this fixture is not a substitute for them; the fix is to say so in
the artifact rather than to let a 1.0 stand unqualified.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "agi_live" / "DNU_AGI_PROOF.json"


@pytest.fixture(scope="module")
def proof() -> dict:
    assert ARTIFACT.is_file(), "the checked-in DNU proof artifact is required"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_a_pass_requires_coverage(proof):
    """passed: True is incompatible with unmet category minimums."""
    if proof.get("unsupported_claims"):
        assert proof.get("passed") is not True, (
            "the artifact reports passed: True while its own unsupported_claims "
            f"lists unmet minimums: {proof['unsupported_claims']}"
        )


def test_category_thresholds_cannot_pass_vacuously(proof):
    """An absent category is the worst score on the threshold, not an exemption."""
    checklist = proof.get("verification_checklist", {})
    if proof.get("unsupported_claims"):
        assert checklist.get("category_thresholds_passed") is not True, (
            "category_thresholds_passed: True while categories are below their "
            "minimums — the check iterates only the categories that exist, so "
            "zero-task categories skip it entirely"
        )


def test_the_artifact_does_not_contradict_its_own_verdict(proof):
    """FINAL_VERDICT said NOT PROVEN while the JSON said passed: True."""
    verdict_file = ARTIFACT.parent / "FINAL_VERDICT.txt"
    if not verdict_file.is_file():
        return
    verdict = verdict_file.read_text(encoding="utf-8").strip().upper()
    if "NOT PROVEN" in verdict:
        assert proof.get("passed") is not True, (
            "FINAL_VERDICT.txt says NOT PROVEN but DNU_AGI_PROOF.json says "
            "passed: True — tooling reads the JSON"
        )


# ---------------------------------------------------------------------------
# The runner's logic, independent of whatever artifact happens to be checked in
# ---------------------------------------------------------------------------


def test_coverage_disclosure_names_what_was_not_run():
    """The fixture must not be readable as a frontier benchmark."""
    from tools.agi.run_dnu_agi_proof_battery import build_coverage_disclosure

    scorecard = {
        "total_tasks": 1,
        "categories": {"novel_reasoning": {"attempted": 1, "pass_rate": 1.0}},
    }
    disclosure = build_coverage_disclosure(
        scorecard, ["Category 'coding' has 0 tasks, below minimum of 10"], [{}]
    )

    assert disclosure["coverage_sufficient"] is False
    assert "coding" in disclosure["categories_with_zero_tasks"]
    blob = " ".join(disclosure["does_not_establish"]).upper()
    for benchmark in ("HLE", "GPQA", "SWE-BENCH", "ARC-AGI", "GAIA"):
        assert benchmark in blob, (
            f"{benchmark} is not named as un-run — the fixture can still be read "
            "as a substitute for it"
        )


def test_coverage_disclosure_reports_sufficiency_when_earned():
    from tools.agi.run_dnu_agi_proof_battery import build_coverage_disclosure

    scorecard = {
        "total_tasks": 90,
        "categories": {
            cat: {"attempted": n, "pass_rate": 0.9}
            for cat, n in (
                ("novel_reasoning", 50), ("coding", 10), ("planning", 5),
                ("self_debug", 5), ("transfer", 10), ("research", 10),
            )
        },
    }
    disclosure = build_coverage_disclosure(scorecard, [], [{}] * 90)
    assert disclosure["coverage_sufficient"] is True
    assert disclosure["categories_with_zero_tasks"] == []


def test_the_threshold_flag_is_derived_from_unsupported_claims():
    """Pins the fix at its source, not just in the current artifact."""
    import inspect

    from tools.agi import run_dnu_agi_proof_battery as runner

    src = inspect.getsource(runner)
    assert "category_thresholds_passed = not unsupported_claims" in src, (
        "category_thresholds_passed no longer derives from the category "
        "minimums — a run with zero tasks in five categories will pass it again"
    )


def test_one_task_at_100_percent_is_not_a_pass():
    """The specific shape of the original artifact."""
    from tools.agi.run_dnu_agi_proof_battery import MINIMUM_COUNTS, assign_tier

    unsupported = [
        f"Category '{cat}' has 0 tasks, below minimum of {n}"
        for cat, n in MINIMUM_COUNTS.items()
    ]
    tier = assign_tier(1.0, has_unsupported_claims=bool(unsupported))

    # A perfect rate on no coverage must be capped, never promoted.
    assert tier["tier"] <= 2, (
        f"a 100% rate over one task was promoted to tier {tier['tier']}"
    )
