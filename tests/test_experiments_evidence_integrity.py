"""CP126: evidence-integrity contracts for the latent-cortex experiment grader.

This module produces the verdicts that reach the Verifier Foundry's
reliability ledger and the frontier certificate, so its failure modes are
overstated confidence, not exceptions.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.experiments import (
    CONJECTURE,
    PROVEN,
    REFUTED,
    SUPPORTED,
    ArmResult,
    Claim,
    PairedObservation,
    Task,
    extract_final_numeric_claim,
    grade_paired_treatment_vs_control,
    grade_treatment_vs_control,
    record_claim_to_foundry,
    task_battery,
)


# ── the verifier must grade the FINAL answer ───────────────────────────────


def _task(answer: str) -> Task:
    return Task(prompt="q", answer=answer, depth=1, family="khop", seed=1)


@pytest.mark.parametrize(
    "text",
    [
        "the steps give 7 but the final answer is 3.5",
        "intermediate 7, final answer: -3.5",
        "7 ... therefore 1/2",
        "7 then 1.5e3",
        "7 so the answer is 1,024",
    ],
)
def test_wrong_final_numeric_answer_is_not_scored_correct(text):
    """A wrong FINAL answer must lose even when an earlier token was right.

    The old filter kept a token only if it equalled the ground truth or was
    an integer, so a wrong decimal/fraction/scientific final answer was
    filtered OUT and the earlier correct token became the "final" claim.
    """
    assert _task("7").verify(text) is False


def test_correct_final_answer_still_passes_after_reasoning():
    assert _task("7").verify("first I thought 3, but the answer is 7") is True
    assert _task("-4").verify("answer is -4") is True


def test_extractor_and_verifier_share_one_answer_shape_rule():
    """The extractor claims to use the SAME rule; drift would make voting
    grade a different token than the verifier."""
    assert extract_final_numeric_claim("steps 7 then final 3.5") == "3.5"
    assert extract_final_numeric_claim("ratio 1/2") == "1/2"
    assert extract_final_numeric_claim("no numbers here") == ""


# ── the reliability ledger must not accept injected verdicts ───────────────


def _valid_claim() -> Claim:
    return Claim(
        experiment="exp_test",
        statement="a graded statement",
        tier=SUPPORTED,
        evidence={"families": {"math": {"n": 30}}},
    )


@pytest.mark.parametrize(
    "claim",
    [
        {"experiment": "x", "statement": "s", "tier": "TOTALLY_PROVEN", "evidence": {"a": 1}},
        {"experiment": "", "statement": "s", "tier": PROVEN, "evidence": {"a": 1}},
        {"experiment": "x", "statement": "", "tier": PROVEN, "evidence": {"a": 1}},
        {"experiment": "x", "statement": "s", "tier": PROVEN},  # no evidence
        {"experiment": "x", "statement": "s", "tier": PROVEN, "evidence": {}},
        "not-a-claim",
        None,
    ],
)
def test_foundry_refuses_unvalidated_claims(claim):
    """Any caller could previously inject a SUPPORTED/PROVEN verdict and
    raise a verifier's measured reliability without running anything."""
    assert record_claim_to_foundry(claim, "math") is False


def test_foundry_refuses_an_invalid_domain():
    assert record_claim_to_foundry(_valid_claim(), "") is False


def test_refuted_verdicts_do_not_carry_positive_reliability(monkeypatch):
    """A refutation must not raise a verifier's score almost as much as
    support did — every non-PROVEN tier previously scored 0.6."""
    recorded: list[dict] = []

    class _Foundry:
        @staticmethod
        def record_verdict(**kwargs):
            recorded.append(kwargs)
            return "verdict-1"

    import core.brain.verifiers.foundry as foundry_mod

    monkeypatch.setattr(foundry_mod, "get_verifier_foundry", lambda: _Foundry())

    scores = {}
    for tier in (PROVEN, SUPPORTED, CONJECTURE, REFUTED):
        recorded.clear()
        claim = Claim(
            experiment="exp_test",
            statement="s",
            tier=tier,
            evidence={"families": {"math": {"n": 30}}},
        )
        assert record_claim_to_foundry(claim, "math") is True
        scores[tier] = recorded[0]["score"]
        assert recorded[0]["hard_pass"] is (tier in (PROVEN, SUPPORTED))

    assert scores[REFUTED] == 0.0
    assert scores[CONJECTURE] < scores[SUPPORTED] < scores[PROVEN]


# ── selective omission must not improve a claim ────────────────────────────


def _arm(n: int, successes: int) -> ArmResult:
    return ArmResult(name="arm", n=n, successes=successes)


def test_family_dropped_from_treatment_cannot_be_silently_ignored():
    """Walking only the treatment side let a control family vanish — an
    omission that can only improve the reported claim."""
    claim = grade_treatment_vs_control(
        "exp_omission",
        "treatment beats control",
        treatment_by_family={"math": _arm(40, 40)},
        control_by_family={"math": _arm(40, 5), "coding": _arm(40, 39)},
    )

    assert "coding" in claim.evidence["families_missing_from_treatment"]
    assert claim.tier == CONJECTURE, "an incomplete comparison cannot be SUPPORTED"


def test_complete_comparison_can_still_be_supported():
    claim = grade_treatment_vs_control(
        "exp_complete",
        "treatment beats control",
        treatment_by_family={"math": _arm(40, 40)},
        control_by_family={"math": _arm(40, 5)},
    )

    assert claim.evidence["families_missing_from_treatment"] == []
    assert claim.tier == SUPPORTED


# ── compute validity must travel with the claim ────────────────────────────


def _paired(n: int, treatment_wins: int) -> dict[str, list[PairedObservation]]:
    rows = [
        PairedObservation(
            task_id=f"t{i}",
            family="math",
            treatment_success=i < treatment_wins,
            control_success=False,
            treatment_layer_apps=100,
            control_layer_apps=100,
        )
        for i in range(n)
    ]
    return {"math": rows}


def test_claim_states_whether_compute_was_validated():
    """Several callers legitimately disable compute matching; a claim graded
    that way must not read as clean causal attribution."""
    matched = grade_paired_treatment_vs_control(
        "exp_compute", "s", _paired(30, 25), require_compute=True
    )
    unmatched = grade_paired_treatment_vs_control(
        "exp_compute", "s", _paired(30, 25), require_compute=False
    )

    assert matched.evidence["compute_matched"] is True
    assert unmatched.evidence["compute_matched"] is False


# ── workload dimensions must be bounded ────────────────────────────────────


@pytest.mark.parametrize(
    ("families", "depths", "per_cell"),
    [
        (["khop"], [0], 4),          # zero depth
        (["khop"], [-1], 4),         # negative depth
        (["khop"], [10_000], 4),     # unbounded depth
        (["khop"], [2], 0),          # empty battery that reads as a real run
        (["khop"], [2], -3),
        (["nope"], [2], 4),          # unknown family
        ([], [2], 4),
        (["khop"], [], 4),
    ],
)
def test_task_battery_rejects_invalid_workload_dimensions(families, depths, per_cell):
    with pytest.raises(ValueError):
        task_battery(families, depths, per_cell)


def test_task_battery_still_builds_a_valid_request():
    tasks = task_battery(["khop"], [2, 3], 2, seed=1)
    assert len(tasks) == 4
    assert all(isinstance(task, Task) for task in tasks)
