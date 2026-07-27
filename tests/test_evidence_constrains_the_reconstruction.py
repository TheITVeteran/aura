"""Being handed a piece has to change the outcome.

Reconstructing from published rules is the easy case. The real one is fragments
— a binary, a screenshot, a capture, a log, one file of a format — and having to
infer the rest. Before this, the plan came from her knowledge plus some prose
notes, so evidence in hand did nothing: it could be collected, summarised, and
then have no bearing on what got built or on whether it was right.

Evidence becomes constraints, constraints get checked for coverage, and what is
uncovered is the list of things still to find out. That is what makes a
reconstruction converge rather than simply stop.

Two properties matter more than the extraction details. Confidence must not be
laundered — a symbol read out of a binary and a behaviour guessed from a
screenshot cannot end up sounding equally certain. And the gap report has to
name the strongest unaddressed evidence first, because an unanswered
near-certainty is a hole while an unanswered guess is only a lead.
"""
from __future__ import annotations

import pytest

from core.self_improvement.reconstruction_evidence import (
    Evidence,
    EvidenceKind,
    assess_coverage,
    extract_constraints,
    fuse_evidence,
)
from core.self_improvement.reconstruction_plan import plan_from_payload

PLAN_PAYLOAD = {
    "target": "2048",
    "summary": "sliding tile game with score",
    "components": [{"name": "rules", "responsibility": "merge tiles and score"}],
    "entry_points": ["move", "initial_board", "legal_moves", "apply_action", "render"],
    "worked_examples": [
        {
            "entry_point": "move",
            "argument": {"board": [[2, 2, 0, 0]], "direction": "left"},
            "expected": {"board": [[4, 0, 0, 0]], "score": 4},
        }
    ],
    "invariants": [{"description": "score never negative", "expression": "1 == 1"}],
    "adapter": {
        "initial_state": "initial_board",
        "legal_actions": "legal_moves",
        "apply_action": "apply_action",
    },
}


@pytest.fixture()
def plan():
    parsed, problems = plan_from_payload(PLAN_PAYLOAD)
    assert not problems, problems
    return parsed


def test_a_source_fragment_yields_its_interface() -> None:
    constraints = extract_constraints(
        Evidence(EvidenceKind.SOURCE_FRAGMENT, "def merge_row(row):\n    pass\nclass Board:\n    pass")
    )
    statements = [c.statement for c in constraints]
    assert any("merge_row" in s for s in statements)
    assert any("Board" in s for s in statements)


def test_watching_it_run_yields_behaviour() -> None:
    constraints = extract_constraints(
        Evidence(EvidenceKind.OBSERVED_IO, "board=[2,2] dir=left -> board=[4,0] score=4")
    )
    assert constraints and constraints[0].category == "behaviour"


def test_a_capture_yields_endpoints() -> None:
    constraints = extract_constraints(
        Evidence(EvidenceKind.NETWORK_TRACE, "POST /api/score HTTP/1.1\nGET /api/leaderboard HTTP/1.1")
    )
    statements = " ".join(c.statement for c in constraints)
    assert "/api/score" in statements and "/api/leaderboard" in statements


def test_generic_words_do_not_become_constraints() -> None:
    """"contains the string 'error'" constrains nothing and buries what does."""
    constraints = extract_constraints(
        Evidence(EvidenceKind.BINARY_STRINGS, "error\x00value\x00string\x00Highscore\x00")
    )
    statements = " ".join(c.statement for c in constraints)
    assert "Highscore" in statements
    assert "'error'" not in statements


# ── Confidence is carried, never laundered ─────────────────────────────────

def test_watching_it_run_outranks_reading_a_screenshot() -> None:
    observed = Evidence(EvidenceKind.OBSERVED_IO, "x -> y")
    guessed = Evidence(EvidenceKind.UI_CAPTURE, "New Game")
    assert observed.confidence > guessed.confidence


def test_prior_art_is_the_weakest_evidence() -> None:
    """Something she built before is a lead, not an observation of this target."""
    assert Evidence(EvidenceKind.PRIOR_ART, "x").confidence < Evidence(
        EvidenceKind.BINARY_STRINGS, "x"
    ).confidence


def test_two_independent_sources_corroborate() -> None:
    """The same claim from two sources is worth more than from either.

    Corroboration is on the claim, not on the meaning: two sources asserting
    the same thing merge and gain confidence. Recognising that a screenshot
    label and a manual sentence describe one underlying feature would need
    semantic matching, which is not attempted — those stay separate claims.
    """
    pieces = [
        Evidence(EvidenceKind.UI_CAPTURE, "Undo Move", "screenshot-one"),
        Evidence(EvidenceKind.UI_CAPTURE, "Undo Move", "screenshot-two"),
    ]
    fused = [c for c in fuse_evidence(pieces) if "Undo Move" in c.statement]
    assert len(fused) == 1, "one claim, not two"
    assert fused[0].confidence > Evidence(EvidenceKind.UI_CAPTURE, "x").confidence
    assert "screenshot-one" in fused[0].provenance
    assert "screenshot-two" in fused[0].provenance


def test_nothing_ever_reaches_certainty() -> None:
    pieces = [
        Evidence(EvidenceKind.SOURCE_FRAGMENT, "def go():\n    pass", f"source-{i}")
        for i in range(6)
    ]
    assert all(c.confidence < 1.0 for c in fuse_evidence(pieces))


def test_the_same_fact_is_not_listed_twice() -> None:
    pieces = [
        Evidence(EvidenceKind.SOURCE_FRAGMENT, "def merge_row(row):\n    pass", "a"),
        Evidence(EvidenceKind.SOURCE_FRAGMENT, "def merge_row(row):\n    pass", "b"),
    ]
    fused = fuse_evidence(pieces)
    assert len([c for c in fused if "merge_row" in c.statement]) == 1


# ── Coverage turns evidence into the next questions ────────────────────────

def test_evidence_the_plan_ignores_is_reported_as_a_gap(plan) -> None:
    constraints = fuse_evidence(
        [
            Evidence(EvidenceKind.NETWORK_TRACE, "GET /api/leaderboard HTTP/1.1", "pcap"),
            Evidence(EvidenceKind.BINARY_STRINGS, "Highscore\x00tile_merge_failed\x00", "binary"),
        ]
    )
    report = assess_coverage(plan, constraints)
    assert report.uncovered
    questions = " ".join(report.next_questions())
    assert "leaderboard" in questions or "Highscore" in questions


def test_evidence_the_plan_answers_is_counted_as_covered(plan) -> None:
    constraints = fuse_evidence(
        [Evidence(EvidenceKind.OBSERVED_IO, "board=[2,2,0,0] dir=left -> board=[4,0,0,0] score=4")]
    )
    report = assess_coverage(plan, constraints)
    assert report.covered


def test_the_strongest_unanswered_evidence_is_asked_about_first(plan) -> None:
    """An unanswered near-certainty is a hole; an unanswered guess is a lead."""
    constraints = fuse_evidence(
        [
            Evidence(EvidenceKind.UI_CAPTURE, "Zzzzq Panel", "screenshot"),
            Evidence(EvidenceKind.SOURCE_FRAGMENT, "def qqzzx_handler(x):\n    pass", "leak"),
        ]
    )
    report = assess_coverage(plan, constraints)
    assert report.next_questions(), "both are unaddressed by the plan"
    assert "qqzzx_handler" in report.next_questions()[0]


def test_no_evidence_is_not_reported_as_a_failure(plan) -> None:
    report = assess_coverage(plan, [])
    assert report.coverage == 1.0
    assert "nothing constrains" in report.summary()


def test_coverage_is_a_fraction_that_can_be_watched(plan) -> None:
    """A reconstruction converges when this rises; that is the whole point."""
    constraints = fuse_evidence(
        [
            Evidence(EvidenceKind.OBSERVED_IO, "board=[2,2,0,0] dir=left -> board=[4,0,0,0] score=4"),
            Evidence(EvidenceKind.NETWORK_TRACE, "GET /api/leaderboard HTTP/1.1", "pcap"),
        ]
    )
    report = assess_coverage(plan, constraints)
    assert 0.0 < report.coverage < 1.0
    assert "coverage" in report.summary()
