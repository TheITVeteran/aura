"""The learner was trained on shape and told it was correctness.

Credit for a response was computed from its length plus whether it contained a
newline, a hyphen-space, a "1." or a code fence. A long, well-formatted
hallucination therefore earned the maximum score available, and a correct
one-line answer was penalized for being short. Nothing about the text's
truth entered the calculation, and nothing had to: the absence of a verdict
was scored as a middling pass.

The turn ledger already records what verified the SERVED answer, on a ranked
scale. That rank is the score now, normalized against the top of the scale —
and a turn nothing graded earns no entry at all, because an unmeasured turn is
not a graded turn.

The homeostasis signal had the same shape. Every nonempty string was reported
as a success, so a canned recovery message off a fallback lane raised
integrity — her own sense that she is working correctly — on exactly the turns
that are evidence she is not.
"""
from __future__ import annotations

from core.brain.inference_gate import InferenceGate
from core.runtime.turn_outcome import TurnOutcome, VerificationGrade, bind_turn


def _gate():
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_credit_basis = ""
    return gate


# ────────────────────────────── credit follows the verdict


def test_an_ungraded_turn_earns_no_credit():
    gate = _gate()
    outcome = TurnOutcome(origin="user_chat")
    cid = outcome.record_candidate("a long, beautifully formatted answer\n- one\n- two", source="cortex")
    outcome.mark_served("a long, beautifully formatted answer\n- one\n- two", candidate_id=cid)

    with bind_turn(outcome):
        score, basis = gate._response_credit_outcome()

    assert score is None
    assert basis.startswith("ungraded:")


def test_an_asserted_answer_is_still_ungraded():
    """ASSERTED means the runtime said so about itself. That is the claim
    under test, not evidence for it."""
    gate = _gate()
    outcome = TurnOutcome(origin="user_chat")
    cid = outcome.record_candidate(
        "trust me", source="cortex", verification=VerificationGrade.ASSERTED
    )
    outcome.mark_served("trust me", candidate_id=cid)

    with bind_turn(outcome):
        score, basis = gate._response_credit_outcome()

    assert score is None
    assert basis == "ungraded:asserted"


def test_a_verified_answer_earns_credit_proportional_to_its_grade():
    gate = _gate()
    outcome = TurnOutcome(origin="user_chat")
    cid = outcome.record_candidate(
        "42", source="cortex", verification=VerificationGrade.EXTERNALLY_VERIFIED
    )
    outcome.mark_served("42", candidate_id=cid)

    with bind_turn(outcome):
        score, basis = gate._response_credit_outcome()

    assert score == 1.0
    assert basis == "graded:externally_verified"


def test_a_weaker_grade_earns_less_than_a_stronger_one():
    gate = _gate()

    def _score(grade):
        outcome = TurnOutcome(origin="user_chat")
        cid = outcome.record_candidate("x", source="cortex", verification=grade)
        outcome.mark_served("x", candidate_id=cid)
        with bind_turn(outcome):
            return gate._response_credit_outcome()[0]

    observed = _score(VerificationGrade.OBSERVED)
    external = _score(VerificationGrade.EXTERNALLY_VERIFIED)

    assert observed is not None
    assert observed < external


def test_a_short_correct_answer_is_not_penalized_for_being_short():
    """The old score multiplied length; a verified one-word answer scored
    below an unverified essay."""
    gate = _gate()
    outcome = TurnOutcome(origin="user_chat")
    cid = outcome.record_candidate(
        "4", source="cortex", verification=VerificationGrade.EXTERNALLY_VERIFIED
    )
    outcome.mark_served("4", candidate_id=cid)

    with bind_turn(outcome):
        score, _basis = gate._response_credit_outcome()

    assert score == 1.0


def test_no_turn_bound_earns_no_credit():
    gate = _gate()

    score, basis = gate._response_credit_outcome()

    assert score is None
    assert basis == "no_turn_bound"


def test_a_verified_draft_that_was_discarded_does_not_buy_credit():
    """The ledger grades what was SERVED. A draft that was verified and then
    replaced has not verified the answer the person got."""
    gate = _gate()
    outcome = TurnOutcome(origin="user_chat")
    outcome.record_candidate(
        "verified draft", source="verifier", verification=VerificationGrade.EXTERNALLY_VERIFIED
    )
    replacement = outcome.record_candidate("what actually went out", source="fallback")
    outcome.mark_served("what actually went out", candidate_id=replacement)

    with bind_turn(outcome):
        score, _basis = gate._response_credit_outcome()

    assert score is None


def test_the_shape_heuristic_is_gone_from_the_source():
    """Length and a list marker say nothing about correctness."""
    import inspect

    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)

    assert "response_len / 500.0" not in source
    assert 'has_structure = any(marker in response_text' not in source


# ────────────────────────────── homeostatic success needs more than text


def _homeostasis():
    from core.consciousness.homeostasis import HomeostasisEngine

    return HomeostasisEngine()


def test_a_fallback_answer_does_not_raise_integrity():
    engine = _homeostasis()
    # A fresh engine sits at the ceiling, where nothing can move up.
    engine.integrity = 0.5
    before = engine.integrity

    engine.on_response_success(response_length=200, fallback=True)

    assert engine.integrity == before, (
        "a canned recovery message raised her sense that she is working correctly"
    )


def test_a_fallback_answer_is_still_counted():
    engine = _homeostasis()

    engine.on_response_success(response_length=200, fallback=True)

    counts = engine.response_outcome_counts()
    assert counts["total"] == 1
    assert counts["fallback"] == 1
    assert counts["succeeded"] == 0


def test_an_ordinary_answer_still_raises_integrity():
    engine = _homeostasis()
    engine.integrity = 0.5
    before = engine.integrity

    engine.on_response_success(response_length=200)

    assert engine.integrity > before


def test_the_success_rate_excludes_fallback_answers():
    engine = _homeostasis()

    engine.on_response_success(response_length=200)
    engine.on_response_success(response_length=200, fallback=True)

    assert engine.get_response_success_rate() == 0.5


def test_zero_evidence_is_distinguishable_from_no_failures():
    """The rate returns 1.0 on a fresh boot, which reads as perfect health
    from nothing at all."""
    engine = _homeostasis()

    assert engine.get_response_success_rate() == 1.0
    assert engine.response_outcome_counts()["total"] == 0


def test_the_verified_endpoint_is_absent_until_something_verifies_one():
    engine = _homeostasis()

    assert engine.last_verified_response_endpoint == ""

    engine.on_response_success(response_length=10, verified=True, endpoint="Cortex")

    assert engine.last_verified_response_endpoint == "Cortex"


def test_the_gate_tells_homeostasis_which_lane_answered():
    import ast
    import inspect

    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "on_response_success"
        ):
            continue
        passed = {keyword.arg for keyword in node.keywords}
        assert {"verified", "fallback", "endpoint"} <= passed, (
            f"the homeostatic success signal still carries only {sorted(passed)}"
        )
        return
    raise AssertionError("the homeostasis success hook was not found")
