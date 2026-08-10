from __future__ import annotations

import json

import pytest

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    build_deterministic_router_receipt,
    router_check,
    validate_deterministic_router_envelope,
)
from core.brain.llm.latent_cortex.objective_program_verifier import (
    solve_objective_program,
    validate_objective_program_solution,
)
from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier


def _route(candidate: str, objective: str = "") -> dict:
    atomic = build_atomic_decomposition(candidate, objective=objective)
    return build_deterministic_router_receipt(
        candidate,
        objective=objective,
        atomic_receipt=atomic,
    )


def test_router_verifies_exact_arithmetic_and_refutes_wrong_claim() -> None:
    good = _route("2 + 2 = 4.")
    bad = _route("2 + 2 = 5.")
    assert good["counts"]["verified"] == 1 and good["hard_pass"] is True
    assert bad["counts"]["refuted"] == 1 and bad["hard_pass"] is False
    assert bad["routes"][0]["tool_receipt"]["execution_mode"] == "pure_local_read_only"


def test_router_compiles_python_and_rejects_syntax_error() -> None:
    good = _route("```python\nvalue = 2 + 2\n```")
    bad = _route("```python\nvalue = (\n```")
    assert good["routes"][0]["outcome"] == "verified"
    assert bad["routes"][0]["outcome"] == "refuted"
    assert bad["routes"][0]["detail"]["failure_code"] == "syntax_error"


def test_router_does_not_false_refute_chunked_code_or_partial_json() -> None:
    body = "\n".join(f"value_{index} = {index}" for index in range(100))
    code = _route(f"```python\n{body}\n```")
    assert code["counts"]["refuted"] == 0
    assert code["counts"]["unsupported"] == len(code["routes"])

    multiline_json = _route('{\n  "answer": 42\n}')
    assert multiline_json["counts"]["refuted"] == 0


def test_router_distinguishes_unsupported_from_unknown_without_false_pass() -> None:
    receipt = _route(
        "According to the source, the simulation proves the plan works.",
        objective="Check the claim",
    )
    assert receipt["checked"] is False
    assert receipt["hard_pass"] is True
    assert receipt["counts"]["unsupported"] >= 1
    assert receipt["counts"]["verified"] == 0


def test_router_receipt_contains_no_candidate_text() -> None:
    candidate = "The private marker is 7 * 6 = 42."
    wire = json.dumps(_route(candidate), sort_keys=True)
    assert candidate not in wire
    assert "private marker" not in wire


def test_text_free_router_envelope_rejects_forged_tool_outcome() -> None:
    candidate = "7 * 6 = 42."
    atomic = build_atomic_decomposition(candidate)
    receipt = build_deterministic_router_receipt(
        candidate,
        objective="",
        atomic_receipt=atomic,
    )
    assert validate_deterministic_router_envelope(receipt, atomic_receipt=atomic) == receipt
    receipt["routes"][0]["outcome"] = "refuted"
    import pytest

    with pytest.raises(ValueError, match="commitment|verdict"):
        validate_deterministic_router_envelope(receipt, atomic_receipt=atomic)


def test_task_verifier_caps_deterministically_refuted_candidate() -> None:
    verifier = EpisodeTaskVerifier("Calculate two plus two")
    row = verifier.evaluate("2 + 2 = 5.")
    assert row["checks"]["deterministic_router"]["valid"] is False
    assert row["score"] <= 0.25
    guidance = verifier.to_receipt()
    assert guidance["deterministic_router"]["counts"]["refuted"] == 1


def test_router_check_returns_neutral_when_no_sound_route_exists() -> None:
    candidate = "The design is elegant."
    atomic = build_atomic_decomposition(candidate)
    check = router_check(candidate, objective="", atomic_receipt=atomic)
    assert check["applicable"] is False
    assert check["score"] is None
    assert check["receipt"]["counts"]["unknown"] == 1


def test_router_executes_public_modular_objective_without_hidden_answer() -> None:
    objective = (
        "Start at the given value and apply each operation modulo 19: start=17. "
        "Operations: -11, *12. You may reason before answering. Finish with exactly "
        "one final line using the envelope FINAL_ANSWER: <JSON object>."
    )
    good = _route('FINAL_ANSWER: {"residue":15}', objective)
    bad = _route('FINAL_ANSWER: {"residue":14}', objective)

    assert good["routes"][0]["verifier"] == "exact_objective_program"
    assert good["routes"][0]["outcome"] == "verified"
    assert bad["routes"][0]["outcome"] == "refuted"
    assert bad["routes"][0]["detail"]["failure_codes"] == [
        "objective_result_mismatch"
    ]
    assert "expected_payload" not in good["routes"][0]["detail"]
    assert len(good["routes"][0]["detail"]["expected_payload_sha256"]) == 64


def test_router_executes_public_boolean_objective_with_bounded_parser() -> None:
    objective = (
        "Evaluate this 2-operation expression with 1=true, 0=false, and xor meaning "
        "exactly one operand is true: ((not 0) or 0). Return a value of 1 or 0. "
        "You may reason before answering."
    )
    receipt = _route('FINAL_ANSWER: {"value":1}', objective)

    assert receipt["routes"][0]["verifier"] == "exact_objective_program"
    assert receipt["routes"][0]["outcome"] == "verified"
    execution = receipt["routes"][0]["detail"]["execution"]
    assert execution["declared_operations"] == 2
    assert execution["executed_operations"] == 2
    assert len(execution["expression_sha256"]) == 64


def test_router_executes_public_stable_nearest_traversal_exactly() -> None:
    objective = (
        "Fresh algorithm task. The input values, in original position order, are "
        "[72, 13, 66, 60, 51, 73]. Select the lower median by numeric value first. "
        "Then repeatedly select one remaining value by minimizing, in order: absolute "
        "distance from the most recently selected value; numeric value; original "
        "zero-based position. Return the complete selected-value sequence. Its checksum "
        "is the sum of one-based output position multiplied by value. You may reason "
        "before the answer."
    )
    correct = 'FINAL_ANSWER: {"sequence":[60,66,72,73,51,13],"checksum":1033}'
    upper_median = 'FINAL_ANSWER: {"sequence":[66,60,51,72,73,13],"checksum":1070}'

    good = _route(correct, objective)
    bad = _route(upper_median, objective)

    assert good["routes"][0]["verifier"] == "exact_objective_program"
    assert good["routes"][0]["outcome"] == "verified"
    assert good["routes"][0]["detail"]["family"] == "stable_nearest_traversal"
    assert bad["routes"][0]["outcome"] == "refuted"


@pytest.mark.parametrize(
    ("candidate", "outcome"),
    [
        ('{"value":1}', "verified"),
        ('{"value":0}', "refuted"),
        ('```json\n{"value":1}\n```', "verified"),
        ('```json\n{"value":0}\n```', "refuted"),
    ],
)
def test_router_checks_uniquely_bounded_json_without_final_answer_marker(
    candidate: str,
    outcome: str,
) -> None:
    objective = (
        "Evaluate this 2-operation expression with 1=true, 0=false, and xor meaning "
        "exactly one operand is true: ((1 and 1) or 0). Return a value of 1 or 0. "
        "You may reason before answering."
    )

    receipt = _route(candidate, objective)

    assert receipt["routes"][0]["verifier"] == "exact_objective_program"
    assert receipt["routes"][0]["outcome"] == outcome


def test_router_does_not_treat_prose_wrapped_json_as_unique_terminal_answer() -> None:
    objective = (
        "Evaluate this 2-operation expression with 1=true, 0=false, and xor meaning "
        "exactly one operand is true: ((1 and 1) or 0). Return a value of 1 or 0. "
        "You may reason before answering."
    )

    receipt = _route('Maybe {"value":1}, but I am not certain.', objective)

    assert receipt["routes"][0]["verifier"] == "none"
    assert receipt["routes"][0]["outcome"] == "unknown"


def test_public_objective_solver_emits_canonical_candidate_and_text_free_receipt() -> None:
    objective = (
        "Start at the given value and apply each operation modulo 19: start=12. "
        "Operations: *18, *12. You may reason before answering. Finish with exactly "
        "one final line using the envelope FINAL_ANSWER: <JSON object>."
    )
    solved = solve_objective_program(objective)
    assert solved is not None
    candidate, receipt = solved

    assert candidate == (
        "Start with 12 modulo 19.\n"
        "Step 1: 12 * 18 = 7 (mod 19).\n"
        "Step 2: 7 * 12 = 8 (mod 19).\n"
        'FINAL_ANSWER: {"residue":8}'
    )
    assert validate_objective_program_solution(
        receipt,
        objective=objective,
        candidate=candidate,
    ) == receipt
    wire = json.dumps(receipt, sort_keys=True)
    assert candidate not in wire
    assert '"residue":8' not in wire
