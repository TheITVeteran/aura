from __future__ import annotations

import json

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    build_deterministic_router_receipt,
    router_check,
    validate_deterministic_router_envelope,
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
