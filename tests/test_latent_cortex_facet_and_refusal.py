"""Worker assertions are not verdicts, and refusals must be attributable.

CP126 94ecfee0: facet rows come from the worker's own receipt, `satisfied`
was coerced with bool() — so the string "false" became True — and the result
was recorded as a CHECKED hard pass feeding the Foundry's reliability
statistics.

CP126 5879d2b5: _record_failure returned only ok and reason while mutating
shared counters, so every refusal was indistinguishable from every other
instance of itself.
"""
from __future__ import annotations

import pytest

from core.brain import latent_cortex_service as mod
from core.brain.latent_cortex_service import LatentCortexService


@pytest.fixture()
def service():
    return LatentCortexService()


@pytest.fixture()
def verdicts(monkeypatch):
    recorded = []

    class _Foundry:
        @staticmethod
        def record_verdict(**kwargs):
            recorded.append(kwargs)

    import core.brain.verifiers.foundry as foundry_mod

    monkeypatch.setattr(foundry_mod, "get_verifier_foundry", lambda: _Foundry())
    return recorded


def _judge(service, satisfied):
    service._record_facet_judgments(
        {"verifier_guidance": {"facet_judgments": [
            {"facet": "grounded", "satisfied": satisfied, "excerpt": "because X"}
        ]}},
        "general",
        "an objective",
    )


# --- 94ecfee0 ------------------------------------------------------------


def test_the_string_false_is_no_longer_a_pass(service, verdicts):
    """bool("false") is True — the coercion inverted the verdict."""
    _judge(service, "false")

    assert verdicts == []


@pytest.mark.parametrize("value", ["true", "", 0, 1, None, [], {"a": 1}])
def test_a_non_boolean_verdict_is_not_invented(service, verdicts, value):
    _judge(service, value)

    assert verdicts == []


def test_a_real_boolean_is_recorded(service, verdicts):
    _judge(service, True)

    assert len(verdicts) == 1
    assert verdicts[0]["hard_pass"] is True
    assert verdicts[0]["score"] == 1.0


def test_a_false_verdict_is_recorded_as_a_failure(service, verdicts):
    _judge(service, False)

    assert len(verdicts) == 1
    assert verdicts[0]["hard_pass"] is False
    assert verdicts[0]["score"] == 0.0


def test_a_worker_assertion_is_not_recorded_as_checked(service, verdicts):
    """It is the worker's claim about its own output; an operator grading
    against the excerpt is what makes it checked."""
    _judge(service, True)

    assert verdicts[0]["checked"] is False
    assert verdicts[0]["meta"]["source"] == "worker_self_assertion"
    assert verdicts[0]["meta"]["independently_checked"] is False


# --- 5879d2b5 ------------------------------------------------------------


def test_a_refusal_carries_a_receipt(service):
    result = service._record_failure("disabled:AURA_LATENT_CORTEX=0")

    assert result["ok"] is False
    receipt = result["refusal_receipt"]
    assert receipt["schema"] == "aura.latent_cortex.refusal_receipt.v1"
    assert receipt["refusal_id"]
    assert receipt["at"] > 0


def test_the_refusal_is_classified(service):
    receipt = service._record_failure("client_failed:worker_not_ready")["refusal_receipt"]

    assert receipt["reason_class"] == "client_failed"
    assert receipt["reason"] == "client_failed:worker_not_ready"


def test_two_refusals_are_distinguishable(service):
    first = service._record_failure("busy")["refusal_receipt"]
    second = service._record_failure("busy")["refusal_receipt"]

    assert first["refusal_id"] != second["refusal_id"]
    assert second["failure_streak"] > first["failure_streak"]


def test_the_stage_is_recorded(service):
    receipt = service._record_failure("x", stage="dispatch")["refusal_receipt"]

    assert receipt["stage"] == "dispatch"


def test_supplied_evidence_is_bounded(service):
    receipt = service._record_failure(
        "x", evidence={f"k{i}": "v" * 900 for i in range(40)}
    )["refusal_receipt"]

    assert receipt.get("evidence_truncated") is True
    assert all(
        not isinstance(value, str) or len(value) <= 400 for value in receipt.values()
    )


def test_the_reason_contract_is_unchanged(service):
    """Callers read result['reason']; that must keep working."""
    assert service._record_failure("invalid_question")["reason"] == "invalid_question"
