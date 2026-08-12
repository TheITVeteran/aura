from __future__ import annotations

import pytest

from core.brain.llm.unified_recurrent_shadow_canary import (
    REFUTED,
    SUPPORTED,
    UnifiedRecurrentShadowCanaryError,
    adjudicate_shadow_canary,
    run_shadow_canary,
    seal_shadow_canary_plan,
)
from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    RECEIPT_SCHEMA,
    seal_shadow_probe_receipt,
)

PACKAGE = "fixture-package"
CONTROLLER = "c" * 64


def _cases() -> list[dict[str, object]]:
    return [
        {
            "task_id": "khop-1",
            "family": "khop",
            "public_token_ids": [1, 201, 1],
            "expected_token_ids": [12, 999],
            "max_tokens": 2,
        },
        {
            "task_id": "modular-1",
            "family": "modular",
            "public_token_ids": [1, 204, 1],
            "expected_token_ids": [13, 999],
            "max_tokens": 2,
        },
    ]


def _receipt(
    request_sha256: str,
    family: str,
    *,
    base_exact: bool,
    shadow_exact: bool,
    base_latency_ms: int = 10,
    shadow_latency_ms: int = 30,
) -> dict[str, object]:
    same = base_exact == shadow_exact
    return seal_shadow_probe_receipt(
        {
            "schema": RECEIPT_SCHEMA,
            "request_sha256": request_sha256,
            "status": "completed",
            "reason": "matched_shadow_probe_completed",
            "package_id": PACKAGE,
            "controller_sha256": CONTROLLER,
            "family": family,
            "recurrence_depth": 4,
            "input_token_count": 3,
            "expected_token_count": 2,
            "max_tokens": 2,
            "base_token_count": 2,
            "base_output_sha256": "a" * 64,
            "base_exact_match": base_exact,
            "base_stopped_on_eos": True,
            "base_latency_ms": base_latency_ms,
            "shadow_token_count": 2,
            "shadow_output_sha256": ("a" if same else "b") * 64,
            "shadow_exact_match": shadow_exact,
            "shadow_stopped_on_eos": True,
            "shadow_latency_ms": shadow_latency_ms,
            "outputs_equal": same,
            "output_exposed": False,
            "serving_authority": False,
        }
    )


def _positive() -> tuple[dict[str, object], list[dict[str, object]]]:
    plan, normalized = seal_shadow_canary_plan(
        _cases(),
        package_id=PACKAGE,
        controller_sha256=CONTROLLER,
    )
    observations = [
        {
            "status": "completed",
            "reason": "matched_shadow_probe_completed",
            "receipt": _receipt(
                row["request_sha256"],
                row["family"],
                base_exact=index == 1,
                shadow_exact=True,
            ),
        }
        for index, row in enumerate(normalized)
    ]
    return plan, observations


def test_canary_support_requires_exact_gain_with_zero_regression() -> None:
    plan, observations = _positive()

    verdict = adjudicate_shadow_canary(plan, observations)

    assert verdict["verdict"] == SUPPORTED
    assert verdict["supported"] is True
    assert verdict["measurements"]["wrong_to_right"] == 1
    assert verdict["measurements"]["right_to_wrong"] == 0
    assert verdict["output_exposed"] is False
    assert verdict["serving_authority"] is False
    assert all("text" not in row and "tokens" not in row for row in verdict["evidence"])


def test_one_base_to_shadow_regression_refutes_the_canary() -> None:
    plan, observations = _positive()
    request = plan["cases"][1]
    observations[1]["receipt"] = _receipt(
        request["request_sha256"],
        request["family"],
        base_exact=True,
        shadow_exact=False,
    )

    verdict = adjudicate_shadow_canary(plan, observations)

    assert verdict["verdict"] == REFUTED
    assert verdict["checks"]["all_shadow_answers_exact"] is False
    assert verdict["checks"]["maximum_right_to_wrong"] is False


def test_latency_is_measured_and_can_refute_promotion() -> None:
    plan, observations = _positive()
    request = plan["cases"][0]
    observations[0]["receipt"] = _receipt(
        request["request_sha256"],
        request["family"],
        base_exact=False,
        shadow_exact=True,
        base_latency_ms=10,
        shadow_latency_ms=200,
    )

    verdict = adjudicate_shadow_canary(plan, observations)

    assert verdict["supported"] is False
    assert verdict["checks"]["maximum_aggregate_latency_ratio"] is False


def test_replayed_or_rebound_receipt_is_malformed_evidence() -> None:
    plan, observations = _positive()
    observations[1]["receipt"] = observations[0]["receipt"]

    verdict = adjudicate_shadow_canary(plan, observations)

    assert verdict["supported"] is False
    assert verdict["measurements"]["completed"] == 1
    assert "unified_recurrent_shadow_probe_request_binding_differs" in (
        verdict["evidence"][1]["errors"]
    )


def test_tampered_plan_is_rejected_as_infrastructure_not_negative_evidence() -> None:
    plan, observations = _positive()
    plan["decision_rule"]["minimum_wrong_to_right"] = 0

    with pytest.raises(UnifiedRecurrentShadowCanaryError, match="plan invalid"):
        adjudicate_shadow_canary(plan, observations)


def test_callback_and_receipt_state_disagreement_refutes_evidence() -> None:
    plan, observations = _positive()
    observations[0]["status"] = "unavailable"
    observations[0]["reason"] = "probe_unavailable"

    verdict = adjudicate_shadow_canary(plan, observations)

    assert verdict["supported"] is False
    assert verdict["measurements"]["completed"] == 1
    assert "shadow_canary_result_receipt_state_differs" in (
        verdict["evidence"][0]["errors"]
    )


def test_malformed_case_request_has_stable_canary_error_boundary() -> None:
    cases = _cases()
    cases[0]["public_token_ids"] = [True]

    with pytest.raises(
        UnifiedRecurrentShadowCanaryError,
        match="case request invalid",
    ):
        seal_shadow_canary_plan(
            cases,
            package_id=PACKAGE,
            controller_sha256=CONTROLLER,
        )


@pytest.mark.asyncio
async def test_runner_executes_each_case_and_returns_only_no_output_evidence() -> None:
    calls = 0

    async def probe(public, expected, *, max_tokens):
        nonlocal calls
        plan, normalized = seal_shadow_canary_plan(
            _cases(),
            package_id=PACKAGE,
            controller_sha256=CONTROLLER,
        )
        del plan, public, expected, max_tokens
        row = normalized[calls]
        receipt = _receipt(
            row["request_sha256"],
            row["family"],
            base_exact=calls == 1,
            shadow_exact=True,
        )
        calls += 1
        return {
            "status": "completed",
            "reason": "matched_shadow_probe_completed",
            "receipt": receipt,
        }

    result = await run_shadow_canary(
        _cases(),
        package_id=PACKAGE,
        controller_sha256=CONTROLLER,
        probe=probe,
    )

    assert calls == 2
    assert result["verdict"]["supported"] is True
    assert "text" not in result
    assert "tokens" not in result
