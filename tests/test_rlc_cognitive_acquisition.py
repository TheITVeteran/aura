"""Governed RLC acquisition: one request, one fetch, one continuation."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.cognitive_acquisition import (
    build_acquisition_receipt,
    build_acquisition_request,
    build_continuation_receipt,
    validate_acquisition_receipt,
    validate_acquisition_request,
    validate_continuation_receipt,
)
from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256

OBJECTIVE = "Compare scheduler lock strategies and justify the safer design."
FIRST_TEXT = "A lease-based single owner is safer because expiry enables recovery."
OLD_MEMORY = {
    "source": "memory",
    "text": "Earlier run used an unbounded lock.",
    "content_sha256": "1" * 64,
}
NEW_MEMORY = {
    "source": "memory.semantic.a1b2c3d4",
    "text": "A bounded lease recovered after owner death.",
    "content_sha256": "2" * 64,
}


def _episode_receipt(action: str = "search_memory") -> dict:
    transition = {
        "schema": "aura.rlc.action_transition.v1",
        "action": action,
        "outcome": "succeeded",
        "checked": True,
        "metrics": {"gain": 0.4, "cost": 0.1},
    }
    return {
        "cognitive_action_trace": [
            {
                "decision": {"action": action},
                "transition": transition,
            }
        ]
    }


def test_request_binds_problem_answer_action_and_existing_source_inventory():
    receipt = _episode_receipt()

    request = build_acquisition_request(
        objective=OBJECTIVE,
        first_text=FIRST_TEXT,
        first_receipt=receipt,
        cognitive_context=[OLD_MEMORY],
    )

    assert request is not None
    assert request["action"] == "search_memory"
    assert request["max_acquisitions"] == 1
    assert request["max_continuation_rounds"] == 1
    assert request["worker_performed_io"] is False
    assert request["before_inventory"] == [["memory", "1" * 64]]
    assert "tentative answer" in request["retrieval_query"]
    assert validate_acquisition_request(
        request,
        objective=OBJECTIVE,
        first_text=FIRST_TEXT,
        first_receipt=receipt,
        cognitive_context=[OLD_MEMORY],
    ) == request

    tampered = deepcopy(request)
    tampered["retrieval_query"] += " injected"
    with pytest.raises(ValueError, match="differs"):
        validate_acquisition_request(
            tampered,
            objective=OBJECTIVE,
            first_text=FIRST_TEXT,
            first_receipt=receipt,
            cognitive_context=[OLD_MEMORY],
        )


def test_non_retrieval_action_creates_no_request():
    assert (
        build_acquisition_request(
            objective=OBJECTIVE,
            first_text=FIRST_TEXT,
            first_receipt=_episode_receipt("decompose"),
            cognitive_context=[OLD_MEMORY],
        )
        is None
    )


def test_request_accepts_worker_slot_text_commitments_without_raw_text():
    request = build_acquisition_request(
        objective=OBJECTIVE,
        first_text=FIRST_TEXT,
        first_receipt=_episode_receipt(),
        cognitive_context=[
            {
                "slot": 2,
                "source": "memory",
                "text_sha256": "a" * 64,
            }
        ],
    )

    assert request is not None
    assert request["before_inventory"] == [["memory", "a" * 64]]


def test_acquisition_distinguishes_repeated_and_new_context():
    request = build_acquisition_request(
        objective=OBJECTIVE,
        first_text=FIRST_TEXT,
        first_receipt=_episode_receipt(),
        cognitive_context=[OLD_MEMORY],
    )
    assert request is not None

    repeated = build_acquisition_receipt(
        request,
        acquired_context=[
            {
                **OLD_MEMORY,
                "source": "memory.semantic.different",
            }
        ],
        ingress_receipt={"schema": "ingress", "result": "same"},
        elapsed_s=0.1,
    )
    assert repeated["status"] == "completed_no_new_context"
    assert repeated["new_context_count"] == 0
    assert repeated["continuation_rounds_authorized"] == 0
    assert validate_acquisition_receipt(repeated, request=request) == repeated

    changed = build_acquisition_receipt(
        request,
        acquired_context=[OLD_MEMORY, NEW_MEMORY],
        ingress_receipt={"schema": "ingress", "result": "new"},
        elapsed_s=0.2,
    )
    assert changed["status"] == "completed_new_context"
    assert changed["new_inventory"] == [
        ["memory.semantic.a1b2c3d4", "2" * 64]
    ]
    assert changed["continuation_rounds_authorized"] == 1
    tampered = deepcopy(changed)
    tampered["new_context_count"] = 0
    with pytest.raises(ValueError):
        validate_acquisition_receipt(tampered, request=request)


def test_evidence_acquisition_counts_only_the_reference_adapter():
    request = build_acquisition_request(
        objective=OBJECTIVE,
        first_text=FIRST_TEXT,
        first_receipt=_episode_receipt("retrieve_evidence"),
        cognitive_context=[
            {
                "source": "world_model",
                "text": "Existing world summary.",
            }
        ],
    )
    assert request is not None
    receipt = build_acquisition_receipt(
        request,
        acquired_context=[
            {
                "source": "world_model",
                "text": "A changed but non-retrieved world summary.",
            }
        ],
        ingress_receipt={
            "present_sources": ["world_model"],
            "absent_sources": ["reference"],
        },
        elapsed_s=0.1,
    )

    assert receipt["status"] == "failed"
    assert receipt["error_code"] == "reference_source_unavailable"
    assert receipt["after_inventory"] == []


def test_continuation_receipt_binds_both_results_and_returned_round():
    request = build_acquisition_request(
        objective=OBJECTIVE,
        first_text=FIRST_TEXT,
        first_receipt=_episode_receipt(),
        cognitive_context=[OLD_MEMORY],
    )
    assert request is not None
    acquisition = build_acquisition_receipt(
        request,
        acquired_context=[NEW_MEMORY],
        ingress_receipt={"schema": "ingress"},
        elapsed_s=0.2,
    )
    first = {"ok": True, "text": FIRST_TEXT, "receipt": _episode_receipt()}
    second = {"ok": True, "text": "Refined answer.", "receipt": {"round": 2}}

    receipt = build_continuation_receipt(
        request,
        acquisition,
        first_result=first,
        second_result=second,
        returned_round=2,
        continuation_reason="second_episode_succeeded",
    )

    assert receipt["returned_round"] == 2
    assert receipt["second_attempted"] is True
    assert receipt["second_succeeded"] is True
    assert receipt["request"] == request
    assert receipt["acquisition"] == acquisition
    payload = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_sha256(payload)
    assert validate_continuation_receipt(receipt) == receipt
    tampered = deepcopy(receipt)
    tampered["returned_round"] = 1
    with pytest.raises(ValueError):
        validate_continuation_receipt(tampered)


@pytest.mark.asyncio
async def test_service_runs_at_most_one_continuation_with_new_context(monkeypatch):
    from core.brain import cognitive_ingress
    from core.brain.latent_cortex_service import LatentCortexService

    service = LatentCortexService()
    calls: list[dict] = []
    first = {
        "ok": True,
        "text": FIRST_TEXT,
        "receipt": _episode_receipt(),
    }
    second = {
        "ok": True,
        "text": "The bounded lease is safer and the new run confirms recovery.",
        "receipt": {"round": 2},
    }

    async def fake_deep_reason(question=None, *, messages=None, **kwargs):
        calls.append(dict(kwargs))
        return first if len(calls) == 1 else second

    ingress = SimpleNamespace(
        stakes=0.8,
        uncertainty=0.6,
        epistemic_genesis=object(),
        epistemic_state=object(),
        memory_result=object(),
        to_receipt=lambda: {"schema": "aura.cognitive_ingress.v1"},
    )
    acquired_queries: list[tuple[str, str]] = []

    async def fake_ingress(*args, **kwargs):
        acquired_queries.append(
            (kwargs["retrieval_query"], kwargs["acquisition_source"])
        )
        return ingress

    monkeypatch.setattr(service, "deep_reason", fake_deep_reason)
    monkeypatch.setattr(cognitive_ingress, "assemble_cognitive_ingress_async", fake_ingress)
    monkeypatch.setattr(
        cognitive_ingress,
        "cognitive_context_items",
        lambda _ingress: [NEW_MEMORY],
    )

    result = await service.deep_reason_with_acquisition(
        OBJECTIVE,
        stakes=0.7,
        uncertainty=0.7,
        timeout_s=60.0,
        foreground_request=True,
        cognitive_context=[OLD_MEMORY],
    )

    assert result is second
    assert len(calls) == 2
    assert len(acquired_queries) == 1
    assert acquired_queries[0][1] == "memory"
    assert calls[0]["publish_workspace_conclusion"] is False
    assert calls[1]["publish_workspace_conclusion"] is True
    assert calls[1]["cognitive_context"] == [NEW_MEMORY]
    continuation = result["receipt"]["cognitive_acquisition"]
    assert continuation["returned_round"] == 2
    assert continuation["acquisition_cap_exhausted"] is True
    assert continuation["continuation_cap_exhausted"] is True


@pytest.mark.asyncio
async def test_service_does_not_continue_when_acquisition_repeats_context(monkeypatch):
    from core.brain import cognitive_ingress
    from core.brain.latent_cortex_service import LatentCortexService

    service = LatentCortexService()
    calls = 0
    first = {
        "ok": True,
        "text": FIRST_TEXT,
        "receipt": _episode_receipt(),
    }

    async def fake_deep_reason(question=None, *, messages=None, **kwargs):
        nonlocal calls
        calls += 1
        return first

    async def fake_ingress(*args, **kwargs):
        return SimpleNamespace(
            stakes=0.7,
            uncertainty=0.7,
            epistemic_genesis=object(),
            epistemic_state=object(),
            memory_result=object(),
            to_receipt=lambda: {"schema": "aura.cognitive_ingress.v1"},
        )

    broadcasts: list[str] = []

    async def fake_broadcast(result, *, objective, stakes):
        broadcasts.append(str(result["text"]))

    monkeypatch.setattr(service, "deep_reason", fake_deep_reason)
    monkeypatch.setattr(service, "_broadcast_conclusion", fake_broadcast)
    monkeypatch.setattr(cognitive_ingress, "assemble_cognitive_ingress_async", fake_ingress)
    monkeypatch.setattr(
        cognitive_ingress,
        "cognitive_context_items",
        lambda _ingress: [OLD_MEMORY],
    )

    result = await service.deep_reason_with_acquisition(
        OBJECTIVE,
        stakes=0.7,
        uncertainty=0.7,
        timeout_s=60.0,
        foreground_request=True,
        cognitive_context=[OLD_MEMORY],
    )

    assert result is first
    assert calls == 1
    assert broadcasts == [FIRST_TEXT]
    continuation = result["receipt"]["cognitive_acquisition"]
    assert continuation["continuation_reason"] == "no_new_context"
    assert continuation["second_attempted"] is False
