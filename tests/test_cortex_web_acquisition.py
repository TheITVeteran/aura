from __future__ import annotations

import pytest

from core.brain.cortex_web_acquisition import (
    CORTEX_WEB_ACQUISITION_SCHEMA,
    acquire_live_web_evidence,
    should_acquire_live_web,
)


def test_live_web_selection_uses_temporal_need_and_local_coverage():
    assert should_acquire_live_web(
        "What is the latest Python release?",
        "Python release",
        local_context_is_new=True,
    ) == (True, "live_or_source_sensitive_objective")
    assert should_acquire_live_web(
        "Explain the pumping lemma.",
        "pumping lemma",
        local_context_is_new=False,
    ) == (True, "local_reference_uncovered")
    assert should_acquire_live_web(
        "Explain the pumping lemma.",
        "pumping lemma",
        local_context_is_new=True,
    ) == (False, "local_reference_sufficient")
    assert should_acquire_live_web(
        "What is in my private notes today?",
        "My notes say the access token is sk-example-secret.",
        local_context_is_new=False,
    ) == (False, "private_or_local_objective")


@pytest.mark.asyncio
async def test_live_web_runs_through_governed_orchestrator_and_returns_typed_evidence():
    class Orchestrator:
        calls: list[tuple] = []

        async def execute_tool(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {
                "ok": True,
                "answer": "Python 3.14 is the current stable feature series.",
                "sources": [
                    {
                        "title": "Python downloads",
                        "text": "Download the latest Python release.",
                        "url": "https://python.org/downloads/",
                    }
                ],
            }

    orchestrator = Orchestrator()
    acquired = await acquire_live_web_evidence(
        orchestrator,
        objective="What is the current Python release?",
        retrieval_query="current Python release",
        cognitive_context=[{"source": "world_model", "text": "Python exists."}],
        selection_reason="live_or_source_sensitive_objective",
    )

    assert len(orchestrator.calls) == 1
    args, kwargs = orchestrator.calls[0]
    assert args[0] == "web_search"
    assert args[1]["query"] == "What is the current Python release?"
    assert args[1]["retain"] is False
    assert kwargs["origin"] == "latent_cortex"
    context = kwargs["payload_context"]
    assert context["effect_scope"] == "read_only"
    assert context["foreground_cognitive_acquisition"] is True
    assert context["user_explicitly_authorized"] is False
    assert acquired.receipt["schema"] == CORTEX_WEB_ACQUISITION_SCHEMA
    assert acquired.receipt["completed"] is True
    assert acquired.receipt["worker_performed_io"] is False
    assert acquired.receipt["service_performed_io"] is True
    assert any(
        item["source"] == "capability.web_search"
        and item["instruction_authority"] is False
        for item in acquired.context or []
    )


@pytest.mark.asyncio
async def test_live_web_fails_honestly_without_an_executor():
    acquired = await acquire_live_web_evidence(
        object(),
        objective="What changed today?",
        retrieval_query="changes today",
        cognitive_context=None,
        selection_reason="live_or_source_sensitive_objective",
    )

    assert acquired.context is None
    assert acquired.receipt["attempted"] is False
    assert acquired.receipt["status"] == "executor_unavailable"
    assert len(acquired.receipt["receipt_sha256"]) == 64


@pytest.mark.asyncio
async def test_live_web_contains_noncanonical_adapter_results():
    class Orchestrator:
        async def execute_tool(self, *_args, **_kwargs):
            return {"ok": True, "results": [{"opaque": object()}]}

    original_context = [{"source": "world_model", "text": "Known locally."}]
    acquired = await acquire_live_web_evidence(
        Orchestrator(),
        objective="What changed online today?",
        retrieval_query="changes today",
        cognitive_context=original_context,
        selection_reason="live_or_source_sensitive_objective",
    )

    assert acquired.context is original_context
    assert acquired.receipt["completed"] is False
    assert acquired.receipt["status"] == "noncanonical_result"
    assert acquired.receipt["result_sha256"] is None
    assert len(acquired.receipt["receipt_sha256"]) == 64
