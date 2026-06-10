"""Recall telemetry: memory quality must be a measured quantity.

Pins that every RAG-bridge retrieval lands in telemetry (hits, misses,
skips, latency), that aggregates are correct, and that the identity
contract can surface recall quality from real samples.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.memory.recall_telemetry import RecallTelemetry, get_recall_telemetry  # noqa: E402


def test_hit_rate_and_latency_aggregates():
    t = RecallTelemetry(window=50)
    t.record(query_words=6, candidates=8, kept=3, latency_ms=12.0)
    t.record(query_words=5, candidates=4, kept=0, latency_ms=8.0)
    t.record(query_words=2, candidates=0, kept=0, latency_ms=0.4, skipped_reason="query_below_threshold")

    snap = t.snapshot()
    assert snap["lifetime"]["retrievals"] == 3
    assert snap["lifetime"]["attempted"] == 2
    assert snap["lifetime"]["hits"] == 1
    assert snap["lifetime"]["skips"] == 1
    assert snap["lifetime"]["hit_rate"] == 0.5
    assert snap["window"]["attempted"] == 2
    assert snap["window"]["latency_p50_ms"] in (8.0, 12.0)
    assert snap["recent"][-1]["skipped_reason"] == "query_below_threshold"


def test_window_is_bounded():
    t = RecallTelemetry(window=10)
    for i in range(50):
        t.record(query_words=5, candidates=1, kept=1, latency_ms=1.0)
    snap = t.snapshot()
    assert snap["window"]["size"] == 10
    assert snap["lifetime"]["retrievals"] == 50


def test_rag_bridge_records_skip_for_short_query():
    from core.memory import rag_bridge

    before = get_recall_telemetry().snapshot()["lifetime"]["retrievals"]
    result = asyncio.run(rag_bridge.fetch_deep_context("hey"))
    after = get_recall_telemetry().snapshot()

    assert result == ""
    assert after["lifetime"]["retrievals"] == before + 1
    assert after["recent"][-1]["skipped_reason"] == "query_below_threshold"


def test_rag_bridge_records_facade_unavailable():
    from core.container import ServiceContainer
    from core.memory import rag_bridge

    ServiceContainer.clear()
    result = asyncio.run(
        rag_bridge.fetch_deep_context("what did we decide about the journal demo")
    )
    snap = get_recall_telemetry().snapshot()

    assert result == ""
    assert snap["recent"][-1]["skipped_reason"] == "memory_facade_unavailable"


def test_rag_bridge_records_hit_with_latency(monkeypatch):
    from core.container import ServiceContainer
    from core.memory import rag_bridge

    class Facade:
        def search(self, query, limit):
            return [{"content": "we picked the journal demo", "timestamp": 0}]

    class Retriever:
        async def rerank_and_format(self, results, limit):
            return "[just now] we picked the journal demo"

    monkeypatch.setattr(rag_bridge, "temporal_retriever", Retriever())
    ServiceContainer.register_instance("memory_facade", Facade(), required=False)
    try:
        result = asyncio.run(
            rag_bridge.fetch_deep_context("what did we decide about the demo plan")
        )
    finally:
        ServiceContainer.clear()

    snap = get_recall_telemetry().snapshot()
    last = snap["recent"][-1]
    assert "journal demo" in result
    assert last["hit"] is True
    assert last["candidates"] == 1
    assert last["latency_ms"] >= 0.0


def test_identity_contract_surfaces_recall_quality():
    get_recall_telemetry().record(
        query_words=6, candidates=5, kept=2, latency_ms=9.0
    )
    from core.conversation.chat_preflight import _live_internals_summary

    lines = _live_internals_summary()
    assert any("Memory recall (recent)" in line for line in lines)
