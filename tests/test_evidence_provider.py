"""Tests for the ReAct evidence provider (real repo reads + memory recall)."""
from __future__ import annotations

import pytest

from core.brain.evidence_provider import (
    EvidenceProvider,
    _salient_terms,
    reference_query_candidates,
)


def test_salient_terms_picks_identifiers():
    terms = _salient_terms("how does the SubprocessGateway handle governance_runtime_active")
    assert "SubprocessGateway" in terms
    assert "governance_runtime_active" in terms


@pytest.mark.asyncio
async def test_repo_evidence_reads_real_spans():
    # SubprocessGateway is a real symbol in this repo; the provider should find it
    # and return real path:line source spans.
    prov = EvidenceProvider(memory_facade=None)
    spans = await prov.gather(
        "explain how SubprocessGateway routes through effect governance",
        task_type="repo_audit",
        limit=6,
    )
    assert spans
    repo_spans = [s for s in spans if s.source == "repo"]
    assert repo_spans
    assert any("subprocess_gateway" in s.ref.lower() for s in repo_spans)
    assert any(s.ref and ":" in s.ref for s in repo_spans)


@pytest.mark.asyncio
async def test_named_path_is_read():
    prov = EvidenceProvider(memory_facade=None)
    spans = await prov.gather(
        "what is in core/brain/verifiers/base.py", task_type="repo_audit", limit=4
    )
    assert any("base.py" in s.ref for s in spans)


@pytest.mark.asyncio
async def test_render_pack_returns_strings():
    prov = EvidenceProvider(memory_facade=None)
    pack = await prov.render_pack(
        "how does SubprocessGateway work", task_type="architecture", limit=4
    )
    assert all(isinstance(p, str) for p in pack)


@pytest.mark.asyncio
async def test_memory_evidence_uses_facade():
    class _Facade:
        async def search(self, query: str, limit: int = 5):
            return [{"content": "Bryan prefers Python for tooling.", "id": "m1"}]

    prov = EvidenceProvider(memory_facade=_Facade())
    spans = await prov.gather("what language does Bryan prefer", task_type="factual", limit=4)
    assert any(s.source == "memory" and "Python" in s.text for s in spans)


def test_reference_query_uses_subject_not_requested_answer_format():
    queries = reference_query_candidates(
        "Explain the Dijkstra shortest-path algorithm in one complete response. "
        "Include: (1) the invariant, (2) pseudocode, and (3) a worked example."
    )

    assert queries[0] == "Dijkstra shortest-path algorithm"
    assert "complete response" not in " ".join(queries).lower()
    assert "worked example" not in " ".join(queries).lower()


def test_reference_query_ignores_visible_speaker_identity_prefix():
    queries = reference_query_candidates(
        "ChatGPT here. Explain the Dijkstra shortest-path algorithm in one "
        "complete response. Include: (1) the invariant and (2) pseudocode."
    )

    assert queries[0] == "Dijkstra shortest-path algorithm"
    assert all("chatgpt" not in query.lower() for query in queries)


def test_reference_query_adds_explicit_technical_facets():
    queries = reference_query_candidates(
        "Explain the Dijkstra shortest-path algorithm. Include binary-heap and "
        "array complexity plus its failure with negative weights."
    )

    assert "Dijkstra algorithm negative weight" in queries
    assert "Dijkstra algorithm binary heap complexity" in queries
    assert "Dijkstra algorithm array complexity" in queries


@pytest.mark.asyncio
async def test_factual_evidence_reads_and_prioritizes_offline_reference(monkeypatch):
    from types import SimpleNamespace

    class _Corpus:
        queries = []

        def search(self, query, limit, *, deadline_s):
            self.queries.append(query)
            assert query.startswith("Dijkstra")
            assert limit >= 6
            assert deadline_s <= 0.25
            return [
                SimpleNamespace(
                    source="wikipedia",
                    title="Edge disjoint shortest pair algorithm",
                    snippet="A related graph algorithm.",
                    rank=-3.0,
                ),
                SimpleNamespace(
                    source="wikipedia",
                    title="Dijkstra's algorithm",
                    snippet="Finds shortest paths in graphs with non-negative edge weights.",
                    rank=-1.0,
                ),
            ]

    monkeypatch.setattr(
        "core.knowledge.local_corpus.get_local_corpus_store", lambda: _Corpus()
    )

    spans = await EvidenceProvider(memory_facade=None).reference_evidence(
        "Explain the Dijkstra shortest-path algorithm in one complete response.",
        limit=2,
    )

    assert spans[0].source == "reference"
    assert "Dijkstra's algorithm" in spans[0].ref
    assert "non-negative" in spans[0].text


@pytest.mark.asyncio
async def test_factual_evidence_keeps_distinct_sections_from_same_article(monkeypatch):
    from types import SimpleNamespace

    class _Corpus:
        def search(self, query, limit, *, deadline_s):
            if "negative weight" in query:
                snippet = "Negative weights require Bellman-Ford instead."
            elif "complexity" in query:
                snippet = "A binary heap costs O(E log V); an array costs O(V^2)."
            else:
                snippet = "Dijkstra finds shortest paths in weighted graphs."
            return [
                SimpleNamespace(
                    source="wikipedia",
                    title="Dijkstra's algorithm",
                    snippet=snippet,
                    rank=-1.0,
                )
            ]

    monkeypatch.setattr(
        "core.knowledge.local_corpus.get_local_corpus_store", lambda: _Corpus()
    )
    spans = await EvidenceProvider(memory_facade=None).reference_evidence(
        "ChatGPT here. Explain the Dijkstra algorithm, including negative weights "
        "and binary heap and array complexity.",
        limit=4,
    )

    rendered = "\n".join(span.render() for span in spans)
    assert "Bellman-Ford" in rendered
    assert "O(E log V)" in rendered
