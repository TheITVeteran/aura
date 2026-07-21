"""Real retrieval into the workspace (CP239).

The integrated eval proved the model uses PLANTED facts (0->56%). This
adapter backs retrieval with Aura's real memory, and -- critically --
separates a retrieval MISS from a reasoning FAILURE, because those need
opposite fixes and conflating them wastes an integration project.
"""
from __future__ import annotations

import pytest

from core.learning.facade_retrieval import (
    FacadeRetrieval,
    RetrievalAttribution,
    recall_at,
)


class _StubFacade:
    def __init__(self, results):
        self._results = results

    def search_sync(self, query, limit=5):
        return self._results[:limit]


def test_retrieves_and_normalizes_real_memory_dicts():
    facade = FacadeRetrieval(_StubFacade([
        {"content": "Paris is the capital of France.", "score": 0.9},
        {"text": "The Seine flows through Paris.", "score": 0.7},
    ]))
    passages = facade.retrieve("capital of France", limit=5)
    assert passages == [
        "Paris is the capital of France.",
        "The Seine flows through Paris.",
    ]
    assert facade.to_receipt()["passages_returned"] == 2


def test_passages_are_length_bounded():
    """An unbounded memory blob crowds the question out of the window --
    a plumbing failure that masquerades as a reasoning one."""
    facade = FacadeRetrieval(_StubFacade([{"content": "x" * 5000}]), max_chars=100)
    assert len(facade.retrieve("q", limit=1)[0]) == 100


def test_unavailable_organ_returns_nothing_never_fabricates():
    class NoSearch:
        pass

    facade = FacadeRetrieval(NoSearch())
    assert facade.retrieve("q", limit=3) == []
    assert facade.to_receipt()["hit_rate"] == 0.0


def test_a_failing_search_degrades_to_empty_not_a_crash():
    class Broken:
        def search_sync(self, query, limit=5):
            raise RuntimeError("index offline")

    facade = FacadeRetrieval(Broken())
    assert facade.retrieve("q", limit=3) == []
    assert facade.to_receipt()["empty_results"] == 1


def test_empty_and_whitespace_passages_are_dropped():
    facade = FacadeRetrieval(_StubFacade([{"content": "  "}, {"content": "real"}]))
    assert facade.retrieve("q", limit=5) == ["real"]


# ── recall_at: did retrieval actually surface the answer? ───────────────


def test_recall_finds_the_answer_bearing_passage():
    passages = ["irrelevant note", "the node is 42", "another"]
    r = recall_at(passages, "42")
    assert r["recalled"] is True
    assert r["rank"] == 1


def test_recall_reports_a_miss_when_the_fact_is_absent():
    r = recall_at(["nothing useful", "still nothing"], "42")
    assert r["recalled"] is False
    assert r["rank"] is None


def test_recall_respects_the_cutoff():
    passages = ["miss", "miss", "the answer is 42"]
    assert recall_at(passages, "42", k=2)["recalled"] is False
    assert recall_at(passages, "42", k=3)["recalled"] is True


# ── Attribution: retrieval's fault or reasoning's fault? ────────────────


def test_attribution_separates_the_two_failure_modes():
    attr = RetrievalAttribution()
    # fact retrieved AND used
    attr.observe(recalled=True, solved=True)
    attr.observe(recalled=True, solved=True)
    # fact retrieved but NOT used -> reasoning failure
    attr.observe(recalled=True, solved=False)
    # fact never retrieved -> retrieval failure (not reasoning's fault)
    attr.observe(recalled=False, solved=False)
    report = attr.report()
    assert report["retrieval_recall_rate"] == 0.75          # 3 of 4 retrieved
    assert report["use_rate_when_recalled"] == pytest.approx(2 / 3, abs=1e-3)
    assert report["reasoning_failures_despite_recall"] == 1


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="answer must be non-empty"):
        recall_at(["x"], "")
    with pytest.raises(ValueError, match="limit"):
        FacadeRetrieval(_StubFacade([])).retrieve("q", limit=0)
