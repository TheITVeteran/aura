"""World-scale ingestion: unrestricted reads, distillation, belief/memory updates, governed writes."""
from __future__ import annotations

import asyncio

import pytest

from core.world_model.world_ingestion import (
    IngestDocument,
    WorldIngestionEngine,
    get_world_ingestion_engine,
)

_HTML = """
<html><head><title>Mars Facts</title></head><body>
<script>var x = 1;</script>
<h1>Mars</h1>
<p>Mars is the fourth planet from the Sun and has two small moons.</p>
<p>The planet has a thin atmosphere that is mostly carbon dioxide.</p>
<p>Click here</p>
</body></html>
"""


def _engine(**kw):
    # No real network: every fetch returns the canned page.
    return WorldIngestionEngine(fetcher=lambda url: (200, _HTML), min_requests_interval_s=0.0, **kw)


# ── extraction + distillation ────────────────────────────────────────────────

def test_extract_strips_scripts_and_tags():
    eng = _engine()
    doc = asyncio.run(eng.fetch("https://example.com/mars"))
    assert doc.ok
    assert doc.title == "Mars Facts"
    assert "var x" not in doc.text
    assert "Mars is the fourth planet" in doc.text


def test_distill_keeps_declarative_facts():
    eng = _engine()
    report = eng.ingest_text(
        "Mars is the fourth planet from the Sun and has two small moons. "
        "The atmosphere is mostly carbon dioxide and is very thin. Click here.",
        source="unit",
    )
    assert any("fourth planet" in f for f in report.facts)
    assert all("Click here" not in f for f in report.facts)  # boilerplate dropped


# ── unrestricted reads + ingestion into sinks ────────────────────────────────

def test_ingest_url_writes_beliefs_and_memory():
    beliefs, memories = {}, []
    eng = WorldIngestionEngine(
        fetcher=lambda url: (200, _HTML),
        belief_sink=lambda k, v, c: beliefs.__setitem__(k, (v, c)),
        memory_sink=lambda content, meta: memories.append((content, meta)),
        min_requests_interval_s=0.0,
    )
    report = asyncio.run(eng.ingest_url("https://anything.example/page", source_trust=0.7))
    assert report.beliefs_written > 0
    assert report.memories_written > 0
    assert beliefs  # belief sink received world_fact:* keys
    assert all(k.startswith("world_fact:") for k in beliefs)
    assert memories and memories[0][1]["source"].endswith("page")


def test_search_parses_results():
    page = (
        '<a class="result__a" href="https://a.com">Alpha result</a>'
        '<a class="result__a" href="https://b.com">Beta result</a>'
    )
    eng = WorldIngestionEngine(fetcher=lambda url: (200, page), min_requests_interval_s=0.0)
    results = asyncio.run(eng.search("anything", limit=5))
    assert len(results) == 2
    assert results[0].url == "https://a.com" and "Alpha" in results[0].title


def test_failed_fetch_reports_error_not_crash():
    eng = WorldIngestionEngine(fetcher=lambda url: (404, ""), min_requests_interval_s=0.0)
    report = asyncio.run(eng.ingest_url("https://missing.example"))
    assert report.error.startswith("fetch_failed")


def test_fetcher_exception_is_degraded_not_raised():
    def _boom(url):
        raise RuntimeError("dns blew up")

    eng = WorldIngestionEngine(fetcher=_boom, min_requests_interval_s=0.0)
    doc = asyncio.run(eng.fetch("https://x"))
    assert doc.status == 0 and doc.text == ""


# ── governed state-changing writes ───────────────────────────────────────────

def test_get_via_state_changing_request_is_treated_as_read():
    eng = _engine()
    out = asyncio.run(eng.state_changing_request("GET", "https://example.com"))
    assert out["allowed"] is True


def test_irreversible_post_blocked_when_governance_denies(monkeypatch):
    import core.values.value_model as vmod

    class _Judgment:
        permitted = True
        requires_confirmation = True  # irreversible + unconfirmed → hold

    class _VM:
        def evaluate_with_will(self, action):
            return _Judgment()

    monkeypatch.setattr(vmod, "get_value_model", lambda: _VM())
    eng = _engine()
    out = asyncio.run(eng.state_changing_request("POST", "https://api.example/do",
                                                 reversible=False, confirmed=False))
    assert out["allowed"] is False
    assert out["reason"] == "blocked_by_governance"


def test_confirmed_reversible_post_allowed(monkeypatch):
    import core.values.value_model as vmod

    class _Judgment:
        permitted = True
        requires_confirmation = False

    class _VM:
        def evaluate_with_will(self, action):
            return _Judgment()

    monkeypatch.setattr(vmod, "get_value_model", lambda: _VM())
    eng = WorldIngestionEngine(fetcher=lambda url: (200, "ok"), min_requests_interval_s=0.0)
    out = asyncio.run(eng.state_changing_request("POST", "https://api.example/do",
                                                 reversible=True, confirmed=True))
    assert out["allowed"] is True and out["status"] == 200


def test_governance_unavailable_fails_closed_on_writes(monkeypatch):
    import core.values.value_model as vmod

    def _raise():
        raise RuntimeError("no value model")

    monkeypatch.setattr(vmod, "get_value_model", _raise)
    eng = _engine()
    out = asyncio.run(eng.state_changing_request("DELETE", "https://api.example/x"))
    assert out["allowed"] is False  # writes fail closed without governance


# ── anomaly → hypothesis ─────────────────────────────────────────────────────

def test_contradicting_fact_flagged_as_anomaly(monkeypatch):
    eng = WorldIngestionEngine(fetcher=lambda url: (200, _HTML), min_requests_interval_s=0.0)
    # Pretend we already believe something different for one fact's key.
    monkeypatch.setattr(eng, "_contradicts_belief", lambda fact: "fourth planet" in fact)
    raised = []
    monkeypatch.setattr(eng, "_raise_hypothesis", lambda fact, source: raised.append(fact))
    report = eng.ingest_text("Mars is the fourth planet from the Sun and has two small moons.",
                             source="unit")
    assert report.anomalies and raised


# ── singleton ────────────────────────────────────────────────────────────────

def test_singleton_is_stable():
    assert get_world_ingestion_engine() is get_world_ingestion_engine()
