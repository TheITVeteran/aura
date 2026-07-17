"""Contract tests: the epistemic firewall guarding latent-cortex ingress.

Deep recurrence amplifies its seeds, so admission is the safety boundary:
- corroboration counts independent sources, not repetitions;
- observed facts outrank claims, claims outrank inferences;
- irreconcilable conflicts refuse BOTH sides and force a felt caution;
- the whole decision is receipted and deterministic.
"""
from __future__ import annotations

import pytest

from core.brain.cognitive_ingress import (
    assemble_cognitive_ingress,
    cognitive_context_items,
)
from core.brain.epistemic_firewall import (
    EPISTEMIC_FIREWALL_SCHEMA,
    EpistemicFirewall,
    EvidenceItem,
)

NOW = 1_700_000_000.0


def _item(text: str, origin: str, **kwargs) -> EvidenceItem:
    return EvidenceItem(text=text, origin=origin, **kwargs)


# ── Independence clustering ─────────────────────────────────────────────


def test_duplicate_reports_collapse_to_one_independent_source():
    firewall = EpistemicFirewall()
    verdict = firewall.review(
        "database migration status",
        [
            _item("the database migration finished cleanly last night", "feed_a"),
            _item("the database migration finished cleanly last night", "feed_b"),
            _item("the database migration finished cleanly last night", "feed_c"),
        ],
        now=NOW,
    )
    assert len(verdict.admitted) == 1
    assert verdict.admitted[0]["cluster_size"] == 3
    assert verdict.admitted[0]["independent_sources"] == 3
    assert len(verdict.refused) == 2
    assert all("duplicate_of" in row["reason"] for row in verdict.refused)


def test_same_origin_repetition_is_never_corroboration():
    firewall = EpistemicFirewall()
    verdict = firewall.review(
        "server status",
        [
            _item("the primary server is healthy and serving traffic", "monitor#1"),
            _item("all disk arrays report nominal utilization today", "monitor#1"),
        ],
        now=NOW,
    )
    # Same origin ⇒ one cluster ⇒ one admitted representative.
    assert len(verdict.clusters) == 1
    assert len(verdict.admitted) == 1
    assert verdict.admitted[0]["independent_sources"] == 1


# ── Conflict graph + resolution ─────────────────────────────────────────


def test_numeric_disagreement_between_claims_forces_abstention():
    firewall = EpistemicFirewall()
    verdict = firewall.review(
        "how many replicas does the cluster run",
        [
            _item("the production cluster runs 12 database replicas", "source_a"),
            _item("the production cluster runs 3 database replicas", "source_b"),
        ],
        now=NOW,
    )
    assert verdict.abstain is True
    assert "unresolved_conflict" in verdict.reasons
    assert not verdict.admitted
    assert verdict.conflicts[0]["method"] == "numeric_disagreement"
    caution = verdict.caution_text()
    assert "conflict" in caution
    assert len(caution) <= 400


def test_observed_fact_outranks_conflicting_claim():
    firewall = EpistemicFirewall()
    verdict = firewall.review(
        "how many replicas does the cluster run",
        [
            _item(
                "the production cluster runs 12 database replicas",
                "tool_receipt",
                kind="observed_fact",
            ),
            _item("the production cluster runs 3 database replicas", "rumor_feed"),
        ],
        now=NOW,
    )
    assert verdict.abstain is False
    assert len(verdict.admitted) == 1
    assert verdict.admitted[0]["kind"] == "observed_fact"
    losers = [row for row in verdict.refused if row.get("reason") == "conflict_loser"]
    assert len(losers) == 1
    assert verdict.conflicts[0]["resolved_by"] == "provenance_rank"


def test_decisive_freshness_resolves_same_rank_conflicts():
    firewall = EpistemicFirewall()
    verdict = firewall.review(
        "current deployment version",
        [
            _item(
                "the deployment currently serves version 4 in production",
                "status_page",
                observed_at=NOW - 30.0,
            ),
            _item(
                "the deployment currently serves version 2 in production",
                "old_cache",
                observed_at=NOW - 7 * 24 * 3600.0,
            ),
        ],
        now=NOW,
    )
    assert verdict.abstain is False
    assert len(verdict.admitted) == 1
    assert verdict.admitted[0]["origin"] == "status_page"
    assert verdict.conflicts[0]["resolved_by"] == "decisive_freshness"


def test_polarity_disagreement_is_detected():
    firewall = EpistemicFirewall()
    verdict = firewall.review(
        "did the nightly backup complete",
        [
            _item("the nightly backup job completed for every volume", "log_a"),
            _item("the nightly backup job never completed for every volume", "log_b"),
        ],
        now=NOW,
    )
    assert verdict.conflicts
    assert verdict.conflicts[0]["method"] == "polarity_disagreement"
    assert verdict.abstain is True


# ── Freshness, coverage, bounds ─────────────────────────────────────────


def test_stale_evidence_is_flagged_but_not_silently_dropped():
    firewall = EpistemicFirewall()
    verdict = firewall.review(
        "archive layout",
        [
            _item(
                "the archive keeps quarterly snapshots under the vault directory",
                "old_notes",
                observed_at=NOW - 365 * 24 * 3600.0,
            )
        ],
        now=NOW,
    )
    assert len(verdict.admitted) == 1
    assert verdict.admitted[0]["stale"] is True


def test_thin_coverage_requests_more_retrieval():
    firewall = EpistemicFirewall(min_coverage=0.5)
    verdict = firewall.review(
        "compare the scheduler arbitration redesign against the allostasis engine",
        [_item("the cafeteria menu rotates weekly", "wiki")],
        now=NOW,
    )
    assert verdict.needs_more_retrieval is True
    assert "insufficient_coverage" in verdict.reasons
    assert verdict.uncovered_terms


def test_no_valid_evidence_fails_closed():
    firewall = EpistemicFirewall()
    verdict = firewall.review("anything", [], now=NOW)
    assert verdict.needs_more_retrieval is True
    assert "no_valid_evidence" in verdict.reasons
    assert not verdict.admitted

    bad = firewall.review(
        "anything",
        [EvidenceItem(text="", origin="x"), EvidenceItem(text="ok text", origin="")],
        now=NOW,
    )
    assert not bad.admitted
    assert any(reason.startswith("invalid_item") for reason in bad.reasons)


def test_item_overflow_is_clipped_and_receipted():
    firewall = EpistemicFirewall()
    items = [
        _item(
        f"{chr(97 + index) * 3} {chr(97 + index) * 4} report", f"src_{index}"
        )
        for index in range(24)
    ]
    verdict = firewall.review("topic", items, now=NOW)
    assert "item_overflow_clipped" in verdict.reasons
    assert len(verdict.admitted) <= 4
    budget_refusals = [
        row for row in verdict.refused if row.get("reason") == "slot_budget"
    ]
    assert budget_refusals, "past-budget representatives must be receipted"
    receipt = verdict.to_receipt()
    assert receipt["schema"] == EPISTEMIC_FIREWALL_SCHEMA
    assert receipt["admitted"] and receipt["refused"] is not None


def test_review_is_deterministic():
    firewall = EpistemicFirewall()
    items = [
        _item("service alpha handles ingest and fanout", "a"),
        _item("service beta handles archival and replay", "b"),
    ]
    first = firewall.review("services", items, now=NOW).to_receipt()
    second = firewall.review("services", items, now=NOW).to_receipt()
    assert first == second


def test_constructor_validates_thresholds():
    with pytest.raises(ValueError):
        EpistemicFirewall(duplicate_jaccard=0.0)
    with pytest.raises(ValueError):
        EpistemicFirewall(conflict_jaccard=0.9, duplicate_jaccard=0.5)
    with pytest.raises(ValueError):
        EpistemicFirewall(max_admitted=0)


# ── Ingress integration ─────────────────────────────────────────────────


@pytest.fixture()
def registry(monkeypatch):
    services: dict[str, object] = {}
    import core.brain.cognitive_ingress as ingress_mod

    monkeypatch.setattr(ingress_mod, "_get_service", lambda name: services.get(name))
    return services


def test_conflicting_recall_seeds_caution_not_a_winner(registry):
    class ConflictedMemory:
        def search(self, objective, limit=4):
            return [
                {"content": "the launch happened on day 12 of the month"},
                {"content": "the launch happened on day 3 of the month"},
            ]

    registry["memory_facade"] = ConflictedMemory()
    ingress = assemble_cognitive_ingress(None, "when did the launch happen")
    signal = next(s for s in ingress.signals if s.source == "memory")
    assert signal.present is True
    assert signal.firewall["abstain"] is True
    assert signal.context_text == ""
    assert "conflict" in signal.caution_text
    items = cognitive_context_items(ingress)
    sources = [item["source"] for item in items]
    assert "epistemic_caution" in sources
    assert "memory" not in sources


def test_conflicting_recall_raises_allocation_uncertainty(registry):
    class CleanMemory:
        def search(self, objective, limit=4):
            return [{"content": "the launch happened on day 12 of the month"}] * 2

    class ConflictedMemory:
        def search(self, objective, limit=4):
            return [
                {"content": "the launch happened on day 12 of the month"},
                {"content": "the launch happened on day 3 of the month"},
            ]

    registry["memory_facade"] = CleanMemory()
    clean = assemble_cognitive_ingress(None, "when did the launch happen")
    registry["memory_facade"] = ConflictedMemory()
    conflicted = assemble_cognitive_ingress(None, "when did the launch happen")
    assert conflicted.uncertainty > clean.uncertainty


def test_clean_recall_still_seeds_memory_slot(registry):
    class CleanMemory:
        def search(self, objective, limit=4):
            return [
                {
                    "content": "the launch happened on day 12 and shipped cleanly",
                    "verified": True,
                    "timestamp": NOW,
                }
            ]

    registry["memory_facade"] = CleanMemory()
    ingress = assemble_cognitive_ingress(None, "when did the launch happen")
    signal = next(s for s in ingress.signals if s.source == "memory")
    assert signal.context_text
    assert signal.firewall["admitted"][0]["kind"] == "observed_fact"
    assert signal.caution_text == ""
    items = cognitive_context_items(ingress)
    assert any(item["source"] == "memory" for item in items)
    assert all(item["source"] != "epistemic_caution" for item in items)
