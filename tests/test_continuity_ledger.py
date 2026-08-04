"""The ledger's contract: bounded, non-decaying, attributed.

These are the three properties the truncating rolling summary violated, and
each one maps to a way the live conversation actually broke.
"""

from __future__ import annotations

import pytest

from core.brain.llm.continuity_ledger import ContinuityLedger, LedgerEntry


def _chatter(n: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(n):
        out.append({"role": "user", "content": f"Filler question {i} about nothing at all?"})
        out.append({"role": "assistant", "content": f"Filler reply {i} saying very little."})
    return out


DISCLOSURE = "I've always wanted to teach myself physics and get really good at it"


def _seeded() -> ContinuityLedger:
    ledger = ContinuityLedger()
    ledger.observe([{"role": "user", "content": DISCLOSURE}])
    return ledger


def test_early_disclosure_survives_deep_burial():
    """The physics sentence must outlive 200 turns of filler.

    This is the live failure: she asked what he always wanted to learn, he
    said physics, and forty turns later nothing in her context knew it.
    """
    ledger = _seeded()
    ledger.observe(_chatter(200))
    assert "physics" in ledger.render().lower()


def test_render_never_exceeds_its_budget():
    ledger = _seeded()
    ledger.observe(_chatter(400))
    for budget in (0, 120, 400, 1000, 3200, 20000):
        assert len(ledger.render(budget)) <= budget, budget


def test_retained_entries_are_byte_identical_not_re_summarised():
    """Non-decay: what survives is what was written.

    The previous design re-truncated its own summary every compaction, so a
    sentence degraded into a prefix of a prefix. Nothing here may shorten a
    retained entry.
    """
    ledger = _seeded()
    for _ in range(20):
        ledger.observe(_chatter(10))
    rendered = ledger.render(20000)
    assert DISCLOSURE in rendered


def test_budget_pressure_drops_whole_entries_and_says_so():
    ledger = _seeded()
    ledger.observe(_chatter(50))
    tight = ledger.render(600)
    assert len(tight) <= 600
    # Whole lines only — no entry may be cut mid-sentence.
    for line in tight.splitlines():
        if line.startswith("- ["):
            assert not line.endswith("…")
    assert "not shown" in tight


def test_pinned_entries_are_never_evicted():
    ledger = ContinuityLedger()
    ledger.pin("The flight to Lisbon is on the 14th")
    ledger.observe(_chatter(500))
    assert "Lisbon" in ledger.render(20000)


def test_repeated_statements_do_not_duplicate():
    ledger = ContinuityLedger()
    for _ in range(5):
        ledger.observe([{"role": "user", "content": DISCLOSURE}])
    entries = [e for e in ledger.entries if "physics" in e.text.lower()]
    assert len(entries) == 1
    assert entries[0].mentions == 5


def test_round_trip_persistence_preserves_entries():
    ledger = _seeded()
    ledger.observe(_chatter(20))
    restored = ContinuityLedger.from_dict(ledger.to_dict())
    assert restored.render(20000) == ledger.render(20000)
    assert restored.turn == ledger.turn


def test_malformed_persisted_payload_does_not_raise():
    assert ContinuityLedger.from_dict(None).entries == []
    assert ContinuityLedger.from_dict({"entries": ["not a dict"]}).entries == []


def test_speaker_name_is_not_hardcoded():
    """Her record of a person must use that person's name, not a baked-in one."""
    ledger = _seeded()
    assert "Ada" in ledger.render(20000, speaker_name="Ada")


def test_capacity_is_bounded_under_unbounded_input():
    ledger = ContinuityLedger()
    ledger.observe(_chatter(5000))
    from core.brain.llm.continuity_ledger import ledger_capacity

    assert len(ledger.entries) <= ledger_capacity()


def test_salience_prefers_disclosure_over_stale_topic():
    ledger = ContinuityLedger()
    ledger.observe([{"role": "user", "content": DISCLOSURE}])
    disclosure = ledger.entries[0]
    topic = LedgerEntry(
        kind="subject", text="Some passing topic", speaker="user",
        first_turn=1, last_turn=1,
    )
    assert disclosure.salience(300) > topic.salience(300)
