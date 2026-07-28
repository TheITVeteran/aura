"""A reused event id must not erase a different mutation from the ledger.

CP126 (high), core/brain/live_mind_contract.py: "Caller-controlled event IDs
can suppress distinct mutations. Merge identity is only the presence and
text of event_id. Reusing another event's ID causes a different mutation to
be silently dropped, with no comparison of its hashes or metadata."

This ledger is the audit trail for what was done to a user-facing answer —
which stage rewrote it, how many characters went in and out, the before and
after hashes. The id is caller-supplied, so any caller that reuses one (a
retry, a copied receipt, a replayed envelope) deleted a genuinely different
mutation from the record, and what remained still read as a complete
history.

The dedupe itself is wanted: copied receipts really do arrive twice. So the
two cases are separated. Same id and same content is the duplicate this
exists to drop. Same id and DIFFERENT content is a collision — the entry is
kept under its content identity and the collision is recorded, because
losing a real mutation from an audit ledger is worse than carrying a
redundant one.
"""
from __future__ import annotations

import pytest

from core.brain.live_mind_contract import merge_text_mutations


def _mutation(event_id: str, stage: str, *, after_sha: str = "1" * 64) -> dict:
    return {
        "event_id": event_id,
        "stage": stage,
        "method": "shape",
        "reasons": ["voice"],
        "deterministic": True,
        "before_chars": 10,
        "after_chars": 20,
        "before_sha256": "0" * 64,
        "after_sha256": after_sha,
    }


class TestGenuineDuplicatesStillCollapse:
    def test_the_same_event_twice_is_one_entry(self):
        merged = merge_text_mutations(
            [_mutation("E1", "shape")], [_mutation("E1", "shape")],
        )
        assert len(merged) == 1

    def test_identical_legacy_entries_still_collapse(self):
        merged = merge_text_mutations(
            [_mutation("", "shape")], [_mutation("", "shape")],
        )
        assert len(merged) == 1


class TestCollisionsDoNotEraseHistory:
    def test_a_reused_id_with_different_content_keeps_both(self):
        """The defect: one of these used to vanish."""
        merged = merge_text_mutations(
            [_mutation("E1", "shape")], [_mutation("E1", "truncate")],
        )
        assert len(merged) == 2
        assert {m["stage"] for m in merged} == {"shape", "truncate"}

    def test_a_differing_hash_alone_is_enough_to_keep_both(self):
        """Content identity includes the hashes, not just the stage — two
        rewrites at the same stage with different output are different."""
        merged = merge_text_mutations(
            [_mutation("E1", "shape", after_sha="1" * 64)],
            [_mutation("E1", "shape", after_sha="2" * 64)],
        )
        assert len(merged) == 2

    def test_the_collision_is_recorded_not_silent(self, monkeypatch):
        # Patch where the name is USED, not where it is defined: the module
        # does `from core.runtime.errors import record_degradation`, so the
        # reference is bound at import and patching the source module has
        # no effect on it.
        import core.brain.live_mind_contract as mod

        recorded: list = []
        monkeypatch.setattr(
            mod, "record_degradation", lambda *a, **k: recorded.append(a),
        )
        merge_text_mutations(
            [_mutation("E1", "shape")], [_mutation("E1", "truncate")],
        )
        assert recorded, "an id collision was handled silently"

    def test_three_way_collision_keeps_all_distinct_content(self):
        merged = merge_text_mutations(
            [_mutation("E1", "shape")],
            [_mutation("E1", "truncate")],
            [_mutation("E1", "repair")],
        )
        assert len(merged) == 3


class TestTheLedgerStaysWellFormed:
    def test_sequences_are_renumbered_contiguously(self):
        merged = merge_text_mutations(
            [_mutation("E1", "shape")], [_mutation("E2", "truncate")],
        )
        assert [m["sequence"] for m in merged] == [1, 2]

    def test_order_is_preserved(self):
        merged = merge_text_mutations(
            [_mutation("E1", "first")], [_mutation("E2", "second")],
        )
        assert [m["stage"] for m in merged] == ["first", "second"]

    def test_empty_input_is_safe(self):
        assert merge_text_mutations() == []
        assert merge_text_mutations([], []) == []

    def test_the_ledger_stays_bounded(self):
        from core.brain.live_mind_contract import _MAX_TEXT_MUTATIONS

        many = [_mutation(f"E{i}", f"stage{i}") for i in range(_MAX_TEXT_MUTATIONS + 50)]
        assert len(merge_text_mutations(many)) <= _MAX_TEXT_MUTATIONS
