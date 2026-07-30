"""A retry queue must not become a duplicate generator.

CP126 (critical), core/autonomy/memory_persister.py: "Queue replay can
duplicate every successful write. Replay does not check or mark dedup keys,
and a failed final queue rewrite is silently ignored; successful records
remain on disk and are committed again on every replay."

Two halves that compound. The first commit path consults ``_is_duplicate``
before writing; replay did not, so a queued record was re-committed on every
sweep. And the queue rewrite that drains committed records ended in
``pass  # no-op: intentional`` — so when it failed, every record just
committed stayed on disk and came back next time.

Together they turn a retry queue into an unbounded duplicate generator
against Aura's real episodic memory, facts and beliefs.
"""
from __future__ import annotations

import json

import pytest

from core.autonomy.memory_persister import FactRecord, MemoryPersister


@pytest.fixture
def persister(tmp_path):
    return MemoryPersister(
        queue_path=tmp_path / "queue.jsonl",
        dedup_path=tmp_path / "dedup.json",
    )


def _queue(persister, record: dict) -> None:
    persister._queue_path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _fact_record(text: str = "the sky is blue") -> dict:
    fact = FactRecord(fact=text, confidence=0.9)
    return {"kind": "fact", "item_title": "t", "payload": dict(fact.__dict__)}


class TestReplayCommitsOnce:
    def test_a_record_already_committed_is_not_committed_again(self, persister):
        """The exact scenario: the rewrite failed, so the record is back."""
        commits: list[str] = []
        persister._commit_fact = lambda _t, f: (commits.append(f.hash_key()) or (True, "", None))

        _queue(persister, _fact_record())
        assert persister.replay_queue() == 1

        _queue(persister, _fact_record())          # rewrite failed; record returns
        assert persister.replay_queue() == 0
        assert len(commits) == 1

    def test_the_duplicate_is_drained_not_left_queued(self, persister):
        """Leaving it queued is what produced the repeat in the first place."""
        persister._commit_fact = lambda _t, _f: (True, "", None)
        _queue(persister, _fact_record())
        persister.replay_queue()
        _queue(persister, _fact_record())
        persister.replay_queue()
        assert persister._queue_path.read_text(encoding="utf-8").strip() == ""

    def test_a_distinct_record_still_commits(self, persister):
        """Over-suppression would silently lose real memories."""
        commits: list[str] = []
        persister._commit_fact = lambda _t, f: (commits.append(f.hash_key()) or (True, "", None))
        _queue(persister, _fact_record("the sky is blue"))
        persister.replay_queue()
        _queue(persister, _fact_record("the grass is green"))
        assert persister.replay_queue() == 1
        assert len(commits) == 2


class TestFailedCommitsStayQueued:
    def test_a_failed_record_is_retained(self, persister):
        persister._commit_fact = lambda _t, _f: (False, "backend down", None)
        _queue(persister, _fact_record())
        assert persister.replay_queue() == 0
        assert "the sky is blue" in persister._queue_path.read_text(encoding="utf-8")

    def test_a_failed_record_is_not_marked_committed(self, persister):
        """Marking on failure would silently drop the memory forever."""
        persister._commit_fact = lambda _t, _f: (False, "backend down", None)
        _queue(persister, _fact_record())
        persister.replay_queue()

        commits: list[str] = []
        persister._commit_fact = lambda _t, f: (commits.append(f.hash_key()) or (True, "", None))
        assert persister.replay_queue() == 1
        assert len(commits) == 1


class TestTheRewriteFailureIsVisible:
    def test_a_failed_queue_rewrite_records_a_degradation(self, persister, monkeypatch):
        """Was `pass # no-op: intentional`. A queue that cannot drain is a
        real durability fault even though dedup now absorbs the duplicate."""
        import core.autonomy.memory_persister as mod

        recorded: list = []
        monkeypatch.setattr(mod, "record_degradation", lambda *a, **k: recorded.append(a))

        def _boom(*_a, **_kw):
            raise OSError("disk full")

        monkeypatch.setattr(mod, "atomic_write_text", _boom)
        persister._commit_fact = lambda _t, _f: (True, "", None)
        _queue(persister, _fact_record())
        persister.replay_queue()
        assert recorded, "a failed queue rewrite was silent"


class TestMalformedQueueEntries:
    def test_an_unknown_kind_is_skipped_not_crashed(self, persister):
        _queue(persister, {"kind": "from_the_future", "item_title": "t", "payload": {}})
        assert persister.replay_queue() == 0

    def test_an_unparseable_payload_is_skipped(self, persister):
        _queue(persister, {"kind": "fact", "item_title": "t", "payload": {"fact": None}})
        assert persister.replay_queue() == 0

    def test_an_empty_queue_is_safe(self, persister):
        assert persister.replay_queue() == 0
