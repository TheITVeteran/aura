"""The retry queue must not turn into a duplicate generator.

Two defects, and they compound. The first commit path consults the dedup
ledger before writing; ``replay_queue`` did not — so every retry sweep
re-committed the same episode, fact or belief into real memory. And the final
queue rewrite swallowed its failure with ``pass  # no-op: intentional``, so a
queue that could not be drained kept every already-committed record on disk
and replayed it forever, silently.

Either alone is a bug. Together they are unbounded duplication of Aura's
memory with no signal that it is happening.
"""
from __future__ import annotations

import json

import pytest

from core.autonomy import memory_persister as module
from core.autonomy.memory_persister import (
    BeliefUpdate,
    EpisodicEvent,
    FactRecord,
    MemoryPersister,
)


@pytest.fixture()
def persister(tmp_path):
    return MemoryPersister(
        queue_path=tmp_path / "queue.jsonl",
        dedup_path=tmp_path / "dedup.json",
    )


def _queue(persister, kind, payload, title="t"):
    persister._queue_path.parent.mkdir(parents=True, exist_ok=True)
    persister._queue_path.write_text(
        json.dumps({"kind": kind, "item_title": title, "payload": payload}) + "\n",
        encoding="utf-8",
    )


_FACT = {"fact": "the kettle boils at 100C", "confidence": 0.9}


def _accept_all(persister, calls):
    """Make every commit path succeed and count the calls."""
    def _ok(_title, record):
        calls.append(record)
        return True, "id", None

    persister._commit_episodic = _ok
    persister._commit_fact = _ok
    persister._commit_belief = _ok


# --- replay consults the dedup ledger -----------------------------------


def test_an_already_committed_record_is_not_committed_again(persister):
    calls = []
    _accept_all(persister, calls)
    _queue(persister, "fact", _FACT)
    persister._mark_committed(FactRecord(**_FACT).hash_key())

    committed = persister.replay_queue()

    assert calls == []
    assert committed == 0


def test_a_duplicate_is_drained_rather_than_left_to_repeat(persister):
    """Leaving it queued is what produced the repeat in the first place."""
    _accept_all(persister, [])
    _queue(persister, "fact", _FACT)
    persister._mark_committed(FactRecord(**_FACT).hash_key())

    persister.replay_queue()

    assert persister._queue_path.read_text(encoding="utf-8").strip() == ""


def test_replaying_twice_commits_once(persister):
    """The defect, end to end: two sweeps used to mean two memories."""
    calls = []
    _accept_all(persister, calls)
    _queue(persister, "fact", _FACT)

    first = persister.replay_queue()
    _queue(persister, "fact", _FACT)  # the record is offered again
    second = persister.replay_queue()

    assert first == 1
    assert second == 0
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("fact", _FACT),
        ("episodic", {"summary": "did a thing", "started_at": 1000.0}),
        ("belief", {"topic": "t", "position": "p", "rationale": "r", "confidence": 0.5}),
    ],
)
def test_every_tier_is_deduplicated_on_replay(persister, kind, payload):
    calls = []
    _accept_all(persister, calls)
    _queue(persister, kind, payload)

    persister.replay_queue()
    _queue(persister, kind, payload)
    persister.replay_queue()

    assert len(calls) == 1


def test_a_successful_commit_is_marked_before_the_queue_is_rewritten(persister):
    """Order matters: a crash between the two must not re-commit."""
    import inspect

    source = inspect.getsource(MemoryPersister.replay_queue)
    mark_at = source.index("_save_dedup()")
    rewrite_at = source.index("atomic_write_text")
    assert mark_at < rewrite_at


# --- a failed queue rewrite is reported ---------------------------------


def test_a_failed_queue_rewrite_is_recorded_not_swallowed(persister, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        module, "record_degradation",
        lambda *a, **k: recorded.append(k.get("action", "")) or object(),
    )
    monkeypatch.setattr(
        module, "atomic_write_text",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    _accept_all(persister, [])
    _queue(persister, "fact", _FACT)

    committed = persister.replay_queue()

    assert committed == 1  # the commit itself still succeeded
    assert any("not draining" in action for action in recorded)


def test_the_silent_no_op_is_gone():
    """Behaviour-adjacent: the explanatory comment necessarily quotes the
    line it removed, so scan CODE only."""
    import inspect

    code = "\n".join(
        line for line in inspect.getsource(MemoryPersister.replay_queue).splitlines()
        if not line.strip().startswith("#")
    )
    assert "pass  # no-op: intentional" not in code


def test_a_failed_dedup_save_is_reported(persister, monkeypatch):
    """The dedup ledger is what prevents re-commit; losing it silently
    reintroduces the duplication this whole change exists to stop."""
    recorded = []
    monkeypatch.setattr(
        module, "record_degradation",
        lambda *a, **k: recorded.append(k.get("action", "")) or object(),
    )
    monkeypatch.setattr(
        module, "atomic_write_text",
        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only fs")),
    )

    persister._mark_committed("abc")
    persister._save_dedup()

    assert any("dedup marks" in action for action in recorded)


def test_a_failed_dedup_save_does_not_abort_the_sweep(persister, monkeypatch):
    monkeypatch.setattr(
        module, "atomic_write_text",
        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only fs")),
    )
    _accept_all(persister, [])
    _queue(persister, "fact", _FACT)

    # Must not raise: a maintenance sweep that dies abandons everything
    # queued behind the record it choked on.
    assert persister.replay_queue() == 1


def test_dedup_marks_survive_a_failed_rewrite(persister, monkeypatch):
    """This is what makes the failure survivable: the record stays queued,
    but the mark stops it committing twice."""
    calls = []
    _accept_all(persister, calls)
    _queue(persister, "fact", _FACT)
    real_write = module.atomic_write_text
    monkeypatch.setattr(
        module, "atomic_write_text",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )

    persister.replay_queue()
    monkeypatch.setattr(module, "atomic_write_text", real_write)
    persister.replay_queue()  # the same record is still on disk

    assert len(calls) == 1


# --- malformed and forward-compatible entries ---------------------------


def test_an_unknown_kind_does_not_crash_the_sweep(persister):
    calls = []
    _accept_all(persister, calls)
    persister._queue_path.parent.mkdir(parents=True, exist_ok=True)
    persister._queue_path.write_text(
        json.dumps({"kind": "from_the_future", "item_title": "t", "payload": {}}) + "\n"
        + json.dumps({"kind": "fact", "item_title": "t", "payload": _FACT}) + "\n",
        encoding="utf-8",
    )

    committed = persister.replay_queue()

    assert committed == 1  # the good record still went through


def test_an_unparseable_payload_does_not_crash_the_sweep(persister):
    calls = []
    _accept_all(persister, calls)
    persister._queue_path.parent.mkdir(parents=True, exist_ok=True)
    persister._queue_path.write_text(
        json.dumps({"kind": "fact", "item_title": "t", "payload": {"fact": None}}) + "\n"
        + json.dumps({"kind": "fact", "item_title": "t", "payload": _FACT}) + "\n",
        encoding="utf-8",
    )

    assert persister.replay_queue() >= 1


def test_a_failed_commit_stays_queued(persister):
    """Retry must still work — dedup only skips what actually committed."""
    def _fail(_title, _record):
        return False, "", None

    persister._commit_fact = _fail
    _queue(persister, "fact", _FACT)

    committed = persister.replay_queue()

    assert committed == 0
    assert "kettle" in persister._queue_path.read_text(encoding="utf-8")


def test_replay_record_is_forward_compatible():
    assert module._replay_record("from_the_future", {}) is None
    assert isinstance(module._replay_record("fact", _FACT), FactRecord)
    assert isinstance(
        module._replay_record("episodic", {"summary": "s", "started_at": 1.0}),
        EpisodicEvent,
    )
    assert isinstance(
        module._replay_record(
            "belief", {"topic": "t", "position": "p", "rationale": "r", "confidence": 0.1}
        ),
        BeliefUpdate,
    )
