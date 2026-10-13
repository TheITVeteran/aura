"""What she learned about a world is kept in a root that has never held it.

`remember` made the directory before it opened its governed scope and wrote
inside it. On the desktop the directory already existed, so the only sign was a
governance violation logged on every call. In a run with a fresh state root,
which is every campaign, the directory was refused under enforced governance,
nothing was written, and `recall` came back empty for the whole run.
"""

from __future__ import annotations

import pytest

from core.runtime import what_she_learned

pytestmark = pytest.mark.unit


def test_a_fresh_root_keeps_what_she_learned_under_enforced_governance(tmp_path, monkeypatch) -> None:
    kept_in = tmp_path / "state" / "worlds"
    monkeypatch.setattr(what_she_learned, "_KEPT_IN", kept_in)
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    assert not kept_in.exists()
    assert what_she_learned.remember("the tower", {"moves": ["left", "up"]}) is True
    assert what_she_learned.recall("the tower") == {"moves": ["left", "up"]}


def test_a_record_read_once_is_not_read_again_until_it_is_written_and_each_reader_has_its_own_copy(tmp_path, monkeypatch):
    """Offline 2026-10-12 every decision read and checked every record she had kept, a thirtieth of a game's play."""
    import core.runtime.what_she_learned as kept

    monkeypatch.setattr(kept, "_KEPT_IN", tmp_path)
    monkeypatch.setattr(kept, "_RECALLED", {})
    assert kept.remember("a world", {"moves": {"right": [1, 2]}})
    reads = []
    original = kept.validate_indexed_state
    monkeypatch.setattr(kept, "validate_indexed_state", lambda held, **k: reads.append(1) or original(held, **k))
    first = kept.recall("a world")
    first["moves"]["right"].append(99)                       # her copy, not the kept one
    again = kept.recall("a world")
    assert again == {"moves": {"right": [1, 2]}} and len(reads) == 1
    import os
    import time
    assert kept.remember("a world", {"moves": {"right": [3]}})
    os.utime(tmp_path / f"{kept.named('a world')}.json", ns=(time.time_ns(), time.time_ns() + 10_000_000))
    written = len(reads)
    assert kept.recall("a world") == {"moves": {"right": [3]}} and len(reads) == written + 1
    assert kept.recall("a world") == {"moves": {"right": [3]}} and len(reads) == written + 1
