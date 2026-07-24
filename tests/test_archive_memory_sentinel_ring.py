"""Contracts for append-only memory-sentinel proof archival."""

from __future__ import annotations

import json

import pytest

from tools import archive_memory_sentinel_ring as archiver
from tools.archive_memory_sentinel_ring import RingArchiveError, select_new_records


def _line(observed_at: float) -> bytes:
    return json.dumps(
        {
            "at": observed_at,
            "observation_source": "host",
            "guard_stage": "compute",
        },
        separators=(",", ":"),
    ).encode("ascii")


def _ring(*values: float) -> bytes:
    return b"".join(_line(value) + b"\n" for value in values)


def test_archiver_selects_only_strict_monotonic_suffix():
    selected = select_new_records(
        _ring(1.0, 2.0, 3.0),
        last_at=2.0,
        source_replaced=False,
    )
    assert [value for value, _line_raw in selected] == [3.0]


def test_archiver_accepts_compaction_with_overlap():
    selected = select_new_records(
        _ring(2.0, 3.0, 4.0),
        last_at=3.0,
        source_replaced=True,
    )
    assert [value for value, _line_raw in selected] == [4.0]


def test_archiver_rejects_rotation_gap():
    with pytest.raises(RingArchiveError, match="source_ring_rotation_gap"):
        select_new_records(
            _ring(4.0, 5.0),
            last_at=3.0,
            source_replaced=True,
        )


@pytest.mark.parametrize(
    "raw,error",
    [
        (_ring(2.0, 1.0), "source_ring_order_invalid"),
        (_line(1.0), "source_ring_partial_record"),
        (b"{}\n", "source_sample_invalid"),
    ],
)
def test_archiver_rejects_malformed_source(raw: bytes, error: str):
    with pytest.raises(RingArchiveError, match=error):
        select_new_records(raw, last_at=None, source_replaced=False)


def test_archiver_writes_self_hashed_terminal_receipt(tmp_path, monkeypatch):
    source = tmp_path / "ring.jsonl"
    source.write_bytes(_ring(1.0, 2.0))
    states = iter(("current", "gone"))
    monkeypatch.setattr(archiver, "_target_state", lambda *_args: next(states))
    monkeypatch.setattr(archiver.time, "sleep", lambda _seconds: None)
    # Identity is confirmed through the canonical resource observer, not psutil
    # directly — observe the target pid with the expected create_time.
    monkeypatch.setattr(
        "core.runtime.resource_observation.get_resource_observer",
        lambda: type(
            "Obs",
            (),
            {"process": lambda self, _pid: type("Proc", (), {"create_time": 100.0})()},
        )(),
    )
    destination = tmp_path / "archive.jsonl"
    state = tmp_path / "state.json"
    receipt = tmp_path / "receipt.json"

    result = archiver.archive(
        source=source,
        destination=destination,
        state_path=state,
        receipt_path=receipt,
        target_pid=123,
        interval_s=1.0,
    )

    assert destination.read_bytes() == source.read_bytes()
    assert result["sample_count"] == 2
    assert result["archive_sha256"] == archiver._sha(source.read_bytes())
    persisted = json.loads(receipt.read_text())
    material = dict(persisted)
    claimed = material.pop("receipt_sha256")
    assert claimed == archiver._sha(archiver._canonical(material))
    assert json.loads(state.read_text())["status"] == "passed"
