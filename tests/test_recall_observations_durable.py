"""Live recall evidence survives process boundaries without entering recall I/O."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.memory.recall_observations import RecallObservationRing
from tools.fit_actr_retrieval import _observed_samples


def test_durable_observations_are_readable_by_a_fresh_owner(tmp_path):
    path = tmp_path / "recalls.db"
    live = RecallObservationRing(db_path=path, background_persistence=False)
    for offset in range(30):
        live.record(
            [offset + 0.9, offset + 0.4, offset - 0.2, offset - 0.8],
            returned_count=2,
        )
    assert live.flush() == 120

    separate_process_equivalent = RecallObservationRing(db_path=path, persistence=False)
    assert separate_process_equivalent.load_persisted() == 120
    assert separate_process_equivalent.stats()["rankings"] == 30
    assert separate_process_equivalent.samples() == live.samples()


def test_observed_fitter_reads_durable_store_not_process_singleton(tmp_path):
    path = tmp_path / "recalls.db"
    live = RecallObservationRing(db_path=path, background_persistence=False)
    for offset in range(30):
        live.record(
            [offset + 1.0, offset + 0.5, offset, offset - 0.5],
            returned_count=1,
        )
    live.flush()

    samples, stats = _observed_samples(path)

    assert len(samples) == 120
    assert stats["rankings"] == 30
    assert stats["pending_persistence"] == 0


def test_durable_store_and_pending_queue_are_bounded(tmp_path):
    path = tmp_path / "recalls.db"
    ring = RecallObservationRing(capacity=5, db_path=path, background_persistence=False)
    ring.record([9.0, 8.0, 7.0, 6.0], returned_count=2)
    ring.flush()
    ring.record([5.0, 4.0, 3.0, 2.0], returned_count=2)
    ring.flush()

    reader = RecallObservationRing(capacity=5, db_path=path, persistence=False)
    assert reader.load_persisted() == 5
    assert [obs.activation for obs in reader.observations()] == [6.0, 5.0, 4.0, 3.0, 2.0]
    assert ring.stats()["dropped_pending"] == 0


def test_schema_cannot_store_memory_content_or_identity(tmp_path):
    path = tmp_path / "recalls.db"
    ring = RecallObservationRing(db_path=path, background_persistence=False)
    ring.record([0.8, 0.2], returned_count=1)
    ring.flush()

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(recall_observations)")}

    assert columns == {
        "sequence",
        "activation",
        "rank",
        "candidates",
        "returned",
        "recorded_at",
    }
    assert ring.stats()["content_fields_stored"] == []


def test_prelabel_rows_remain_unmeasured_during_schema_upgrade(tmp_path):
    path = tmp_path / "recalls.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE recall_observations (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                activation REAL NOT NULL,
                rank INTEGER NOT NULL,
                candidates INTEGER NOT NULL,
                recorded_at REAL NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO recall_observations "
            "(activation, rank, candidates, recorded_at) VALUES (0.5, 0, 2, 1.0)"
        )

    writer = RecallObservationRing(db_path=path, background_persistence=False)
    writer.record([0.9, 0.1], returned_count=1)
    assert writer.flush() == 2

    reader = RecallObservationRing(db_path=path, persistence=False)
    assert reader.load_persisted() == 2
    assert reader.samples() == [(0.9, 1), (0.1, 0)]


@pytest.mark.asyncio
async def test_async_recording_flushes_off_loop_in_shared_io_pool(tmp_path):
    ring = RecallObservationRing(db_path=tmp_path / "recalls.db")
    ring.record([0.9, 0.3, -0.1], returned_count=1)

    for _ in range(100):
        if ring.stats()["pending_persistence"] == 0:
            break
        await asyncio.sleep(0.01)

    stats = ring.stats()
    assert stats["pending_persistence"] == 0
    assert stats["persisted_observations"] == 3
    assert stats["flush_failures"] == 0


def test_worker_thread_recording_persists_without_an_event_loop(tmp_path):
    path = tmp_path / "recalls.db"
    ring = RecallObservationRing(db_path=path)

    worker = threading.Thread(
        target=lambda: ring.record([0.9, 0.4, -0.2], returned_count=1),
        name="recall-worker-without-loop",
    )
    worker.start()
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    deadline = time.monotonic() + 2.0
    while ring.stats()["pending_persistence"] and time.monotonic() < deadline:
        time.sleep(0.01)

    reader = RecallObservationRing(db_path=path, persistence=False)
    assert reader.load_persisted() == 3
    assert [sample.activation for sample in reader.observations()] == [0.9, 0.4, -0.2]


def test_concurrent_records_have_one_coalesced_persistence_owner(tmp_path, monkeypatch):
    ring = RecallObservationRing(db_path=tmp_path / "recalls.db")
    active = 0
    peak_active = 0
    activity_lock = threading.Lock()
    original = ring._persist_batch

    def observed_persist(batch):
        nonlocal active, peak_active
        with activity_lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            time.sleep(0.01)
            return original(batch)
        finally:
            with activity_lock:
                active -= 1

    monkeypatch.setattr(ring, "_persist_batch", observed_persist)
    with ThreadPoolExecutor(max_workers=8) as callers:
        futures = [
            callers.submit(
                ring.record,
                [float(i), float(i) - 0.5],
                returned_count=1,
            )
            for i in range(40)
        ]
        assert all(future.result(timeout=2.0) == 2 for future in futures)

    assert ring.flush() == 80
    assert peak_active == 1


def test_persistence_failure_never_fails_recall_and_is_observable(tmp_path, monkeypatch):
    ring = RecallObservationRing(db_path=tmp_path / "recalls.db", background_persistence=False)

    def broken(_batch):
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(ring, "_persist_batch", broken)
    assert ring.record([0.7, 0.2], returned_count=1) == 2
    assert ring.flush() == 0
    stats = ring.stats()
    assert stats["observations"] == 2
    assert stats["pending_persistence"] == 2
    assert stats["flush_failures"] == 1
    assert stats["retry_in_s"] > 0.0


def test_invalid_activation_is_skipped_without_losing_the_ranking(tmp_path):
    ring = RecallObservationRing(db_path=tmp_path / "recalls.db", persistence=False)
    assert ring.record([1.0, "bad", float("nan"), 0.2], returned_count=1) == 2
    assert [obs.rank for obs in ring.observations()] == [0, 3]
    assert [obs.returned for obs in ring.observations()] == [True, False]
