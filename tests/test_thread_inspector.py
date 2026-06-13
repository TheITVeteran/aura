"""Thread inventory observability — leak-hunt diagnosis surface."""
from __future__ import annotations

import threading

from core.runtime.thread_inspector import normalize_thread_name, thread_summary


def test_normalize_collapses_pool_worker_indices():
    assert normalize_thread_name("Aura.Events_3") == "Aura.Events"
    assert normalize_thread_name("ThreadPoolExecutor-2_0") == "ThreadPoolExecutor"
    assert normalize_thread_name("AuraVision-1") == "AuraVision"
    assert normalize_thread_name("asyncio_5") == "asyncio"
    # Names without worker indices are preserved.
    assert normalize_thread_name("MainThread") == "MainThread"
    assert normalize_thread_name("Aura.HeartBeat") == "Aura.HeartBeat"
    # Degenerate input.
    assert normalize_thread_name("") == "unnamed"
    assert normalize_thread_name(None) == "unnamed"  # type: ignore[arg-type]


def test_summary_shape_and_counts():
    summary = thread_summary()
    assert summary["total"] >= 1
    assert summary["daemon"] + summary["non_daemon"] == summary["total"]
    assert summary["distinct_groups"] >= 1
    assert isinstance(summary["groups"], dict)
    assert sum(summary["groups"].values()) <= summary["total"]
    # MainThread is always present and never collapsed away.
    assert "MainThread" in summary["groups"]


def test_summary_buckets_a_spawned_pool_under_one_group():
    started = threading.Event()
    release = threading.Event()

    def _worker():
        started.set()
        release.wait(timeout=5.0)

    workers = [
        threading.Thread(target=_worker, name=f"LeakProbeTest_{i}", daemon=True)
        for i in range(4)
    ]
    for w in workers:
        w.start()
    try:
        # Wait until at least one is live so the enumerate sees them.
        assert started.wait(timeout=5.0)
        summary = thread_summary(top=50)
        assert summary["groups"].get("LeakProbeTest", 0) >= 1
    finally:
        release.set()
        for w in workers:
            w.join(timeout=5.0)
