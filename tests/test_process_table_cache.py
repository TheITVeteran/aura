"""The host process-table scan must not run on every observation call.

The Jul 24 boot-stall dumps caught psutil.process_iter running ON the
event loop via resource_observation.processes(); a full host scan costs
seconds of syscalls on macOS, and several subsystems call it per tick.
A short-TTL cache bounds the cost for every caller at once; failures
are never cached so recovery retries immediately.
"""
from __future__ import annotations

from core.runtime.resource_observation import HostResourceObserver


def test_process_table_is_cached_within_ttl(monkeypatch):
    observer = HostResourceObserver()
    calls = {"n": 0}
    real_iter = __import__("psutil").process_iter

    def counting_iter(*args, **kwargs):
        calls["n"] += 1
        return real_iter(*args, **kwargs)

    monkeypatch.setattr("psutil.process_iter", counting_iter)
    first = observer.process_table()
    second = observer.process_table()
    assert calls["n"] == 1, "the second call within the TTL must hit the cache"
    assert second is first

    observer._process_table_cache = (
        observer._process_table_cache[0] - 10.0,
        observer._process_table_cache[1],
    )
    observer.process_table()
    assert calls["n"] == 2, "an expired cache must rescan"


def test_scan_failures_are_not_cached(monkeypatch):
    observer = HostResourceObserver()

    def broken_iter(*args, **kwargs):
        raise OSError("simulated scan failure")

    monkeypatch.setattr("psutil.process_iter", broken_iter)
    failed = observer.process_table()
    assert failed.available is False
    assert observer._process_table_cache is None, (
        "a failed scan must not be served to later callers"
    )
