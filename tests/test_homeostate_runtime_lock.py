"""The homeostate singletons must be constructible from a fresh state.

get_homeostate_reactor() acquires the module lock and then calls
get_homeostate_engine(), which acquires it again. With a plain
threading.Lock that nested acquire self-deadlocked the event loop at
boot PHASE 5.2 — every desktop boot wedged forever at
`homeostate_summary = await start_homeostate_runtime()` (reproduced
2026-07-24, 280+s of continuous 5s loop stalls). These tests run the
constructors on a worker thread with a hard join timeout so a
reintroduced deadlock fails the suite instead of hanging it.
"""
from __future__ import annotations

import threading


def _bounded(fn, timeout_s: float = 10.0):
    result: dict = {}

    def run():
        result["value"] = fn()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    assert not worker.is_alive(), (
        f"{fn.__name__} deadlocked: the module lock is not reentrant"
    )
    return result.get("value")


def test_reactor_and_scheduler_construct_without_deadlock():
    from core.runtime.homeostate import (
        get_convergence_scheduler,
        get_homeostate_reactor,
        reset_homeostate_for_test,
    )

    reset_homeostate_for_test()
    reactor = _bounded(get_homeostate_reactor)
    assert reactor is not None
    scheduler = _bounded(get_convergence_scheduler)
    assert scheduler is not None
    # Second calls return the same singletons, still without deadlock.
    assert _bounded(get_homeostate_reactor) is reactor
    assert _bounded(get_convergence_scheduler) is scheduler
