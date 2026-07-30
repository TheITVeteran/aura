"""The last rung before the runtime kills itself must pull something.

2026-07-29, second Orca Demo attempt:

    🚨 [MEMWATCH] LETHAL ceiling: managed RSS 48237MB ≥ 43008MB.
       Reclaimed (killed=0). Next confirmation aborts.

Nothing was killed, and the process exited. Underneath it sat 18.3GB, 5.0GB
and 1.6GB of respawnable child. terminate_heavy_child_workers matched command
lines against ("mlx_worker.py", "MTLCompilerService"), but the resident 32B is
started through multiprocessing, so its command line is the generic spawn_main
bootstrap and no marker could ever appear in it.
"""
from __future__ import annotations

from typing import Any

import psutil  # noqa: F401 - fixture patches mw.psutil
import pytest

from core.resilience import memory_watchdog as mw

#: Verbatim from the live process (pid 31863, 18.3GB) on 2026-07-29.
_REAL_WORKER_CMD = (
    "/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/"
    "Versions/3.12/Resources/Python.app/Contents/MacOS/Python -c "
    "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=7, "
    "pipe_handle=9)"
)
_SENTINEL_CMD = (
    "/opt/homebrew/.../Python /Users/bryan/.aura/live-source/tools/"
    "memory_sentinel.py --pid 31812 --lethal-mb 43008.0 --interval 0.5"
)


class _FakeChild:
    def __init__(self, pid: int, rss_gb: float, cmd: str) -> None:
        self.pid = pid
        self._rss = int(rss_gb * (1024**3))
        self._cmd = cmd
        self.terminated = False

    def cmdline(self):
        return self._cmd.split(" ")

    def memory_info(self):
        return type("MI", (), {"rss": self._rss})()

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


@pytest.fixture
def live_tree(monkeypatch):
    children = [
        _FakeChild(31863, 18.3, _REAL_WORKER_CMD),
        _FakeChild(32308, 5.0, _REAL_WORKER_CMD),
        _FakeChild(32604, 1.6, _REAL_WORKER_CMD),
        _FakeChild(24822, 0.03, _SENTINEL_CMD),
    ]
    monkeypatch.setattr(mw, "_child_processes", lambda *a, **k: children)
    # memory_watchdog uses the repo's wrapped psutil, not the bare module.
    monkeypatch.setattr(
        mw.psutil, "wait_procs", lambda procs, timeout=0: (list(procs), [])
    )
    return children


def test_a_multiprocessing_worker_is_recognised_as_killable(live_tree) -> None:
    """The regression itself: killed=0 against a tree full of workers."""
    killed = mw.terminate_heavy_child_workers()
    assert killed >= 1, (
        "no worker was recognised — this is the 'Reclaimed (killed=0)' that "
        "preceded the abort"
    )


def test_the_out_of_process_sentinel_is_never_killed(live_tree) -> None:
    """It guards the thing being reclaimed; it must outlive the reclaim."""
    mw.terminate_heavy_child_workers()
    sentinel = next(c for c in live_tree if "memory_sentinel.py" in c._cmd)
    assert not sentinel.terminated


def test_reclaim_stops_once_the_shortfall_is_covered(live_tree) -> None:
    """Getting under the ceiling should cost one model reload, not the tree."""
    killed = mw.terminate_heavy_child_workers(
        free_at_least_bytes=int(6 * (1024**3))
    )
    assert killed == 1, f"killed {killed} workers to free 6GB"
    biggest = next(c for c in live_tree if c.pid == 31863)
    assert biggest.terminated, "largest-first: the 18.3GB worker goes first"
    assert not any(c.terminated for c in live_tree if c.pid in (32308, 32604))


def test_watchdog_asks_for_exactly_its_shortfall(monkeypatch) -> None:
    """The amount comes from the ceiling breach, not from a constant."""
    asked: dict[str, int] = {}

    def _terminator(*, free_at_least_bytes=None):
        asked["bytes"] = free_at_least_bytes
        return 1

    dog = mw.MemoryWatchdog(
        worker_terminator=_terminator,
        gc_collect=lambda: 0,
        ladder_shed=lambda: (0, 0),
        process_exit=lambda code: None,
    )
    sample = type("S", (), {"managed_rss_mb": 48237.0, "swap_used_gb": 0.0})()
    killed = dog._terminate_workers(sample, already_freed=0)

    expected = int((48237.0 - dog.thresholds.hard_mb) * (1024 * 1024))
    assert killed == 1
    assert asked["bytes"] == expected


def test_swap_driven_reclaim_still_sheds_workers(monkeypatch) -> None:
    """RSS under the ceiling does not mean there is nothing to reclaim.

    The hard tier also fires on swap exhaustion. Asking for a shortfall of
    zero bytes there would kill nothing — the same empty rung, wearing the
    budget as a disguise — so with no number to aim at it sheds everything
    eligible, which is what this tier always did.
    """
    asked: list[Any] = []

    def _terminator(*, free_at_least_bytes=None):
        asked.append(free_at_least_bytes)
        return 2

    dog = mw.MemoryWatchdog(
        worker_terminator=_terminator,
        gc_collect=lambda: 0,
        ladder_shed=lambda: (0, 0),
        process_exit=lambda code: None,
    )
    sample = type("S", (), {"managed_rss_mb": 1000.0, "swap_used_gb": 9.9})()

    assert dog._terminate_workers(sample) == 2, "swap pressure must still shed"
    assert asked == [None], "with no RSS breach there is no byte budget to ask for"
