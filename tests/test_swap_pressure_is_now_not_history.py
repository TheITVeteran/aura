"""She declared swap exhaustion with 24.8 GB of RAM free.

Live 2026-07-28. Bryan: "she keeps resetting and erroring." The log said::

    RuntimeError: swap exhaustion: managed RSS 34494MB, swap 16.9GB

Measured on the same host at the same moment: 64% RAM used, **24.8 GB
available**, and the machine perfectly responsive. The reading was not wrong —
17.9 GB really had been written to swap — it was simply about the past.

``swap used`` is a high-water mark on macOS. Pages written to the swap file
stay accounted there long after the pressure that caused them is gone, and the
number does not fall when memory frees up. So ``swap_used >= swap_hard_gb``
latches: once a heavy session has swapped, every subsequent check with the
model resident fires, forever, on a number describing an event an hour old.

Each firing was a fail-closed CRITICAL. Those pinned ``deg_threat`` at 1.00,
which pinned existential threat, which is what the Ulysses covenant reads
before refusing heavy compute at 0.6 — so a swap spike that had already passed
went on silently refusing every build she was asked for.

The fix is to test headroom instead of history. Free space falls when pressure
is real and recovers when it passes, which is exactly the property the old
test lacked.
"""
from __future__ import annotations

import pytest

from core.resilience.memory_watchdog import (
    MemorySample,
    _swap_is_exhausted,
    _Thresholds,
)

THRESHOLDS = _Thresholds.from_environment(64.0)


def _sample(*, used_gb: float, free_gb: float) -> MemorySample:
    """A host with the model resident, varying only the swap picture."""
    return MemorySample(
        core_rss_mb=1200.0,
        child_rss_mb=33000.0,
        swap_used_gb=used_gb,
        system_percent=64.0,
        total_ram_gb=64.0,
        sampled_at=0.0,
        swap_free_gb=free_gb,
    )


def test_the_live_false_alarm_no_longer_fires() -> None:
    """17.9 GB written earlier, 9 GB of headroom now — this is not an emergency."""
    assert not _swap_is_exhausted(_sample(used_gb=17.9, free_gb=9.0), THRESHOLDS)


def test_a_genuinely_full_swap_file_still_fires() -> None:
    """The check must not be defanged — a full file is where the loop stalls."""
    assert _swap_is_exhausted(_sample(used_gb=8.0, free_gb=0.3), THRESHOLDS)


def test_a_healthy_host_is_quiet() -> None:
    assert not _swap_is_exhausted(_sample(used_gb=0.5, free_gb=18.0), THRESHOLDS)


@pytest.mark.parametrize("free_gb", [0.0, 0.1, 0.5, 1.0])
def test_no_headroom_is_always_an_emergency(free_gb: float) -> None:
    if free_gb == 0.0:
        pytest.skip("zero means 'unreadable' — covered by the fallback test")
    assert _swap_is_exhausted(_sample(used_gb=4.0, free_gb=free_gb), THRESHOLDS)


def test_history_alone_never_decides_it() -> None:
    """The whole defect in one assertion: same history, different present."""
    latched = _sample(used_gb=17.9, free_gb=12.0)
    real = _sample(used_gb=17.9, free_gb=0.4)
    assert not _swap_is_exhausted(latched, THRESHOLDS)
    assert _swap_is_exhausted(real, THRESHOLDS)


def test_an_unreadable_free_figure_falls_back_rather_than_going_blind() -> None:
    """Better a stale signal than a watchdog that can never fire."""
    assert _swap_is_exhausted(_sample(used_gb=17.9, free_gb=0.0), THRESHOLDS)
    assert not _swap_is_exhausted(_sample(used_gb=1.0, free_gb=0.0), THRESHOLDS)
