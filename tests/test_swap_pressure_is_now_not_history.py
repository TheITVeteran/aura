"""She declared swap exhaustion with 31.6 GB of RAM free.

Live 2026-07-28. Bryan: "she keeps resetting and erroring." The log said::

    RuntimeError: swap exhaustion: managed RSS 34494MB, swap 16.9GB

Measured on the same host at the same moment: 54% RAM used, **31.6 GB
available**, machine responsive. Each firing was a fail-closed CRITICAL; those
pinned ``deg_threat`` at 1.00 → existential threat 1.00 → the Ulysses covenant
refuses heavy compute at 0.6 — so a phantom emergency silently refused every
build she was asked for.

Two readings had to be discarded to get here, and both were discarded against
measurements, not argument:

* ``swap used`` is a high-water mark. Pages written to swap stay accounted long
  after the pressure is gone, so it latched at 17.9 GB against a 7.7 GB
  threshold and fired forever on an event an hour past.
* ``swap free`` was better and still wrong. macOS sizes the swap file
  dynamically and runs it near full by design — 1.10 GB free of 11.8 GB here,
  having grown to 18.4 GB and shrunk back on its own.

What decides it is RAM the host can still hand out, because that is what the
next allocation draws on. Swap headroom is corroboration; availability is the
gate.
"""
from __future__ import annotations

import pytest

from core.resilience.memory_watchdog import (
    MemorySample,
    _swap_is_exhausted,
    _Thresholds,
)

THRESHOLDS = _Thresholds.from_environment(64.0)


def _sample(*, used_gb: float, free_gb: float, available_gb: float) -> MemorySample:
    """A host with the model resident, varying only the memory picture."""
    return MemorySample(
        core_rss_mb=1200.0,
        child_rss_mb=33000.0,
        swap_used_gb=used_gb,
        system_percent=54.0,
        total_ram_gb=64.0,
        sampled_at=0.0,
        swap_free_gb=free_gb,
        available_gb=available_gb,
    )


def test_the_live_false_alarm_is_gone() -> None:
    """The exact reading that was firing every check, with 31.6 GB free."""
    assert not _swap_is_exhausted(
        _sample(used_gb=10.7, free_gb=1.10, available_gb=31.6), THRESHOLDS
    )


def test_a_latched_high_water_mark_decides_nothing() -> None:
    assert not _swap_is_exhausted(
        _sample(used_gb=17.9, free_gb=0.40, available_gb=24.8), THRESHOLDS
    )


def test_a_real_emergency_still_fires() -> None:
    """Swap full AND the host out of RAM — this is where the loop stalls."""
    assert _swap_is_exhausted(
        _sample(used_gb=10.7, free_gb=0.30, available_gb=1.2), THRESHOLDS
    )


def test_a_healthy_host_is_quiet() -> None:
    assert not _swap_is_exhausted(
        _sample(used_gb=0.5, free_gb=9.0, available_gb=40.0), THRESHOLDS
    )


@pytest.mark.parametrize("available_gb", [0.5, 1.0, 3.0, 5.0])
def test_scarce_ram_with_no_swap_headroom_is_an_emergency(available_gb: float) -> None:
    assert _swap_is_exhausted(
        _sample(used_gb=10.0, free_gb=0.2, available_gb=available_gb), THRESHOLDS
    )


@pytest.mark.parametrize("available_gb", [8.0, 16.0, 31.6, 48.0])
def test_ample_ram_never_is(available_gb: float) -> None:
    """However bad the swap file looks, headroom means no emergency."""
    assert not _swap_is_exhausted(
        _sample(used_gb=17.9, free_gb=0.1, available_gb=available_gb), THRESHOLDS
    )


def test_an_unreadable_availability_falls_back_rather_than_going_blind() -> None:
    """A stale signal beats a watchdog that can never fire."""
    assert _swap_is_exhausted(
        _sample(used_gb=17.9, free_gb=0.2, available_gb=0.0), THRESHOLDS
    )
    assert not _swap_is_exhausted(
        _sample(used_gb=1.0, free_gb=9.0, available_gb=0.0), THRESHOLDS
    )
