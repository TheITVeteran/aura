"""One disk reading, one red line, consumed by everyone who has an opinion.

Ten subsystems each called ``psutil.disk_usage("/")`` and compared it against
thresholds they declared themselves — allostasis (amber 92 / red 98),
survival_driver (95 / 98), fictional_ai_synthesis (>90), core_monitor (98), and
six more. On macOS "/" is a sealed read-only volume sharing an APFS container
with /System/Volumes/Data, so what that call reports depends on which of the
two it resolves to: raw psutil returns 2.2% for "/" on this host while the
volume Aura actually writes to sits at 56%.

To be exact about what this does and does not fix: the "resource strain: disk
at 99%" alarms of 2026-08-13 were CORRECT — the volume really was at 19GB free
of 1.8TB. This is not a false-reading fix. What it removes is a vital that
several subsystems each sample their own way against their own numbers, which
cannot be reasoned about: they can disagree about whether the disk is full
while all claiming to measure "the disk".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.runtime.disk_budget import (
    DISK_AMBER_PERCENT,
    DISK_RED_PERCENT,
    DiskBudgetRefusal,
    ensure_headroom_for,
    free_space,
    state_volume_percent,
    state_volume_usage,
)
from core.security.enforcement import _data_volume_percent

CORE = Path(__file__).resolve().parents[2] / "core"
INTERFACE = Path(__file__).resolve().parents[2] / "interface"


def test_the_reading_comes_from_the_volume_aura_writes_to() -> None:
    from core.runtime.state_ownership import state_root

    expected = free_space(state_root()).percent

    assert abs(state_volume_percent() - expected) < 0.5


def test_the_reading_is_a_real_percentage() -> None:
    assert 0.0 <= state_volume_percent() <= 100.0


def test_usage_has_the_psutil_shape_so_call_sites_swap_cleanly() -> None:
    """Call sites read .total/.free/.percent; a partial shim breaks them."""
    usage = state_volume_usage()

    assert usage.total > 0
    assert 0 <= usage.free <= usage.total
    assert usage.used + usage.free == pytest.approx(usage.total, rel=1e-6)
    assert usage.percent == pytest.approx(
        100.0 * usage.used / usage.total, abs=0.5
    )


def test_the_monitor_delegates_rather_than_deriving_its_own() -> None:
    """A psutil that cannot answer must not change the canonical reading."""

    class Broken:
        @staticmethod
        def disk_usage(path: str) -> object:
            raise OSError("unreadable")

    assert _data_volume_percent(Broken()) == pytest.approx(
        state_volume_percent(), abs=0.5
    )


def test_no_subsystem_probes_the_wrong_mount_again() -> None:
    """The ratchet. This is the defect class, not a single site."""
    offenders: list[str] = []
    for root in (CORE, INTERFACE):
        for path in root.rglob("*.py"):
            if path.name == "disk_budget.py":
                continue  # the canonical reading's own fallback
            if path.name == "enforcement.py":
                continue  # documented last-resort fallback, delegates first
            text = path.read_text(encoding="utf-8", errors="ignore")
            if 'disk_usage("/")' in text or "disk_usage('/')" in text:
                offenders.append(str(path))
    assert not offenders, (
        "these probe a mount that is not where Aura writes; call "
        f"state_volume_percent()/state_volume_usage() instead: {offenders}"
    )


def test_the_red_line_has_one_owner() -> None:
    """Allostasis and the reactive driver must not restate the number."""
    from core.autonomic.allostasis import default_vital_specs

    disk = next(v for v in default_vital_specs() if v.key == "disk_percent")

    assert disk.red == DISK_RED_PERCENT
    assert disk.amber == DISK_AMBER_PERCENT


def test_survival_driver_shares_the_same_lines() -> None:
    from core.autonomic.survival_driver import SurvivalDriver

    driver = SurvivalDriver()

    assert driver.disk_critical_threshold == DISK_RED_PERCENT
    assert driver.disk_warning_threshold == DISK_AMBER_PERCENT


def test_a_write_that_would_cross_the_red_line_is_refused() -> None:
    """The budget half: nothing could ever say no to a 17GB fuse."""
    space = free_space("/")
    # Ask for everything free plus a margin — must be declined, not attempted.
    absurd = int((space.free_gb + 50.0) * (1024**3))

    with pytest.raises(DiskBudgetRefusal):
        ensure_headroom_for(absurd, purpose="test fuse", path="/")


def test_an_ordinary_write_is_not_refused() -> None:
    ensure_headroom_for(1024, purpose="test small write", path="/")
