"""The disk alarm must watch the volume Aura writes to.

The ResourceMonitor sampled psutil.disk_usage("/"). On macOS "/" is a sealed
read-only volume sharing an APFS container with /System/Volumes/Data, and what
that reading reports depends on which of the two it resolves to: raw psutil
returns 2.2% for "/" on this host while the data volume is at 72%.

To be exact about what this does and does not fix: the "resource strain: disk
at 99%" alarms of 2026-08-13 were CORRECT — the volume really was at 19GB free
of 1.8TB. This is not a false-reading fix. It removes the dependence on which
mount "/" resolves to, so the alarm cannot go quiet while the volume holding
her state fills up.
"""

from __future__ import annotations

from core.runtime import resource_psutil as psutil
from core.security.enforcement import _data_volume_percent


def test_the_reading_comes_from_the_state_volume() -> None:
    from core.runtime.state_ownership import state_root

    expected = psutil.disk_usage(str(state_root())).percent

    assert abs(_data_volume_percent(psutil) - expected) < 0.5


def test_the_reading_is_a_real_percentage() -> None:
    value = _data_volume_percent(psutil)

    assert 0.0 <= value <= 100.0


def test_an_unreadable_state_root_falls_back_rather_than_raising() -> None:
    """A monitor that raises stops monitoring; one that guesses zero lies."""

    class Broken:
        @staticmethod
        def disk_usage(path):
            raise OSError("unreadable")

    assert _data_volume_percent(Broken()) == 0.0
