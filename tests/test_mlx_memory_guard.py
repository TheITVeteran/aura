"""Host-safety contracts for the MLX memory envelope (CP214).

An unguarded evaluation tool drove this host to 103 GB and forced a
shutdown. Training was already enveloped; evaluation was not. These tests
pin the properties that make that unrepeatable: limits are derived from
real host RAM, over-host limits are refused rather than granted, the
envelope restores what it changed, and the reclaim hook is available to
long loops.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.runtime.mlx_memory_guard import (  # noqa: E402
    DEFAULT_FRACTION,
    MLX_MEMORY_GUARD_SCHEMA,
    MemoryEnvelope,
    host_memory_bytes,
    mlx_memory_envelope,
)


def test_host_memory_is_read_not_assumed():
    host = host_memory_bytes()
    assert host >= 2 * 1024**3
    assert host < 4 * 1024**4  # sanity: not a nonsense reading


def test_default_limit_is_a_fraction_of_real_host_ram():
    with mlx_memory_envelope() as envelope:
        receipt = envelope.to_receipt()
        assert receipt["schema"] == MLX_MEMORY_GUARD_SCHEMA
        expected = DEFAULT_FRACTION * receipt["host_memory_gb"]
        assert receipt["memory_limit_gb"] == pytest.approx(expected, rel=0.02)
        # Headroom for the OS is the entire point.
        assert receipt["memory_limit_gb"] < receipt["host_memory_gb"]


def test_over_host_limit_is_refused_not_granted():
    """The failure mode being prevented: a limit above physical RAM means
    the machine swaps to death instead of the run failing."""
    host_gb = host_memory_bytes() / 1024**3
    with pytest.raises(ValueError, match="exceeds physical RAM"):
        with mlx_memory_envelope(memory_gb=host_gb * 2):
            pass


def test_unusable_and_out_of_range_settings_are_refused():
    with pytest.raises(ValueError, match="2 GiB floor"):
        with mlx_memory_envelope(memory_gb=0.5):
            pass
    with pytest.raises(ValueError, match="fraction"):
        with mlx_memory_envelope(fraction=0.99):
            pass
    with pytest.raises(ValueError, match="cache limit cannot exceed"):
        with mlx_memory_envelope(memory_gb=4.0, cache_gb=8.0):
            pass


def test_limits_are_restored_on_exit_and_on_error():
    before = mx.set_memory_limit(mx.set_memory_limit(8 * 1024**3))
    with mlx_memory_envelope(memory_gb=4.0):
        inside = mx.set_memory_limit(mx.set_memory_limit(4 * 1024**3))
        assert inside == 4 * 1024**3
    after = mx.set_memory_limit(mx.set_memory_limit(before))
    assert after == before

    with pytest.raises(RuntimeError, match="boom"):
        with mlx_memory_envelope(memory_gb=4.0):
            raise RuntimeError("boom")
    restored = mx.set_memory_limit(mx.set_memory_limit(before))
    assert restored == before, "limits must restore even when the body raises"


def test_reclaim_honours_cadence_and_force():
    envelope = MemoryEnvelope(
        memory_bytes=4 * 1024**3,
        cache_bytes=1024**3,
        wired_bytes=4 * 1024**3,
        reclaim_every=8,
    )
    assert envelope.reclaim(8) is True
    assert envelope.reclaim(3) is False
    assert envelope.reclaim(3, force=True) is True
    assert envelope.reclaim(None, force=True) is True


def test_receipt_is_complete_enough_to_audit_a_run():
    with mlx_memory_envelope(memory_gb=4.0, cache_gb=1.0) as envelope:
        receipt = envelope.to_receipt()
    assert set(receipt) == {
        "schema",
        "memory_limit_gb",
        "cache_limit_gb",
        "wired_limit_gb",
        "reclaim_every",
        "host_memory_gb",
    }
    assert receipt["memory_limit_gb"] == pytest.approx(4.0, rel=1e-6)
    assert receipt["cache_limit_gb"] == pytest.approx(1.0, rel=1e-6)


# ── Reading pressure correctly, not alarmingly ──────────────────────────


def test_host_pressure_reports_reclaimable_not_just_free():
    """'Pages free' excludes inactive/purgeable memory the kernel reclaims
    on demand, so a healthy host can read ~1GB free while 65% of RAM is
    actually available. Aborting a good run on that number is a false
    alarm; the signals that preceded this host's jetsam kill were SWAP and
    COMPRESSOR growth."""
    from core.runtime.mlx_memory_guard import host_pressure

    report = host_pressure()
    if not report.get("available"):
        pytest.skip("vm_stat unavailable on this platform")
    assert report["reclaimable_gb"] >= report["free_gb"]
    assert 0.0 <= report["available_fraction"] <= 1.0
    assert report["host_gb"] > 0
    assert isinstance(report["under_pressure"], bool)


def test_pressure_verdict_keys_on_swap_and_compression():
    """The verdict must not fire merely because free memory looks small."""
    from core.runtime.mlx_memory_guard import host_pressure

    report = host_pressure()
    if not report.get("available"):
        pytest.skip("vm_stat unavailable on this platform")
    if report["swap_used_gb"] < 2.0 and report["compressed_gb"] < 0.25 * report[
        "host_gb"
    ] and report["reclaimable_gb"] >= 0.08 * report["host_gb"]:
        assert report["under_pressure"] is False
