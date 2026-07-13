"""Contract tests for the allocation-growth attribution surface.

Built for the confirmed idle leak (275MB/h on a model-free kernel,
2026-07-13): totals prove THAT memory grows; only per-site snapshot diffs
say WHERE. The surface must be honest when tracing is off, arm a baseline
on first call, and attribute deliberate growth to the allocating site.
"""

from __future__ import annotations

import tracemalloc

import pytest

from core.runtime.runtime_hygiene import RuntimeHygieneManager


@pytest.fixture()
def hygiene():
    manager = RuntimeHygieneManager()
    yield manager
    manager.rearm_allocation_baseline()


class TestAllocationGrowth:
    def test_honest_when_tracing_off(self, hygiene):
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        report = hygiene.allocation_growth()
        assert report["available"] is False
        assert report["reason"] == "tracemalloc_off"
        assert "AURA_RUNTIME_HYGIENE_TRACEMALLOC" in report["hint"]

    def test_first_call_arms_baseline_then_diffs_attribute_growth(self, hygiene):
        tracemalloc.start(5)
        try:
            first = hygiene.allocation_growth()
            assert first["available"] is True
            assert first["baseline_set"] is True

            # Deliberate retained growth from THIS site. Runtime-computed
            # UNIQUE strings — a constant expression would fold to one
            # shared object and retain only pointer storage.
            retained = [f"leak-attribution-proof-{i}" * 50 for i in range(20000)]

            report = hygiene.allocation_growth(top_n=10)
            assert report["baseline_set"] is False
            assert report["growth_mb_total"] > 1.0
            assert len(report["top_sites"]) <= 10
            top_tracebacks = " ".join(
                line for site in report["top_sites"] for line in site["traceback"]
            )
            assert "test_allocation_growth_surface" in top_tracebacks, (
                "the deliberately-leaking site must appear in the top diffs"
            )
            assert retained  # keep the allocation alive through the snapshot
        finally:
            tracemalloc.stop()

    def test_rearm_resets_baseline(self, hygiene):
        tracemalloc.start(1)
        try:
            assert hygiene.allocation_growth()["baseline_set"] is True
            assert hygiene.allocation_growth()["baseline_set"] is False
            hygiene.rearm_allocation_baseline()
            assert hygiene.allocation_growth()["baseline_set"] is True
        finally:
            tracemalloc.stop()
