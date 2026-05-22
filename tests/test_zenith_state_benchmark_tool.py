from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class _BenchmarkState:
    def __init__(self, *, version: int = 4, origin: str = ""):
        self.version = version
        self.transition_origin = origin
        self.cognition = SimpleNamespace(working_memory=[])
        self.world = SimpleNamespace(known_entities={})

    async def derive_async(self, label: str, *, origin: str = ""):
        await asyncio.sleep(0)
        return _BenchmarkState(version=self.version + 1, origin=origin or label)


class _BrokenBenchmarkState(_BenchmarkState):
    async def derive_async(self, label: str, *, origin: str = ""):
        await asyncio.sleep(0)
        message = f"derive failed for {label}:{origin}"
        raise RuntimeError(message)


def _state_factory():
    return _BenchmarkState()


@pytest.mark.asyncio
async def test_zenith_state_benchmark_reports_derivation_success():
    from tools.benchmarks.zenith_v2_state_benchmark import measure_state_derivation_overhead

    step = await measure_state_derivation_overhead(state_factory=_state_factory)

    assert step.ok is True
    assert step.name == "state_derivation_overhead"
    assert step.details["state_version"] == 4
    assert step.details["derived_version"] == 5


@pytest.mark.asyncio
async def test_zenith_state_benchmark_reports_event_loop_health():
    from tools.benchmarks.zenith_v2_state_benchmark import measure_event_loop_lag

    step = await measure_event_loop_lag(
        state_factory=_state_factory,
        iterations=2,
        lag_threshold_ms=500.0,
    )

    assert step.ok is True
    assert step.details["iterations"] == 2
    assert step.details["lag_samples"] == []


@pytest.mark.asyncio
async def test_zenith_state_benchmark_records_structured_failure(monkeypatch):
    import tools.benchmarks.zenith_v2_state_benchmark as benchmark

    records = []

    monkeypatch.setattr(
        benchmark,
        "record_degradation",
        lambda *args, **kwargs: records.append((args, kwargs)),
    )

    report = await benchmark.run_zenith_v2_state_benchmark(
        state_factory=lambda: _BrokenBenchmarkState()
    )

    assert report.ok is False
    assert report.steps[-1].name == "benchmark_failed_closed"
    assert records
    assert "failed closed" in records[0][1]["action"]
