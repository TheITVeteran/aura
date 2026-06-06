"""tests/test_somatic_throttle.py — Unit tests for SomaticComputeSentinel.

Verifies parameter cuts occur under real or coupled hardware stress.
"""
from pathlib import Path
from types import SimpleNamespace

import psutil

import core.brain.llm.somatic_throttle as throttle_module
from core.brain.llm.somatic_throttle import SomaticComputeSentinel


def _governor(throttle: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(get_throttle_factor=lambda: throttle)


def _install_probe_readings(monkeypatch, *, arousal: float, cpu_percent: float, memory_percent: float):
    affect = SimpleNamespace(current=SimpleNamespace(arousal=arousal))
    monkeypatch.setattr(throttle_module, "resolve_affect_engine", lambda: affect)
    monkeypatch.setattr(
        "research.protocols.resource_quotas.get_compute_governor",
        lambda: _governor(),
    )
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: cpu_percent)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=memory_percent))
    return affect


def test_somatic_throttle_normal(monkeypatch):
    # Normal/unstressed parameters remain unchanged.
    _install_probe_readings(monkeypatch, arousal=0.2, cpu_percent=15.0, memory_percent=40.0)
    sentinel = SomaticComputeSentinel()
    opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
    adjusted = sentinel.adjust_generation_options(opts.copy())
        
    assert adjusted["max_tokens"] == 512
    assert adjusted["temperature"] == 0.7
    assert adjusted["recurrent_depth"] == 0.8


def test_somatic_throttle_resource_stressed(monkeypatch):
    # Stressed hardware caps max_tokens and adjusts temp/lane depth.
    _install_probe_readings(monkeypatch, arousal=0.2, cpu_percent=50.0, memory_percent=89.0)
    sentinel = SomaticComputeSentinel()
    opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
    adjusted = sentinel.adjust_generation_options(opts.copy())
        
    assert adjusted["max_tokens"] == 256
    assert adjusted["temperature"] == 0.3
    assert adjusted["recurrent_depth"] == 0.4


def test_somatic_throttle_critical(monkeypatch):
    # Critical parameters restrict max_tokens to 128.
    _install_probe_readings(monkeypatch, arousal=0.95, cpu_percent=95.0, memory_percent=95.0)
    sentinel = SomaticComputeSentinel()
    opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_lane_depth": 0.8}
    adjusted = sentinel.adjust_generation_options(opts.copy())

    assert adjusted["max_tokens"] == 128
    assert adjusted["temperature"] == 0.15
    assert adjusted["recurrent_lane_depth"] == 0.2


def test_somatic_throttle_high_arousal_without_resource_pressure_is_not_critical(monkeypatch):
    _install_probe_readings(monkeypatch, arousal=0.97, cpu_percent=12.0, memory_percent=60.0)
    sentinel = SomaticComputeSentinel()
    opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
    adjusted = sentinel.adjust_generation_options(opts.copy())

    assert adjusted["max_tokens"] == 512
    assert adjusted["temperature"] == 0.7
    assert adjusted["recurrent_depth"] == 0.8


def test_somatic_throttle_does_not_swallow_generic_exceptions():
    source = (
        Path(__file__).resolve().parent.parent
        / "core"
        / "brain"
        / "llm"
        / "somatic_throttle.py"
    ).read_text(encoding="utf-8")

    assert "except Exception" not in source
    assert "except BaseException" not in source


def test_mlx_generation_throttle_hook_uses_typed_boundary():
    source = (
        Path(__file__).resolve().parent.parent
        / "core"
        / "brain"
        / "llm"
        / "mlx_client.py"
    ).read_text(encoding="utf-8")
    throttle_block = source.split("SomaticComputeSentinel", 1)[1].split("foreground_owner_cm", 1)[0]

    assert "except Exception" not in throttle_block
    assert "continued generation without somatic parameter throttle" in throttle_block


def test_somatic_throttle_records_expected_probe_failures(monkeypatch):
    records = []

    class FailingAffectResolver:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            raise RuntimeError("affect offline")

    monkeypatch.setattr(
        "core.brain.llm.somatic_throttle.record_degradation",
        lambda subsystem, exc, **metadata: records.append((subsystem, exc, metadata)),
    )
    monkeypatch.setattr(
        "core.brain.llm.somatic_throttle.resolve_affect_engine",
        FailingAffectResolver(),
    )
    monkeypatch.setattr(
        "research.protocols.resource_quotas.get_compute_governor",
        lambda: _governor(),
    )
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 5.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=20.0))

    adjusted = SomaticComputeSentinel().adjust_generation_options({"max_tokens": 512})

    assert adjusted["max_tokens"] == 512
    assert records
    assert records[0][0] == "somatic_throttle"
    assert records[0][2]["action"] == "using neutral arousal for generation throttle"
