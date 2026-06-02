"""tests/test_somatic_throttle.py — Unit tests for SomaticComputeSentinel.

Verifies parameter cuts occur under real or coupled hardware stress.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.brain.llm.somatic_throttle import SomaticComputeSentinel


def _governor(throttle: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(get_throttle_factor=lambda: throttle)


def test_somatic_throttle_normal():
    # Test normal/unstressed parameters remain unchanged
    with (
        patch("core.brain.llm.somatic_throttle.resolve_affect_engine") as mock_resolve,
        patch("research.protocols.resource_quotas.get_compute_governor", return_value=_governor()),
        patch("psutil.cpu_percent", return_value=15.0),
        patch("psutil.virtual_memory") as mock_vm,
    ):
        mock_affect = MagicMock()
        mock_affect.current = SimpleNamespace(arousal=0.2)
        mock_resolve.return_value = mock_affect
        
        mock_vm_obj = MagicMock()
        mock_vm_obj.percent = 40.0
        mock_vm.return_value = mock_vm_obj
        
        sentinel = SomaticComputeSentinel()
        opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
        adjusted = sentinel.adjust_generation_options(opts.copy())
        
        assert adjusted["max_tokens"] == 512
        assert adjusted["temperature"] == 0.7
        assert adjusted["recurrent_depth"] == 0.8


def test_somatic_throttle_resource_stressed():
    # Test stressed hardware caps max_tokens and adjusts temp/lane depth.
    with (
        patch("core.brain.llm.somatic_throttle.resolve_affect_engine") as mock_resolve,
        patch("research.protocols.resource_quotas.get_compute_governor", return_value=_governor()),
        patch("psutil.cpu_percent", return_value=50.0),
        patch("psutil.virtual_memory") as mock_vm,
    ):
        mock_affect = MagicMock()
        mock_affect.current = SimpleNamespace(arousal=0.2)
        mock_resolve.return_value = mock_affect
        
        mock_vm_obj = MagicMock()
        mock_vm_obj.percent = 89.0
        mock_vm.return_value = mock_vm_obj
        
        sentinel = SomaticComputeSentinel()
        opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
        adjusted = sentinel.adjust_generation_options(opts.copy())
        
        assert adjusted["max_tokens"] == 256
        assert adjusted["temperature"] == 0.3
        assert adjusted["recurrent_depth"] == 0.4


def test_somatic_throttle_critical():
    # Test critical parameters restrict max_tokens to 128
    with (
        patch("core.brain.llm.somatic_throttle.resolve_affect_engine") as mock_resolve,
        patch("research.protocols.resource_quotas.get_compute_governor", return_value=_governor()),
        patch("psutil.cpu_percent", return_value=95.0),  # High CPU
        patch("psutil.virtual_memory") as mock_vm,
    ):
        mock_affect = MagicMock()
        mock_affect.current = SimpleNamespace(arousal=0.95)  # Critical arousal!
        mock_resolve.return_value = mock_affect
        
        mock_vm_obj = MagicMock()
        mock_vm_obj.percent = 95.0  # Critical RAM
        mock_vm.return_value = mock_vm_obj
        
        sentinel = SomaticComputeSentinel()
        opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_lane_depth": 0.8}
        adjusted = sentinel.adjust_generation_options(opts.copy())
        
        assert adjusted["max_tokens"] == 128
        assert adjusted["temperature"] == 0.15
        assert adjusted["recurrent_lane_depth"] == 0.2


def test_somatic_throttle_high_arousal_without_resource_pressure_is_not_critical():
    with (
        patch("core.brain.llm.somatic_throttle.resolve_affect_engine") as mock_resolve,
        patch("research.protocols.resource_quotas.get_compute_governor", return_value=_governor()),
        patch("psutil.cpu_percent", return_value=12.0),
        patch("psutil.virtual_memory") as mock_vm,
    ):
        mock_affect = MagicMock()
        mock_affect.current = SimpleNamespace(arousal=0.97)
        mock_resolve.return_value = mock_affect

        mock_vm_obj = MagicMock()
        mock_vm_obj.percent = 60.0
        mock_vm.return_value = mock_vm_obj

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

    monkeypatch.setattr(
        "core.brain.llm.somatic_throttle.record_degradation",
        lambda subsystem, exc, **metadata: records.append((subsystem, exc, metadata)),
    )
    monkeypatch.setattr(
        "core.brain.llm.somatic_throttle.resolve_affect_engine",
        MagicMock(side_effect=RuntimeError("affect offline")),
    )

    with (
        patch("research.protocols.resource_quotas.get_compute_governor", return_value=_governor()),
        patch("psutil.cpu_percent", return_value=5.0),
        patch("psutil.virtual_memory") as mock_vm,
    ):
        mock_vm_obj = MagicMock()
        mock_vm_obj.percent = 20.0
        mock_vm.return_value = mock_vm_obj

        adjusted = SomaticComputeSentinel().adjust_generation_options({"max_tokens": 512})

    assert adjusted["max_tokens"] == 512
    assert records
    assert records[0][0] == "somatic_throttle"
    assert records[0][2]["action"] == "using neutral arousal for generation throttle"
