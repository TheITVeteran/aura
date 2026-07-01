import importlib
from types import SimpleNamespace


def test_optional_startup_validation_uses_spec_discovery_for_native_packages(monkeypatch):
    import core.startup.validator as validator

    imported: list[str] = []
    original_import_module = importlib.import_module

    def guarded_import_module(name: str, *args, **kwargs):
        imported.append(name)
        if name == "mlx_whisper":
            raise AssertionError("optional mlx_whisper check must not execute native import")
        return original_import_module(name, *args, **kwargs)

    def fake_find_spec(name: str):
        if name in {"mlx_whisper", "yaml"}:
            return object()
        return None

    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    monkeypatch.setattr(validator.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr("core.brain.llm.model_registry.get_local_backend", lambda: "mlx")

    results = validator.check_optional_packages()
    by_name = {result.name: result for result in results}

    assert by_name["Optional: mlx_whisper"].passed is True
    assert by_name["Optional: yaml"].passed is True
    assert "mlx_whisper" not in imported


def test_audio_device_check_uses_sounddevice_backend(monkeypatch):
    import sys

    from core.senses.sensory_registry import (
        SensoryCapabilityFlags,
        get_capabilities,
        set_capabilities,
    )
    from core.startup.validator import check_audio_device

    previous = get_capabilities()
    fake_sounddevice = SimpleNamespace(
        query_devices=lambda: [
            {"name": "Output", "max_input_channels": 0},
            {"name": "Mic", "max_input_channels": 1},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    set_capabilities(SensoryCapabilityFlags(hearing_enabled=True))
    try:
        result = check_audio_device()
    finally:
        set_capabilities(previous)

    assert result.passed is True
    assert "audio input device" in result.message


def test_audio_device_check_warns_when_sounddevice_has_no_inputs(monkeypatch):
    import sys

    from core.senses.sensory_registry import (
        SensoryCapabilityFlags,
        get_capabilities,
        set_capabilities,
    )
    from core.startup.validator import check_audio_device

    previous = get_capabilities()
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(query_devices=lambda: [{"name": "Output", "max_input_channels": 0}]),
    )
    set_capabilities(SensoryCapabilityFlags(hearing_enabled=True))
    try:
        result = check_audio_device()
    finally:
        set_capabilities(previous)

    assert result.passed is False
    assert "No audio devices found" in result.message
