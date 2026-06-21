"""Tests for the layering-clean runtime-settings accessor and the first dead
setting wired through it: voice.output_enabled (docs/SETTINGS_WIRING_AUDIT.md)."""
import asyncio
import json

import pytest

from core.runtime import runtime_settings as rs


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_SETTINGS_PATH", str(tmp_path / "runtime.json"))
    rs.clear_runtime_settings_cache()
    yield
    rs.clear_runtime_settings_cache()


def _write_settings(tmp_path, data):
    (tmp_path / "runtime.json").write_text(json.dumps(data), encoding="utf-8")
    rs.clear_runtime_settings_cache()


def test_missing_file_returns_default(tmp_path):
    assert rs.get_runtime_setting("voice.output_enabled", True) is True
    assert rs.get_runtime_setting("voice.output_enabled", False) is False
    assert rs.get_runtime_setting("nope.key") is None


def test_present_key_returned(tmp_path):
    _write_settings(tmp_path, {"voice.output_enabled": False})
    assert rs.get_runtime_setting("voice.output_enabled", True) is False


def test_missing_key_uses_default(tmp_path):
    _write_settings(tmp_path, {"other.key": 1})
    assert rs.get_runtime_setting("voice.output_enabled", True) is True


def test_corrupt_file_falls_back_to_default(tmp_path):
    (tmp_path / "runtime.json").write_text("{ not valid json", encoding="utf-8")
    rs.clear_runtime_settings_cache()
    assert rs.get_runtime_setting("voice.output_enabled", True) is True


def test_live_update_reflected(tmp_path):
    _write_settings(tmp_path, {"voice.output_enabled": True})
    assert rs.get_runtime_setting("voice.output_enabled", True) is True
    _write_settings(tmp_path, {"voice.output_enabled": False})
    assert rs.get_runtime_setting("voice.output_enabled", True) is False


def test_voice_output_predicate_default_on(tmp_path):
    from core.senses.voice_engine import _user_voice_output_enabled

    assert _user_voice_output_enabled() is True


def test_voice_output_disabled_suppresses_synthesis(tmp_path):
    _write_settings(tmp_path, {"voice.output_enabled": False})

    from core.senses.voice_engine import SovereignVoiceEngine, _user_voice_output_enabled

    assert _user_voice_output_enabled() is False

    # Uninitialized instance: synthesize_speech must short-circuit at the user
    # gate (before acquiring locks or touching the TTS engine), returning "".
    engine = object.__new__(SovereignVoiceEngine)
    engine.speaking_enabled = True
    result = asyncio.run(engine.synthesize_speech("hello there"))
    assert result == ""
