from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path

import pytest

from core.runtime import runtime_settings
from core.runtime.settings_control_plane import RuntimeSettingsStore


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    settings_path = tmp_path / "runtime.json"
    monkeypatch.setenv("AURA_SETTINGS_PATH", str(settings_path))
    runtime_settings.clear_runtime_settings_cache()
    yield settings_path
    runtime_settings.clear_runtime_settings_cache()


def test_voice_engine_initializes_from_verified_settings_not_legacy_launch_env(
    _isolated_settings,
    monkeypatch,
    tmp_path,
) -> None:
    store = RuntimeSettingsStore(_isolated_settings)
    store.patch(
        {
            "voice.input_enabled": False,
            "voice.output_enabled": False,
        },
        expected_revision=0,
        request_id="voice-init-settings",
    )
    runtime_settings.clear_runtime_settings_cache()
    monkeypatch.setenv("AURA_AUTO_LISTEN", "1")

    from core.senses.voice_engine import SovereignVoiceEngine

    engine = SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))

    assert engine.microphone_enabled is False
    assert engine.auto_listen_enabled is False
    assert engine.speaking_enabled is False
    assert engine.should_auto_listen() is False


def test_disabling_voice_input_closes_the_resident_stream(
    tmp_path,
) -> None:
    from core.senses.voice_engine import SovereignVoiceEngine

    calls: list[str] = []

    class Stream:
        def stop(self) -> None:
            calls.append("stop")

        def close(self) -> None:
            calls.append("close")

    engine = SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
    engine.microphone_enabled = True
    engine.auto_listen_enabled = True
    engine._mic_listening = True
    engine._mic_stream = Stream()
    engine._signal_mycelium = lambda *_args, **_kwargs: None

    receipt = engine.apply_runtime_setting("voice.input_enabled", True, False)

    assert receipt == {
        "owner": "voice_input",
        "status": "applied",
        "detail": "microphone input disabled and capture stopped",
    }
    assert calls == ["stop", "close"]
    assert engine._mic_stream is None
    assert engine._mic_listening is False


@pytest.mark.asyncio
async def test_enabling_auto_listen_starts_and_verifies_capture_on_owner_loop(
    tmp_path,
) -> None:
    from core.senses.voice_engine import SovereignVoiceEngine

    engine = SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
    engine.microphone_enabled = True
    engine.auto_listen_enabled = False
    engine._mic_listening = False

    async def _start() -> bool:
        assert asyncio.get_running_loop() is engine.loop
        engine._mic_listening = True
        return True

    engine.start_listening = _start
    receipt = await asyncio.to_thread(
        engine.apply_runtime_setting,
        "voice.auto_listen",
        False,
        True,
    )

    assert receipt == {
        "owner": "voice_input",
        "status": "applied",
        "detail": "server microphone capture started and verified",
    }
    assert engine.auto_listen_enabled is True
    assert engine._mic_listening is True


def test_disabling_voice_output_interrupts_active_playback(tmp_path) -> None:
    from core.senses.voice_engine import SovereignVoiceEngine

    class Player:
        terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

    engine = SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
    player = Player()
    engine._current_afplay = player
    engine.interrupt_flag = threading.Event()
    engine.loop = None

    receipt = engine.apply_runtime_setting("voice.output_enabled", True, False)

    assert receipt["status"] == "applied"
    assert receipt["owner"] == "voice_output"
    assert engine.speaking_enabled is False
    assert engine.interrupt_flag.is_set()
    assert player.terminated is True


def test_settings_route_registers_truthful_voice_owner(
    _isolated_settings,
    monkeypatch,
) -> None:
    from core.senses import voice_engine as voice_module
    from interface.routes import settings

    calls: list[tuple[str, object, object]] = []

    class Voice:
        def apply_runtime_setting(self, key, previous, value):
            calls.append((key, previous, value))
            return {
                "owner": "voice_input",
                "status": "applied",
                "detail": "resident capture stopped and verified",
            }

    monkeypatch.setattr(voice_module, "get_voice_engine", lambda: Voice())
    monkeypatch.setattr(settings, "_STORE", None)
    monkeypatch.setattr(settings, "_SETTINGS_PATH", _isolated_settings)

    store = settings.get_settings()
    result = store.patch(
        {"voice.input_enabled": False},
        expected_revision=0,
        request_id="voice-route-owner",
    )

    assert calls == [("voice.input_enabled", True, False)]
    assert result.application["voice.input_enabled"] == {
        "owner": "voice_input",
        "status": "applied",
        "detail": "resident capture stopped and verified",
    }


def test_production_voice_paths_do_not_construct_competing_engines() -> None:
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "aura_main.py", *sorted((root / "core").rglob("*.py"))]
    offenders: list[str] = []
    for path in candidates:
        if path.name == "voice_engine.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            if name == "SovereignVoiceEngine":
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    assert offenders == []
