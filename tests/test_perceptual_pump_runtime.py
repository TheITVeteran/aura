from __future__ import annotations

from core.container import ServiceContainer
from core.perception.perceptual_pump import (
    AudioState,
    PerceptualFrame,
    PerceptualPump,
    ScreenState,
    SystemState,
    UserState,
    _collect_screen_state,
    frame_to_runtime_body,
)
from core.world_state import WorldState


class _CompletedProcess:
    def __init__(self, stdout: str):
        self.returncode = 0
        self.stdout = stdout


def test_screen_state_probe_uses_read_only_subprocess_gateway(monkeypatch) -> None:
    calls = []

    class Gateway:
        def run(self, argv, **kwargs):
            calls.append((argv, kwargs))
            if "frontmost" in argv[-1]:
                return _CompletedProcess("Notes\n")
            return _CompletedProcess("Aura Journal\n")

    monkeypatch.setattr(
        "core.runtime.subprocess_gateway.get_subprocess_gateway",
        lambda: Gateway(),
    )

    state = _collect_screen_state("")

    assert state.active_app == "Notes"
    assert state.window_title == "Aura Journal"
    assert state.screen_changed is True
    assert state.change_magnitude == 0.3
    assert [call[1]["source"] for call in calls] == [
        "perceptual_pump.screen.app",
        "perceptual_pump.screen.title",
    ]
    assert all(call[1]["read_only"] is True for call in calls)


def test_perceptual_frame_maps_into_runtime_body_observed_vector() -> None:
    frame = PerceptualFrame(
        frame_id=7,
        screen=ScreenState(
            active_app="Notes",
            window_title="Aura Journal",
            content_hash="abc123",
            screen_changed=True,
            change_magnitude=0.65,
        ),
        audio=AudioState(
            rms_energy=0.42,
            voice_activity=True,
            transcript_snippet="Hey Aura",
            transcript_changed=True,
        ),
        system=SystemState(
            cpu_percent=35.0,
            memory_percent=45.0,
            thermal_pressure=0.10,
            battery_percent=85.0,
            battery_charging=True,
        ),
        user=UserState(idle_seconds=2.0, presence=0.95),
    )

    body = frame_to_runtime_body(frame)
    observed = body.observed_vector()

    assert body.screen_novelty == 0.65
    assert body.audio_energy == 0.42
    assert body.voice_present is True
    assert body.foreground_app_familiar == 0.8
    assert observed["screen_novelty"] == 0.65
    assert observed["audio_energy"] == 0.42
    assert observed["voice_present"] == 1.0
    assert observed["app_familiarity"] == 0.8


def test_perceptual_pump_updates_world_state_with_grounded_frame() -> None:
    ServiceContainer.clear()
    world = WorldState()
    ServiceContainer.register_instance("world_state", world, required=False)
    pump = PerceptualPump()
    frame = PerceptualFrame(
        frame_id=1,
        screen=ScreenState(
            active_app="Google Chrome",
            window_title="Climate news",
            content_hash="hash-1",
            screen_changed=True,
            change_magnitude=1.0,
        ),
        audio=AudioState(
            rms_energy=0.31,
            voice_activity=True,
            transcript_snippet="Open a few articles",
            transcript_changed=True,
        ),
        system=SystemState(
            cpu_percent=22.5,
            memory_percent=51.0,
            thermal_pressure=0.08,
            battery_percent=91.0,
            battery_charging=True,
        ),
        user=UserState(idle_seconds=1.0, presence=1.0),
    )

    try:
        pump._update_world_state(frame)

        assert world.active_foreground_app == "Google Chrome"
        assert world.active_window_title == "Climate news"
        assert world.screen_content_hash == "hash-1"
        assert world.ambient_audio_level == 0.31
        assert world.voice_activity_detected is True
        assert world.last_voice_transcript == "Open a few articles"
        assert world.cpu_percent == 22.5
        assert world.memory_percent == 51.0
        assert world.thermal_pressure == 0.08
        descriptions = [event["description"] for event in world.get_salient_events()]
        assert "App switched to Google Chrome" in descriptions
        assert "Voice detected: Open a few articles" in descriptions
    finally:
        ServiceContainer.clear()
