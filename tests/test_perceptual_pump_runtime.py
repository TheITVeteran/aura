from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from core.container import ServiceContainer
from core.perception.multimodal_sync import Modality
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
    import core.perception.perceptual_pump as pump_mod

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
    monkeypatch.setattr(pump_mod, "_collect_native_screen_state", lambda: None)

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


def test_screen_state_probe_uses_native_frontmost_metadata_before_subprocess(monkeypatch) -> None:
    import core.perception.perceptual_pump as pump_mod

    class Gateway:
        def __init__(self):
            self.calls = []

        def run(self, *_args, **_kwargs):
            self.calls.append((_args, _kwargs))
            raise AssertionError("native screen probe should avoid AppleScript subprocess")

    gateway = Gateway()
    monkeypatch.setattr(
        "core.runtime.subprocess_gateway.get_subprocess_gateway",
        lambda: gateway,
    )
    monkeypatch.setattr(
        pump_mod,
        "_collect_native_screen_state",
        lambda: ScreenState(active_app="Google Chrome", window_title="Climate News"),
    )

    state = _collect_screen_state("")

    assert state.active_app == "Google Chrome"
    assert state.window_title == "Climate News"
    assert state.screen_changed is True
    assert state.change_magnitude == 0.3


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
        assert world.last_voice_activity_at == frame.audio.timestamp
        assert world.last_voice_transcript == "Open a few articles"
        assert world.last_voice_transcript_at == frame.audio.timestamp
        assert world.last_audio_source_assessment["source"] == "perceptual_pump_audio"
        assert world.last_audio_source_assessment["transcript_changed"] is True
        assert world.cpu_percent == 22.5
        assert world.memory_percent == 51.0
        assert world.thermal_pressure == 0.08
        descriptions = [event["description"] for event in world.get_salient_events()]
        assert "App switched to Google Chrome" in descriptions
        assert "Voice detected: Open a few articles" in descriptions
    finally:
        ServiceContainer.clear()


def test_perceptual_pump_tick_fuses_modalities_and_publishes_reconciled_beliefs(
    monkeypatch,
) -> None:
    import asyncio

    world = WorldState()
    ServiceContainer.clear()
    ServiceContainer.register_instance("world_state", world, required=False)
    pump = PerceptualPump()
    pump._frame_count = 9

    monkeypatch.setattr(pump, "_cognitive_load_throttle_active", lambda: False)
    monkeypatch.setattr(
        "core.perception.perceptual_pump._collect_screen_state",
        lambda _prev: ScreenState(
            active_app="Google Chrome",
            window_title="Aura verification",
            content_hash="ocr-digest",
            screen_changed=True,
            change_magnitude=0.8,
            available=True,
            source="unit_screen",
            confidence=0.95,
            missing_reason=None,
        ),
    )
    monkeypatch.setattr(
        "core.perception.perceptual_pump._collect_audio_state",
        lambda: AudioState(
            rms_energy=0.3,
            voice_activity=True,
            transcript_snippet="private spoken request",
            transcript_full="private spoken request",
            transcript_changed=True,
            available=True,
            source="unit_audio",
            confidence=0.90,
            missing_reason=None,
        ),
    )
    monkeypatch.setattr(
        "core.perception.perceptual_pump._collect_system_state",
        lambda: SystemState(
            cpu_percent=21.0,
            memory_percent=42.0,
            thermal_pressure=0.05,
            battery_percent=90.0,
            battery_charging=True,
            available=True,
            source="unit_system",
            confidence=0.99,
            missing_reason=None,
        ),
    )
    monkeypatch.setattr(
        "core.perception.perceptual_pump._collect_user_state",
        lambda: UserState(
            idle_seconds=1.0,
            presence=1.0,
            available=True,
            source="unit_user",
            confidence=0.8,
            missing_reason=None,
        ),
    )

    try:
        asyncio.run(pump._tick())

        frame = pump.latest_frame
        assert frame is not None and frame.fusion is not None
        assert frame.fusion.missing == {}
        assert frame.fusion.has_usable(Modality.VISION) is True
        assert frame.fusion.has_usable(Modality.SPEECH) is True
        assert frame.fusion.confidence > 0.65
        assert world.get_belief("perception.fusion_confidence") == frame.fusion.confidence
        assert world.get_belief("perception.device.cpu_percent") == 21.0
        assert world.last_audio_source_assessment["visual_speech_evidence"] is False

        status_text = repr(pump.get_status())
        assert "private spoken request" not in status_text
        assert "audio_transcript_not_visual_speech" in status_text
    finally:
        ServiceContainer.clear()


def test_screen_probe_timeout_does_not_propagate_and_sets_backoff(monkeypatch) -> None:
    import subprocess

    import core.perception.perceptual_pump as pump_mod

    monkeypatch.setattr(pump_mod, "_LAST_SCREEN_PROBE_TIMEOUT_AT", 0.0)
    degradation_calls = []
    monkeypatch.setattr(
        pump_mod,
        "record_degradation",
        lambda *args, **kwargs: degradation_calls.append((args, kwargs)),
    )

    class HangingGateway:
        def __init__(self):
            self.calls = 0
            self.kwargs = []

        def run(self, argv, **kwargs):
            self.calls += 1
            self.kwargs.append(kwargs)
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    gateway = HangingGateway()
    monkeypatch.setattr(
        "core.runtime.subprocess_gateway.get_subprocess_gateway",
        lambda: gateway,
    )
    monkeypatch.setattr(pump_mod, "_collect_native_screen_state", lambda: None)

    # A hung osascript must yield an empty state, not an exception — a
    # TimeoutExpired here killed the entire pump task in live runtime.
    state = _collect_screen_state("")
    assert state.active_app == ""
    assert gateway.calls == 1
    assert gateway.kwargs[0]["timeout"] == pump_mod._SCREEN_PROBE_TIMEOUT_S
    assert degradation_calls
    assert degradation_calls[-1][1]["severity"] == "warning"
    assert degradation_calls[-1][1]["enforce_failure_policy"] is False
    assert "backed off" in degradation_calls[-1][1]["action"]

    # Backoff: the next collection skips the subprocess entirely.
    state = _collect_screen_state("")
    assert gateway.calls == 1
    assert pump_mod._LAST_SCREEN_PROBE_TIMEOUT_AT > 0.0


def test_pump_runtime_errors_cover_subprocess_failures() -> None:
    import subprocess

    from core.perception.perceptual_pump import _PUMP_RUNTIME_ERRORS

    assert issubclass(subprocess.TimeoutExpired, _PUMP_RUNTIME_ERRORS)


def test_throttle_stretches_subprocess_sensor_cadence(monkeypatch) -> None:
    import asyncio

    pump = PerceptualPump()
    screen_calls = []

    async def _run() -> None:
        monkeypatch.setattr(pump, "_cognitive_load_throttle_active", lambda: True)
        monkeypatch.setattr(
            "core.perception.perceptual_pump._collect_screen_state",
            lambda prev: screen_calls.append(prev) or ScreenState(),
        )
        # SCREEN_EVERY_N ticks would normally trigger a screen probe;
        # under throttle it must not.
        for _ in range(pump.SCREEN_EVERY_N):
            await pump._tick()
        assert screen_calls == []

        # At the throttled cadence the probe still happens — perception
        # slows down under load, it never stops.
        for _ in range(pump.SCREEN_EVERY_N * (pump.THROTTLED_MULTIPLIER - 1)):
            await pump._tick()
        assert len(screen_calls) == 1

    asyncio.run(_run())


def test_throttle_reads_foreground_inference_lane(monkeypatch) -> None:
    pump = PerceptualPump()

    class Gate:
        def get_conversation_status(self):
            return {"foreground_owned": True, "active_generations": 1}

    ServiceContainer.register_instance("inference_gate", Gate(), required=False)
    try:
        assert pump._cognitive_load_throttle_active() is True
    finally:
        ServiceContainer.clear()


def test_throttle_uses_lightweight_foreground_probe_without_full_lane_status(monkeypatch) -> None:
    pump = PerceptualPump()
    status_calls = []

    class Gate:
        def _foreground_user_turn_active(self):
            return False

        def _foreground_owner_active(self):
            return True

        def get_conversation_status(self):
            status_calls.append("called")
            return {"foreground_owned": False, "active_generations": 0}

    ServiceContainer.register_instance("inference_gate", Gate(), required=False)
    try:
        assert pump._cognitive_load_throttle_active() is True
        assert status_calls == []
    finally:
        ServiceContainer.clear()


def test_throttle_still_honors_memory_pressure_after_lightweight_probe(monkeypatch) -> None:
    pump = PerceptualPump()
    pump._system = SystemState(memory_percent=90.0)
    status_calls = []

    class Gate:
        def _foreground_user_turn_active(self):
            return False

        def _foreground_owner_active(self):
            return False

        def get_conversation_status(self):
            status_calls.append("called")
            return {"foreground_owned": False, "active_generations": 0}

    ServiceContainer.register_instance("inference_gate", Gate(), required=False)
    try:
        assert pump._cognitive_load_throttle_active() is True
        assert status_calls == []
    finally:
        ServiceContainer.clear()


def test_substrate_injections_are_serialized_off_event_loop_and_shutdown_cleanly(
    monkeypatch,
) -> None:
    pump = PerceptualPump()
    worker_started = threading.Event()
    release_worker = threading.Event()
    calls: list[tuple[str, int, str, int]] = []

    class Engine:
        def step(self, _body, event, recurrent_cycles):
            frame_id = int(event.label.rsplit("_", 1)[-1])
            calls.append(
                (
                    "start",
                    frame_id,
                    threading.current_thread().name,
                    threading.get_ident(),
                )
            )
            if frame_id == 1:
                worker_started.set()
                assert release_worker.wait(timeout=2.0)
            calls.append(
                (
                    "end",
                    frame_id,
                    threading.current_thread().name,
                    threading.get_ident(),
                )
            )
            assert recurrent_cycles == 2
            return SimpleNamespace(
                valence=0.1,
                arousal=0.2,
                curiosity=0.3,
            )

    class Substrate:
        def inject_perceptual_frame(self, _payload):
            return None

        def adapt_projections(self, _payload, *, lr):
            assert lr == 0.005

    async def idle_pump_loop() -> None:
        await asyncio.Event().wait()

    async def run() -> tuple[int, dict]:
        loop_thread = threading.get_ident()
        monkeypatch.setattr(pump, "_pump_loop", idle_pump_loop)
        await pump.start()
        executor = pump._substrate_executor
        assert executor is not None

        first = asyncio.create_task(
            pump._inject_into_substrate(PerceptualFrame(frame_id=1))
        )
        for _ in range(200):
            if worker_started.is_set():
                break
            await asyncio.sleep(0.001)
        assert worker_started.is_set()

        second = asyncio.create_task(
            pump._inject_into_substrate(PerceptualFrame(frame_id=2))
        )
        heartbeats = 0
        for _ in range(8):
            await asyncio.sleep(0.005)
            heartbeats += 1
        assert heartbeats == 8
        assert not first.done()
        assert not second.done()

        release_worker.set()
        await asyncio.gather(first, second)
        status = pump.get_status()
        await pump.stop()

        assert pump._substrate_executor is None
        assert getattr(executor, "_shutdown", False) is True
        return loop_thread, status

    ServiceContainer.clear()
    ServiceContainer.register_instance("phenomenal_engine", Engine(), required=False)
    ServiceContainer.register_instance("conscious_substrate", Substrate(), required=False)
    try:
        loop_thread, status = asyncio.run(run())
    finally:
        release_worker.set()
        ServiceContainer.clear()

    assert [(stage, frame_id) for stage, frame_id, _thread, _ident in calls] == [
        ("start", 1),
        ("end", 1),
        ("start", 2),
        ("end", 2),
    ]
    worker_names = {thread for _stage, _frame_id, thread, _ident in calls}
    worker_idents = {ident for _stage, _frame_id, _thread, ident in calls}
    assert len(worker_names) == 1
    assert len(worker_idents) == 1
    assert next(iter(worker_names)).startswith("AuraPerceptualSubstrate")
    assert loop_thread not in worker_idents
    worker_status = status["substrate_worker"]
    assert worker_status["owner"] == "dedicated_single_worker"
    assert worker_status["ordered"] is True
    assert worker_status["active"] is False
    assert worker_status["last_ms"] > 0.0
