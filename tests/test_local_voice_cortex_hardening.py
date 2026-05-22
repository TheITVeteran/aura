import asyncio
from types import SimpleNamespace

import pytest

from core import local_voice_cortex as voice_module


class ContainerScenario:
    @staticmethod
    def get(_name, default=None):
        return default


class StreamScenario:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.closed = False

    def start_stream(self):
        self.started = True

    def is_active(self):
        return not self.stopped

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class AudioInterfaceScenario:
    def __init__(self, stream):
        self.stream = stream
        self.open_kwargs = None

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        return self.stream


class VadScenario:
    def __init__(self):
        self.calls = 0

    def is_speech(self, data, rate):
        self.calls += 1
        assert rate == 16000
        assert len(data) == 640
        return self.calls == 1


@pytest.fixture(autouse=True)
def local_voice_environment(monkeypatch):
    monkeypatch.setattr(voice_module, "_PYAUDIO_IMPORT_ATTEMPTED", False)
    monkeypatch.setattr(voice_module, "_WEBRTCVAD_IMPORT_ATTEMPTED", False)
    monkeypatch.setattr(voice_module, "_MLX_WHISPER_IMPORT_ATTEMPTED", False)
    monkeypatch.setattr(voice_module, "ServiceContainer", ContainerScenario)
    monkeypatch.setattr(voice_module, "get_vram_manager", lambda: None)
    monkeypatch.setattr(
        voice_module,
        "pyaudio",
        SimpleNamespace(paInt16=8, paContinue=0),
    )


def test_vad_frame_splitter_preserves_partial_audio():
    cortex = voice_module.LocalVoiceCortex()

    frame_bytes = cortex.VAD_FRAME_BYTES
    chunks = list(cortex._iter_vad_frames((b"a" * (frame_bytes * 2)) + b"tail"))

    assert len(chunks) == 2
    assert all(len(chunk) == frame_bytes for chunk in chunks)
    assert bytes(cortex._pending_audio) == b"tail"

    remaining = list(cortex._iter_vad_frames(b"b" * (frame_bytes - 4)))

    assert len(remaining) == 1
    assert len(remaining[0]) == frame_bytes
    assert cortex._pending_audio == bytearray()


@pytest.mark.asyncio
async def test_listen_loop_processes_vad_segment_without_cpu_spin():
    cortex = voice_module.LocalVoiceCortex()
    stream = StreamScenario()
    cortex.audio_interface = AudioInterfaceScenario(stream)
    cortex.audio_queue = asyncio.Queue(maxsize=100)
    cortex._shutdown_event = asyncio.Event()
    cortex.vad = VadScenario()
    cortex.is_listening = True
    processed_segments = []

    async def process_segment(frames):
        processed_segments.append(frames)
        cortex.is_listening = False
        cortex._shutdown_event.set()

    cortex._process_audio_segment = process_segment
    for _index in range(52):
        await cortex.audio_queue.put(b"\0" * cortex.VAD_FRAME_BYTES)

    await asyncio.wait_for(cortex.listen_loop(), timeout=1.0)

    assert stream.started is True
    assert stream.stopped is True
    assert stream.closed is True
    assert cortex.vad.calls == 52
    assert len(processed_segments) == 1
    assert len(processed_segments[0]) == 52


@pytest.mark.asyncio
async def test_start_keeps_listener_offline_when_voice_dependencies_are_missing(monkeypatch):
    records = []
    monkeypatch.setattr(voice_module, "pyaudio", None)
    monkeypatch.setattr(voice_module, "webrtcvad", object())
    monkeypatch.setattr(voice_module, "mlx_whisper", object())
    monkeypatch.setattr(voice_module, "np", object())
    monkeypatch.setattr(
        voice_module,
        "record_degradation",
        lambda subsystem, error, **kwargs: records.append((subsystem, error, kwargs)),
    )
    cortex = voice_module.LocalVoiceCortex()

    started = await cortex.start()

    assert started is False
    assert cortex.is_listening is False
    assert records
    assert records[0][0] == "local_voice_cortex"
    assert "kept microphone listener offline" in records[0][2]["action"]


@pytest.mark.asyncio
async def test_speak_reports_failure_when_no_voice_output_exists(monkeypatch):
    records = []
    monkeypatch.setattr(voice_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        voice_module,
        "record_degradation",
        lambda subsystem, error, **kwargs: records.append((subsystem, error, kwargs)),
    )
    cortex = voice_module.LocalVoiceCortex()

    spoken = await cortex.speak("hello")

    assert spoken is False
    assert records
    assert records[-1][0] == "local_voice_cortex"
    assert "all local voice cortex speech fallbacks failed" in records[-1][2]["action"]
