from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.runtime.model_lane_control import ModelLaneController
from core.runtime.receipts import ReceiptStore
from core.runtime.shutdown_coordinator import clear_shutdown_request
from core.senses import voice_engine


def _controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelLaneController:
    monkeypatch.setenv("AURA_LANE_BUDGET_GB", "46")
    return ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_discovery=None,
    )


@pytest.fixture(autouse=True)
def _shutdown_state() -> None:
    clear_shutdown_request()
    yield
    clear_shutdown_request()


def test_whisper_model_holds_and_releases_process_identified_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, monkeypatch)
    constructor_calls: list[tuple[str, dict[str, object]]] = []
    loading_preemptibility: list[bool] = []

    class _Whisper:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            constructor_calls.append((model_name, dict(kwargs)))
            loading_preemptibility.append(
                bool(controller.snapshot()["owners"][0]["preemptible"])
            )

    monkeypatch.setattr(voice_engine, "_get_whisper_model_class", lambda: _Whisper)
    engine = voice_engine.SovereignVoiceEngine(
        whisper_model="base",
        data_dir=str(tmp_path / "voice"),
        model_lane_controller=controller,
    )

    assert engine._init_stt() is True
    owner = controller.snapshot()["owners"][0]
    assert owner["model_path"] == "faster-whisper/base"
    assert owner["declared_gb"] == pytest.approx(0.5)
    assert owner["metadata"]["model_role"] == "stt"
    assert constructor_calls[0][0] == "base"
    assert loading_preemptibility == [False]
    assert owner["preemptible"] is True

    engine.on_stop()

    assert engine.stt_model is None
    assert controller.snapshot()["owners"] == []


def test_piper_model_holds_and_releases_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, monkeypatch)
    voice_root = tmp_path / "voice"
    model_dir = voice_root / "piper_voices"
    model_dir.mkdir(parents=True)
    (model_dir / "en_US-amy-medium.onnx").write_bytes(b"model")
    (model_dir / "en_US-amy-medium.onnx.json").write_text("{}", encoding="utf-8")

    class _Piper:
        @staticmethod
        def load(model_path: str, *, config_path: str):
            return {"model_path": model_path, "config_path": config_path}

    monkeypatch.setattr(voice_engine, "PiperVoice", _Piper)
    engine = voice_engine.SovereignVoiceEngine(
        data_dir=str(voice_root),
        model_lane_controller=controller,
    )
    engine.use_xtts = False
    engine.use_piper = True

    engine._init_tts()

    assert engine._tts_initialized is True
    owner = controller.snapshot()["owners"][0]
    assert owner["declared_gb"] == pytest.approx(0.25)
    assert owner["metadata"]["model_role"] == "piper"

    engine.on_stop()

    assert engine._piper_voice is None
    assert controller.snapshot()["owners"] == []


def test_stt_discards_loaded_model_when_lane_activation_is_lost(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    released: list[str] = []

    class _Lease:
        @staticmethod
        def set_preemptible(preemptible: bool) -> bool:
            assert preemptible is True
            return False

        @staticmethod
        def release(*, reason: str) -> bool:
            released.append(reason)
            return True

    class _Whisper:
        def __init__(self, _model_name: str, **_kwargs: object) -> None:
            self.initialized = True

    monkeypatch.setattr(voice_engine, "_get_whisper_model_class", lambda: _Whisper)
    engine = voice_engine.SovereignVoiceEngine(
        whisper_model="base",
        data_dir=str(tmp_path / "voice"),
    )
    monkeypatch.setattr(engine, "_acquire_voice_model_lane", lambda **_kwargs: _Lease())

    assert engine._init_stt() is False
    assert engine.stt_model is None
    assert engine._stt_lane_lease is None
    assert engine._stt_initialized is False
    assert released == ["stt_model_load_failed"]


@pytest.mark.asyncio
async def test_stt_preemption_refuses_exact_active_model_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, monkeypatch)

    class _Whisper:
        def __init__(self, _model_name: str, **_kwargs: object) -> None:
            self.initialized = True

    monkeypatch.setattr(voice_engine, "_get_whisper_model_class", lambda: _Whisper)
    engine = voice_engine.SovereignVoiceEngine(
        data_dir=str(tmp_path / "voice"),
        model_lane_controller=controller,
    )
    assert engine._init_stt() is True
    entered = threading.Event()
    release = threading.Event()

    def _hold_model() -> None:
        with engine.stt_model_session() as model:
            assert model is not None
            entered.set()
            assert release.wait(2.0)

    holder = asyncio.create_task(asyncio.to_thread(_hold_model))
    assert await asyncio.to_thread(entered.wait, 2.0)
    assert engine._stt_active_users == 1
    assert await engine._evict_stt_lane(object(), "candidate") is False

    release.set()
    await holder
    assert engine._stt_active_users == 0
    assert await engine._evict_stt_lane(object(), "candidate") is True
    assert controller.snapshot()["owners"] == []
    engine.on_stop()


def test_voice_preemption_tracks_model_use_not_passive_capture(tmp_path: Path) -> None:
    engine = voice_engine.SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
    engine._mic_listening = True
    engine.is_speaking = True

    assert asyncio.run(engine._evict_stt_lane(object(), "candidate")) is True
    assert asyncio.run(engine._evict_tts_lane(object(), "candidate")) is False

    engine._mic_listening = False
    engine.is_speaking = False
    engine.on_stop()


def test_websocket_stt_reuses_registered_canonical_voice_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.senses import voice_socket_logic

    class _Model:
        @staticmethod
        def transcribe(_audio, **_kwargs):
            return iter((SimpleNamespace(text="hello"),)), {"language": "en"}

    model = _Model()

    class _CanonicalVoice:
        whisper_model_name = "base"
        stt_model = model

        @staticmethod
        def ensure_stt() -> bool:
            return True

        @staticmethod
        @contextlib.contextmanager
        def stt_model_session():
            yield model

    monkeypatch.setattr(
        voice_socket_logic,
        "get_runtime_service",
        lambda name, default=None: _CanonicalVoice() if name == "voice_engine" else default,
    )

    proxy = voice_socket_logic.get_whisper_model("tiny")
    assert proxy is not model
    segments, info = proxy.transcribe([0.0], beam_size=1)
    assert [segment.text for segment in segments] == ["hello"]
    assert info == {"language": "en"}
