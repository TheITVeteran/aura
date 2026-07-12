from __future__ import annotations

import numpy as np
import pytest

import core.perception.visual_speech_auto_avsr as auto_avsr
from core.perception.visual_speech import BackendPrediction
from core.perception.visual_speech_auto_avsr import AutoAVSRBackend, AutoAVSRConfig


class FakeLease:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def set_preemptible(self, preemptible: bool) -> bool:
        self.events.append(f"preemptible:{preemptible}")
        return True

    async def release(self, *, reason: str) -> bool:
        self.events.append(f"release:{reason}")
        return True


class FakeRuntime:
    def __init__(self, _config: AutoAVSRConfig, events: list[str]) -> None:
        self.events = events
        self.loaded = False
        self.fail_load = False

    def load(self) -> None:
        self.events.append("runtime:load")
        if self.fail_load:
            raise RuntimeError("load failed")
        self.loaded = True

    def infer(self, mouth_crops: np.ndarray) -> BackendPrediction:
        self.events.append(f"runtime:infer:{mouth_crops.shape[0]}")
        return BackendPrediction(
            transcript="unit visual prediction",
            confidence=None,
            calibrated=False,
            backend="auto_avsr",
            model_id="unit-auto-avsr",
        )

    def clear(self) -> None:
        self.events.append("runtime:clear")
        self.loaded = False


def _install_fake_lane(monkeypatch, events: list[str], runtime: FakeRuntime) -> FakeLease:
    import core.runtime.model_lane_control as lane

    lease = FakeLease(events)

    async def acquire(**kwargs):
        events.append(f"acquire:{kwargs['owner_id']}:{kwargs['preemptible']}")
        return lease

    async def run(operation, *, operation_name, timeout_s=None):
        events.append(f"owned:{operation_name}:{timeout_s}")
        return operation()

    monkeypatch.setattr(lane, "acquire_in_process_model_lane", acquire)
    monkeypatch.setattr(lane, "run_owned_model_thread_call", run)
    monkeypatch.setattr(auto_avsr, "_AutoAVSRRuntime", lambda _config: runtime)
    return lease


@pytest.mark.asyncio
async def test_backend_load_infer_and_unload_are_fenced_by_model_lane(monkeypatch) -> None:
    events: list[str] = []
    config = AutoAVSRConfig(beam_size=2)
    runtime = FakeRuntime(config, events)
    _install_fake_lane(monkeypatch, events, runtime)
    backend = AutoAVSRBackend(config)
    monkeypatch.setattr(backend, "available", lambda: (True, "ready"))
    crops = np.full((40, 96, 96, 3), 127, dtype=np.uint8)

    prediction = await backend.infer(crops, fps=25.0)
    unloaded = await backend.unload(reason="unit_done")

    assert prediction.transcript == "unit visual prediction"
    assert unloaded is True
    assert events == [
        "acquire:auto-avsr-visual-speech:False",
        "owned:auto-avsr-visual-speech-load:180.0",
        "runtime:load",
        "preemptible:True",
        "preemptible:False",
        "owned:auto-avsr-visual-speech-inference:180.0",
        "runtime:infer:40",
        "preemptible:True",
        "owned:auto-avsr-unload-clear:None",
        "runtime:clear",
        "release:unit_done",
    ]
    status = backend.get_status()
    assert status["loads"] == 1
    assert status["inferences"] == 1
    assert status["loaded"] is False
    assert status["lane_owned"] is False


@pytest.mark.asyncio
async def test_failed_load_clears_runtime_and_releases_lane(monkeypatch) -> None:
    events: list[str] = []
    config = AutoAVSRConfig()
    runtime = FakeRuntime(config, events)
    runtime.fail_load = True
    _install_fake_lane(monkeypatch, events, runtime)
    backend = AutoAVSRBackend(config)
    monkeypatch.setattr(backend, "available", lambda: (True, "ready"))

    with pytest.raises(RuntimeError, match="load failed"):
        await backend.infer(np.zeros((40, 96, 96, 3), dtype=np.uint8), fps=25.0)

    assert "runtime:clear" in events
    assert "release:auto_avsr_load_failed" in events
    assert backend.get_status()["lane_owned"] is False
    assert backend.get_status()["loads"] == 0


def test_availability_rejects_manifest_mismatch_before_model_load(tmp_path) -> None:
    model_root = tmp_path / "auto_avsr"
    model_root.mkdir()
    (model_root / "manifest.json").write_text('{"checkpoint_sha256": "wrong"}')
    backend = AutoAVSRBackend(AutoAVSRConfig(model_root=model_root))

    available, reason = backend.available()

    assert available is False
    assert reason == "manifest_mismatch:checkpoint_sha256"


@pytest.mark.live
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_checkpoint_runs_inside_model_lane_and_unloads() -> None:
    backend = AutoAVSRBackend(AutoAVSRConfig(beam_size=2, torch_threads=4))
    available, reason = backend.available()
    assert available, f"Auto-AVSR model unavailable: {reason}"
    crops = np.zeros((40, 96, 96, 3), dtype=np.uint8)
    for index in range(40):
        opening = 4 + index % 8
        crops[index, 36 - opening : 60 + opening, 18:78] = 180

    loaded = False
    try:
        prediction = await backend.infer(crops, fps=25.0)
        loaded = True
        status = backend.get_status()

        assert prediction.backend == "auto_avsr"
        assert prediction.calibrated is False
        assert prediction.confidence is None
        assert status["loaded"] is True
        assert status["lane_owned"] is True
        assert status["integrity_verified"] is True
        assert status["loads"] == 1
        assert status["inferences"] == 1
    finally:
        unloaded = await backend.unload(reason="live_model_proof_complete")
        assert unloaded is loaded
