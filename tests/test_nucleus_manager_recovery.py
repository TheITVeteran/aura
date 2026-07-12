from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from core.brain.llm.nucleus_manager import NucleusManager
from core.runtime.errors import get_degradation_tracker


def _manager_with_missing_models(tmp_path) -> NucleusManager:
    manager = NucleusManager()
    manager.bus = None
    manager.brainstem_path = str(tmp_path / "missing-brainstem")
    manager.cortex_path = str(tmp_path / "missing-cortex")
    return manager


@pytest.mark.asyncio
async def test_missing_model_path_marks_lane_unavailable(tmp_path):
    tracker = get_degradation_tracker()
    tracker.reset()
    manager = _manager_with_missing_models(tmp_path)
    manager.models.pop("brainstem", None)

    loaded = await manager.load_model("brainstem")

    assert loaded is False
    assert manager.models["brainstem"]["loaded"] is False
    assert "missing-brainstem" in manager.models["brainstem"]["last_error"]
    assert tracker.count("nucleus_manager", "warning") >= 1


@pytest.mark.asyncio
async def test_generate_text_missing_models_returns_deterministic_offline_marker(tmp_path):
    tracker = get_degradation_tracker()
    tracker.reset()
    manager = _manager_with_missing_models(tmp_path)

    text = await manager.generate_text_async("hello")

    assert text == "[NUCLEUS ERROR] Internal inference offline."
    assert manager.models["cortex"]["loaded"] is False
    assert manager.models["brainstem"]["loaded"] is False
    assert tracker.count("nucleus_manager", "degraded") >= 1


@pytest.mark.asyncio
async def test_generate_stream_missing_brainstem_entry_returns_offline_marker(tmp_path):
    tracker = get_degradation_tracker()
    tracker.reset()
    manager = _manager_with_missing_models(tmp_path)
    manager.models.pop("brainstem", None)

    chunks = [
        chunk
        async for chunk in manager.generate_stream_async("status", origin="health_monitor")
    ]

    assert chunks == ["[NUCLEUS ERROR] Internal inference offline."]
    assert "brainstem" in manager.models
    assert manager.models["brainstem"]["loaded"] is False
    assert tracker.count("nucleus_manager", "degraded") >= 1


@pytest.mark.asyncio
async def test_unload_models_clears_loaded_entries_even_when_cache_clear_is_unavailable(tmp_path):
    manager = _manager_with_missing_models(tmp_path)
    manager.models["cortex"].update({
        "model": object(),
        "tokenizer": object(),
        "loaded": True,
        "cache": object(),
    })

    await manager.unload_models()

    assert manager.models["cortex"]["model"] is None
    assert manager.models["cortex"]["tokenizer"] is None
    assert manager.models["cortex"]["loaded"] is False
    assert manager.models["cortex"]["cache"] is None


@pytest.mark.asyncio
async def test_loaded_nucleus_model_holds_lane_lease_until_unload(
    tmp_path,
    monkeypatch,
):
    from core.runtime import model_lane_control
    from core.utils import gpu_sentinel

    model_path = tmp_path / "brainstem-7b"
    model_path.mkdir()
    manager = _manager_with_missing_models(tmp_path)
    manager.brainstem_path = str(model_path)
    captured: list[dict[str, object]] = []

    class _Lease:
        released = False

        async def release(self, *, reason):
            captured.append({"release_reason": reason})
            self.released = True
            return True

    lease = _Lease()

    async def _acquire(**kwargs):
        captured.append(kwargs)
        return lease

    class _Sentinel:
        def acquire(self, **_kwargs):
            return True

        def release(self):
            return None

    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", _acquire)
    monkeypatch.setattr(gpu_sentinel, "get_gpu_sentinel", lambda: _Sentinel())
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(load=lambda *_args, **_kwargs: (object(), object())),
    )

    assert await manager.load_model("brainstem") is True
    assert manager.models["brainstem"]["lane_lease"] is lease
    assert captured[0]["purpose"] == "serve"
    assert captured[0]["preemptible"] is False

    await manager.unload_models()

    assert lease.released is True
    assert manager.models["brainstem"]["lane_lease"] is None


@pytest.mark.asyncio
async def test_concurrent_nucleus_loaders_materialize_one_owned_model(
    tmp_path,
    monkeypatch,
) -> None:
    from core.runtime import model_lane_control
    from core.utils import gpu_sentinel

    model_path = tmp_path / "brainstem-7b"
    model_path.mkdir()
    manager = _manager_with_missing_models(tmp_path)
    manager.brainstem_path = str(model_path)
    acquire_calls = 0
    load_calls = 0
    load_entered = threading.Event()
    release_load = threading.Event()

    class _Lease:
        async def release(self, *, reason):
            return bool(reason)

    async def _acquire(**_kwargs):
        nonlocal acquire_calls
        acquire_calls += 1
        return _Lease()

    class _Sentinel:
        def acquire(self, **_kwargs):
            return True

        def release(self):
            return None

    def _load(*_args, **_kwargs):
        nonlocal load_calls
        load_calls += 1
        load_entered.set()
        assert release_load.wait(timeout=2.0)
        return object(), object()

    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", _acquire)
    monkeypatch.setattr(gpu_sentinel, "get_gpu_sentinel", lambda: _Sentinel())
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=_load))

    first = asyncio.create_task(manager.load_model("brainstem"))
    assert await asyncio.to_thread(load_entered.wait, 1.0)
    second = asyncio.create_task(manager.load_model("brainstem"))
    await asyncio.sleep(0.02)
    release_load.set()

    assert await asyncio.gather(first, second) == [True, True]
    assert acquire_calls == 1
    assert load_calls == 1

    await manager.unload_models()


@pytest.mark.asyncio
async def test_nucleus_unload_waits_for_exact_worker_model_use(tmp_path) -> None:
    manager = _manager_with_missing_models(tmp_path)
    entered = threading.Event()
    release_generation = threading.Event()
    release_reasons: list[str] = []

    class _Lease:
        async def release(self, *, reason: str) -> bool:
            release_reasons.append(reason)
            return True

    model = object()
    tokenizer = object()
    manager.models["cortex"].update(
        {
            "model": model,
            "tokenizer": tokenizer,
            "loaded": True,
            "lane_lease": _Lease(),
        }
    )

    def _hold_generation() -> None:
        with manager._model_thread_context("cortex") as (_entry, owned_model, owned_tokenizer):
            assert owned_model is model
            assert owned_tokenizer is tokenizer
            entered.set()
            assert release_generation.wait(2.0)

    generation = asyncio.create_task(asyncio.to_thread(_hold_generation))
    assert await asyncio.to_thread(entered.wait, 1.0)
    unload = asyncio.create_task(
        manager._unload_model_entry("cortex", reason="optimizer_completed")
    )
    await asyncio.sleep(0.05)

    assert unload.done() is False
    assert release_reasons == []
    assert manager.models["cortex"]["model"] is model

    release_generation.set()
    await generation
    await asyncio.wait_for(unload, timeout=1.0)

    assert manager.models["cortex"]["model"] is None
    assert manager.models["cortex"]["tokenizer"] is None
    assert manager.models["cortex"]["loaded"] is False
    assert release_reasons == ["nucleus_unload:optimizer_completed"]
