from __future__ import annotations

import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_local_code_model_holds_lane_lease_through_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from core.brain.llm import local_code_model
    from core.runtime import model_lane_control

    model_path = tmp_path / "coder-7b"
    model_path.mkdir()
    captured: list[dict[str, object]] = []

    class _Lease:
        released = False

        async def set_preemptible(self, preemptible):
            captured.append({"activated_preemptible": preemptible})
            return True

        async def release(self, *, reason):
            captured.append({"release_reason": reason})
            self.released = True
            return True

    lease = _Lease()

    async def _acquire(**kwargs):
        captured.append(kwargs)
        return lease

    class _Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "formatted-code-prompt"

        def encode(self, text):
            return list(range(max(1, len(text.split()))))

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = lambda *_args, **_kwargs: (object(), _Tokenizer())
    mlx_lm.generate = lambda *_args, **_kwargs: "def answer():\n    return 42"
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", _acquire)
    monkeypatch.setattr(local_code_model, "_model", None)
    monkeypatch.setattr(local_code_model, "_tokenizer", None)
    monkeypatch.setattr(local_code_model, "_loaded_path", None)
    monkeypatch.setattr(local_code_model, "_loaded_identity", None)
    monkeypatch.setattr(local_code_model, "_lane_lease", None)
    identity = local_code_model.TrustedModelIdentity(
        real_path=str(model_path),
        checkpoint_fingerprint="a" * 64,
        checkpoint_files=1,
        behavior_bundle_sha256="b" * 64,
        architecture="Qwen2ForCausalLM",
        max_context_tokens=4096,
        trust_manifest_sha256="c" * 64,
    )
    monkeypatch.setattr(local_code_model, "_validate_model_trust", lambda _path: identity)

    model = local_code_model.LocalCodeModel(str(model_path))
    result = await model.think("write answer")

    assert "return 42" in result
    receipt = model.last_receipt()
    assert receipt is not None
    assert receipt["route"] == "local_mlx_unsteered_no_cloud"
    assert receipt["steering_hooks"] == ()
    assert receipt["validation_status"] == "complete_nonempty"
    assert receipt["checkpoint_fingerprint"] == "a" * 64
    assert captured[0]["purpose"] == "serve"
    assert captured[0]["preemptible"] is False
    assert captured[1] == {"activated_preemptible": True}
    assert local_code_model._lane_lease is lease

    await model.close()

    assert lease.released is True
    assert local_code_model._model is None
    assert local_code_model._lane_lease is None


@pytest.mark.asyncio
async def test_local_code_model_refuses_eviction_during_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from core.brain.llm import local_code_model
    from core.runtime import model_lane_control

    model_path = tmp_path / "coder-7b"
    model_path.mkdir()
    generation_started = threading.Event()
    finish_generation = threading.Event()

    class _Lease:
        async def set_preemptible(self, preemptible):
            return bool(preemptible)

        async def release(self, *, reason):
            return bool(reason)

    async def _acquire(**_kwargs):
        return _Lease()

    class _Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "formatted-code-prompt"

        def encode(self, text):
            return list(range(max(1, len(text.split()))))

    def _generate(*_args, **_kwargs):
        generation_started.set()
        assert finish_generation.wait(timeout=2.0)
        return "done"

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = lambda *_args, **_kwargs: (object(), _Tokenizer())
    mlx_lm.generate = _generate
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", _acquire)
    monkeypatch.setattr(local_code_model, "_model", None)
    monkeypatch.setattr(local_code_model, "_tokenizer", None)
    monkeypatch.setattr(local_code_model, "_loaded_path", None)
    monkeypatch.setattr(local_code_model, "_loaded_identity", None)
    monkeypatch.setattr(local_code_model, "_lane_lease", None)
    monkeypatch.setattr(local_code_model, "_active_generations", 0)
    monkeypatch.setattr(local_code_model, "_eviction_in_progress", False)
    identity = local_code_model.TrustedModelIdentity(
        real_path=str(model_path),
        checkpoint_fingerprint="a" * 64,
        checkpoint_files=1,
        behavior_bundle_sha256="b" * 64,
        architecture="Qwen2ForCausalLM",
        max_context_tokens=4096,
        trust_manifest_sha256="c" * 64,
    )
    monkeypatch.setattr(local_code_model, "_validate_model_trust", lambda _path: identity)

    model = local_code_model.LocalCodeModel(str(model_path))
    generation = asyncio.create_task(model.think("hold"))
    assert await asyncio.to_thread(generation_started.wait, 1.0)
    generation.cancel()
    await asyncio.sleep(0.02)

    assert generation.done() is False
    assert local_code_model._active_generations == 1

    evicted = await local_code_model._evict_local_code_model(
        SimpleNamespace(model_path=str(model_path)),
        "unit-test",
    )
    assert evicted is False
    assert local_code_model._model is not None

    finish_generation.set()
    with pytest.raises(asyncio.CancelledError):
        await generation
    assert local_code_model._active_generations == 0
    await model.close()
