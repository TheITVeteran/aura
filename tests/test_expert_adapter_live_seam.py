"""Contract tests for the expert-adapter live seam.

Covers the chain Bryan's "more weights" build rides on:
worker attach/detach bookkeeping → client IPC guards → native
reload_model_artifact (the retired monkey-patch's replacement) → library
async residency honesty → router hook safety.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.brain.expert_lora_library import (
    ExpertLoRALibrary,
    LoRAAdapter,
)
from core.brain.llm.mlx_client import MLXLocalClient

pytestmark = pytest.mark.unit


# ── worker helpers (fake module tree, real bookkeeping) ──────────────────────

class LoRALinear:  # noqa: N801 - type NAME is the worker's detection contract
    def __init__(self, linear):
        self.linear = linear


class PlainLinear:
    pass


class FakeModel:
    def __init__(self):
        self.modules_by_name = {"layers.0.q_proj": PlainLinear()}
        self.updates = []

    def named_modules(self):
        return list(self.modules_by_name.items())

    def update_modules(self, tree):
        self.updates.append(tree)


def test_worker_attach_records_only_newly_wrapped_layers(monkeypatch):
    from core.brain.llm import mlx_worker

    model = FakeModel()
    # a pre-existing LoRA layer (the personality adapter) must NOT be recorded
    model.modules_by_name["layers.0.v_proj"] = LoRALinear(PlainLinear())

    def fake_load_adapters(m, path):
        m.modules_by_name["layers.0.q_proj"] = LoRALinear(
            m.modules_by_name["layers.0.q_proj"]
        )

    monkeypatch.setattr("mlx_lm.tuner.utils.load_adapters", fake_load_adapters)
    wrapped = mlx_worker._attach_expert_adapter(model, "/fake/adapter")
    assert [name for name, _ in wrapped] == ["layers.0.q_proj"]


def test_worker_attach_failure_unwinds_partial_wrap(monkeypatch):
    from core.brain.llm import mlx_worker

    model = FakeModel()

    def failing_load_adapters(m, path):
        m.modules_by_name["layers.0.q_proj"] = LoRALinear(
            m.modules_by_name["layers.0.q_proj"]
        )
        raise RuntimeError("weights corrupt")

    monkeypatch.setattr("mlx_lm.tuner.utils.load_adapters", failing_load_adapters)
    with pytest.raises(RuntimeError, match="weights corrupt"):
        mlx_worker._attach_expert_adapter(model, "/fake/adapter")
    # the unwind restored the original module via update_modules
    assert len(model.updates) == 1


def test_worker_detach_restores_wrapped_module():
    from core.brain.llm import mlx_worker

    model = FakeModel()
    original = model.modules_by_name["layers.0.q_proj"]
    wrapper = LoRALinear(original)
    restored = mlx_worker._detach_expert_adapter(model, [("layers.0.q_proj", wrapper)])
    assert restored == 1
    assert len(model.updates) == 1


def test_worker_dispatch_handles_set_expert_adapter():
    source = open("core/brain/llm/mlx_worker.py", encoding="utf-8").read()
    assert 'elif action == "set_expert_adapter":' in source
    # KV caches must be invalidated on weight change (CP126: invalidation is
    # now proven-or-fatal, validation precedes mutation, and a failed attach
    # rolls back to the previous identity instead of silently going bare).
    handler = source.split('elif action == "set_expert_adapter":', 1)[1][:8000]
    assert "prompt_cache_lru.clear()" in handler
    assert "_clear_mlx_cache" in handler
    assert "metal_semaphore" in handler
    assert "_validate_expert_adapter_dir" in handler
    assert "_unrestorable_wrapped" in handler
    assert "restored_previous" in handler
    assert "cache_invalidated" in handler


# ── client guards ─────────────────────────────────────────────────────────────

class ProcessProbe:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


@pytest.fixture
def client():
    c = MLXLocalClient(model_path="Qwen2.5-32B-cortex-test")
    yield c
    c.close()


async def test_set_expert_adapter_refuses_without_worker(client, tmp_path):
    res = await client.set_expert_adapter(str(tmp_path / "some-adapter"))
    assert res == {"ok": False, "reason": "worker_not_ready"}


async def test_set_expert_adapter_refuses_mid_generation(client, tmp_path):
    client._req_q = object()
    client._process = ProcessProbe(alive=True)
    client._init_done = True
    client._active_generations = 1
    res = await client.set_expert_adapter(str(tmp_path))
    client._req_q = None
    assert res == {"ok": False, "reason": "generation_active"}


async def test_set_expert_adapter_refuses_missing_dir(client):
    client._req_q = object()
    client._process = ProcessProbe(alive=True)
    client._init_done = True
    res = await client.set_expert_adapter("/nonexistent/adapter-xyz")
    client._req_q = None
    assert res["ok"] is False and res["reason"].startswith("adapter_missing")


async def test_reload_model_artifact_refuses_missing_dir(client):
    res = await client.reload_model_artifact("/nonexistent/fused-xyz")
    assert res["ok"] is False and res["reason"].startswith("artifact_missing")


async def test_reload_model_artifact_defers_while_busy(client, tmp_path):
    artifact = tmp_path / "fused-gen1"
    artifact.mkdir()
    client._active_generations = 1
    res = await client.reload_model_artifact(str(artifact))
    assert res["ok"] is True and res["mode"] == "deferred"
    assert client.model_path == str(artifact)
    assert client._deferred_reboot_reason == "promoted_artifact_swap"
    client._deferred_reboot_reason = None
    client._active_generations = 0


async def test_reload_model_artifact_recycles_when_idle(client, tmp_path, monkeypatch):
    artifact = tmp_path / "fused-gen2"
    artifact.mkdir()
    calls = {}

    async def fake_reboot(reason="", mark_failed=True):
        calls["reason"] = reason
        calls["mark_failed"] = mark_failed

    monkeypatch.setattr(client, "reboot_worker", fake_reboot)
    res = await client.reload_model_artifact(str(artifact))
    assert res["ok"] is True and res["mode"] == "recycled"
    assert calls == {"reason": "promoted_artifact_swap", "mark_failed": False}
    assert client.model_path == str(artifact)


async def test_promoted_swap_deferred_reboot_is_not_a_failure(client, monkeypatch):
    calls = {}

    async def fake_reboot(reason="", mark_failed=True):
        calls["reason"] = reason
        calls["mark_failed"] = mark_failed

    monkeypatch.setattr(client, "reboot_worker", fake_reboot)
    await client._resolve_deferred_reboot("promoted_artifact_swap")
    assert calls == {"reason": "promoted_artifact_swap", "mark_failed": False}


def test_monkey_patch_is_gone():
    source = open("core/learning/live_learner.py", encoding="utf-8").read()
    assert "types.MethodType" not in source
    assert "async def patch_mlx_client_for_hot_swap" not in source
    # the in-orchestrator model load (the ~20GB memory bomb) must never return
    assert "self_or_client._model = new_model" not in source


# ── library async residency ───────────────────────────────────────────────────

class RecordingAsyncApplier:
    def __init__(self, load_ok=True):
        self.load_ok = load_ok
        self.loaded = []
        self.unloaded = []

    async def load(self, adapter):
        self.loaded.append(adapter.name)
        return self.load_ok

    async def unload(self, adapter):
        self.unloaded.append(adapter.name)
        return True


@pytest.fixture
def library(tmp_path):
    lib = ExpertLoRALibrary(manifest_path=tmp_path / "library.json", max_resident=1)
    lib.register(LoRAAdapter(name="math-a", path=str(tmp_path / "a"),
                             task_types={"math"}, keywords={"arithmetic"}, quality=0.8))
    lib.register(LoRAAdapter(name="logic-b", path=str(tmp_path / "b"),
                             task_types={"logic"}, keywords={"deduction"}, quality=0.7))
    return lib


async def test_activate_async_reflects_applier_truth(library):
    applier = RecordingAsyncApplier(load_ok=False)
    ok = await library.activate_async("math-a", applier)
    assert ok is False
    assert library.resident() == []  # a refused swap is never claimed resident


async def test_activate_async_evicts_lru_through_applier(library):
    applier = RecordingAsyncApplier()
    assert await library.activate_async("math-a", applier)
    assert await library.activate_async("logic-b", applier)
    assert applier.unloaded == ["math-a"]  # max_resident=1 evicted the LRU
    assert library.resident() == ["logic-b"]


async def test_select_and_activate_async_is_env_gated(library, monkeypatch):
    monkeypatch.delenv("AURA_EXPERT_LORA_LIBRARY", raising=False)
    applier = RecordingAsyncApplier()
    got = await library.select_and_activate_async("solve arithmetic", "math", applier)
    assert got is None and applier.loaded == []

    monkeypatch.setenv("AURA_EXPERT_LORA_LIBRARY", "1")
    got = await library.select_and_activate_async("solve arithmetic", "math", applier)
    assert got is not None and got.name == "math-a"
    assert applier.loaded == ["math-a"]


# ── router hook safety ────────────────────────────────────────────────────────

async def test_router_hook_is_default_off(monkeypatch):
    from core.brain.llm_health_router import HealthAwareLLMRouter

    monkeypatch.delenv("AURA_EXPERT_LORA_ROUTING", raising=False)
    router = HealthAwareLLMRouter.__new__(HealthAwareLLMRouter)  # no full init needed
    # must be a no-op that never raises, even half-constructed
    await router._maybe_route_expert_adapter("prompt", {})
