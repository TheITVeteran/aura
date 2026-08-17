from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.orchestrator.main import RobustOrchestrator
from core.orchestrator.mixins.learning_evolution import (
    _has_healthy_local_inference_endpoint,
)


class _RetryableLocalGate:
    def __init__(self, _orchestrator):
        self._initialized = False
        self._init_error = "local_worker_unavailable"
        self.attempts = 0

    async def initialize(self):
        self.attempts += 1
        self._initialized = self.attempts >= 2
        self._init_error = None if self._initialized else "local_worker_unavailable"

    def initialization_receipt(self):
        return {
            "mode": "local",
            "reason": self._init_error or "",
        }


class _RaisingLocalGate:
    def __init__(self, _orchestrator):
        self._initialized = False

    async def initialize(self):
        raise RuntimeError("local_runtime_broken")


@pytest.mark.asyncio
async def test_failed_local_gate_remains_unready_and_retries_same_instance(
    monkeypatch, service_container
):
    import core.brain.inference_gate as gate_module
    import core.orchestrator.main as main_module

    monkeypatch.setenv("AURA_FULL_TEST_BOOT", "1")
    monkeypatch.setattr(gate_module, "InferenceGate", _RetryableLocalGate)
    monkeypatch.setattr(main_module, "_record_main_degradation", lambda *_args, **_kwargs: None)
    orchestrator = RobustOrchestrator.__new__(RobustOrchestrator)
    orchestrator._inference_gate = None

    assert await orchestrator._ensure_inference_gate_ready("first_attempt") is False
    gate = orchestrator._inference_gate
    assert gate is ServiceContainer.get("inference_gate")
    assert gate._initialized is False
    assert gate.attempts == 1

    assert await orchestrator._ensure_inference_gate_ready("second_attempt") is True
    assert orchestrator._inference_gate is gate
    assert gate._initialized is True
    assert gate.attempts == 2


@pytest.mark.asyncio
async def test_raised_local_initialization_never_forges_readiness(
    monkeypatch, service_container
):
    import core.brain.inference_gate as gate_module
    import core.orchestrator.main as main_module

    monkeypatch.setenv("AURA_FULL_TEST_BOOT", "1")
    monkeypatch.setattr(gate_module, "InferenceGate", _RaisingLocalGate)
    monkeypatch.setattr(main_module, "_record_main_degradation", lambda *_args, **_kwargs: None)
    orchestrator = RobustOrchestrator.__new__(RobustOrchestrator)
    orchestrator._inference_gate = None

    assert await orchestrator._ensure_inference_gate_ready("raised_attempt") is False
    assert orchestrator._inference_gate._initialized is False
    assert ServiceContainer.get("inference_gate") is orchestrator._inference_gate


def test_learning_admission_counts_only_healthy_local_endpoints():
    health = {
        "local-ready": True,
        "local-down": False,
        "off-host-ready": True,
    }
    router = SimpleNamespace(
        endpoints={
            "local-ready": SimpleNamespace(name="local-ready", is_local=True),
            "local-down": SimpleNamespace(name="local-down", is_local=True),
            "off-host-ready": SimpleNamespace(name="off-host-ready", is_local=False),
        },
        health_monitor=SimpleNamespace(peek_healthy=lambda name: health[name]),
    )

    assert _has_healthy_local_inference_endpoint(router) is True
    health["local-ready"] = False
    assert _has_healthy_local_inference_endpoint(router) is False


def test_orchestrator_sources_make_no_retired_model_recovery_claims():
    from core.ops import resilient_boot
    from core.orchestrator import boot, main
    from core.orchestrator.mixins import (
        learning_evolution,
        message_handling,
        message_pipeline,
    )

    retired_claims = ("gemini", "cloud-only", "cloud fallback", "cloud recovery")
    for module in (
        boot,
        main,
        message_pipeline,
        learning_evolution,
        message_handling,
        resilient_boot,
    ):
        source = inspect.getsource(module).lower()
        for claim in retired_claims:
            assert claim not in source, f"{module.__name__} still contains {claim!r}"
