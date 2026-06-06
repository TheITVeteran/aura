"""tests/test_agi_integration.py
================================
Tests for the AGIIntegrationLayer coordinator — tick execution,
telemetry aggregation, modulation retrieval, inference callbacks,
and graceful degradation.
"""

import asyncio
import inspect
import threading
import time
import numpy as np
import pytest
from types import SimpleNamespace

from core.agi.agi_integration import AGIIntegrationLayer
from core.brain.homeostatic_modulator import InferenceModulation
from core.container import ServiceContainer


class RecordedCall:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class CallRecorder:
    def __init__(self, result=None, *, side_effect=None):
        self.result = result
        self.side_effect = side_effect
        self.calls = []
        self.call_args = None

    @property
    def call_count(self):
        return len(self.calls)

    def __call__(self, *args, **kwargs):
        call = RecordedCall(args, kwargs)
        self.calls.append(call)
        self.call_args = call
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        return self.result

    def assert_called_once(self):
        assert len(self.calls) == 1

    def assert_not_called(self):
        assert not self.calls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def integration_layer():
    """Create a fresh AGIIntegrationLayer for each test."""
    layer = AGIIntegrationLayer()
    yield layer


@pytest.fixture
def base_modulation():
    return InferenceModulation(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        logit_bias={},
        head_weights=np.ones(32, dtype=np.float32),
        urgency=0.5,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_integration_layer_initializes_cleanly(integration_layer):
    assert integration_layer.tick_count == 0
    assert integration_layer.last_tick_time == 0.0
    assert not integration_layer._running
    assert integration_layer.modulator is not None
    assert integration_layer.feedback_loop is not None


# ---------------------------------------------------------------------------
# Single tick execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_tick_increments_counter(integration_layer, monkeypatch):
    """A single tick should increment tick_count and update last_tick_time."""
    monkeypatch.setattr(ServiceContainer, "get", lambda *args, **kwargs: None)
    await integration_layer._run_tick()

    assert integration_layer.tick_count == 1
    assert integration_layer.last_tick_time > 0.0


@pytest.mark.asyncio
async def test_run_tick_steps_precision_engine(integration_layer, monkeypatch):
    """Tick should call precision.step() when the engine is registered."""
    precision = SimpleNamespace(step=CallRecorder())

    def lookup(name, default=None):
        if name == "precision_engine":
            return precision
        return default

    monkeypatch.setattr(ServiceContainer, "get", lookup)
    await integration_layer._run_tick()

    precision.step.assert_called_once()


@pytest.mark.asyncio
async def test_run_tick_contraction_every_30_ticks(integration_layer, monkeypatch):
    """Dimensional contraction should trigger every 30 ticks."""
    expansion = SimpleNamespace(evaluate_contraction=CallRecorder([]))

    def lookup(name, default=None):
        if name == "dimensional_expansion":
            return expansion
        return default

    monkeypatch.setattr(ServiceContainer, "get", lookup)
    # Run 29 ticks — no contraction yet
    for _ in range(29):
        await integration_layer._run_tick()
    expansion.evaluate_contraction.assert_not_called()

    # 30th tick triggers contraction
    await integration_layer._run_tick()
    expansion.evaluate_contraction.assert_called_once()


@pytest.mark.asyncio
async def test_run_tick_saves_projection_every_300s(integration_layer, monkeypatch):
    """Projection weights save should trigger after 300 seconds elapse."""
    # Force last_save_time into the past
    integration_layer.last_save_time = time.time() - 301.0

    projection = SimpleNamespace(save=CallRecorder())
    integration_layer.modulator.projection = projection

    monkeypatch.setattr(ServiceContainer, "get", lambda *args, **kwargs: None)
    await integration_layer._run_tick()

    projection.save.assert_called_once()


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------

def test_on_inference_complete_returns_metrics(integration_layer, base_modulation, monkeypatch):
    """Callback should return surprise/coherence dict."""
    substrate = SimpleNamespace(
        idx_valence=0,
        idx_arousal=1,
        x=np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    def lookup(name, default=None):
        if name == "liquid_substrate":
            return substrate
        return default

    monkeypatch.setattr(ServiceContainer, "get", lookup)
    metrics = integration_layer.on_inference_complete(
        output_text="test output",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.15],
        modulation=base_modulation,
    )

    assert "surprise" in metrics
    assert "coherence" in metrics
    assert isinstance(metrics["surprise"], float)
    assert isinstance(metrics["coherence"], float)


def test_on_inference_complete_graceful_degradation(integration_layer, base_modulation, monkeypatch):
    """If feedback loop throws, callback should return safe defaults."""
    monkeypatch.setattr(
        integration_layer.feedback_loop,
        "process_output",
        CallRecorder(side_effect=RuntimeError("boom")),
    )
    metrics = integration_layer.on_inference_complete(
        output_text="test",
        token_ids=[1],
        logprobs=None,
        modulation=base_modulation,
    )

    assert metrics == {"surprise": 0.5, "coherence": 0.0}


# ---------------------------------------------------------------------------
# Modulation retrieval
# ---------------------------------------------------------------------------

def test_get_modulation_returns_inference_modulation(integration_layer, monkeypatch):
    """get_modulation should return an InferenceModulation dataclass."""
    monkeypatch.setattr(ServiceContainer, "get", lambda *args, **kwargs: None)
    mod = integration_layer.get_modulation()

    assert isinstance(mod, InferenceModulation)
    assert 0.0 < mod.temperature <= 1.5
    assert 0.0 < mod.top_p <= 1.0


def test_get_modulation_graceful_degradation(integration_layer, monkeypatch):
    """If modulator throws, get_modulation should return safe defaults."""
    monkeypatch.setattr(
        integration_layer.modulator,
        "compute_modulation",
        CallRecorder(side_effect=RuntimeError("broken")),
    )
    mod = integration_layer.get_modulation()

    assert isinstance(mod, InferenceModulation)
    assert mod.temperature == 0.7
    assert mod.top_p == 0.9


# ---------------------------------------------------------------------------
# Telemetry aggregation
# ---------------------------------------------------------------------------

def test_get_unified_telemetry_minimal(integration_layer, monkeypatch):
    """Telemetry should include integration block even with no services."""
    monkeypatch.setattr(ServiceContainer, "get", lambda *args, **kwargs: None)
    telemetry = integration_layer.get_unified_telemetry()

    assert "integration" in telemetry
    assert telemetry["integration"]["ticks"] == 0
    assert "uptime_seconds" in telemetry["integration"]


def test_get_unified_telemetry_with_services(integration_layer, monkeypatch):
    """Telemetry should aggregate data from all registered services."""
    precision = SimpleNamespace(get_state_dict=CallRecorder({"arousal": 0.6, "fatigue": 0.2}))

    substrate = SimpleNamespace(
        idx_valence=0,
        idx_arousal=1,
        idx_frustration=2,
        idx_curiosity=3,
        idx_focus=4,
        x=np.array([0.7, 0.5, 0.1, 0.3, 0.8], dtype=np.float64),
        sync_lock=threading.Lock(),
    )

    free_energy = SimpleNamespace(smoothed_fe=0.42, current_action="explore")

    expansion = SimpleNamespace(get_status=CallRecorder({"current_dim": 18, "expanded_count": 2}))

    registry = SimpleNamespace(
        synthesized_actuators={"a": 1},
        actuators={"a": 1, "b": 2, "c": 3},
    )

    def lookup(name, default=None):
        mapping = {
            "precision_engine": precision,
            "liquid_substrate": substrate,
            "free_energy_engine": free_energy,
            "dimensional_expansion": expansion,
            "actuator_registry": registry,
        }
        return mapping.get(name, default)

    monkeypatch.setattr(ServiceContainer, "get", lookup)
    telemetry = integration_layer.get_unified_telemetry()

    assert "precision" in telemetry
    assert telemetry["precision"]["arousal"] == 0.6

    assert "substrate" in telemetry
    assert telemetry["substrate"]["valence"] == 0.7
    assert telemetry["substrate"]["frustration"] == 0.1

    assert "free_energy" in telemetry
    assert telemetry["free_energy"]["smoothed_free_energy"] == 0.42

    assert "dimensional_expansion" in telemetry
    assert telemetry["dimensional_expansion"]["current_dim"] == 18
    assert telemetry["dimensional_expansion"]["expanded_count"] == 2

    assert "actuators" in telemetry
    assert telemetry["actuators"]["synthesized_count"] == 1
    assert telemetry["actuators"]["total_count"] == 3


# ---------------------------------------------------------------------------
# Tick loop resilience
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_tick_survives_precision_engine_crash(integration_layer, monkeypatch):
    """If PrecisionEngine.step() throws, the tick should complete gracefully."""
    precision = SimpleNamespace(step=CallRecorder(side_effect=RuntimeError("FHN divergence")))

    def lookup(name, default=None):
        if name == "precision_engine":
            return precision
        return default

    monkeypatch.setattr(ServiceContainer, "get", lookup)
    # Should NOT raise
    await integration_layer._run_tick()

    assert integration_layer.tick_count == 1


@pytest.mark.asyncio
async def test_run_tick_survives_expansion_crash(integration_layer, monkeypatch):
    """If evaluate_contraction() throws, the tick should complete gracefully."""
    expansion = SimpleNamespace(
        evaluate_contraction=CallRecorder(side_effect=ValueError("matrix singular"))
    )

    def lookup(name, default=None):
        if name == "dimensional_expansion":
            return expansion
        return default

    # Force tick_count to 29 so the 30th tick triggers contraction
    integration_layer.tick_count = 29

    monkeypatch.setattr(ServiceContainer, "get", lookup)
    await integration_layer._run_tick()

    assert integration_layer.tick_count == 30
    expansion.evaluate_contraction.assert_called_once()


# ---------------------------------------------------------------------------
# Start / Stop lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_stop_lifecycle(integration_layer, monkeypatch):
    """Start should set _running, stop should clear it and save projection."""
    task = SimpleNamespace(cancel=CallRecorder())

    def create_tracked_task(awaitable, **_kwargs):
        if inspect.isawaitable(awaitable):
            awaitable.close()
        return task

    tracker = SimpleNamespace(create_task=CallRecorder(side_effect=create_tracked_task))

    projection = SimpleNamespace(save=CallRecorder())
    integration_layer.modulator.projection = projection

    monkeypatch.setattr("core.utils.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr(ServiceContainer, "register", CallRecorder())
    await integration_layer.start()

    assert integration_layer._running is True
    tracker.create_task.assert_called_once()

    await integration_layer.stop()
    assert integration_layer._running is False
    task.cancel.assert_called_once()
    projection.save.assert_called_once()


@pytest.mark.asyncio
async def test_double_start_is_idempotent(integration_layer, monkeypatch):
    """Calling start() twice should not spawn a second loop."""
    task = SimpleNamespace(cancel=CallRecorder())

    def create_tracked_task(awaitable, **_kwargs):
        if inspect.isawaitable(awaitable):
            awaitable.close()
        return task

    tracker = SimpleNamespace(create_task=CallRecorder(side_effect=create_tracked_task))

    monkeypatch.setattr("core.utils.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr(ServiceContainer, "register", CallRecorder())
    await integration_layer.start()
    await integration_layer.start()

    assert tracker.create_task.call_count == 1
    await integration_layer.stop()
