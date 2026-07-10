import asyncio
from types import SimpleNamespace

import pytest

from core.brain.inference_gate import InferenceGate
from core.consciousness.attention_schema import AttentionSchema
from core.consciousness.free_energy import FreeEnergyState
from core.runtime.errors import get_degradation_tracker


class CancellationProbe:
    def __init__(self):
        self.calls = []

    def __call__(self, name, default=None):
        self.calls.append(SimpleNamespace(name=name, default=default))
        raise asyncio.CancelledError()


# ==============================================================================
# InferenceGate Phi Gating Tests
# ==============================================================================

def test_inference_gate_get_system_phi_redundancy(monkeypatch):
    """Verify that _get_system_phi correctly probes the redundant sources."""
    # 1. Test ClosedCausalLoop _loop_state fallback
    closed_loop = SimpleNamespace(_loop_state=SimpleNamespace(phi_estimate=0.75))

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: closed_loop if name == "closed_causal_loop" else default),
    )
    phi = InferenceGate._get_system_phi()
    assert phi == 0.75

    # 2. Test PhiComputer fallback
    phi_computer = SimpleNamespace(latest_phi=0.45)

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(lambda name, default=None: None))
    monkeypatch.setattr("core.consciousness.phi_compute.get_phi_computer", lambda: phi_computer)
    phi = InferenceGate._get_system_phi()
    assert phi == 0.45

    # 3. Test PhiCore fallback
    phi_core = SimpleNamespace(_last_result=SimpleNamespace(phi_s=0.25))

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: phi_core if name == "phi_core" else default),
    )
    monkeypatch.setattr("core.consciousness.phi_compute.get_phi_computer", lambda: None)
    phi = InferenceGate._get_system_phi()
    assert phi == 0.25

    # 4. Test neutral fallback
    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(lambda name, default=None: None))
    monkeypatch.setattr("core.consciousness.phi_compute.get_phi_computer", lambda: None)
    phi = InferenceGate._get_system_phi()
    assert phi == 0.5


def test_inference_gate_get_system_phi_records_typed_probe_failure(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()

    def _service_get(name, default=None):
        if name == "closed_causal_loop":
            raise RuntimeError("loop unavailable")
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(_service_get))
    monkeypatch.setattr("core.consciousness.phi_compute.get_phi_computer", lambda: None)
    phi = InferenceGate._get_system_phi()

    assert phi == 0.5
    records = tracker.recent(subsystem="inference_gate", limit=1)
    assert records
    assert records[-1].action == "continued phi lookup after closed causal loop probe failed"
    tracker.reset()


def test_inference_gate_get_system_phi_propagates_cancellation(monkeypatch):
    probe = CancellationProbe()
    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(probe))
    with pytest.raises(asyncio.CancelledError):
        InferenceGate._get_system_phi()
    assert len(probe.calls) == 1


def test_inference_gate_source_has_no_raw_broad_exception_catches():
    from pathlib import Path

    source = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    assert "except Exception" not in source
    assert "except BaseException" not in source


def test_inference_gate_adaptive_max_tokens_phi_scaling(monkeypatch):
    """Verify Phi scaling remains bounded by the foreground compute profile."""
    monkeypatch.setenv("AURA_FOREGROUND_CHAT_SIMPLE_MAX_TOKENS", "2048")

    # Under high Phi (phi = 0.5 -> scale = 0.6 + 1.0 = 1.6)
    monkeypatch.setattr(InferenceGate, "_get_system_phi", classmethod(lambda cls: 0.5))
    adapted = InferenceGate._adaptive_max_tokens_for_prompt(
        prompt="Hello",
        base_tokens=1000,
        origin="user",
        requested_tier="primary",
        is_background=False
    )
    assert 1500 <= adapted <= 1700

    # Under low Phi (phi = 0.0 -> scale = max(0.5, 0.6 + 0) = 0.6)
    monkeypatch.setattr(InferenceGate, "_get_system_phi", classmethod(lambda cls: 0.0))
    adapted = InferenceGate._adaptive_max_tokens_for_prompt(
        prompt="Hello",
        base_tokens=1000,
        origin="user",
        requested_tier="primary",
        is_background=False
    )
    assert 550 <= adapted <= 650

    # Under nominal Phi (phi = 0.2 -> scale = 0.6 + 0.4 = 1.0)
    monkeypatch.setattr(InferenceGate, "_get_system_phi", classmethod(lambda cls: 0.2))
    adapted = InferenceGate._adaptive_max_tokens_for_prompt(
        prompt="Hello",
        base_tokens=1000,
        origin="user",
        requested_tier="primary",
        is_background=False
    )
    assert 950 <= adapted <= 1050


# ==============================================================================
# AttentionSchema Free Energy Gating Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_attention_schema_free_energy_gating_unrestricted(monkeypatch):
    """Verify that focus shifts occur normally under low Free Energy (F <= 0.6)."""
    schema = AttentionSchema()
    
    # Establish initial focus
    await schema.set_focus(
        content="Analyzing the neural mesh",
        source="curiosity",
        priority=0.4
    )
    assert schema.current_focus.source == "curiosity"
    
    # Under low Free Energy (e.g. F = 0.2), shifting to a different source should succeed
    mock_fe_state = FreeEnergyState(
        surprise=0.1,
        complexity=0.1,
        free_energy=0.2,
        valence=0.6,
        arousal=0.2,
        dominant_action="explore"
    )
    fe_engine = SimpleNamespace(current=mock_fe_state)

    monkeypatch.setattr("core.consciousness.free_energy.get_free_energy_engine", lambda: fe_engine)
    await schema.set_focus(
        content="Responding to query",
        source="affective_steering",
        priority=0.3
    )
    assert schema.current_focus.source == "affective_steering"
    assert schema.current_focus.content == "Responding to query"


@pytest.mark.asyncio
async def test_attention_schema_free_energy_gating_rigidity(monkeypatch, tmp_path):
    """Verify focus stability/rigidity is enforced under high Free Energy (F > 0.6)."""
    import core.consciousness.attention_schema as attention_module
    from core.runtime.receipts import ReceiptStore

    receipt_store = ReceiptStore(tmp_path / "attention-policy-receipts")
    monkeypatch.setattr(
        attention_module,
        "get_receipt_store",
        lambda: receipt_store,
    )
    schema = AttentionSchema()
    
    # Establish initial focus
    await schema.set_focus(
        content="Analyzing the neural mesh",
        source="curiosity",
        priority=0.5
    )
    assert schema.current_focus.source == "curiosity"
    
    # Under high Free Energy (e.g. F = 0.8), shift is blocked if priority is too low
    mock_fe_state = FreeEnergyState(
        surprise=0.8,
        complexity=0.8,
        free_energy=0.8,
        valence=-0.6,
        arousal=0.8,
        dominant_action="update_beliefs"
    )
    fe_engine = SimpleNamespace(current=mock_fe_state)

    monkeypatch.setattr("core.consciousness.free_energy.get_free_energy_engine", lambda: fe_engine)
    # Shift with low priority (0.4) is below the rigidity threshold (0.3 + 0.8 * 0.4 = 0.62)
    blocked_focus = await schema.set_focus(
        content="Responding to query",
        source="affective_steering",
        priority=0.4
    )
    # Should retain the original focus
    assert schema.current_focus.source == "curiosity"
    assert schema.current_focus.content == "Analyzing the neural mesh"
    assert blocked_focus == schema.current_focus

    # Shift with same source is NOT blocked
    await schema.set_focus(
        content="Deepening neural mesh analysis",
        source="curiosity",
        priority=0.1
    )
    assert schema.current_focus.source == "curiosity"
    assert schema.current_focus.content == "Deepening neural mesh analysis"

    # Shift with extremely high priority (0.7) exceeding threshold (0.62) should be allowed
    await schema.set_focus(
        content="Emergency shutdown response",
        source="safety_governor",
        priority=0.7
    )
    assert schema.current_focus.source == "safety_governor"
    assert schema.current_focus.content == "Emergency shutdown response"
