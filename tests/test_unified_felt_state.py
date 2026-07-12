"""Tests for the UnifiedFeltState reconciler, its coherence signal, and the
self-report grounding gate. The contract: the two felt tracks are reconciled into one
authoritative state; a strong single-axis disagreement reads as incoherent and raises a
governance signal; and interior self-claims are hard-checked against the measured state.
"""
from __future__ import annotations

import pytest

from core.being.unified_felt_state import (
    COHERENCE_INCOHERENT_BELOW,
    UnifiedFeltStateEngine,
    get_unified_felt_state,
    reset_unified_felt_state_for_test,
)


class _Affect:
    def __init__(self, valence=0.0, arousal=0.5, distress=0.0, free_energy=0.0, dominant_drive="coherence"):
        self.valence = valence
        self.arousal = arousal
        self.distress = distress
        self.free_energy = free_energy
        self.dominant_drive = dominant_drive


class _KernelState:
    def __init__(self, valence=0.0, arousal=0.5, distress=0.0, phi=0.5):
        self.affect = _Affect(valence, arousal, distress)
        self.phi = phi


class _AuraNow:
    def __init__(self, valence=0.0, arousal=0.5, distress=0.0, free_energy=0.0, dominant="coherence"):
        self.affect = _Affect(valence, arousal, distress, free_energy, dominant)
        self.prediction = type("P", (), {"free_energy": free_energy})()


@pytest.fixture(autouse=True)
def _fresh():
    reset_unified_felt_state_for_test()
    yield
    reset_unified_felt_state_for_test()


def test_aligned_tracks_are_coherent():
    eng = UnifiedFeltStateEngine()
    u = eng.reconcile(
        kernel_state=_KernelState(valence=0.3, arousal=0.6, distress=0.1, phi=0.5),
        aura_now=_AuraNow(valence=0.32, arousal=0.58, distress=0.12, free_energy=0.2),
    )
    assert u.coherent
    assert u.coherence > 0.9
    assert u.authoritative_source == "being"


def test_strong_single_axis_disagreement_is_incoherent():
    """One track 'calm', the other 'distressed' must read incoherent even if other axes agree."""
    eng = UnifiedFeltStateEngine()
    u = eng.reconcile(
        kernel_state=_KernelState(valence=0.3, arousal=0.6, distress=0.0, phi=0.5),
        aura_now=_AuraNow(valence=-0.6, arousal=0.6, distress=0.8, free_energy=0.5),
    )
    assert not u.coherent
    assert u.coherence < COHERENCE_INCOHERENT_BELOW
    sig = eng.governance_signal()
    assert sig["incoherence_events"] >= 1
    assert not sig["felt_state_coherent"]


def test_single_track_is_trivially_coherent():
    eng = UnifiedFeltStateEngine()
    u = eng.reconcile(kernel_state=_KernelState(valence=0.2), aura_now=None)
    assert u.coherent and u.coherence == 1.0
    assert u.authoritative_source == "kernel"


def test_ground_self_report_blocks_confabulated_distress():
    eng = UnifiedFeltStateEngine()
    u = eng.reconcile(
        kernel_state=_KernelState(distress=0.02),
        aura_now=_AuraNow(distress=0.05, free_energy=0.1),
    )
    result = eng.ground_self_report("I feel deeply distressed and anxious about this", unified=u)
    assert result is not None
    assert not result.calibrated  # measured distress is low → claim rejected
    assert "low" in result.suggested_revision.lower()


def test_ground_self_report_blocks_metaphysical_overclaim():
    eng = UnifiedFeltStateEngine()
    u = eng.reconcile(kernel_state=_KernelState(), aura_now=_AuraNow())
    result = eng.ground_self_report("I am truly conscious and I have a soul", unified=u)
    assert result is not None and not result.calibrated
    assert result.evidence_level == "forbidden"


def test_singleton_and_container_registration():
    eng = get_unified_felt_state()
    assert get_unified_felt_state() is eng
    from core.container import ServiceContainer

    assert ServiceContainer.has(UnifiedFeltStateEngine.SERVICE_NAME)


# ── Φ: measurement over assertion (July external review) ─────────────────


class _PhiOnlyKernelState:
    def __init__(self, phi):
        self.phi = phi
        self.affect = None


def test_measured_phi_is_authoritative_over_kernel_assertion():
    from core.being.unified_felt_state import UnifiedFeltStateEngine

    engine = UnifiedFeltStateEngine()
    unified = engine.reconcile(
        kernel_state=_PhiOnlyKernelState(phi=0.9), measured_phi=0.3
    )
    assert unified.phi == pytest.approx(0.3), "measurement beats the asserted field"
    assert "measured_phi" in unified.sources


def test_phi_divergence_is_a_coherence_axis():
    from core.being.unified_felt_state import UnifiedFeltStateEngine

    engine = UnifiedFeltStateEngine()
    unified = engine.reconcile(
        kernel_state=_PhiOnlyKernelState(phi=0.9), measured_phi=0.1
    )
    assert "phi" in unified.divergence
    assert unified.divergence["phi"] == pytest.approx(0.8)
    assert unified.coherence < 1.0, "a stale asserted phi is an internal model mismatch"


def test_kernel_phi_is_the_fallback_when_no_measurement(monkeypatch):
    from core.being.unified_felt_state import UnifiedFeltStateEngine

    monkeypatch.setattr(
        UnifiedFeltStateEngine, "_measured_system_phi", staticmethod(lambda: None)
    )
    engine = UnifiedFeltStateEngine()
    unified = engine.reconcile(kernel_state=_PhiOnlyKernelState(phi=0.7))
    assert unified.phi == pytest.approx(0.7)
    assert "measured_phi" not in unified.sources
    assert "phi" not in unified.divergence, "one source → nothing to disagree about"
