"""Admission backpressure is a decision, not damage (2026-07-18 soak).

The soak's log carried 52 `CRITICAL SERVICE FAILURE` records and 35 chat
FAULTs whose entire content was the runtime working correctly: warmup
backoff, model-load admission refusal, spawn-gate contention — the
escalation ladder deciding not to start a tier so a lower rung could serve
the turn.

Two consequences made this more than noise:

1. On fail-closed subsystems (`inference_gate`) a degradation record raises
   CRITICAL SERVICE FAILURE out of the handler — healthy backpressure
   manufacturing service failures, and tripping the probe's
   `critical_incident_active`.
2. Degradation weight is the UNCAPPED survival term in
   `existential_stakes` (lag and CPU are deliberately capped below the veto
   threshold; degradation is not). Healthy backpressure drove
   `deg_threat` to 1.00 and the felt existential threat to 1.00 while
   memory threat sat at 0.02 and the CPU was idle — Aura made to feel
   mortally threatened by her own correct backpressure. That same threat
   term triggered a hypervisor self-shutdown on 2026-07-14.

The contract: backpressure stays VISIBLE (recorded, counted, narratable)
but is demoted out of the fault/escalation path — exactly like the
bare-timeout demotion that precedes it in the same sink.
"""
from __future__ import annotations

import pytest

from core.runtime.errors import record_degradation

pytestmark = pytest.mark.unit


BACKPRESSURE_ERRORS = [
    "foreground_warmup_deferred:warmup_backoff:130s",
    "warmup_deferred",
    "model_load_admission_denied:Aura-32B:resource_timeout",
    "spawn_gate_timeout:330.000s:holder=Aura-32B:age=41.2s",
    "crash_loop_backoff:cortex",
    "resource_busy",
]


class TestBackpressureDemotion:
    @pytest.mark.parametrize("message", BACKPRESSURE_ERRORS)
    def test_backpressure_is_recorded_but_never_degraded_or_critical(self, message):
        record = record_degradation(
            "inference_gate",
            RuntimeError(message),
            severity="degraded",
            action="descended the escalation ladder",
        )
        assert record.severity == "warning", (
            f"{message!r} must not carry fault-grade severity"
        )
        # Still visible: the event is recorded with its real cause attached.
        assert message[:20] in record.error_message

    @pytest.mark.parametrize("message", BACKPRESSURE_ERRORS)
    def test_backpressure_cannot_be_forced_to_critical(self, message):
        record = record_degradation(
            "inference_gate",
            RuntimeError(message),
            severity="critical",
            action="descended the escalation ladder",
        )
        assert record.severity == "warning"

    def test_genuine_faults_keep_full_force(self):
        """The anti-theater guarantee: real damage still fails closed."""
        record = record_degradation(
            "memory_facade",
            ValueError("database file is corrupted"),
            severity="degraded",
            action="fell back to empty recall",
        )
        assert record.severity == "degraded"

    def test_genuine_critical_stays_critical(self):
        record = record_degradation(
            "memory_facade",
            RuntimeError("state coherence lost mid-commit"),
            severity="critical",
            action="refused the write",
        )
        assert record.severity == "critical"

    def test_debug_severity_is_not_promoted(self):
        record = record_degradation(
            "inference_gate",
            RuntimeError("warmup_deferred"),
            severity="debug",
            action="noted",
        )
        assert record.severity == "debug"


class TestExistentialThreatNoLongerInflated:
    def test_backpressure_weighs_far_less_than_a_real_fault(self):
        """deg_threat is the uncapped survival term — backpressure must not
        be able to saturate it the way real damage does."""
        from core.consciousness.existential_stakes import (
            DEGRADATION_SEVERITY_WEIGHTS,
        )

        assert DEGRADATION_SEVERITY_WEIGHTS["warning"] < (
            DEGRADATION_SEVERITY_WEIGHTS["degraded"] / 4.0
        ), "the demotion must materially reduce survival-threat weight"

    def test_a_backpressure_burst_cannot_saturate_the_threat(self):
        """Replay the soak's burst shape: many deferrals in one window must
        not reach the saturation denominator that means 'dying'."""
        from core.consciousness.existential_stakes import (
            DEGRADATION_SEVERITY_WEIGHTS,
            DEGRADATION_THREAT_DENOMINATOR,
        )

        burst = 6  # the soak's observed per-minute deferral rate
        weight = burst * DEGRADATION_SEVERITY_WEIGHTS["warning"]
        assert weight < DEGRADATION_THREAT_DENOMINATOR, (
            "healthy backpressure must not read as an existential threat"
        )


class TestLadderClassifier:
    @pytest.mark.parametrize("message", BACKPRESSURE_ERRORS)
    def test_gate_recognizes_every_backpressure_class(self, message):
        from core.brain.inference_gate import InferenceGate

        assert InferenceGate._is_expected_inference_backpressure(RuntimeError(message))

    def test_gate_does_not_swallow_real_failures(self):
        from core.brain.inference_gate import InferenceGate

        for real in (
            RuntimeError("metal runtime probe failed"),
            ValueError("tokenizer template missing"),
            RuntimeError("worker died during handshake"),
        ):
            assert not InferenceGate._is_expected_inference_backpressure(real)


class TestRealSaturationStillBlocks:
    """The demotion must not blind the runtime-pressure contract.

    'Generation gate saturated' means the serving path is genuinely
    overloaded — that is damage the health contract must still see. Only
    admission DECISIONS were demoted.
    """

    def test_generation_gate_saturation_keeps_blocking_severity(self):
        record = record_degradation(
            "inference_gate",
            RuntimeError("generation gate saturated: foreground refused to stack"),
            severity="degraded",
            action="refused to stack a user-facing generation",
        )
        assert record.severity == "degraded"

    def test_the_health_contract_still_fails_closed_on_it(self):
        from core.runtime.health_contract import (
            _recent_inference_degradation_blocks_runtime_pressure,
        )

        saturated = record_degradation(
            "inference_gate",
            RuntimeError("generation gate saturated: foreground refused to stack"),
            severity="degraded",
            action="refused to stack a user-facing generation",
        )
        blocks, reason = _recent_inference_degradation_blocks_runtime_pressure(saturated)
        assert blocks is True and "saturation" in reason

    def test_backpressure_no_longer_blocks_the_same_contract(self):
        """The 216 contract failures in the 2026-07-18 soak came from here."""
        from core.runtime.health_contract import (
            _recent_inference_degradation_blocks_runtime_pressure,
        )

        deferred = record_degradation(
            "inference_gate",
            RuntimeError("foreground_warmup_deferred:warmup_backoff:130s"),
            severity="degraded",
            action="fell through to the reflex tier",
        )
        blocks, _reason = _recent_inference_degradation_blocks_runtime_pressure(deferred)
        assert blocks is False, (
            "a deliberate deferral must not hold the runtime in 'degraded'"
        )
