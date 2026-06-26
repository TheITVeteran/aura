"""Proves the phenomenal substrate is *causally* live, not stored-and-ignored.

Two gaps the critique flagged:

  1. The live heartbeat uses ``PhenomenalIntegrator._pulse_blocking``, which computed the
     phenomenal state but never routed it to planner/memory/attention/self-model — so the
     rich downstream coupling was dead on the traced runtime path.
  2. ``SelfAwareness`` calls ``set_agency/set_embodiment/set_continuity/set_presence`` on the
     ``PhenomenalEngine``, but those methods did not exist → a no-op bridge.

These tests lock both as causal: routing fires from the blocking path, and the self-model's
signals actually move the phenomenal inference (not just get stored).
"""
from __future__ import annotations

from core.affect.phenomenal_integration import PhenomenalIntegrator
from core.phenomenal_substrate.experience_engine import PhenomenalEngine
from core.phenomenal_substrate.types import RuntimeBody, Event


# ── Fix A: the blocking heartbeat path routes downstream ─────────────────────


class _RecordingOrchestrator:
    """Minimal orchestrator exposing the four downstream sinks the integrator routes to."""

    def __init__(self) -> None:
        self.affect_consumed = []
        self.write_weights = None
        self.routed_broadcast = None
        self.presence = None

        outer = self

        class _Planner:
            def consume_affect(self, state):
                outer.affect_consumed.append(state)

        class _Memory:
            def set_write_weights(self, weights):
                outer.write_weights = weights

        class _Attention:
            def route(self, broadcast):
                outer.routed_broadcast = broadcast

        class _SelfModel:
            def update_presence(self, *, self_presence, mineness, integration):
                outer.presence = (self_presence, mineness, integration)

        self.planner = _Planner()
        self.memory = _Memory()
        self.attention = _Attention()
        self.self_model = _SelfModel()


def test_blocking_pulse_routes_to_all_downstream_systems():
    integrator = PhenomenalIntegrator()
    orch = _RecordingOrchestrator()

    state = integrator._pulse_blocking(orchestrator=orch, event_label="heartbeat")

    assert state is not None, "blocking pulse should produce a state"
    # The whole point: the blocking heartbeat must drive every downstream sink.
    assert orch.affect_consumed, "planner.consume_affect was never called from the blocking path"
    assert orch.write_weights is not None, "memory.set_write_weights was never called"
    assert orch.routed_broadcast is not None, "attention.route was never called"
    assert orch.presence is not None, "self_model.update_presence was never called"
    # And it routed *this* state, not a stale one.
    assert orch.affect_consumed[-1] is state
    assert orch.write_weights == state.memory_weights


def test_blocking_pulse_survives_a_broken_downstream_sink():
    """One bad sink must not abort routing to the others (best-effort coupling)."""

    class _PartlyBrokenOrch(_RecordingOrchestrator):
        def __init__(self) -> None:
            super().__init__()

            class _BoomPlanner:
                def consume_affect(self, state):
                    error = RuntimeError("planner exploded")
                    raise error

            self.planner = _BoomPlanner()

    orch = _PartlyBrokenOrch()
    integrator = PhenomenalIntegrator()
    state = integrator._pulse_blocking(orchestrator=orch, event_label="heartbeat")

    assert state is not None
    # planner blew up, but memory/attention/self-model still got routed
    assert orch.write_weights is not None
    assert orch.routed_broadcast is not None
    assert orch.presence is not None


# ── Fix B: self-model signals actually modulate phenomenal inference ─────────


def _neutral_body() -> RuntimeBody:
    return RuntimeBody(
        energy=0.5,
        continuity=0.5,
        agency=0.2,
        safety=0.5,
        social_contact=0.5,
        novelty=0.2,
        uncertainty=0.6,
        compute_pressure=0.2,
        memory_pressure=0.2,
        error_pressure=0.2,
    )


def test_self_signals_are_stored_and_clamped():
    eng = PhenomenalEngine()
    eng.set_agency(2.0)        # out of range → clamps to 1.0
    eng.set_continuity(-1.0)   # clamps to 0.0
    eng.set_embodiment(0.7)
    eng.set_presence(0.9)
    assert eng._self_signals == {
        "agency": 1.0,
        "continuity": 0.0,
        "embodiment": 0.7,
        "presence": 0.9,
    }


def test_self_signals_modulate_the_body_before_inference():
    eng = PhenomenalEngine()
    body = _neutral_body()

    blended = eng._apply_self_signals(body)  # no signals yet → unchanged
    assert blended is body

    eng.set_agency(1.0)
    eng.set_continuity(1.0)
    eng.set_embodiment(1.0)
    eng.set_presence(1.0)

    blended = eng._apply_self_signals(body)
    # agency blends up from 0.2 toward 1.0
    assert blended.agency > body.agency
    # continuity blends up from 0.5 toward 1.0
    assert blended.continuity > body.continuity
    # embodiment raises grounded safety
    assert blended.safety > body.safety
    # presence reduces uncertainty (being-here resolves ambiguity)
    assert blended.uncertainty < body.uncertainty
    # original body is untouched (immutable blend)
    assert body.agency == 0.2


def test_self_signals_change_the_resulting_experience_state():
    """End-to-end: the same event yields a different ExperienceState once the
    self-model asserts high agency/presence — proof the bridge is causal."""
    event = Event(label="heartbeat", source="test")

    baseline_eng = PhenomenalEngine()
    baseline = baseline_eng.step(_neutral_body(), event)

    signalled_eng = PhenomenalEngine()
    signalled_eng.set_agency(1.0)
    signalled_eng.set_presence(1.0)
    signalled_eng.set_continuity(1.0)
    signalled = signalled_eng.step(_neutral_body(), event)

    # A strong sense of agency/presence/continuity must shift the phenomenal outcome.
    assert (
        signalled.policy_priors != baseline.policy_priors
        or signalled.self_presence != baseline.self_presence
        or signalled.integration != baseline.integration
    ), "self-model signals did not change the phenomenal state — bridge is not causal"
