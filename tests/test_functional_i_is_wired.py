"""The functional "I" reaches policy, or it is not a self-model.

FunctionalIAttractor computes continuity, coherence, integrity, identity
tension, agency readiness and first-person confidence.
ClosedLoopPolicyCoupler maps that vector onto temperature, top-p, max tokens,
planning depth, verification threshold, memory retrieval depth, tool risk
budget and model-size preference.

Neither had a production call path. Both were reachable only from
tools/ablation_runner.py and tools/agi/run_causal_agency_lesion.py — an
ablation harness measuring a controller that controls nothing in the live
system, which is a null result about an organ that was never installed.

The attractor states its own condition in its docstring: real only when
derived from live evidence, injected back into the causal vector, and
changing policy downstream. These tests hold all three legs, because a
"causal" self-model is exactly as causal as its worst-wired leg.
"""

from __future__ import annotations

import inspect

import pytest

from core.being.policy_coupler import ClosedLoopPolicy, ClosedLoopPolicyCoupler
from core.being.runtime import BeingRuntime, get_being_runtime
from core.being.self_model_attractor import FunctionalIAttractor


def test_being_runtime_constructs_both_controllers() -> None:
    runtime = get_being_runtime()
    assert isinstance(runtime.self_attractor, FunctionalIAttractor)
    assert isinstance(runtime.policy_coupler, ClosedLoopPolicyCoupler)
    assert runtime.policy_coupler.production_mode is True


def test_leg_one_derivation_happens_in_the_sample_path() -> None:
    """The attractor is updated where the evidence is, under the same lock."""

    source = inspect.getsource(BeingRuntime._refresh_causal_self_vector)
    assert "_refresh_functional_i" in source

    refresh = inspect.getsource(BeingRuntime._refresh_functional_i)
    assert "self.self_attractor.update" in refresh
    assert "self.policy_coupler.modulate" in refresh


def test_leg_two_feedback_reaches_the_causal_vector() -> None:
    """An attractor that reads the vector and never writes it is a monitor."""

    refresh = inspect.getsource(BeingRuntime._refresh_functional_i)
    assert "vector_contributions" in refresh
    assert "replace(" in refresh, (
        "signals are frozen; a mutation that skipped replace() would also skip "
        "rewriting provenance, leaving an attractor-derived value attributed "
        "to its original sensor"
    )


def test_leg_three_policy_reaches_generation() -> None:
    """The leg whose absence made the other two decorative."""

    from core.brain.cognitive_engine import (
        _apply_functional_i_constraint,
        _live_mind_generation_controls,
    )

    controls_source = inspect.getsource(_live_mind_generation_controls)
    assert "_apply_functional_i_constraint" in controls_source

    constraint_source = inspect.getsource(_apply_functional_i_constraint)
    assert "closed_loop_policy" in constraint_source


class _StubRuntime:
    def __init__(self, policy: ClosedLoopPolicy | None) -> None:
        self._policy = policy

    def closed_loop_policy(self) -> ClosedLoopPolicy | None:
        return self._policy


@pytest.fixture
def _patched_runtime(monkeypatch):
    def _install(policy: ClosedLoopPolicy | None) -> None:
        import core.being.runtime as being_runtime

        monkeypatch.setattr(being_runtime, "get_being_runtime", lambda: _StubRuntime(policy))

    return _install


def test_a_strained_self_lowers_temperature(_patched_runtime) -> None:
    from core.brain.cognitive_engine import _apply_functional_i_constraint

    _patched_runtime(ClosedLoopPolicy(temperature=0.31, top_p=0.74))
    temperature, top_p, loops = _apply_functional_i_constraint(0.66, 0.90, 1)
    assert temperature == pytest.approx(0.31)
    assert top_p == pytest.approx(0.74)
    assert loops == 1


def test_a_calm_self_never_raises_temperature(_patched_runtime) -> None:
    """Tighten only.

    A confident self-model buying MORE randomness is not what any term in the
    coupler measures, and it would turn the self-model into a licence.
    """

    from core.brain.cognitive_engine import _apply_functional_i_constraint

    _patched_runtime(ClosedLoopPolicy(temperature=1.10, top_p=0.99))
    temperature, top_p, _loops = _apply_functional_i_constraint(0.58, 0.88, 1)
    assert temperature == pytest.approx(0.58)
    assert top_p == pytest.approx(0.88)


def test_high_verification_pressure_buys_a_second_pass(_patched_runtime) -> None:
    from core.brain.cognitive_engine import _apply_functional_i_constraint

    _patched_runtime(ClosedLoopPolicy(verification_threshold=0.75))
    _t, _p, loops = _apply_functional_i_constraint(0.58, 0.88, 1)
    assert loops == 2


def test_no_policy_leaves_the_turn_untouched(_patched_runtime) -> None:
    """Absence of a self-model is not evidence of a calm one."""

    from core.brain.cognitive_engine import _apply_functional_i_constraint

    _patched_runtime(None)
    temperature, top_p, loops = _apply_functional_i_constraint(0.58, 0.88, 1)
    assert (temperature, top_p, loops) == (0.58, 0.88, 1)


def test_the_coupler_actually_responds_to_identity_tension() -> None:
    """Not the wiring — the controller itself, on its own terms.

    If a strained "I" and a settled one produced the same policy, the wiring
    above would be carrying a constant and the whole thing would still be
    decorative.
    """

    from core.being.causal_self_state import CausalSelfVector
    from core.being.self_model_attractor import SelfAttractorState

    vector = CausalSelfVector(signals={})
    coupler = ClosedLoopPolicyCoupler(production_mode=True)

    def _state(tension: float, integrity: float, readiness: float) -> SelfAttractorState:
        return SelfAttractorState(
            attractor_id="probe",
            updated_at=0.0,
            identity_name="Aura",
            continuity_hash="h",
            continuity=1.0 - tension,
            coherence=1.0 - tension,
            integrity=integrity,
            agency_readiness=readiness,
            identity_tension=tension,
            first_person_confidence=1.0 - tension,
            claim_policy="functional_i_claim_allowed",
            current_i_statement="",
        )

    settled = coupler.modulate(vector=vector, self_state=_state(0.05, 0.95, 0.95))
    strained = coupler.modulate(vector=vector, self_state=_state(0.85, 0.30, 0.30))

    assert strained.temperature < settled.temperature
    assert strained.verification_threshold > settled.verification_threshold
    assert strained.tool_risk_budget < settled.tool_risk_budget
    assert strained.planning_depth >= settled.planning_depth
