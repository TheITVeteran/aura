"""A diverged ODE step used to become zeros, silently.

``_commit_worker_state_transform`` is the single point every substrate write
funnels through, and its first act was ``np.nan_to_num(..., nan=0.0)`` on the
proposed state. That is coercion, not recovery: it does not detect divergence
and it does not restore anything. x[0..6] are valence, arousal, dominance,
frustration, curiosity, energy and focus, so a diverged step reset every
affective reading to the middle mid-conversation with nothing logged. An
unrecorded divergence is indistinguishable from a calm mind.

These tests hold the corrective mechanism in both directions: a diverged state
must be replaced by the last VERIFIED SOUND one and recorded, and a sound state
must pass through untouched. A layer that rewrites every state satisfies the
first and fails the second.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.substrate_recovery import (
    DivergenceRecovery,
    check_soundness,
    probe_divergence_recovery,
)


class TestSoundnessIsJudgedBeforeCoercion:
    def test_a_normal_state_is_sound(self):
        report = check_soundness(np.array([0.5, -0.5, 0.0, 1.0, -1.0]))
        assert report.sound is True
        assert report.reasons == ()

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ([np.nan, 0.1], "nan"),
            ([np.inf, 0.1], "inf"),
            ([-np.inf, 0.1], "inf"),
            ([2.5, 0.1], "out_of_bounds"),
        ],
    )
    def test_each_divergence_shape_is_named(self, state, expected):
        report = check_soundness(np.array(state, dtype=np.float64))
        assert report.sound is False
        assert expected in report.reasons

    def test_an_empty_state_is_not_silently_fine(self):
        assert check_soundness(np.array([])).sound is False

    def test_the_report_carries_the_diagnostic(self):
        report = check_soundness(np.array([np.nan, np.nan, np.inf, 0.2]))
        assert report.non_finite_count == 3
        assert "nan" in report.reasons and "inf" in report.reasons


class TestRecoveryRestoresHerActualPreviousState:
    def test_a_diverged_step_restores_the_last_sound_state(self):
        recovery = DivergenceRecovery()
        sound = np.array([0.4, -0.2, 0.9, 0.0])
        recovery.observe(sound)

        outcome = recovery.recover(np.array([np.nan, np.inf, 0.9, 0.0]))
        assert outcome.recovered is True
        assert outcome.state is not None
        assert np.all(np.isfinite(outcome.state))
        # Her real previous condition, not zeros.
        assert np.allclose(outcome.state, sound)

    def test_zeros_are_not_the_recovery(self):
        """The whole point: a mind that diverged is not a mind at neutral."""
        recovery = DivergenceRecovery()
        recovery.observe(np.array([0.8, -0.7, 0.6, 0.5]))
        outcome = recovery.recover(np.array([np.nan, np.nan, np.nan, np.nan]))
        assert outcome.state is not None
        assert not np.allclose(outcome.state, np.zeros(4))

    def test_a_sound_state_passes_through_untouched(self):
        recovery = DivergenceRecovery()
        sound = np.array([0.4, -0.2, 0.9, 0.0])
        recovery.observe(sound)
        outcome = recovery.recover(sound)
        assert outcome.recovered is False
        assert np.allclose(outcome.state, sound)

    def test_divergence_with_no_sound_checkpoint_says_so(self):
        """Inventing a state here is the failure this module exists to end."""
        recovery = DivergenceRecovery()
        outcome = recovery.recover(np.array([np.nan, np.nan]))
        assert outcome.recovered is False
        assert outcome.state is None
        assert outcome.reason == "no_checkpoint"

    def test_the_checkpoint_ring_holds_only_sound_states(self):
        recovery = DivergenceRecovery()
        recovery.observe(np.array([0.1, 0.1]))
        recovery.observe(np.array([np.nan, 0.1]))
        recovery.observe(np.array([0.3, 0.3]))
        assert recovery.checkpoint_count == 2
        outcome = recovery.recover(np.array([np.inf, np.inf]))
        assert np.allclose(outcome.state, np.array([0.3, 0.3]))


class TestRepeatedDivergenceChangesTheDynamics:
    def test_escalation_damps_the_integration(self):
        """Restoring into unchanged dynamics that just diverged buys one step."""
        recovery = DivergenceRecovery(escalate_after=3)
        recovery.observe(np.array([0.5, 0.5]))
        assert recovery.damping == pytest.approx(1.0)

        for _ in range(3):
            recovery.recover(np.array([np.nan, np.nan]))
        assert recovery.damping < 1.0
        assert recovery.escalations >= 1

    def test_damping_never_reaches_zero(self):
        """Damping to zero stops the mind rather than stabilising it."""
        recovery = DivergenceRecovery(escalate_after=1)
        recovery.observe(np.array([0.5, 0.5]))
        for _ in range(50):
            recovery.recover(np.array([np.nan, np.nan]))
        assert recovery.damping > 0.0

    def test_sound_steps_earn_the_damping_back(self):
        recovery = DivergenceRecovery(escalate_after=1)
        recovery.observe(np.array([0.5, 0.5]))
        recovery.recover(np.array([np.nan, np.nan]))
        damped = recovery.damping
        assert damped < 1.0
        for _ in range(20):
            recovery.observe(np.array([0.5, 0.5]))
        assert recovery.damping > damped

    def test_metrics_report_what_happened(self):
        recovery = DivergenceRecovery()
        recovery.observe(np.array([0.5, 0.5]))
        recovery.recover(np.array([np.nan, 0.5]))
        metrics = recovery.as_metrics()
        assert metrics["divergences"] == 1
        assert metrics["recoveries"] == 1
        assert "nan" in metrics["last_reasons"]


class TestTheSubstrateUsesIt:
    def test_the_commit_point_recovers_rather_than_zeroing(self):
        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=8))
        sound = np.linspace(0.1, 0.8, 8)
        substrate._divergence_recovery.observe(sound)

        diverged = np.full(8, np.nan)
        recovered = substrate._recover_diverged_state(diverged, source="test")
        assert np.all(np.isfinite(recovered))
        assert not np.allclose(recovered, np.zeros(8)), (
            "a diverged step became zeros again; that is the defect"
        )
        assert np.allclose(recovered, sound)

    def test_a_sound_proposal_is_not_altered(self):
        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=8))
        proposed = np.linspace(-0.5, 0.5, 8)
        out = substrate._recover_diverged_state(proposed, source="test")
        assert np.allclose(out, proposed)

    def test_the_substrate_exposes_its_recovery_record(self):
        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=8))
        metrics = substrate.substrate_recovery_metrics()
        assert set(metrics) >= {"divergences", "recoveries", "escalations", "damping"}


def test_the_eval_arena_recovery_probe_is_real():
    outcome = probe_divergence_recovery()
    assert outcome.measured is True
    assert outcome.passed is True
    assert outcome.evidence["restored"] is True
    assert outcome.evidence["sound_state_untouched"] is True
