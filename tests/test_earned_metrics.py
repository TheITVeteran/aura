"""Contract tests for earned names.

A quantity keeps a name only while it predicts the thing the name refers to,
on data it was not fitted to, better than chance reproduces. These tests pin
each half of that: a direction that genuinely tracks its target validates, and
every way of not tracking it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.qualia_engine import QualiaEngine
from core.verify.earned_metric import EarnedAxis, recurrence_verdict


def _axis(name: str = "valence", **overrides) -> EarnedAxis:
    kwargs = dict(
        min_samples=64,
        holdout_fraction=0.3,
        min_holdout_r=0.35,
        max_p=0.05,
        ridge_penalty=1.0,
        permutations=100,
        capacity=2048,
    )
    kwargs.update(overrides)
    return EarnedAxis(name, **kwargs)


# ---------------------------------------------------------------------------
# EarnedAxis
# ---------------------------------------------------------------------------


def test_an_axis_reports_nothing_before_it_is_validated():
    axis = _axis()
    assert axis.value([1.0, 2.0, 3.0]) is None
    assert not axis.validated
    assert "no observations" in axis.last_fit.reason


def test_an_axis_that_predicts_its_target_earns_the_name():
    axis = _axis()
    rng = np.random.default_rng(11)
    # The target genuinely lives along one direction, with noise on top.
    for _ in range(200):
        state = rng.normal(size=6)
        target = 0.8 * state[3] - 0.4 * state[1] + rng.normal(scale=0.25)
        axis.observe(state, target)

    fit = axis.fit()
    assert fit.validated, fit.reason
    assert abs(fit.holdout_r) >= 0.35
    assert fit.permutation_p <= 0.05
    assert axis.value(rng.normal(size=6)) is not None


def test_an_axis_fitted_to_noise_never_earns_the_name():
    """The whole point: an arbitrary direction stays unnamed."""

    axis = _axis()
    rng = np.random.default_rng(12)
    for _ in range(200):
        axis.observe(rng.normal(size=6), rng.normal())

    fit = axis.fit()
    assert not fit.validated
    assert axis.value(rng.normal(size=6)) is None, (
        "an unvalidated axis must report nothing, not a plausible-looking number"
    )


def test_too_few_observations_cannot_validate():
    axis = _axis(min_samples=64)
    rng = np.random.default_rng(13)
    for _ in range(20):
        state = rng.normal(size=4)
        axis.observe(state, float(state[0]))

    fit = axis.fit()
    assert not fit.validated
    assert "64 needed" in fit.reason


def test_validation_uses_a_chronological_split_not_a_random_one():
    """A random split on an autocorrelated trajectory leaks the answer.

    Here the target is pure drift: it is perfectly predictable from position in
    time but has no stable relationship to the state. A random split would put
    a near-neighbour of every holdout point in training and validate. A
    chronological split puts the entire holdout beyond anything seen, where the
    extrapolation fails — which is the honest verdict.
    """

    axis = _axis()
    rng = np.random.default_rng(14)
    for step in range(200):
        # State is noise; the target ramps with time. Nothing in the state
        # carries the ramp, so no direction should be able to predict it.
        axis.observe(rng.normal(size=5), float(step) * 0.05)

    assert not axis.fit().validated


def test_a_constant_target_does_not_validate_on_a_division_artefact():
    axis = _axis()
    rng = np.random.default_rng(15)
    for _ in range(200):
        axis.observe(rng.normal(size=4), 0.5)
    assert not axis.fit().validated


def test_a_resized_substrate_discards_incompatible_history():
    axis = _axis()
    rng = np.random.default_rng(16)
    for _ in range(100):
        axis.observe(rng.normal(size=4), rng.normal())
    axis.observe(rng.normal(size=8), 0.1)
    # Old observations described a different space; mixing them would fit a
    # direction that never existed.
    assert axis.snapshot()["observations"] == 1


def test_non_finite_observations_are_refused():
    axis = _axis()
    axis.observe([float("nan"), 1.0], 0.5)
    axis.observe([1.0, 2.0], float("inf"))
    assert axis.snapshot()["observations"] == 0


def test_validation_is_lost_again_when_the_relationship_breaks():
    """The name is held on a lease, not granted permanently."""

    axis = _axis(capacity=200)
    rng = np.random.default_rng(17)
    for _ in range(200):
        state = rng.normal(size=5)
        axis.observe(state, 0.9 * state[2] + rng.normal(scale=0.1))
    assert axis.fit().validated

    for _ in range(200):
        axis.observe(rng.normal(size=5), rng.normal())
    assert not axis.fit().validated
    assert axis.value(rng.normal(size=5)) is None


# ---------------------------------------------------------------------------
# Surrogate threshold
# ---------------------------------------------------------------------------


def _noise(rng, n=20, width=32):
    return [rng.normal(size=width) for _ in range(n)]


def _cycle(rng, period=3, n=20, width=32):
    anchors = [rng.normal(size=width) for _ in range(period)]
    return [anchors[i % period] + rng.normal(scale=0.02, size=width) for i in range(n)]


def test_recurrence_needs_enough_history():
    assert recurrence_verdict([[1.0, 0.0]], percentile=95.0, surrogates=8) is None


def test_recurrence_threshold_is_derived_from_the_trajectory():
    """Two trajectories with different geometry must not get the same bar.

    That is the failure mode of the 0.85 constant: it describes no substrate in
    particular.
    """

    rng = np.random.default_rng(18)
    tight = recurrence_verdict(
        [rng.normal(scale=0.01, size=16) + 5.0 for _ in range(20)],
        percentile=95.0,
        surrogates=64,
    )
    scattered = recurrence_verdict(_noise(rng), percentile=95.0, surrogates=64)

    assert tight is not None and scattered is not None
    assert tight.threshold > scattered.threshold, (
        "a trajectory whose states are all near-identical must demand a higher "
        "bar for recurrence than one that wanders"
    )


def test_recurrence_rejects_a_degenerate_trajectory():
    assert (
        recurrence_verdict(
            [[0.0, 0.0] for _ in range(10)], percentile=95.0, surrogates=8
        )
        is None
    )


def test_recurrence_separates_a_cycle_from_noise():
    """The property the first attempt at this did not have.

    Shuffling the states and asking how similar each is to its predecessors
    leaves a 3-cycle looking exactly like its own shuffle — every state still
    has near-twins somewhere, whatever the order. Measured that way: 0.139 on
    noise against 0.163 on a clean cycle, which is no detector at all. The
    statistic has to be lag-resolved, because *when* similar states occur is
    what a shuffle actually destroys.
    """

    detected = 0
    false_positives = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        detected += int(
            recurrence_verdict(
                _cycle(rng), percentile=95.0, surrogates=128, seed=seed
            ).recurrent
        )
        rng = np.random.default_rng(seed)
        false_positives += int(
            recurrence_verdict(
                _noise(rng), percentile=95.0, surrogates=128, seed=seed
            ).recurrent
        )

    assert detected == 60, f"a clean 3-cycle must always register: {detected}/60"
    assert false_positives <= 9, (
        f"noise must clear a 95th-percentile bar about 5% of the time, "
        f"not {false_positives}/60"
    )


def test_recurrence_finds_the_period():
    rng = np.random.default_rng(31)
    verdict = recurrence_verdict(
        _cycle(rng, period=4), percentile=95.0, surrogates=64, seed=31
    )
    assert verdict is not None
    assert verdict.dominant_lag == 4, (
        f"a 4-cycle's strongest lag should be 4, got {verdict.dominant_lag}"
    )


# ---------------------------------------------------------------------------
# QualiaEngine v3
# ---------------------------------------------------------------------------


def _run(engine: QualiaEngine, state, *, affect_target=None, ignited=True):
    return engine.process(
        state=np.asarray(state, dtype=float),
        velocity=np.zeros_like(np.asarray(state, dtype=float)),
        predictive_metrics={"current_surprise": 0.3, "free_energy": 0.2, "precision": 0.8},
        workspace_snapshot={"ignited": ignited, "ignition_level": 0.7},
        phi=0.3,
        affect_target=affect_target,
    )


def test_qualia_does_not_call_a_dimension_valence_without_evidence():
    engine = QualiaEngine()
    rng = np.random.default_rng(19)
    descriptor = _run(engine, rng.normal(size=8))

    assert "valence" not in descriptor.conceptual
    assert "state_dim_0" in descriptor.conceptual
    assert descriptor.axis_fits["valence"]["validated"] is False


def test_qualia_earns_the_valence_name_when_the_substrate_predicts_affect():
    engine = QualiaEngine()
    rng = np.random.default_rng(20)
    for _ in range(200):
        state = rng.normal(size=8)
        _run(
            engine,
            state,
            affect_target={
                "valence": 0.9 * state[5] + rng.normal(scale=0.1),
                "arousal": rng.normal(),  # unrelated: must NOT validate
            },
        )

    fits = engine.layer_2.axis_fits()
    assert fits["valence"]["validated"], fits["valence"]["reason"]
    assert not fits["arousal"]["validated"], (
        "an axis fed an unrelated target must not earn its name alongside one that did"
    )
    assert "valence" in engine.validated_axes()
    assert "arousal" not in engine.validated_axes()

    descriptor = _run(engine, rng.normal(size=8))
    assert "valence" in descriptor.conceptual
    assert "state_dim_1" in descriptor.conceptual, "arousal stays under its neutral name"


def test_recurrence_threshold_is_measured_not_assumed():
    engine = QualiaEngine()
    rng = np.random.default_rng(21)
    for _ in range(12):
        _run(engine, rng.normal(size=16))

    descriptor = engine.get_last_descriptor()
    assert descriptor is not None
    assert descriptor.self_reference_threshold is not None
    assert descriptor.self_reference_threshold != 0.85, (
        "the constant this replaced must not reappear as a coincidence"
    )
    assert descriptor.recurrence["surrogates"] > 0
    assert "statistic" in descriptor.recurrence


def test_the_engine_discriminates_recurrence_from_drift():
    """End to end through the live layer, not just the statistic in isolation."""

    def rate(build):
        fires = total = 0
        for seed in range(20):
            rng = np.random.default_rng(seed)
            engine = QualiaEngine()
            for index, state in enumerate(build(rng)):
                fired = _run(engine, state).self_referential
                if index >= 10:
                    total += 1
                    fires += int(fired)
        return fires / total

    noise_rate = rate(lambda rng: [rng.normal(size=32) for _ in range(30)])
    cycle_rate = rate(
        lambda rng: (
            lambda anchors: [
                anchors[i % 3] + rng.normal(scale=0.02, size=32) for i in range(30)
            ]
        )([rng.normal(size=32) for _ in range(3)])
    )

    assert cycle_rate > 0.95, f"a 3-cycle must register: {cycle_rate:.3f}"
    assert noise_rate < 0.15, (
        f"drift must stay near the nominal 5% bar, not {noise_rate:.3f}"
    )


def test_richness_never_claims_to_be_validated():
    engine = QualiaEngine()
    rng = np.random.default_rng(23)
    descriptor = _run(engine, rng.normal(size=8))

    assert descriptor.richness_validated is False
    assert descriptor.richness_basis == "hand_chosen_weights_unfitted"
    assert descriptor.feature_aggregate == descriptor.phenomenal_richness
    assert 0.0 <= descriptor.feature_aggregate <= 1.0


def test_descriptor_dict_carries_the_evidence_with_the_numbers():
    engine = QualiaEngine()
    payload = _run(engine, np.random.default_rng(24).normal(size=8)).to_dict()

    for key in (
        "feature_aggregate",
        "richness_validated",
        "richness_basis",
        "self_reference_threshold",
        "axis_fits",
    ):
        assert key in payload, f"{key} must travel with the numbers it qualifies"


def test_engine_snapshot_reports_which_axes_are_earned():
    engine = QualiaEngine()
    snapshot = engine.get_snapshot()
    assert snapshot["validated_axes"] == []
    assert snapshot["richness_validated"] is False


@pytest.mark.parametrize("width", [1, 2, 3, 8, 64])
def test_engine_survives_any_substrate_width(width):
    engine = QualiaEngine()
    descriptor = _run(engine, np.linspace(-1.0, 1.0, width))
    assert descriptor.conceptual
    assert 0.0 <= descriptor.feature_aggregate <= 1.0
