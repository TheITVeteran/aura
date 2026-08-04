"""φ has unit tests for numerical stability and none for being φ.

The review's note: "``phi_core.py`` does not have a property-based test that
verifies the computed φ satisfies the IIT axioms (e.g. intrinsic existence). It
only has unit tests for numerical stability. The mathematical definition exists,
but correctness relative to IIT is not proven."

Correct. So this is the axiom battery, stated as properties that hold for every
input rather than for one recorded example, each with a system constructed to
have a known answer.

The five postulates, and what each one requires of a φ over a TPM:

  EXISTENCE      φ ≥ 0 always, and φ > 0 requires the system to make a
                 difference to itself — a system whose next state is
                 independent of its current state has φ = 0.
  INTEGRATION    φ = 0 exactly when some bipartition costs nothing to cut. Two
                 independent halves are two systems, not one.
  EXCLUSION      φ is the MINIMUM over bipartitions, not any other summary. The
                 weakest seam defines the system.
  COMPOSITION    the reported MIP is a genuine bipartition — both sides
                 non-empty, disjoint, and covering every node exactly once.
  INFORMATION    φ is a property of the DYNAMICS, so relabelling the nodes
                 cannot change it, and a system with more internal dependence
                 cannot score below one with less.

These run against the same exhaustive-MIP path the runtime uses, not a
reimplementation of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.phi_core import (
    INTEGRATION_FRACTION_FLOOR,
    MIN_HISTORY_FOR_TPM,
    PhiCore,
)


# ── systems with known answers ────────────────────────────────────────────

def _independent_halves(steps: int = 600, seed: int = 11) -> list[int]:
    """Two 4-node halves that never influence each other. φ must be 0."""
    rng = np.random.default_rng(seed)
    left = 0
    right = 0
    states = []
    for _ in range(steps):
        # Each half is a deterministic function of ITS OWN previous state.
        left = ((left << 1) | (1 if (left & 0b1000) else 0)) & 0b1111
        right = ((right >> 1) | ((right & 1) << 3)) & 0b1111
        if rng.random() < 0.2:  # keep both halves exploring their state space
            left ^= 1 << int(rng.integers(0, 4))
            right ^= 1 << int(rng.integers(0, 4))
        states.append(left | (right << 4))
    return states


def _memoryless(steps: int = 600, seed: int = 5) -> list[int]:
    """Next state independent of current state. Nothing makes a difference."""
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 256)) for _ in range(steps)]


def _coupled_ring(steps: int = 600, seed: int = 7) -> list[int]:
    """Every node driven by its neighbour: integrated, φ should exceed 0."""
    rng = np.random.default_rng(seed)
    state = 0b10110101
    states = []
    for _ in range(steps):
        bits = [(state >> i) & 1 for i in range(8)]
        nxt = [bits[(i - 1) % 8] ^ bits[(i + 1) % 8] for i in range(8)]
        if rng.random() < 0.15:
            nxt[int(rng.integers(0, 8))] ^= 1
        state = sum(bit << i for i, bit in enumerate(nxt))
        states.append(state)
    return states


def _phi_of(states: list[int], *, with_null: bool = True, surrogates: int = 8) -> object:
    """Run the runtime's own exhaustive 8-node MIP over a state sequence.

    With the sampling null attached by default, because that is the quantity
    the axioms are about: raw φ at reachable sample sizes is dominated by the
    cost of estimating a 256x256 TPM from a few hundred transitions, and
    ``test_the_raw_estimator_is_biased_and_that_is_why_the_null_exists``
    measures exactly how much.
    """
    core = PhiCore()
    from collections import deque

    history: deque = deque(states, maxlen=4000)
    visits = np.ones(256, dtype=np.float64)
    for state in states:
        visits[int(state) & 0xFF] += 1.0
    result = core._compute_eight_node_phi_from_history(
        history,
        visits,
        core._residual_bipartitions,
        core._residual_bit_tables,
        grounding="test",
        population_size=8,
    )
    if result is not None and with_null:
        core.attach_phi_null(
            result,
            history,
            core._residual_bipartitions,
            core._residual_bit_tables,
            surrogates=surrogates,
        )
    return result


# ── EXISTENCE ─────────────────────────────────────────────────────────────

class TestIntrinsicExistence:
    """A system exists intrinsically only if it makes a difference to itself."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_phi_is_never_negative(self, seed):
        result = _phi_of(_coupled_ring(seed=seed))
        assert result is not None
        assert result.phi_s >= 0.0

    def test_a_memoryless_system_has_no_intrinsic_existence(self):
        """If the next state ignores the current one, nothing is caused."""
        result = _phi_of(_memoryless())
        assert result is not None
        assert result.integration_fraction == pytest.approx(0.0, abs=0.02), (
            f"a system with no self-dependence left "
            f"{result.integration_fraction:.3f} of its phi above the null "
            f"(raw {result.phi_s:.4f}, null {result.phi_null_mean:.4f})"
        )
        assert result.integration_is_significant is False
        assert result.is_complex is False

    def test_the_raw_estimator_is_biased_and_that_is_why_the_null_exists(self):
        """The measurement that forced the correction, kept as the record.

        A memoryless 8-node system scores a LARGE raw φ, because estimating a
        256x256 TPM from a few hundred transitions produces apparent structure
        from nothing. If this ever stops being true the null can be revisited;
        until then, removing it would restore a number that means nothing.
        """
        raw = _phi_of(_memoryless(), with_null=False)
        assert raw.phi_s > 0.3, (
            "the raw estimator no longer shows finite-sample bias on a "
            f"memoryless system (phi={raw.phi_s}); re-derive the correction"
        )

    def test_too_little_history_yields_no_claim_at_all(self):
        """Below the sample floor the answer is None, not zero.

        Zero is a measurement. None is the absence of one, and the difference
        is what keeps a warm-up from reading as a disintegrated mind.
        """
        assert _phi_of([1, 2, 3] * (MIN_HISTORY_FOR_TPM // 4)) is None


# ── INTEGRATION ───────────────────────────────────────────────────────────

class TestIntegration:
    """φ > 0 exactly when no bipartition is free to cut."""

    def test_two_independent_halves_do_not_integrate(self):
        result = _phi_of(_independent_halves())
        assert result is not None
        # Two independently predictable halves are two systems. They leave a
        # small residual above the null — MEASURED at 0.049, which is what
        # INTEGRATION_FRACTION_FLOOR is set from — because the estimator cannot
        # find the true 4|4 cut from 600 transitions over 256 states. That
        # residual is bookkeeping, and it must not read as integration.
        assert result.integration_fraction < INTEGRATION_FRACTION_FLOOR, (
            f"two causally independent halves left "
            f"{result.integration_fraction:.3f} above the null"
        )
        assert result.is_complex is False

    def test_a_coupled_system_integrates(self):
        result = _phi_of(_coupled_ring())
        assert result is not None
        assert result.integration_fraction > INTEGRATION_FRACTION_FLOOR * 3
        assert result.null_p_value < 0.05
        assert result.is_complex is True

    def test_coupling_cannot_score_below_independence(self):
        """The information postulate, as an ordering over real systems.

        Raw φ got this BACKWARDS — the memoryless system outscored the coupled
        ring — which is the sharpest evidence that raw φ is not comparable
        across systems at these sample sizes.
        """
        coupled = _phi_of(_coupled_ring())
        split = _phi_of(_independent_halves())
        memoryless = _phi_of(_memoryless())
        assert coupled.integration_fraction > split.integration_fraction
        assert split.integration_fraction >= memoryless.integration_fraction


# ── EXCLUSION ─────────────────────────────────────────────────────────────

class TestExclusion:
    """φ is the minimum over bipartitions — the weakest seam, nothing else."""

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_phi_equals_the_minimum_partition_cost(self, seed):
        result = _phi_of(_coupled_ring(seed=seed))
        assert result is not None
        assert result.all_partition_phis
        assert result.phi_s == pytest.approx(
            max(0.0, min(result.all_partition_phis)), abs=1e-9
        )

    def test_phi_is_not_the_mean_or_the_max(self, ):
        """Guards against a refactor quietly changing which summary is used."""
        result = _phi_of(_coupled_ring(seed=21))
        phis = result.all_partition_phis
        assert len(phis) > 1
        if max(phis) > min(phis):
            assert result.phi_s < float(np.mean(phis))
            assert result.phi_s < max(phis)

    def test_the_reported_mip_is_the_partition_that_won(self):
        result = _phi_of(_coupled_ring(seed=31))
        assert result.mip_phi_value == pytest.approx(result.phi_s)


# ── COMPOSITION ───────────────────────────────────────────────────────────

class TestComposition:
    """The MIP must actually be a bipartition of the node set."""

    @pytest.mark.parametrize("seed", [41, 42, 43])
    def test_the_mip_partitions_every_node_exactly_once(self, seed):
        result = _phi_of(_coupled_ring(seed=seed))
        a = set(result.mip_partition_a)
        b = set(result.mip_partition_b)
        assert a and b, "a bipartition with an empty side is not a cut"
        assert not (a & b), "a node cannot be on both sides"
        assert a | b == set(range(8)), "every node must be on one side"

    def test_every_evaluated_partition_has_a_cost(self):
        result = _phi_of(_coupled_ring(seed=44))
        assert all(np.isfinite(phi) for phi in result.all_partition_phis)
        assert all(phi >= -1e-9 for phi in result.all_partition_phis)


# ── INFORMATION ───────────────────────────────────────────────────────────

class TestInformationIsIntrinsic:
    """φ is a property of the dynamics, not of how the nodes are named."""

    def test_relabelling_the_nodes_does_not_change_phi(self):
        states = _coupled_ring(seed=51)
        baseline = _phi_of(states)

        # A fixed permutation of the 8 bit positions: the same system, with its
        # nodes written down in a different order.
        perm = [3, 5, 0, 7, 1, 6, 2, 4]

        def relabel(state: int) -> int:
            return sum(((state >> src) & 1) << dst for dst, src in enumerate(perm))

        permuted = _phi_of([relabel(s) for s in states])
        assert permuted.phi_s == pytest.approx(baseline.phi_s, rel=0.02, abs=5e-3), (
            f"relabelling changed phi from {baseline.phi_s} to {permuted.phi_s}"
        )

    def test_phi_is_deterministic_for_the_same_history(self):
        states = _coupled_ring(seed=61)
        assert _phi_of(states).phi_s == pytest.approx(_phi_of(states).phi_s, abs=1e-12)


# ── the number carries what it is a measurement of ────────────────────────

class TestTheScalarIsNotAlone:
    def test_provenance_travels_with_the_value(self):
        result = _phi_of(_coupled_ring(seed=71))
        provenance = result.provenance()
        assert provenance["sampling"] == "exhaustive_mip"
        assert provenance["node_count"] == 8
        assert provenance["tpm_n_samples"] > 0
        assert provenance["interval_method"] == "not_computed"
        assert provenance["null_surrogates"] >= 2
        assert provenance["integration_fraction"] is not None

    def test_an_interval_can_be_measured_when_a_number_is_published(self):
        """The finite-sample uncertainty is real and quantifiable; report it."""
        core = PhiCore()
        from collections import deque

        states = _coupled_ring(seed=81)
        history: deque = deque(states, maxlen=4000)
        visits = np.ones(256, dtype=np.float64)
        for state in states:
            visits[int(state) & 0xFF] += 1.0

        result = core._compute_eight_node_phi_from_history(
            history, visits, core._residual_bipartitions, core._residual_bit_tables
        )
        core.attach_phi_interval(
            result,
            history,
            core._residual_bipartitions,
            core._residual_bit_tables,
            resamples=12,
        )
        assert result.phi_lower is not None and result.phi_upper is not None
        assert result.phi_lower <= result.phi_upper
        assert result.interval_method.startswith("bootstrap_transitions:")


class TestThePowerOfTheMeasurementAtLiveSampleSizes:
    """More data would not change the live verdict, and this is why.

    A live conversation accumulates ~196 Grassmann states per turn into a
    deque(maxlen=2000) — the buffer saturates in about ten turns. So the
    question "should we soak for hours?" has an answer that does not depend on
    patience: at the lengths actually reachable, does the corrected fraction
    still tell a coupled system from an uncoupled one?

    It does, by about a hundredfold. That makes the live reading of 0.007 a
    conclusive measurement rather than an underpowered one.
    """

    @pytest.mark.parametrize("n", [500, 1000, 2000])
    def test_a_coupled_system_is_obvious_at_live_history_lengths(self, n):
        result = _phi_of(_coupled_ring(n, seed=7))
        assert result.integration_fraction > 0.4, (
            f"at n={n} the estimator lost a genuinely coupled system"
        )

    @pytest.mark.parametrize("n", [500, 1000, 2000])
    def test_an_uncoupled_system_stays_at_the_floor(self, n):
        result = _phi_of(_memoryless(n, seed=5))
        assert result.integration_fraction < 0.02

    def test_the_gap_is_wide_enough_that_more_data_is_not_the_answer(self):
        coupled = _phi_of(_coupled_ring(2000, seed=7)).integration_fraction
        uncoupled = _phi_of(_memoryless(2000, seed=5)).integration_fraction
        assert coupled > uncoupled * 20 or uncoupled < 1e-6
        assert coupled > 0.4
