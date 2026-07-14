"""
Tests for IIT 4.0 Exclusion Postulate implementation in phi_core.py.
Validates compute_max_phi_complex() and related methods.
"""

import time

import numpy as np
import pytest


def _make_phi_core_with_history(n_steps=200):
    """Create a PhiCore with enough state history to compute phi."""
    from core.consciousness.phi_core import N_NODES, PhiCore

    pc = PhiCore()
    rng = np.random.RandomState(42)

    # Generate correlated state transitions to produce non-trivial phi
    x = rng.randn(N_NODES).astype(np.float32) * 0.5
    for _ in range(n_steps):
        # Correlated dynamics: each node depends on its neighbors
        noise = rng.randn(N_NODES).astype(np.float32) * 0.2
        x = 0.7 * x + 0.3 * np.roll(x, 1) + noise
        # Record as substrate state (needs at least 8 elements)
        state = np.zeros(16, dtype=np.float32)
        state[:N_NODES] = x
        pc.record_state(state)

    return pc


def test_phi_core_init_exclusion_fields():
    """PhiCore has exclusion postulate tracking fields."""
    from core.consciousness.phi_core import PhiCore

    pc = PhiCore()
    assert pc._max_phi_complex is None
    assert pc._max_phi_value == 0.0
    assert pc._max_phi_complex_names == []


def test_compute_max_phi_complex_returns_result():
    """compute_max_phi_complex returns a (subset, phi) tuple."""
    pc = _make_phi_core_with_history(200)
    result = pc.compute_max_phi_complex()

    assert result is not None, "Should return a result with enough history"
    subset, phi_val = result
    assert isinstance(subset, tuple)
    assert len(subset) >= 2, "Max phi complex must have at least 2 nodes"
    assert phi_val >= 0.0, "Phi must be non-negative"


def test_compute_max_phi_complex_stores_state():
    """After computation, internal state is updated."""
    pc = _make_phi_core_with_history(200)
    pc.compute_max_phi_complex()

    assert pc._max_phi_complex is not None
    assert pc._max_phi_value >= 0.0
    assert len(pc._max_phi_complex_names) == len(pc._max_phi_complex)


def test_max_phi_subset_nodes_valid():
    """The returned subset contains valid node indices."""
    from core.consciousness.phi_core import N_NODES

    pc = _make_phi_core_with_history(200)
    subset, _ = pc.compute_max_phi_complex()

    for idx in subset:
        assert 0 <= idx < N_NODES, f"Invalid node index {idx}"


def test_compute_phi_for_subset_2nodes():
    """_compute_phi_for_subset works for a 2-node subset."""
    pc = _make_phi_core_with_history(200)
    tpm = pc.build_tpm()
    p = pc._get_stationary_distribution()

    phi = pc._compute_phi_for_subset(tpm, p, (0, 1))
    assert phi >= 0.0


def test_compute_phi_for_subset_full():
    """Phi for the full 8-node subset should match compute_phi result closely."""
    from core.consciousness.phi_core import N_NODES

    pc = _make_phi_core_with_history(200)
    full_result = pc.compute_phi()
    assert full_result is not None

    tpm = pc.build_tpm()
    p = pc._get_stationary_distribution()
    full_subset = tuple(range(N_NODES))
    subset_phi = pc._compute_phi_for_subset(tpm, p, full_subset)

    # Should be close to the full phi_s (same computation, different code path)
    assert abs(subset_phi - full_result.phi_s) < 0.01, (
        f"Full subset phi ({subset_phi:.5f}) should match compute_phi ({full_result.phi_s:.5f})"
    )


def test_max_phi_geq_full_phi():
    """The max-phi complex must have phi >= the full system's phi."""
    pc = _make_phi_core_with_history(200)
    full_result = pc.compute_phi()
    assert full_result is not None

    exclusion_result = pc.compute_max_phi_complex()
    assert exclusion_result is not None
    _, max_phi = exclusion_result

    assert max_phi >= full_result.phi_s - 1e-6, (
        f"Max phi ({max_phi:.5f}) must be >= full system phi ({full_result.phi_s:.5f})"
    )


def test_get_status_includes_exclusion():
    """get_status includes exclusion postulate fields after computation."""
    pc = _make_phi_core_with_history(200)
    pc.compute_phi()  # This triggers compute_max_phi_complex internally

    status = pc.get_status()
    assert "not a consciousness meter" in status["claim_boundary"]
    assert "exclusion_max_phi" in status
    assert "exclusion_complex_nodes" in status
    assert "exclusion_complex_names" in status
    assert "exclusion_is_full_system" in status
    assert "exclusion_complex_size" in status


def test_get_phi_statement_includes_exclusion():
    """get_phi_statement mentions exclusion postulate after computation."""
    pc = _make_phi_core_with_history(200)
    pc.compute_phi()

    statement = pc.get_phi_statement()
    assert "EXCLUSION" in statement
    assert "CONSCIOUS COMPLEX" not in statement
    assert "not a consciousness proof" in statement


def test_exclusion_not_in_status_before_compute():
    """Before any computation, exclusion fields are absent from status."""
    from core.consciousness.phi_core import PhiCore

    pc = PhiCore()
    status = pc.get_status()
    assert "exclusion_max_phi" not in status


def test_compute_max_phi_caching():
    """Repeated calls within interval return cached result."""
    pc = _make_phi_core_with_history(200)
    result1 = pc.compute_max_phi_complex()
    assert result1 is not None

    # Second call should hit cache (interval not elapsed)
    result2 = pc.compute_max_phi_complex()
    assert result2 is not None
    assert result1[0] == result2[0]
    assert result1[1] == result2[1]


def test_insufficient_history_returns_none():
    """compute_max_phi_complex returns None with insufficient history."""
    from core.consciousness.phi_core import PhiCore

    pc = PhiCore()
    # Only record 10 states (below MIN_HISTORY_FOR_TPM)
    for i in range(10):
        state = np.ones(16, dtype=np.float32) * i * 0.1
        pc.record_state(state)

    result = pc.compute_max_phi_complex()
    assert result is None


def test_phi_for_subset_bipartition_basic():
    """_phi_for_subset_bipartition produces non-negative results."""
    pc = _make_phi_core_with_history(200)
    tpm = pc.build_tpm()
    p = pc._get_stationary_distribution()

    # Build a 3-node subset TPM
    subset = (0, 1, 2)
    # We'll call the internal method via _compute_phi_for_subset
    phi = pc._compute_phi_for_subset(tpm, p, subset)
    assert phi >= 0.0


def test_vectorized_affective_phi_preserves_value_and_runtime_budget():
    """Exact 8-node validation must not starve the live event loop for seconds."""
    pc = _make_phi_core_with_history(200)

    started = time.perf_counter()
    result = pc.compute_affective_phi()
    elapsed = time.perf_counter() - started

    assert result is not None
    assert result.phi_s == pytest.approx(0.2355138393131479, abs=1e-5)
    assert result.mip_partition_a == [0]
    assert result.mip_partition_b == [1, 2, 3, 4, 5, 6, 7]
    assert elapsed < 1.5, f"exact affective phi took {elapsed:.3f}s"


def test_vectorized_partition_math_matches_scalar_reference():
    from core.consciousness.phi_core import PhiCore

    rng = np.random.default_rng(7)
    tpm = rng.random((8, 8))
    tpm /= tpm.sum(axis=1, keepdims=True)
    stationary = rng.random(8)
    stationary /= stationary.sum()
    extract_a = np.array([state & 0b11 for state in range(8)], dtype=np.int32)
    extract_b = np.array([(state >> 2) & 1 for state in range(8)], dtype=np.int32)

    tpm_a = PhiCore._marginal_tpm_from_extract(
        tpm,
        stationary,
        extract_target=extract_a,
        extract_other=extract_b,
        n_target_states=4,
        n_other_states=2,
    )
    tpm_b = PhiCore._marginal_tpm_from_extract(
        tpm,
        stationary,
        extract_target=extract_b,
        extract_other=extract_a,
        n_target_states=2,
        n_other_states=4,
    )

    scalar_phi = 0.0
    for state in range(8):
        cut = np.array(
            [
                tpm_a[extract_a[state], extract_a[next_state]]
                * tpm_b[extract_b[state], extract_b[next_state]]
                for next_state in range(8)
            ],
            dtype=np.float64,
        )
        cut /= cut.sum()
        kl = np.sum(tpm[state] * np.log(tpm[state] / (cut + 1e-10)))
        scalar_phi += stationary[state] * max(0.0, float(kl))

    vector_phi = PhiCore._partition_kl_phi(
        tpm,
        stationary,
        tpm_a,
        tpm_b,
        extract_a,
        extract_b,
    )
    assert vector_phi == pytest.approx(scalar_phi, abs=1e-12)
