"""Tests for non-parametric generation helpers (normalization + cosine gating).

The full generate_with_memory loop is validated end-to-end against the real model in
aura_bench/nonparametric_probe.py (generation 5/5). These cover the pure helpers that
make the cosine-gated λ model-independent.
"""
from __future__ import annotations

import numpy as np

from core.brain.nonparametric_generation import cosine_from_l2, normalize


def test_normalize_unit_norm():
    v = normalize(np.array([3.0, 4.0, 0.0, 0.0]))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_normalize_zero_vector_safe():
    v = normalize(np.zeros(4))
    assert np.all(np.isfinite(v))


def test_cosine_from_l2_identical_is_one():
    # distance 0 between unit vectors → cosine 1
    assert abs(cosine_from_l2(0.0) - 1.0) < 1e-9


def test_cosine_from_l2_orthogonal_is_zero():
    # two orthogonal unit vectors are sqrt(2) apart → cosine 0
    assert abs(cosine_from_l2(np.sqrt(2.0))) < 1e-6


def test_cosine_from_l2_opposite_is_minus_one():
    # antipodal unit vectors are distance 2 apart → cosine -1
    assert abs(cosine_from_l2(2.0) - (-1.0)) < 1e-9


def test_cosine_monotonic_in_distance():
    assert cosine_from_l2(0.1) > cosine_from_l2(0.5) > cosine_from_l2(1.0)
