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


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def test_logits_processor_boosts_recalled_token():
    import mlx.core as mx

    from core.brain.nonparametric_generation import make_nonparametric_logits_processor
    from core.brain.nonparametric_memory import NonParametricMemory

    mem = NonParametricMemory(dim=4)
    kvec = normalize(np.array([1.0, 0, 0, 0]))
    mem.add(kvec, token_id=0, token="x", weight=1.0)   # recall favors token 0

    class FakeModel:
        def model(self, seq):
            n = int(seq.shape[1])
            return mx.array(np.tile(kvec.astype(np.float32), (1, n, 1)))

    proc = make_nonparametric_logits_processor(FakeModel(), mem, free_energy=1.0)
    tokens = mx.array([1, 2, 3])
    logits = mx.array(np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32))  # model favors token 2
    out = np.array(proc(tokens, logits)).reshape(-1)
    # exact-key recall clears the gate → the recalled token (0) wins over the model's pick (2)
    assert int(np.argmax(out)) == 0


def test_logits_processor_fail_open_on_far_neighbor():
    import mlx.core as mx

    from core.brain.nonparametric_generation import make_nonparametric_logits_processor
    from core.brain.nonparametric_memory import NonParametricMemory

    mem = NonParametricMemory(dim=4)
    mem.add(normalize(np.array([0.0, 0.0, 0.0, 1.0])), token_id=0, token="x")  # orthogonal to query

    class FakeModel:
        def model(self, seq):
            n = int(seq.shape[1])
            q = normalize(np.array([1.0, 0, 0, 0]))
            return mx.array(np.tile(q.astype(np.float32), (1, n, 1)))

    proc = make_nonparametric_logits_processor(FakeModel(), mem)
    logits = mx.array(np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32))
    out = np.array(proc(mx.array([1, 2]), logits)).reshape(-1)
    # far neighbor (cos≈0 < min_cos) → λ gated to 0 → logits unchanged
    assert np.allclose(out, np.array([0.0, 0.0, 5.0, 0.0]))
