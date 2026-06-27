"""Tests for non-parametric memory — growable token-level capacity for a fixed model."""
from __future__ import annotations

import numpy as np
import pytest

from core.brain.nonparametric_memory import NonParametricMemory


@pytest.fixture
def mem(tmp_path):
    return NonParametricMemory(dim=4, path=tmp_path / "npm", base_lambda=0.4, max_lambda=0.7)


def test_add_and_len(mem):
    assert mem.add(np.array([1.0, 0, 0, 0]), token_id=7, token="seven")
    assert len(mem) == 1


def test_add_rejects_wrong_dim(mem):
    assert mem.add(np.array([1.0, 0, 0]), token_id=1) is False


def test_query_returns_nearest_first(mem):
    mem.add(np.array([1.0, 0, 0, 0]), 1, "a")
    mem.add(np.array([0.0, 1.0, 0, 0]), 2, "b")
    mem.add(np.array([0.9, 0.1, 0, 0]), 3, "c")
    nbrs = mem.query(np.array([1.0, 0, 0, 0]), k=2)
    assert nbrs[0].token_id in (1, 3)         # the two closest to [1,0,0,0]
    assert nbrs[0].distance <= nbrs[1].distance


def test_query_empty_store_returns_nothing(mem):
    assert mem.query(np.array([1.0, 0, 0, 0])) == []


def test_knn_probs_sum_to_one(mem):
    mem.add(np.array([1.0, 0, 0, 0]), 1, "a")
    mem.add(np.array([0.0, 1.0, 0, 0]), 2, "b")
    nbrs = mem.query(np.array([1.0, 0, 0, 0]), k=2)
    probs = mem.knn_probs(nbrs)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs[1] > probs[2]                # nearer neighbor gets more mass


def test_adaptive_lambda_higher_for_closer_neighbor(mem):
    mem.add(np.array([1.0, 0, 0, 0]), 1, "a")
    near = mem.query(np.array([1.0, 0, 0, 0]))      # distance ~0
    far_mem = NonParametricMemory(dim=4)
    far_mem.add(np.array([1.0, 0, 0, 0]), 1, "a")
    far = far_mem.query(np.array([0.0, 0.0, 0.0, 5.0]))  # far
    assert mem.adaptive_lambda(near) > far_mem.adaptive_lambda(far)


def test_adaptive_lambda_zero_without_neighbors(mem):
    assert mem.adaptive_lambda([]) == 0.0


def test_adaptive_lambda_higher_with_free_energy(mem):
    mem.add(np.array([1.0, 0, 0, 0]), 1, "a")
    nbrs = mem.query(np.array([1.0, 0, 0, 0]))
    low_fe = mem.adaptive_lambda(nbrs, free_energy=0.0)
    high_fe = mem.adaptive_lambda(nbrs, free_energy=1.0)
    assert high_fe > low_fe                    # a surprised model trusts memory more


def test_low_phi_zeroes_lambda(mem):
    mem.add(np.array([1.0, 0, 0, 0]), 1, "a")
    nbrs = mem.query(np.array([1.0, 0, 0, 0]))
    assert mem.adaptive_lambda(nbrs, phi=0.01) == 0.0   # fragmented cognition → no recall trust


def test_lambda_never_exceeds_max(mem):
    mem.add(np.array([1.0, 0, 0, 0]), 1, "a")
    nbrs = mem.query(np.array([1.0, 0, 0, 0]))
    assert mem.adaptive_lambda(nbrs, free_energy=1.0, phi=0.55) <= 0.7


def test_interpolate_blends_toward_memory(mem):
    # memory strongly recalls token 42; model favored token 7
    mem.add(np.array([1.0, 0, 0, 0]), 42, "answer", weight=1.0)
    lm = {7: 0.6, 42: 0.1, 9: 0.3}
    blended = mem.interpolate(lm, np.array([1.0, 0, 0, 0]), free_energy=1.0)
    assert blended[42] > lm[42]                 # memory pulled probability toward 42
    assert abs(sum(blended.values()) - 1.0) < 1e-6


def test_interpolate_fail_open_no_neighbors(mem):
    lm = {7: 0.6, 9: 0.4}
    assert mem.interpolate(lm, np.array([1.0, 0, 0, 0])) == lm   # empty store → unchanged


def test_interpolate_fail_open_zero_lambda(mem):
    mem.add(np.array([1.0, 0, 0, 0]), 42, "x")
    lm = {7: 0.6, 9: 0.4}
    # phi below DORMANT forces lambda 0 → model distribution unchanged
    assert mem.interpolate(lm, np.array([1.0, 0, 0, 0]), phi=0.0) == lm


def test_eviction_bounded(tmp_path):
    m = NonParametricMemory(dim=2, path=tmp_path / "b", max_entries=64)
    for i in range(300):
        m.add(np.array([float(i), 1.0]), token_id=i, weight=1.0)
    assert len(m) <= 64
    assert m.stats()["evicted"] > 0


def test_persist_round_trip(tmp_path):
    p = tmp_path / "store"
    m1 = NonParametricMemory(dim=3, path=p)
    m1.add(np.array([1.0, 2.0, 3.0]), token_id=5, token="five", weight=2.0)
    m1.persist()
    m2 = NonParametricMemory(dim=3, path=p)
    assert len(m2) == 1
    nbrs = m2.query(np.array([1.0, 2.0, 3.0]))
    assert nbrs[0].token_id == 5 and nbrs[0].token == "five"


def test_dim_mismatch_on_load_starts_fresh(tmp_path):
    p = tmp_path / "store"
    m1 = NonParametricMemory(dim=3, path=p)
    m1.add(np.array([1.0, 2.0, 3.0]), token_id=5)
    m1.persist()
    # a different base model → different hidden dim → must NOT mix vector spaces
    m2 = NonParametricMemory(dim=8, path=p)
    assert len(m2) == 0


def test_apply_to_logits_flag_off_is_identity(mem, monkeypatch):
    monkeypatch.delenv("AURA_NONPARAMETRIC_MEMORY", raising=False)
    logits = np.array([0.1, 0.2, 5.0, 0.3], dtype=np.float32)
    out = mem.apply_to_logits(logits, np.array([1.0, 0, 0, 0]))
    assert np.allclose(out, logits)            # default-off: never touches generation


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def test_apply_to_logits_flag_on_reweights(mem, monkeypatch):
    monkeypatch.setenv("AURA_NONPARAMETRIC_MEMORY", "1")
    mem.add(np.array([1.0, 0, 0, 0]), token_id=0, token="x", weight=1.0)  # recall favors token 0
    logits = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32)            # model favors token 2
    out = mem.apply_to_logits(logits, np.array([1.0, 0, 0, 0]), free_energy=1.0)
    # compare probabilities (out is log-probs, logits are raw) — token 0's share must rise
    assert _softmax(out)[0] > _softmax(logits)[0]
