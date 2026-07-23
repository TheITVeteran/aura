"""Tests for non-parametric memory — growable token-level capacity for a fixed model."""
from __future__ import annotations

import numpy as np
import pytest

from core.brain.nonparametric_memory import (
    NonParametricMemory,
    get_nonparametric_memory,
    reset_nonparametric_memory,
    validate_nonparametric_memory_identity,
)


@pytest.fixture
def mem(tmp_path):
    return NonParametricMemory(dim=4, path=tmp_path / "npm", base_lambda=0.4, max_lambda=0.7)


def test_add_and_len(mem):
    assert mem.add(np.array([1.0, 0, 0, 0]), token_id=7, token="seven")
    assert len(mem) == 1
    assert mem.stats()["allocated_bytes"] <= (64 * 4 * 4) + (64 * 4)


def test_identity_receipt_is_stable_cached_and_invalidated_by_content(mem):
    mem.add(np.array([1.0, 0, 0, 0]), token_id=7, token="seven")

    first, first_work = mem.identity_receipt_with_work()
    second, second_work = mem.identity_receipt_with_work()

    assert validate_nonparametric_memory_identity(first) == first
    assert second == first
    assert first_work == first["source_bytes"]
    assert second_work == 0

    mem.add(np.array([0.0, 1.0, 0, 0]), token_id=8, token="eight")
    changed, changed_work = mem.identity_receipt_with_work()
    assert changed["content_sha256"] != first["content_sha256"]
    assert changed["receipt_sha256"] != first["receipt_sha256"]
    assert changed_work == changed["source_bytes"]


def test_identity_receipt_rejects_rehashed_content_lie(mem):
    mem.add(np.array([1.0, 0, 0, 0]), token_id=7, token="seven")
    receipt = mem.identity_receipt()
    tampered = {**receipt, "content_sha256": "0" * 64}

    with pytest.raises(ValueError, match="identity is invalid"):
        validate_nonparametric_memory_identity(tampered)


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
    assert m1.persist() is True
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


def test_inconsistent_persistence_metadata_fails_closed(tmp_path):
    p = tmp_path / "corrupt"
    np.save(p.with_suffix(".keys.npy"), np.ones((2, 3), dtype=np.float32))
    p.with_suffix(".meta.json").write_text(
        '{"token_ids":[1],"tokens":["one"],"weights":[1.0],"ts":[1.0]}',
        encoding="utf-8",
    )

    loaded = NonParametricMemory(dim=3, path=p)

    assert len(loaded) == 0


def test_nonfinite_persistence_metadata_fails_before_identity_hash(tmp_path):
    p = tmp_path / "nonfinite"
    np.save(p.with_suffix(".keys.npy"), np.ones((1, 3), dtype=np.float32))
    p.with_suffix(".meta.json").write_text(
        '{"token_ids":[1],"tokens":["one"],"weights":[1.0],"ts":[NaN]}',
        encoding="utf-8",
    )

    loaded = NonParametricMemory(dim=3, path=p)

    assert len(loaded) == 0


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


def test_apply_to_logits_preserves_relative_mass_for_unrecalled_tokens(mem, monkeypatch):
    monkeypatch.setenv("AURA_NONPARAMETRIC_MEMORY", "1")
    mem.add(np.array([1.0, 0, 0, 0]), token_id=0, token="x", weight=1.0)
    logits = np.array([0.0, 1.0, 3.0, -0.5], dtype=np.float32)

    before = _softmax(logits)
    after = _softmax(
        mem.apply_to_logits(
            logits,
            np.array([1.0, 0, 0, 0]),
            free_energy=1.0,
        )
    )

    assert np.isclose(after[1] / after[2], before[1] / before[2], rtol=1e-5)


def test_singleton_refuses_cross_model_hidden_dimension():
    reset_nonparametric_memory()
    try:
        assert get_nonparametric_memory(4) is not None
        assert get_nonparametric_memory(8) is None
    finally:
        reset_nonparametric_memory()


# ── Anisotropy-corrected similarity (July proof-driven fixes) ─────────────


def _unit(v):
    import numpy as np

    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_similarity_raw_fallback_before_mu_ready(tmp_path):
    """Before the query mean converges, similarity is raw cosine and the
    gate is the strict 0.98 — exact re-encounters pass, everything else
    is blocked (measured: unrelated prompts score raw 0.81-0.93)."""

    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    key = _unit([1, 2, 3, 4, 5, 6, 7, 8])
    mem.add(key, 42)
    assert not mem.similarity_ready()
    assert mem.min_similarity() == mem.MIN_SIM_RAW

    exact = mem.query(key, k=1)[0]
    assert exact.similarity > 0.99

    related_but_not_exact = mem.query(_unit([1, 2, 3, 4, 5, 6, 7, 9]), k=1)[0]
    assert related_but_not_exact.similarity < mem.MIN_SIM_RAW or exact.similarity > related_but_not_exact.similarity


def test_similarity_centers_once_mu_ready(tmp_path):
    """After MU_READY_N query samples, similarity subtracts the common
    direction: two keys sharing a dominant component but differing in
    their distinctive parts must separate."""
    import numpy as np

    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    common = np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.float32)
    a = _unit(common + np.array([3, 0, 0, 0, 0, 0, 0, 0]))
    b = _unit(common + np.array([0, 3, 0, 0, 0, 0, 0, 0]))
    mem.add(a, 1)

    raw_cos = float(np.dot(a, b))
    assert raw_cos > 0.9, "the anisotropy setup: raw cosine is inflated"

    # Feed the mean with samples of the common direction.
    rng = np.random.default_rng(7)
    for _ in range(mem.MU_READY_N):
        noise = rng.normal(0, 0.5, size=8).astype(np.float32)
        mem.query(_unit(common + noise), k=1)
    assert mem.similarity_ready()
    assert mem.min_similarity() == mem.MIN_SIM_CENTERED

    centered = mem.query(b, k=1)[0]
    assert centered.similarity < 0.6, (
        f"centered similarity must strip the common direction (got {centered.similarity:.3f})"
    )
    same = mem.query(a, k=1)[0]
    assert same.similarity > 0.9, "identical keys stay similar after centering"


def test_interpolate_filters_below_gate_neighbors(tmp_path):
    """Below-gate entries must not leak probability mass (the cross-fact
    digit-leakage failure the July proof caught)."""

    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    target = _unit([1, 0, 0, 0, 0, 0, 0, 0.2])
    other = _unit([0, 1, 0, 0, 0, 0, 0, 0.2])
    mem.add(target, 111)
    mem.add(other, 222)

    blended = mem.interpolate({111: 0.01, 999: 0.99}, target, k=4, lam_override=0.9)
    assert blended[111] > blended.get(222, 0.0), (
        "the exact-match entry must dominate; the unrelated entry is filtered"
    )


def test_query_mu_persists_across_reload(tmp_path):
    import numpy as np

    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    mem.add(_unit([1, 1, 1, 1, 1, 1, 1, 1]), 5)
    for _ in range(20):
        mem.query(_unit(np.random.default_rng(3).normal(1, 0.1, 8)), k=1)
    assert mem.similarity_ready()
    assert mem.persist()

    reloaded = NonParametricMemory(dim=8, path=tmp_path / "npm")
    assert reloaded.similarity_ready(), "the anisotropy mean must survive restarts"


def test_neighbors_carry_store_index(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    a = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    b = _unit([0, 1, 0, 0, 0, 0, 0, 0])
    mem.add(a, 1)
    mem.add(b, 2)
    nearest = mem.query(a, k=1)[0]
    assert nearest.index == 0 and nearest.token_id == 1


class TestRecallPathsShareTheConfidenceGate:
    """apply_to_logits must not be more permissive than interpolate().

    interpolate() drops every neighbour below the active similarity gate.
    apply_to_logits fed the raw query result straight into knn_probs, so a
    single below-threshold neighbour still produced nonzero kNN mass and
    shifted the token distribution — the two recall paths enforced different
    standards, and the one wired to logits was the weaker one.
    """

    def _store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_NONPARAMETRIC_MEMORY", "1")
        from core.brain.nonparametric_memory import NonParametricMemory

        return NonParametricMemory(dim=8, path=str(tmp_path / "npm"))

    def test_below_gate_neighbour_cannot_shift_logits_via_lam_override(
        self, tmp_path, monkeypatch
    ):
        """The exact path that made this exploitable.

        adaptive_lambda independently returns ~0 for weak neighbours, which
        masked the missing gate in normal use. lam_override BYPASSES
        adaptive_lambda, so without the gate a below-threshold neighbour did
        reach knn_probs: measured against the pre-fix code, a neighbour at
        similarity 0.8896 (gate 0.98) shifted the logits by 3.47.
        """
        import numpy as np

        store = self._store(tmp_path, monkeypatch)
        # Similarity ~0.89 — comfortably below the 0.98 raw gate.
        store.add(
            np.array([1, 1, 1, 1, 1, 1, 1, -0.3], dtype=np.float32),
            token_id=5,
            token="y",
        )
        query = np.ones(8, dtype=np.float32)

        neighbours = store.query(query, k=8)
        assert neighbours, "fixture must produce a neighbour"
        assert neighbours[0].similarity < store.min_similarity(), (
            "fixture neighbour must be BELOW the gate for this test to mean anything"
        )

        out = np.asarray(
            store.apply_to_logits(
                np.zeros(16, dtype=np.float32), query, lam_override=0.5
            ),
            dtype=np.float64,
        )
        assert np.allclose(out, 0.0), (
            "a below-gate neighbour must not reach the recall distribution "
            "even when adaptive_lambda is bypassed"
        )

    def test_above_gate_neighbour_still_recalls(self, tmp_path, monkeypatch):
        """The gate must not disable legitimate recall."""
        import numpy as np

        store = self._store(tmp_path, monkeypatch)
        store.add(
            np.array([1, 1, 1, 1, 1, 1, 1, 0.55], dtype=np.float32),
            token_id=5,
            token="y",
        )
        query = np.ones(8, dtype=np.float32)

        neighbours = store.query(query, k=8)
        assert neighbours[0].similarity >= store.min_similarity()

        out = np.asarray(
            store.apply_to_logits(np.zeros(16, dtype=np.float32), query),
            dtype=np.float64,
        )
        assert not np.allclose(out, 0.0), "above-gate recall must still apply"

    def test_gate_is_the_same_function_both_paths_use(self, tmp_path, monkeypatch):
        store = self._store(tmp_path, monkeypatch)
        source = open("core/brain/nonparametric_memory.py", encoding="utf-8").read()
        body = source.split("def apply_to_logits", 1)[1][:2000]
        assert "self.min_similarity()" in body
        assert "nb.similarity >= min_sim" in body
        assert isinstance(store.min_similarity(), float)
