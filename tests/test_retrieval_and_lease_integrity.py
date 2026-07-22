"""CP126: retrieval metric coherence, lease safety, and flag truthiness.

* ``d5cc1faf`` — candidates were selected by raw Euclidean distance while
  the confidence gate judged them by mean-centered cosine. Raw hidden states
  share a dominant common direction, so distance is dominated by vector norm
  and that shared direction: the semantically strongest neighbour could be
  ranked outside the top k and never reach the gate at all.
* ``5a7bdf9c`` — the nucleus unload dropped the lane-lease handle in the
  same statement that read it, before the release was known to have
  succeeded, so a failed release stranded ownership with nothing left to
  retry or report with.
* ``8eda805e`` — a mandatory search handoff was disabled by the mere
  PRESENCE of an environment variable, so "0", "false" and "disabled" all
  switched the guard off.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.brain.nonparametric_memory import NonParametricMemory


@pytest.fixture
def anisotropic_store():
    """A store with the anisotropy the production hidden space actually has.

    Every key shares a dominant common direction, which is what makes
    Euclidean distance and centered cosine disagree.
    """
    rng = np.random.default_rng(7)
    dim = 64
    store = NonParametricMemory(dim=dim, max_entries=500)
    common = rng.normal(size=dim)
    common /= np.linalg.norm(common)

    def make(signature, scale):
        vector = 3.0 * common + signature
        return (vector / np.linalg.norm(vector) * scale).astype(np.float32)

    signature = rng.normal(size=dim)
    # The semantically right neighbour: same signature direction, far larger
    # norm — which is exactly what a distance scan punishes.
    store.add(make(signature, 12.0), token_id=111, token="right")
    for i in range(60):
        store.add(make(rng.normal(size=dim), 1.0), token_id=200 + i, token=f"decoy{i}")

    # Drive the running mean with diverse traffic, as decoding would.
    for _ in range(store.MU_READY_N + 5):
        store.query(make(rng.normal(size=dim), 1.0), k=1)

    return store, make(signature, 1.0), make


class TestSelectionUsesTheGateMetric:
    def test_the_right_neighbour_is_retrieved(self, anisotropic_store):
        store, query, _ = anisotropic_store
        tokens = [n.token for n in store.query(query, k=5)]
        assert "right" in tokens

    def test_it_ranks_first(self, anisotropic_store):
        store, query, _ = anisotropic_store
        assert store.query(query, k=5)[0].token == "right"

    def test_the_old_distance_scan_would_have_missed_it(self, anisotropic_store):
        """The defect, demonstrated rather than asserted.

        Reproduces the previous selector on the same store: the correct
        neighbour ranks nearly last by Euclidean distance, so no gate
        threshold could have recovered it.
        """
        store, query, _ = anisotropic_store
        n = store._size
        dist_sq = (
            store._key_norms[:n]
            + float(np.dot(query, query))
            - 2.0 * (store._keys[:n] @ query)
        )
        order = np.argsort(np.sqrt(np.maximum(dist_sq, 0.0)))
        euclidean_top5 = [store._tokens[i] for i in order[:5]]
        assert "right" not in euclidean_top5
        # And it is not marginal: it sorts to the far end.
        assert int(np.where(order == 0)[0][0]) > n // 2

    def test_results_are_sorted_by_similarity(self, anisotropic_store):
        store, query, _ = anisotropic_store
        results = store.query(query, k=8)
        sims = [n.similarity for n in results]
        assert sims == sorted(sims, reverse=True)

    def test_similarities_match_an_independent_computation(self, anisotropic_store):
        store, query, _ = anisotropic_store
        mu = store._query_mu
        for neighbor in store.query(query, k=5):
            key = store._keys[neighbor.index]
            a = query - mu
            b = key - mu
            expected = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            assert neighbor.similarity == pytest.approx(expected, abs=1e-4)

    def test_distance_is_still_reported(self, anisotropic_store):
        store, query, _ = anisotropic_store
        for neighbor in store.query(query, k=5):
            key = store._keys[neighbor.index]
            assert neighbor.distance == pytest.approx(
                float(np.linalg.norm(query - key)), rel=1e-3,
            )

    def test_k_is_respected_and_bounded(self, anisotropic_store):
        store, query, _ = anisotropic_store
        assert len(store.query(query, k=3)) == 3
        assert len(store.query(query, k=10_000)) == store._size


class TestDegenerateQueries:
    def test_a_query_on_the_common_direction_returns_nothing(self):
        """After centering it has no direction left to compare.

        Ranking on a near-zero denominator would be numerical noise
        presented as confidence, so retrieval reports nothing and the gate
        refuses.
        """
        rng = np.random.default_rng(3)
        dim = 32
        store = NonParametricMemory(dim=dim, max_entries=100)
        for i in range(20):
            store.add(rng.normal(size=dim).astype(np.float32), token_id=i, token=f"t{i}")
        fixed = rng.normal(size=dim).astype(np.float32)
        # Drive the mean to the query itself.
        for _ in range(store.MU_READY_N + 5):
            store.query(fixed, k=1)
        assert store.query(fixed, k=5) == []

    def test_the_degenerate_case_is_counted(self):
        rng = np.random.default_rng(3)
        dim = 32
        store = NonParametricMemory(dim=dim, max_entries=100)
        for i in range(20):
            store.add(rng.normal(size=dim).astype(np.float32), token_id=i, token=f"t{i}")
        fixed = rng.normal(size=dim).astype(np.float32)
        for _ in range(store.MU_READY_N + 5):
            store.query(fixed, k=1)
        store.query(fixed, k=5)
        assert int(store._stats.get("degenerate_query", 0)) > 0

    def test_a_non_finite_query_is_still_rejected(self):
        store = NonParametricMemory(dim=8, max_entries=10)
        store.add(np.ones(8, dtype=np.float32), token_id=1, token="a")
        assert store.query(np.full(8, np.nan, dtype=np.float32), k=1) == []

    def test_an_empty_store_returns_nothing(self):
        store = NonParametricMemory(dim=8, max_entries=10)
        assert store.query(np.ones(8, dtype=np.float32), k=1) == []


class TestSearchHandoffFlagTruthiness:
    def test_off_values_do_not_disable_the_guard(self, monkeypatch):
        from core.brain.llm.runtime_wiring import _env_flag_enabled

        for value in ("0", "false", "no", "off", "disabled", "", "   "):
            monkeypatch.setenv("AURA_TEST_FLAG", value)
            assert _env_flag_enabled("AURA_TEST_FLAG") is False, value

    def test_on_values_enable_it(self, monkeypatch):
        from core.brain.llm.runtime_wiring import _env_flag_enabled

        for value in ("1", "true", "TRUE", "yes", "on", "enabled"):
            monkeypatch.setenv("AURA_TEST_FLAG", value)
            assert _env_flag_enabled("AURA_TEST_FLAG") is True, value

    def test_an_unrecognised_value_is_off(self, monkeypatch):
        from core.brain.llm.runtime_wiring import _env_flag_enabled

        # A typo must not silently disable a guard.
        monkeypatch.setenv("AURA_TEST_FLAG", "ture")
        assert _env_flag_enabled("AURA_TEST_FLAG") is False

    def test_an_absent_flag_is_off(self, monkeypatch):
        from core.brain.llm.runtime_wiring import _env_flag_enabled

        monkeypatch.delenv("AURA_TEST_FLAG", raising=False)
        assert _env_flag_enabled("AURA_TEST_FLAG") is False

    def test_the_handoff_policy_uses_it(self):
        import inspect

        from core.brain.llm import runtime_wiring

        source = inspect.getsource(runtime_wiring.should_force_tool_handoff)
        assert '_env_flag_enabled("AURA_EMBODIED_CHALLENGE")' in source
        assert 'os.environ.get("AURA_EMBODIED_CHALLENGE")' not in source


class TestLeaseIsHeldUntilReleaseSucceeds:
    def _source(self) -> str:
        import inspect

        from core.brain.llm import nucleus_manager

        return inspect.getsource(nucleus_manager.NucleusManager._unload_model_entry_locked)

    def test_the_handle_is_not_dropped_while_reading_it(self):
        source = self._source()
        assert 'lease, entry["lane_lease"] = entry.get("lane_lease"), None' not in source
        assert 'lease = entry.get("lane_lease")' in source

    def test_a_failed_release_retains_the_lease(self):
        source = self._source()
        block = source.split("except (OSError", 1)[1]
        assert 'entry["lane_lease"] = lease' in block

    def test_a_failed_release_records_what_to_recover(self):
        source = self._source()
        assert "lane_lease_release_failed" in source

    def test_a_failed_release_is_critical(self):
        source = self._source()
        assert 'severity="critical"' in source

    def test_the_lease_is_cleared_only_after_success(self):
        source = self._source()
        # The clear happens after the try/except, not inside it.
        assert source.rindex('entry["lane_lease"] = None') > source.index("except (OSError")
