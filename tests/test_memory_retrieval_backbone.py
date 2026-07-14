"""Contracts for the memory-retrieval backbone (July 2026 external review).

The review's load-bearing finding: "the intelligence amplifiers are real, the
consciousness scaffolding is real, but the memory retrieval backbone undercuts
them" — rag.retrieve_memories was substring match with compute_cosine_similarity
sitting unused right above it. These tests pin the fixes:

  1. retrieval is real TF-IDF cosine (ranks by relevance, not presence);
  2. conceptual gravitation has a feeder and a consolidation caller;
  3. one-shot episodic binding IS the hippocampal index (documented mechanism).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestTfidfRetrieval:
    def test_ranks_by_relevance_not_mere_presence(self):
        from core.memory.rag import retrieve_memories

        memories = [
            {"id": "a", "text": "the cat sat on the mat in the sun"},
            {"id": "b", "text": "quantum chromodynamics binds quarks with gluons"},
            {"id": "c", "text": "a cat and another cat chased a third cat"},
        ]
        results = retrieve_memories("cat", memories, top_k=3)
        ids = [r["id"] for r in results]
        # c (cat×3) must outrank a (cat×1); b (no term) must not appear
        assert ids[0] == "c"
        assert "b" not in ids
        assert results[0]["score"] > results[1]["score"]

    def test_rare_terms_outweigh_common_ones(self):
        from core.memory.rag import retrieve_memories

        memories = [
            {"id": "common", "text": "the the the the the the report"},
            {"id": "rare", "text": "the bekenstein bound report"},
        ]
        # "bekenstein" is rare across the set → should dominate the match
        results = retrieve_memories("the bekenstein report", memories, top_k=2)
        assert results[0]["id"] == "rare"

    def test_exact_phrase_gets_a_bounded_bonus(self):
        from core.memory.rag import retrieve_memories

        memories = [
            {"id": "scattered", "text": "meeting notes about the budget and the timeline"},
            {"id": "verbatim", "text": "the budget timeline meeting is on friday"},
        ]
        results = retrieve_memories("budget timeline meeting", memories, top_k=2)
        assert results[0]["id"] == "verbatim"
        assert results[0]["score"] <= 1.0        # bonus stays bounded

    def test_no_query_terms_returns_empty(self):
        from core.memory.rag import retrieve_memories

        assert retrieve_memories("", [{"id": "a", "text": "x"}]) == []
        assert retrieve_memories("cat", []) == []

    def test_uses_the_previously_dead_helpers(self):
        """Guard against regression to substring: the cosine helper must matter."""
        import inspect

        from core.memory import rag

        source = inspect.getsource(rag.retrieve_memories)
        assert "compute_cosine_similarity" in source
        assert "compute_term_freq" in source
        assert "query in text" not in source  # the old substring test is gone


class TestGravitationWiring:
    def test_facade_search_feeds_co_access_events(self):
        """The feeder lives inside MemoryFacade.search itself — pin the source
        so a refactor cannot silently orphan the gravitation engine again."""
        import inspect

        from core.memory.memory_facade import MemoryFacade

        src = inspect.getsource(MemoryFacade.search)
        assert "conceptual_gravitation" in src
        assert "record_recall" in src
        assert "end_turn" in src

    def test_consolidate_has_a_dream_cycle_caller(self):
        import inspect

        from core.resilience import dream_cycle

        src = inspect.getsource(dream_cycle.DreamCycle)
        assert "_consolidate_gravitation" in src
        assert ".consolidate(" in src or "gravitation.consolidate" in src

    def test_gravitation_nudges_recalled_pairs_closer(self):
        import numpy as np

        from core.memory.conceptual_gravitation import ConceptualGravitationEngine

        engine = ConceptualGravitationEngine()
        # co-recall a pair three turns running
        for _ in range(3):
            engine.record_recall("a")
            engine.record_recall("b")
            engine.end_turn()

        store = {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0]),
        }

        class Store:
            def get_embedding(self, i):
                return store.get(i)

            def set_embedding(self, i, v):
                store[i] = v

        before = float(np.dot(store["a"], store["b"]))
        engine.consolidate(Store())
        after = float(np.dot(store["a"], store["b"]))
        assert after > before  # cosine similarity increased


class TestKnowledgeSearchIsRanked:
    @pytest.fixture(autouse=True)
    def _isolate_from_ambient_governance(self, monkeypatch):
        """These are search/ranking unit tests; approve writes explicitly.

        add_knowledge consults the live constitutional core, whose present-
        state (AuraNow) policy is process-global — an earlier test that left
        the shared runtime in a 'stabilization first' posture caused every
        write here to be denied (aura_now_defer) and the searches to scan an
        empty table (order-dependence register, 2 victims).  Governance
        behavior has its own tests; here it is stubbed to approve.
        """
        from core.memory.knowledge_graph import PersistentKnowledgeGraph

        def _approve(self, *args, return_decision=False, **kwargs):
            return (True, None) if return_decision else True

        monkeypatch.setattr(PersistentKnowledgeGraph, "_approve_memory_write", _approve)

    def test_fts5_ranks_by_relevance_not_substring(self, tmp_path):
        from core.memory.knowledge_graph import PersistentKnowledgeGraph

        kg = PersistentKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
        assert kg._fts_enabled, "FTS5 must be active on a fresh database"
        kg.add_knowledge("the mitochondria is the powerhouse of the cell", "fact")
        kg.add_knowledge("cell membranes regulate transport", "fact")
        kg.add_knowledge("cooking pasta requires boiling water", "fact")
        results = kg.search_knowledge("mitochondria cell")
        contents = [r["content"] for r in results]
        assert contents and "mitochondria" in contents[0]
        assert all("pasta" not in c for c in contents)

    def test_multi_word_query_matches_any_order(self, tmp_path):
        """LIKE '%a b%' required adjacency; FTS must not."""
        from core.memory.knowledge_graph import PersistentKnowledgeGraph

        kg = PersistentKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
        kg.add_knowledge("the budget for the timeline was approved", "fact")
        results = kg.search_knowledge("timeline budget")
        assert results, "out-of-order terms must still match under FTS"

    def test_backfill_indexes_preexisting_rows(self, tmp_path):
        import sqlite3

        from core.memory.knowledge_graph import PersistentKnowledgeGraph

        db = tmp_path / "kg.db"
        # simulate a pre-FTS database: raw schema + rows, no fts table
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE knowledge (
            id TEXT PRIMARY KEY, content TEXT, type TEXT, source TEXT,
            confidence REAL, created_at REAL, last_accessed REAL,
            access_count INTEGER, metadata TEXT)""")
        conn.execute(
            "INSERT INTO knowledge VALUES ('k1', 'ancient rows about gluons', 'fact', 's', 0.9, 0, 0, 0, '{}')"
        )
        conn.commit()
        conn.close()

        kg = PersistentKnowledgeGraph(db_path=str(db))
        results = kg.search_knowledge("gluons")
        assert any(r["id"] == "k1" for r in results)


class TestPromotionGateThreshold:
    def test_default_pass_rate_allows_marginal_residual(self):
        """1.0 stalled self-improvement (one flaky behavioral test blocked every
        promotion). Audits and syntax stay hard; the residual rate is 0.95."""
        from core.self_improvement.promotion_gate import LabPromotionGate

        gate = LabPromotionGate()
        assert gate.min_pass_rate == pytest.approx(0.95)
        assert gate.require_syntax_valid is True
        assert gate.require_surface_preserved is True


class TestOneShotIsHippocampalBinding:
    def test_bind_then_partial_cue_completes_the_assembly(self, tmp_path):
        import sqlite3

        from core.memory.hippocampus import HippocampalIndex

        db = tmp_path / "hippo.db"

        def conn_factory():
            return sqlite3.connect(db)

        index = HippocampalIndex(conn_factory)
        # ONE exposure — one-shot episodic binding
        index.bind("episode-1", ["angry", "crow", "rooftop", "storm"])
        # a PARTIAL cue set reinstates the whole episode (CA3 pattern completion)
        completed = index.pattern_complete(["crow", "storm"], limit=5)
        assert any(eid == "episode-1" for eid, _ in completed)


# ── Hybrid semantic retrieval (July capability raise) ─────────────────────


class _FakeDenseEngine:
    """Deterministic 'semantic' backend: axis per known concept."""

    _AXES = {"feline": 0, "cat": 0, "kitten": 0, "finance": 1, "budget": 1}

    def _vec(self, text):
        import numpy as np

        v = np.zeros(4, dtype=np.float32)
        for word, axis in self._AXES.items():
            if word in text.lower():
                v[axis] += 1.0
        if not v.any():
            v[3] = 1.0
        return v

    _model = object()  # non-None → "real backend up"

    def embed(self, text):
        return self._vec(text)

    def embed_batch(self, texts):
        return [self._vec(t) for t in texts]


@pytest.fixture
def semantic_rag(monkeypatch):
    from core.memory import rag

    rag.reset_semantic_state_for_test()
    monkeypatch.setenv("AURA_SEMANTIC_RAG", "1")
    monkeypatch.setattr(rag, "_get_embed_engine", lambda: _FakeDenseEngine())
    yield rag
    rag.reset_semantic_state_for_test()


def test_semantic_match_outranks_lexical_overlap(semantic_rag):
    """The claim the vault's vocabulary always made: MEANING retrieves,
    not just shared tokens. 'kitten' shares zero tokens with 'cat' but
    must outrank a lexically-overlapping but unrelated memory."""
    memories = [
        {"text": "the cat sat by the feline door", "id": "about-cats"},
        {"text": "my monthly budget review is due", "id": "about-money"},
    ]
    results = semantic_rag.retrieve_memories("kitten photos", memories, top_k=2)
    assert results, "hybrid retrieval returned nothing"
    assert results[0]["id"] == "about-cats"
    assert results[0]["retrieval"] == "hybrid_semantic"


def test_tfidf_fallback_when_backend_down(monkeypatch):
    from core.memory import rag

    rag.reset_semantic_state_for_test()
    monkeypatch.setenv("AURA_SEMANTIC_RAG", "0")
    memories = [{"text": "the retry budget is three attempts", "id": "m1"}]
    results = rag.retrieve_memories("retry budget", memories, top_k=1)
    assert results and results[0]["retrieval"] == "tfidf"
    rag.reset_semantic_state_for_test()


def test_too_many_cold_texts_defers_to_background_warm(semantic_rag, monkeypatch):
    """Bounded work: a query over a huge cold corpus must not stall — it
    falls back to TF-IDF and warms the cache for the next query."""
    from core.memory import rag

    monkeypatch.setattr(rag, "_SEMANTIC_MAX_UNCACHED", 3)
    memories = [{"text": f"memory number {i} about topic {i}", "id": str(i)} for i in range(10)]
    results = semantic_rag.retrieve_memories("memory number 4", memories, top_k=3)
    assert results
    assert all(r["retrieval"] == "tfidf" for r in results), "cold corpus → lexical this query"


def test_semantic_cache_is_bounded(semantic_rag, monkeypatch):
    from core.memory import rag

    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_MAX", 5)
    memories = [{"text": f"unique text {i}", "id": str(i)} for i in range(4)]
    for query in ("cat", "budget", "kitten"):
        semantic_rag.retrieve_memories(query, memories, top_k=2)
    assert len(rag._SEMANTIC_CACHE) <= 5
