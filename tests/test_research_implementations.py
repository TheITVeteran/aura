"""Tests for research-inspired implementations.

Covers:
  - Conceptual Gravitation (C.O.R.E.)
  - Three-layer Knowledge Compression (Concord DTU)
  - Navigating Spreading-out Graph (Cognitive SSD)
  - STDP Online Learning (BrainCog)
"""
import numpy as np
import pytest


# ── Conceptual Gravitation ───────────────────────────────────────────────────

class TestConceptualGravitation:

    def test_co_access_recording(self):
        from core.memory.conceptual_gravitation import ConceptualGravitationEngine
        engine = ConceptualGravitationEngine()
        engine.record_recall("mem_1")
        engine.record_recall("mem_2")
        engine.record_recall("mem_3")
        engine.end_turn()
        # Should have 3 pairs: (1,2), (1,3), (2,3)
        assert len(engine._co_accesses) == 3

    def test_gravitation_nudges_embeddings(self):
        from core.memory.conceptual_gravitation import ConceptualGravitationEngine

        class MockStore:
            def __init__(self):
                self.embeddings = {}
            def get_embedding(self, id):
                return self.embeddings.get(id)
            def set_embedding(self, id, emb):
                self.embeddings[id] = emb

        store = MockStore()
        store.embeddings["a"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store.embeddings["b"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        engine = ConceptualGravitationEngine()
        # Record enough co-accesses to trigger gravity
        for _ in range(5):
            engine.record_recall("a")
            engine.record_recall("b")
            engine.end_turn()

        stats = engine.consolidate(store)
        assert stats["nudged"] >= 1

        # Embeddings should be closer after nudging
        dist_after = np.linalg.norm(store.embeddings["a"] - store.embeddings["b"])
        dist_before = np.linalg.norm(np.array([1, 0, 0]) - np.array([0, 1, 0]))
        assert dist_after < dist_before

    def test_embeddings_stay_normalized(self):
        from core.memory.conceptual_gravitation import ConceptualGravitationEngine

        class MockStore:
            def __init__(self):
                self.embeddings = {}
            def get_embedding(self, id):
                return self.embeddings.get(id)
            def set_embedding(self, id, emb):
                self.embeddings[id] = emb

        store = MockStore()
        store.embeddings["x"] = np.random.randn(32).astype(np.float32)
        store.embeddings["x"] /= np.linalg.norm(store.embeddings["x"])
        store.embeddings["y"] = np.random.randn(32).astype(np.float32)
        store.embeddings["y"] /= np.linalg.norm(store.embeddings["y"])

        engine = ConceptualGravitationEngine()
        for _ in range(10):
            engine.record_recall("x")
            engine.record_recall("y")
            engine.end_turn()

        engine.consolidate(store)
        assert abs(np.linalg.norm(store.embeddings["x"]) - 1.0) < 0.01
        assert abs(np.linalg.norm(store.embeddings["y"]) - 1.0) < 0.01


# ── Knowledge Compression ────────────────────────────────────────────────────

class TestKnowledgeCompression:

    def test_compress_turns(self):
        from core.memory.knowledge_compression import KnowledgeCompressor
        compressor = KnowledgeCompressor()
        turns = [
            {"role": "user", "content": "I'm thinking about moving to Austin."},
            {"role": "assistant", "content": "Austin is great. Good music scene, tech jobs, warm weather."},
            {"role": "user", "content": "What about Denver?"},
            {"role": "assistant", "content": "Denver is colder but the mountains are unbeatable."},
        ]
        atom = compressor.compress_turns(turns)

        assert atom.turn_count == 4
        assert len(atom.readable) > 0
        assert atom.vector is not None
        assert atom.vector.shape == (32,)
        assert "Austin" in atom.entities or "Denver" in atom.entities

    def test_sentiment_computation(self):
        from core.memory.knowledge_compression import KnowledgeCompressor
        compressor = KnowledgeCompressor()

        happy_turns = [{"role": "user", "content": "I'm so happy and grateful, everything is awesome!"}]
        sad_turns = [{"role": "user", "content": "I'm sad and frustrated, everything is terrible."}]

        happy_atom = compressor.compress_turns(happy_turns)
        sad_atom = compressor.compress_turns(sad_turns)

        assert happy_atom.sentiment > 0
        assert sad_atom.sentiment < 0

    def test_merge_atoms(self):
        from core.memory.knowledge_compression import KnowledgeCompressor, KnowledgeAtom
        compressor = KnowledgeCompressor()

        a1 = KnowledgeAtom(readable="User asked about Python.", entities=["Python"], turn_count=2,
                           vector=np.array([1.0] + [0.0]*31, dtype=np.float32))
        a2 = KnowledgeAtom(readable="Discussed Rust performance.", entities=["Rust"], turn_count=3,
                           vector=np.array([0.0, 1.0] + [0.0]*30, dtype=np.float32))

        merged = compressor.merge_atoms([a1, a2])
        assert merged.turn_count == 5
        assert "Python" in merged.entities or "Rust" in merged.entities
        assert merged.vector is not None

    def test_compressed_context_output(self):
        from core.memory.knowledge_compression import KnowledgeCompressor
        compressor = KnowledgeCompressor()
        for i in range(5):
            compressor.compress_turns([
                {"role": "user", "content": f"Message number {i} about topic {i}."},
                {"role": "assistant", "content": f"Response to topic {i}."},
            ])

        context = compressor.get_compressed_context(max_chars=500)
        assert len(context) > 0
        assert len(context) <= 600  # Slight overshoot OK due to joiner


# ── Navigating Graph ─────────────────────────────────────────────────────────

class TestNavigatingGraph:

    def test_add_and_search(self):
        from core.memory.navigating_graph import NavigatingGraph
        nsg = NavigatingGraph(dim=32)

        # Add 50 random memories
        np.random.seed(42)
        for i in range(50):
            emb = np.random.randn(32).astype(np.float32)
            nsg.add(f"mem_{i}", emb, metadata={"index": i})

        # Search for something similar to mem_0
        query = nsg.get_embedding("mem_0")
        results = nsg.search(query, top_k=5)

        assert len(results) > 0
        assert results[0]["id"] == "mem_0"  # Should find itself
        assert results[0]["distance"] < 0.01

    def test_search_returns_nearest(self):
        from core.memory.navigating_graph import NavigatingGraph
        nsg = NavigatingGraph(dim=3)

        # Add three known points
        nsg.add("north", np.array([0, 0, 1], dtype=np.float32))
        nsg.add("east", np.array([1, 0, 0], dtype=np.float32))
        nsg.add("northeast", np.array([0.7, 0, 0.7], dtype=np.float32))

        # Query close to north
        results = nsg.search(np.array([0.1, 0, 0.9], dtype=np.float32), top_k=2)
        assert results[0]["id"] == "north"

    def test_remove_node(self):
        from core.memory.navigating_graph import NavigatingGraph
        nsg = NavigatingGraph(dim=8)
        nsg.add("a", np.random.randn(8).astype(np.float32))
        nsg.add("b", np.random.randn(8).astype(np.float32))
        nsg.add("c", np.random.randn(8).astype(np.float32))

        nsg.remove("b")
        assert "b" not in nsg._nodes
        results = nsg.search(np.random.randn(8).astype(np.float32), top_k=5)
        assert all(r["id"] != "b" for r in results)

    def test_rebuild_graph(self):
        from core.memory.navigating_graph import NavigatingGraph
        nsg = NavigatingGraph(dim=16)
        np.random.seed(123)
        for i in range(100):
            nsg.add(f"n_{i}", np.random.randn(16).astype(np.float32))

        nsg.rebuild()
        assert nsg._inserts_since_rebuild == 0

        # Search should still work after rebuild
        results = nsg.search(np.random.randn(16).astype(np.float32), top_k=5)
        assert len(results) == 5

    def test_get_set_embedding(self):
        from core.memory.navigating_graph import NavigatingGraph
        nsg = NavigatingGraph(dim=4)
        original = np.array([1, 0, 0, 0], dtype=np.float32)
        nsg.add("test", original)

        emb = nsg.get_embedding("test")
        assert emb is not None
        assert np.allclose(emb, original / np.linalg.norm(original))

        new_emb = np.array([0, 1, 0, 0], dtype=np.float32)
        nsg.set_embedding("test", new_emb)
        updated = nsg.get_embedding("test")
        assert np.allclose(updated, new_emb / np.linalg.norm(new_emb))

    def test_large_graph_performance(self):
        from core.memory.navigating_graph import NavigatingGraph
        import time

        nsg = NavigatingGraph(dim=128)
        np.random.seed(0)
        for i in range(1000):
            nsg.add(f"m_{i}", np.random.randn(128).astype(np.float32))

        query = np.random.randn(128).astype(np.float32)
        start = time.time()
        results = nsg.search(query, top_k=10)
        elapsed = time.time() - start

        assert len(results) == 10
        assert elapsed < 1.0  # Should be well under 1 second for 1K nodes


# ── STDP Learning ────────────────────────────────────────────────────────────

class TestSTDPLearning:
    def test_process_singleton_registers_at_requested_live_dimension(self, monkeypatch):
        from core.consciousness import stdp_learning

        registrations = []
        monkeypatch.setattr(stdp_learning, "_instance", None)
        monkeypatch.setattr(
            stdp_learning.ServiceContainer,
            "has",
            classmethod(lambda cls, name: False),
        )
        monkeypatch.setattr(
            stdp_learning.ServiceContainer,
            "get",
            classmethod(lambda cls, name, default=None: default),
        )
        monkeypatch.setattr(
            stdp_learning.ServiceContainer,
            "register_instance",
            classmethod(
                lambda cls, name, instance, **metadata: registrations.append(
                    (name, instance, metadata)
                )
            ),
        )

        engine = stdp_learning.get_stdp_engine(n_neurons=512)

        assert engine.n == 512
        assert engine._eligibility.shape == (512, 512)
        assert registrations[0][0] == "stdp_engine"
        assert registrations[0][1] is engine
        assert registrations[0][2]["required_for"] == (
            "liquid substrate reward-modulated plasticity"
        )

    def test_process_singleton_resizes_to_match_new_substrate(self, monkeypatch):
        from core.consciousness import stdp_learning

        monkeypatch.setattr(
            stdp_learning,
            "_instance",
            stdp_learning.STDPLearningEngine(n_neurons=8),
        )
        monkeypatch.setattr(
            stdp_learning.ServiceContainer,
            "has",
            classmethod(lambda cls, name: False),
        )
        monkeypatch.setattr(
            stdp_learning.ServiceContainer,
            "get",
            classmethod(lambda cls, name, default=None: default),
        )
        monkeypatch.setattr(
            stdp_learning.ServiceContainer,
            "register_instance",
            classmethod(lambda cls, name, instance, **metadata: None),
        )

        engine = stdp_learning.get_stdp_engine(n_neurons=16)

        assert engine.n == 16
        assert engine._eligibility.shape == (16, 16)


    def test_record_spikes_updates_eligibility(self):
        from core.consciousness.stdp_learning import STDPLearningEngine
        engine = STDPLearningEngine(n_neurons=8)

        # Fire neurons 0 and 3 at different times
        activations_t1 = np.zeros(8, dtype=np.float32)
        activations_t1[0] = 1.0
        engine.record_spikes(activations_t1, t=0.0)

        activations_t2 = np.zeros(8, dtype=np.float32)
        activations_t2[3] = 1.0
        engine.record_spikes(activations_t2, t=5.0)

        # Eligibility should be non-zero between neurons 0 and 3
        assert engine._eligibility[0, 3] != 0.0

    def test_reward_modulates_learning(self):
        from core.consciousness.stdp_learning import STDPLearningEngine
        engine = STDPLearningEngine(n_neurons=8)

        # Create some eligibility
        activations = np.ones(8, dtype=np.float32)
        engine.record_spikes(activations, t=0.0)

        # High surprise → larger weight delta
        dw_high = engine.deliver_reward(surprise=0.9, prediction_error=0.5)

        engine2 = STDPLearningEngine(n_neurons=8)
        engine2.record_spikes(activations, t=0.0)
        dw_low = engine2.deliver_reward(surprise=0.1, prediction_error=0.5)

        assert np.linalg.norm(dw_high) >= np.linalg.norm(dw_low)

    def test_apply_to_connectivity(self):
        from core.consciousness.stdp_learning import STDPLearningEngine
        engine = STDPLearningEngine(n_neurons=4)

        W = np.eye(4, dtype=np.float32) * 0.5
        dw = np.ones((4, 4), dtype=np.float32) * 0.001

        W_new = engine.apply_to_connectivity(W, dw)

        # Diagonal should be zero (no self-connections)
        assert all(W_new[i, i] == 0.0 for i in range(4))
        # Off-diagonal should have changed
        assert W_new[0, 1] > 0

    def test_weight_clipping(self):
        from core.consciousness.stdp_learning import STDPLearningEngine, WEIGHT_CLIP
        engine = STDPLearningEngine(n_neurons=4)

        W = np.ones((4, 4), dtype=np.float32) * 100.0
        dw = np.ones((4, 4), dtype=np.float32) * 100.0

        W_new = engine.apply_to_connectivity(W, dw)
        assert np.all(np.abs(W_new) <= WEIGHT_CLIP + 0.01)

    def test_learning_rate_scales_with_surprise(self):
        from core.consciousness.stdp_learning import STDPLearningEngine
        engine = STDPLearningEngine(n_neurons=4)

        activations = np.ones(4, dtype=np.float32)
        engine.record_spikes(activations, t=0.0)

        engine.deliver_reward(surprise=0.0, prediction_error=0.1)
        lr_low = engine._learning_rate

        engine.deliver_reward(surprise=1.0, prediction_error=0.1)
        lr_high = engine._learning_rate

        assert lr_high > lr_low

    def test_status_reports(self):
        from core.consciousness.stdp_learning import STDPLearningEngine
        engine = STDPLearningEngine(n_neurons=8)
        status = engine.get_status()
        assert "learning_rate" in status
        assert "total_updates" in status
        assert "eligibility_norm" in status
