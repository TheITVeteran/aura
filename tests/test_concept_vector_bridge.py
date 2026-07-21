from __future__ import annotations

import asyncio

import numpy as np

from core.brain import concept_vector_bridge as bridge_module


def test_fallback_is_stable_across_bridge_instances(monkeypatch):
    monkeypatch.setattr(
        bridge_module,
        "get_runtime_service",
        lambda _name, default=None: default,
    )

    first = bridge_module.ConceptVectorBridge()
    second = bridge_module.ConceptVectorBridge()
    vector_a = asyncio.run(first.generate_concept_vector("stable shared concept"))
    vector_b = asyncio.run(second.generate_concept_vector("stable shared concept"))

    assert vector_a == vector_b
    assert len(vector_a) == bridge_module.ConceptVectorBridge.VECTOR_DIM
    assert np.isclose(np.linalg.norm(vector_a), 1.0)


def test_fallback_preserves_more_lexical_similarity_for_related_text(monkeypatch):
    monkeypatch.setattr(
        bridge_module,
        "get_runtime_service",
        lambda _name, default=None: default,
    )
    bridge = bridge_module.ConceptVectorBridge()

    anchor = np.asarray(asyncio.run(bridge.generate_concept_vector("memory continuity")))
    related = np.asarray(
        asyncio.run(bridge.generate_concept_vector("durable memory continuity"))
    )
    unrelated = np.asarray(
        asyncio.run(bridge.generate_concept_vector("thermal battery pressure"))
    )

    assert float(np.dot(anchor, related)) > float(np.dot(anchor, unrelated))


def test_shared_vector_memory_provider_replaces_boot_fallback(monkeypatch):
    services = {}
    monkeypatch.setattr(
        bridge_module,
        "get_runtime_service",
        lambda name, default=None: services.get(name, default),
    )
    bridge = bridge_module.ConceptVectorBridge()

    fallback = asyncio.run(bridge.generate_concept_vector("provider upgrade"))

    class Embedder:
        @staticmethod
        def embed(_text):
            return np.asarray([3.0, 1.0], dtype=np.float32)

    services["vector_memory_engine"] = type(
        "VectorMemory",
        (),
        {"embedder": Embedder()},
    )()
    upgraded = asyncio.run(bridge.generate_concept_vector("provider upgrade"))

    assert upgraded == [3.0, 1.0]
    assert upgraded != fallback
    assert bridge._concept_sources["provider upgrade"] == "provider"
