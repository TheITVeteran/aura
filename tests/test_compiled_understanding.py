"""Contract tests: the Compiled Understanding Layer.

The layer's promises, proven on real machinery (SQLite library, real
grounding math, injected compiler lanes):
- concepts are digested ONCE and served from cache forever after;
- compiler identity is honest (solver vs resident vs heuristic fallback);
- verification is deterministic grounding against the actual sources;
- bridges connect recently-used concepts and refuse to invent connections;
- reuse evidence exports through the governed gateway for consolidation;
- every degraded path (no corpus, no lanes, broken library) returns honest
  empties, never fabricated understanding.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.knowledge.compiled_understanding import (
    COMPILER_DEEP_SOLVER,
    COMPILER_HEURISTIC,
    CompiledUnderstandingService,
    ConceptCompiler,
    ConceptDigest,
    DigestLibrary,
    concept_key,
    extract_concepts,
    grounding_score,
)


class StubCorpus:
    """Deterministic corpus with provenance-shaped hits."""

    class Hit:
        def __init__(self, title, snippet, doc_id):
            self.title = title
            self.snippet = snippet
            self.source = "wikipedia"
            self.rank = -1.0
            self.doc_id = doc_id

    def __init__(self, docs=None):
        self.docs = docs if docs is not None else {
            "hash join": "A hash join builds a hash table on the smaller "
            "relation and probes it with the larger relation; it wins when "
            "equality predicates dominate and memory holds the build side.",
            "merge join": "A merge join requires both inputs sorted on the "
            "join key and streams them together; it wins on pre-sorted "
            "inputs and bounded memory.",
        }

    def search(self, query, limit=3):
        hits = []
        for index, (title, text) in enumerate(self.docs.items()):
            if any(word in text.lower() or word in title for word in query.lower().split()):
                hits.append(self.Hit(title, text, index))
        return hits[:limit]


def _solver_think(response="A precise digest: the hash join builds a table "
                  "on the smaller relation and probes with the larger; its "
                  "constraint is memory for the build side; analogy: a "
                  "phone book lookup; failure mode: skewed keys."):
    async def think(prompt):
        return response

    return think


# ── Concept extraction + keys ───────────────────────────────────────────


def test_concept_extraction_prefers_content_bigrams():
    concepts = extract_concepts(
        "Compare the hash join with the merge join under memory pressure."
    )
    assert "hash join" in concepts
    assert all("the" not in c.split() for c in concepts)


def test_concept_key_is_canonical():
    assert concept_key("  Hash-Join!  ") == concept_key("hash join")
    with pytest.raises(ValueError):
        concept_key("!!!")


def test_grounding_score_separates_grounded_from_drifted():
    material = "the hash join builds a hash table on the smaller relation"
    grounded = grounding_score("hash join builds a table on the smaller relation", material)
    drifted = grounding_score("quantum entanglement enables faster-than-light gossip", material)
    assert grounded > 0.8 and drifted < 0.2


# ── Library ─────────────────────────────────────────────────────────────


def test_library_round_trip_hits_and_bridges(tmp_path):
    lib = DigestLibrary(tmp_path / "digests.db")
    digest = ConceptDigest(
        key="hash join", digest_text="dense digest", compiler=COMPILER_DEEP_SOLVER,
        grounding=0.9, verified=True,
    )
    lib.put(digest)
    fetched = lib.get("hash join")
    assert fetched is not None and fetched.digest_text == "dense digest"
    for _ in range(4):
        lib.record_use("hash join")
    assert lib.get("hash join").hits == 4
    lib.put(ConceptDigest(key="merge join", digest_text="d2",
                          compiler=COMPILER_DEEP_SOLVER, grounding=0.9, verified=True))
    lib.add_bridge("hash join", "merge join")
    assert "merge join" in lib.get("hash join").bridges
    assert "hash join" in lib.get("merge join").bridges
    heavy = lib.heavily_used(min_hits=3)
    assert [d.key for d in heavy] == ["hash join"]
    stats = lib.stats()
    assert stats["digests"] == 2 and stats["verified"] == 2


def test_library_missing_db_reads_return_empty(tmp_path):
    lib = DigestLibrary(tmp_path / "sub" / "digests.db")
    assert lib.get("anything") is None
    assert lib.heavily_used() == []


# ── Compiler ────────────────────────────────────────────────────────────


def test_compiler_uses_injected_lane_and_verifies_grounding():
    compiler = ConceptCompiler(think=_solver_think())
    digest = asyncio.run(
        compiler.compile(
            "hash join",
            [{"text": StubCorpus().docs["hash join"], "title": "hash join",
              "source": "wikipedia", "doc_id": 0}],
        )
    )
    assert digest.compiler == COMPILER_DEEP_SOLVER
    assert digest.verified and digest.grounding > 0.35
    assert digest.sources[0]["title"] == "hash join"


def test_compiler_falls_back_to_heuristic_and_says_so():
    async def dead_lane(prompt):
        raise RuntimeError("no capacity")

    compiler = ConceptCompiler(think=dead_lane)
    digest = asyncio.run(
        compiler.compile(
            "hash join",
            [{"text": StubCorpus().docs["hash join"], "title": "hash join",
              "source": "wikipedia", "doc_id": 0}],
        )
    )
    assert digest.compiler == COMPILER_HEURISTIC
    assert "hash" in digest.digest_text.lower()
    # Extractive text is grounded by construction.
    assert digest.verified


def test_compiler_refuses_empty_material():
    compiler = ConceptCompiler(think=_solver_think())
    with pytest.raises(ValueError):
        asyncio.run(compiler.compile("hash join", [{"text": "  "}]))


def test_bridge_compiler_refuses_no_connection():
    async def honest_lane(prompt):
        return "NO SUBSTANTIVE CONNECTION"

    compiler = ConceptCompiler(think=honest_lane)
    assert asyncio.run(compiler.compile_bridge("hash join", "haiku")) is None


# ── Service ─────────────────────────────────────────────────────────────


def _service(tmp_path, think=None, corpus=None):
    return CompiledUnderstandingService(
        library=DigestLibrary(tmp_path / "digests.db"),
        compiler=ConceptCompiler(think=think or _solver_think()),
        corpus=corpus or StubCorpus(),
    )


def test_understand_compiles_once_then_serves_cache(tmp_path):
    service = _service(tmp_path)
    first = asyncio.run(service.understand("How does the hash join work?"))
    assert first["compiled_now"] == ["hash join"]
    assert "hash join" in first["context"]

    second = asyncio.run(service.understand("Explain the hash join again"))
    assert second["compiled_now"] == []
    assert "hash join" in second["context"]
    assert service.get_status()["cache_hits"] >= 1
    assert service.library.get("hash join").hits >= 1


def test_understand_bounds_foreground_compiles(tmp_path):
    service = _service(tmp_path)
    receipt = asyncio.run(
        service.understand(
            "Compare hash join versus merge join tradeoffs", max_compiles=1
        )
    )
    assert len(receipt["compiled_now"]) == 1
    assert receipt["uncompiled"], "the rest must be left for idle cycles, honestly"


def test_understand_without_corpus_is_honest(tmp_path):
    class EmptyCorpus:
        def search(self, query, limit=3):
            return []

    service = _service(tmp_path, corpus=EmptyCorpus())
    receipt = asyncio.run(service.understand("explain the flux capacitor"))
    assert receipt["context"] == ""
    assert receipt["digest_keys"] == []
    assert receipt["uncompiled"], "unfindable concepts are reported, not invented"


def test_unverified_digests_are_flagged(tmp_path):
    drifting = _solver_think("Entirely unrelated mystical prose about moons and tides.")
    service = _service(tmp_path, think=drifting)
    receipt = asyncio.run(service.understand("How does the hash join work?"))
    assert receipt["unverified_digests"] == ["hash join"]


def test_bridge_cycle_connects_recent_concepts(tmp_path):
    service = _service(tmp_path)
    asyncio.run(service.understand("hash join details", max_compiles=1))
    asyncio.run(service.understand("merge join details", max_compiles=1))
    receipt = asyncio.run(service.bridge_cycle(max_pairs=2))
    assert receipt["attempted"] >= 1
    assert receipt["bridged"], receipt
    left, right = receipt["bridged"][0]
    assert right in service.library.get(left).bridges


def test_reuse_evidence_exports_via_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    service = _service(tmp_path)
    asyncio.run(service.understand("hash join plan"))
    for _ in range(3):
        asyncio.run(service.understand("hash join plan"))
    out = tmp_path / "evidence.json"
    payload = service.export_reuse_evidence(out)
    assert payload["candidates"], "reused digests must appear as evidence"
    assert payload["candidates"][0]["key"] == "hash join"
    written = json.loads(out.read_text())
    assert written["schema"] == "aura.digest_reuse_evidence.v1"


def test_service_registers_in_spine():
    from core.service_names import ServiceNames

    assert ServiceNames.COMPILED_UNDERSTANDING == "compiled_understanding"


def test_deep_deliberation_feeds_digests_into_the_episode(tmp_path, monkeypatch):
    """The live seam: compiled digests enter the latent episode as a system
    message; a failed/absent layer degrades to the bare question."""
    import core.brain.deep_deliberation as dd
    import core.knowledge.compiled_understanding as cu

    captured: dict = {}

    class StubLatentService:
        async def deep_reason(self, question=None, *, messages=None, **kwargs):
            captured["question"] = question
            captured["messages"] = messages
            return {"ok": True, "text": "the deliberate answer", "receipt": {}}

    class StubBrain:
        async def think(self, prompt, **kwargs):
            return "How does the hash join build and probe its table?"

    service = _service(tmp_path)
    monkeypatch.setattr(cu, "_INSTANCE", service)
    monkeypatch.setattr(
        dd, "resolve_brain", lambda *_a, **_k: StubBrain()
    )
    import core.brain.latent_cortex_service as lcs

    monkeypatch.setattr(
        lcs, "get_latent_cortex_service", lambda *_a, **_k: StubLatentService()
    )

    engine = dd.DeepDeliberationEngine()
    result = asyncio.run(
        engine.deliberate("How does the hash join work?", timeout_s=30.0)
    )
    assert result.used_latent_cortex is True
    assert captured["messages"] is not None, "digest context must reach the episode"
    system = captured["messages"][0]
    assert system["role"] == "system"
    assert "hash join" in system["content"]
    assert captured["messages"][1]["role"] == "user"
