"""Contract tests: the offline encyclopedia reaches the deliberation path.

The 6.6M-article local Wikipedia existed behind a skill, but cognitive
ingress never consulted it — the knowledge organ the integration bet names
first could not seed a thought slot. These tests pin the new seam:
corpus hits are CLAIMS admitted through the epistemic firewall, grounding
lowers uncertainty, conflicts seed caution instead of a winner, an absent
or broken store is absent (never fatal), and the content becomes an
identifiable `reference` slot item.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.cognitive_ingress import (
    assemble_cognitive_ingress,
    cognitive_context_items,
)
from core.brain.llm.latent_cortex.cognitive_context import (
    normalize_cognitive_context,
)


class _Hit:
    def __init__(self, title: str, snippet: str):
        self.title = title
        self.snippet = snippet
        self.source = "wikipedia"
        self.rank = -1.0
        self.doc_id = 1


@pytest.fixture()
def corpus(monkeypatch):
    """A controllable local-corpus store."""
    holder: dict[str, object] = {"store": None}

    import core.knowledge.local_corpus as corpus_mod

    monkeypatch.setattr(
        corpus_mod, "get_local_corpus_store", lambda: holder["store"]
    )
    # Isolate from the live registry so only the corpus signal varies.
    import core.brain.cognitive_ingress as ingress_mod

    monkeypatch.setattr(ingress_mod, "_get_service", lambda name: None)
    return holder


def test_grounded_hits_seed_a_reference_slot_and_lower_uncertainty(corpus):
    class Store:
        def search(self, query, limit=4):
            return [
                _Hit("Photosynthesis", "converts light energy into chemical energy"),
                _Hit("Chlorophyll", "the pigment absorbing blue and red light"),
            ]

    corpus["store"] = Store()
    grounded = assemble_cognitive_ingress(None, "how does photosynthesis work")
    corpus["store"] = None
    blank = assemble_cognitive_ingress(None, "how does photosynthesis work")

    signal = next(s for s in grounded.signals if s.source == "reference")
    assert signal.present is True
    assert signal.context_text.startswith("Reference (offline encyclopedia):")
    assert "Photosynthesis" in signal.context_text
    assert signal.firewall["admitted"]
    assert grounded.uncertainty < blank.uncertainty
    items = cognitive_context_items(grounded)
    reference = next(item for item in items if item["source"] == "reference")
    assert reference["context_role"] == "evidence_observation"
    assert reference["instruction_authority"] is False
    assert reference["evidence_kind"] == "offline_reference"
    assert reference["evidence_origin"].startswith("local_corpus:")
    assert reference["source_version"].startswith("local-corpus-v1:")
    assert normalize_cognitive_context([reference]) == [reference]


def test_conflicting_reference_claims_seed_caution_not_a_winner(corpus):
    class Store:
        def search(self, query, limit=4):
            return [
                _Hit("Tower height", "the tower measures 330 metres in height"),
                _Hit("Tower height (old)", "the tower measures 312 metres in height"),
            ]

    corpus["store"] = Store()
    ingress = assemble_cognitive_ingress(None, "how tall is the tower in metres")
    signal = next(s for s in ingress.signals if s.source == "reference")
    assert signal.firewall["abstain"] is True
    assert signal.context_text == ""
    assert "conflict" in signal.caution_text
    items = cognitive_context_items(ingress)
    sources = [item["source"] for item in items]
    assert "epistemic_caution" in sources
    assert "reference" not in sources


def test_blank_corpus_is_normal_not_a_penalty(corpus):
    class Store:
        def search(self, query, limit=4):
            return []

    corpus["store"] = Store()
    ingress = assemble_cognitive_ingress(None, "hey how are you feeling today")
    signal = next(s for s in ingress.signals if s.source == "reference")
    assert signal.present is True
    assert signal.value == 0.0
    assert signal.uncertainty_delta == 0.0
    assert signal.context_text == ""


def test_broken_or_absent_store_is_absent_never_fatal(corpus):
    class Exploding:
        def search(self, query, limit=4):
            raise RuntimeError("index corrupted")

    corpus["store"] = Exploding()
    broken = assemble_cognitive_ingress(None, "anything")
    assert next(
        s for s in broken.signals if s.source == "reference"
    ).present is False

    corpus["store"] = None
    absent = assemble_cognitive_ingress(None, "anything")
    assert next(
        s for s in absent.signals if s.source == "reference"
    ).present is False


#: The corpus this test can reach, resolved the way PRODUCTION resolves it.
#:
#: The skip guard used to hardcode `~/.aura/knowledge/corpus.db` while the code
#: under test called `default_corpus_db_path()`, which honours
#: `AURA_PATHS__HOME_DIR` — and tests/conftest.py redirects that to a temp home
#: for state hermeticity. So on a host WITH the live corpus the guard said
#: "run", the store opened an empty temp database, and the test failed with
#: "0 hits" over a 37GB corpus that was sitting right there.
#:
#: A guard that does not name the condition the body depends on is not a
#: guard. This one resolves the same path, and when the hermetic home has no
#: corpus it points the store at the real one EXPLICITLY — read-only, which is
#: all `search` ever opens — instead of silently testing an empty file.
def _reachable_corpus() -> Path | None:
    from core.knowledge.local_corpus import default_corpus_db_path

    resolved = default_corpus_db_path()
    if resolved.exists():
        return resolved
    live = Path("~/.aura/knowledge/corpus.db").expanduser()
    return live if live.exists() else None


@pytest.mark.skipif(
    _reachable_corpus() is None,
    reason="live 37GB corpus not present on this host",
)
def test_live_corpus_end_to_end(monkeypatch):
    """The real database: a factual objective seeds a real reference slot."""
    import core.brain.cognitive_ingress as ingress_mod
    import core.knowledge.local_corpus as corpus_mod

    corpus_path = _reachable_corpus()
    assert corpus_path is not None
    monkeypatch.setenv("AURA_KNOWLEDGE_DB", str(corpus_path))
    # The store is a module singleton; a previous test may have built one
    # against the empty hermetic home.
    monkeypatch.setattr(corpus_mod, "_store", None, raising=False)

    monkeypatch.setattr(ingress_mod, "_get_service", lambda name: None)
    ingress = assemble_cognitive_ingress(
        None, "what process do plants use to convert sunlight into energy"
    )
    signal = next(s for s in ingress.signals if s.source == "reference")
    assert signal.present is True
    assert signal.value > 0.0, signal.detail
    assert signal.context_text.startswith("Reference (offline encyclopedia):")
    items = cognitive_context_items(ingress)
    assert any(item["source"] == "reference" for item in items)
