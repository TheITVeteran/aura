"""Local reference-knowledge substrate — offline corpus behind FTS5.

Pins: ranked retrieval with provenance, FTS-injection safety, honest
misses on empty/absent corpora, wikitext cleaning, bounded resumable
ingest accounting, and the REFERENCE lane in intentional retrieval.
"""
from __future__ import annotations

from pathlib import Path

from core.knowledge.local_corpus import LocalCorpusStore


def _seeded_store(tmp_path: Path) -> LocalCorpusStore:
    store = LocalCorpusStore(tmp_path / "corpus.db")
    store.add_documents([
        (
            "Photosynthesis",
            "Photosynthesis is the process by which green plants convert "
            "light energy into chemical energy, producing oxygen from "
            "carbon dioxide and water in their chloroplasts.",
            "wikipedia",
        ),
        (
            "Chlorophyll",
            "Chlorophyll is the green pigment in chloroplasts that absorbs "
            "light for photosynthesis, most strongly in the blue and red "
            "portions of the electromagnetic spectrum.",
            "wikipedia",
        ),
        (
            "Apollo 11",
            "Apollo 11 was the American spaceflight that first landed "
            "humans on the Moon in July 1969, commanded by Neil Armstrong.",
            "wikipedia",
        ),
    ])
    return store


class TestLocalCorpusStore:
    def test_search_returns_ranked_hits_with_provenance(self, tmp_path):
        store = _seeded_store(tmp_path)
        hits = store.search("green pigment light absorption", limit=2)
        assert hits, "seeded corpus must answer a topical query"
        assert hits[0].title == "Chlorophyll"
        assert hits[0].source == "wikipedia"
        assert "pigment" in hits[0].snippet.lower()
        payload = hits[0].to_memory_dict()
        assert payload["metadata"]["provenance"] == "local_corpus"
        assert payload["metadata"]["store"] == "reference"

    def test_and_semantics_with_or_fallback(self, tmp_path):
        store = _seeded_store(tmp_path)
        # No single document contains both topics: AND finds nothing,
        # the OR fallback still surfaces the individually relevant docs.
        hits = store.search("photosynthesis Armstrong", limit=3)
        titles = {h.title for h in hits}
        assert titles & {"Photosynthesis", "Apollo 11"}

    def test_fts_syntax_cannot_be_injected(self, tmp_path):
        store = _seeded_store(tmp_path)
        # FTS5 operators/quotes in user text must be neutralized, not raise.
        for hostile in (
            'moon" OR title:"',
            "NEAR(photosynthesis, 1)",
            "col:*  ( ) ^ \" '",
            "-",
        ):
            hits = store.search(hostile, limit=3)
            assert isinstance(hits, list)

    def test_absent_corpus_is_an_honest_miss(self, tmp_path):
        store = LocalCorpusStore(tmp_path / "never_created.db")
        assert store.search("anything", limit=3) == []
        assert store.document_count() == 0
        assert store.status()["exists"] is False

    def test_empty_query_returns_empty(self, tmp_path):
        store = _seeded_store(tmp_path)
        assert store.search("", limit=3) == []
        assert store.search("  !!  ", limit=3) == []

    def test_meta_roundtrip_and_status(self, tmp_path):
        store = _seeded_store(tmp_path)
        store.set_meta("wikipedia_pages_processed", "1234")
        assert store.get_meta("wikipedia_pages_processed") == "1234"
        status = store.status()
        assert status["documents"] == 3
        assert status["exists"] is True


class TestWikitextCleaning:
    def test_clean_strips_markup_preserves_prose(self):
        from tools.knowledge_substrate.ingest_wikipedia import clean_wikitext

        raw = (
            "{{Infobox planet|name=Mars}}\n"
            "'''Mars''' is the [[Solar System|fourth planet]] from the "
            "[[Sun]].<ref>NASA fact sheet</ref>\n"
            "== Exploration ==\n"
            "{| class=\"wikitable\"\n|rover|Curiosity\n|}\n"
            "Rovers include [https://nasa.gov Curiosity] and Perseverance."
            "<!-- hidden editorial note -->\n"
            "[[File:Mars.jpg|thumb|Mars photo]]"
        )
        cleaned = clean_wikitext(raw)
        assert "Mars is the fourth planet from the Sun." in cleaned
        assert "Exploration." in cleaned
        assert "Curiosity and Perseverance" in cleaned
        for artifact in ("{{", "[[", "<ref", "wikitable", "hidden editorial"):
            assert artifact not in cleaned

    def test_nested_templates_removed(self):
        from tools.knowledge_substrate.ingest_wikipedia import strip_templates

        assert strip_templates("a {{outer {{inner}} rest}} b").split() == ["a", "b"]


class TestBoundedIngest:
    def _write_fixture_dump(self, path: Path) -> None:
        import bz2

        pages = []
        for index in range(6):
            body = (
                f"'''Topic {index}''' is a test subject with enough prose to clear "
                "the minimum body threshold for indexing. " * 8
            )
            pages.append(
                "<page>"
                f"<title>Topic {index}</title><ns>0</ns><id>{index + 1}</id>"
                f"<revision><id>{index + 1}</id><text>{body}</text></revision>"
                "</page>"
            )
        # A redirect and a non-article page: both must be skipped.
        pages.append(
            "<page><title>Alias</title><ns>0</ns><id>90</id>"
            "<redirect title=\"Topic 0\"/>"
            "<revision><id>90</id><text>#REDIRECT [[Topic 0]]</text></revision></page>"
        )
        pages.append(
            "<page><title>Talk:Topic 0</title><ns>1</ns><id>91</id>"
            "<revision><id>91</id><text>discussion page prose</text></revision></page>"
        )
        xml = (
            '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
            + "".join(pages)
            + "</mediawiki>"
        )
        path.write_bytes(bz2.compress(xml.encode("utf-8")))

    def test_ingest_is_bounded_and_resumable(self, tmp_path):
        from tools.knowledge_substrate.ingest_wikipedia import ingest

        dump = tmp_path / "fixture.xml.bz2"
        self._write_fixture_dump(dump)
        db = tmp_path / "corpus.db"

        first = ingest(dump, db_path=db, max_pages=4)
        assert first["stop_reason"] == "max_pages"
        assert first["documents_indexed"] == 4

        second = ingest(dump, db_path=db, resume=True)
        assert second["stop_reason"] == "dump_exhausted"
        # Only the remaining articles were added; redirects/talk skipped.
        store = LocalCorpusStore(db)
        assert store.document_count() == 6
        hits = store.search("Topic 5 test subject", limit=2)
        assert hits and hits[0].title == "Topic 5"


class TestReferenceRetrievalLane:
    def test_reference_store_registers_and_serves(self, tmp_path, monkeypatch):
        import core.knowledge.local_corpus as corpus_mod
        from core.memory.intentional_retrieval import (
            IntentionalRetriever,
            MemoryStoreType,
        )

        store = _seeded_store(tmp_path)
        monkeypatch.setattr(corpus_mod, "_store", store)

        retriever = IntentionalRetriever()
        wired = retriever.wire_default_stores()
        assert MemoryStoreType.REFERENCE.value in wired

        adapter = retriever._adapters[MemoryStoreType.REFERENCE]
        results = list(adapter("who first landed humans on the moon", 3))
        assert results
        assert any("Apollo 11" in str(r.get("content", "")) for r in results)
        assert all(
            r.get("metadata", {}).get("provenance") == "local_corpus" for r in results
        )

    def test_empty_corpus_does_not_register_reference_lane(self, tmp_path, monkeypatch):
        import core.knowledge.local_corpus as corpus_mod
        from core.memory.intentional_retrieval import (
            IntentionalRetriever,
            MemoryStoreType,
        )

        monkeypatch.setattr(
            corpus_mod, "_store", LocalCorpusStore(tmp_path / "empty.db")
        )
        retriever = IntentionalRetriever()
        wired = retriever.wire_default_stores()
        assert MemoryStoreType.REFERENCE.value not in wired


class TestContinuousGrowth:
    def test_retained_document_insert_is_deduped_by_artifact(self, tmp_path):
        """The corpus accretes from verified lived research — once per
        artifact, provenance-tagged, searchable immediately."""
        store = _seeded_store(tmp_path)
        first = store.add_retained_document(
            "quantum error correction thresholds",
            "Verified research note: surface codes tolerate ~1% physical "
            "error rates; below threshold, logical error falls exponentially "
            "with code distance.",
            artifact_id="artifact-123",
        )
        again = store.add_retained_document(
            "quantum error correction thresholds",
            "duplicate content",
            artifact_id="artifact-123",
        )
        assert first is True
        assert again is False, "same artifact must not be re-inserted"
        hits = store.search("surface codes physical error threshold", limit=2)
        assert hits and hits[0].source == "web_retained"

    def test_rebuild_swap_replaces_corpus_atomically(self, tmp_path):
        """Refresh mode: a new dump ingests beside the live corpus and
        swaps in atomically only on completion."""
        import bz2
        import os

        live = tmp_path / "corpus.db"
        LocalCorpusStore(live).add_documents(
            [("Old article", "stale content from the previous snapshot " * 10,
              "wikipedia")]
        )

        body = "Fresh snapshot content about lunar geology and basalt plains. " * 8
        xml = (
            '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
            "<page><title>Lunar geology</title><ns>0</ns><id>1</id>"
            f"<revision><id>1</id><text>{body}</text></revision></page>"
            "</mediawiki>"
        )
        dump = tmp_path / "new.xml.bz2"
        dump.write_bytes(bz2.compress(xml.encode("utf-8")))

        from tools.knowledge_substrate.ingest_wikipedia import ingest

        rebuild = tmp_path / "corpus.db.rebuild.tmp"
        summary = ingest(dump, db_path=rebuild)
        assert summary["stop_reason"] == "dump_exhausted"
        os.replace(rebuild, live)

        refreshed = LocalCorpusStore(live)
        assert refreshed.document_count() == 1
        hits = refreshed.search("lunar geology basalt", limit=1)
        assert hits and hits[0].title == "Lunar geology"
        assert refreshed.search("stale content previous", limit=1)[0:0] == []
