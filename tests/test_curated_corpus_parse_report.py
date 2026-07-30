"""A corpus that silently loses entries must not look like a smaller corpus.

CP126 (high), core/autonomy/curated_media_loader.py: "Malformed bullets are
silently dropped. The docstring promises a category-level marker, but
unmatched bullets and entries before the exact library heading are skipped
with no count, warning, or parse report."

The docstring described a marker that does not exist. Skipping a bad line is
the right behaviour — one typo must not empty the library — but a corpus
that quietly loses entries to a formatting change is indistinguishable from
a corpus that is genuinely that size, and the only symptom is Aura never
mentioning those films again.

The report deliberately does NOT count preamble bullets as loss. Measured
against the real corpus, all twelve skipped bullets were instructional prose
before the "# The library" heading and every actual entry parsed. Counting
those as loss would make the report cry wolf on a healthy file, and a report
that cries wolf gets muted — which is how the original silence returns.
"""
from __future__ import annotations

import pytest

from core.autonomy.curated_media_loader import (
    load_corpus,
    load_corpus_with_report,
)

CORPUS = """- an instruction bullet before the heading
- another instruction bullet
# The library
## Films
- **Solaris** — Tarkovsky — slow and strange.
- this bullet is malformed
## Channels
- **Veritasium** — https://youtube.com/veritasium — science.
"""


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.md"
    path.write_text(CORPUS, encoding="utf-8")
    return path


class TestRealLossesAreCounted:
    def test_a_malformed_library_bullet_is_counted(self, corpus):
        _items, report = load_corpus_with_report(corpus)
        assert report.unmatched == 1
        assert report.dropped == 1
        assert report.complete is False

    def test_the_failing_line_is_sampled(self, corpus):
        _items, report = load_corpus_with_report(corpus)
        assert any("malformed" in s for s in report.samples)

    def test_the_good_entries_still_load(self, corpus):
        items, _report = load_corpus_with_report(corpus)
        assert [i.title for i in items] == ["Solaris", "Veritasium"]

    def test_a_loss_records_a_degradation(self, corpus, monkeypatch):
        import core.autonomy.curated_media_loader as mod

        recorded: list = []
        monkeypatch.setattr(mod, "record_degradation", lambda *a, **k: recorded.append(a))
        load_corpus_with_report(corpus)
        assert recorded, "a dropped library entry was silent"


class TestPreambleIsNotALoss:
    def test_preamble_bullets_are_not_counted_as_dropped(self, corpus):
        _items, report = load_corpus_with_report(corpus)
        assert report.before_library_heading == 2
        assert report.dropped == 1  # the malformed one only

    def test_preamble_is_still_reported(self, corpus):
        """Not loss, but not hidden either."""
        _items, report = load_corpus_with_report(corpus)
        assert report.to_dict()["before_library_heading"] == 2

    def test_preamble_is_not_sampled(self, corpus):
        _items, report = load_corpus_with_report(corpus)
        assert not any("instruction bullet" in s for s in report.samples)


class TestTheRealCorpusIsHealthy:
    def test_every_real_library_entry_parses(self):
        from pathlib import Path

        real = Path(__file__).resolve().parents[1] / "aura/knowledge/bryan-curated-media.md"
        if not real.exists():
            pytest.skip("curated corpus not present")
        items, report = load_corpus_with_report(real)
        assert report.unmatched == 0, report.samples
        assert report.complete is True
        assert len(items) == report.parsed > 0


class TestTheLegacyApiIsUnchanged:
    def test_load_corpus_still_returns_a_list(self, corpus):
        assert [i.title for i in load_corpus(corpus)] == ["Solaris", "Veritasium"]

    def test_a_missing_file_returns_empty(self, tmp_path):
        items, report = load_corpus_with_report(tmp_path / "nope.md")
        assert items == []
        assert report.total_bullets == 0

    def test_the_report_is_serializable(self, corpus):
        payload = load_corpus_with_report(corpus)[1].to_dict()
        assert payload["schema"] == "aura.curated_corpus_parse.v1"
