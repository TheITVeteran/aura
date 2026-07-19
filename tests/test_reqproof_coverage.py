"""Corpus-coverage tests: zero-unmapped, staleness, strict map schema."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.reqproof.coverage import (
    CoverageError,
    _sha256_text,
    check_coverage,
    load_coverage_map,
    load_manifest,
    range_text,
)

CORPUS_TEXT = """First obligation line.
Second obligation line.

Fourth line after a blank.
"""


def write_fixture(
    root: Path,
    *,
    entries: list[dict] | None = None,
    corpus_text: str = CORPUS_TEXT,
) -> None:
    sources = root / "config" / "requirement_sources"
    sources.mkdir(parents=True, exist_ok=True)
    (sources / "MINI.txt").write_text(corpus_text, encoding="utf-8")
    (sources / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpora": {
                    "mini": {
                        "snapshot": "config/requirement_sources/MINI.txt",
                        "original_path": "/nowhere/mini.txt",
                        "original_sha256": "0" * 64,
                        "description": "test corpus",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    if entries is None:
        lines = corpus_text.splitlines()
        entries = [
            {
                "corpus": "mini",
                "lines": "1-2",
                "sha256": _sha256_text(range_text(lines, 1, 2)),
                "class": "normative",
                "requirements": ["REQ-001"],
            },
            {
                "corpus": "mini",
                "lines": "4-4",
                "sha256": _sha256_text(range_text(lines, 4, 4)),
                "class": "rationale",
                "requirements": [],
                "reason": "explanatory only",
            },
        ]
    (root / "config" / "requirement_coverage_map.json").write_text(
        json.dumps({"schema_version": 1, "entries": entries}), encoding="utf-8"
    )


class TestHappyPath:
    def test_full_coverage_has_no_defects(self, tmp_path):
        write_fixture(tmp_path)
        defects, report = check_coverage(tmp_path, registry_ids={"REQ-001"})
        assert defects == []
        assert report["unmapped_lines"] == 0
        assert report["entries_by_class"]["normative"] == 1

    def test_blank_lines_do_not_require_mapping(self, tmp_path):
        write_fixture(tmp_path)
        defects, _ = check_coverage(tmp_path, registry_ids={"REQ-001"})
        assert not [d for d in defects if d.defect_class == "unmapped-passage"]


class TestDefects:
    def test_unmapped_passage_detected_with_span(self, tmp_path):
        lines = CORPUS_TEXT.splitlines()
        entries = [
            {
                "corpus": "mini",
                "lines": "1-1",
                "sha256": _sha256_text(range_text(lines, 1, 1)),
                "class": "normative",
                "requirements": ["REQ-001"],
            }
        ]
        write_fixture(tmp_path, entries=entries)
        defects, report = check_coverage(tmp_path, registry_ids={"REQ-001"})
        unmapped = [d for d in defects if d.defect_class == "unmapped-passage"]
        assert {d.subject for d in unmapped} == {"mini:L2", "mini:L4"}
        assert report["unmapped_lines"] == 2

    def test_editing_corpus_under_map_is_stale_coverage(self, tmp_path):
        write_fixture(tmp_path)
        sources = tmp_path / "config" / "requirement_sources"
        (sources / "MINI.txt").write_text(
            CORPUS_TEXT.replace("First obligation", "Rewritten obligation"),
            encoding="utf-8",
        )
        defects, _ = check_coverage(tmp_path, registry_ids={"REQ-001"})
        assert "stale-coverage" in {d.defect_class for d in defects}

    def test_range_beyond_corpus_is_stale_coverage(self, tmp_path):
        entries = [
            {
                "corpus": "mini",
                "lines": "1-99",
                "sha256": "0" * 64,
                "class": "normative",
                "requirements": ["REQ-001"],
            }
        ]
        write_fixture(tmp_path, entries=entries)
        defects, _ = check_coverage(tmp_path, registry_ids={"REQ-001"})
        assert "stale-coverage" in {d.defect_class for d in defects}

    def test_unknown_requirement_reference_is_orphan(self, tmp_path):
        write_fixture(tmp_path)
        defects, _ = check_coverage(tmp_path, registry_ids={"OTHER-999"})
        assert "coverage-orphan-ref" in {d.defect_class for d in defects}

    def test_missing_snapshot_is_reported(self, tmp_path):
        write_fixture(tmp_path)
        (tmp_path / "config" / "requirement_sources" / "MINI.txt").unlink()
        defects, _ = check_coverage(tmp_path, registry_ids={"REQ-001"})
        assert "missing-corpus" in {d.defect_class for d in defects}


class TestStrictMapSchema:
    def test_normative_without_requirements_rejected(self, tmp_path):
        entries = [
            {
                "corpus": "mini",
                "lines": "1-2",
                "sha256": "0" * 64,
                "class": "normative",
                "requirements": [],
            }
        ]
        write_fixture(tmp_path, entries=entries)
        with pytest.raises(CoverageError, match="maps to no requirements"):
            load_coverage_map(tmp_path)

    def test_rationale_without_reason_rejected(self, tmp_path):
        entries = [
            {
                "corpus": "mini",
                "lines": "1-2",
                "sha256": "0" * 64,
                "class": "rationale",
                "requirements": [],
            }
        ]
        write_fixture(tmp_path, entries=entries)
        with pytest.raises(CoverageError, match="must record a reason"):
            load_coverage_map(tmp_path)

    def test_unknown_entry_class_rejected(self, tmp_path):
        entries = [
            {
                "corpus": "mini",
                "lines": "1-2",
                "sha256": "0" * 64,
                "class": "skipped",
                "requirements": [],
            }
        ]
        write_fixture(tmp_path, entries=entries)
        with pytest.raises(CoverageError, match="class"):
            load_coverage_map(tmp_path)

    def test_unknown_entry_field_rejected(self, tmp_path):
        entries = [
            {
                "corpus": "mini",
                "lines": "1-2",
                "sha256": "0" * 64,
                "class": "normative",
                "requirements": ["REQ-001"],
                "confidence": "high",
            }
        ]
        write_fixture(tmp_path, entries=entries)
        with pytest.raises(CoverageError, match="unknown fields"):
            load_coverage_map(tmp_path)

    def test_invalid_line_range_rejected(self, tmp_path):
        entries = [
            {
                "corpus": "mini",
                "lines": "9-2",
                "sha256": "0" * 64,
                "class": "normative",
                "requirements": ["REQ-001"],
            }
        ]
        write_fixture(tmp_path, entries=entries)
        with pytest.raises(CoverageError, match="invalid"):
            load_coverage_map(tmp_path)

    def test_manifest_unknown_field_rejected(self, tmp_path):
        write_fixture(tmp_path)
        manifest_path = tmp_path / "config" / "requirement_sources" / "MANIFEST.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["corpora"]["mini"]["trust_me"] = True
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CoverageError, match="unknown fields"):
            load_manifest(tmp_path)


class TestRealRepositoryCoverage:
    """The checked-in corpora, manifest, and map must actually verify."""

    ROOT = Path(__file__).resolve().parents[1]

    def test_manifest_and_snapshots_exist(self):
        corpora = load_manifest(self.ROOT)
        assert set(corpora) == {
            "anima-rationis",
            "capabilities-pdf",
            "context-prompt",
            "second-criticism",
        }
        for corpus in corpora.values():
            assert (self.ROOT / corpus.snapshot).is_file(), corpus.snapshot

    def test_checked_in_map_is_hash_current_and_complete(self):
        from tools.reqproof.schema import load_registry

        registry = load_registry(self.ROOT / "config" / "requirement_registry.json")
        defects, report = check_coverage(
            self.ROOT, registry_ids=set(registry.by_id())
        )
        stale = [d for d in defects if d.defect_class == "stale-coverage"]
        unmapped = [d for d in defects if d.defect_class == "unmapped-passage"]
        assert stale == [], stale[:5]
        assert unmapped == [], unmapped[:5]
        assert report["unmapped_lines"] == 0
