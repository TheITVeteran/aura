"""Tracker extraction and migration tests: determinism, strictness, staleness."""
from __future__ import annotations

from pathlib import Path

import pytest
from reqproof_testkit import mini_tracker

from tools.reqproof.migrate import (
    MigrationError,
    build_registry,
    migrate,
    parse_scope_refs,
    parse_state,
)
from tools.reqproof.tracker_parse import TrackerParseError, parse_tracker

ROOT = Path(__file__).resolve().parents[1]
REAL_TRACKER = ROOT / "docs" / "AURA_EXECUTION_TRACKER.md"


class TestStrictParsing:
    def test_real_tracker_parses_with_expected_families(self):
        extraction = parse_tracker(REAL_TRACKER)
        tables = {row.table for row in extraction.table_rows}
        assert tables == {"master", "self_model", "mq", "foundation"}
        lists = {item.list_key for item in extraction.items}
        assert "passf" in lists and "matrix" in lists
        assert len(extraction.table_rows) >= 140
        assert len(extraction.declared_ids()) >= 200

    def test_missing_section_is_a_hard_error(self, tmp_path):
        broken = mini_tracker().replace("#### Pass F: Enterprise Maturity", "#### Renamed")
        path = tmp_path / "t.md"
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(TrackerParseError, match="Pass F"):
            parse_tracker(path)

    def test_broken_numbering_is_a_hard_error(self, tmp_path):
        broken = mini_tracker().replace("1. **First matrix item**", "3. **First matrix item**")
        path = tmp_path / "t.md"
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(TrackerParseError, match="numbering broke"):
            parse_tracker(path)

    def test_malformed_table_row_is_a_hard_error(self, tmp_path):
        broken = mini_tracker().replace(
            "| `BETA-001` | `OPEN` | Do the beta work. | Pass F 1 |",
            "| `BETA-001` | `OPEN` | Do the beta work. |",
        )
        path = tmp_path / "t.md"
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(TrackerParseError, match="cells"):
            parse_tracker(path)

    def test_duplicate_table_ids_are_a_hard_error(self, tmp_path):
        broken = mini_tracker().replace("`FND-01-TEST`", "`SM-01-TEST`")
        path = tmp_path / "t.md"
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(TrackerParseError, match="duplicate"):
            parse_tracker(path)

    def test_unknown_status_is_a_migration_error(self, tmp_path):
        path = tmp_path / "t.md"
        path.write_text(mini_tracker(master_status="DONE-ISH"), encoding="utf-8")
        extraction = parse_tracker(path)
        with pytest.raises(MigrationError, match="unrecognized status"):
            build_registry(extraction, allowlist={})

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("OPEN", ("open", "")),
            ("OPEN 2026-07-13", ("open", "2026-07-13")),
            ("IN PROGRESS (SOURCE GREEN; LIVE OPEN) 2026-07-14", ("in_progress", "2026-07-14")),
            ("COMPLETE 2026-07-10", ("complete", "2026-07-10")),
            ("WITHDRAWN FROM RELEASE SCOPE 2026-07-15", ("withdrawn", "2026-07-15")),
            ("DEFERRED", ("deferred", "")),
        ],
    )
    def test_state_parsing(self, raw, expected):
        assert parse_state(raw, context="test") == expected

    def test_scope_ref_expansion(self):
        assert parse_scope_refs("Matrix 1; Pass F 12-13") == [
            "MATRIX-01",
            "PASSF-12",
            "PASSF-13",
        ]
        assert parse_scope_refs("Matrix 2 and 17; Addendum 22") == [
            "MATRIX-02",
            "MATRIX-17",
            "ADDENDUM-22",
        ]
        assert parse_scope_refs("Pass F 9-10; Matrix 15-16") == [
            "PASSF-09",
            "PASSF-10",
            "MATRIX-15",
            "MATRIX-16",
        ]


class TestDeterminismAndStaleness:
    def test_same_tracker_builds_byte_identical_registry(self, tmp_path):
        path = tmp_path / "t.md"
        path.write_text(mini_tracker(), encoding="utf-8")
        extraction = parse_tracker(path)
        first = build_registry(extraction, allowlist={})
        second = build_registry(parse_tracker(path), allowlist={})
        assert first.to_canonical_json() == second.to_canonical_json()

    def test_narrative_prose_edit_does_not_invalidate_registry(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text(mini_tracker(narrative="Version one."), encoding="utf-8")
        b.write_text(mini_tracker(narrative="A totally rewritten story."), encoding="utf-8")
        assert (
            parse_tracker(a).extraction_sha256() == parse_tracker(b).extraction_sha256()
        )

    def test_status_edit_invalidates_extraction(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text(mini_tracker(master_status="OPEN"), encoding="utf-8")
        b.write_text(mini_tracker(master_status="IN PROGRESS 2026-07-16"), encoding="utf-8")
        assert (
            parse_tracker(a).extraction_sha256() != parse_tracker(b).extraction_sha256()
        )

    def test_migrate_is_idempotent_and_detects_staleness(self, tmp_path):
        tracker = tmp_path / "t.md"
        registry = tmp_path / "registry.json"
        allowlist = tmp_path / "allow.json"
        tracker.write_text(mini_tracker(), encoding="utf-8")

        first = migrate(
            tracker_path=tracker,
            registry_path=registry,
            allowlist_path=allowlist,
            write=True,
        )
        assert first["written"] and not first["registry_current"]
        second = migrate(
            tracker_path=tracker,
            registry_path=registry,
            allowlist_path=allowlist,
            write=True,
        )
        assert second["registry_current"] and not second["written"]
        assert second["content_sha256"] == first["content_sha256"]

        tracker.write_text(
            mini_tracker(master_status="IN PROGRESS 2026-07-16"), encoding="utf-8"
        )
        third = migrate(
            tracker_path=tracker,
            registry_path=registry,
            allowlist_path=allowlist,
            write=False,
        )
        assert not third["registry_current"]


class TestMigrationSemantics:
    def test_structure_parent_children_and_minting(self, tmp_path):
        path = tmp_path / "t.md"
        path.write_text(mini_tracker(), encoding="utf-8")
        registry = build_registry(parse_tracker(path), allowlist={})
        by_id = registry.by_id()

        # Master refs became closure requirements and dependencies.
        alpha = by_id["ALPHA-001"]
        assert set(alpha.closure_requires) == {"MATRIX-01", "PASSF-01"}
        assert alpha.depends_on == ("BETA-001",)
        assert alpha.kind == "parent"

        # Table children point at their structural parents bidirectionally.
        assert by_id["SM-01-TEST"].parent == "SELF-MODEL-MIRROR-001"
        assert by_id["MQ-01"].parent == "SELF-MODEL-MIRROR-001"
        assert "SM-01-TEST" in by_id["SELF-MODEL-MIRROR-001"].closure_requires
        assert by_id["FND-01-TEST"].parent == "FOUNDATION-100-001"

        # Nested units become children of their numbered item.
        assert by_id["CTX9-UNIT-001"].parent == "MATRIX-01"
        assert "CTX9-UNIT-001" in by_id["MATRIX-01"].closure_requires

    def test_complete_item_children_inherit_but_stay_unproven(self, tmp_path):
        path = tmp_path / "t.md"
        path.write_text(mini_tracker(child_status="COMPLETE 2026-07-10"), encoding="utf-8")
        registry = build_registry(parse_tracker(path), allowlist={})
        by_id = registry.by_id()
        assert by_id["MATRIX-01"].state == "complete"
        assert by_id["CTX9-UNIT-001"].state == "complete"
        # No machine evidence exists, so the validator must flag both.
        from tools.reqproof.validate import validate_registry

        defects = validate_registry(
            registry, root=tmp_path, commit_exists=lambda commit: True
        )
        unproven = {
            d.subject for d in defects if d.defect_class == "unproven-closure"
        }
        assert {"MATRIX-01", "CTX9-UNIT-001"} <= unproven

    def test_prose_only_ids_are_minted_open(self, tmp_path):
        content = mini_tracker() + "\nLater prose references `GAMMA-001` casually.\n"
        path = tmp_path / "t.md"
        path.write_text(content, encoding="utf-8")
        registry = build_registry(parse_tracker(path), allowlist={})
        gamma = registry.by_id()["GAMMA-001"]
        assert gamma.state == "open"
        assert gamma.mandatory
        assert gamma.notes == "minted_from_prose"

    def test_allowlisted_tokens_are_not_minted(self, tmp_path):
        content = mini_tracker() + "\nMentions `SHA-256-STYLE` token.\n"
        path = tmp_path / "t.md"
        path.write_text(content, encoding="utf-8")
        registry = build_registry(
            parse_tracker(path),
            allowlist={"SHA-256-STYLE": "not a requirement"},
        )
        assert "SHA-256-STYLE" not in registry.by_id()
