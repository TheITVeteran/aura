"""Contracts for the A2 FMEA registry.

The registry is only worth having if it cannot rot: every referenced
detection/mitigation module must import, the generated doc must match the
code, and gaps must be explicit and shrink-only.
"""
from __future__ import annotations

import importlib

import pytest

from core.runtime.fmea import (
    FMEA_REGISTRY,
    BlastRadius,
    FailureMode,
    Severity,
    all_referenced_modules,
    detection_gaps,
    failure_modes_for,
    mitigation_gaps,
    registry_summary,
)

pytestmark = pytest.mark.unit

# The pinned gap allowlist: adding a gap is a conscious act (extend here
# WITH a note in the entry); closing one shrinks this set forever.
KNOWN_MITIGATION_GAPS = {"FM-MEM-001"}
KNOWN_DETECTION_GAPS: set[str] = set()


class TestRegistryIntegrity:
    def test_ids_are_unique(self):
        ids = [m.id for m in FMEA_REGISTRY]
        assert len(ids) == len(set(ids))

    def test_every_entry_is_complete(self):
        for mode in FMEA_REGISTRY:
            assert isinstance(mode, FailureMode)
            for field_name in ("id", "subsystem", "mode", "cause", "effect", "detection", "mitigation"):
                assert str(getattr(mode, field_name)).strip(), f"{mode.id}.{field_name} empty"
            assert isinstance(mode.blast_radius, BlastRadius)
            assert isinstance(mode.severity, Severity)

    def test_every_entry_cites_a_real_occurrence_or_notes_analysis(self):
        """No template filler: a mode either happened (cited) or the notes
        explain why it is structurally reachable."""
        for mode in FMEA_REGISTRY:
            assert mode.occurrences or mode.notes, (
                f"{mode.id} cites no occurrence and gives no analysis note"
            )

    def test_referenced_modules_all_import(self):
        """A mitigation that was deleted or renamed must fail the build."""
        for module_path in sorted(all_referenced_modules()):
            importlib.import_module(module_path)

    def test_catastrophic_and_critical_modes_have_detection_modules(self):
        """The worst classes may not rely on prose-only detection."""
        for mode in FMEA_REGISTRY:
            if mode.severity in {Severity.CATASTROPHIC, Severity.CRITICAL}:
                if mode.detection.strip().upper() != "GAP":
                    assert mode.detection_modules, (
                        f"{mode.id} is {mode.severity} but names no detection module"
                    )


class TestGapDiscipline:
    def test_mitigation_gaps_match_the_pinned_allowlist(self):
        assert {m.id for m in mitigation_gaps()} == KNOWN_MITIGATION_GAPS

    def test_detection_gaps_match_the_pinned_allowlist(self):
        assert {m.id for m in detection_gaps()} == KNOWN_DETECTION_GAPS

    def test_gap_entries_carry_notes(self):
        for mode in [*mitigation_gaps(), *detection_gaps()]:
            assert mode.notes, f"{mode.id} is a gap but explains nothing about closing it"


class TestLookups:
    def test_failure_modes_for_subsystem(self):
        assert any(m.id == "FM-LANE-001" for m in failure_modes_for("mlx_client"))
        assert failure_modes_for("nonexistent-subsystem-xyz") == []

    def test_summary_counts(self):
        summary = registry_summary()
        assert summary["total"] == len(FMEA_REGISTRY)
        assert summary["mitigation_gaps"] == len(KNOWN_MITIGATION_GAPS)


class TestDocDrift:
    def test_generated_doc_matches_registry(self):
        """docs/FMEA.md is GENERATED; if fmea.py changes and the doc does
        not, this fails — run `make fmea-doc`."""
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root / "tools"))
        try:
            import render_fmea
        finally:
            sys.path.pop(0)

        doc_path = repo_root / "docs" / "FMEA.md"
        assert doc_path.exists(), "docs/FMEA.md missing — run `make fmea-doc`"
        assert doc_path.read_text(encoding="utf-8") == render_fmea.render()
