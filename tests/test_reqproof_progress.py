"""Evidence-weighted progress and total-checkpoint forecast tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from reqproof_testkit import make_registry_dict, make_requirement, write_evidence_receipt

from tools.reqproof.evidence import EvidenceLedger, add_entry
from tools.reqproof.progress import (
    CheckpointRecord,
    ProgressError,
    ProgressPolicy,
    ScopeBaseline,
    build_progress_report,
    build_scope_migration_receipt,
    parse_checkpoint_blame,
    render_markdown,
)
from tools.reqproof.schema import EVIDENCE_CLASSES, Registry

COMMIT = "a" * 40


def policy() -> ProgressPolicy:
    return ProgressPolicy(
        class_weights={name: 2 for name in EVIDENCE_CLASSES},
        optimistic_points_per_checkpoint=8,
        conservative_points_per_checkpoint=2,
        medium_confidence_verified_basis_points=2500,
        high_confidence_verified_basis_points=7500,
        legacy_engineering_estimate_basis_points=2700,
    )


def registry(**overrides) -> Registry:
    return Registry.from_dict(make_registry_dict([make_requirement(**overrides)]))


def records() -> tuple[CheckpointRecord, ...]:
    return (
        CheckpointRecord("2026-07-20-01", "one", COMMIT, True),
        CheckpointRecord("2026-07-20-02", "two", COMMIT, True),
        CheckpointRecord("2026-07-20-03", "three", "b" * 40, False),
    )


class TestPolicy:
    def test_policy_is_hashed_and_exhaustive(self):
        current = policy()
        parsed = ProgressPolicy.from_dict(current.to_dict())
        assert parsed.compute_content_sha256() == current.compute_content_sha256()

        tampered = current.to_dict()
        tampered["class_weights"]["test"] = 99
        with pytest.raises(ProgressError, match="content hash mismatch"):
            ProgressPolicy.from_dict(tampered)

        missing = current.to_dict()
        del missing["class_weights"]["soak"]
        with pytest.raises(ProgressError, match="exhaustive"):
            ProgressPolicy.from_dict(missing, verify_hash=False)

    def test_scope_baseline_rejects_denominator_shrink_and_growth(self):
        original_registry = registry(
            acceptance=["first", "second"],
            evidence_required=["implementation", "test"],
        )
        baseline = ScopeBaseline.from_registry(original_registry)
        shrunk = registry(
            acceptance=["first"],
            evidence_required=["implementation", "test"],
        )
        with pytest.raises(ProgressError, match="denominator shrank"):
            build_progress_report(
                root=Path("."),
                registry=shrunk,
                ledger=EvidenceLedger.empty_for(shrunk),
                policy=policy(),
                scope_baseline=baseline,
                checkpoint_records=(),
                commit_exists=lambda commit: True,
            )

        grown = registry(
            acceptance=["first", "second", "third"],
            evidence_required=["implementation", "test"],
        )
        with pytest.raises(ProgressError, match="baseline is stale"):
            build_progress_report(
                root=Path("."),
                registry=grown,
                ledger=EvidenceLedger.empty_for(grown),
                policy=policy(),
                scope_baseline=baseline,
                checkpoint_records=(),
                commit_exists=lambda commit: True,
            )

    def test_scope_migration_removes_only_cross_modal_cells(self):
        corrected = registry(
            acceptance=["static", "live"],
            acceptance_evidence_required=[
                ["implementation", "test"],
                ["implementation", "test", "live"],
            ],
            evidence_required=["implementation", "test", "live"],
        )
        cartesian = ScopeBaseline(
            fingerprints=tuple(
                sorted(
                    f"TEST-001::A{index}::{class_name}"
                    for index in (1, 2)
                    for class_name in ("implementation", "test", "live")
                )
            )
        )
        baseline, receipt = build_scope_migration_receipt(
            cartesian,
            corrected,
            reason="remove cross-modality cells",
        )
        assert len(baseline.fingerprints) == 5
        assert receipt["removed_cells"] == ["TEST-001::A1::live"]
        assert receipt["invariants"]["acceptance_units_before"] == 2
        assert receipt["invariants"]["acceptance_units_after"] == 2
        assert not receipt["invariants"]["acceptance_obligation_shrink"]

        shrunk = registry(
            acceptance=["static"],
            evidence_required=["implementation", "test"],
        )
        with pytest.raises(ProgressError, match="acceptance-obligation shrink"):
            build_scope_migration_receipt(
                cartesian,
                shrunk,
                reason="invalid shrink",
            )

class TestCheckpointInventory:
    def test_blame_parser_counts_records_and_push_state(self):
        blame = (
            f"{COMMIT} 1 1 1\n"
            "\t## Checkpoint 2026-07-20-01: First\n"
            f"{'b' * 40} 2 2 1\n"
            "\tNot a checkpoint\n"
            f"{'c' * 40} 3 3 1\n"
            "\t## Checkpoint 2026-07-20-02: Second\n"
        )
        parsed = parse_checkpoint_blame(blame, {COMMIT})
        assert [record.record_id for record in parsed] == [
            "2026-07-20-01",
            "2026-07-20-02",
        ]
        assert [record.pushed for record in parsed] == [True, False]

    def test_duplicate_checkpoint_record_id_is_rejected(self):
        blame = (
            f"{COMMIT} 1 1 1\n"
            "\t## Checkpoint 2026-07-20-01: First\n"
            f"{'b' * 40} 2 2 1\n"
            "\t## Checkpoint 2026-07-20-01: Duplicate\n"
        )
        with pytest.raises(ProgressError, match="duplicate checkpoint"):
            parse_checkpoint_blame(blame, {COMMIT, "b" * 40})


class TestProgressMath:
    def test_empty_ledger_gets_no_credit_and_counts_total_records(self, tmp_path: Path):
        current_registry = registry()
        report = build_progress_report(
            root=tmp_path,
            registry=current_registry,
            ledger=EvidenceLedger.empty_for(current_registry),
            policy=policy(),
            scope_baseline=ScopeBaseline.from_registry(current_registry),
            checkpoint_records=records(),
            commit_exists=lambda commit: True,
        )
        assert report["completion"]["machine_certified_percent"] == "0.00"
        assert report["scope"]["acceptance_evidence_cells_total"] == 2
        assert report["checkpoint_inventory"] == {
            "records_in_tracker": 3,
            "pushed_checkpoint_records": 2,
            "unpushed_checkpoint_records": 1,
            "distinct_pushed_commits": 1,
            "records_on_shared_commits": 2,
            "unpushed_record_ids": ["2026-07-20-03"],
        }
        assert report["forecast"]["total_checkpoint_records_low"] == 3
        assert report["forecast"]["total_checkpoint_records_high"] == 4

    def test_partial_acceptance_evidence_gets_exact_partial_credit(self, tmp_path: Path):
        artifact = write_evidence_receipt(
            tmp_path,
            "proof.json",
            targets=[("TEST-001", "implementation", ["A1"])],
            commit=COMMIT,
        )
        current_registry = registry()
        ledger = add_entry(
            EvidenceLedger.empty_for(current_registry),
            current_registry,
            requirement_id="TEST-001",
            evidence_class="implementation",
            acceptance_ids=("A1",),
            ref="proof.json",
            commit=COMMIT,
            recorded_at="2026-07-20",
            root=tmp_path,
        )
        report = build_progress_report(
            root=tmp_path,
            registry=current_registry,
            ledger=ledger,
            policy=policy(),
            scope_baseline=ScopeBaseline.from_registry(current_registry),
            checkpoint_records=records(),
            commit_exists=lambda commit: True,
        )
        assert report["completion"]["machine_certified_percent"] == "50.00"
        assert report["scope"]["acceptance_evidence_cells_verified"] == 1
        assert report["forecast"]["confidence"] == "medium"

        artifact.write_text("mutated", encoding="utf-8")
        stale = build_progress_report(
            root=tmp_path,
            registry=current_registry,
            ledger=ledger,
            policy=policy(),
            scope_baseline=ScopeBaseline.from_registry(current_registry),
            checkpoint_records=records(),
            commit_exists=lambda commit: True,
        )
        assert stale["completion"]["machine_certified_percent"] == "0.00"

    def test_markdown_is_deterministic_and_carries_report_hash(self, tmp_path: Path):
        current_registry = registry()
        report = build_progress_report(
            root=tmp_path,
            registry=current_registry,
            ledger=EvidenceLedger.empty_for(current_registry),
            policy=policy(),
            scope_baseline=ScopeBaseline.from_registry(current_registry),
            checkpoint_records=records(),
            commit_exists=lambda commit: True,
        )
        first = render_markdown(report)
        second = render_markdown(json.loads(json.dumps(report)))
        assert first == second
        assert report["report_sha256"] in first
        assert "Machine-certified completion" in first
