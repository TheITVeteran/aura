"""Current dependency-aware requirement docket tests."""

from __future__ import annotations

from pathlib import Path

from reqproof_testkit import make_registry_dict, make_requirement, write_evidence_receipt

from tools.reqproof.docket import build_docket_report, render_markdown
from tools.reqproof.evidence import EvidenceLedger, add_entry
from tools.reqproof.schema import Registry

COMMIT = "a" * 40


def registry(*requirements: dict) -> Registry:
    return Registry.from_dict(make_registry_dict(list(requirements)))


def test_docket_separates_claims_evidence_and_dependency_readiness(tmp_path: Path):
    current = registry(
        make_requirement(id="ACTIVE-001", state="in_progress"),
        make_requirement(id="BLOCKED-001", depends_on=["OPEN-001"]),
        make_requirement(id="DONE-001", state="complete"),
        make_requirement(id="OPEN-001"),
        make_requirement(id="READY-001"),
    )
    report = build_docket_report(
        root=tmp_path,
        registry=current,
        ledger=EvidenceLedger.empty_for(current),
        commit_exists=lambda commit: True,
    )
    rows = {row["id"]: row for row in report["requirements"]}

    assert rows["ACTIVE-001"]["disposition"] == "in_progress"
    assert rows["DONE-001"]["disposition"] == "claimed_complete_needs_evidence"
    assert rows["READY-001"]["disposition"] == "ready_for_implementation"
    assert rows["BLOCKED-001"]["disposition"] == "blocked_by_dependency"
    assert rows["BLOCKED-001"]["direct_dependency_blockers"] == ["OPEN-001"]
    assert rows["DONE-001"]["evidence_cells_verified"] == 0


def test_exact_evidence_can_certify_without_reclassifying_open_work(tmp_path: Path):
    write_evidence_receipt(
        tmp_path,
        "proof.json",
        targets=[("DONE-001", "implementation", ["A1"])],
        commit=COMMIT,
    )
    current = registry(
        make_requirement(
            id="DONE-001",
            state="complete",
            evidence_required=["implementation"],
        ),
        make_requirement(
            id="OPEN-001",
            evidence_required=["implementation"],
        ),
    )
    ledger = EvidenceLedger.empty_for(current)
    ledger = add_entry(
        ledger,
        current,
        requirement_id="DONE-001",
        evidence_class="implementation",
        acceptance_ids=("A1",),
        ref="proof.json",
        commit=COMMIT,
        recorded_at="2026-08-05",
        root=tmp_path,
    )
    report = build_docket_report(
        root=tmp_path,
        registry=current,
        ledger=ledger,
        commit_exists=lambda commit: True,
    )
    rows = {row["id"]: row for row in report["requirements"]}

    assert rows["DONE-001"]["disposition"] == "certified_complete"
    assert rows["OPEN-001"]["disposition"] == "ready_for_implementation"
    assert report["summary"]["dispositions"]["certified_complete"] == 1


def test_report_and_markdown_are_deterministic(tmp_path: Path):
    current = registry(make_requirement(id="ACTIVE-001", state="in_progress"))
    ledger = EvidenceLedger.empty_for(current)
    first = build_docket_report(
        root=tmp_path,
        registry=current,
        ledger=ledger,
        commit_exists=lambda commit: True,
    )
    second = build_docket_report(
        root=tmp_path,
        registry=current,
        ledger=ledger,
        commit_exists=lambda commit: True,
    )

    assert first == second
    assert render_markdown(first) == render_markdown(second)
    assert first["report_sha256"] in render_markdown(first)


def test_certified_local_evidence_does_not_hide_graph_blockers(tmp_path: Path):
    write_evidence_receipt(
        tmp_path,
        "proof.json",
        targets=[("PARENT-001", "implementation", ["A1"])],
        commit=COMMIT,
    )
    current = registry(
        make_requirement(id="CHILD-001"),
        make_requirement(id="DEP-001"),
        make_requirement(
            id="PARENT-001",
            state="complete",
            kind="parent",
            closure_requires=["CHILD-001"],
            depends_on=["DEP-001"],
            evidence_required=["implementation"],
        ),
    )
    ledger = add_entry(
        EvidenceLedger.empty_for(current),
        current,
        requirement_id="PARENT-001",
        evidence_class="implementation",
        acceptance_ids=("A1",),
        ref="proof.json",
        commit=COMMIT,
        recorded_at="2026-08-05",
        root=tmp_path,
    )
    report = build_docket_report(
        root=tmp_path,
        registry=current,
        ledger=ledger,
        commit_exists=lambda commit: True,
    )
    parent = next(row for row in report["requirements"] if row["id"] == "PARENT-001")

    assert parent["disposition"] == "claimed_complete_needs_children"
    assert parent["closure_blockers"] == ["CHILD-001"]
    assert parent["direct_dependency_blockers"] == ["DEP-001"]


def test_mandatory_withdrawn_requirement_is_not_returned_to_work_queue(tmp_path: Path):
    current = registry(make_requirement(id="OLD-001", state="withdrawn"))
    report = build_docket_report(
        root=tmp_path,
        registry=current,
        ledger=EvidenceLedger.empty_for(current),
        commit_exists=lambda commit: True,
    )

    assert report["requirements"][0]["disposition"] == "withdrawn"
