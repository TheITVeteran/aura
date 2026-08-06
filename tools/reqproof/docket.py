#!/usr/bin/env python3
"""Generate Aura's current dependency-aware requirement docket.

The append-only execution tracker preserves history, while the requirement
registry preserves scope. This report joins the registry to the verified
evidence ledger so an operator can see what is current without treating prose,
code presence, or a historical ``complete`` label as proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reqproof.evidence import (  # noqa: E402
    DEFAULT_EVIDENCE_LEDGER_PATH,
    EvidenceLedger,
    EvidenceLedgerError,
    load_evidence_ledger,
    verify_ledger_binding,
)
from tools.reqproof.migrate import DEFAULT_REGISTRY_PATH  # noqa: E402
from tools.reqproof.schema import CLOSED_STATES, Registry, Requirement, load_registry  # noqa: E402
from tools.reqproof.validate import (  # noqa: E402
    default_commit_exists,
    verified_acceptance_coverage,
)

DOCKET_SCHEMA_VERSION = 1
DEFAULT_REPORT_PATH = ROOT / "artifacts" / "reqproof" / "DOCKET_REPORT.json"
DEFAULT_MARKDOWN_PATH = ROOT / "docs" / "AURA_REMAINING_DOCKET.md"


class DocketError(ValueError):
    """Docket inputs violated the current-scope contract."""


def _content_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _acceptance_ids(requirement: Requirement) -> tuple[str, ...]:
    return tuple(f"A{index}" for index in range(1, len(requirement.acceptance) + 1))


def _missing_cells(
    requirement: Requirement,
    coverage: dict[str, set[str]],
) -> tuple[str, ...]:
    return tuple(
        f"{evidence_class}:{acceptance_id}"
        for evidence_class in requirement.evidence_required
        for acceptance_id in _acceptance_ids(requirement)
        if acceptance_id not in coverage.get(evidence_class, set())
    )


def build_docket_report(
    *,
    root: Path,
    registry: Registry,
    ledger: EvidenceLedger,
    commit_exists: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    try:
        verify_ledger_binding(ledger, registry)
    except EvidenceLedgerError as exc:
        raise DocketError(str(exc)) from exc
    if commit_exists is None:
        commit_exists = default_commit_exists(root)

    by_id = registry.by_id()
    entries_by_requirement = ledger.entries_by_requirement()
    local_complete: dict[str, bool] = {}
    coverages: dict[str, dict[str, set[str]]] = {}
    missing_by_id: dict[str, tuple[str, ...]] = {}
    for requirement in registry.requirements:
        coverage = verified_acceptance_coverage(
            requirement,
            requirement.evidence,
            entries_by_requirement.get(requirement.id, ()),
            root,
            commit_exists,
        )
        missing = _missing_cells(requirement, coverage)
        coverages[requirement.id] = coverage
        missing_by_id[requirement.id] = missing
        local_complete[requirement.id] = not missing

    rows: list[dict[str, Any]] = []
    disposition_counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = {}
    for requirement in registry.requirements:
        direct_dependencies = tuple(
            dependency
            for dependency in requirement.depends_on
            if dependency in by_id
            and (by_id[dependency].state not in CLOSED_STATES or not local_complete[dependency])
        )
        closure_blockers = tuple(
            child
            for child in requirement.closure_requires
            if child in by_id
            and (by_id[child].state not in CLOSED_STATES or not local_complete[child])
        )
        missing = missing_by_id[requirement.id]
        if not requirement.mandatory:
            disposition = "nonmandatory"
        elif requirement.state == "complete" and missing:
            disposition = "claimed_complete_needs_evidence"
        elif requirement.state == "complete" and closure_blockers:
            disposition = "claimed_complete_needs_children"
        elif requirement.state == "complete" and direct_dependencies:
            disposition = "claimed_complete_needs_dependencies"
        elif requirement.state == "complete":
            disposition = "certified_complete"
        elif requirement.state == "in_progress":
            disposition = "in_progress"
        elif requirement.state in ("deferred", "blocked", "withdrawn"):
            disposition = requirement.state
        elif direct_dependencies:
            disposition = "blocked_by_dependency"
        else:
            disposition = "ready_for_implementation"

        disposition_counts[disposition] += 1
        family = requirement.id.split("-", 1)[0]
        family_counts.setdefault(family, Counter())[disposition] += 1
        rows.append(
            {
                "id": requirement.id,
                "title": requirement.title,
                "family": family,
                "kind": requirement.kind,
                "mandatory": requirement.mandatory,
                "state_claim": requirement.state,
                "disposition": disposition,
                "owner": requirement.owner,
                "acceptance_units": len(requirement.acceptance),
                "evidence_cells_total": len(requirement.acceptance)
                * len(requirement.evidence_required),
                "evidence_cells_verified": sum(
                    len(values) for values in coverages[requirement.id].values()
                ),
                "missing_evidence_cells": list(missing),
                "direct_dependency_blockers": list(direct_dependencies),
                "closure_blockers": list(closure_blockers),
                "parent": requirement.parent,
            }
        )

    priority = {
        "in_progress": 0,
        "claimed_complete_needs_evidence": 1,
        "claimed_complete_needs_children": 2,
        "claimed_complete_needs_dependencies": 3,
        "ready_for_implementation": 4,
        "blocked_by_dependency": 5,
        "blocked": 6,
        "deferred": 7,
        "withdrawn": 8,
        "certified_complete": 9,
        "nonmandatory": 10,
    }
    rows.sort(key=lambda row: (priority[row["disposition"]], row["id"]))
    body: dict[str, Any] = {
        "schema_version": DOCKET_SCHEMA_VERSION,
        "inputs": {
            "registry_sha256": registry.compute_content_sha256(),
            "evidence_ledger_sha256": ledger.compute_content_sha256(),
        },
        "summary": {
            "requirements_total": len(rows),
            "mandatory_requirements": sum(row["mandatory"] for row in rows),
            "dispositions": dict(sorted(disposition_counts.items())),
            "in_progress_ids": [row["id"] for row in rows if row["disposition"] == "in_progress"],
            "claimed_complete_needs_evidence_ids": [
                row["id"] for row in rows if row["disposition"] == "claimed_complete_needs_evidence"
            ],
            "ready_for_implementation_ids": [
                row["id"] for row in rows if row["disposition"] == "ready_for_implementation"
            ],
        },
        "families": [
            {"family": family, "counts": dict(sorted(counts.items()))}
            for family, counts in sorted(family_counts.items())
        ],
        "requirements": rows,
        "non_claims": [
            "A state_claim is historical tracker metadata, not certified completion.",
            "Ready for implementation means direct dependencies are not open; it does not waive closure children or evidence.",
            "The docket does not infer evidence from source presence, checkpoint prose, or test names.",
        ],
    }
    body["report_sha256"] = _content_sha256(body)
    return body


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    counts = summary["dispositions"]
    lines = [
        "# Aura Remaining Docket",
        "",
        "> Generated by `tools/reqproof/docket.py`. Do not edit by hand.",
        "",
        "## Current Truth",
        "",
        f"- Requirements: **{summary['requirements_total']}** total / **{summary['mandatory_requirements']}** mandatory",
        f"- In progress: **{counts.get('in_progress', 0)}**",
        f"- Claimed complete but missing evidence: **{counts.get('claimed_complete_needs_evidence', 0)}**",
        f"- Ready for implementation: **{counts.get('ready_for_implementation', 0)}**",
        f"- Blocked by direct dependency: **{counts.get('blocked_by_dependency', 0)}**",
        f"- Machine-certified complete: **{counts.get('certified_complete', 0)}**",
        "",
        "## Active Requirements",
        "",
        "| ID | Claim | Missing cells | Direct blockers |",
        "|---|---|---:|---|",
    ]
    for row in report["requirements"]:
        if row["disposition"] != "in_progress":
            continue
        blockers = ", ".join(row["direct_dependency_blockers"]) or "none"
        lines.append(
            f"| `{row['id']}` | `{row['state_claim']}` | "
            f"{len(row['missing_evidence_cells'])} | {blockers} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Backfill",
            "",
            "These requirements carry a historical complete claim but are not machine-certified:",
            "",
        ]
    )
    for requirement_id in summary["claimed_complete_needs_evidence_ids"]:
        lines.append(f"- `{requirement_id}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The JSON report contains every requirement, exact missing acceptance/evidence cells, direct dependency blockers, closure blockers, and family counts. Historical checkpoint prose remains audit history and cannot award evidence credit.",
            "",
            f"Report SHA-256: `{report['report_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--evidence-ledger", default=str(DEFAULT_EVIDENCE_LEDGER_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN_PATH))
    args = parser.parse_args()

    registry = load_registry(Path(args.registry))
    ledger = load_evidence_ledger(Path(args.evidence_ledger))
    report = build_docket_report(root=ROOT, registry=registry, ledger=ledger)
    _atomic_write(Path(args.report), json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(Path(args.markdown), render_markdown(report))
    print(
        json.dumps(
            {
                "dispositions": report["summary"]["dispositions"],
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
