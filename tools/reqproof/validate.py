#!/usr/bin/env python3
"""Structural validation and defect detection for the requirement registry.

Every defect is a typed fingerprint ``(defect_class, subject)`` so the gate
can ratchet on exact identities rather than counts: a pre-existing defect is
tolerated only while it is pinned in the checked-in defect baseline, a NEW
fingerprint always fails, and a fixed fingerprint makes the baseline stale
(shrink-only refresh, mirroring tools/lint_governance.py).

Defect classes:

* ``duplicate-id``          — case-insensitive requirement ID collision.
* ``orphan-ref``            — depends_on/closure_requires/parent names a
                              requirement that does not exist.
* ``parent-mismatch``       — child declares a parent whose closure_requires
                              does not include the child.
* ``closure-cycle``         — closure_requires graph is not a DAG.
* ``dependency-cycle``      — depends_on graph is not a DAG.
* ``false-closure``         — complete requirement with a non-closed member
                              of closure_requires: a broad row may never be
                              more closed than its children.
* ``unproven-closure``      — complete requirement missing verifiable
                              evidence for a required evidence class.
* ``impossible-evidence``   — evidence ref whose target is missing, whose
                              hash mismatches, or whose commit is unknown.
* ``contradictory-status``  — complete requirement whose mandatory
                              dependency is still open.
* ``withdrawn-required``    — withdrawn requirement still required for the
                              closure of a live requirement.
* ``prose-minted``          — requirement minted from a prose-only reference
                              and still lacking a real defining row.
* ``stale-migration``       — registry was generated from a different
                              normative tracker extraction than the current
                              one.
* ``prose-only-token``      — ID-like token in the tracker that is neither a
                              requirement nor allowlisted.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from tools.reqproof.evidence import (
    EvidenceLedgerError,
    resolve_evidence_target,
    sha256_file,
)
from tools.reqproof.schema import CLOSED_STATES, EvidenceRef, Registry, Requirement
from tools.reqproof.tracker_parse import TrackerExtraction

BLOCKING_ALWAYS = frozenset(
    {
        "duplicate-id",
        "orphan-ref",
        "parent-mismatch",
        "closure-cycle",
        "impossible-evidence",
        "stale-migration",
        "prose-only-token",
        # Coverage classes (tools/reqproof/coverage.py): zero-unmapped is a
        # standing gate, not a release-time aspiration.
        "missing-corpus",
        "stale-coverage",
        "coverage-orphan-ref",
        "unmapped-passage",
    }
)

# Pre-existing tracker debt that is pinned by exact fingerprint in the defect
# baseline: a NEW fingerprint always fails the gate, a fixed one makes the
# baseline stale (shrink-only). ``dependency-cycle`` lives here because the
# master table's cross-references are genuinely mutual today; the closure
# graph (which decides what can close) must stay a hard DAG regardless.
RATCHETED_CLASSES = frozenset(
    {
        "dependency-cycle",
        "false-closure",
        "unproven-closure",
        "contradictory-status",
        "withdrawn-required",
        "prose-minted",
    }
)

DEFECT_CLASSES = tuple(sorted(BLOCKING_ALWAYS | RATCHETED_CLASSES))


@dataclass(frozen=True)
class Defect:
    defect_class: str
    subject: str
    detail: str

    @property
    def fingerprint(self) -> str:
        return f"{self.defect_class}::{self.subject}"

    def to_dict(self) -> dict[str, str]:
        return {
            "defect_class": self.defect_class,
            "subject": self.subject,
            "detail": self.detail,
        }


def _sha256_file(path: Path) -> str:
    """Compatibility wrapper retained for tests and local callers."""
    return sha256_file(path)


def default_commit_exists(root: Path) -> Callable[[str], bool]:
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    gateway = get_subprocess_gateway()

    def check(commit: str) -> bool:
        result = gateway.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            timeout=30,
            read_only=True,
            source="reqproof_evidence_commit_probe",
        )
        return result.returncode == 0

    return check


def _detect_cycles(
    ids: Iterable[str], edges: dict[str, tuple[str, ...]], defect_class: str
) -> list[Defect]:
    """Iterative three-color DFS; reports one defect per cycle found."""
    white, gray, black = 0, 1, 2
    color: dict[str, int] = {req_id: white for req_id in ids}
    defects: list[Defect] = []
    for start in sorted(color):
        if color[start] != white:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        path: list[str] = []
        while stack:
            node, edge_index = stack.pop()
            if edge_index == 0:
                color[node] = gray
                path.append(node)
            targets = tuple(t for t in edges.get(node, ()) if t in color)
            advanced = False
            for next_index in range(edge_index, len(targets)):
                target = targets[next_index]
                if color[target] == gray:
                    cycle_start = path.index(target)
                    cycle = path[cycle_start:] + [target]
                    # Canonical identity: the sorted member set, so the same
                    # cycle discovered from a different entry point (or with a
                    # rotated path) pins to one fingerprint.
                    members = "+".join(sorted(set(cycle)))
                    defects.append(
                        Defect(
                            defect_class=defect_class,
                            subject=members,
                            detail="cycle: " + " -> ".join(cycle),
                        )
                    )
                elif color[target] == white:
                    stack.append((node, next_index + 1))
                    stack.append((target, 0))
                    advanced = True
                    break
            if not advanced:
                color[node] = black
                path.pop()
    return defects


def _verify_evidence(
    requirement: Requirement,
    evidence_refs: tuple[EvidenceRef, ...],
    root: Path,
    commit_exists: Callable[[str], bool],
) -> list[Defect]:
    defects: list[Defect] = []
    for evidence in evidence_refs:
        subject = f"{requirement.id}::{evidence.ref}"
        try:
            target = resolve_evidence_target(root, evidence.ref)
        except EvidenceLedgerError as exc:
            defects.append(
                Defect(
                    defect_class="impossible-evidence",
                    subject=subject,
                    detail=str(exc),
                )
            )
            continue
        actual = _sha256_file(target)
        if actual != evidence.sha256:
            defects.append(
                Defect(
                    defect_class="impossible-evidence",
                    subject=subject,
                    detail=(
                        "evidence content hash mismatch: "
                        f"recorded {evidence.sha256[:12]}…, actual {actual[:12]}…"
                    ),
                )
            )
            continue
        if not commit_exists(evidence.commit):
            defects.append(
                Defect(
                    defect_class="impossible-evidence",
                    subject=subject,
                    detail=f"evidence commit {evidence.commit} unknown to repository",
                )
            )
    return defects


def _verified_evidence_classes(
    requirement: Requirement,
    evidence_refs: tuple[EvidenceRef, ...],
    root: Path,
    commit_exists: Callable[[str], bool],
) -> set[str]:
    verified: set[str] = set()
    for evidence in evidence_refs:
        try:
            target = resolve_evidence_target(root, evidence.ref)
        except EvidenceLedgerError:
            continue
        if _sha256_file(target) != evidence.sha256:
            continue
        if not commit_exists(evidence.commit):
            continue
        verified.add(evidence.evidence_class)
    return verified


def validate_registry(
    registry: Registry,
    *,
    root: Path,
    extraction: TrackerExtraction | None = None,
    prose_allowlist: dict[str, str] | None = None,
    commit_exists: Callable[[str], bool] | None = None,
    evidence_by_requirement: Mapping[str, tuple[EvidenceRef, ...]] | None = None,
) -> list[Defect]:
    """Run every detector; returns defects sorted by fingerprint."""
    if commit_exists is None:
        commit_exists = default_commit_exists(root)
    prose_allowlist = prose_allowlist or {}
    evidence_by_requirement = evidence_by_requirement or {}
    by_id = registry.by_id()
    defects: list[Defect] = []

    lowered: dict[str, str] = {}
    for requirement in registry.requirements:
        key = requirement.id.lower()
        if key in lowered:
            defects.append(
                Defect(
                    defect_class="duplicate-id",
                    subject=requirement.id,
                    detail=f"collides with {lowered[key]} (case-insensitive)",
                )
            )
        else:
            lowered[key] = requirement.id

    for requirement in registry.requirements:
        for field_name, refs in (
            ("depends_on", requirement.depends_on),
            ("closure_requires", requirement.closure_requires),
            ("parent", (requirement.parent,) if requirement.parent else ()),
        ):
            for ref in refs:
                if ref not in by_id:
                    defects.append(
                        Defect(
                            defect_class="orphan-ref",
                            subject=f"{requirement.id}::{ref}",
                            detail=f"{field_name} references unknown requirement {ref}",
                        )
                    )
        if requirement.parent and requirement.parent in by_id:
            parent = by_id[requirement.parent]
            if requirement.id not in parent.closure_requires:
                defects.append(
                    Defect(
                        defect_class="parent-mismatch",
                        subject=requirement.id,
                        detail=(
                            f"parent {parent.id} does not list this requirement "
                            "in closure_requires"
                        ),
                    )
                )

    all_ids = list(by_id)
    defects.extend(
        _detect_cycles(
            all_ids,
            {req.id: req.closure_requires for req in registry.requirements},
            "closure-cycle",
        )
    )
    defects.extend(
        _detect_cycles(
            all_ids,
            {req.id: req.depends_on for req in registry.requirements},
            "dependency-cycle",
        )
    )

    for requirement in registry.requirements:
        evidence_refs = tuple(requirement.evidence) + tuple(
            evidence_by_requirement.get(requirement.id, ())
        )
        defects.extend(
            _verify_evidence(requirement, evidence_refs, root, commit_exists)
        )

        if requirement.state == "complete":
            for child_id in requirement.closure_requires:
                child = by_id.get(child_id)
                if child is not None and child.state not in CLOSED_STATES:
                    defects.append(
                        Defect(
                            defect_class="false-closure",
                            subject=f"{requirement.id}::{child_id}",
                            detail=(
                                f"complete requirement requires {child_id} "
                                f"which is {child.state}"
                            ),
                        )
                    )
            verified = _verified_evidence_classes(
                requirement, evidence_refs, root, commit_exists
            )
            missing = [
                cls for cls in requirement.evidence_required if cls not in verified
            ]
            if missing:
                defects.append(
                    Defect(
                        defect_class="unproven-closure",
                        subject=requirement.id,
                        detail=(
                            "complete without verifiable evidence for: "
                            + ", ".join(missing)
                        ),
                    )
                )
            for dep_id in requirement.depends_on:
                dep = by_id.get(dep_id)
                if dep is not None and dep.mandatory and dep.state == "open":
                    defects.append(
                        Defect(
                            defect_class="contradictory-status",
                            subject=f"{requirement.id}::{dep_id}",
                            detail=(
                                f"complete requirement depends on {dep_id} "
                                "which is still open"
                            ),
                        )
                    )

        if requirement.state != "withdrawn":
            for child_id in requirement.closure_requires:
                child = by_id.get(child_id)
                if child is not None and child.state == "withdrawn":
                    defects.append(
                        Defect(
                            defect_class="withdrawn-required",
                            subject=f"{requirement.id}::{child_id}",
                            detail=(
                                f"live requirement requires withdrawn {child_id}; "
                                "either withdraw the obligation explicitly or "
                                "restore the child"
                            ),
                        )
                    )

        if requirement.notes == "minted_from_prose":
            defects.append(
                Defect(
                    defect_class="prose-minted",
                    subject=requirement.id,
                    detail="obligation exists only as a prose reference",
                )
            )

    if extraction is not None:
        current_hash = extraction.extraction_sha256()
        if registry.generated_from.tracker_extraction_sha256 != current_hash:
            defects.append(
                Defect(
                    defect_class="stale-migration",
                    subject=registry.generated_from.tracker_path,
                    detail=(
                        "registry generated from extraction "
                        f"{registry.generated_from.tracker_extraction_sha256[:12]}… "
                        f"but tracker now extracts to {current_hash[:12]}…; "
                        "rerun tools/reqproof/migrate.py --write"
                    ),
                )
            )
        known = set(by_id) | set(prose_allowlist)
        for token in extraction.all_id_tokens:
            if token not in known:
                defects.append(
                    Defect(
                        defect_class="prose-only-token",
                        subject=token,
                        detail=(
                            "tracker references an ID-like token that is neither "
                            "a registry requirement nor allowlisted in "
                            "config/reqproof_prose_token_allowlist.json"
                        ),
                    )
                )

    return sorted(defects, key=lambda d: (d.defect_class, d.subject, d.detail))
