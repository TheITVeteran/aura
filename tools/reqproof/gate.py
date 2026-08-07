#!/usr/bin/env python3
"""The requirement-to-proof gate (SCOPE-001 / PROGRESS-CONTROL-001).

One command that decides, from machine state only, whether the requirement
control plane is sound and whether release is permitted. Two modes:

* ``--mode structural`` (checkpoint gate, wired into release preflight):
  - the registry must parse strictly and match the current tracker extraction;
  - the separately hashed evidence ledger must bind the exact registry;
  - the closure graph must be a DAG with no orphans or duplicates;
  - recorded evidence must verify (existing file, matching hash, known commit);
  - every corpus passage must be mapped (zero-unmapped, always);
  - ratcheted defect classes (pre-existing tracker debt such as
    unproven closures and dependency cycles) must match the checked-in
    fingerprint baseline exactly: any NEW fingerprint fails, any fixed
    fingerprint makes the baseline stale and requires an explicit
    shrink-only refresh (``--refresh-baseline``).

* ``--mode release``:
  everything structural, plus zero defects of any class and zero mandatory
  requirements outside a closed state. Release is meant to be impossible
  until the program is actually done; today this mode reports honestly that
  release is blocked and by exactly how much.

The report is deterministic (no timestamps) and written atomically to
``artifacts/reqproof/GATE_REPORT.json`` so two runs on the same state are
byte-identical.

Numbers this gate does NOT produce yet (explicit non-claims): no completion
percentage, no checkpoint forecast, no evidence weighting. Those belong to
the progress engine built on top of this substrate; until it exists the only
honest published numbers are the raw counts in the report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reqproof.coverage import check_coverage  # noqa: E402
from tools.reqproof.evidence import (  # noqa: E402
    DEFAULT_EVIDENCE_LEDGER_PATH,
    EvidenceLedgerError,
    load_evidence_ledger,
    verify_ledger_binding,
)
from tools.reqproof.migrate import (  # noqa: E402
    DEFAULT_ALLOWLIST_PATH,
    DEFAULT_REGISTRY_PATH,
    load_prose_allowlist,
)
from tools.reqproof.progress import (  # noqa: E402
    DEFAULT_SCOPE_BASELINE_PATH,
    ProgressError,
    load_scope_baseline,
    verify_scope_baseline,
)
from tools.reqproof.schema import (  # noqa: E402
    CLOSED_STATES,
    RegistrySchemaError,
    load_registry,
)
from tools.reqproof.tracker_parse import (  # noqa: E402
    TRACKER_RELPATH,
    TrackerParseError,
    parse_tracker,
)
from tools.reqproof.validate import (  # noqa: E402
    BLOCKING_ALWAYS,
    RATCHETED_CLASSES,
    Defect,
    validate_registry,
)

DEFAULT_BASELINE_PATH = ROOT / "config" / "reqproof_defect_baseline.json"
DEFAULT_REPORT_PATH = ROOT / "artifacts" / "reqproof" / "GATE_REPORT.json"
_STRUCTURAL_PROOF_ID = "reqproof-structural-gate-audit"
_STRUCTURAL_RECEIPT_PREFIX = "artifacts/reqproof/evidence/reqproof-structural-gate-audit/"


def load_defect_baseline(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or not isinstance(data.get("fingerprints"), list)
        or not all(isinstance(item, str) for item in data["fingerprints"])
    ):
        raise ValueError(
            f"defect baseline {path} must be "
            '{"schema_version": 1, "fingerprints": [...]}'
        )
    fingerprints = list(data["fingerprints"])
    if fingerprints != sorted(set(fingerprints)):
        raise ValueError(f"defect baseline {path} must be sorted and unique")
    return fingerprints


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _is_self_refreshable_stale_defect(root: Path, defect: Defect) -> bool:
    """Return true only for stale evidence emitted by this gate's own proof."""
    if defect.defect_class != "stale-evidence":
        return False
    _requirement, separator, ref = defect.subject.partition("::")
    if not separator or not ref.startswith(_STRUCTURAL_RECEIPT_PREFIX):
        return False
    target = root / ref
    try:
        resolved = target.resolve(strict=True)
        if target.is_symlink() or not resolved.is_relative_to(root.resolve()):
            return False
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(data, dict)
        and data.get("proof_id") == _STRUCTURAL_PROOF_ID
        and data.get("verdict") == "pass"
    )


def run_gate(
    *,
    root: Path,
    mode: str,
    registry_path: Path,
    tracker_path: Path,
    allowlist_path: Path,
    evidence_ledger_path: Path,
    scope_baseline_path: Path,
    baseline_path: Path,
    report_path: Path,
    refresh_baseline: bool = False,
    refresh_self_evidence: bool = False,
) -> tuple[int, dict]:
    failures: list[str] = []
    defects: list[Defect] = []
    counts: dict[str, int] = {}
    coverage_report: dict = {}
    registry = None
    evidence_ledger = None
    scope_baseline = None

    try:
        registry = load_registry(registry_path)
    except RegistrySchemaError as exc:
        failures.append(f"registry: {exc}")
    try:
        evidence_ledger = load_evidence_ledger(evidence_ledger_path)
    except EvidenceLedgerError as exc:
        failures.append(f"evidence ledger: {exc}")
    try:
        scope_baseline = load_scope_baseline(scope_baseline_path)
    except ProgressError as exc:
        failures.append(f"scope baseline: {exc}")
    try:
        extraction = parse_tracker(tracker_path)
    except TrackerParseError as exc:
        failures.append(f"tracker: {exc}")
        extraction = None

    if registry is not None and evidence_ledger is not None:
        try:
            verify_ledger_binding(evidence_ledger, registry)
        except EvidenceLedgerError as exc:
            failures.append(f"evidence ledger: {exc}")
            evidence_ledger = None
    if registry is not None and scope_baseline is not None:
        try:
            verify_scope_baseline(scope_baseline, registry)
        except ProgressError as exc:
            failures.append(f"scope baseline: {exc}")
            scope_baseline = None

    if registry is not None and extraction is not None:
        allowlist = load_prose_allowlist(allowlist_path)
        defects.extend(
            validate_registry(
                registry,
                root=root,
                extraction=extraction,
                prose_allowlist=allowlist,
                evidence_entries_by_requirement=(
                    evidence_ledger.entries_by_requirement()
                    if evidence_ledger is not None
                    else {}
                ),
            )
        )
        coverage_defects, coverage_report = check_coverage(
            root, registry_ids=set(registry.by_id())
        )
        defects.extend(coverage_defects)

    self_refresh_ignored: list[str] = []
    if refresh_self_evidence:
        if mode != "structural":
            failures.append("self-evidence refresh is structural-only")
        else:
            retained: list[Defect] = []
            for defect in defects:
                if _is_self_refreshable_stale_defect(root, defect):
                    self_refresh_ignored.append(defect.fingerprint)
                else:
                    retained.append(defect)
            defects = retained
            if not self_refresh_ignored:
                failures.append("self-evidence refresh found no stale structural receipt")

    for defect in defects:
        counts[defect.defect_class] = counts.get(defect.defect_class, 0) + 1

    blocking = [d for d in defects if d.defect_class in BLOCKING_ALWAYS]
    ratcheted = [d for d in defects if d.defect_class in RATCHETED_CLASSES]
    unknown_class = [
        d
        for d in defects
        if d.defect_class not in BLOCKING_ALWAYS
        and d.defect_class not in RATCHETED_CLASSES
    ]
    if unknown_class:
        failures.append(
            "internal: defect classes without a blocking policy: "
            + ", ".join(sorted({d.defect_class for d in unknown_class}))
        )

    for defect in blocking:
        failures.append(f"{defect.fingerprint}: {defect.detail}")

    baseline = load_defect_baseline(baseline_path)
    current_fingerprints = sorted({d.fingerprint for d in ratcheted})
    new_fingerprints = sorted(set(current_fingerprints) - set(baseline))
    fixed_fingerprints = sorted(set(baseline) - set(current_fingerprints))

    if refresh_baseline:
        if new_fingerprints and baseline_path.exists():
            # Shrink-only once a baseline exists. Initial seeding (no file
            # yet) is itself an explicit, diff-reviewed act.
            failures.append(
                "baseline refresh refused: refresh is shrink-only but new "
                f"defect fingerprints exist: {new_fingerprints[:5]}"
            )
        else:
            _atomic_write_json(
                baseline_path,
                {"schema_version": 1, "fingerprints": current_fingerprints},
            )
            baseline = current_fingerprints
            new_fingerprints = []
            fixed_fingerprints = []
    else:
        for fingerprint in new_fingerprints:
            detail = next(
                d.detail for d in ratcheted if d.fingerprint == fingerprint
            )
            failures.append(f"NEW ratcheted defect {fingerprint}: {detail}")
        if fixed_fingerprints:
            failures.append(
                "defect baseline is STALE (defects were fixed — good): run "
                "tools/reqproof/gate.py --refresh-baseline to shrink it; "
                f"fixed: {fixed_fingerprints[:5]}"
            )

    summary: dict = {}
    if registry is not None:
        states: dict[str, int] = {}
        mandatory_open = 0
        for requirement in registry.requirements:
            states[requirement.state] = states.get(requirement.state, 0) + 1
            if requirement.mandatory and requirement.state not in CLOSED_STATES:
                mandatory_open += 1
        summary = {
            "requirements": len(registry.requirements),
            "states": dict(sorted(states.items())),
            "mandatory_not_closed": mandatory_open,
            "registry_revision": registry.registry_revision,
            "registry_content_sha256": registry.compute_content_sha256(),
            "tracker_extraction_sha256": (
                registry.generated_from.tracker_extraction_sha256
            ),
            "evidence_ledger_entries": (
                len(evidence_ledger.entries) if evidence_ledger is not None else 0
            ),
            "evidence_ledger_sha256": (
                evidence_ledger.compute_content_sha256()
                if evidence_ledger is not None
                else ""
            ),
            "scope_baseline_cells": (
                len(scope_baseline.fingerprints) if scope_baseline is not None else 0
            ),
            "scope_baseline_sha256": (
                scope_baseline.compute_content_sha256()
                if scope_baseline is not None
                else ""
            ),
        }
        if mode == "release":
            if mandatory_open:
                failures.append(
                    f"release blocked: {mandatory_open} mandatory requirements "
                    "are not closed"
                )
            if defects:
                failures.append(
                    f"release blocked: {len(defects)} defects of any class must "
                    "be zero at release"
                )

    verdict = "pass" if not failures else "fail"
    report = {
        "schema_version": 1,
        "mode": mode,
        "verdict": verdict,
        "summary": summary,
        "defect_counts": dict(sorted(counts.items())),
        "defects": [d.to_dict() for d in defects],
        "ratchet": {
            "baseline_fingerprints": len(baseline),
            "current_fingerprints": len(current_fingerprints),
            "new": new_fingerprints,
            "fixed_pending_refresh": fixed_fingerprints,
        },
        "coverage": coverage_report,
        "failures": failures,
        "self_evidence_refresh": {
            "enabled": refresh_self_evidence,
            "ignored_stale_receipts": sorted(self_refresh_ignored),
            "provisional_until_replaced": bool(self_refresh_ignored),
        },
        "non_claims": [
            "No completion percentage or checkpoint forecast is published by "
            "this gate yet; only raw counts are honest at this stage.",
            "A structural pass asserts registry/coverage integrity only — it "
            "is not evidence that any requirement's engineering work is done.",
        ],
    }
    _atomic_write_json(report_path, report)
    return (0 if verdict == "pass" else 1), report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("structural", "release"), default="structural")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--tracker", default=str(ROOT / TRACKER_RELPATH))
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST_PATH))
    parser.add_argument("--evidence-ledger", default=str(DEFAULT_EVIDENCE_LEDGER_PATH))
    parser.add_argument("--scope-baseline", default=str(DEFAULT_SCOPE_BASELINE_PATH))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument(
        "--refresh-baseline",
        action="store_true",
        help="shrink-only refresh of the ratcheted defect baseline",
    )
    parser.add_argument(
        "--refresh-self-evidence",
        action="store_true",
        help=(
            "during structural proof recapture, ignore only this gate's own "
            "verified stale receipt; every other defect remains blocking"
        ),
    )
    args = parser.parse_args()
    code, report = run_gate(
        root=ROOT,
        mode=args.mode,
        registry_path=Path(args.registry),
        tracker_path=Path(args.tracker),
        allowlist_path=Path(args.allowlist),
        evidence_ledger_path=Path(args.evidence_ledger),
        scope_baseline_path=Path(args.scope_baseline),
        baseline_path=Path(args.baseline),
        report_path=Path(args.report),
        refresh_baseline=bool(args.refresh_baseline),
        refresh_self_evidence=bool(args.refresh_self_evidence),
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "mode": report["mode"],
                "summary": report["summary"],
                "defect_counts": report["defect_counts"],
                "failures": report["failures"][:20],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
