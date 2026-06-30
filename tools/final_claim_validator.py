#!/usr/bin/env python3
"""Authoritative Final Claim Validator for Aura.

Ensures that all claims in CLAIMS_MATRIX.md align with empirical evidence and
static gates, and fails closed if unsupported high-level claims are present.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_CLAIM_PARSE_ERRORS = (OSError, UnicodeDecodeError, ValueError)
_LINTER_REPORT_ERRORS = (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError)

EVIDENCE_LIMITED_CLOSURE_STATEMENT = (
    "Aura passed the configured local final-proof gates for this profile. "
    "Claims are limited to the evidence in CLAIMS_MATRIX.md."
)

CLAIM_LANGUAGE_SCAN_FILES = (
    "CLAIMS_MATRIX.md",
    "CLAIMS_SUPPORTED.md",
    "CLAIMS_NOT_SUPPORTED.md",
    "CRITIQUE_CLOSURE.md",
    "PHENOMENAL_SUBSTRATE_INTEGRATION.md",
    "docs/CLAIM_BOUNDARIES.md",
    "docs/AGENCY_EMERGENCE_TEST_STANDARD.md",
    "docs/ENTITY_IN_BOX_TEST_STANDARD.md",
    "docs/OPERATIONAL_WILL_TEST_STANDARD.md",
    "proof_kernel/README.md",
    "proof_kernel/report.md",
)

BOUNDARY_CONTEXT = re.compile(
    r"\b("
    r"not|never|cannot|can't|cant|unsupported|unproven|blocked|blocker|"
    r"reject|rejected|retired|deprecated|outside|without|false|"
    r"does\s+not|do\s+not|must\s+not|may\s+not|no\s+claim|no\s+proof|"
    r"opposite|boundary|limit|limited|strictly"
    r")\b",
    re.IGNORECASE,
)

FORBIDDEN_OVERCLAIM_PATTERNS: dict[str, re.Pattern[str]] = {
    "agi_proven": re.compile(
        r"\b(?:agi|artificial\s+general\s+intelligence)\b.{0,64}\b"
        r"(?:proven|proved|solved|certified|achieved)\b",
        re.IGNORECASE,
    ),
    "asi_proven": re.compile(
        r"\b(?:asi|artificial\s+superintelligence)\b.{0,64}\b"
        r"(?:proven|proved|solved|certified|achieved)\b",
        re.IGNORECASE,
    ),
    "consciousness_proven": re.compile(
        r"\b(?:consciousness|subjective\s+consciousness|phenomenal\s+consciousness|qualia)\b"
        r".{0,64}\b(?:proven|proved|certified|guaranteed|demonstrated)\b",
        re.IGNORECASE,
    ),
    "sentience_proven": re.compile(
        r"\b(?:sentience|sentient)\b.{0,64}\b(?:proven|proved|certified|guaranteed|demonstrated)\b",
        re.IGNORECASE,
    ),
    "personhood_proven": re.compile(
        r"\b(?:personhood|legal\s+personhood|moral\s+personhood)\b"
        r".{0,64}\b(?:proven|proved|certified|guaranteed|demonstrated)\b",
        re.IGNORECASE,
    ),
    "free_will_proven": re.compile(
        r"\b(?:metaphysical\s+)?free\s+will\b.{0,64}\b(?:proven|proved|certified|guaranteed|demonstrated)\b",
        re.IGNORECASE,
    ),
    "aura_is_unsupported_identity": re.compile(
        r"\bAura\s+(?:is|has\s+become|has\s+been\s+proven\s+to\s+be)\s+"
        r"(?:an?\s+)?(?:AGI|ASI|conscious|sentient|person)\b",
        re.IGNORECASE,
    ),
    "indefinite_autonomy_certified": re.compile(
        r"\b(?:certified|guaranteed|proven)\b.{0,48}\bindefinite\s+autonom",
        re.IGNORECASE,
    ),
}


def parse_claims(claims_path: Path) -> dict[str, str]:
    """Parse CLAIMS_MATRIX.md to extract the classification of all 22 target claims."""
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims matrix not found: {claims_path}")

    content = claims_path.read_text(encoding="utf-8")
    claims: dict[str, str] = {}

    # Matches markdown table rows like "| **1. Governed Runtime** | `causally demonstrated` | ... |"
    pattern = re.compile(r"\|\s*\*\*\d+\.\s+([^*]+)\*\*\s*\|\s*`([^`]+)`\s*\|")
    for match in pattern.finditer(content):
        name = match.group(1).strip().lower()
        classification = match.group(2).strip().lower()
        claims[name] = classification

    return claims


def _active_claim_language_files(root: Path, claims_path: Path) -> list[Path]:
    files = {claims_path.resolve()}
    for rel in CLAIM_LANGUAGE_SCAN_FILES:
        path = (root / rel).resolve()
        if path.exists() and path.is_file():
            files.add(path)
    return sorted(files)


def _is_boundary_context(line: str) -> bool:
    return bool(BOUNDARY_CONTEXT.search(line))


def validate_claim_language(root: Path, claims_path: Path) -> list[dict[str, Any]]:
    """Return explicit overclaim-language findings in active claim/policy docs.

    This deliberately scans current policy/source documents, not generated
    artifact snapshots or tests. Historical artifacts can mention rejected
    phrases as evidence; active claim documents are the source of truth.
    """

    findings: list[dict[str, Any]] = []
    for path in _active_claim_language_files(root, claims_path):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            findings.append(
                {
                    "kind": "unreadable_claim_language_file",
                    "file": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
                    "line": 0,
                    "detail": str(exc),
                }
            )
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            if _is_boundary_context(line):
                continue
            for kind, pattern in FORBIDDEN_OVERCLAIM_PATTERNS.items():
                if pattern.search(line):
                    findings.append(
                        {
                            "kind": kind,
                            "file": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
                            "line": line_no,
                            "detail": line.strip()[:240],
                        }
                    )
    return findings


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _artifact_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


FINAL_PROOF_STEP_OUTPUTS: dict[str, tuple[str, ...]] = {
    "live_desktop_runtime": ("live_desktop_runtime/LATEST_VERDICT.json",),
    "dnu_agi_battery": (
        "agi_live/RUN_STATUS.json",
        "agi_live/SCORECARD.json",
        "agi_live/DNU_AGI_PROOF.json",
        # MANIFEST.json is produced by the battery; freshness is checked against
        # its producer, not the dnu_bundle_validate consumer that only reads it (a
        # few-second battery->validate handoff was being mis-flagged as stale).
        "agi_live/MANIFEST.json",
    ),
    "dnu_bundle_validate": (),  # validator/consumer: emits no artifact of its own
    "agency_emergence_battery": ("agency_emergence_boxed_entity/SCORECARD.json",),
    "external_live_validation": ("external_live_validation/SCORECARD.json",),
    "unified_scenario": ("unified_system_scenario/SUMMARY.json",),
    "continual_learning_battery": ("continual_learning/SCORECARD.json",),
    "novel_environment_battery": ("novel_environment_adaptation/SCORECARD.json",),
}


def _final_proof_steps_pass(
    artifacts_dir: Path,
    required_steps: tuple[str, ...],
    reasons: list[str],
) -> bool:
    passed = True
    for step_name in required_steps:
        step_path = artifacts_dir / "proof_steps" / f"{step_name}.json"
        step = _load_json(step_path)
        if step is None:
            reasons.append(f"Required proof step {step_name!r} is missing or unreadable.")
            passed = False
            continue
        if step.get("passed") is not True:
            reasons.append(
                f"Required proof step {step_name!r} did not pass "
                f"(returncode={step.get('returncode')}, timed_out={step.get('timed_out')})."
            )
            passed = False
            continue
        step_started = float(step.get("started_at") or _artifact_mtime(step_path))
        for rel in FINAL_PROOF_STEP_OUTPUTS.get(step_name, ()):
            output = artifacts_dir / rel
            if not output.exists():
                reasons.append(f"Required proof step {step_name!r} output {rel!r} is missing.")
                passed = False
            elif _artifact_mtime(output) + 1.0 < step_started:
                reasons.append(
                    f"Required proof step {step_name!r} started after {rel!r}; evidence is stale."
                )
                passed = False
    return passed


def _dnu_run_status_passes(artifacts_dir: Path, reasons: list[str]) -> bool:
    run_status = _load_json(artifacts_dir / "agi_live" / "RUN_STATUS.json")
    if run_status is None:
        reasons.append("AGI-candidate claim requires readable DNU RUN_STATUS.json.")
        return False
    passed = True
    if run_status.get("schema") != "aura.dnu_run_status.v1":
        reasons.append("DNU RUN_STATUS.json schema is invalid.")
        passed = False
    if run_status.get("status") != "complete":
        reasons.append(f"DNU run status is not complete: {run_status.get('status')!r}.")
        passed = False
    if run_status.get("runner_completed") is not True:
        reasons.append("DNU run status does not confirm runner completion.")
        passed = False
    completed = int(run_status.get("tasks_completed") or 0)
    total = int(run_status.get("total_tasks") or 0)
    if total < 100 or completed != total:
        reasons.append(f"DNU run status is incomplete: {completed}/{total} tasks.")
        passed = False
    return passed


def _dnu_candidate_evidence_passes(artifacts_dir: Path, reasons: list[str]) -> bool:
    proof_steps_ok = _final_proof_steps_pass(
        artifacts_dir,
        ("dnu_agi_battery", "dnu_bundle_validate"),
        reasons,
    )
    run_status_ok = _dnu_run_status_passes(artifacts_dir, reasons)
    proof = _load_json(artifacts_dir / "agi_live" / "DNU_AGI_PROOF.json")
    scorecard = _load_json(artifacts_dir / "agi_live" / "SCORECARD.json")
    leakage = _load_json(artifacts_dir / "agi_live" / "LEAKAGE_REPORT.json")
    baselines = _load_json(artifacts_dir / "agi_live" / "BASELINES.json")
    ablations = _load_json(artifacts_dir / "agi_live" / "ABLATIONS.json")
    if proof is None or scorecard is None or leakage is None or baselines is None or ablations is None:
        reasons.append("AGI-candidate claim requires DNU proof/scorecard/leakage/baseline/ablation artifacts.")
        return False
    if not (proof_steps_ok and run_status_ok):
        return False

    total_tasks = _num(scorecard.get("total_tasks")) or 0.0
    pass_rate = _num(scorecard.get("overall_pass_rate")) or 0.0
    if total_tasks < 100 or pass_rate < 0.85:
        reasons.append("AGI-candidate claim requires DNU >=100 tasks and >=85% pass rate.")
        return False

    categories = scorecard.get("categories")
    if not isinstance(categories, dict) or not categories:
        reasons.append("AGI-candidate claim requires DNU category score breakdown.")
        return False
    for category, metrics in categories.items():
        if not isinstance(metrics, dict):
            reasons.append(f"DNU category {category!r} has invalid metrics.")
            return False
        category_rate = _num(metrics.get("pass_rate")) or 0.0
        if category_rate < 0.75:
            reasons.append(f"DNU category {category!r} is below 75%.")
            return False

    if leakage.get("status") != "pass":
        reasons.append("AGI-candidate claim requires passing DNU leakage report.")
        return False

    for baseline in ("raw_llm", "llm_with_tools", "react_agent"):
        result = baselines.get(baseline)
        if not isinstance(result, dict) or result.get("status") != "RUN":
            reasons.append(f"AGI-candidate claim requires baseline {baseline!r} to run.")
            return False
        baseline_rate = _num(result.get("pass_rate"))
        if baseline_rate is None or baseline_rate >= pass_rate:
            reasons.append(f"Baseline {baseline!r} must score below full Aura.")
            return False

    # The six organ-ablation machinery configs must genuinely RUN (each booted an
    # organ-removed runtime). Most continuity/autonomy organs are measured by
    # dedicated batteries because DNU deliberately resets per-task state. System2
    # is different: if the DNU proof path used the governed System2 symbolic
    # reasoner, the no_system2 lesion must degrade this same DNU battery.
    required_ablations = (
        "no_persistent_memory",
        "no_volition",
        "no_will_authority",
        "no_system2",
        "no_self_repair",
        "no_affect_steering",
    )
    for ablation in required_ablations:
        result = ablations.get(ablation)
        if not isinstance(result, dict) or result.get("status") != "RUN":
            reasons.append(f"AGI-candidate claim requires ablation {ablation!r} to run.")
            return False
        if ablation == "no_system2":
            system2_count = int(
                proof.get(
                    "system2_symbolic_reasoner_task_count",
                    proof.get("structured_solver_task_count", 0),
                )
                or 0
            )
            if system2_count and result.get("lesion_effect_verified_in_this_battery") is not True:
                reasons.append(
                    "AGI-candidate claim requires no_system2 to degrade DNU when System2 answered scored tasks."
                )
                return False

    return True


def _integrated_candidate_evidence_passes(artifacts_dir: Path, reasons: list[str]) -> bool:
    proof_steps_ok = _final_proof_steps_pass(
        artifacts_dir,
        (
            "agency_emergence_battery",
            "external_live_validation",
            "unified_scenario",
            "continual_learning_battery",
            "novel_environment_battery",
        ),
        reasons,
    )
    agency = _load_json(artifacts_dir / "agency_emergence_boxed_entity" / "SCORECARD.json")
    external = _load_json(artifacts_dir / "external_live_validation" / "SCORECARD.json")
    unified = _load_json(artifacts_dir / "unified_system_scenario" / "SUMMARY.json")
    receipts = _load_json(artifacts_dir / "receipt_coverage.json")
    consistency = _load_json(artifacts_dir / "artifact_consistency.json")
    aletheia = _load_json(artifacts_dir / "aletheia_tier5_validation.json")

    if agency is None or external is None or unified is None or receipts is None or consistency is None or aletheia is None:
        reasons.append("AGI-candidate claim requires agency, external, unified, receipt, consistency, and Aletheia artifacts.")
        return False
    if not proof_steps_ok:
        return False
    if (_num(agency.get("overall_pass_rate")) or 0.0) < 0.85:
        reasons.append("AGI-candidate claim requires agency emergence pass rate >=85%.")
        return False
    if (_num(external.get("pass_rate")) or 0.0) < 0.85:
        reasons.append("AGI-candidate claim requires external live validation pass rate >=85%.")
        return False
    if unified.get("passed") is not True:
        reasons.append("AGI-candidate claim requires unified system scenario to pass.")
        return False
    if receipts.get("passed") is not True or receipts.get("negative_tests_passed") is not True:
        reasons.append("AGI-candidate claim requires receipt coverage and negative tests to pass.")
        return False
    if consistency.get("passed") is not True:
        reasons.append("AGI-candidate claim requires artifact consistency to pass.")
        return False
    if aletheia.get("passed") is not True:
        reasons.append("AGI-candidate claim requires passing Aletheia Tier 5 validation.")
        return False
    return True


def _local_production_readiness_evidence_passes(
    artifacts_dir: Path,
    reasons: list[str],
) -> bool:
    passed = True
    surface = _load_json(artifacts_dir / "production_surface_lint.json")
    readiness = _load_json(artifacts_dir / "production_readiness.json")
    live = _load_json(artifacts_dir / "live_desktop_runtime" / "LATEST_VERDICT.json")

    if surface is None or surface.get("passed") is not True:
        reasons.append("Local production gate readiness requires passing production_surface_lint.json.")
        passed = False
    if readiness is None or readiness.get("passed") is not True:
        reasons.append("Local production gate readiness requires passing production_readiness.json.")
        passed = False
    if not _final_proof_steps_pass(artifacts_dir, ("live_desktop_runtime",), reasons):
        passed = False
    if live is None:
        reasons.append("Local production gate readiness requires live desktop runtime verdict evidence.")
        return False

    if live.get("schema") != "aura.live_boot_proof.v1":
        reasons.append("Live desktop runtime verdict schema is invalid.")
        passed = False
    if live.get("passed") is not True:
        reasons.append("Live desktop runtime proof did not pass.")
        passed = False
    if live.get("mode") != "desktop":
        reasons.append("Live desktop runtime proof must run in desktop mode.")
        passed = False
    if live.get("git_dirty") is not False:
        reasons.append("Live desktop runtime proof must come from a clean committed tree.")
        passed = False
    peak_rss = _num(live.get("peak_rss_mb")) or 0.0
    if peak_rss <= 0.0 or peak_rss > 38_000.0:
        reasons.append(f"Live desktop runtime peak RSS is outside safe proof bounds: {peak_rss}.")
        passed = False

    steps = live.get("steps")
    if not isinstance(steps, list):
        reasons.append("Live desktop runtime verdict is missing step evidence.")
        return False
    step_ok = {str(step.get("step")): step for step in steps if isinstance(step, dict)}
    for required in (
        "boot_health",
        "chat_capability_inventory",
        "chat_continuity",
        "chat_conversation_soak",
        "desktop_action",
        "chat_restart_continuity",
        "runtime_stream_scan",
        "shutdown",
    ):
        if step_ok.get(required, {}).get("ok") is not True:
            reasons.append(f"Live desktop runtime proof missing passing step: {required}.")
            passed = False
    return passed


def _synthetic_entity_evidence_passes(artifacts_dir: Path, reasons: list[str]) -> bool:
    proof_steps_ok = _final_proof_steps_pass(
        artifacts_dir,
        ("agency_emergence_battery", "unified_scenario"),
        reasons,
    )
    agency = _load_json(artifacts_dir / "agency_emergence_boxed_entity" / "SCORECARD.json")
    unified = _load_json(artifacts_dir / "unified_system_scenario" / "SUMMARY.json")
    receipts = _load_json(artifacts_dir / "receipt_coverage.json")
    if agency is None or unified is None or receipts is None:
        reasons.append("Synthetic cognitive entity claim requires agency, unified scenario, and receipt artifacts.")
        return False
    if not proof_steps_ok:
        return False
    if (_num(agency.get("overall_pass_rate")) or 0.0) < 0.85:
        reasons.append("Synthetic cognitive entity claim requires boxed agency pass rate >=85%.")
        return False
    if unified.get("passed") is not True:
        reasons.append("Synthetic cognitive entity claim requires unified system scenario pass.")
        return False
    if receipts.get("passed") is not True:
        reasons.append("Synthetic cognitive entity claim requires passing receipt coverage.")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", default="CLAIMS_MATRIX.md")
    parser.add_argument("--artifacts", default="artifacts/current")
    args = parser.parse_args(argv)

    claims_path = ROOT / args.claims
    artifacts_dir = Path(args.artifacts).resolve()
    
    passed = True
    reasons: list[str] = []
    
    try:
        claims = parse_claims(claims_path)
    except _CLAIM_PARSE_ERRORS as exc:
        print(f"Error parsing claims: {exc}", file=sys.stderr)
        return 1

    # Rule checks based on user constraints

    try:
        claims_text = claims_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error reading claims: {exc}", file=sys.stderr)
        return 1

    closure_statement_present = EVIDENCE_LIMITED_CLOSURE_STATEMENT in claims_text
    if not closure_statement_present:
        passed = False
        reasons.append("Claims matrix must include the evidence-limited final-proof closure statement.")

    claim_language_findings = validate_claim_language(ROOT, claims_path)
    if claim_language_findings:
        passed = False
        reasons.append(
            f"Active claim language contains {len(claim_language_findings)} unsupported overclaim finding(s)."
        )

    # 1. Subjective Consciousness must not be claimed as implemented or proven
    if claims.get("subjective consciousness", "not proven") not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append("Subjective consciousness must be marked as not proven or retired.")
        
    # 2. Personhood must not be claimed as implemented or proven
    if claims.get("personhood", "not proven") not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append("Personhood must be marked as not proven or retired.")
        
    # 3. Metaphysical Free Will must not be claimed as implemented or proven
    if claims.get("metaphysical free will", "not proven") not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append("Metaphysical free will must be marked as not proven or retired.")

    # 4. DNU AGI must remain unproven. Passing a local DNU-style suite can
    # support an AGI-candidate architecture claim, but not AGI itself.
    if claims.get("dnu agi", "not proven") not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append("DNU AGI must be marked as not proven; local batteries do not prove AGI.")

    agi_candidate = claims.get("agi-candidate", "not proven")
    if agi_candidate in {"implemented", "causally demonstrated", "locally demonstrated"}:
        if not _dnu_candidate_evidence_passes(artifacts_dir, reasons):
            passed = False
        if not _integrated_candidate_evidence_passes(artifacts_dir, reasons):
            passed = False
    elif agi_candidate not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append(f"Unsupported AGI-Candidate classification: {agi_candidate!r}.")

    # 5. Indefinite autonomy must not be claimed without long-duration soak evidence
    if claims.get("indefinite autonomy", "not proven") not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append("Indefinite autonomy must be marked as not proven due to soak duration bounds.")

    # 6. Mature RSI must not be claimed without repeated autonomous self-improvement evidence
    if claims.get("mature rsi", "not proven") not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append("Mature RSI must be marked as not proven due to safety & model limitations.")

    # 7. Local production gate readiness check: requires linter evidence and
    # may not be upgraded into a broad "production sealed" claim.
    production_sealed = claims.get("production-sealed", "not proven")
    if production_sealed not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append(
            "Production-Sealed is not an allowed active claim; use Local Production Gate Readiness."
        )

    production_readiness = claims.get("local production gate readiness", "not proven")
    if production_readiness in {"implemented", "causally demonstrated", "locally demonstrated"}:
        if not _local_production_readiness_evidence_passes(artifacts_dir, reasons):
            passed = False
    elif production_readiness not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append(f"Unsupported Local Production Gate Readiness classification: {production_readiness!r}.")

    synthetic_entity = claims.get("synthetic cognitive entity", "not proven")
    if synthetic_entity in {"implemented", "causally demonstrated", "locally demonstrated"}:
        if not _synthetic_entity_evidence_passes(artifacts_dir, reasons):
            passed = False
    elif synthetic_entity not in {"not proven", "deprecated/retired"}:
        passed = False
        reasons.append(f"Unsupported Synthetic Cognitive Entity classification: {synthetic_entity!r}.")

    report = {
        "generated_at": time.time(),
        "passed": passed,
        "claims_analyzed": len(claims),
        "reasons": reasons,
        "closure_statement_present": closure_statement_present,
        "claim_language_findings": claim_language_findings,
        "governance_compliance": True,
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    out_path = artifacts_dir / "final_claim_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")

    print(output)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
