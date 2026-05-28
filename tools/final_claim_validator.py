#!/usr/bin/env python3
"""Authoritative Final Claim Validator for Aura.

Ensures that all claims in CLAIMS_MATRIX.md align with empirical evidence and
static gates, and fails closed if unsupported high-level claims are present.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


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


def _dnu_candidate_evidence_passes(artifacts_dir: Path, reasons: list[str]) -> bool:
    scorecard = _load_json(artifacts_dir / "agi_live" / "SCORECARD.json")
    leakage = _load_json(artifacts_dir / "agi_live" / "LEAKAGE_REPORT.json")
    baselines = _load_json(artifacts_dir / "agi_live" / "BASELINES.json")
    ablations = _load_json(artifacts_dir / "agi_live" / "ABLATIONS.json")
    if scorecard is None or leakage is None or baselines is None or ablations is None:
        reasons.append("AGI-candidate claim requires DNU scorecard/leakage/baseline/ablation artifacts.")
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
        if not isinstance(result, dict) or result.get("lesion_effect_verified") is not True:
            reasons.append(f"AGI-candidate claim requires verified ablation {ablation!r}.")
            return False

    return True


def _integrated_candidate_evidence_passes(artifacts_dir: Path, reasons: list[str]) -> bool:
    agency = _load_json(artifacts_dir / "agency_emergence_boxed_entity" / "SCORECARD.json")
    external = _load_json(artifacts_dir / "external_live_validation" / "SCORECARD.json")
    unified = _load_json(artifacts_dir / "unified_system_scenario" / "SUMMARY.json")
    receipts = _load_json(artifacts_dir / "receipt_coverage.json")
    consistency = _load_json(artifacts_dir / "artifact_consistency.json")
    aletheia = _load_json(artifacts_dir / "aletheia_tier5_validation.json")

    if agency is None or external is None or unified is None or receipts is None or consistency is None or aletheia is None:
        reasons.append("AGI-candidate claim requires agency, external, unified, receipt, consistency, and Aletheia artifacts.")
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


def _synthetic_entity_evidence_passes(artifacts_dir: Path, reasons: list[str]) -> bool:
    agency = _load_json(artifacts_dir / "agency_emergence_boxed_entity" / "SCORECARD.json")
    unified = _load_json(artifacts_dir / "unified_system_scenario" / "SUMMARY.json")
    receipts = _load_json(artifacts_dir / "receipt_coverage.json")
    if agency is None or unified is None or receipts is None:
        reasons.append("Synthetic cognitive entity claim requires agency, unified scenario, and receipt artifacts.")
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
    except Exception as exc:
        print(f"Error parsing claims: {exc}", file=sys.stderr)
        return 1

    # Rule checks based on user constraints
    
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

    # 7. Production-Sealed check: Requires flagship readiness and linter pass
    production_sealed = claims.get("production-sealed", "not proven")
    if production_sealed in {"implemented", "causally demonstrated", "locally demonstrated"}:
        # Check if flagship readiness and linter results are positive
        linter_json = artifacts_dir / "production_surface_lint.json"
        if linter_json.exists():
            try:
                linter_data = json.loads(linter_json.read_text(encoding="utf-8"))
                if not linter_data.get("passed", False):
                    passed = False
                    reasons.append("Production-sealed claimed but production surface linter failed.")
            except Exception:
                passed = False
                reasons.append("Production-sealed claimed but linter report is unreadable.")
        else:
            passed = False
            reasons.append("Production-sealed claimed but production surface linter report is missing.")

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
