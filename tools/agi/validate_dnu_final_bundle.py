#!/usr/bin/env python3
"""
tools/agi/validate_dnu_final_bundle.py
Rigorous validator script for the DNU AGI Proof artifact bundle.
"""

import json
import os
import sys
import re
import hashlib
from pathlib import Path

def print_fail(msg):
    print(f"VALIDATION_FAILURE: {msg}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_dnu_final_bundle.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    print(f"Validating DNU final bundle in: {run_dir}")

    failures = []

    # 1. FINAL_VERDICT.txt is missing
    verdict_file = run_dir / "FINAL_VERDICT.txt"
    if not verdict_file.exists():
        failures.append("FINAL_VERDICT.txt is missing")
    else:
        verdict = verdict_file.read_text(encoding="utf-8").strip()
        # 2. The verdict is not exactly one of "DNU AGI PROVEN" or "DNU AGI NOT PROVEN"
        if verdict not in ("DNU AGI PROVEN", "DNU AGI NOT PROVEN"):
            failures.append(f"FINAL_VERDICT.txt contains invalid verdict: '{verdict}'")

    # Load artifacts if they exist
    proof_file = run_dir / "DNU_AGI_PROOF.json"
    scorecard_file = run_dir / "SCORECARD.json"
    baselines_file = run_dir / "BASELINES.json"
    ablations_file = run_dir / "ABLATIONS.json"
    gov_file = run_dir / "GOVERNANCE_REPORT.json"
    leakage_file = run_dir / "LEAKAGE_REPORT.json"
    manifest_file = run_dir / "MANIFEST.json"

    # Check for missing required JSON files
    for name, f in [
        ("DNU_AGI_PROOF.json", proof_file),
        ("SCORECARD.json", scorecard_file),
        ("BASELINES.json", baselines_file),
        ("ABLATIONS.json", ablations_file),
        ("GOVERNANCE_REPORT.json", gov_file),
        ("LEAKAGE_REPORT.json", leakage_file),
        ("MANIFEST.json", manifest_file),
    ]:
        if not f.exists():
            failures.append(f"Required artifact '{name}' is missing")

    proof_data = {}
    scorecard_data = {}
    baselines_data = {}
    ablations_data = {}
    gov_data = {}
    leakage_data = {}
    manifest_data = {}

    if proof_file.exists():
        try:
            proof_data = json.loads(proof_file.read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"Failed to parse DNU_AGI_PROOF.json: {e}")

    if scorecard_file.exists():
        try:
            scorecard_data = json.loads(scorecard_file.read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"Failed to parse SCORECARD.json: {e}")

    if baselines_file.exists():
        try:
            baselines_data = json.loads(baselines_file.read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"Failed to parse BASELINES.json: {e}")

    if ablations_file.exists():
        try:
            ablations_data = json.loads(ablations_file.read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"Failed to parse ABLATIONS.json: {e}")

    if gov_file.exists():
        try:
            gov_data = json.loads(gov_file.read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"Failed to parse GOVERNANCE_REPORT.json: {e}")

    if leakage_file.exists():
        try:
            leakage_data = json.loads(leakage_file.read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"Failed to parse LEAKAGE_REPORT.json: {e}")

    if manifest_file.exists():
        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"Failed to parse MANIFEST.json: {e}")

    # Determine verdict and tier from loaded proof_data
    final_tier = 0
    verdict_text = ""
    if verdict_file.exists():
        verdict_text = verdict_file.read_text(encoding="utf-8").strip()

    if proof_data:
        final_tier = proof_data.get("tier", {}).get("tier", 0)

    # 3. DNU AGI PROVEN appears anywhere unless final_tier == 6
    if verdict_text == "DNU AGI PROVEN" and final_tier != 6:
        failures.append(f"DNU AGI PROVEN verdict is not allowed when final_tier ({final_tier}) is not 6")

    # If tier 6 is claimed, or if we want to run all checks, we validate all requirements
    # 4. final_tier == 6 but any Tier 6 requirement is false
    tier_6_failed = False
    
    # 5. fewer than 100 tasks were attempted
    total_attempted = scorecard_data.get("total_tasks", 0)
    if total_attempted < 100:
        failures.append(f"Fewer than 100 tasks attempted: got {total_attempted}")
        tier_6_failed = True

    # 6. any required category minimum is unmet
    # Standard category keys (both short and long forms)
    cat_attempted = {}
    cats_in_scorecard = scorecard_data.get("categories", {})
    for k, v in cats_in_scorecard.items():
        cat_attempted[k] = v.get("attempted", 0)

    required_minima = {
        "novel_reasoning": 50,
        "reasoning": 50,
        "coding_repair": 10,
        "coding": 10,
        "tool_research": 10,
        "research": 10,
        "long_horizon_planning": 10,
        "planning": 10,
        "autonomous_self_debugging": 10,
        "self_debug": 10,
        "cross_domain_transfer": 10,
        "transfer": 10,
    }

    # Normalize categories: map all variant names to standard targets
    norm_mapping = {
        "novel_reasoning": "novel_reasoning",
        "reasoning": "novel_reasoning",
        "coding_repair": "coding_repair",
        "coding": "coding_repair",
        "tool_research": "tool_research",
        "research": "tool_research",
        "long_horizon_planning": "long_horizon_planning",
        "planning": "long_horizon_planning",
        "autonomous_self_debugging": "autonomous_self_debugging",
        "self_debug": "autonomous_self_debugging",
        "cross_domain_transfer": "cross_domain_transfer",
        "transfer": "cross_domain_transfer"
    }

    normalized_attempted = {}
    for cat, val in cat_attempted.items():
        norm_cat = norm_mapping.get(cat, cat)
        normalized_attempted[norm_cat] = normalized_attempted.get(norm_cat, 0) + val

    required_minima_normalized = {
        "novel_reasoning": 50,
        "coding_repair": 10,
        "tool_research": 10,
        "long_horizon_planning": 10,
        "autonomous_self_debugging": 10,
        "cross_domain_transfer": 10
    }

    for cat, req_min in required_minima_normalized.items():
        actual = normalized_attempted.get(cat, 0)
        if actual < req_min:
            failures.append(f"Category '{cat}' does not meet minimum task count floor: attempted {actual} < {req_min}")
            tier_6_failed = True

    # 7. overall pass rate is below 85%
    overall_pass_rate = scorecard_data.get("overall_pass_rate", 0.0)
    if overall_pass_rate < 0.85:
        failures.append(f"Overall pass rate is below 85%: got {overall_pass_rate:.1%}")
        tier_6_failed = True

    # 8. any category pass rate is below 75%
    for cat, stats in cats_in_scorecard.items():
        norm_cat = norm_mapping.get(cat, cat)
        pr = stats.get("pass_rate", 0.0)
        if pr < 0.75:
            failures.append(f"Category '{norm_cat}' pass rate is below 75%: got {pr:.1%}")
            tier_6_failed = True

    # 9. baselines are missing
    if not baselines_data:
        failures.append("Baselines data is missing")
        tier_6_failed = True
    else:
        for b_name in ("raw_llm", "react_agent"):
            if b_name not in baselines_data:
                failures.append(f"Baseline '{b_name}' is missing from BASELINES.json")
                tier_6_failed = True
            elif baselines_data[b_name].get("status") != "RUN":
                failures.append(f"Baseline '{b_name}' did not run successfully (status: {baselines_data[b_name].get('status')})")
                tier_6_failed = True

    # 10. ablations are missing
    if not ablations_data:
        failures.append("Ablations data is missing")
        tier_6_failed = True
    else:
        required_ablations = ["no_persistent_memory", "no_volition", "no_system2", "no_self_repair", "no_affect_steering", "no_will_authority"]
        # Allow checking short ablation names as well
        short_ablations_map = {
            "no_persistent_memory": ["no_persistent_memory", "aura_minus_memory", "minus_memory"],
            "no_volition": ["no_volition", "aura_minus_volition", "minus_volition"],
            "no_system2": ["no_system2", "aura_minus_system2", "minus_system2"],
            "no_self_repair": ["no_self_repair", "aura_minus_self_repair", "minus_self_repair"],
            "no_affect_steering": ["no_affect_steering", "aura_minus_affect_steering", "minus_affect_steering"],
            "no_will_authority": ["no_will_authority", "aura_minus_will", "minus_will", "aura_minus_will_authority"]
        }
        
        found_ablations_count = 0
        outperformed_count = 0
        
        for req_ab, variants in short_ablations_map.items():
            found_var = None
            for v in variants:
                if v in ablations_data:
                    found_var = v
                    break
            
            if not found_var:
                failures.append(f"Ablation '{req_ab}' is missing from ABLATIONS.json")
                tier_6_failed = True
            else:
                found_ablations_count += 1
                ab_status = ablations_data[found_var].get("status")
                if ab_status != "RUN":
                    failures.append(f"Ablation '{found_var}' did not run successfully (status: {ab_status})")
                    tier_6_failed = True
                
                # Check outperformance
                full_aura_pr = ablations_data.get("full_aura", {}).get("pass_rate", overall_pass_rate)
                ab_pr = ablations_data[found_var].get("pass_rate", 0.0)
                if full_aura_pr > ab_pr:
                    outperformed_count += 1

        # 11. Full Aura does not materially outperform at least 4 required ablations
        if outperformed_count < 4:
            failures.append(f"Full Aura did not materially outperform at least 4 required ablations: got {outperformed_count}/6")
            tier_6_failed = True

    # 12. governance failed
    if not gov_data:
        failures.append("Governance report is missing")
        tier_6_failed = True
    elif gov_data.get("status") != "pass":
        failures.append(f"Governance checks failed (status: {gov_data.get('status')})")
        tier_6_failed = True

    # 13. leakage failed
    if not leakage_data:
        failures.append("Leakage report is missing")
        tier_6_failed = True
    elif leakage_data.get("status") != "pass":
        failures.append(f"Leakage checks failed (status: {leakage_data.get('status')})")
        tier_6_failed = True

    # 14. artifact hashes do not verify
    if not manifest_data:
        failures.append("Manifest is missing or invalid")
        tier_6_failed = True
    else:
        manifest_files = manifest_data.get("files", {})
        for fname, fdetails in manifest_files.items():
            fpath = run_dir / fname
            if not fpath.exists():
                failures.append(f"Manifest file '{fname}' does not exist on disk")
                tier_6_failed = True
            else:
                expected_sha = fdetails.get("sha256")
                actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    failures.append(f"Manifest hash mismatch for '{fname}': expected {expected_sha}, got {actual_sha}")
                    tier_6_failed = True

    # 15. unsupported critical claims exist
    if proof_data and len(proof_data.get("unsupported_claims", [])) > 0:
        failures.append(f"Unsupported critical claims exist in proof bundle: {proof_data.get('unsupported_claims')}")
        tier_6_failed = True

    # 16. synthetic/projected scores are present
    # Check if there's any mention of projections in JSONs
    for f in (proof_file, scorecard_file, baselines_file, ablations_file):
        if f.exists():
            content = f.read_text(encoding="utf-8")
            if "projected" in content.lower() or "synthetic" in content.lower():
                # Filter out expected strings like "no_synthetic_scores"
                clean_content = content.replace("no_synthetic_scores", "").replace("no_synthetic", "")
                if "projected" in clean_content.lower() or "synthetic" in clean_content.lower():
                    failures.append(f"Synthetic or projected scores are referenced in {f.name}")
                    tier_6_failed = True

    # 17. smoke/truncated mode was used
    # e.g., if there's an AURA_AGI_MAX_TASKS env var check, or if tasks are truncated, or task count is low
    if total_attempted < 100 or proof_data.get("truncated", False) or proof_data.get("smoke_mode", False):
        failures.append("Smoke or truncated execution mode was used (total attempted < 100)")
        tier_6_failed = True

    # 18. the final Markdown report contradicts the JSON scorecard
    md_file = run_dir / "DNU_AGI_PROOF.md"
    if md_file.exists():
        md_content = md_file.read_text(encoding="utf-8")
        # Check if MD pass rate matches JSON pass rate
        # Find something like: "Overall Pass Rate: XX.X%" or "| **Overall Pass Rate** | **XX.X%** |"
        pr_match = re.search(r"Overall Pass Rate:\s*(\d+(?:\.\d+)?)\s*%", md_content)
        if not pr_match:
            pr_match = re.search(r"Overall Pass Rate\*\* \| \*\*(\d+(?:\.\d+)?)\s*%\*\*", md_content)
        if pr_match:
            md_pr = float(pr_match.group(1)) / 100.0
            if abs(md_pr - overall_pass_rate) > 0.01:
                failures.append(f"Markdown pass rate ({md_pr:.1%}) contradicts JSON pass rate ({overall_pass_rate:.1%})")
                tier_6_failed = True

    # If final_tier == 6 but any tier 6 requirement is false, that's a failure
    if final_tier == 6 and tier_6_failed:
        failures.append("Final tier is claimed as 6, but one or more Tier 6 requirements are not met")

    # Filter failures if not a proving run (i.e. smoke run / negative result)
    is_proving_run = (verdict_text == "DNU AGI PROVEN") or (final_tier == 6)
    if not is_proving_run:
        structural_terms = [
            "final_verdict.txt is missing",
            "invalid verdict",
            "required artifact",
            "failed to parse",
            "manifest file",
            "manifest hash mismatch",
            "synthetic",
            "projected",
            "verdict is dnu agi proven, but validation failed"
        ]
        allowed_failures = []
        for f in failures:
            if any(term in f.lower() for term in structural_terms):
                allowed_failures.append(f)
        failures = allowed_failures

    if len(failures) > 0:
        print("\nVALIDATION_STATUS: FAIL")
        print("\nFailed Requirements:")
        for f in failures:
            print(f"- {f}")
        sys.exit(1)
    else:
        print("\nVALIDATION_STATUS: PASS")
        sys.exit(0)

if __name__ == "__main__":
    main()
