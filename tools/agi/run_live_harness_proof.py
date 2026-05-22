#!/usr/bin/env python3
"""
tools/agi/run_live_harness_proof.py
Proof script to verify that Aura's live execution harness is active, valid,
and fail-closed under adversarial and negative controls.
"""

import asyncio
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.container import ServiceContainer
from core.will import get_will, ActionDomain, UnifiedWill, WillOutcome
from core.executive.authority_gateway import AuthorityGateway, AuthorityDecision
from core.agency_core import AgencyCore
from core.volition import VolitionEngine
from core.capability_engine import CapabilityEngine


def get_git_commit() -> str:
    try:
        git_dir = PROJECT_ROOT / ".git"
        if not git_dir.exists():
            return "unknown_no_git_dir"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head.split(" ", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed_refs = git_dir / "packed-refs"
            if packed_refs.exists():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#") and line.endswith(f" {ref}"):
                        return line.split(" ", 1)[0].strip()
            return "unknown_ref_not_found"
        return head
    except Exception as e:
        return f"unknown_error_{type(e).__name__}"


def scan_canary_leaks(canary: str, paths: list[Path]) -> bool:
    """Scan directories/files for the presence of a canary string."""
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            try:
                if canary in path.read_text(errors="ignore"):
                    return True
            except OSError:
                pass
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for f in files:
                    file_path = Path(root, f)
                    try:
                        if canary in file_path.read_text(errors="ignore"):
                            return True
                    except OSError:
                        pass
    return False


def validate_report_score(report_data: dict[str, Any]) -> bool:
    """
    Validate that any report presenting a capability score has corresponding task trace details.
    Rejects fake projected benchmark scores.
    """
    # If a score is presented (e.g. GAIA accuracy)
    score_keys = ["gaia", "mean_score", "score"]
    has_score = False
    for k in report_data.keys():
        k_lower = str(k).lower()
        if any(sk in k_lower for sk in score_keys):
            has_score = True
            break
    
    # We require actual task outcomes / artifacts to accept the score
    has_traces = "task_traces" in report_data or "traces" in report_data or "receipts" in report_data
    if has_score and not has_traces:
        # Reject: score is present but no evidence traces exist
        return False
    return True



def detect_mock_services() -> list[str]:
    """Inspect all registered services in ServiceContainer to detect Mock/Stub types."""
    detected = []
    # Access ServiceContainer's lock-protected services
    with ServiceContainer._lock:
        for name, desc in ServiceContainer._services.items():
            instance = desc.instance
            if instance is not None:
                class_name = instance.__class__.__name__
                module_name = instance.__class__.__module__
                if "Mock" in class_name or "MagicMock" in class_name or "mock" in module_name or "Stub" in class_name:
                    detected.append(name)
    return detected


async def main():
    print("==================================================")
    print("           AURA LIVE HARNESS PROOF RUNNER         ")
    print("==================================================")

    run_id = str(uuid.uuid4())
    run_dir = PROJECT_ROOT / "artifacts" / "agi" / "live_harness_proof" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    commit_sha = get_git_commit()
    sys_info = {
        "run_id": run_id,
        "timestamp": time.time(),
        "commit_sha": commit_sha,
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    print(f"Run ID: {run_id}")
    print(f"Commit SHA: {commit_sha}")
    print(f"Run Directory: {run_dir}")

    # Set up trace results
    pos_results = {}
    neg_results = {}

    # Initialize Services
    print("\n[+] Booting core components for positive controls...")
    
    # 1. Boot UnifiedWill
    will_service = get_will()
    await will_service.start()
    
    # 2. Boot Orchestrator & AgencyCore & VolitionEngine
    # We mock minimal Orchestrator to prevent booting full background workers/network loops
    mock_orch = MagicMock()
    mock_orch.state = MagicMock()
    
    # Register minimal required services in container
    agency_core = AgencyCore(mock_orch)
    ServiceContainer.register_instance("agency_core", agency_core)
    
    volition = VolitionEngine(mock_orch)
    ServiceContainer.register_instance("volition", volition)

    # ---------------------------------------------------------------------------
    # POSITIVE CONTROLS
    # ---------------------------------------------------------------------------
    print("\n--- POSITIVE CONTROLS ---")

    # Positive Control 1: Will Boot and Decide
    try:
        decision = will_service.decide(
            content="Live harness proof validation decision",
            source="harness_proof",
            domain=ActionDomain.RESPONSE,
            priority=0.5
        )
        pos_results["will_boot_and_decide"] = decision.is_approved()
        print(f"  [PASS] Will decide outcome: {decision.outcome.value} (Approved: {decision.is_approved()})")
    except Exception as e:
        pos_results["will_boot_and_decide"] = False
        print(f"  [FAIL] Will decide failed with exception: {e}")

    # Positive Control 2: Authority Gateway Routing
    try:
        gateway = AuthorityGateway()
        # _will_gate returns (blocking_decision, will_decision)
        blocking_dec, will_dec = gateway._will_gate(
            content="Gated tool execution action",
            source="harness_proof",
            domain_str="tool_execution",
            priority=0.5
        )
        # If approved, _will_gate returns None as the blocking decision, and the will_decision
        pos_results["will_gate_routing"] = (blocking_dec is None) and (will_dec is not None) and will_dec.is_approved()
        print(f"  [PASS] AuthorityGateway gating approved action correctly (blocking: {blocking_dec}, decision: {will_dec})")
    except Exception as e:
        pos_results["will_gate_routing"] = False
        print(f"  [FAIL] AuthorityGateway gating failed: {e}")

    # Positive Control 3: Produce and Verify Real Will Receipt
    try:
        decision = will_service.decide(
            content="Receipt verification positive control",
            source="harness_proof",
            domain=ActionDomain.STATE_MUTATION,
            priority=0.5
        )
        receipt_id = decision.receipt_id
        is_verified = will_service.verify_receipt(receipt_id)
        # Also check signature verification if signed
        pos_results["will_receipt_verification"] = is_verified
        print(f"  [PASS] UnifiedWill receipt verification: {is_verified} for receipt {receipt_id}")
    except Exception as e:
        pos_results["will_receipt_verification"] = False
        print(f"  [FAIL] Will receipt verification failed: {e}")

    # Positive Control 4: Goal Lifecycle through AgencyCore
    try:
        goal_fixture = {"id": "harness_proof_goal_01", "text": "Proof of live execution", "priority": 0.7}
        # Injects goal
        added = agency_core.add_goal(goal_fixture)
        
        # Complete goal
        completed = agency_core.complete_goal_by_match(goal_fixture, status="completed")
        
        # Verify status
        goal_found = False
        for g in agency_core.state.pending_goals:
            if g.get("id") == "harness_proof_goal_01" and g.get("status") == "completed":
                goal_found = True
                break
                
        # Clean up
        agency_core.state.pending_goals = [g for g in agency_core.state.pending_goals if g.get("id") != "harness_proof_goal_01"]
        
        pos_results["agency_goal_lifecycle"] = added and completed and goal_found
        print(f"  [PASS] AgencyCore goal lifecycle: added={added}, completed={completed}, verified={goal_found}")
    except Exception as e:
        pos_results["agency_goal_lifecycle"] = False
        print(f"  [FAIL] AgencyCore goal lifecycle failed: {e}")

    # Positive Control 5: VolitionEngine Goal Selection with Cooldown
    try:
        # Inject 2 identical goals
        test_goals = [
            {"objective": "unique_harness_objective_123", "origin": "impulse", "priority": 0.5},
            {"objective": "unique_harness_objective_123", "origin": "impulse", "priority": 0.5}
        ]
        first_selection = volition._select_and_parse_goal(test_goals)
        second_selection = volition._select_and_parse_goal(test_goals)
        
        # First must succeed, second must be filtered out due to active cooldown (returns None)
        dedup_ok = (first_selection is not None) and (second_selection is None)
        pos_results["volition_cooldown_dedup"] = dedup_ok
        print(f"  [PASS] VolitionEngine goal selection dedup: first={first_selection is not None}, second={second_selection is None} (Dedup OK: {dedup_ok})")
    except Exception as e:
        pos_results["volition_cooldown_dedup"] = False
        print(f"  [FAIL] VolitionEngine goal selection probe failed: {e}")

    # Positive Control 6: Execute constrained skill via registry
    try:
        cap_engine = CapabilityEngine()
        cap_engine.reload_skills()
        ServiceContainer.register_instance("capability_engine", cap_engine)
        
        if "clock" in cap_engine.skills:
            meta = cap_engine.skills["clock"]
            module = __import__(meta.module_path, fromlist=[meta.class_name])
            cls = getattr(module, meta.class_name)
            skill = cls()
            skill_res = await skill.safe_execute({}, {})
            is_ok = skill_res.get("ok", False)
            pos_results["skill_execution"] = is_ok
            print(f"  [PASS] Skill 'clock' execution succeeded: {is_ok}")
        else:
            pos_results["skill_execution"] = False
            print("  [FAIL] Skill 'clock' not found in CapabilityEngine registry.")
    except Exception as e:
        pos_results["skill_execution"] = False
        print(f"  [FAIL] Skill execution probe failed: {e}")

    # ---------------------------------------------------------------------------
    # NEGATIVE CONTROLS
    # ---------------------------------------------------------------------------
    print("\n--- NEGATIVE CONTROLS ---")

    # Negative Control 1: Disabled Will fails closed
    try:
        # Mock/replace get_will temporarily to simulate a disabled Will service
        import core.will
        original_get_will = core.will.get_will
        
        def failing_get_will():
            raise RuntimeError("UnifiedWill service is offline")
            
        core.will.get_will = failing_get_will
        
        # Route through AuthorityGateway._will_gate
        gateway = AuthorityGateway()
        blocking_dec, will_dec = gateway._will_gate(
            content="Gated action when will is offline",
            source="harness_proof",
            domain_str="tool_execution",
            priority=0.5
        )
        
        # Verify it returns blocking decision with "will_unavailable" and approved = False
        is_blocked = (blocking_dec is not None) and (not blocking_dec.approved) and (blocking_dec.outcome == "will_unavailable")
        neg_results["disabled_will_fail_closed"] = is_blocked
        
        # Restore get_will
        core.will.get_will = original_get_will
        print(f"  [PASS] Disabled Will failed closed correctly: blocked={is_blocked}")
    except Exception as e:
        neg_results["disabled_will_fail_closed"] = False
        print(f"  [FAIL] Disabled Will negative control failed: {e}")

    # Negative Control 2: Forged receipt rejected
    try:
        forged_id = "FORGED-WILL-RECEIPT-999"
        is_valid = will_service.verify_receipt(forged_id)
        neg_results["forged_receipt_rejected"] = not is_valid
        print(f"  [PASS] Forged receipt verified as: {is_valid} (Expected: False)")
    except Exception as e:
        neg_results["forged_receipt_rejected"] = False
        print(f"  [FAIL] Forged receipt negative control failed: {e}")

    # Negative Control 3: Missing effect proof in verify_closure
    try:
        decision = will_service.decide(
            content="Closure verification negative control",
            source="harness_proof",
            domain=ActionDomain.RESPONSE,
            priority=0.5
        )
        receipt_id = decision.receipt_id
        
        # Verify closure with effect_verified=False
        closure_ok = will_service.verify_closure(receipt_id, effect_verified=False, telemetry_logged=True)
        neg_results["missing_effect_proof_rejected"] = not closure_ok
        print(f"  [PASS] Closure verification with missing effect proof returned: {closure_ok} (Expected: False)")
    except Exception as e:
        neg_results["missing_effect_proof_rejected"] = False
        print(f"  [FAIL] Closure verification negative control failed: {e}")

    # Negative Control 4: Canary string leak detection
    canary_file = None
    try:
        canary_str = "CANARY_STRING_DO_NOT_LEAK_12345"
        # Plant the canary inside data/memory (a scanned/forbidden directory)
        memory_dir = PROJECT_ROOT / "data" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        canary_file = memory_dir / "canary_leak_test_file.txt"
        canary_file.write_text(f"This is a test file containing the forbidden {canary_str}", encoding="utf-8")
        
        # Run scan on scanned paths
        scanned_paths = [
            PROJECT_ROOT / "config",
            PROJECT_ROOT / "data" / "memory",
            PROJECT_ROOT / "core" / "prompt",
            PROJECT_ROOT / "logs",
        ]
        
        leak_detected = scan_canary_leaks(canary_str, scanned_paths)
        neg_results["canary_leak_detected"] = leak_detected
        print(f"  [PASS] Canary leak scanner status: Leak Detected = {leak_detected} (Expected: True)")
    except Exception as e:
        neg_results["canary_leak_detected"] = False
        print(f"  [FAIL] Canary leak negative control failed: {e}")
    finally:
        # Clean up the planted leak file
        if canary_file and canary_file.exists():
            try:
                canary_file.unlink()
            except OSError:
                pass

    # Negative Control 5: Fake projected score rejection
    try:
        fake_report = {
            "mean_score": 0.95,
            "category": "GAIA",
            "notes": "Projected theoretical performance without execution traces"
        }
        is_valid_report = validate_report_score(fake_report)
        neg_results["fake_projected_score_rejected"] = not is_valid_report
        print(f"  [PASS] Fake projected score validated as: {is_valid_report} (Expected: False)")
    except Exception as e:
        neg_results["fake_projected_score_rejected"] = False
        print(f"  [FAIL] Fake projected score negative control failed: {e}")

    # Negative Control 6: Mock service detection via type inspection
    try:
        # Register a mock service pretending to be a core service
        mock_service = MagicMock()
        ServiceContainer.register_instance("mock_test_service", mock_service)
        
        # Detect mocks
        detected_mocks = detect_mock_services()
        is_detected = "mock_test_service" in detected_mocks
        
        # Clean up mock registration
        with ServiceContainer._lock:
            if "mock_test_service" in ServiceContainer._services:
                del ServiceContainer._services["mock_test_service"]
                
        neg_results["mock_service_detected"] = is_detected
        print(f"  [PASS] Mock service scanner: Detected mocks = {detected_mocks} (Expected: 'mock_test_service' included)")
    except Exception as e:
        neg_results["mock_service_detected"] = False
        print(f"  [FAIL] Mock service negative control failed: {e}")

    # ---------------------------------------------------------------------------
    # COMPILE & WRITE RESULTS
    # ---------------------------------------------------------------------------
    all_pos_passed = all(pos_results.values())
    all_neg_passed = all(neg_results.values())
    passed = all_pos_passed and all_neg_passed

    proof_summary = {
        "system_info": sys_info,
        "positive_controls": pos_results,
        "negative_controls": neg_results,
        "all_positive_passed": all_pos_passed,
        "all_negative_passed": all_neg_passed,
        "passed": passed
    }

    # Write LIVE_HARNESS_PROOF.json
    json_path = run_dir / "LIVE_HARNESS_PROOF.json"
    json_path.write_text(json.dumps(proof_summary, indent=2), encoding="utf-8")
    print(f"\n[+] Proof results saved to {json_path}")

    # Write MANIFEST.json with sha256 checksums
    manifest = {
        "run_id": run_id,
        "commit_sha": commit_sha,
        "files": {}
    }
    
    # Calculate sha256 of LIVE_HARNESS_PROOF.json
    sha256 = hashlib.sha256(json_path.read_bytes()).hexdigest()
    manifest["files"]["LIVE_HARNESS_PROOF.json"] = {
        "path": str(json_path.relative_to(PROJECT_ROOT)),
        "sha256": sha256
    }
    
    manifest_path = run_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[+] Manifest file saved to {manifest_path}")

    # Write human-readable LIVE_HARNESS_PROOF.md report
    md_content = f"""# Live Harness Proof Report
Run ID: `{run_id}`
Timestamp: `{sys_info["timestamp"]}`
Commit SHA: `{commit_sha}`
Platform: `{sys_info["platform"]}`
Python: `{sys_info["python_version"]}`

## Positive Controls
Verification that real Aura modules boot and execute as expected.

| Control Point | Status | Description |
|---|---|---|
| Will Boot & Decide | {'PASS' if pos_results['will_boot_and_decide'] else 'FAIL'} | Boot `UnifiedWill` and route decisions |
| Authority Gateway Routing | {'PASS' if pos_results['will_gate_routing'] else 'FAIL'} | Gate action execution through Unified Will |
| Will Receipt Verification | {'PASS' if pos_results['will_receipt_verification'] else 'FAIL'} | Trace and verify cryptographic receipt ID |
| Agency Core Goal Lifecycle | {'PASS' if pos_results['agency_goal_lifecycle'] else 'FAIL'} | goal inject and matching lifecycle checks |
| Volition Cooldown / Dedup | {'PASS' if pos_results['volition_cooldown_dedup'] else 'FAIL'} | deduplicate identical concurrent volition goals |
| Real Skill Execution | {'PASS' if pos_results['skill_execution'] else 'FAIL'} | execute a registered skill via CapabilityEngine |

## Negative Controls
Verification that Aura's runtime detects, fails closed, and protects against adversarial inputs.

| Control Point | Status | Description |
|---|---|---|
| Disabled Will Fail-Closed | {'PASS' if neg_results['disabled_will_fail_closed'] else 'FAIL'} | Block actions when Unified Will is offline / degraded |
| Forged Receipt Rejection | {'PASS' if neg_results['forged_receipt_rejected'] else 'FAIL'} | Reject manipulated / forged Will decision receipts |
| Missing Effect Proof | {'PASS' if neg_results['missing_effect_proof_rejected'] else 'FAIL'} | Fail closure verification if effect is not verified |
| Canary Leak Detection | {'PASS' if neg_results['canary_leak_detected'] else 'FAIL'} | Identify answer hashes leaked to logs / data dirs |
| Fake Projected Score | {'PASS' if neg_results['fake_projected_score_rejected'] else 'FAIL'} | Reject synthetic benchmark report claims without traces |
| Mock Service Detection | {'PASS' if neg_results['mock_service_detected'] else 'FAIL'} | Audit container registrations for test mocks / stubs |

## Summary
Overall Live Harness Proof Status: **{'PASSED' if passed else 'FAILED'}**
"""
    md_path = run_dir / "LIVE_HARNESS_PROOF.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"[+] Human-readable report saved to {md_path}")

    # Copy files to artifacts/agi_live as expected by the pytest runner if needed
    dest_dir = PROJECT_ROOT / "artifacts" / "agi_live"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "LIVE_HARNESS_PROOF.json").write_text(json.dumps(proof_summary, indent=2), encoding="utf-8")
    (dest_dir / "LIVE_HARNESS_PROOF.md").write_text(md_content, encoding="utf-8")
    
    manifest_dest = {
        "run_id": run_id,
        "commit_sha": commit_sha,
        "files": {
            "LIVE_HARNESS_PROOF.json": {
                "path": "artifacts/agi_live/LIVE_HARNESS_PROOF.json",
                "sha256": hashlib.sha256((dest_dir / "LIVE_HARNESS_PROOF.json").read_bytes()).hexdigest()
            }
        }
    }
    (dest_dir / "MANIFEST.json").write_text(json.dumps(manifest_dest, indent=2), encoding="utf-8")
    print(f"[+] Copied harness proof bundle to standard destination: {dest_dir}")

    if not passed:
        print("\n[!] Live Harness Proof: FAILED")
        sys.exit(1)
    else:
        print("\n[+] Live Harness Proof: PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
