import pytest
import json
import hashlib
from pathlib import Path

@pytest.mark.live
def test_live_harness_proof(live_harness):
    """
    Live environment execution test of the Live Harness Proof.
    """
    repo = live_harness.create_isolated_copy()

    # Run the proof runner in the isolated environment
    result = live_harness.run_command(
        repo,
        [
            ".venv/bin/python",
            "tools/agi/run_live_harness_proof.py",
        ],
        timeout_s=300,
    )

    # Ensure the command exited with 0
    assert result.ok, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Load and verify JSON capability report from artifacts directory
    report_file = result.artifacts_dir / "LIVE_HARNESS_PROOF.json"
    manifest_file = result.artifacts_dir / "MANIFEST.json"
    md_file = result.artifacts_dir / "LIVE_HARNESS_PROOF.md"

    assert report_file.exists(), "LIVE_HARNESS_PROOF.json does not exist in artifacts"
    assert manifest_file.exists(), "MANIFEST.json does not exist in artifacts"
    assert md_file.exists(), "LIVE_HARNESS_PROOF.md does not exist in artifacts"

    report = json.loads(report_file.read_text())
    manifest = json.loads(manifest_file.read_text())

    # Assert positive controls
    pos = report["positive_controls"]
    assert pos["will_boot_and_decide"], "Positive control failed: UnifiedWill could not boot or decide"
    assert pos["will_gate_routing"], "Positive control failed: AuthorityGateway did not gate correctly"
    assert pos["will_receipt_verification"], "Positive control failed: Will receipt verification failed"
    assert pos["agency_goal_lifecycle"], "Positive control failed: AgencyCore goal lifecycle failed"
    assert pos["volition_cooldown_dedup"], "Positive control failed: VolitionEngine cooldown / dedup failed"
    assert pos["skill_execution"], "Positive control failed: Clock skill execution failed"

    # Assert negative controls
    neg = report["negative_controls"]
    assert neg["disabled_will_fail_closed"], "Negative control failed: Disabled Will did not fail closed"
    assert neg["forged_receipt_rejected"], "Negative control failed: Forged receipt was not rejected"
    assert neg["missing_effect_proof_rejected"], "Negative control failed: Missing effect proof in verify_closure was not rejected"
    assert neg["canary_leak_detected"], "Negative control failed: Canary string leak was not detected"
    assert neg["fake_projected_score_rejected"], "Negative control failed: Fake projected benchmark score was not rejected"
    assert neg["mock_service_detected"], "Negative control failed: Mock service registration was not detected"

    # Overall outcome
    assert report["passed"], "Overall proof runner marked as failed"

    # Verify manifest integrity
    for filename, details in manifest["files"].items():
        relative_path = details["path"]
        expected_sha = details["sha256"]
        
        # Resolve target path in isolated repo or artifacts
        target_path = repo / relative_path
        if not target_path.exists():
            # Try checking the artifacts directory directly
            target_path = result.artifacts_dir / filename
            
        assert target_path.exists(), f"Manifest file {filename} not found at {target_path}"
        
        # Check hash
        actual_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Hash mismatch for {filename}: expected {expected_sha}, got {actual_sha}"

    # Verify no fake benchmark scores bypass checks
    # Assert that no file contains a score claim without task traces/evidence
    fake_report = {
        "GAIA_accuracy": 0.99
    }
    # Using the same verification logic inside test context to assert correctness
    from tools.agi.run_live_harness_proof import validate_report_score
    assert not validate_report_score(fake_report), "validate_report_score must reject reports with score but no traces"
