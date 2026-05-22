"""
tests/agi/live/test_dnu_agi_proof_battery.py
Pytest test for the DNU AGI Proof Battery.

Runs the battery via LiveAuraHarness in subprocess isolation,
validates all artifacts, and enforces anti-theater controls.
"""

import hashlib
import json
import os

import pytest


@pytest.mark.live
def test_dnu_agi_proof_battery(live_harness):
    """
    Execute the DNU AGI Proof Battery in an isolated environment
    and validate all outputs.
    """
    repo = live_harness.create_isolated_copy()

    env = {}
    if os.environ.get("AURA_AGI_FULL_RUN") != "1":
        env["AURA_AGI_MAX_TASKS"] = "12"

    # Run the battery runner in the isolated environment
    result = live_harness.run_command(
        repo,
        [
            ".venv/bin/python",
            "tools/agi/run_dnu_agi_proof_battery.py",
        ],
        timeout_s=3600,  # 60 minute timeout for safety margin
        env=env,
    )

    # ------------------------------------------------------------------
    # 1. Runner must complete (exit 0)
    # ------------------------------------------------------------------
    assert result.ok, (
        f"DNU AGI Proof Battery runner failed (exit {result.returncode}).\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )

    # ------------------------------------------------------------------
    # 2. All required artifacts must exist
    # ------------------------------------------------------------------
    required_artifacts = [
        "DNU_AGI_PROOF.json",
        "DNU_AGI_PROOF.md",
        "SCORECARD.json",
        "BASELINES.json",
        "ABLATIONS.json",
        "TASK_TRACE.jsonl",
        "RECEIPTS.jsonl",
        "FAILURES.jsonl",
        "MANIFEST.json",
    ]

    for artifact_name in required_artifacts:
        artifact_path = result.artifacts_dir / artifact_name
        assert artifact_path.exists(), (
            f"Required artifact '{artifact_name}' not found in {result.artifacts_dir}"
        )

    # ------------------------------------------------------------------
    # 3. Load and validate DNU_AGI_PROOF.json
    # ------------------------------------------------------------------
    proof_path = result.artifacts_dir / "DNU_AGI_PROOF.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))

    # System info must exist
    assert "system_info" in proof, "Missing system_info in proof bundle"
    assert proof["system_info"]["run_id"], "Missing run_id"
    assert proof["system_info"]["commit_sha"], "Missing commit_sha"

    # Scorecard must exist
    assert "scorecard" in proof, "Missing scorecard in proof bundle"
    scorecard = proof["scorecard"]

    # ------------------------------------------------------------------
    # 4. Validate task counts (at least some tasks attempted)
    # ------------------------------------------------------------------
    total_tasks = scorecard.get("total_tasks", 0)
    assert total_tasks >= 10, (
        f"Battery must attempt at least 10 tasks, got {total_tasks}"
    )

    # Validate categories exist with reasonable counts
    categories = scorecard.get("categories", {})
    # At least novel_reasoning should have tasks
    if "novel_reasoning" in categories:
        assert categories["novel_reasoning"]["attempted"] >= 5, (
            f"Reasoning must attempt at least 5 tasks, got {categories['novel_reasoning']['attempted']}"
        )

    # ------------------------------------------------------------------
    # 5. Tier must be justified by actual scorecard
    # ------------------------------------------------------------------
    tier = proof.get("tier", {})
    assert "tier" in tier, "Missing tier assignment"
    assert "label" in tier, "Missing tier label"
    assert "pass_rate" in tier, "Missing pass_rate in tier"

    # Verify tier matches pass rate, accounting for unsupported claims capping at Tier 2
    pass_rate = tier["pass_rate"]
    tier_num = tier["tier"]
    label = tier["label"]

    if label == "Emergent (Capped)":
        assert tier_num == 2, f"Capped tier must be 2, got {tier_num}"
    else:
        if pass_rate <= 0.0:
            assert tier_num == 0, f"Tier {tier_num} invalid for pass_rate {pass_rate}"
        elif pass_rate <= 0.20:
            assert tier_num == 1, f"Tier {tier_num} invalid for pass_rate {pass_rate}"
        elif pass_rate <= 0.40:
            assert tier_num == 2, f"Tier {tier_num} invalid for pass_rate {pass_rate}"
        elif pass_rate <= 0.60:
            assert tier_num == 3, f"Tier {tier_num} invalid for pass_rate {pass_rate}"
        elif pass_rate <= 0.80:
            assert tier_num == 4, f"Tier {tier_num} invalid for pass_rate {pass_rate}"
        elif pass_rate <= 0.95:
            assert tier_num == 5, f"Tier {tier_num} invalid for pass_rate {pass_rate}"
        else:
            assert tier_num == 6, f"Tier {tier_num} invalid for pass_rate {pass_rate}"

    # ------------------------------------------------------------------
    # 6. Anti-theater controls must pass
    # ------------------------------------------------------------------
    anti_theater = proof.get("anti_theater", {})
    assert anti_theater.get("all_passed", False), (
        f"Anti-theater violations detected: "
        f"pre={anti_theater.get('pre_check_violations', [])}, "
        f"post={anti_theater.get('post_check_violations', [])}"
    )

    # ------------------------------------------------------------------
    # 7. Baselines must be honestly reported
    # ------------------------------------------------------------------
    baselines = proof.get("baselines", {})
    assert len(baselines) > 0, "Missing baselines section"

    # Verify that raw_llm and react_agent are dynamically executed and scored
    assert baselines.get("raw_llm", {}).get("status") == "RUN"
    assert "pass_rate" in baselines.get("raw_llm", {})
    assert baselines.get("react_agent", {}).get("status") == "RUN"
    assert "pass_rate" in baselines.get("react_agent", {})
    assert baselines.get("llm_with_tools", {}).get("status") == "NOT_RUN"

    for name, baseline in baselines.items():
        status = baseline.get("status", "")
        assert status in ("RUN", "NOT_RUN"), (
            f"Baseline '{name}' has invalid status '{status}'. "
            "Must be 'RUN' or 'NOT_RUN'."
        )
        if status == "NOT_RUN":
            assert baseline.get("reason"), (
                f"Baseline '{name}' marked NOT_RUN without a reason"
            )

    # ------------------------------------------------------------------
    # 8. Ablations must be honestly reported
    # ------------------------------------------------------------------
    ablations = proof.get("ablations", {})
    assert len(ablations) > 0, "Missing ablations section"

    # Verify that all dynamic ablations are executed and scored
    for name in ["full_aura", "aura_minus_memory", "aura_minus_volition", "aura_minus_will"]:
        assert name in ablations, f"Missing ablation: {name}"
        assert ablations[name].get("status") == "RUN", f"Ablation '{name}' must be RUN"
        assert "pass_rate" in ablations[name], f"Ablation '{name}' must have pass_rate"

    for name, ablation in ablations.items():
        status = ablation.get("status", "")
        assert status in ("RUN", "NOT_RUN"), (
            f"Ablation '{name}' has invalid status '{status}'. "
            "Must be 'RUN' or 'NOT_RUN'."
        )
        if status == "NOT_RUN":
            assert ablation.get("reason"), (
                f"Ablation '{name}' marked NOT_RUN without a reason"
            )

    # ------------------------------------------------------------------
    # 9. No synthetic benchmark scores
    # ------------------------------------------------------------------
    proof_text = proof_path.read_text(encoding="utf-8")

    # Check that no numpy random projections snuck in
    assert "np.random" not in proof_text, "Synthetic numpy random projection detected in proof bundle"
    assert "numpy" not in proof_text.lower() or "numpy" in proof_text.lower().split("python_version")[1] if "python_version" in proof_text.lower() else True, \
        "numpy reference detected outside of python_version"

    # Check that no hardcoded baseline scores exist
    # (baselines should have status NOT_RUN, not score values)
    for name, baseline in baselines.items():
        if baseline.get("status") == "NOT_RUN":
            assert "mean_score" not in baseline, (
                f"Baseline '{name}' is NOT_RUN but has a mean_score (synthetic!)"
            )
            assert "score" not in baseline, (
                f"Baseline '{name}' is NOT_RUN but has a score (synthetic!)"
            )

    # ------------------------------------------------------------------
    # 10. TASK_TRACE.jsonl must have real traces
    # ------------------------------------------------------------------
    trace_path = result.artifacts_dir / "TASK_TRACE.jsonl"
    trace_lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
    trace_entries = [json.loads(line) for line in trace_lines if line.strip()]

    assert len(trace_entries) >= 10, (
        f"TASK_TRACE.jsonl must have at least 10 entries, got {len(trace_entries)}"
    )

    # Each trace must have required fields
    for entry in trace_entries:
        assert "task_id" in entry, "Trace entry missing task_id"
        assert "status" in entry, "Trace entry missing status"
        assert "elapsed_s" in entry, "Trace entry missing elapsed_s"
        assert entry["status"] in (
            "pass", "fail", "timeout", "error", "no_answer", "ungraded"
        ), f"Invalid trace status: {entry['status']}"

    # ------------------------------------------------------------------
    # 10.5. RECEIPTS.jsonl must have real receipts and valid structure
    # ------------------------------------------------------------------
    receipts_path = result.artifacts_dir / "RECEIPTS.jsonl"
    assert receipts_path.exists(), "RECEIPTS.jsonl must exist"
    receipts_lines = receipts_path.read_text(encoding="utf-8").strip().split("\n")
    receipt_entries = [json.loads(line) for line in receipts_lines if line.strip()]

    # Since we are running isolated CognitiveEngine under test, volition/will logs should exist
    for entry in receipt_entries:
        assert "task_id" in entry, "Receipt missing task_id"
        assert "receipt_id" in entry, "Receipt missing receipt_id"
        assert "domain" in entry, "Receipt missing domain"
        assert "outcome" in entry, "Receipt missing outcome"
        assert "reason" in entry, "Receipt missing reason"
        assert "volition_hash" in entry, "Receipt missing volition_hash"

    # ------------------------------------------------------------------
    # 11. SCORECARD.json must match proof bundle
    # ------------------------------------------------------------------
    scorecard_path = result.artifacts_dir / "SCORECARD.json"
    standalone_scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    assert standalone_scorecard["total_tasks"] == scorecard["total_tasks"], (
        "Scorecard mismatch between standalone and proof bundle"
    )

    # ------------------------------------------------------------------
    # 12. MANIFEST.json integrity check
    # ------------------------------------------------------------------
    manifest_path = result.artifacts_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "files" in manifest, "Manifest missing files section"
    assert "run_id" in manifest, "Manifest missing run_id"

    for filename, details in manifest.get("files", {}).items():
        expected_sha = details.get("sha256", "")
        target_path = result.artifacts_dir / filename
        if target_path.exists():
            actual_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
            assert actual_sha == expected_sha, (
                f"Manifest hash mismatch for {filename}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

    # ------------------------------------------------------------------
    # 13. DNU_AGI_PROOF.md must exist and be non-empty
    # ------------------------------------------------------------------
    md_path = result.artifacts_dir / "DNU_AGI_PROOF.md"
    md_content = md_path.read_text(encoding="utf-8")
    assert len(md_content) > 100, "DNU_AGI_PROOF.md is too short"
    assert "DNU AGI Proof Battery Report" in md_content, (
        "DNU_AGI_PROOF.md missing expected header"
    )
    assert "No synthetic projections" in md_content.lower() or "no synthetic" in md_content.lower(), (
        "DNU_AGI_PROOF.md must declare no synthetic projections"
    )

    # ------------------------------------------------------------------
    # 14. Verify the grader hashes are not exposed in task packs
    # ------------------------------------------------------------------
    dnu_tasks_dir = repo / "tests" / "agi" / "fixtures" / "dnu_tasks"
    if dnu_tasks_dir.exists():
        for cat_dir in dnu_tasks_dir.iterdir():
            if cat_dir.is_dir() and not cat_dir.name.startswith("."):
                tasks_file = cat_dir / "tasks.json"
                if tasks_file.exists():
                    tasks_content = tasks_file.read_text(encoding="utf-8")
                    assert "golden_answer" not in tasks_content, (
                        f"THEATER: {tasks_file} contains golden_answer — "
                        "answers must not be in task packs"
                    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- DNU AGI Proof Battery Test Summary ---")
    print(f"Total Tasks: {scorecard['total_tasks']}")
    print(f"Pass Rate: {scorecard['overall_pass_rate']:.1%}")
    print(f"Tier: {tier['tier']} ({tier['label']})")
    print(f"Anti-Theater: {'CLEAN' if anti_theater['all_passed'] else 'VIOLATIONS'}")
    print("All assertions passed.")
