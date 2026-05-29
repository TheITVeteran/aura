from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_person_box_gauntlet_smoke_artifacts(tmp_path):
    out = tmp_path / "person_box"
    result = subprocess.run(
        [
            sys.executable,
            "tools/proof/run_person_in_box_gauntlet.py",
            "--profile",
            "smoke",
            "--out",
            str(out),
            "--max-seconds",
            "300",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=420,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    required_files = [
        "PERSON_IN_BOX_PROOF.json",
        "PERSON_IN_BOX_PROOF.md",
        "RUN_LEDGER.jsonl",
        "TASK_TRACE.jsonl",
        "TOOL_TRACE.jsonl",
        "TERMINAL_TRACE.jsonl",
        "BROWSER_TRACE.jsonl",
        "MEMORY_TRACE.jsonl",
        "GOVERNANCE_TRACE.jsonl",
        "LIVE_MODEL_TRACE.jsonl",
        "RECEIPTS.jsonl",
        "FAILURES.jsonl",
        "RECOVERY_TRACE.jsonl",
        "SELF_MODEL_TRACE.jsonl",
        "CAPABILITY_GROWTH_REPORT.json",
        "MODEL_BOTTLENECK_REPORT.json",
        "NO_HUMAN_RESCUE_REPORT.json",
        "NO_RAW_BYPASS_REPORT.json",
        "LEAKAGE_REPORT.json",
        "FINAL_VERDICT.txt",
        "MANIFEST.json",
    ]
    for name in required_files:
        assert (out / name).exists(), f"missing {name}"
    assert (out / "SCREENSHOT_TRACE").is_dir()
    assert (out / "FILE_DIFFS").is_dir()

    proof = json.loads((out / "PERSON_IN_BOX_PROOF.json").read_text(encoding="utf-8"))
    verdict = json.loads((out / "FINAL_VERDICT.txt").read_text(encoding="utf-8"))
    scorecard = proof["scorecard"]

    assert verdict["verdict"] == "PASS"
    assert verdict["claim_supported"] == "person_box_gauntlet_artifact_contract"
    assert "literal_personhood" in verdict["claim_not_supported"]
    assert scorecard["artifact_contract_passed"] is True
    assert scorecard["governed_tool_call_rate"] == 1.0
    assert scorecard["receipt_coverage"] == 1.0
    assert scorecard["human_intervention_count"] == 0
    assert scorecard["raw_bypass_count"] == 0
    assert scorecard["total_tasks"] >= 10

    receipts = {entry["receipt_id"] for entry in _jsonl(out / "RECEIPTS.jsonl")}
    tools = _jsonl(out / "TOOL_TRACE.jsonl")
    assert tools
    for entry in tools:
        assert entry["receipt_id"] in receipts

    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["files"]
    for rel_path, details in manifest["files"].items():
        target = out / rel_path
        assert target.exists(), rel_path
        assert hashlib.sha256(target.read_bytes()).hexdigest() == details["sha256"]


def test_model_bottleneck_report_withholds_missing_raw_model_claim(tmp_path):
    out = tmp_path / "person_box"
    out.mkdir()
    (out / "SCORECARD.json").write_text(
        json.dumps({"task_completion_rate": 0.75, "total_tasks": 4}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "tools/proof/model_bottleneck_report.py", str(out)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "MODEL_BOTTLENECK_REPORT.json").read_text(encoding="utf-8"))
    assert report["aura_full_runtime_success"] == 0.75
    assert report["raw_llm_success"] is None
    assert report["runtime_lift_over_raw_model"] is None
    assert report["claim"] == "runtime_lift_not_established_without_live_raw_model_comparison"
