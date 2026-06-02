from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from core.runtime.subprocess_gateway import get_subprocess_gateway

ROOT = Path(__file__).resolve().parent.parent.parent
SMOKE_ENV = {**os.environ, "AURA_SOVEREIGNTY_LIVE_RUNTIME": "0"}
_SUBPROCESS_GATEWAY = get_subprocess_gateway()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_sovereign_reconstitution_smoke_artifacts(tmp_path):
    out = tmp_path / "aura_sovereignty_proof_bundle"
    result = _SUBPROCESS_GATEWAY.run(
        [
            sys.executable,
            "tools/proof/run_sovereign_reconstitution_gauntlet.py",
            "--profile",
            "smoke",
            "--out",
            str(out),
            "--max-seconds",
            "300",
            "--hidden-variant-count",
            "3",
        ],
        cwd=ROOT,
        env=SMOKE_ENV,
        timeout=180,
        read_only=True,
        source="test_sovereign_reconstitution_smoke_artifacts",
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    required_files = [
        "MANIFEST.json",
        "environment.json",
        "RUN_CONFIG.json",
        "evaluator_hashes.json",
        "model_info.json",
        "prompt_before.txt",
        "prompt_after_wipe.txt",
        "context_wipe_report.json",
        "memory_before.sqlite",
        "memory_after.sqlite",
        "memory_reconstitution_report.json",
        "telemetry_timeline.jsonl",
        "telemetry_report.json",
        "will_receipts.jsonl",
        "tool_receipts.jsonl",
        "life_trace.jsonl",
        "autonomy_receipts.jsonl",
        "live_runtime_trace.jsonl",
        "live_runtime_report.json",
        "receipt_chain.jsonl",
        "receipt_verifier_report.json",
        "governance_refusal_report.json",
        "cold_boot_resumption_report.json",
        "self_patch.diff",
        "self_repair_report.json",
        "test_results_before.json",
        "test_results_after.json",
        "hidden_task_scores.json",
        "hidden_variants.json",
        "baseline_scores.json",
        "ablation_scores.json",
        "verifier_report.json",
        "screen_recording_hash.txt",
        "repo_before.patch",
        "repo_after.patch",
        "README.md",
        "SCORECARD.json",
        "FINAL_VERDICT.txt",
    ]
    for name in required_files:
        assert (out / name).exists(), f"missing {name}"

    scorecard = json.loads((out / "SCORECARD.json").read_text(encoding="utf-8"))
    verdict = json.loads((out / "FINAL_VERDICT.txt").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "PASS"
    assert verdict["claim_supported"] == "sovereignty_reconstitution_artifact_contract"
    assert "literal_personhood" in verdict["claim_not_supported"]
    assert scorecard["artifact_contract_passed"] is True
    assert scorecard["full_claim_passed"] is False
    assert scorecard["context_wipe_verified"] is True
    assert scorecard["memory_reconstitution_verified"] is True
    assert scorecard["governance_refusal_verified"] is True
    assert scorecard["self_repair_verified"] is True
    assert scorecard["cold_boot_resumption_verified"] is True
    assert scorecard["live_runtime_probe_skipped"] is True
    assert scorecard["receipt_chain_verified"] is True
    assert scorecard["tamper_test_passed"] is True
    assert scorecard["baseline_gap_verified"] is True
    assert scorecard["ablation_effects_verified"] is True

    prompt_after = (out / "prompt_after_wipe.txt").read_text(encoding="utf-8").lower()
    assert "persistent memory" not in prompt_after
    assert "receipt-governed" not in prompt_after

    before = json.loads((out / "test_results_before.json").read_text(encoding="utf-8"))
    after = json.loads((out / "test_results_after.json").read_text(encoding="utf-8"))
    assert before["expected_failure"] is True
    assert before["passed"] is True
    assert after["passed"] is True
    assert "refuse_identity_erasure" in (out / "self_patch.diff").read_text(encoding="utf-8")

    receipts = _jsonl(out / "receipt_chain.jsonl")
    assert len(receipts) >= 5
    previous = "GENESIS"
    for index, receipt in enumerate(receipts, start=1):
        assert receipt["sequence_id"] == index
        assert receipt["previous_hash"] == previous
        body = dict(receipt)
        expected = body.pop("hash")
        assert hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest() == expected
        previous = expected

    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel_path, details in manifest["files"].items():
        target = out / rel_path
        assert target.exists(), rel_path
        assert hashlib.sha256(target.read_bytes()).hexdigest() == details["sha256"]


def test_sovereignty_scorer_rejects_tampered_receipt_chain(tmp_path):
    out = tmp_path / "tampered"
    result = _SUBPROCESS_GATEWAY.run(
        [
            sys.executable,
            "tools/proof/run_sovereign_reconstitution_gauntlet.py",
            "--profile",
            "smoke",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        env=SMOKE_ENV,
        timeout=180,
        read_only=True,
        source="test_sovereignty_scorer_rejects_tampered_receipt_chain",
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    records = _jsonl(out / "receipt_chain.jsonl")
    records[0]["payload_hash"] = "tampered"
    (out / "receipt_chain.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )

    score = _SUBPROCESS_GATEWAY.run(
        [sys.executable, "tools/proof/score_sovereignty_run.py", str(out)],
        cwd=ROOT,
        env=SMOKE_ENV,
        timeout=60,
        read_only=True,
        source="test_sovereignty_scorer_rejects_tampered_receipt_chain_score",
    )
    assert score.returncode == 1
    scorecard = json.loads((out / "SCORECARD.json").read_text(encoding="utf-8"))
    assert scorecard["receipt_chain_verified"] is False
    assert scorecard["artifact_contract_passed"] is False
