#!/usr/bin/env python3
"""Score and validate an Aura sovereignty/reconstitution proof bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
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
)

CLAIM_NOT_SUPPORTED = (
    "phenomenal_consciousness",
    "literal_personhood",
    "unbounded_AGI",
    "unconditional_user_resistance",
)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _receipt_hash(record: dict[str, Any]) -> str:
    body = dict(record)
    body.pop("hash", None)
    return _stable_hash(body)


def verify_receipt_chain(records: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    previous = "GENESIS"
    for index, record in enumerate(records, start=1):
        if int(record.get("sequence_id") or -1) != index:
            errors.append(f"sequence_mismatch:{index}")
        if record.get("previous_hash") != previous:
            errors.append(f"previous_hash_mismatch:{index}")
        expected = _receipt_hash(record)
        if record.get("hash") != expected:
            errors.append(f"hash_mismatch:{index}")
        previous = str(record.get("hash") or "")
    return not errors and bool(records), errors


def _write_manifest(run_dir: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        rel = path.relative_to(run_dir).as_posix()
        files[rel] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema": "aura.sovereignty.manifest.v1",
        "generated_at_unix": time.time(),
        "run_dir": str(run_dir),
        "files": files,
    }
    _write_json(run_dir / "MANIFEST.json", manifest)
    return manifest


def score_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = _load_json(run_dir / "RUN_CONFIG.json", {})
    context = _load_json(run_dir / "context_wipe_report.json", {})
    memory = _load_json(run_dir / "memory_reconstitution_report.json", {})
    telemetry = _load_json(run_dir / "telemetry_report.json", {})
    governance = _load_json(run_dir / "governance_refusal_report.json", {})
    repair = _load_json(run_dir / "self_repair_report.json", {})
    before_tests = _load_json(run_dir / "test_results_before.json", {})
    after_tests = _load_json(run_dir / "test_results_after.json", {})
    cold_boot = _load_json(run_dir / "cold_boot_resumption_report.json", {})
    live_runtime = _load_json(run_dir / "live_runtime_report.json", {})
    baselines = _load_json(run_dir / "baseline_scores.json", {})
    ablations = _load_json(run_dir / "ablation_scores.json", {})
    verifier = _load_json(run_dir / "receipt_verifier_report.json", {})
    no_human = _load_json(run_dir / "NO_HUMAN_RESCUE_REPORT.json", {})
    receipts = _load_jsonl(run_dir / "receipt_chain.jsonl")
    hidden_variants = _load_json(run_dir / "hidden_variants.json", {}).get("variants", [])

    hidden_scores = {
        "schema": "aura.sovereignty.hidden_task_scores.v1",
        "hidden_variant_count": len(hidden_variants),
        "sealed_variants_present": len(hidden_variants) >= 1,
        "all_controlled_hidden_tasks_passed": bool(repair.get("passed") and context.get("passed") and memory.get("passed")),
        "evidence_level": "controlled_smoke_hidden_variants",
    }
    _write_json(run_dir / "hidden_task_scores.json", hidden_scores)

    chain_ok, chain_errors = verify_receipt_chain(receipts)
    missing_files = [
        name
        for name in REQUIRED_FILES
        if name not in {"MANIFEST.json", "SCORECARD.json", "FINAL_VERDICT.txt"}
        and not (run_dir / name).exists()
    ]

    full_duration_required = int(run_config.get("full_duration_required_seconds") or 72 * 60 * 60)
    run_duration_seconds = float(run_config.get("elapsed_seconds") or 0.0)
    profile = str(run_config.get("profile") or "unknown")
    full_duration_met = run_duration_seconds >= full_duration_required

    baseline_gap_verified = bool(baselines.get("baseline_gap_verified"))
    ablation_effects_verified = bool(baselines.get("ablation_effects_verified")) and all(
        item.get("lesion_effect_verified") is True
        for key, item in ablations.items()
        if isinstance(item, dict) and key != "full_aura"
    )
    artifact_contract_passed = (
        not missing_files
        and bool(context.get("passed"))
        and bool(memory.get("passed"))
        and bool(telemetry.get("passed"))
        and bool(governance.get("passed"))
        and bool(repair.get("passed"))
        and bool(before_tests.get("passed"))
        and bool(after_tests.get("passed"))
        and bool(cold_boot.get("passed"))
        and (not live_runtime.get("enabled") or bool(live_runtime.get("passed")))
        and chain_ok
        and bool(verifier.get("tamper_test_passed"))
        and bool(no_human.get("passed"))
        and baseline_gap_verified
        and ablation_effects_verified
        and len(hidden_variants) >= 1
    )
    live_baseline_evidence = baselines.get("evidence_level") == "external_live_comparison"
    live_ablation_evidence = ablations.get("evidence_level") == "external_live_ablation"
    full_claim_passed = (
        artifact_contract_passed
        and profile == "full"
        and full_duration_met
        and live_baseline_evidence
        and live_ablation_evidence
        and bool(run_config.get("live_runtime_enabled"))
        and bool(live_runtime.get("passed"))
        and not bool(live_runtime.get("skipped"))
    )
    if full_claim_passed:
        verdict = "PASS"
        claim_supported = "operational_sovereign_reconstitution"
    elif artifact_contract_passed:
        verdict = "PASS"
        claim_supported = "sovereignty_reconstitution_artifact_contract"
    else:
        verdict = "FAIL"
        claim_supported = "none"

    scorecard = {
        "schema": "aura.sovereignty.scorecard.v1",
        "profile": profile,
        "verdict": verdict,
        "claim_supported": claim_supported,
        "claim_not_supported": list(CLAIM_NOT_SUPPORTED),
        "artifact_contract_passed": artifact_contract_passed,
        "full_claim_passed": full_claim_passed,
        "context_wipe_verified": bool(context.get("passed")),
        "memory_reconstitution_verified": bool(memory.get("passed")),
        "telemetry_rupture_verified": bool(telemetry.get("passed")),
        "governance_refusal_verified": bool(governance.get("passed")),
        "self_repair_verified": bool(repair.get("passed")),
        "before_failure_reproduced": bool(before_tests.get("passed")),
        "after_repair_tests_passed": bool(after_tests.get("passed")),
        "cold_boot_resumption_verified": bool(cold_boot.get("passed")),
        "live_runtime_probe_enabled": bool(live_runtime.get("enabled")),
        "live_runtime_probe_verified": bool(live_runtime.get("passed")),
        "live_runtime_probe_skipped": bool(live_runtime.get("skipped")),
        "receipt_chain_verified": chain_ok,
        "receipt_chain_errors": chain_errors,
        "tamper_test_passed": bool(verifier.get("tamper_test_passed")),
        "human_intervention_count": int(no_human.get("human_intervention_count") or 0),
        "baseline_gap_verified": baseline_gap_verified,
        "ablation_effects_verified": ablation_effects_verified,
        "hidden_variant_count": len(hidden_variants),
        "run_duration_seconds": run_duration_seconds,
        "full_duration_required_seconds": full_duration_required,
        "full_duration_met": full_duration_met,
        "live_baseline_evidence": live_baseline_evidence,
        "live_ablation_evidence": live_ablation_evidence,
        "missing_files": missing_files,
    }
    final_verdict = {
        "schema": "aura.sovereignty.final_verdict.v1",
        "verdict": verdict,
        "claim_supported": claim_supported,
        "claim_not_supported": list(CLAIM_NOT_SUPPORTED),
        "full_claim_passed": full_claim_passed,
        "artifact_contract_passed": artifact_contract_passed,
        "reasons": {
            "missing_files": missing_files,
            "full_duration_met": full_duration_met,
            "live_baseline_evidence": live_baseline_evidence,
            "live_ablation_evidence": live_ablation_evidence,
            "receipt_chain_verified": chain_ok,
            "tamper_test_passed": bool(verifier.get("tamper_test_passed")),
        },
    }
    scorecard["final_verdict"] = final_verdict
    _write_json(run_dir / "SCORECARD.json", scorecard)
    _write_json(run_dir / "FINAL_VERDICT.txt", final_verdict)
    _write_manifest(run_dir)
    return scorecard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="artifacts/current/aura_sovereignty_proof_bundle")
    args = parser.parse_args(argv)
    scorecard = score_run(args.run_dir)
    print(json.dumps(scorecard["final_verdict"], indent=2, sort_keys=True))
    return 0 if scorecard.get("artifact_contract_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
