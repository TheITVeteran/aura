from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.verify_semantic_neural_decode_canary import (
    REPO_ROOT,
    _sha,
    verify_canary,
)

ARTIFACT = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/cp550_semantic_decode_canary/result.json"
)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_artifact(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["model_identity"] = {
        "path": str(model.resolve()),
        "config_sha256": _file_sha(model / "config.json"),
        "weights_index_sha256": _file_sha(model / "model.safetensors.index.json"),
    }
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return artifact, model


def _write_journal(artifact: Path, destination: Path) -> None:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    previous = "0" * 64
    events = [
        {
            "event": "campaign_started",
            "source_commit": payload["source_commit"],
            "seed": payload["seed"],
            "tasks_per_difficulty": payload["tasks_per_difficulty"],
            "task_count": payload["task_count"],
            "arm_count": 5,
        }
    ]
    for index, raw_output in enumerate(payload["raw_outputs"], start=1):
        events.append(
            {
                "event": "decode_committed",
                "completed": index,
                "total": len(payload["raw_outputs"]),
                "row": {
                    "task_id": raw_output["task_id"],
                    "arm": raw_output["arm"],
                    "response_sha256": hashlib.sha256(
                        raw_output["response"].encode()
                    ).hexdigest(),
                },
                "raw_output": raw_output,
            }
        )
    events.append(
        {
            "event": "campaign_completed",
            "admitted": payload["admitted"],
            "report_receipt_sha256": payload["receipt_sha256"],
        }
    )
    lines = []
    last_decode_receipt = ""
    for index, event in enumerate(events):
        body = {
            "schema": "aura.rlc.semantic_neural_decode_journal.v1",
            "previous_receipt_sha256": previous,
            **event,
        }
        receipt = _sha(body)
        lines.append(json.dumps({**body, "receipt_sha256": receipt}, sort_keys=True))
        previous = receipt
        if index == len(events) - 2:
            last_decode_receipt = receipt
    payload["journal_last_decode_receipt_sha256"] = last_decode_receipt
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    completed = json.loads(lines[-1])
    completed["report_receipt_sha256"] = payload["receipt_sha256"]
    completed_body = {
        key: value for key, value in completed.items() if key != "receipt_sha256"
    }
    completed["receipt_sha256"] = _sha(completed_body)
    lines[-1] = json.dumps(completed, sort_keys=True)
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_semantic_decode_verifier_regrades_and_replays_frozen_canary(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    report = verify_canary(artifact, model_path=model)
    assert report["verified"] is True
    assert report["independent_exact_by_arm"] == {
        "ordinary_base": 0,
        "matched_wire_base": 0,
        "treatment": 27,
        "coefficient_lesion": 9,
        "matched_wrong_state": 0,
    }
    assert report["gain_count"] == 27
    assert report["regression_count"] == 0
    assert report["treatment_state_replay_count"] == 27
    assert report["paired_discordant_count"] == 27
    assert report["paired_one_sided_exact_p"] == pytest.approx(2**-27)


def test_semantic_decode_verifier_checks_receipt_chained_journal(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    journal = tmp_path / "journal.jsonl"
    _write_journal(artifact, journal)
    report = verify_canary(artifact, model_path=model, journal_path=journal)
    assert report["journal_event_count"] == 137
    assert report["journal_decode_count"] == 135

    lines = journal.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[20])
    tampered["row"]["task_id"] = "tampered"
    lines[20] = json.dumps(tampered, sort_keys=True)
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt chain broke"):
        verify_canary(artifact, model_path=model, journal_path=journal)


def test_semantic_decode_verifier_binds_resident_manifest(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    manifest = tmp_path / "active.json"
    manifest_payload = {
        "active_model_path": str(model.resolve()),
        "base_model": "base",
        "fused_at": 1,
        "schema_version": 2,
        "tag": "resident-test",
    }
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["resident_manifest_identity"] = {
        "path": str(manifest.resolve()),
        "sha256": _file_sha(manifest),
        "active_model_path": str(model.resolve()),
        "schema_version": 2,
        "base_model": "base",
        "tag": "resident-test",
        "fused_at": 1,
    }
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_canary(
        artifact,
        model_path=model,
        resident_manifest_path=manifest,
    )
    assert report["resident_manifest_identity"]["tag"] == "resident-test"
    assert "resident model bound by" in report["claim_boundary"]
    assert report["producer_claim_boundary_legacy"] is True

    manifest_payload["tag"] = "changed"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resident manifest identity mismatch"):
        verify_canary(
            artifact,
            model_path=model,
            resident_manifest_path=manifest,
        )


def test_semantic_decode_verifier_rejects_resealed_response_tamper(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    row = next(row for row in payload["raw_outputs"] if row["arm"] == "treatment")
    row["response"] = "FINAL_ANSWER: {}"
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary disagrees with raw output"):
        verify_canary(artifact, model_path=model)
