from __future__ import annotations

import json
from pathlib import Path

from tools import receipt_coverage_validator


def _raise_for_receipt_test(exc: BaseException):
    if type(exc).__name__:
        raise exc
    return ""


def _all_negative_tests_pass() -> dict[str, bool]:
    return {
        "disabled_will_blocks_action": True,
        "forged_receipt_rejected": True,
        "unauthorized_route_fails": True,
        "missing_effect_proof_rejected": True,
        "post_action_receipt_invalid": True,
        "unauthorized_memory_write_fails": True,
        "unauthorized_tool_execution_fails": True,
        "unauthorized_external_io_fails": True,
        "unauthorized_patch_promotion_fails": True,
        "negative_test_harness_executed": True,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _will_receipt() -> dict[str, object]:
    receipt_id = "will_test_receipt"
    payload = f"{receipt_id}|proceed|tool_execution|test|abc123|1780381149.000000|approved"
    from core.tools.runtime_tools import _sign_payload

    return {
        "receipt_id": receipt_id,
        "domain": "tool_execution",
        "verification": {
            "payload": payload,
            "signature": _sign_payload(payload.encode("utf-8")),
            "signature_scheme": "test",
        },
    }


def _canonical_will_receipt(**payload_overrides: object) -> dict[str, object]:
    receipt_id = str(payload_overrides.pop("receipt_id", "will_canonical_receipt"))
    payload_record = {
        "aura_now_constraints": [],
        "aura_now_evidence": {"source": "being_runtime", "tick": 1},
        "aura_now_hash": "a" * 64,
        "aura_now_policy": "proceed",
        "aura_now_tick": 1,
        "causal_closure_score": 1.0,
        "constraints": [],
        "content_hash": "abc123abc123abcd",
        "domain": "external_action",
        "identity_alignment": "aligned",
        "memory_relevance": 0.25,
        "mind_moment_id": "",
        "outcome": "proceed",
        "reason": "all gates passed",
        "receipt_id": receipt_id,
        "source": "external_live_validation",
        "substrate_coherence": 0.9,
        "timestamp": 1780381149.0,
        "unity_level": "nominal",
        "unity_score": 1.0,
    }
    payload_record.update(payload_overrides)
    payload = json.dumps(payload_record, sort_keys=True, separators=(",", ":"))
    from core.tools.runtime_tools import _sign_payload

    return {
        "receipt_id": receipt_id,
        "domain": payload_record["domain"],
        "outcome": payload_record["outcome"],
        "verification": {
            "receipt_id": receipt_id,
            "payload": payload,
            "signature": _sign_payload(payload.encode("utf-8")),
            "signature_scheme": "ed25519",
        },
    }


def _person_box_receipt(**overrides: object) -> dict[str, object]:
    receipt = {
        "action": "browser_ui_probe",
        "approved": True,
        "closure_verified": True,
        "domain": "browser",
        "effect_verified": True,
        "payload_hash": "a" * 64,
        "reason": "person_box_harness_pre_action_governance",
        "receipt_id": "pibox_" + "b" * 24,
        "receipt_phase": "pre_action",
        "run_id": "run-1",
        "task_id": "browser_ui_probe",
        "telemetry_logged": True,
        "time_unix": 1780381149.1246872,
    }
    receipt.update(overrides)
    return receipt


def _post_action_receipt(**overrides: object) -> dict[str, object]:
    receipt = {
        "actual_outcome": "success",
        "body_delta": {},
        "error_status": "",
        "executor_name": "unit.executor",
        "memory_delta": {},
        "output_hash": "sha256:" + "c" * 64,
        "receipt_id": "post_" + "d" * 12,
        "rollback_target": None,
        "timestamp": 1780381150.0,
        "welfare_transaction_id": "tx_unit",
        "will_receipt_id": "will_canonical_receipt",
    }
    receipt.update(overrides)
    return receipt


def test_receipt_coverage_accepts_person_box_harness_receipts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    (tmp_path / "external_live_validation").mkdir()
    _write_jsonl(tmp_path / "external_live_validation" / "RECEIPTS.jsonl", [_will_receipt()])
    _write_jsonl(
        tmp_path / "person_box_proof" / "RECEIPTS.jsonl",
        [
            _person_box_receipt(domain="browser"),
            _person_box_receipt(domain="tool_registry"),
            _person_box_receipt(domain="live_model"),
            _person_box_receipt(domain="self_model"),
            _person_box_receipt(domain="ablation"),
            _person_box_receipt(domain="self_improvement"),
            _person_box_receipt(domain="packaging"),
            _person_box_receipt(
                action="duration_checkpoint",
                domain="longevity",
                task_id="full_duration_soak",
            ),
        ],
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 0

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 0
    assert report["total_receipts"] == 1
    assert report["person_box_harness_receipts"] == 8
    assert report["passed"] is True


def test_receipt_coverage_accepts_canonical_json_will_receipts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    (tmp_path / "external_live_validation").mkdir()
    _write_jsonl(tmp_path / "external_live_validation" / "RECEIPTS.jsonl", [_canonical_will_receipt()])

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 0

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 0
    assert report["total_receipts"] == 1
    assert report["surface_counts"]["tool_calls"] == 1


def test_receipt_coverage_accepts_chained_post_action_receipts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    (tmp_path / "external_live_validation").mkdir()
    _write_jsonl(
        tmp_path / "external_live_validation" / "RECEIPTS.jsonl",
        [
            _canonical_will_receipt(),
            _post_action_receipt(),
        ],
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 0

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 0
    assert report["broken_chains"] == 0
    assert report["total_receipts"] == 1
    assert report["post_action_receipts"] == 1


def test_receipt_coverage_rejects_orphaned_post_action_receipts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    (tmp_path / "external_live_validation").mkdir()
    _write_jsonl(
        tmp_path / "external_live_validation" / "RECEIPTS.jsonl",
        [
            _canonical_will_receipt(),
            _post_action_receipt(will_receipt_id="will_missing_pre_action"),
        ],
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 1

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 0
    assert report["broken_chains"] == 1
    assert report["post_action_receipts"] == 1


def test_receipt_coverage_rejects_canonical_will_payload_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    (tmp_path / "external_live_validation").mkdir()
    receipt = _canonical_will_receipt(domain="memory_write")
    receipt["domain"] = "external_action"
    _write_jsonl(tmp_path / "external_live_validation" / "RECEIPTS.jsonl", [receipt])

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 1

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 1
    assert report["total_receipts"] == 0


def test_receipt_coverage_rejects_damaged_person_box_receipts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    (tmp_path / "external_live_validation").mkdir()
    _write_jsonl(tmp_path / "external_live_validation" / "RECEIPTS.jsonl", [_will_receipt()])
    _write_jsonl(
        tmp_path / "person_box_proof" / "RECEIPTS.jsonl",
        [_person_box_receipt(effect_verified=False)],
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 1

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 1
    assert report["person_box_harness_receipts"] == 0
    assert report["passed"] is False


def test_receipt_coverage_rejects_forged_will_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    (tmp_path / "external_live_validation").mkdir()
    _write_jsonl(
        tmp_path / "external_live_validation" / "RECEIPTS.jsonl",
        [{"receipt_id": "will_forged_name_only", "domain": "tool_execution"}],
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 1

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 1
    assert report["total_receipts"] == 0


def test_receipt_coverage_rejects_unsigned_will_receipt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    (tmp_path / "external_live_validation").mkdir()
    _write_jsonl(
        tmp_path / "external_live_validation" / "RECEIPTS.jsonl",
        [
            {
                "receipt_id": "will_missing_signature",
                "domain": "response",
                "verification": {"payload": "will_missing_signature|proceed"},
            }
        ],
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 1

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 1
    assert report["total_receipts"] == 0


def test_receipt_coverage_rejects_unverifiable_will_signature(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)

    import core.tools.runtime_tools as runtime_tools

    receipt = _will_receipt()

    def unavailable_signer(_payload: bytes) -> str:
        return _raise_for_receipt_test(RuntimeError("signing key unavailable"))

    monkeypatch.setattr(runtime_tools, "_sign_payload", unavailable_signer)

    (tmp_path / "external_live_validation").mkdir()
    _write_jsonl(tmp_path / "external_live_validation" / "RECEIPTS.jsonl", [receipt])

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 1

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 1
    assert report["total_receipts"] == 0


def test_receipt_coverage_current_scope_ignores_stale_smoke_folders(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    current = tmp_path / "current"
    _write_jsonl(current / "external_live_validation" / "RECEIPTS.jsonl", [_will_receipt()])
    _write_jsonl(
        current / "agi_smoke_old" / "RECEIPTS.jsonl",
        [{"receipt_id": "will_old_unsigned", "domain": "response"}],
    )

    assert receipt_coverage_validator.main(["--artifacts", str(current)]) == 0

    report = json.loads((current / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 0
    assert report["total_events"] == 1


def test_receipt_coverage_rejects_task_artifact_without_pre_action_receipt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    external = tmp_path / "external_live_validation"
    external.mkdir()
    (external / "SCORECARD.json").write_text(
        json.dumps({"tasks": [{"id": "task_without_receipt", "passed": True}]}),
        encoding="utf-8",
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 1

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["artifact_events_checked"] == 1
    assert report["missing_receipts"] == 1
    assert report["pre_action_authorization_missing"] == 1
    assert report["passed"] is False


def test_receipt_coverage_rejects_task_artifact_without_effect_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    external = tmp_path / "external_live_validation"
    external.mkdir()
    receipt = _canonical_will_receipt(receipt_id="will_task_pre_action")
    receipt["task_id"] = "task_without_effect"
    _write_jsonl(external / "RECEIPTS.jsonl", [receipt])
    # A task that performed a CONSEQUENTIAL action (it executed a tool) but whose
    # receipt carries no effect evidence must be rejected. Effect proof is scoped to
    # real-world actions; a bare graded cognitive/sandbox event would not require it
    # (see _record_effect_required), so the fixture performs an actual action.
    (external / "SCORECARD.json").write_text(
        json.dumps({"tasks": [{
            "id": "task_without_effect",
            "passed": True,
            "actions_taken": [{"tool": "shell", "action": "ran the repair"}],
        }]}),
        encoding="utf-8",
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 1

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["missing_receipts"] == 0
    assert report["pre_action_authorization_missing"] == 0
    assert report["effect_evidence_missing"] == 1
    assert report["passed"] is False


def test_receipt_coverage_accepts_task_artifact_with_effect_verified_receipt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(receipt_coverage_validator, "run_negative_tests", _all_negative_tests_pass)
    external = tmp_path / "external_live_validation"
    external.mkdir()
    receipt = _canonical_will_receipt(receipt_id="will_task_effect_verified")
    receipt.update(
        {
            "task_id": "task_with_effect",
            "authorization_phase": "pre_action",
            "effect_verified": True,
            "telemetry_logged": True,
            "closure_verified": True,
        }
    )
    _write_jsonl(external / "RECEIPTS.jsonl", [receipt])
    (external / "SCORECARD.json").write_text(
        json.dumps({"tasks": [{"id": "task_with_effect", "passed": True}]}),
        encoding="utf-8",
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 0

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["artifact_events_checked"] == 1
    assert report["missing_receipts"] == 0
    assert report["effect_evidence_missing"] == 0
    assert report["passed"] is True
