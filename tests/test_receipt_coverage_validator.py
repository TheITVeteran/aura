from __future__ import annotations

import json
from pathlib import Path

from tools import receipt_coverage_validator


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
    from core.runtime_tools import _sign_payload

    return {
        "receipt_id": receipt_id,
        "domain": "tool_execution",
        "verification": {
            "payload": payload,
            "signature": _sign_payload(payload.encode("utf-8")),
            "signature_scheme": "test",
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
        ],
    )

    assert receipt_coverage_validator.main(["--artifacts", str(tmp_path)]) == 0

    report = json.loads((tmp_path / "receipt_coverage.json").read_text(encoding="utf-8"))
    assert report["invalid_receipts"] == 0
    assert report["total_receipts"] == 1
    assert report["person_box_harness_receipts"] == 7
    assert report["passed"] is True


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
