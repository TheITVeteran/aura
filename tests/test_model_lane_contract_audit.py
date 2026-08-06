from __future__ import annotations

from pathlib import Path

from tools.closeout.audit_model_lane_contract import ROOT, audit


def _passing_ownership(**_kwargs):
    return {
        "passed": True,
        "inventory_entries": 1,
        "owned_paths": 1,
        "load_references": 1,
        "findings": [],
    }


def test_repository_model_lane_contract_is_complete() -> None:
    report = audit(ROOT, ownership_runner=_passing_ownership)

    assert report["passed"] is True
    assert report["issues"] == []
    assert report["checked_functions"] >= 21
    assert report["unified_admission_contract"] == {
        "policy_owner": "ResourceAdmissionController",
        "durable_capacity_owner": "ModelLaneController",
        "production_transaction": "_model_load_admission_context",
        "anti_thrash_owner": "MLXLocalClient._ensure_worker_alive",
    }


def test_model_lane_contract_fails_closed_when_required_call_is_removed(
    tmp_path: Path,
) -> None:
    for relative in (
        "core/runtime/control_plane.py",
        "core/runtime/model_lane_control.py",
        "core/runtime/subprocess_gateway.py",
        "core/brain/llm/mlx_client.py",
        "tools/live_resource_pressure_proof.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = (ROOT / relative).read_text(encoding="utf-8")
        if relative == "core/runtime/model_lane_control.py":
            source = source.replace(
                "await self.reconcile_expired_compensations()",
                "await self._persist_missing_terminal_receipts()",
                1,
            )
        target.write_text(source, encoding="utf-8")

    report = audit(tmp_path, ownership_runner=_passing_ownership)

    assert report["passed"] is False
    assert any(
        "ModelLaneController.reserve lost calls ['reconcile_expired_compensations']"
        in issue
        for issue in report["issues"]
    )


def test_model_lane_contract_rejects_scheduler_capacity_split(tmp_path: Path) -> None:
    for relative in (
        "core/runtime/control_plane.py",
        "core/runtime/model_lane_control.py",
        "core/runtime/subprocess_gateway.py",
        "core/brain/llm/mlx_client.py",
        "tools/live_resource_pressure_proof.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = (ROOT / relative).read_text(encoding="utf-8")
        if relative == "core/brain/llm/mlx_client.py":
            source = source.replace(
                "lane_decision = await lane_controller.reserve(",
                "lane_decision = await lane_controller.cancel(",
                1,
            )
        target.write_text(source, encoding="utf-8")

    report = audit(tmp_path, ownership_runner=_passing_ownership)

    assert report["passed"] is False
    assert any(
        "_model_load_admission_context lost calls ['reserve']" in issue
        for issue in report["issues"]
    )


def test_model_lane_contract_rejects_removed_retry_storm_backoff(tmp_path: Path) -> None:
    for relative in (
        "core/runtime/control_plane.py",
        "core/runtime/model_lane_control.py",
        "core/runtime/subprocess_gateway.py",
        "core/brain/llm/mlx_client.py",
        "tools/live_resource_pressure_proof.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = (ROOT / relative).read_text(encoding="utf-8")
        if relative == "core/brain/llm/mlx_client.py":
            source = source.replace(
                "if request_is_background and self._model_load_admission_backoff_active():",
                "if request_is_background and False:",
                1,
            )
        target.write_text(source, encoding="utf-8")

    report = audit(tmp_path, ownership_runner=_passing_ownership)

    assert report["passed"] is False
    assert any(
        "MLXLocalClient._ensure_worker_alive lost calls "
        "['_model_load_admission_backoff_active']" in issue
        for issue in report["issues"]
    )


def test_model_lane_contract_propagates_ownership_failure() -> None:
    report = audit(
        ROOT,
        ownership_runner=lambda **_kwargs: {
            "passed": False,
            "findings": [{"code": "unowned_model_load"}],
        },
    )

    assert report["passed"] is False
    assert "model load ownership audit did not pass" in report["issues"]
