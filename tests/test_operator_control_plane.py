from __future__ import annotations

import pytest

from core.runtime.control_plane import (
    DesiredServiceSpec,
    PressureSnapshot,
    ResourceAdmissionController,
    RuntimeControlPlane,
)
from core.runtime.operator_control_plane import (
    SCHEMA,
    collect_runtime_control_plane_status,
)
from core.runtime.receipts import ReceiptStore


def _admission(store: ReceiptStore) -> ResourceAdmissionController:
    return ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=40.0),
        receipt_store=store,
    )


@pytest.mark.asyncio
async def test_operator_report_unifies_healthy_runtime_state(tmp_path):
    store = ReceiptStore(tmp_path / "receipts")
    plane = RuntimeControlPlane(admission=_admission(store))
    plane.register_service(
        DesiredServiceSpec(name="scheduler", critical=True),
        start=lambda: None,
        stop=lambda: None,
        probe=lambda: True,
        adopt_running=True,
    )
    await plane.reconcile_once()

    report = collect_runtime_control_plane_status(
        plane=plane,
        receipt_store=store,
    )

    assert report["schema"] == SCHEMA
    assert report["status"] == "healthy"
    assert report["ready"] is True
    assert report["summary"]["managed_services"] == 1
    assert report["summary"]["ready_services"] == 1
    assert report["blockers"] == []
    assert report["services"]["scheduler"]["observed_state"] == "ready"
    assert "runtime_control_plane" in report["conditions"]
    assert "resource_admission" in report["conditions"]
    assert report["receipt_storage"]["high_volume_ledger_available"] is True
    assert len(report["digest"]) == 64
    store.close()


@pytest.mark.asyncio
async def test_operator_report_names_critical_open_circuit_and_remediation(tmp_path):
    store = ReceiptStore(tmp_path / "receipts")
    plane = RuntimeControlPlane(admission=_admission(store))

    async def fail_start() -> None:
        raise RuntimeError("scheduler boot failed")

    plane.register_service(
        DesiredServiceSpec(
            name="scheduler",
            critical=True,
            restart_limit=1,
        ),
        start=fail_start,
        stop=lambda: None,
        probe=lambda: False,
    )
    await plane.reconcile_once()

    report = collect_runtime_control_plane_status(
        plane=plane,
        receipt_store=store,
    )

    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["summary"]["open_circuits"] == 1
    blocker = next(item for item in report["blockers"] if item["subject"] == "scheduler")
    assert blocker["severity"] == "critical"
    assert blocker["state"] == "circuit_open"
    assert "resetting the restart circuit" in blocker["remediation"]
    assert "scheduler boot failed" in blocker["last_error"]
    store.close()


def test_reliability_router_exposes_control_plane_endpoint():
    from core.resilience.diagnostics_dashboard import create_diagnostics_router

    router = create_diagnostics_router()
    paths = {route.path for route in router.routes}

    assert "/reliability/control-plane" in paths


def test_operator_cli_uses_shared_control_plane_report(monkeypatch):
    from core.runtime import operator_cli, operator_control_plane

    monkeypatch.setattr(
        operator_control_plane,
        "collect_runtime_control_plane_status",
        lambda: {"schema": SCHEMA, "ready": True, "status": "healthy"},
    )

    result = operator_cli.run_command(["control-plane"])

    assert result == {
        "command": "control-plane",
        "ok": True,
        "report": {"schema": SCHEMA, "ready": True, "status": "healthy"},
    }


def test_operator_report_projects_shutdown_phase_and_admission(tmp_path):
    from core.runtime.shutdown_coordinator import request_shutdown

    store = ReceiptStore(tmp_path / "receipts")
    plane = RuntimeControlPlane(admission=_admission(store))
    request_shutdown("operator-report-test")

    report = collect_runtime_control_plane_status(
        plane=plane,
        receipt_store=store,
    )

    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["shutdown"]["request"]["first_reason"] == "operator-report-test"
    blocker = next(
        item for item in report["blockers"] if item["kind"] == "runtime_shutdown"
    )
    assert blocker["subject"] == "shutdown_coordinator"
    store.close()
