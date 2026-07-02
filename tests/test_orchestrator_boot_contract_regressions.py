from pathlib import Path

from core.container import ServiceContainer
from core.orchestrator import RobustOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_setup_registers_output_gate_for_runtime_health_contract():
    ServiceContainer.clear()
    try:
        orchestrator = RobustOrchestrator()
        orchestrator.setup()

        assert ServiceContainer.get("output_gate", default=None) is orchestrator.output_gate
    finally:
        ServiceContainer.clear()


def test_final_boot_complete_uses_fresh_runtime_health_check():
    boot_source = (PROJECT_ROOT / "core" / "orchestrator" / "boot.py").read_text(
        encoding="utf-8"
    )
    final_boot_slice = boot_source.split("# ── Final Success State", 1)[1].split(
        "except (ImportError, AttributeError, RuntimeError) as e:",
        1,
    )[0]

    assert "self.status.healthy = bool(self.health_check())" in final_boot_slice
    assert "final runtime health check failed" in final_boot_slice
    assert "Cortex prewarm" in final_boot_slice
    assert "launcher readiness remains gated" in final_boot_slice


def test_boot_phase_health_contract_does_not_emit_runtime_critical_summary():
    boot_source = (PROJECT_ROOT / "core" / "orchestrator" / "boot.py").read_text(
        encoding="utf-8"
    )
    health_slice = boot_source.split("# ── Runtime Health Contract", 1)[1].split(
        "# ── Startup Validation",
        1,
    )[0]

    assert "runtime_ready_for_health_log" in health_slice
    assert "log_health_report() if runtime_ready_for_health_log else evaluate_health()" in health_slice
    assert "HEALTH CONTRACT DETAIL: boot pending critical liveness" in health_slice
    scheduler_slice = boot_source.split("# ── Canonical Scheduler Heartbeat", 1)[1].split(
        "# ── Runtime Health Contract",
        1,
    )[0]
    assert "Scheduler heartbeat disabled for foreground-only boot" not in scheduler_slice
    assert "await asyncio.wait_for(scheduler.start(), timeout=5.0)" in scheduler_slice


def test_canonical_boot_refreshes_health_before_manifest():
    aura_main = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    boot_slice = aura_main.split("async def _boot_runtime_orchestrator", 1)[1].split(
        "def _refresh_orchestrator_health_before_manifest",
        1,
    )[0]

    assert "await _enforce_boot_probes(ready_label)" in boot_slice
    assert "readiness_snapshot = _refresh_orchestrator_health_before_manifest(orchestrator, ready_label)" in boot_slice
    assert "readiness_snapshot=readiness_snapshot" in boot_slice
    assert "_schedule_runtime_manifest_ready_refresh(" in boot_slice
    assert "initial_readiness=readiness_snapshot" in boot_slice
    assert boot_slice.index("_refresh_orchestrator_health_before_manifest") < boot_slice.index(
        "_write_runtime_manifest("
    )


def test_runtime_manifest_pre_ready_snapshot_gets_bounded_refresh_task():
    aura_main = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    refresh_slice = aura_main.split("def _schedule_runtime_manifest_ready_refresh", 1)[1].split(
        "def _register_runtime_singletons",
        1,
    )[0]

    assert "if bool(initial_readiness.get(\"ready\")):" in refresh_slice
    assert "AURA_RUNTIME_MANIFEST_READY_REFRESH_SECONDS" in refresh_slice
    assert "_refresh_runtime_manifest_until_ready(" in refresh_slice
    assert "runtime_manifest.ready_refresh" in refresh_slice
    assert "if bool(snapshot.get(\"ready\")):" in refresh_slice
    assert "readiness_snapshot=snapshot" in refresh_slice


def test_runtime_manifest_unready_refresh_logs_on_change_not_every_poll(monkeypatch, caplog):
    # caplog, not capsys: asserting on stdout couples this test to whichever
    # root logging handlers earlier tests happened to install (observed as an
    # order-dependence failure in chunked suite runs).
    import logging

    import aura_main
    import core.runtime.health_contract as health_contract

    class Orchestrator:
        @staticmethod
        def health_check():
            return False

    def report():
        return {
            "status": "critical",
            "failures": {
                "critical": [{"container_key": "inference_gate"}],
                "important": [],
            },
        }

    monkeypatch.setattr(health_contract, "runtime_health_report", report)
    monkeypatch.setattr(health_contract, "required_probe_status", lambda contract: {})
    monkeypatch.setattr(
        health_contract,
        "required_probe_blockers",
        lambda status: ["runtime_required_probes", "probe:inference"],
    )
    monkeypatch.setattr(aura_main, "_MANIFEST_UNREADY_LOG_INTERVAL_S", 9999.0)
    aura_main._MANIFEST_UNREADY_LOG_STATE.clear()

    with caplog.at_level(logging.WARNING, logger="Aura.Main"):
        first = aura_main._refresh_orchestrator_health_before_manifest(
            Orchestrator(),
            "Server",
        )
        second = aura_main._refresh_orchestrator_health_before_manifest(
            Orchestrator(),
            "Server",
        )

    assert first["ready"] is False
    assert second["ready"] is False
    unready_warnings = [
        record
        for record in caplog.records
        if "Runtime health still not clean before manifest" in record.getMessage()
    ]
    assert len(unready_warnings) == 1


def test_runtime_manifest_records_pre_ready_boot_contract_snapshot(tmp_path):
    from core.runtime.runtime_manifest import build_runtime_manifest

    manifest = build_runtime_manifest(
        profile="desktop",
        ready_label="Server",
        project_root=PROJECT_ROOT,
        artifact_root=tmp_path,
        readiness_snapshot={
            "ready": False,
            "status": "booting",
            "critical": ["inference_gate"],
            "important": [],
            "required_probe_blockers": ["probe:inference"],
        },
    )

    assert manifest["readiness_snapshot"]["ready"] is False
    assert manifest["readiness_snapshot"]["critical"] == ["inference_gate"]
    assert manifest["readiness_snapshot"]["required_probe_blockers"] == ["probe:inference"]


def test_runtime_manifest_does_not_mark_registered_unready_role_healthy(tmp_path):
    from core.runtime.runtime_manifest import build_runtime_manifest

    class UnreadyOutputGate:
        @staticmethod
        def is_ready():
            return False

    ServiceContainer.clear()
    try:
        ServiceContainer.register_instance(
            "output_gate",
            UnreadyOutputGate(),
            required=False,
        )
        manifest = build_runtime_manifest(
            profile="desktop",
            ready_label="Server",
            project_root=PROJECT_ROOT,
            artifact_root=tmp_path,
        )

        service = manifest["services"]["output_gate"]
        role = manifest["service_roles"]["output_gate"]
        assert service["health_status"] == "liveness_failed"
        assert role["health_status"] == "liveness_failed"
        assert role["health_evidence"]["output_gate"]["liveness"] == "failed"
        assert "output_gate" in manifest["disabled_subsystems"]
    finally:
        ServiceContainer.clear()


def test_foreground_start_keeps_scheduler_heartbeat_alive():
    main_source = (PROJECT_ROOT / "core" / "orchestrator" / "main.py").read_text(
        encoding="utf-8"
    )
    scheduler_section = main_source.split(
        "# HARDENING: Register Periodic Metabolic/Substrate Tasks",
        1,
    )[1]
    foreground_slice = scheduler_section.split("if _foreground_only_runtime():", 1)[1].split(
        "else:",
        1,
    )[0]

    assert "heartbeat remains active for runtime health" in foreground_slice
    assert "if not scheduler.is_alive():" in foreground_slice
    assert "await asyncio.wait_for(scheduler.start(), timeout=5.0)" in foreground_slice


def test_foreground_boot_defers_mycelium_infrastructure_mapping():
    boot_source = (PROJECT_ROOT / "core" / "orchestrator" / "boot.py").read_text(
        encoding="utf-8"
    )
    assert boot_source.count("mycelium.setup()") == 1
    assert "orchestrator.mycelium.background_mapping" not in boot_source
    assert "mapping_scheduled = mycelium.setup()" in boot_source
    assert "mycelium.get_infrastructure_report()[\"mapping_state\"]" in boot_source


def test_orchestrator_main_loop_refreshes_watchdog_heartbeat():
    main_source = (PROJECT_ROOT / "core" / "orchestrator" / "main.py").read_text(
        encoding="utf-8"
    )
    loop_slice = main_source.split(
        "logger.info(\"🚩 [ORCHESTRATOR] Main Heartbeat Active",
        1,
    )[1].split("await asyncio.sleep(0.05)", 1)[0]

    assert "self.status.cycle_count += 1" in loop_slice
    assert "self._update_heartbeat()" in loop_slice
    assert "watchdog.heartbeat(\"orchestrator_loop\")" in loop_slice
