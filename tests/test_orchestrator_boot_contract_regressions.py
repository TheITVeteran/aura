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
    assert "_refresh_orchestrator_health_before_manifest(orchestrator, ready_label)" in boot_slice
    assert boot_slice.index("_refresh_orchestrator_health_before_manifest") < boot_slice.index(
        "_write_runtime_manifest("
    )


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
