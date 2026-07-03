"""tests/test_reliability_hardening.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Comprehensive test suite for reliability-grade reliability hardening.

Covers all 10 phases:
1. Fault taxonomy & FMEA
2. Triple Modular Redundancy
3. Design-by-Contract
4. SLO Monitor
5. Verified State Machines
6. Chaos Framework
7. Distributed Tracing
8. Canary & Rollback
9. CI gate (tested via compile check)
10. Diagnostics Dashboard
"""
import json
import os
import tempfile
import time
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Fault Taxonomy & FMEA
# ═══════════════════════════════════════════════════════════════════════

class TestFaultTaxonomy:
    def test_registry_singleton(self):
        from core.resilience.fault_taxonomy import get_fault_registry
        r1 = get_fault_registry()
        r2 = get_fault_registry()
        assert r1 is r2

    def test_builtin_faults_registered(self):
        from core.resilience.fault_taxonomy import get_fault_registry
        reg = get_fault_registry()
        defns = reg.all_definitions()
        assert len(defns) >= 20, f"Expected >= 20 builtins, got {len(defns)}"

    def test_severity_ordering(self):
        from core.resilience.fault_taxonomy import FaultSeverity
        assert FaultSeverity.CATASTROPHIC < FaultSeverity.CRITICAL
        assert FaultSeverity.CRITICAL < FaultSeverity.MARGINAL
        assert FaultSeverity.MARGINAL < FaultSeverity.NEGLIGIBLE

    def test_rpn_calculation(self):
        from core.resilience.fault_taxonomy import (
            FaultDefinition, FaultDomain, FaultSeverity,
            FaultProbability, DetectionDifficulty, RecoveryStrategy,
        )
        defn = FaultDefinition(
            fault_id="TEST-001", name="Test",
            description="Test fault",
            domain=FaultDomain.INFERENCE,
            severity=FaultSeverity.CRITICAL,       # 2
            probability=FaultProbability.OCCASIONAL, # 3
            detection=DetectionDifficulty.HIGH,      # 2
            recovery=RecoveryStrategy.AUTOMATIC_RESTART,
            mttr_seconds=30,
            blast_radius="Test",
        )
        assert defn.rpn == 2 * 3 * 2  # 12

    def test_record_fault(self):
        from core.resilience.fault_taxonomy import FaultRegistry
        reg = FaultRegistry()
        record = reg.record_fault("F01", "test_subsystem", details="Test failure")
        assert record.fault_id == "F01"
        assert record.subsystem == "test_subsystem"
        assert reg.fault_count("F01") == 1

    def test_recent_faults_window(self):
        from core.resilience.fault_taxonomy import FaultRegistry
        reg = FaultRegistry()
        reg.record_fault("F01", "sub1", details="old")
        reg.record_fault("F02", "sub2", details="new")
        recent = reg.recent_faults(window_s=10)
        assert len(recent) == 2

    def test_rpn_report(self):
        from core.resilience.fault_taxonomy import FaultRegistry
        reg = FaultRegistry()
        report = reg.rpn_report()
        assert len(report) >= 20
        # Sorted by RPN descending
        rpns = [r["rpn"] for r in report]
        assert rpns == sorted(rpns, reverse=True)

    def test_status(self):
        from core.resilience.fault_taxonomy import FaultRegistry
        reg = FaultRegistry()
        reg.record_fault("F01", "test")
        status = reg.status()
        assert status["total_faults"] == 1
        assert "definitions_count" in status


class TestFMEARegistry:
    def test_all_faults_have_fmea_entries(self):
        from core.resilience.fmea_registry import get_fmea_registry
        fmea = get_fmea_registry()
        summary = fmea.coverage_summary()
        assert summary["coverage_pct"] >= 95

    def test_faults_above_rpn(self):
        from core.resilience.fmea_registry import get_fmea_registry
        fmea = get_fmea_registry()
        high_risk = fmea.faults_above_rpn(5)
        assert isinstance(high_risk, list)

    def test_full_report(self):
        from core.resilience.fmea_registry import get_fmea_registry
        fmea = get_fmea_registry()
        report = fmea.full_report()
        assert len(report) >= 15
        for entry in report:
            assert "fault_id" in entry
            assert "rpn" in entry
            assert "mitigations" in entry


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Triple Modular Redundancy
# ═══════════════════════════════════════════════════════════════════════

class TestTMR:
    def test_simplex_mode(self):
        from core.resilience.tmr import TMRVoter
        voter = TMRVoter("test")
        result = voter.execute(lambda: 42)
        assert result.value == 42
        assert result.unanimous

    def test_unanimous_vote(self):
        os.environ["AURA_TMR_ENABLED"] = "1"
        try:
            # Need to reimport to pick up env var
            import importlib
            import core.resilience.tmr as tmr_mod
            importlib.reload(tmr_mod)
            voter = tmr_mod.TMRVoter("test_unanimous")
            result = voter.execute(
                lambda: 42, lambda: 42, lambda: 42,
            )
            assert result.value == 42
            assert result.unanimous
            assert not result.had_divergence
        finally:
            os.environ.pop("AURA_TMR_ENABLED", None)
            importlib.reload(tmr_mod)

    def test_majority_vote_with_divergence(self):
        os.environ["AURA_TMR_ENABLED"] = "1"
        try:
            import importlib
            import core.resilience.tmr as tmr_mod
            importlib.reload(tmr_mod)
            voter = tmr_mod.TMRVoter("test_diverge")
            result = voter.execute(
                lambda: 42, lambda: 42, lambda: 99,
            )
            assert result.value == 42
            assert not result.unanimous
            assert result.had_divergence
            assert "C" in result.divergent_channels
        finally:
            os.environ.pop("AURA_TMR_ENABLED", None)
            importlib.reload(tmr_mod)

    def test_channel_error_handling(self):
        os.environ["AURA_TMR_ENABLED"] = "1"
        try:
            import importlib
            import core.resilience.tmr as tmr_mod
            importlib.reload(tmr_mod)
            voter = tmr_mod.TMRVoter("test_error")

            channel_calls: list[int] = []

            def failing():
                channel_calls.append(1)  # recording pattern: raise sites leave evidence
                raise ValueError("channel failure")

            result = voter.execute(
                lambda: 42, lambda: 42, failing,
            )
            assert result.value == 42
            assert result.had_divergence
        finally:
            os.environ.pop("AURA_TMR_ENABLED", None)
            importlib.reload(tmr_mod)

    def test_status(self):
        from core.resilience.tmr import TMRVoter
        voter = TMRVoter("test_status")
        voter.execute(lambda: 1)
        status = voter.status()
        assert status["name"] == "test_status"
        assert status["total_votes"] == 1


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Design-by-Contract
# ═══════════════════════════════════════════════════════════════════════

class TestContracts:
    def test_precondition_pass(self):
        from core.resilience.contracts import precondition

        @precondition(lambda x: x > 0, message="x must be positive")
        def double(x):
            return x * 2

        assert double(5) == 10

    def test_precondition_violation_logged(self):
        from core.resilience.contracts import precondition, get_contract_tracker

        tracker = get_contract_tracker()
        before = tracker.count("precondition")

        @precondition(lambda x: x > 0, message="x must be positive")
        def double(x):
            return x * 2

        result = double(-1)  # Violation logged, but continues
        assert result == -2
        assert tracker.count("precondition") > before

    def test_postcondition_pass(self):
        from core.resilience.contracts import postcondition

        @postcondition(lambda r: r > 0, message="result must be positive")
        def abs_val(x):
            return abs(x)

        assert abs_val(-5) == 5

    def test_postcondition_violation_logged(self):
        from core.resilience.contracts import postcondition, get_contract_tracker

        tracker = get_contract_tracker()
        before = tracker.count("postcondition")

        @postcondition(lambda r: r > 0, message="result must be positive")
        def negate(x):
            return -x

        result = negate(5)  # Returns -5, postcondition fails
        assert result == -5
        assert tracker.count("postcondition") > before

    def test_tracker_status(self):
        from core.resilience.contracts import get_contract_tracker
        tracker = get_contract_tracker()
        status = tracker.status()
        assert "total_violations" in status
        assert "by_kind" in status
        assert status["enforcement_mode"] == "log_continue"


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: SLO Monitor
# ═══════════════════════════════════════════════════════════════════════

class TestSLOMonitor:
    def test_default_slos_registered(self):
        from slo.slo_monitor import SLOMonitor
        mon = SLOMonitor()
        status = mon.status()
        assert status["slo_count"] >= 25

    def test_record_within_slo(self):
        from slo.slo_monitor import SLOMonitor
        mon = SLOMonitor()
        within = mon.record("boot_cold_p95_ms", 5000)
        assert within is True

    def test_record_violation(self):
        from slo.slo_monitor import SLOMonitor
        mon = SLOMonitor()
        within = mon.record("boot_cold_p95_ms", 999999)
        assert within is False

    def test_burn_rate(self):
        from slo.slo_monitor import SLOMonitor
        mon = SLOMonitor()
        # Record many violations to trigger burn rate
        for _ in range(100):
            mon.record("boot_cold_p95_ms", 999999)
        tracker = mon.get_tracker("boot_cold_p95_ms")
        assert tracker is not None
        assert tracker.burn_rate() > 1.0
        assert tracker.budget_status().value in ("SLOW_BURN", "FAST_BURN", "EXHAUSTED")

    def test_budget_remaining(self):
        from slo.slo_monitor import SLOMonitor
        mon = SLOMonitor()
        for _ in range(10):
            mon.record("will_decision_p95_ms", 1.0)  # Within SLO
        tracker = mon.get_tracker("will_decision_p95_ms")
        assert tracker is not None
        assert tracker.budget_remaining_pct() == 100.0

    def test_percentile(self):
        from slo.slo_monitor import SLOMonitor
        mon = SLOMonitor()
        for i in range(100):
            mon.record("health_endpoint_p99_ms", float(i))
        tracker = mon.get_tracker("health_endpoint_p99_ms")
        p50 = tracker.percentile(50)
        assert p50 is not None
        assert 45 <= p50 <= 55


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Verified State Machines
# ═══════════════════════════════════════════════════════════════════════

class TestVerifiedStateMachine:
    def test_valid_machine_verifies(self):
        from core.resilience.verified_state_machine import create_component_health_machine
        sm = create_component_health_machine()
        warnings = sm.verify()
        assert isinstance(warnings, list)

    def test_transition_succeeds(self):
        from core.resilience.verified_state_machine import create_component_health_machine
        sm = create_component_health_machine()
        assert sm.current == "HEALTHY"
        record = sm.transition("DEGRADED")
        assert record.guard_passed
        assert sm.current == "DEGRADED"

    def test_illegal_transition_raises(self):
        from core.resilience.verified_state_machine import (
            create_component_health_machine, IllegalTransitionError,
        )
        sm = create_component_health_machine()
        with pytest.raises(IllegalTransitionError):
            sm.transition("FAILED")  # HEALTHY → FAILED not allowed

    def test_deadlock_detection(self):
        from core.resilience.verified_state_machine import (
            VerifiedStateMachine, DeadlockDetectedError,
        )
        with pytest.raises(DeadlockDetectedError):
            sm = VerifiedStateMachine(
                name="deadlock_test",
                states={"A", "B"},
                initial="A",
                transitions={("A", "B"): None},
                # B has no outgoing transitions and is not terminal
            )
            sm.verify()

    def test_unreachable_state_detection(self):
        from core.resilience.verified_state_machine import (
            VerifiedStateMachine, UnreachableStateError,
        )
        with pytest.raises(UnreachableStateError):
            sm = VerifiedStateMachine(
                name="unreachable_test",
                states={"A", "B", "C"},
                initial="A",
                transitions={
                    ("A", "B"): None,
                    ("B", "A"): None,
                    # C is unreachable
                    ("C", "A"): None,
                },
            )
            sm.verify()

    def test_guard_rejection(self):
        from core.resilience.verified_state_machine import VerifiedStateMachine
        sm = VerifiedStateMachine(
            name="guard_test",
            states={"A", "B"},
            initial="A",
            transitions={
                ("A", "B"): lambda: False,  # Guard always rejects
                ("B", "A"): None,
            },
        )
        record = sm.transition("B")
        assert not record.guard_passed
        assert sm.current == "A"  # Didn't transition

    def test_boot_lifecycle_machine(self):
        from core.resilience.verified_state_machine import create_boot_lifecycle_machine
        sm = create_boot_lifecycle_machine()
        sm.transition("LOADING_CONFIG")
        sm.transition("LOADING_MODEL")
        sm.transition("STARTING_SERVICES")
        sm.transition("HEALTH_CHECK")
        sm.transition("READY")
        assert sm.current == "READY"

    def test_shutdown_lifecycle_machine(self):
        from core.resilience.verified_state_machine import create_shutdown_lifecycle_machine
        sm = create_shutdown_lifecycle_machine()
        sm.transition("DRAINING")
        sm.transition("STOPPING_SERVICES")
        sm.transition("FLUSHING_STATE")
        sm.transition("TERMINATED")
        assert sm.current == "TERMINATED"

    def test_status(self):
        from core.resilience.verified_state_machine import create_component_health_machine
        sm = create_component_health_machine()
        status = sm.status()
        assert status["current"] == "HEALTHY"
        assert "HEALTHY" in status["states"]


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Chaos Framework
# ═══════════════════════════════════════════════════════════════════════

class TestChaosFramework:
    def test_memory_pressure_fault(self):
        from tools.chaos.chaos_framework import MemoryPressureFault
        fault = MemoryPressureFault(target_mb=1)
        fault.inject()
        fault.revert()
        assert fault.name() == "memory_pressure_1mb"

    def test_latency_fault(self):
        from tools.chaos.chaos_framework import LatencyFault
        fault = LatencyFault(delay_ms=10, jitter_ms=0)
        fault.inject()
        t0 = time.time()
        fault.apply_delay()
        elapsed = time.time() - t0
        assert elapsed >= 0.005  # At least 5ms (some tolerance)
        fault.revert()
        fault.apply_delay()  # Should be instant now

    def test_experiment_passes(self):
        from tools.chaos.chaos_framework import (
            ChaosFramework, ChaosExperiment, LatencyFault,
        )
        framework = ChaosFramework()
        experiment = ChaosExperiment(
            name="test_pass",
            hypothesis="No-op fault doesn't break anything",
            fault=LatencyFault(delay_ms=0),
            duration_s=0.1,
            success_criteria=lambda m: True,
        )
        result = framework.run(experiment)
        assert result.status.value == "passed"

    def test_kill_switch(self):
        from tools.chaos.chaos_framework import ChaosFramework
        framework = ChaosFramework()
        assert framework.status()["running"] is False

    def test_pass_rate(self):
        from tools.chaos.chaos_framework import ChaosFramework
        framework = ChaosFramework()
        assert framework.pass_rate() == 1.0  # No experiments = 100%


# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Distributed Tracing
# ═══════════════════════════════════════════════════════════════════════

class TestTracing:
    def test_span_creation(self):
        from core.observability.tracing import Tracer
        tracer = Tracer(enabled=True, sample_rate=1.0)
        with tracer.span("test_op") as span:
            span.set_attribute("key", "value")
        assert span.duration_ms is not None
        assert span.status.value == "OK"

    def test_nested_spans(self):
        from core.observability.tracing import Tracer
        tracer = Tracer(enabled=True, sample_rate=1.0)
        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                pass
        assert child.parent_span_id == parent.span_id
        assert child.trace_id == parent.trace_id

    def test_error_recording(self):
        from core.observability.tracing import Tracer
        tracer = Tracer(enabled=True, sample_rate=1.0)
        with pytest.raises(ValueError):
            with tracer.span("failing") as span:
                raise ValueError("test error")
        assert span.status.value == "ERROR"
        assert len(span.events) == 1
        assert span.events[0].name == "exception"

    def test_disabled_tracing(self):
        from core.observability.tracing import Tracer
        tracer = Tracer(enabled=False)
        with tracer.span("noop") as span:
            span.set_attribute("key", "val")
        assert span.trace_id == "0"  # Noop span

    def test_otlp_export(self):
        from core.observability.tracing import Tracer
        tracer = Tracer(enabled=True, sample_rate=1.0)
        with tracer.span("export_test"):
            pass
        export = tracer.export_json(limit=5)
        data = json.loads(export)
        assert "resourceSpans" in data

    def test_status(self):
        from core.observability.tracing import Tracer
        tracer = Tracer(enabled=True, sample_rate=1.0)
        with tracer.span("status_test"):
            pass
        status = tracer.status()
        assert status["enabled"] is True
        assert status["total_spans"] >= 1


# ═══════════════════════════════════════════════════════════════════════
# Phase 8: Canary & Rollback
# ═══════════════════════════════════════════════════════════════════════

class TestCanary:
    def test_canary_passes(self):
        from infrastructure.canary import CanaryController, CanaryConfig, CanaryMetrics
        config = CanaryConfig(phase_duration_s=0.01)  # Fast test
        controller = CanaryController(config=config)
        result = controller.run_canary(
            canary_fn=lambda: None,
            collect_metrics=lambda: CanaryMetrics(
                error_rate=0.01, latency_p95_ms=50,
            ),
        )
        assert result.passed

    def test_canary_rollback_on_error_rate(self):
        from infrastructure.canary import CanaryController, CanaryConfig, CanaryMetrics
        config = CanaryConfig(phase_duration_s=0.01, max_error_rate=0.05)
        controller = CanaryController(config=config)

        call_count = 0
        def metrics():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CanaryMetrics(error_rate=0.01, latency_p95_ms=50)
            return CanaryMetrics(error_rate=0.20, latency_p95_ms=50)  # High error

        result = controller.run_canary(
            canary_fn=lambda: None,
            collect_metrics=metrics,
        )
        assert not result.passed
        assert "Error rate" in result.rollback_reason

    def test_smoke_test_failure(self):
        from infrastructure.canary import CanaryController, CanaryConfig, CanaryMetrics
        config = CanaryConfig(phase_duration_s=0.01)
        controller = CanaryController(config=config)
        result = controller.run_canary(
            canary_fn=lambda: None,
            collect_metrics=lambda: CanaryMetrics(),
            smoke_test=lambda: False,
        )
        assert not result.passed


class TestRollback:
    def test_checkpoint_and_rollback_applies_state(self):
        from infrastructure.rollback import RollbackController
        with tempfile.TemporaryDirectory() as tmp:
            controller = RollbackController(checkpoint_dir=Path(tmp))
            applied: list[dict] = []
            controller.register_state_applier(lambda s: applied.append(s) or True)
            cp = controller.checkpoint("v1.0", state_collector=lambda: {"key": "val"})
            assert cp.verified
            success = controller.rollback("v1.0")
            assert success
            assert applied == [{"key": "val"}]  # state actually restored

    def test_rollback_with_state_but_no_applier_fails_closed(self):
        """A rollback that cannot restore its persisted state must not
        report success — pretending to restore is worse than failing."""
        from infrastructure.rollback import RollbackController
        with tempfile.TemporaryDirectory() as tmp:
            controller = RollbackController(checkpoint_dir=Path(tmp))
            controller.checkpoint("v1.0", state_collector=lambda: {"key": "val"})
            assert controller.rollback("v1.0") is False

    def test_stateless_checkpoint_rollback_succeeds_without_applier(self):
        from infrastructure.rollback import RollbackController
        with tempfile.TemporaryDirectory() as tmp:
            controller = RollbackController(checkpoint_dir=Path(tmp))
            controller.checkpoint("marker-only")
            assert controller.rollback("marker-only") is True

    def test_verify_hooks(self):
        from infrastructure.rollback import RollbackController
        controller = RollbackController()
        controller.register_verify_hook(lambda: True)
        controller.register_verify_hook(lambda: True)
        assert controller.verify() is True

    def test_status(self):
        from infrastructure.rollback import RollbackController
        controller = RollbackController()
        controller.checkpoint("test")
        status = controller.status()
        assert status["checkpoint_count"] == 1


# ═══════════════════════════════════════════════════════════════════════
# Phase 10: Diagnostics Dashboard
# ═══════════════════════════════════════════════════════════════════════

class TestDiagnosticsDashboard:
    def test_collect_diagnostics(self):
        from core.resilience.diagnostics_dashboard import collect_reliability_diagnostics
        diag = collect_reliability_diagnostics()
        assert "subsystems" in diag
        assert "summary" in diag
        assert diag["summary"]["total_subsystems"] >= 4

    def test_fault_taxonomy_in_diagnostics(self):
        from core.resilience.diagnostics_dashboard import collect_reliability_diagnostics
        diag = collect_reliability_diagnostics()
        assert "fault_taxonomy" in diag["subsystems"]

    def test_slo_monitor_in_diagnostics(self):
        from core.resilience.diagnostics_dashboard import collect_reliability_diagnostics
        diag = collect_reliability_diagnostics()
        assert "slo_monitor" in diag["subsystems"]

    def test_contracts_in_diagnostics(self):
        from core.resilience.diagnostics_dashboard import collect_reliability_diagnostics
        diag = collect_reliability_diagnostics()
        assert "contracts" in diag["subsystems"]


# ═══════════════════════════════════════════════════════════════════════
# Hardening regressions: defects found in the reliability-stack audit.
# Every test here pins a bug that shipped in the first cut — deadlocks,
# severity fidelity, alert storms, dead SLOs, broken traceability.
# ═══════════════════════════════════════════════════════════════════════

class TestAuditRegressions:
    def test_fault_registry_status_does_not_deadlock(self):
        """status() used to call rpn_report() while holding the non-reentrant
        registry lock — the first health query hung its thread forever."""
        import threading

        from core.resilience.fault_taxonomy import FaultRegistry

        reg = FaultRegistry()
        reg.record_fault("F01", "test")
        result: list[dict] = []
        worker = threading.Thread(target=lambda: result.append(reg.status()))
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive(), "FaultRegistry.status() deadlocked"
        assert result and result[0]["total_faults"] == 1
        assert len(result[0]["top_rpn"]) == 5

    def test_chaos_status_does_not_deadlock(self):
        """ChaosFramework.status() had the same held-lock self-call via
        pass_rate()."""
        import threading

        from tools.chaos.chaos_framework import ChaosFramework

        framework = ChaosFramework()
        result: list[dict] = []
        worker = threading.Thread(target=lambda: result.append(framework.status()))
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive(), "ChaosFramework.status() deadlocked"
        assert result and result[0]["pass_rate"] == 1.0

    def test_record_fault_severity_override(self):
        """Undefined fault IDs default to MARGINAL; callers with live
        severity knowledge (record_degradation) must be able to keep it."""
        from core.resilience.fault_taxonomy import FaultRegistry, FaultSeverity

        reg = FaultRegistry()
        rec = reg.record_fault(
            "RUNTIME-TEST-SUBSYS", "test", severity=FaultSeverity.CRITICAL,
        )
        assert rec.severity == FaultSeverity.CRITICAL

    def test_degradation_severity_reaches_fault_registry(self):
        """A critical degradation must land as a CRITICAL fault record, not
        the MARGINAL default (the _sev_map used to be built and dropped)."""
        from core.resilience.fault_taxonomy import FaultSeverity, get_fault_registry
        from core.runtime.errors import record_degradation

        registry = get_fault_registry()
        before = registry.fault_count("RUNTIME-AEROTEST-SEV")
        record_degradation(
            "aerotest.sev", ValueError("boom"), severity="critical",
            action="test-only",
        )
        assert registry.fault_count("RUNTIME-AEROTEST-SEV") == before + 1
        recent = registry.faults_by_subsystem("aerotest.sev")
        assert recent and recent[-1].severity == FaultSeverity.CRITICAL

    def test_count_per_window_slo_actually_violates(self):
        """error_events_per_hour records value 1.0 per event; the old
        per-sample comparison (1.0 > 10) could never violate."""
        from slo.slo_monitor import SLOMonitor

        mon = SLOMonitor()
        results = [mon.record("error_events_per_hour", 1.0) for _ in range(12)]
        assert all(results[:10]), "first 10 events are within budget"
        assert results[10] is False, "11th event in the window must violate"
        assert results[11] is False

    def test_slo_alert_cooldown_prevents_storm(self):
        """A persistently violating SLO must not append one alert per sample."""
        from slo.slo_monitor import SLOMonitor

        mon = SLOMonitor()
        for _ in range(200):
            mon.record("boot_cold_p95_ms", 999999.0)
        alerts = mon.status()["recent_alerts"]
        # Cooldown is 60s: a tight loop lands exactly one alert.
        assert len([a for a in alerts if a["slo"] == "boot_cold_p95_ms"]) == 1

    def test_slo_concurrent_record_and_status(self):
        """Sample recording races status() window scans; the deque must be
        lock-protected (unlocked iteration raises RuntimeError)."""
        import threading

        from slo.slo_monitor import SLOMonitor

        mon = SLOMonitor()
        errors: list[BaseException] = []

        def hammer_records():
            try:
                for i in range(2000):
                    mon.record("tick_duration_p95_ms", float(i % 700))
            except (RuntimeError, ValueError) as exc:
                errors.append(exc)

        def hammer_status():
            try:
                for _ in range(60):
                    mon.status()
            except (RuntimeError, ValueError) as exc:
                errors.append(exc)

        threads = [threading.Thread(target=hammer_records) for _ in range(3)]
        threads += [threading.Thread(target=hammer_status) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
        assert not errors, f"concurrent SLO access raised: {errors[:3]}"

    def test_state_machine_guard_may_reenter_machine(self):
        """Guards/actions run under the machine lock; a guard consulting the
        machine (current, allowed_transitions) used to self-deadlock."""
        import threading

        from core.resilience.verified_state_machine import VerifiedStateMachine

        sm = VerifiedStateMachine(
            name="reentrant_guard",
            states={"A", "B"},
            initial="A",
            transitions={
                ("A", "B"): None,
                ("B", "A"): None,
            },
        )
        sm._transitions[("A", "B")] = lambda: sm.current == "A"  # reentrant guard
        done: list[str] = []
        worker = threading.Thread(target=lambda: done.append(sm.transition("B").to_state))
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive(), "reentrant guard deadlocked the machine"
        assert done == ["B"] and sm.current == "B"

    def test_invariant_decorator_leaves_static_and_nested_alone(self):
        """@invariant used to wrap ANY public callable attribute — corrupting
        staticmethods, classmethods, and nested classes."""
        from core.resilience.contracts import invariant

        @invariant(lambda self: self.n >= 0, message="n must stay >= 0")
        class Counter:
            class Config:  # nested class must survive untouched
                flag = True

            def __init__(self) -> None:
                self.n = 0

            def bump(self) -> int:
                self.n += 1
                return self.n

            @staticmethod
            def double(x: int) -> int:
                return x * 2

            @classmethod
            def make(cls) -> "Counter":
                return cls()

        c = Counter()
        assert c.bump() == 1                  # instance method wrapped + works
        assert Counter.double(4) == 8         # staticmethod not rebound
        assert isinstance(Counter.make(), Counter)  # classmethod not rebound
        assert Counter.Config.flag is True    # nested class untouched

    def test_trace_sampling_is_per_trace_not_per_span(self):
        """Head-based sampling: children re-rolling the dice produced orphan
        spans and broken traces. The root decision must propagate."""
        from core.observability.tracing import Tracer

        tracer = Tracer(enabled=True, sample_rate=0.0)  # roots never sampled
        with tracer.span("root") as root:
            with tracer.span("child") as child:
                pass
        assert root.trace_id == "0" and child.trace_id == "0"  # both dropped

        tracer_all = Tracer(enabled=True, sample_rate=1.0)
        with tracer_all.span("root") as root:
            with tracer_all.span("child") as child:
                pass
        assert child.parent_span_id == root.span_id  # both kept, linked

    def test_force_sample_survives_unsampled_root(self):
        from core.observability.tracing import Tracer

        tracer = Tracer(enabled=True, sample_rate=0.0)
        with tracer.span("root"):
            with tracer.span("forensic", force_sample=True) as span:
                pass
        assert span.trace_id != "0"

    def test_will_refuse_fault_is_negligible_and_recovered(self):
        """Refusals are governance working as designed — they must not
        pollute fault health as unrecovered MARGINAL faults."""
        from core.resilience.fault_taxonomy import (
            FaultRegistry, FaultSeverity,
        )

        reg = FaultRegistry()
        defn = reg.get_definition("WILL-REFUSE")
        assert defn is not None
        assert defn.severity == FaultSeverity.NEGLIGIBLE
        rec = reg.record_fault("WILL-REFUSE", "will.test", recovered=True)
        assert rec.recovered and rec.severity == FaultSeverity.NEGLIGIBLE


class TestTraceability:
    """FMEA/runbook linkage must resolve to real repo artifacts — a fault
    catalog pointing at files that don't exist is theater."""

    def test_every_runbook_reference_resolves(self):
        from core.resilience.fault_taxonomy import get_fault_registry

        repo = Path(__file__).resolve().parents[1]
        missing = [
            (d.fault_id, d.runbook)
            for d in get_fault_registry().all_definitions()
            if d.runbook and not (repo / d.runbook).is_file()
        ]
        assert not missing, f"fault runbooks missing on disk: {missing}"

    def test_every_mitigation_path_resolves(self):
        from core.resilience.fmea_registry import get_fmea_registry

        repo = Path(__file__).resolve().parents[1]
        missing = []
        for entry in get_fmea_registry().full_report():
            for mit in entry["mitigations"]:
                if mit["impl"] and not (repo / mit["impl"]).exists():
                    missing.append((mit["action_id"], mit["impl"]))
        assert not missing, f"FMEA mitigation paths missing on disk: {missing}"
