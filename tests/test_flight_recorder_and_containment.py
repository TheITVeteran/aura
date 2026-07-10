"""Contracts for A5 (flight-recorder event feeds), C2 (triage trend +
narrator classes), C3 (boot fault containment), C4 (startup budget gate).

The A5 ring itself (mmap codec, SIGKILL survival, death reports) is covered
by tests/test_flight_recorder.py. This suite pins the EVENT feeds folded
into that same ring: degradations, K6 condition flips, and reconciler
actions must land as crash-survivable event frames, without re-entering
the subsystems they observe.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from core.runtime.flight_recorder import (
    FlightRecorder,
    inspect_ring_file,
    set_flight_recorder_for_test,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_recorder_singleton():
    set_flight_recorder_for_test(None)
    yield
    set_flight_recorder_for_test(None)


@pytest.fixture()
def started_recorder(tmp_path):
    instance = FlightRecorder(tmp_path / "flight", slot_count=64)
    instance.start_sync()
    set_flight_recorder_for_test(instance)
    yield instance
    instance.close()


def _event_frames(instance: FlightRecorder) -> list:
    inspection = inspect_ring_file(instance.ring_path)
    assert inspection is not None and inspection.readable
    return [
        frame
        for frame in inspection.frames
        if frame.payload.get("mode") == "event"
    ]


class TestFlightRecorderEventFeeds:
    def test_degradations_feed_the_ring(self, started_recorder):
        from core.runtime.errors import record_degradation

        record_degradation(
            "flight_test_subsystem",
            RuntimeError("boom"),
            severity="warning",
            action="unit test",
        )
        events = _event_frames(started_recorder)
        assert any(
            frame.payload["stage"] == "event:degradation_warning"
            and frame.payload["extra"]["src"] == "flight_test_subsystem"
            and "boom" in frame.payload["extra"]["sum"]
            for frame in events
        ), "a warning degradation must land as an event frame"

    def test_condition_flips_feed_the_ring_creation_and_refresh_do_not(
        self, started_recorder
    ):
        from core.runtime.conditions import ConditionType, get_component_conditions

        conditions = get_component_conditions("flight_lane_test")
        conditions.set(ConditionType.READY, True, reason="Warm")  # creation ≠ flip
        conditions.set(ConditionType.READY, False, reason="Down")  # flip
        conditions.set(ConditionType.READY, False, reason="StillDown")  # refresh

        flips = [
            frame
            for frame in _event_frames(started_recorder)
            if frame.payload["stage"] == "event:condition_transition"
            and frame.payload["extra"]["src"] == "flight_lane_test"
        ]
        assert len(flips) == 1, "flips recorded; creation and refresh are not"
        assert "Ready=False" in flips[0].payload["extra"]["sum"]

    def test_reconciler_actions_feed_the_ring_with_lane(self, started_recorder):
        from core.runtime.flight_recorder import record_event

        assert record_event(
            kind="reconcile_evicted",
            source="lane_reconciler",
            summary="budget eviction",
            lane="solver",
        )
        events = _event_frames(started_recorder)
        assert events[-1].payload["extra"]["lane"] == "solver"
        assert events[-1].payload["stage"] == "event:reconcile_evicted"

    def test_event_intake_never_refreshes_slow_fields(self, started_recorder, monkeypatch):
        """Event feeds fire from inside other subsystems' locks
        (conditions.set, record_degradation). A slow-field refresh re-enters
        those subsystems (all_conditions_report / degradation tracker) —
        deadlock. Event frames must be pure memcpy."""

        def _forbidden():  # pragma: no cover - guard
            raise AssertionError("record_event triggered a slow-field refresh")

        monkeypatch.setattr(started_recorder, "_refresh_slow_fields", _forbidden)
        for index in range(2 * 5 + 1):  # cross the refresh period boundary
            assert started_recorder.record_event(
                kind="probe", source="test", summary=f"event {index}"
            )

    def test_events_ride_on_the_last_known_tick(self, started_recorder):
        started_recorder.record_frame(tick=41, stage="thinking", mode="focus")
        started_recorder.record_event(kind="probe", source="test", summary="x")
        events = _event_frames(started_recorder)
        assert events[-1].tick == 41, "events carry the last tick for ordering"

    def test_module_entry_point_is_noop_until_started(self, tmp_path):
        from core.runtime.flight_recorder import record_event

        assert record_event(kind="probe", source="test", summary="x") is False
        unstarted = FlightRecorder(tmp_path / "flight", slot_count=64)
        set_flight_recorder_for_test(unstarted)
        assert record_event(kind="probe", source="test", summary="x") is False


class TestTriageTrend:
    def test_trend_names_new_resolved_and_moved_classes(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from crash_triage import compute_trend
        finally:
            sys.path.pop(0)

        previous = {
            "generated_at": 100.0,
            "classes": [
                {"fingerprint": "stall:gateway_index", "count": 70},
                {"fingerprint": "process_death:vanished", "count": 36},
            ],
        }
        current = {
            "classes": [
                {"fingerprint": "stall:gateway_index", "count": 73},
                {"fingerprint": "stall:new_thing", "count": 2},
            ],
        }
        trend = compute_trend(current, previous)
        assert trend["new_classes"] == ["stall:new_thing"]
        assert trend["resolved_classes"] == ["process_death:vanished"]
        assert trend["count_deltas"] == {"stall:gateway_index": 3}

    def test_narrator_consumes_triage_classes(self, tmp_path):
        from core.observability.incident_narrator import IncidentNarrator

        triage = {
            "generated_at": 9_999_999_999.0,
            "classes": [
                {
                    "kind": "process_death",
                    "fingerprint": "process_death:vanished",
                    "count": 36,
                    "last_seen": 9_999_999_999.0,
                }
            ],
            "trend": {"new_classes": ["process_death:vanished"]},
        }
        (tmp_path / "triage.json").write_text(json.dumps(triage), encoding="utf-8")
        narrator = IncidentNarrator(error_log_root=tmp_path)
        items = narrator._collect_triage_classes(cutoff=0.0)
        kinds = {item.kind for item in items}
        assert "triage_class_process_death" in kinds
        assert "triage_new_class" in kinds
        by_kind = {item.kind: item for item in items}
        assert by_kind["triage_class_process_death"].severity == "critical"
        assert "36 occurrence" in by_kind["triage_class_process_death"].summary

    def test_narrator_consumes_conditions(self):
        from core.observability.incident_narrator import IncidentNarrator
        from core.runtime.conditions import ConditionType, get_component_conditions

        get_component_conditions("cortex_lane").set(
            ConditionType.READY, False, reason="CrashLoopBackOff", message="trip=2"
        )
        items = IncidentNarrator._collect_component_conditions(cutoff=0.0)
        assert any(
            "cortex_lane" in item.summary and "CrashLoopBackOff" in item.summary
            for item in items
        )


class TestBootFaultContainment:
    """C3: every boot step in the granular boot tables must execute inside
    an error boundary — one organ's failed init degrades a capability,
    never the boot."""

    def test_boot_step_tables_are_exception_bounded(self):
        boot_dir = REPO_ROOT / "core" / "orchestrator" / "mixins" / "boot"
        tables_found = 0
        for path in boot_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                # Find `for name, step in boot_steps:`-style loops and require
                # the step call inside them to be wrapped in try/except.
                if not isinstance(node, ast.AsyncFor | ast.For):
                    continue
                iter_src = ast.dump(node.iter)
                if "boot_steps" not in iter_src:
                    continue
                tables_found += 1
                has_bounded_call = any(
                    isinstance(child, ast.Try)
                    and any(
                        isinstance(grand, ast.Await | ast.Call)
                        for stmt in child.body
                        for grand in ast.walk(stmt)
                    )
                    for stmt in node.body
                    for child in ast.walk(stmt)
                    if isinstance(child, ast.Try)
                )
                assert has_bounded_call, (
                    f"{path.name}: a boot_steps loop executes steps without "
                    "a try/except boundary"
                )
        assert tables_found >= 1, "expected at least one boot_steps table"

    def test_fmea_host_and_organism_modes_have_mitigations_or_declared_gaps(self):
        from core.runtime.fmea import FMEA_REGISTRY, BlastRadius

        for mode in FMEA_REGISTRY:
            if mode.blast_radius in {BlastRadius.HOST, BlastRadius.ORGANISM}:
                mitigated = mode.mitigation.strip().upper() != "GAP"
                assert mitigated or mode.notes, (
                    f"{mode.id}: {mode.blast_radius} blast radius with neither "
                    "a mitigation nor an explanatory gap note"
                )


class TestStartupBudget:
    def test_over_budget_boot_is_a_violation(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from live_surface_probe import startup_budget_violations
        finally:
            sys.path.pop(0)

        payload = {"runtime_age_s": 400.0, "progress": 48, "status": "booting"}
        violations = startup_budget_violations(payload, budget_s=180.0)
        assert violations and "48%" in violations[0]

    def test_completed_boot_and_degraded_runtime_pass(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from live_surface_probe import startup_budget_violations
        finally:
            sys.path.pop(0)

        # Completed boot: fine at any age.
        assert startup_budget_violations(
            {"runtime_age_s": 4000.0, "progress": 100, "status": "ready"}, 180.0
        ) == []
        # Post-latch degraded presentation: NOT a startup violation.
        assert startup_budget_violations(
            {"runtime_age_s": 4000.0, "progress": 100, "status": "degraded"}, 180.0
        ) == []

    def test_wedged_startup_probe_is_a_violation(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from live_surface_probe import startup_budget_violations
        finally:
            sys.path.pop(0)

        payload = {
            "runtime_age_s": 50.0,
            "progress": 48,
            "status": "booting",
            "probes": {"startup": {"ok": False, "reason": "startup wedged"}},
        }
        violations = startup_budget_violations(payload, budget_s=180.0)
        assert any("wedged" in v for v in violations)
