from __future__ import annotations

import asyncio
import time

import pytest

from core.runtime.control_plane import (
    AdmissionOutcome,
    AdmissionPriority,
    AdmissionRequest,
    DesiredServiceSpec,
    DesiredServiceState,
    ObservedServiceState,
    PressureSnapshot,
    ResourceAdmissionController,
    RuntimeControlPlane,
    WorkClass,
)
from core.runtime.receipts import ReceiptStore


def _normal_pressure() -> PressureSnapshot:
    return PressureSnapshot(memory_percent=42.0, thermal_level=0, loop_lag_s=0.01)


@pytest.mark.asyncio
async def test_background_admission_fails_closed_under_memory_pressure(tmp_path):
    store = ReceiptStore(tmp_path / "receipts")
    controller = ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=94.0),
        receipt_store=store,
    )
    request = AdmissionRequest(
        owner="autonomy.research",
        work_class=WorkClass.BACKGROUND,
        priority=AdmissionPriority.BACKGROUND,
        timeout_s=0,
    )

    decision = await controller.acquire(request)

    assert decision.outcome == AdmissionOutcome.DEFERRED
    assert decision.reason == "critical_memory_pressure_94.0"
    assert decision.receipt_id
    receipt = store.get(decision.receipt_id)
    assert receipt.kind == "resource_admission"
    assert receipt.decision == "deferred"
    assert receipt.pressure["memory_percent"] == 94.0
    reloaded = ReceiptStore(tmp_path / "receipts")
    assert reloaded.reload_from_disk() == 1
    durable = reloaded.get(decision.receipt_id)
    assert durable is not None
    assert durable.kind == "resource_admission"
    assert durable.request_id == request.request_id


@pytest.mark.asyncio
async def test_background_admission_recovers_after_fresh_healthy_loop_sample():
    pressure = {
        "loop_lag_s": 0.0,
        "loop_lag_sample_age_s": 20.0,
        "loop_lag_sample_fresh": False,
        "loop_monitor_alive": True,
    }
    controller = ResourceAdmissionController(pressure_provider=lambda: dict(pressure))
    controller._pressure_cache_s = 0.0

    def request() -> AdmissionRequest:
        return AdmissionRequest(
            owner="mlx.model_load:background",
            work_class=WorkClass.MODEL_LOAD,
            priority=AdmissionPriority.BACKGROUND,
            timeout_s=0,
        )

    stale = await controller.acquire(request())

    assert stale.outcome == AdmissionOutcome.DEFERRED
    assert stale.reason == "event_loop_signal_unavailable"

    pressure.update(
        loop_lag_s=0.012,
        loop_lag_sample_age_s=0.05,
        loop_lag_sample_fresh=True,
    )
    recovered = await controller.acquire(request())

    assert recovered.admitted is True
    await controller.release(recovered.lease_id)


@pytest.mark.asyncio
async def test_repeated_unaudited_denials_coalesce_until_state_changes(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("AURA_ADMISSION_RECEIPT_HEARTBEAT_S", "3600")
    pressure = {"memory": 94.0}
    store = ReceiptStore(tmp_path / "receipts")
    controller = ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=pressure["memory"]),
        receipt_store=store,
    )
    controller._pressure_cache_s = 0.0

    def request() -> AdmissionRequest:
        return AdmissionRequest(
            owner="autonomy.research",
            work_class=WorkClass.BACKGROUND,
            priority=AdmissionPriority.BACKGROUND,
            timeout_s=0,
        )

    first = await controller.acquire(request())
    pressure["memory"] = 94.7
    second = await controller.acquire(request())

    assert second.receipt_id == first.receipt_id
    assert second.receipt_replayed is True
    assert store.coverage_stats()["resource_admission"] == 1
    assert controller.status()["counters"]["receipt_coalesced"] == 1

    pressure["memory"] = 40.0
    recovered = await controller.acquire(request())
    assert recovered.admitted is True
    await controller.release(recovered.lease_id)

    pressure["memory"] = 95.0
    regressed = await controller.acquire(request())
    assert regressed.receipt_id != first.receipt_id
    assert regressed.receipt_replayed is False
    assert store.coverage_stats()["resource_admission"] == 2


@pytest.mark.asyncio
async def test_audited_denials_are_never_coalesced(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_ADMISSION_RECEIPT_HEARTBEAT_S", "3600")
    store = ReceiptStore(tmp_path / "receipts")
    controller = ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=94.0),
        receipt_store=store,
    )

    def request() -> AdmissionRequest:
        return AdmissionRequest(
            owner="mlx.model_load:cortex",
            work_class=WorkClass.MODEL_LOAD,
            lane="cortex",
            priority=AdmissionPriority.FOREGROUND,
            timeout_s=0,
            receipt_required=True,
        )

    first = await controller.acquire(request())
    second = await controller.acquire(request())

    assert first.receipt_id != second.receipt_id
    assert first.receipt_replayed is False
    assert second.receipt_replayed is False
    assert store.coverage_stats()["resource_admission"] == 2


@pytest.mark.asyncio
async def test_zero_receipt_heartbeat_disables_coalescing(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_ADMISSION_RECEIPT_HEARTBEAT_S", "0")
    store = ReceiptStore(tmp_path / "receipts")
    controller = ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=94.0),
        receipt_store=store,
    )
    request = lambda: AdmissionRequest(
        owner="maintenance",
        work_class=WorkClass.BACKGROUND,
        priority=AdmissionPriority.BACKGROUND,
        timeout_s=0,
    )

    first = await controller.acquire(request())
    second = await controller.acquire(request())

    assert first.receipt_id != second.receipt_id
    assert store.coverage_stats()["resource_admission"] == 2


@pytest.mark.asyncio
async def test_receipt_coalescing_state_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_ADMISSION_RECEIPT_HEARTBEAT_S", "3600")
    controller = ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=94.0),
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        history_limit=16,
    )

    for index in range(70):
        await controller.acquire(
            AdmissionRequest(
                owner=f"background-owner-{index}",
                work_class=WorkClass.BACKGROUND,
                priority=AdmissionPriority.BACKGROUND,
                timeout_s=0,
            )
        )

    assert controller.status()["receipt_state_count"] == 64


@pytest.mark.asyncio
async def test_foreground_inference_remains_admissible_under_moderate_pressure():
    controller = ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=87.0),
    )
    request = AdmissionRequest(
        owner="chat.foreground",
        work_class=WorkClass.INFERENCE,
        lane="cortex",
        priority=AdmissionPriority.FOREGROUND,
        timeout_s=0,
    )

    decision = await controller.acquire(request)

    assert decision.admitted is True
    await controller.release(decision.lease_id)


@pytest.mark.asyncio
async def test_same_lane_serializes_and_request_id_is_idempotent():
    controller = ResourceAdmissionController(pressure_provider=_normal_pressure)
    first_request = AdmissionRequest(
        owner="kernel",
        work_class=WorkClass.INFERENCE,
        lane="cortex",
        request_id="stable-request",
        timeout_s=0,
    )
    first = await controller.acquire(first_request)
    replay = await controller.acquire(first_request)
    blocked = await controller.acquire(
        AdmissionRequest(
            owner="second",
            work_class=WorkClass.INFERENCE,
            lane="cortex",
            timeout_s=0,
        )
    )

    assert first.admitted is True
    assert replay.admitted is True
    assert replay.replayed is True
    assert replay.lease_id == first.lease_id
    assert blocked.outcome == AdmissionOutcome.DEFERRED
    assert blocked.blocking_lease_ids == (first.lease_id,)

    await controller.release(first.lease_id)
    with pytest.raises(KeyError):
        await controller.release(first.lease_id)


@pytest.mark.asyncio
async def test_different_inference_lanes_can_run_concurrently():
    controller = ResourceAdmissionController(pressure_provider=_normal_pressure)
    cortex = await controller.acquire(
        AdmissionRequest(owner="chat", work_class=WorkClass.INFERENCE, lane="cortex", timeout_s=0)
    )
    brainstem = await controller.acquire(
        AdmissionRequest(
            owner="ambient",
            work_class=WorkClass.INFERENCE,
            lane="brainstem",
            timeout_s=0,
        )
    )

    assert cortex.admitted and brainstem.admitted
    assert controller.active_lease_count(WorkClass.INFERENCE) == 2

    await controller.release(cortex.lease_id)
    await controller.release(brainstem.lease_id)


@pytest.mark.asyncio
async def test_same_lane_model_load_can_nest_under_inference_reservation():
    controller = ResourceAdmissionController(pressure_provider=_normal_pressure)
    inference = await controller.acquire(
        AdmissionRequest(
            owner="inference_gate",
            work_class=WorkClass.INFERENCE,
            lane="cortex",
            timeout_s=0,
        )
    )

    model_load = await controller.acquire(
        AdmissionRequest(
            owner="mlx.model_load",
            work_class=WorkClass.MODEL_LOAD,
            lane="cortex",
            timeout_s=0,
        )
    )

    assert inference.admitted and model_load.admitted
    await controller.release(model_load.lease_id)
    await controller.release(inference.lease_id)


@pytest.mark.asyncio
async def test_model_load_is_blocked_by_another_inference_lane():
    controller = ResourceAdmissionController(pressure_provider=_normal_pressure)
    brainstem = await controller.acquire(
        AdmissionRequest(
            owner="ambient_inference",
            work_class=WorkClass.INFERENCE,
            lane="brainstem",
            timeout_s=0,
        )
    )

    cortex_load = await controller.acquire(
        AdmissionRequest(
            owner="mlx.model_load",
            work_class=WorkClass.MODEL_LOAD,
            lane="cortex",
            timeout_s=0,
        )
    )

    assert cortex_load.outcome == AdmissionOutcome.DEFERRED
    assert cortex_load.blocking_lease_ids == (brainstem.lease_id,)
    await controller.release(brainstem.lease_id)


@pytest.mark.asyncio
async def test_legacy_global_inference_scope_conflicts_with_every_lane():
    controller = ResourceAdmissionController(pressure_provider=_normal_pressure)
    legacy = await controller.acquire(
        AdmissionRequest(
            owner="legacy",
            work_class=WorkClass.INFERENCE,
            lane="legacy_global_inference",
            timeout_s=0,
            metadata={"global_inference_scope": True},
        )
    )

    cortex = await controller.acquire(
        AdmissionRequest(
            owner="chat",
            work_class=WorkClass.INFERENCE,
            lane="cortex",
            timeout_s=0,
        )
    )

    assert cortex.outcome == AdmissionOutcome.DEFERRED
    assert cortex.blocking_lease_ids == (legacy.lease_id,)
    controller.release_sync(legacy.lease_id, reason="legacy_test_finished")
    assert controller.active_lease_count(WorkClass.INFERENCE) == 0


@pytest.mark.asyncio
async def test_sync_release_rejects_receipt_bearing_lease_without_losing_it(tmp_path):
    controller = ResourceAdmissionController(
        pressure_provider=_normal_pressure,
        receipt_store=ReceiptStore(tmp_path / "receipts"),
    )
    admitted = await controller.acquire(
        AdmissionRequest(
            owner="audited",
            work_class=WorkClass.MAINTENANCE,
            lane="audit",
            timeout_s=0,
            receipt_required=True,
        )
    )

    with pytest.raises(RuntimeError, match="require async release"):
        controller.release_sync(admitted.lease_id)
    assert controller.active_lease_count() == 1

    await controller.release(admitted.lease_id)


@pytest.mark.asyncio
async def test_evolution_conflicts_with_active_inference_and_times_out(tmp_path):
    store = ReceiptStore(tmp_path / "receipts")
    controller = ResourceAdmissionController(
        pressure_provider=_normal_pressure,
        receipt_store=store,
        poll_interval_s=0.01,
    )
    inference = await controller.acquire(
        AdmissionRequest(owner="chat", work_class=WorkClass.INFERENCE, lane="cortex", timeout_s=0)
    )

    evolution = await controller.acquire(
        AdmissionRequest(
            owner="hephaestus",
            work_class=WorkClass.EVOLUTION,
            priority=AdmissionPriority.BACKGROUND,
            timeout_s=0.03,
        )
    )

    assert evolution.outcome == AdmissionOutcome.TIMED_OUT
    assert evolution.reason == "resource_timeout"
    assert evolution.blocking_lease_ids == (inference.lease_id,)
    assert evolution.receipt_id
    await controller.release(inference.lease_id)


@pytest.mark.asyncio
async def test_foreground_request_can_cooperatively_preempt_lower_priority_work():
    controller = ResourceAdmissionController(
        pressure_provider=_normal_pressure,
        poll_interval_s=0.01,
    )
    evolution_lease_id = ""
    preempt_reasons: list[str] = []

    async def release_on_preempt(reason: str) -> None:
        preempt_reasons.append(reason)
        await controller.release(
            evolution_lease_id,
            reason=reason,
            preempted=True,
        )

    evolution = await controller.acquire(
        AdmissionRequest(
            owner="training",
            work_class=WorkClass.EVOLUTION,
            priority=AdmissionPriority.BACKGROUND,
            preemptible=True,
            timeout_s=0,
        ),
        on_preempt=release_on_preempt,
    )
    evolution_lease_id = evolution.lease_id

    foreground = await controller.acquire(
        AdmissionRequest(
            owner="chat",
            work_class=WorkClass.INFERENCE,
            lane="cortex",
            priority=AdmissionPriority.FOREGROUND,
            timeout_s=0.2,
        )
    )

    assert foreground.admitted is True
    assert preempt_reasons == [f"preempted_by:{foreground.request_id}"]
    assert controller.status()["counters"]["preemptions"] == 1
    await controller.release(foreground.lease_id)


@pytest.mark.asyncio
async def test_cancelled_waiter_is_removed():
    controller = ResourceAdmissionController(
        pressure_provider=_normal_pressure,
        poll_interval_s=0.01,
    )
    holder = await controller.acquire(
        AdmissionRequest(owner="holder", work_class=WorkClass.INFERENCE, lane="cortex", timeout_s=0)
    )
    waiting = asyncio.create_task(
        controller.acquire(
            AdmissionRequest(
                owner="waiting",
                work_class=WorkClass.INFERENCE,
                lane="cortex",
                timeout_s=5,
            )
        )
    )
    await asyncio.sleep(0.03)
    waiting.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert controller.status()["waiters"] == []
    await controller.release(holder.lease_id)


@pytest.mark.asyncio
async def test_expired_lease_is_reclaimed():
    controller = ResourceAdmissionController(pressure_provider=_normal_pressure)
    expired = await controller.acquire(
        AdmissionRequest(
            owner="short",
            work_class=WorkClass.INFERENCE,
            lane="cortex",
            timeout_s=0,
            lease_ttl_s=0.02,
        )
    )
    await asyncio.sleep(0.03)
    replacement = await controller.acquire(
        AdmissionRequest(
            owner="replacement",
            work_class=WorkClass.INFERENCE,
            lane="cortex",
            timeout_s=0,
        )
    )

    assert replacement.admitted is True
    assert controller.status()["counters"]["expired"] == 1
    with pytest.raises(KeyError):
        await controller.release(expired.lease_id)
    await controller.release(replacement.lease_id)


@pytest.mark.asyncio
async def test_reconciler_starts_dependencies_in_order_and_stops_in_reverse(tmp_path):
    store = ReceiptStore(tmp_path / "receipts")
    admission = ResourceAdmissionController(
        pressure_provider=_normal_pressure,
        receipt_store=store,
    )
    plane = RuntimeControlPlane(admission=admission)
    events: list[str] = []
    states = {"database": False, "api": False}

    async def start_database() -> None:
        events.append("start:database")
        states["database"] = True

    async def stop_database() -> None:
        events.append("stop:database")
        states["database"] = False

    async def start_api() -> None:
        events.append("start:api")
        states["api"] = True

    async def stop_api() -> None:
        events.append("stop:api")
        states["api"] = False

    plane.register_service(
        DesiredServiceSpec(name="database", critical=True),
        start=start_database,
        stop=stop_database,
        probe=lambda: states["database"],
    )
    plane.register_service(
        DesiredServiceSpec(name="api", critical=True, dependencies=("database",)),
        start=start_api,
        stop=stop_api,
        probe=lambda: states["api"],
    )

    started = await plane.reconcile_once()

    assert started["converged"] is True
    assert started["critical_ready"] is True
    assert events == ["start:database", "start:api"]
    assert plane.service_status()["api"]["observed_state"] == "ready"
    assert started["actions"][0]["admission_receipt_id"]

    plane.set_desired_state("api", DesiredServiceState.STOPPED)
    plane.set_desired_state("database", DesiredServiceState.STOPPED)
    stopped = await plane.reconcile_once()

    assert stopped["converged"] is True
    assert events[-2:] == ["stop:api", "stop:database"]


def test_reconciler_rejects_dependency_cycles():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    plane.register_service(
        DesiredServiceSpec(name="one", dependencies=("two",)),
        start=lambda: None,
        stop=lambda: None,
        probe=lambda: True,
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        plane.register_service(
            DesiredServiceSpec(name="two", dependencies=("one",)),
            start=lambda: None,
            stop=lambda: None,
            probe=lambda: True,
        )
    assert "two" not in plane.service_status()


def test_stopped_desired_service_registers_as_converged_stopped():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    plane.register_service(
        DesiredServiceSpec(
            name="dormant",
            desired_state=DesiredServiceState.STOPPED,
        ),
        start=lambda: pytest.fail("stopped service must not start"),
        stop=lambda: pytest.fail("never-started service must not stop"),
        probe=lambda: False,
    )

    assert plane.service_status()["dormant"]["observed_state"] == "stopped"
    assert plane.service_status()["dormant"]["reason"] == "registered_stopped"


@pytest.mark.asyncio
async def test_reconciler_opens_circuit_after_bounded_start_failures():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    attempts = 0

    async def fail_start() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boot failed")

    plane.register_service(
        DesiredServiceSpec(
            name="fragile",
            critical=True,
            restart_limit=2,
            restart_window_s=60,
            backoff_initial_s=0.001,
            backoff_max_s=0.001,
        ),
        start=fail_start,
        stop=lambda: None,
        probe=lambda: False,
    )

    await plane.reconcile_once()
    await asyncio.sleep(0.002)
    await plane.reconcile_once()
    report = await plane.reconcile_once()

    assert attempts == 2
    assert report["services"]["fragile"]["observed_state"] == (
        ObservedServiceState.CIRCUIT_OPEN.value
    )
    assert report["critical_ready"] is False


@pytest.mark.asyncio
async def test_dependency_stays_blocked_until_owner_is_ready():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    plane.register_service(
        DesiredServiceSpec(name="dependent", dependencies=("missing",)),
        start=lambda: pytest.fail("blocked service must not start"),
        stop=lambda: None,
        probe=lambda: True,
    )

    report = await plane.reconcile_once()

    status = report["services"]["dependent"]
    assert status["observed_state"] == "blocked"
    assert status["reason"] == "dependency_blocked:missing"
    assert report["actions"] == []


@pytest.mark.asyncio
async def test_dependency_loss_stops_an_already_running_dependent():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    running = {"owner": True, "dependent": True}
    events: list[str] = []
    plane.register_service(
        DesiredServiceSpec(name="owner"),
        start=lambda: running.__setitem__("owner", True),
        stop=lambda: running.__setitem__("owner", False),
        probe=lambda: running["owner"],
        adopt_running=True,
    )
    plane.register_service(
        DesiredServiceSpec(name="dependent", dependencies=("owner",)),
        start=lambda: running.__setitem__("dependent", True),
        stop=lambda: (events.append("stop:dependent"), running.__setitem__("dependent", False)),
        probe=lambda: running["dependent"],
        adopt_running=True,
    )
    plane.set_desired_state("owner", DesiredServiceState.STOPPED)

    report = await plane.reconcile_once()

    assert running == {"owner": False, "dependent": False}
    assert events == ["stop:dependent"]
    assert report["services"]["dependent"]["observed_state"] == "blocked"
    assert report["services"]["dependent"]["reason"] == "dependency_blocked:owner"


@pytest.mark.asyncio
async def test_failed_stop_prevents_duplicate_restart():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    starts = 0

    async def start() -> None:
        nonlocal starts
        starts += 1

    async def stop() -> None:
        raise RuntimeError("cannot stop old instance")

    plane.register_service(
        DesiredServiceSpec(name="wedged", restart_limit=2),
        start=start,
        stop=stop,
        probe=lambda: False,
        adopt_running=True,
    )

    report = await plane.reconcile_once()

    assert starts == 0
    assert report["services"]["wedged"]["observed_state"] == "failed"
    assert report["services"]["wedged"]["reason"] == "stop_failed"

    second = await plane.reconcile_once()
    assert starts == 0
    assert second["services"]["wedged"]["observed_state"] == "failed"


@pytest.mark.asyncio
async def test_unverified_start_is_stopped_before_retry_is_scheduled():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    running = False
    starts = 0
    stops = 0

    async def start() -> None:
        nonlocal running, starts
        starts += 1
        running = True

    async def stop() -> None:
        nonlocal running, stops
        stops += 1
        running = False

    plane.register_service(
        DesiredServiceSpec(name="unverified", restart_limit=2),
        start=start,
        stop=stop,
        probe=lambda: False,
    )

    report = await plane.reconcile_once()

    assert starts == 1
    assert stops == 1
    assert running is False
    assert report["services"]["unverified"]["observed_state"] == "backing_off"
    assert report["services"]["unverified"]["reason"] == "start_failed"
    assert "without passing liveness probe" in report["services"]["unverified"]["last_error"]


@pytest.mark.asyncio
async def test_failed_start_cleanup_blocks_duplicate_start():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    starts = 0

    async def start() -> None:
        nonlocal starts
        starts += 1
        raise RuntimeError("partial launch")

    async def stop() -> None:
        raise RuntimeError("cannot prove partial instance stopped")

    plane.register_service(
        DesiredServiceSpec(name="partial", restart_limit=3),
        start=start,
        stop=stop,
        probe=lambda: False,
    )

    first = await plane.reconcile_once()
    second = await plane.reconcile_once()

    assert starts == 1
    assert first["services"]["partial"]["observed_state"] == "failed"
    assert second["services"]["partial"]["reason"] == "start_cleanup_failed"
    assert "partial launch" in second["services"]["partial"]["last_error"]
    assert "cannot prove partial instance stopped" in second["services"]["partial"]["last_error"]


@pytest.mark.asyncio
async def test_zero_restart_budget_still_permits_initial_start():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    starts = 0

    async def start() -> None:
        nonlocal starts
        starts += 1

    plane.register_service(
        DesiredServiceSpec(name="one_shot", restart_limit=0),
        start=start,
        stop=lambda: None,
        probe=lambda: True,
    )

    report = await plane.reconcile_once()

    assert starts == 1
    assert report["services"]["one_shot"]["observed_state"] == "ready"


@pytest.mark.asyncio
async def test_repeated_health_failures_consume_restart_budget():
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    healthy = True

    async def start() -> None:
        nonlocal healthy
        healthy = True

    async def stop() -> None:
        nonlocal healthy
        healthy = False

    plane.register_service(
        DesiredServiceSpec(
            name="flapping",
            restart_limit=1,
            restart_window_s=60,
        ),
        start=start,
        stop=stop,
        probe=lambda: healthy,
        adopt_running=True,
    )
    healthy = False

    report = await plane.reconcile_once()

    assert report["services"]["flapping"]["observed_state"] == "circuit_open"
    assert report["services"]["flapping"]["reason"] == "probe_failed"


def test_pressure_snapshot_normalizes_mapping_values():
    snapshot = PressureSnapshot.from_mapping(
        {
            "memory_pct": 88,
            "thermal_level": 2,
            "loop_lag_s": 0.5,
            "red_zones": ["memory"],
        }
    )

    assert snapshot.memory_percent == 88.0
    assert snapshot.thermal_level == 2
    assert snapshot.red_zones == ("memory",)
    assert snapshot.to_dict()["red_zones"] == ["memory"]


def test_service_observation_timestamps_are_real():
    before = time.time()
    plane = RuntimeControlPlane(
        admission=ResourceAdmissionController(pressure_provider=_normal_pressure)
    )
    plane.register_service(
        DesiredServiceSpec(name="clock"),
        start=lambda: None,
        stop=lambda: None,
        probe=lambda: True,
    )

    assert plane.service_status()["clock"]["last_transition_at"] >= before
