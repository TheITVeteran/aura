from __future__ import annotations

import asyncio
import json
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from core.reality_reach import (
    ActuationCommand,
    ActuationLease,
    ActuationReceipt,
    ActuationState,
    ActuatorCapability,
    ChannelDeclaration,
    ChannelKind,
    ChannelReading,
    CouplingClass,
    EffectReceipt,
    EvidenceLevel,
    NumericDomain,
    PreparedActuation,
    ReadingStatus,
    RealityLayer,
    RealityReachService,
    Reversibility,
    RollbackReceipt,
    transaction_store,
    transactions,
)
from core.reality_reach.transactions import (
    RealityActuationCoordinator,
    RealityActuationError,
)

NOW_NS = 2_000_000_000_000
MONOTONIC_NS = 10_000_000_000
DIGEST = "sha256:" + "a" * 64


def test_transaction_facade_reexports_canonical_store_contracts() -> None:
    assert transactions.RealityActuationError is transaction_store.RealityActuationError
    assert transactions.TRANSACTION_SCHEMA == transaction_store.TRANSACTION_SCHEMA
    assert transactions.COMMAND_CAPSULE_SCHEMA == transaction_store.COMMAND_CAPSULE_SCHEMA
    assert transactions.RECOVERY_REPORT_SCHEMA == transaction_store.RECOVERY_REPORT_SCHEMA


def _sensor() -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id="test.sensor",
        kind=ChannelKind.SENSOR,
        observable="temperature",
        unit="celsius",
        domain=NumericDomain(0.0, 100.0),
        coupling=CouplingClass.THERMAL,
        reality_layers=(RealityLayer.EFFECTIVE, RealityLayer.DIRECT),
        evidence_level=EvidenceLevel.P2,
        owner="tests",
        resolution=0.1,
        sample_rate_hz=10.0,
        max_latency_s=0.1,
        stale_after_s=5.0,
        reference_id="test.independent.sensor",
        coupling_validated=True,
    )


def _actuator() -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id="test.actuator",
        kind=ChannelKind.ACTUATOR,
        observable="temperature",
        unit="celsius",
        domain=NumericDomain(0.0, 10.0),
        coupling=CouplingClass.THERMAL,
        reality_layers=(RealityLayer.EFFECTIVE, RealityLayer.DIRECT),
        evidence_level=EvidenceLevel.P1,
        owner="tests",
        stale_after_s=5.0,
        coupling_validated=True,
    )


class TransactionAdapter:
    adapter_id = "test.transaction"

    def __init__(
        self,
        *,
        verify: bool = True,
        fail_actuation: bool = False,
        fail_safe_state: bool = False,
    ) -> None:
        self.verify = verify
        self.fail_actuation = fail_actuation
        self.fail_safe_state = fail_safe_state
        self.actuation_calls = 0
        self.rollback_calls = 0
        self.safe_state_calls = 0

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (_actuator(), _sensor())

    def actuator_capabilities(self) -> tuple[ActuatorCapability, ...]:
        return (
            ActuatorCapability(
                adapter_id=self.adapter_id,
                channel_id="test.actuator",
                reversibility=Reversibility.REVERSIBLE,
                magnitude_domain=NumericDomain(0.0, 10.0),
                max_commands_per_minute=10,
                observation_channels=("test.sensor",),
                required_permissions=("hardware.thermal",),
                failure_modes=("thermal_limit",),
                watchdog_timeout_s=1.0,
            ),
        )

    def read(self) -> tuple[ChannelReading, ...]:
        return (
            ChannelReading(
                channel_id="test.sensor",
                value=25.0,
                unit="celsius",
                captured_at_ns=NOW_NS,
                status=ReadingStatus.AVAILABLE,
                source="test.instrument",
            ),
            ChannelReading(
                channel_id="test.actuator",
                value=None,
                unit="celsius",
                captured_at_ns=NOW_NS,
                status=ReadingStatus.UNAVAILABLE,
                source="test.actuator",
            ),
        )

    async def prepare(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
    ) -> PreparedActuation:
        return PreparedActuation(
            preparation_id="test.preparation.1",
            command_sha256=command.sha256,
            lease_sha256=lease.sha256,
            adapter_id=self.adapter_id,
            capability_sha256=self.actuator_capabilities()[0].sha256,
            precondition_sha256=DIGEST,
            rollback_token_sha256=DIGEST,
            prepared_at_ns=NOW_NS,
        )

    async def actuate(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
        prepared: PreparedActuation,
    ) -> ActuationReceipt:
        self.actuation_calls += 1
        if self.fail_actuation:
            raise RuntimeError("transport_failed")
        return ActuationReceipt(
            receipt_id="test.actuation.1",
            command_sha256=command.sha256,
            preparation_sha256=prepared.sha256,
            adapter_id=self.adapter_id,
            state=ActuationState.EXECUTED,
            accepted=True,
            transport_completed=True,
            executed=True,
            recorded_at_ns=NOW_NS,
            detail_sha256=DIGEST,
        )

    async def verify_effect(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> EffectReceipt:
        return EffectReceipt(
            receipt_id="test.effect.1",
            command_sha256=command.sha256,
            actuation_receipt_sha256=actuation.sha256,
            observation_channel_id="test.sensor",
            observation_sha256=DIGEST,
            state=(ActuationState.EFFECT_VERIFIED if self.verify else ActuationState.FAILED),
            target_error=0.1 if self.verify else 2.0,
            independently_observed=self.verify,
            recorded_at_ns=NOW_NS,
        )

    async def cancel(
        self,
        command: ActuationCommand,
        prepared: PreparedActuation | None,
    ) -> ActuationReceipt:
        return ActuationReceipt(
            receipt_id="test.cancel.1",
            command_sha256=command.sha256,
            preparation_sha256=prepared.sha256 if prepared else DIGEST,
            adapter_id=self.adapter_id,
            state=ActuationState.CANCELLED,
            accepted=False,
            transport_completed=False,
            executed=False,
            recorded_at_ns=NOW_NS,
            detail_sha256=DIGEST,
        )

    async def safe_state(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
    ) -> RollbackReceipt:
        self.safe_state_calls += 1
        if self.fail_safe_state:
            raise RuntimeError("private-provider-detail")
        return self._rollback_receipt(command, actuation, ActuationState.SAFE_STATE)

    async def rollback(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> RollbackReceipt:
        self.rollback_calls += 1
        return self._rollback_receipt(command, actuation, ActuationState.ROLLED_BACK)

    def _rollback_receipt(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
        state: ActuationState,
    ) -> RollbackReceipt:
        return RollbackReceipt(
            receipt_id=f"test.{state.value}.1",
            command_sha256=command.sha256,
            actuation_receipt_sha256=actuation.sha256 if actuation else DIGEST,
            adapter_id=self.adapter_id,
            state=state,
            safe_state_observation_sha256=DIGEST,
            independently_observed=True,
            recorded_at_ns=NOW_NS,
        )


async def _executor(**kwargs: Any) -> dict[str, Any]:
    handler = kwargs["effect_handler"]
    verifier = kwargs["effect_verifier"]
    try:
        dispatched = dict(await handler({"will_receipt_id": "test.authority.1"}))
    except Exception as exc:  # noqa: BLE001 - emulate ActionExecutor containment
        return {
            "ok": False,
            "error": f"handler_failed:{type(exc).__name__}",
            "transport_succeeded": False,
            "effect_verified": False,
        }
    verified = dict(await verifier({"result": dispatched}))
    return {**dispatched, **verified, "ok": verified.get("effect_verified") is True}


def _service(adapter: TransactionAdapter) -> RealityReachService:
    service = RealityReachService(
        (adapter,),
        clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
        session_id="test-session",
    )
    service.refresh()
    return service


def _command(service: RealityReachService) -> ActuationCommand:
    return ActuationCommand(
        command_id="test.command.1",
        request_id="test.request.1",
        adapter_id="test.transaction",
        channel_id="test.actuator",
        observable="temperature",
        unit="celsius",
        target=26.0,
        tolerance=0.2,
        magnitude=1.0,
        idempotency_key="test.idempotency.1",
        inventory_sha256=str(service.status()["registry_sha256"]),
        deadline_ns=NOW_NS + 10_000_000_000,
        safe_envelope=NumericDomain(0.0, 2.0),
        preconditions=("sensor_ready",),
        expected_effects=("temperature_changed",),
        abort_predicates=("thermal_limit",),
    )


def _capsule_path(root: Path) -> Path:
    return next(root.glob("*.command"))


def test_actuation_command_round_trips_through_strict_document() -> None:
    adapter = TransactionAdapter()
    command = _command(_service(adapter))

    restored = ActuationCommand.from_dict(command.to_dict())

    assert restored == command
    assert restored.sha256 == command.sha256
    with pytest.raises(ValueError, match="schema"):
        ActuationCommand.from_dict({**command.to_dict(), "unexpected": True})


@pytest.mark.asyncio
async def test_verified_effect_is_durable_and_idempotently_replayed(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )

    first = await coordinator.execute(command)
    second = await coordinator.execute(command)

    assert first["effect_verified"] is True
    assert first["reality_reach_transaction"]["state"] == "effect_verified"
    assert second["replayed"] is True
    assert adapter.actuation_calls == 1


@pytest.mark.asyncio
async def test_failed_independent_verification_rolls_back(tmp_path: Path) -> None:
    adapter = TransactionAdapter(verify=False)
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )

    result = await coordinator.execute(command)

    assert result["effect_verified"] is False
    assert result["reality_reach_transaction"]["state"] == "rolled_back"
    assert adapter.rollback_calls == 1


@pytest.mark.asyncio
async def test_transport_failure_enters_observed_safe_state(tmp_path: Path) -> None:
    adapter = TransactionAdapter(fail_actuation=True)
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )

    result = await coordinator.execute(command)

    assert result["effect_verified"] is False
    assert result["reality_reach_transaction"]["state"] == "safe_state"
    assert adapter.safe_state_calls == 1


@pytest.mark.asyncio
async def test_inventory_drift_refuses_before_transaction_or_effect(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    altered = ActuationCommand(
        **{
            **command.to_dict(),
            "inventory_sha256": DIGEST,
            "safe_envelope": command.safe_envelope,
            "parameters": dict(command.parameters),
            "preconditions": command.preconditions,
            "expected_effects": command.expected_effects,
            "abort_predicates": command.abort_predicates,
        }
    )
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )

    with pytest.raises(RealityActuationError, match="inventory_drift"):
        await coordinator.execute(altered)
    assert adapter.actuation_calls == 0


@pytest.mark.asyncio
async def test_dispatched_crash_state_is_never_automatically_replayed(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    coordinator._transition(
        command,
        expected={ActuationState.PLANNED},
        state=ActuationState.DISPATCHED,
        updates={
            "lease_sha256": DIGEST,
            "preparation_sha256": DIGEST,
            "authority_receipt_id": "test.authority.1",
        },
    )

    result = await coordinator.execute(command)

    assert result["replayed"] is True
    assert result["manual_reconciliation_required"] is True
    assert result["retry_safe"] is False
    assert adapter.actuation_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_state", [ActuationState.DISPATCHED, ActuationState.EXECUTED])
async def test_new_coordinator_restores_interrupted_effect_to_observed_safe_state(
    tmp_path: Path,
    crash_state: ActuationState,
) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    first_process = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    first_process._create(command)
    updates = (
        {
            "lease_sha256": DIGEST,
            "preparation_sha256": DIGEST,
            "authority_receipt_id": "test.authority.1",
        }
        if crash_state is ActuationState.DISPATCHED
        else {"actuation_receipt_sha256": DIGEST}
    )
    first_process._transition(
        command,
        expected={ActuationState.PLANNED},
        state=crash_state,
        updates=updates,
    )

    restarted = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS + 1,
        monotonic_clock_ns=lambda: MONOTONIC_NS + 1,
    )
    first = await restarted.recover_after_restart(command)
    second = await restarted.recover_after_restart(command)

    assert first["reality_reach_transaction"]["state"] == "safe_state"
    assert first["manual_reconciliation_required"] is False
    assert first["retry_safe"] is False
    assert second["reality_reach_transaction"]["state"] == "safe_state"
    assert adapter.safe_state_calls == 1
    assert adapter.actuation_calls == 0


@pytest.mark.asyncio
async def test_restart_recovery_cancels_admitted_work_without_dispatch(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    coordinator._transition(
        command,
        expected={ActuationState.PLANNED},
        state=ActuationState.ADMITTED,
        updates={
            "lease_sha256": DIGEST,
            "preparation_sha256": DIGEST,
            "authority_receipt_id": "test.authority.1",
        },
    )

    result = await RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS + 1,
        monotonic_clock_ns=lambda: MONOTONIC_NS + 1,
    ).recover_after_restart(command)

    assert result["reality_reach_transaction"]["state"] == "cancelled"
    assert result["manual_reconciliation_required"] is False
    assert adapter.actuation_calls == 0
    assert adapter.safe_state_calls == 0


@pytest.mark.asyncio
async def test_restart_recovery_failure_is_indeterminate_and_secret_free(
    tmp_path: Path,
) -> None:
    adapter = TransactionAdapter(fail_safe_state=True)
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    coordinator._transition(
        command,
        expected={ActuationState.PLANNED},
        state=ActuationState.DISPATCHED,
        updates={
            "lease_sha256": DIGEST,
            "preparation_sha256": DIGEST,
            "authority_receipt_id": "test.authority.1",
        },
    )

    result = await RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS + 1,
        monotonic_clock_ns=lambda: MONOTONIC_NS + 1,
    ).recover_after_restart(command)
    record = result["reality_reach_transaction"]

    assert record["state"] == "indeterminate"
    assert record["manual_reconciliation_required"] is True
    assert "private-provider-detail" not in record["last_error"]
    assert record["last_error"].startswith("RuntimeError:sha256:")


@pytest.mark.asyncio
async def test_restart_recovery_refuses_inventory_drift_before_safe_state(
    tmp_path: Path,
) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    coordinator._transition(
        command,
        expected={ActuationState.PLANNED},
        state=ActuationState.DISPATCHED,
        updates={
            "lease_sha256": DIGEST,
            "preparation_sha256": DIGEST,
            "authority_receipt_id": "test.authority.1",
        },
    )
    drifted = ActuationCommand(
        **{
            **command.to_dict(),
            "inventory_sha256": DIGEST,
            "safe_envelope": command.safe_envelope,
            "parameters": dict(command.parameters),
            "preconditions": command.preconditions,
            "expected_effects": command.expected_effects,
            "abort_predicates": command.abort_predicates,
        }
    )

    with pytest.raises(RealityActuationError, match="recovery_inventory_drift"):
        await RealityActuationCoordinator(
            service,
            root=tmp_path,
            executor=_executor,
            wall_clock_ns=lambda: NOW_NS + 1,
            monotonic_clock_ns=lambda: MONOTONIC_NS + 1,
        ).recover_after_restart(drifted)
    assert adapter.safe_state_calls == 0


@pytest.mark.asyncio
async def test_boot_sweep_reconstructs_command_and_recovers_without_caller_object(
    tmp_path: Path,
) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    first_process = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    first_process._create(command)
    first_process._transition(
        command,
        expected={ActuationState.PLANNED},
        state=ActuationState.DISPATCHED,
        updates={
            "lease_sha256": DIGEST,
            "preparation_sha256": DIGEST,
            "authority_receipt_id": "test.authority.1",
        },
    )
    del command
    restarted = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS + 1,
        monotonic_clock_ns=lambda: MONOTONIC_NS + 1,
    )

    report = await restarted.recover_all_after_restart()

    assert report["complete"] is True
    assert report["eligible"] == 1
    assert report["processed"] == 1
    assert report["recovered"][0]["before_state"] == "dispatched"
    assert report["recovered"][0]["after_state"] == "safe_state"
    assert adapter.safe_state_calls == 1
    assert adapter.actuation_calls == 0
    capsule = _capsule_path(tmp_path)
    assert stat.S_IMODE(capsule.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_boot_sweep_reports_legacy_transaction_without_command_capsule(
    tmp_path: Path,
) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    _capsule_path(tmp_path).unlink()

    report = await coordinator.recover_all_after_restart()

    assert report["complete"] is False
    assert report["processed"] == 0
    assert len(report["legacy_unrecoverable_transaction_sha256"]) == 1


@pytest.mark.asyncio
async def test_boot_sweep_bounds_work_and_reports_deferred_transactions(
    tmp_path: Path,
) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    base = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    for index in range(3):
        command = replace(
            base,
            command_id=f"test.command.{index + 1}",
            request_id=f"test.request.{index + 1}",
            idempotency_key=f"test.idempotency.{index + 1}",
        )
        coordinator._create(command)

    report = await coordinator.recover_all_after_restart(max_transactions=2)

    assert report["eligible"] == 3
    assert report["processed"] == 2
    assert report["deferred"] == 1
    assert report["complete"] is False


@pytest.mark.asyncio
async def test_recovery_supervisor_retries_and_wakes_when_adapter_reattaches(
    tmp_path: Path,
) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    coordinator._transition(
        command,
        expected={ActuationState.PLANNED},
        state=ActuationState.DISPATCHED,
        updates={
            "lease_sha256": DIGEST,
            "preparation_sha256": DIGEST,
            "authority_receipt_id": "test.authority.supervisor",
        },
    )
    service.unregister_adapter(adapter.adapter_id)

    task = await coordinator.start_recovery_supervisor(
        min_retry_s=0.01,
        max_retry_s=0.05,
    )
    first = await coordinator.wait_for_recovery_attempt(timeout_s=1.0)
    first_generation = coordinator.status()["restart_recovery"]["generation"]

    assert first["complete"] is False
    assert first["retryable"] is True
    assert first["failures"]
    assert task.done() is False

    service.register_adapter(adapter)
    service.refresh()
    coordinator.notify_adapter_available(adapter.adapter_id)
    recovered = first
    generation = first_generation
    for _attempt in range(3):
        recovered = await coordinator.wait_for_recovery_attempt(
            after_generation=generation,
            timeout_s=1.0,
        )
        generation = coordinator.status()["restart_recovery"]["generation"]
        if recovered["complete"]:
            break

    assert recovered["complete"] is True, recovered
    assert recovered["recovered"][0]["after_state"] == "safe_state"
    assert adapter.safe_state_calls == 1
    assert adapter.actuation_calls == 0
    await coordinator.stop()
    assert task.done() is True


@pytest.mark.asyncio
async def test_recovery_supervisor_is_singleflight_and_quiet_when_complete(
    tmp_path: Path,
) -> None:
    adapter = TransactionAdapter()
    coordinator = RealityActuationCoordinator(
        _service(adapter),
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )

    task = await coordinator.start_recovery_supervisor(
        min_retry_s=0.01,
        max_retry_s=0.05,
    )
    same_task = await coordinator.start_recovery_supervisor(
        min_retry_s=0.02,
        max_retry_s=0.04,
    )
    report = await coordinator.wait_for_recovery_attempt(timeout_s=1.0)
    generation = coordinator.status()["restart_recovery"]["generation"]

    assert same_task is task
    assert report["complete"] is True
    await asyncio.sleep(0.04)
    assert coordinator.status()["restart_recovery"]["generation"] == generation

    restarted_task = await coordinator.start_recovery_supervisor()
    woke = await coordinator.wait_for_recovery_attempt(
        after_generation=generation,
        timeout_s=1.0,
    )
    assert restarted_task is task
    assert woke["complete"] is True
    await coordinator.stop()


@pytest.mark.asyncio
async def test_unobserved_safe_state_remains_retryable_until_restored(
    tmp_path: Path,
) -> None:
    adapter = TransactionAdapter(fail_safe_state=True)
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    coordinator._transition(
        command,
        expected={ActuationState.PLANNED},
        state=ActuationState.DISPATCHED,
        updates={
            "lease_sha256": DIGEST,
            "preparation_sha256": DIGEST,
            "authority_receipt_id": "test.authority.unresolved",
        },
    )

    task = await coordinator.start_recovery_supervisor(
        min_retry_s=0.01,
        max_retry_s=0.05,
    )
    unresolved = await coordinator.wait_for_recovery_attempt(timeout_s=1.0)
    generation = coordinator.status()["restart_recovery"]["generation"]

    assert unresolved["complete"] is False
    assert unresolved["retryable"] is True
    assert unresolved["failures"] == []
    assert unresolved["unresolved"] == [
        {"command_sha256": command.sha256, "state": "indeterminate"}
    ]

    adapter.fail_safe_state = False
    coordinator.notify_adapter_available(adapter.adapter_id)
    restored = unresolved
    for _attempt in range(3):
        restored = await coordinator.wait_for_recovery_attempt(
            after_generation=generation,
            timeout_s=1.0,
        )
        generation = coordinator.status()["restart_recovery"]["generation"]
        if restored["complete"]:
            break

    assert restored["complete"] is True
    assert restored["unresolved"] == []
    assert restored["recovered"][0]["after_state"] == "safe_state"
    assert adapter.actuation_calls == 0
    await coordinator.stop()
    assert task.done() is True


def test_command_capsule_collision_and_tampering_fail_closed(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    with pytest.raises(RealityActuationError, match="command_capsule_collision"):
        coordinator._create(replace(command, command_id="test.command.collision"))

    capsule = _capsule_path(tmp_path)
    document = json.loads(capsule.read_bytes())
    document["command"]["target"] = 99.0
    capsule.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    capsule.chmod(0o600)
    with pytest.raises(RealityActuationError, match="command_capsule_digest_invalid"):
        coordinator._discover_recovery_commands(10)


def test_command_capsule_symlink_replacement_fails_closed(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    capsule = _capsule_path(tmp_path)
    outside = tmp_path / "outside"
    capsule.replace(outside)
    capsule.symlink_to(outside)

    with pytest.raises(RealityActuationError, match="scan_custody_invalid"):
        coordinator._discover_recovery_commands(10)


@pytest.mark.asyncio
async def test_independent_readback_can_reconcile_post_execution_crash(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    coordinator._transition(
        command,
        expected={ActuationState.PLANNED},
        state=ActuationState.EXECUTED,
        updates={"actuation_receipt_sha256": DIGEST},
    )
    effect = EffectReceipt(
        receipt_id="test.reconciled.effect.1",
        command_sha256=command.sha256,
        actuation_receipt_sha256=DIGEST,
        observation_channel_id="test.sensor",
        observation_sha256=DIGEST,
        state=ActuationState.EFFECT_VERIFIED,
        target_error=0.1,
        independently_observed=True,
        recorded_at_ns=NOW_NS,
    )

    result = await coordinator.reconcile(
        command,
        effect,
        authority_receipt_id="test.reconciliation.authority.1",
    )

    assert result["effect_verified"] is True
    assert result["reality_reach_transaction"]["state"] == "manually_reconciled"
    assert result["manual_reconciliation_required"] is False


def test_transaction_digest_tampering_fails_closed(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    path = next(tmp_path.glob("*.json"))
    payload = path.read_text(encoding="utf-8").replace(
        '"last_error":""',
        '"last_error":"tampered"',
    )
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(RealityActuationError, match="digest_invalid"):
        coordinator._load(command)


def test_transaction_root_symlink_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    adapter = TransactionAdapter()
    service = _service(adapter)

    with pytest.raises(RealityActuationError, match="root_symlink_refused"):
        RealityActuationCoordinator(service, root=link, executor=_executor)


def test_transaction_record_symlink_is_refused(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    path = next(tmp_path.glob("*.json"))
    target = tmp_path / "target.json"
    path.replace(target)
    path.symlink_to(target)

    with pytest.raises(RealityActuationError, match="transaction_symlink_refused"):
        coordinator._load(command)


@pytest.mark.asyncio
async def test_reconciliation_requires_canonical_authority_receipt(tmp_path: Path) -> None:
    adapter = TransactionAdapter()
    service = _service(adapter)
    command = _command(service)
    coordinator = RealityActuationCoordinator(
        service,
        root=tmp_path,
        executor=_executor,
        wall_clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
    )
    coordinator._create(command)
    coordinator._transition(
        command,
        expected={ActuationState.PLANNED},
        state=ActuationState.EXECUTED,
        updates={"actuation_receipt_sha256": DIGEST},
    )
    effect = EffectReceipt(
        receipt_id="test.reconciled.effect.1",
        command_sha256=command.sha256,
        actuation_receipt_sha256=DIGEST,
        observation_channel_id="test.sensor",
        observation_sha256=DIGEST,
        state=ActuationState.EFFECT_VERIFIED,
        target_error=0.1,
        independently_observed=True,
        recorded_at_ns=NOW_NS,
    )

    with pytest.raises(RealityActuationError, match="authority_invalid"):
        await coordinator.reconcile(
            command,
            effect,
            authority_receipt_id="invalid authority\nreceipt",
        )
