from __future__ import annotations

import time
from typing import Any

import pytest

from core.reality_reach.actuation import (
    ActuationLease,
    ActuationState,
    RealityAdapter,
)
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.live import RealityReachService
from core.reality_reach.scalar_adapter import (
    ScalarAdapterError,
    ScalarRealityAdapter,
    ScalarResourceProfile,
    ScalarSample,
    ScalarWriteResult,
)
from core.runtime.audit_chain import canonical_json, sha256_hex


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


class _Transport:
    transport_id = "test.scalar"

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.writes: list[tuple[str, float, str, bool]] = []
        self.reads = 0

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        self.reads += 1
        return ScalarSample(
            value=self.value,
            captured_at_ns=time.time_ns(),
            source_event_id=_digest(
                {"resource_id": resource_id, "value": self.value, "reads": self.reads}
            ),
        )

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        self.writes.append((resource_id, value, idempotency_key, recovery))
        self.value = value
        return ScalarWriteResult(
            accepted=True,
            transport_completed=True,
            receipt={"resource_id": resource_id, "accepted": True, "recovery": recovery},
        )


def _profile(*, writable: bool = True, distinct: bool = True) -> ScalarResourceProfile:
    return ScalarResourceProfile(
        resource_id="item.desk_light",
        observable="desk_light_level",
        unit="percent",
        domain=NumericDomain(0.0, 100.0),
        resolution=1.0,
        tolerance=1.0,
        writable=writable,
        physical_identity_sha256=_digest("physical.desk_light"),
        owner="tests.scalar_adapter",
        protocol="testscalar",
        safe_value=0.0 if writable else None,
        max_commands_per_minute=4,
        readback_distinct_from_command=distinct,
    )


async def _command_and_lease(
    adapter: ScalarRealityAdapter,
    service: RealityReachService,
    *,
    target: float,
):
    command = await adapter.compile_command(
        target,
        inventory_sha256=service.status()["registry_sha256"],
        deadline_s=10.0,
        idempotency_key="test.scalar.write",
        source="test",
    )
    now_wall = time.time_ns()
    now_mono = time.monotonic_ns()
    lease = ActuationLease(
        lease_id="lease.test.scalar",
        command_sha256=command.sha256,
        adapter_id=adapter.adapter_id,
        session_id=service.session_id,
        authority_receipt_id="authority.test.scalar",
        issued_at_ns=now_wall,
        expires_at_ns=now_wall + 10_000_000_000,
        issued_monotonic_ns=now_mono,
        expires_monotonic_ns=now_mono + 10_000_000_000,
    )
    return command, lease


@pytest.mark.asyncio
async def test_scalar_adapter_executes_verifies_and_rolls_back() -> None:
    transport = _Transport()
    initial = await transport.read_scalar("item.desk_light")
    adapter = ScalarRealityAdapter(transport, _profile(), initial_sample=initial)
    assert isinstance(adapter, RealityAdapter)
    service = RealityReachService((adapter,), session_id="test.scalar.service")

    command, lease = await _command_and_lease(adapter, service, target=65.0)
    prepared = await adapter.prepare(command, lease)
    actuation = await adapter.actuate(command, lease, prepared)
    effect = await adapter.verify_effect(command, actuation)

    assert actuation.state is ActuationState.EXECUTED
    assert effect.state is ActuationState.EFFECT_VERIFIED
    assert effect.target_error == 0.0
    assert transport.value == 65.0
    rollback = await adapter.rollback(command, actuation)
    assert rollback.state is ActuationState.ROLLED_BACK
    assert transport.value == 0.0
    assert transport.writes[-1][3] is True


@pytest.mark.asyncio
async def test_scalar_adapter_refuses_state_drift_before_dispatch() -> None:
    transport = _Transport()
    adapter = ScalarRealityAdapter(
        transport,
        _profile(),
        initial_sample=await transport.read_scalar("item.desk_light"),
    )
    service = RealityReachService((adapter,), session_id="test.scalar.drift")
    command, lease = await _command_and_lease(adapter, service, target=50.0)
    prepared = await adapter.prepare(command, lease)
    transport.value = 12.0

    with pytest.raises(ScalarAdapterError, match="changed_before_dispatch"):
        await adapter.actuate(command, lease, prepared)
    assert transport.writes == []


@pytest.mark.asyncio
async def test_scalar_adapter_does_not_promote_shared_command_readback() -> None:
    transport = _Transport()
    adapter = ScalarRealityAdapter(
        transport,
        _profile(distinct=False),
        initial_sample=await transport.read_scalar("item.desk_light"),
    )
    service = RealityReachService((adapter,), session_id="test.scalar.common.driver")
    command, lease = await _command_and_lease(adapter, service, target=25.0)
    prepared = await adapter.prepare(command, lease)
    actuation = await adapter.actuate(command, lease, prepared)
    effect = await adapter.verify_effect(command, actuation)

    assert effect.state is ActuationState.FAILED
    assert effect.independently_observed is False


def test_read_only_scalar_adapter_registers_without_executable_channel() -> None:
    transport = _Transport(7.0)
    sample = ScalarSample(
        value=7.0,
        captured_at_ns=time.time_ns(),
        source_event_id=_digest("read-only"),
    )
    adapter = ScalarRealityAdapter(transport, _profile(writable=False), initial_sample=sample)
    service = RealityReachService((adapter,), session_id="test.scalar.read.only")

    assert adapter.actuator_capabilities() == ()
    assert service.executable_actuator_channels() == ()
    assert len(service.declarations()) == 1
