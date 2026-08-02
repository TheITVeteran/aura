from __future__ import annotations

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
)

NOW_NS = 2_000_000_000_000
MONOTONIC_NS = 10_000_000_000
DIGEST = "sha256:" + "a" * 64


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
        sample_rate_hz=1.0,
        max_latency_s=1.0,
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
        max_latency_s=1.0,
        stale_after_s=5.0,
        coupling_validated=True,
    )


def _capability() -> ActuatorCapability:
    return ActuatorCapability(
        adapter_id="test.full",
        channel_id="test.actuator",
        reversibility=Reversibility.REVERSIBLE,
        magnitude_domain=NumericDomain(0.0, 10.0),
        max_commands_per_minute=6,
        observation_channels=("test.sensor",),
        required_permissions=("hardware.thermal",),
        failure_modes=("sensor_dropout", "thermal_limit"),
    )


def _command(parameters: dict[str, float] | None = None) -> ActuationCommand:
    return ActuationCommand(
        command_id="test.command.1",
        request_id="test.request.1",
        adapter_id="test.full",
        channel_id="test.actuator",
        observable="temperature",
        unit="celsius",
        target=5.0,
        tolerance=0.25,
        magnitude=2.0,
        idempotency_key="test.idempotency.1",
        inventory_sha256=DIGEST,
        deadline_ns=NOW_NS + 10_000_000_000,
        safe_envelope=NumericDomain(0.0, 5.0),
        parameters=parameters or {"duration_s": 1.5},
        preconditions=("sensor_ready",),
        expected_effects=("temperature_changed",),
        abort_predicates=("thermal_limit",),
    )


class ReadOnlyActuatorAdapter:
    adapter_id = "test.read_only"

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (_actuator(),)

    def read(self) -> tuple[ChannelReading, ...]:
        return ()


class FullAdapter:
    adapter_id = "test.full"

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (_actuator(), _sensor())

    def actuator_capabilities(self) -> tuple[ActuatorCapability, ...]:
        return (_capability(),)

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
        raise NotImplementedError

    async def actuate(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
        prepared: PreparedActuation,
    ) -> ActuationReceipt:
        raise NotImplementedError

    async def verify_effect(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> EffectReceipt:
        raise NotImplementedError

    async def cancel(
        self,
        command: ActuationCommand,
        prepared: PreparedActuation | None,
    ) -> ActuationReceipt:
        raise NotImplementedError

    async def safe_state(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
    ) -> RollbackReceipt:
        raise NotImplementedError

    async def rollback(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> RollbackReceipt:
        raise NotImplementedError


def test_actuator_declaration_requires_complete_bidirectional_protocol() -> None:
    service = RealityReachService()

    try:
        service.register_adapter(ReadOnlyActuatorAdapter())
    except TypeError as exc:
        assert "complete RealityAdapter protocol" in str(exc)
    else:
        raise AssertionError("read-only actuator adapter was accepted")


def test_actuator_is_executable_only_with_live_observation_route() -> None:
    adapter = FullAdapter()
    service = RealityReachService(
        (adapter,),
        clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: MONOTONIC_NS,
        session_id="test-session",
    )

    assert service.executable_actuator_channels() == ()
    assert service.actuator_adapter("test.actuator") is None

    service.refresh()

    assert service.executable_actuator_channels() == ("test.actuator",)
    assert service.actuator_adapter("test.actuator") is adapter
    assert service.status()["declared_actuator_count"] == 1
    assert service.status()["executable_actuator_count"] == 1


def test_command_parameters_are_frozen_and_content_addressed() -> None:
    parameters = {"duration_s": 1.5}
    command = _command(parameters)
    original_sha256 = command.sha256
    parameters["duration_s"] = 99.0

    assert command.parameters == {"duration_s": 1.5}
    assert command.sha256 == original_sha256

    try:
        command.parameters["duration_s"] = 2.0  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("command parameters were mutable")


def test_lease_requires_matching_session_and_both_clock_deadlines() -> None:
    command = _command()
    lease = ActuationLease(
        lease_id="test.lease.1",
        command_sha256=command.sha256,
        adapter_id="test.full",
        session_id="test-session",
        authority_receipt_id="test.authority.1",
        issued_at_ns=NOW_NS,
        expires_at_ns=NOW_NS + 10_000_000_000,
        issued_monotonic_ns=MONOTONIC_NS,
        expires_monotonic_ns=MONOTONIC_NS + 10_000_000_000,
    )

    assert lease.is_valid(
        now_ns=NOW_NS + 1,
        monotonic_now_ns=MONOTONIC_NS + 1,
        session_id="test-session",
    )
    assert not lease.is_valid(
        now_ns=NOW_NS + 1,
        monotonic_now_ns=MONOTONIC_NS + 1,
        session_id="other-session",
    )
    assert not lease.is_valid(
        now_ns=NOW_NS + 1,
        monotonic_now_ns=MONOTONIC_NS + 10_000_000_000,
        session_id="test-session",
    )


def test_effect_cannot_be_verified_without_independent_observation() -> None:
    try:
        EffectReceipt(
            receipt_id="test.effect.1",
            command_sha256=DIGEST,
            actuation_receipt_sha256=DIGEST,
            observation_channel_id="test.sensor",
            observation_sha256=DIGEST,
            state=ActuationState.EFFECT_VERIFIED,
            target_error=0.1,
            independently_observed=False,
            recorded_at_ns=NOW_NS,
        )
    except ValueError as exc:
        assert "independent observation" in str(exc)
    else:
        raise AssertionError("unobserved effect was marked verified")
