from __future__ import annotations

from dataclasses import replace

import pytest

from core.reality_reach import (
    ChannelDeclaration,
    ChannelKind,
    ChannelRegistry,
    Constraint,
    ConstraintKind,
    CouplingClass,
    EvidenceLevel,
    FailureCode,
    NumericDomain,
    ObjectiveKind,
    ProofRequirement,
    ReachabilityCertificate,
    ReachabilityEngine,
    ReachabilityStatus,
    RealityIR,
    RealityLayer,
)


def _contract(
    *,
    layer: RealityLayer = RealityLayer.DIRECT,
    tolerance: float = 0.5,
    actuators: tuple[str, ...] = ("display.luminance",),
    sensors: tuple[str, ...] = ("meter.luminance",),
    proof: ProofRequirement | None = None,
    constraints: tuple[Constraint, ...] = (),
) -> RealityIR:
    return RealityIR(
        request_id="test.luminance.1",
        objective="Set and independently verify display luminance",
        objective_kind=ObjectiveKind.CONTROL,
        observable="luminance",
        unit="nit",
        target=120.0,
        tolerance=tolerance,
        domain=NumericDomain(0.0, 500.0),
        allowed_actuators=actuators,
        allowed_sensors=sensors,
        constraints=constraints,
        required_proof=proof or ProofRequirement(minimum_evidence=EvidenceLevel.P3),
        reality_layer=layer,
    )


def _actuator(
    *,
    channel_id: str = "display.luminance",
    coupling: CouplingClass = CouplingClass.OPTICAL,
    validated: bool = True,
    reference_id: str = "display.controller",
    tags: tuple[str, ...] = (),
    evidence: EvidenceLevel = EvidenceLevel.P4,
) -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id=channel_id,
        kind=ChannelKind.ACTUATOR,
        observable="luminance",
        unit="nit",
        domain=NumericDomain(0.0, 500.0),
        coupling=coupling,
        reality_layers=(RealityLayer.EFFECTIVE, RealityLayer.DIRECT, RealityLayer.AMBIENT),
        evidence_level=evidence,
        owner="core.display",
        reference_id=reference_id,
        compliance_tags=tags,
        coupling_validated=validated,
    )


def _sensor(
    *,
    channel_id: str = "meter.luminance",
    resolution: float = 0.1,
    reference_id: str = "external.photometer",
    external: bool = True,
    evidence: EvidenceLevel = EvidenceLevel.P4,
) -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id=channel_id,
        kind=ChannelKind.SENSOR,
        observable="luminance",
        unit="nit",
        domain=NumericDomain(0.0, 500.0),
        coupling=CouplingClass.OPTICAL,
        reality_layers=(RealityLayer.EFFECTIVE, RealityLayer.DIRECT, RealityLayer.AMBIENT),
        evidence_level=evidence,
        owner="core.metrology",
        resolution=resolution,
        sample_rate_hz=10.0,
        max_latency_s=0.1,
        reference_id=reference_id,
        calibration_id="nist.photometer.2026",
        external_metrology=external,
        coupling_validated=True,
    )


def _codes(certificate: ReachabilityCertificate) -> set[FailureCode]:
    return {failure.code for failure in certificate.failures}


def test_reality_ir_digest_is_canonical_and_input_sensitive() -> None:
    first = _contract()
    second = _contract()
    changed = replace(first, tolerance=0.25)

    assert first.sha256 == second.sha256
    assert changed.sha256 != first.sha256


def test_missing_channels_produce_explicit_unreachable_certificate() -> None:
    certificate = ReachabilityEngine().analyze(_contract(), ChannelRegistry())

    assert certificate.status == ReachabilityStatus.UNREACHABLE
    assert FailureCode.NO_CHANNEL in _codes(certificate)
    assert certificate.selected_actuators == ()
    assert certificate.selected_sensors == ()
    assert certificate.verify_integrity()


def test_sensor_floor_is_partial_not_a_claim_of_direct_verification() -> None:
    registry = ChannelRegistry((_actuator(), _sensor(resolution=1.0)))
    certificate = ReachabilityEngine().analyze(
        _contract(tolerance=0.5),
        registry,
    )

    assert certificate.status == ReachabilityStatus.PARTIAL
    assert FailureCode.BELOW_SENSOR_FLOOR in _codes(certificate)
    assert certificate.claim_boundary == RealityLayer.DIRECT


def test_shared_reference_cannot_establish_external_effect() -> None:
    registry = ChannelRegistry(
        (
            _actuator(reference_id="display.controller"),
            _sensor(reference_id="display.controller", external=False),
        )
    )
    certificate = ReachabilityEngine().analyze(_contract(), registry)

    assert certificate.status == ReachabilityStatus.PARTIAL
    assert FailureCode.SHARED_REFERENCE in _codes(certificate)


def test_internal_channel_cannot_be_relabelled_as_direct_reality() -> None:
    registry = ChannelRegistry(
        (
            replace(_actuator(), reality_layers=(RealityLayer.INTERNAL,)),
            replace(_sensor(), reality_layers=(RealityLayer.INTERNAL,)),
        )
    )
    certificate = ReachabilityEngine().analyze(_contract(), registry)

    assert certificate.status == ReachabilityStatus.UNREACHABLE
    assert FailureCode.NO_CHANNEL in _codes(certificate)
    assert certificate.selected_actuators == ()
    assert certificate.selected_sensors == ()


def test_ambient_claim_requires_validated_coupling_and_external_metrology() -> None:
    proof = ProofRequirement(
        minimum_evidence=EvidenceLevel.P3,
        external_metrology=True,
    )
    unresolved = ChannelRegistry(
        (
            _actuator(coupling=CouplingClass.UNKNOWN, validated=False),
            _sensor(external=False),
        )
    )
    failed = ReachabilityEngine().analyze(
        _contract(layer=RealityLayer.AMBIENT, proof=proof),
        unresolved,
    )
    assert failed.status == ReachabilityStatus.UNREACHABLE
    assert FailureCode.AMBIENT_IDENTITY_UNRESOLVED in _codes(failed)

    resolved = ChannelRegistry((_actuator(), _sensor(external=True)))
    passed = ReachabilityEngine().analyze(
        _contract(layer=RealityLayer.AMBIENT, proof=proof),
        resolved,
    )
    assert passed.status == ReachabilityStatus.REACHABLE
    assert passed.failures == ()


def test_required_channel_constraint_must_have_declared_compliance() -> None:
    legal = Constraint(
        constraint_id="legal.radio.local-only",
        kind=ConstraintKind.LEGAL,
        description="Emission remains inside the authorized local test envelope",
        applies_to_channels=("display.luminance",),
    )
    missing = ChannelRegistry((_actuator(), _sensor()))
    denied = ReachabilityEngine().analyze(
        _contract(constraints=(legal,)),
        missing,
    )
    assert denied.status == ReachabilityStatus.UNREACHABLE
    assert FailureCode.CONSTRAINT_UNSATISFIED in _codes(denied)

    compliant = ChannelRegistry(
        (
            _actuator(tags=("legal.radio.local-only",)),
            _sensor(),
        )
    )
    accepted = ReachabilityEngine().analyze(
        _contract(constraints=(legal,)),
        compliant,
    )
    assert accepted.status == ReachabilityStatus.REACHABLE


def test_required_evidence_level_is_not_inferred_from_intent() -> None:
    registry = ChannelRegistry(
        (
            _actuator(evidence=EvidenceLevel.P2),
            _sensor(evidence=EvidenceLevel.P2),
        )
    )
    certificate = ReachabilityEngine().analyze(
        _contract(
            proof=ProofRequirement(minimum_evidence=EvidenceLevel.P5),
        ),
        registry,
    )

    assert certificate.status == ReachabilityStatus.PARTIAL
    assert certificate.evidence_ceiling == EvidenceLevel.P2
    assert FailureCode.INSUFFICIENT_EVIDENCE in _codes(certificate)


def test_certificate_rejects_payload_digest_tampering() -> None:
    certificate = ReachabilityEngine().analyze(
        _contract(),
        ChannelRegistry((_actuator(), _sensor())),
    )
    assert certificate.verify_integrity()

    with pytest.raises(ValueError, match="does not match"):
        replace(
            certificate,
            selected_sensors=("meter.tampered",),
            certificate_sha256=certificate.certificate_sha256,
        )


def test_control_contract_requires_actuator_and_observation_channel() -> None:
    with pytest.raises(ValueError, match="require an actuator"):
        _contract(actuators=())
    with pytest.raises(ValueError, match="observation channel"):
        _contract(sensors=())
