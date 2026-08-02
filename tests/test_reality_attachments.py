from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from core.reality_reach import attachments as attachment_module
from core.reality_reach.attachment_authority import AttachmentAuthorityError
from core.reality_reach.attachments import (
    AttachmentAccess,
    ConnectionState,
    DeviceAttachmentBroker,
    DeviceCandidate,
)
from core.reality_reach.body_projection import (
    PhysicalBodyProjection,
    project_adapter_to_body,
    remove_body_projection,
)
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.live import ChannelReading, ReadingStatus, RealityReachService
from core.reality_reach.observation_router import RealityObservationRouter
from core.reality_reach.trust_custody import KeychainAttachmentTrustStore
from core.somatic.body_schema import LimbType


def _declaration(
    channel_id: str = "test.device.temperature",
    *,
    kind: ChannelKind = ChannelKind.SENSOR,
) -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id=channel_id,
        kind=kind,
        observable="temperature",
        unit="celsius",
        domain=NumericDomain(-40.0, 125.0),
        coupling=CouplingClass.NETWORK,
        reality_layers=(RealityLayer.EFFECTIVE,),
        evidence_level=EvidenceLevel.P2,
        owner="tests",
        resolution=0.1,
        sample_rate_hz=1.0 if kind == ChannelKind.SENSOR else 0.0,
        max_latency_s=1.0,
        stale_after_s=10.0,
        reference_id="test.reference.temperature",
        coupling_validated=True,
    )


class Adapter:
    adapter_id = "test.device_adapter"

    def __init__(self) -> None:
        self.declaration = _declaration()
        self.reading = ChannelReading(
            channel_id=self.declaration.channel_id,
            value=21.5,
            unit=self.declaration.unit,
            captured_at_ns=time.time_ns(),
            status=ReadingStatus.AVAILABLE,
            source="test.device",
            uncertainty=0.1,
        )

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (self.declaration,)

    def read(self) -> tuple[ChannelReading, ...]:
        return (self.reading,)

    async def refresh_readback(self) -> ChannelReading:
        return self.reading


def _candidate(*, persistent: bool = True, manifest: str = "b") -> DeviceCandidate:
    now = time.time_ns()
    return DeviceCandidate(
        candidate_id="test.candidate.device",
        connector_id="test.connector",
        device_id="test.device",
        display_name="Test Device",
        transport="test.transport",
        identity_fingerprint="sha256:" + "a" * 64,
        manifest_sha256="sha256:" + manifest * 64,
        access=(AttachmentAccess.OBSERVE,),
        discovered_at_ns=now,
        expires_at_ns=now + 60_000_000_000,
        persistent_identity=persistent,
        metadata={"class": "temperature"},
    )


class Connector:
    connector_id = "test.connector"

    def __init__(self, candidate: DeviceCandidate, *, fail_attach: bool = False) -> None:
        self.candidate = candidate
        self.fail_attach = fail_attach
        self.attach_count = 0
        self.detach_count = 0

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        return (replace(self.candidate, discovered_at_ns=time.time_ns(), expires_at_ns=time.time_ns() + 60_000_000_000),)

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> Adapter:
        assert candidate.identity_fingerprint == self.candidate.identity_fingerprint
        assert access == (AttachmentAccess.OBSERVE,)
        self.attach_count += 1
        if self.fail_attach:
            raise RuntimeError("candidate manifest changed")
        return Adapter()

    async def detach(self, adapter: Adapter) -> None:
        assert adapter.adapter_id == Adapter.adapter_id
        self.detach_count += 1


class FakeKeychain:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.accept_writes = True

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, password: str) -> bool:
        if self.accept_writes:
            self.values[(service, account)] = password
        return self.accept_writes


class FakeAuthorityVerifier:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def verify(self, capability, *, intent, persistent):
        del persistent
        receipt_id = str(capability.get("receipt_id") or "")
        if not receipt_id:
            raise AttachmentAuthorityError("attachment_authority_receipt_missing")
        if receipt_id in self.seen:
            raise AttachmentAuthorityError("attachment_authority_capability_replayed")
        self.seen.add(receipt_id)
        return {
            "capability": {"receipt_id": receipt_id},
            "intent": dict(intent),
            "verified_at_ns": time.time_ns(),
        }

    def validate_persisted(self, evidence, *, intent, persistent):
        del intent, persistent
        return dict(evidence)


@pytest.fixture
def no_body_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        attachment_module,
        "project_adapter_to_body",
        lambda adapter, **_kwargs: PhysicalBodyProjection(adapter.adapter_id, ()),
    )
    monkeypatch.setattr(attachment_module, "remove_body_projection", lambda _item: None)


def _broker(
    state_path: Path,
    *,
    backend: FakeKeychain | None = None,
    authority: FakeAuthorityVerifier | None = None,
    clock_ns=time.time_ns,
) -> tuple[DeviceAttachmentBroker, RealityReachService]:
    service = RealityReachService(session_id="test.attachments")
    router = RealityObservationRouter(service)
    keychain = backend or FakeKeychain()
    return (
        DeviceAttachmentBroker(
            service,
            router,
            state_path=state_path,
            trust_store=KeychainAttachmentTrustStore(keychain, state_path),
            authority_verifier=authority or FakeAuthorityVerifier(),
            clock_ns=clock_ns,
        ),
        service,
    )


def _authority(receipt_id: str) -> dict[str, str]:
    return {"receipt_id": receipt_id}


@pytest.mark.asyncio
async def test_discovery_only_proposes_and_never_invents_attachment(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    broker, service = _broker(tmp_path / "trust.json")
    connector = Connector(_candidate())
    broker.register_connector(connector)

    candidates = await broker.discover()

    assert len(candidates) == 1
    assert connector.attach_count == 0
    assert service.status()["adapter_count"] == 0
    assert broker.requests()[0].state == ConnectionState.PENDING_TRUST


@pytest.mark.asyncio
async def test_persistent_trust_reattaches_after_runtime_migration(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    state_path = tmp_path / "trust.json"
    backend = FakeKeychain()
    first, first_service = _broker(state_path, backend=backend)
    first_connector = Connector(_candidate())
    first.register_connector(first_connector)
    await first.discover()
    attached = await first.authorize_and_attach(
        first.requests()[0].request_id,
        authority_capability=_authority("authority.test.physical.1"),
        persistent=True,
    )
    assert attached.state == ConnectionState.ATTACHED
    assert first_service.status()["adapter_count"] == 1
    await first.stop()

    second, second_service = _broker(state_path, backend=backend)
    second_connector = Connector(_candidate())
    second.register_connector(second_connector)
    await second.discover()

    assert second_connector.attach_count == 1
    assert second_service.status()["adapter_count"] == 1
    assert any(item.state == ConnectionState.ATTACHED for item in second.requests())
    await second.stop()


@pytest.mark.asyncio
async def test_unstable_identity_cannot_receive_persistent_trust(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    broker, _service = _broker(tmp_path / "trust.json")
    broker.register_connector(Connector(_candidate(persistent=False)))
    await broker.discover()

    with pytest.raises(PermissionError, match="not stable"):
        await broker.authorize_and_attach(
            broker.requests()[0].request_id,
            authority_capability=_authority("authority.test.physical.2"),
            persistent=True,
        )


@pytest.mark.asyncio
async def test_attachment_failure_is_explicit_and_revocation_removes_adapter(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    failed, _ = _broker(tmp_path / "failed.json")
    failed.register_connector(Connector(_candidate(), fail_attach=True))
    await failed.discover()
    result = await failed.authorize_and_attach(
        failed.requests()[0].request_id,
        authority_capability=_authority("authority.test.physical.3"),
        persistent=True,
    )
    assert result.state == ConnectionState.ERROR
    assert "candidate manifest changed" in result.error

    broker, service = _broker(tmp_path / "revoke.json")
    connector = Connector(_candidate())
    broker.register_connector(connector)
    await broker.discover()
    attached = await broker.authorize_and_attach(
        broker.requests()[0].request_id,
        authority_capability=_authority("authority.test.physical.4"),
        persistent=True,
    )
    await broker.revoke(_candidate().identity_fingerprint, reason="test revoke")

    assert attached.state == ConnectionState.ATTACHED
    assert connector.detach_count == 1
    assert service.status()["adapter_count"] == 0
    assert broker.requests()[0].state == ConnectionState.REVOKED


@pytest.mark.asyncio
async def test_revocation_does_not_claim_success_before_durable_grant_removal(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    backend = FakeKeychain()
    broker, service = _broker(tmp_path / "revoke-failure.json", backend=backend)
    connector = Connector(_candidate())
    broker.register_connector(connector)
    await broker.discover()
    await broker.authorize_and_attach(
        broker.requests()[0].request_id,
        authority_capability=_authority("authority.test.revoke.failure"),
        persistent=True,
    )
    backend.accept_writes = False

    with pytest.raises(RuntimeError, match="anchor_write_unconfirmed"):
        await broker.revoke(_candidate().identity_fingerprint, reason="test revoke")

    assert broker.status()["trust_grants"] == 1
    assert service.status()["adapter_count"] == 1
    assert connector.detach_count == 0
    assert broker.requests()[0].state == ConnectionState.ATTACHED


@pytest.mark.asyncio
async def test_expired_persistent_grant_cannot_reattach(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    state_path = tmp_path / "expiring-trust.json"
    backend = FakeKeychain()
    now_ns = [time.time_ns()]

    def clock() -> int:
        return now_ns[0]

    first, _ = _broker(state_path, backend=backend, clock_ns=clock)
    first.register_connector(Connector(_candidate()))
    await first.discover()
    await first.authorize_and_attach(
        first.requests()[0].request_id,
        authority_capability=_authority("authority.test.expiring"),
        persistent=True,
        grant_ttl_s=1,
    )
    await first.stop()

    now_ns[0] += 2_000_000_000
    second, second_service = _broker(state_path, backend=backend, clock_ns=clock)
    connector = Connector(_candidate())
    second.register_connector(connector)
    await second.discover()

    assert connector.attach_count == 0
    assert second_service.status()["adapter_count"] == 0
    assert second.status()["trust_grants"] == 0
    assert second.status()["expired_grants"] == 1
    assert second.requests()[0].state == ConnectionState.PENDING_TRUST


@pytest.mark.asyncio
async def test_grant_lifetime_cannot_exceed_policy_ceiling(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    broker, _ = _broker(tmp_path / "bounded-trust.json")
    broker.register_connector(Connector(_candidate()))
    await broker.discover()

    with pytest.raises(ValueError, match="grant_ttl_s"):
        broker.authority_intent(
            broker.requests()[0].request_id,
            persistent=True,
            grant_ttl_s=90 * 24 * 60 * 60 + 1,
        )


@pytest.mark.asyncio
async def test_keychain_outage_keeps_bounded_session_attachment_available(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del tmp_path, no_body_projection
    service = RealityReachService(session_id="test.session-only")
    broker = DeviceAttachmentBroker(
        service,
        RealityObservationRouter(service),
        trust_store=None,
        trust_store_error="KeychainUnavailableError:locked",
        authority_verifier=FakeAuthorityVerifier(),
    )
    broker.register_connector(Connector(_candidate(persistent=False)))
    await broker.discover()

    attached = await broker.authorize_and_attach(
        broker.requests()[0].request_id,
        authority_capability=_authority("authority.test.session"),
        persistent=False,
        grant_ttl_s=60,
    )

    assert attached.state == ConnectionState.ATTACHED
    assert service.status()["adapter_count"] == 1
    assert broker.status()["persistent_trust_ready"] is False
    assert "locked" in broker.status()["trust_custody"]["error"]


@pytest.mark.asyncio
async def test_trust_state_load_never_blocks_the_event_loop() -> None:
    main_thread = threading.get_ident()

    class RecordingTrustStore:
        identity = {"identity_sha256": "sha256:" + "e" * 64}

        def __init__(self) -> None:
            self.load_thread = 0

        def load(self):
            self.load_thread = threading.get_ident()
            return {"grants": []}

        def save(self, body):
            del body
            return {}

        def rotate_and_save(self, body):
            del body
            return {}

        def status(self):
            return {"healthy": True, "error": ""}

    store = RecordingTrustStore()
    service = RealityReachService(session_id="test.off-loop-trust-load")
    broker = DeviceAttachmentBroker(
        service,
        RealityObservationRouter(service),
        trust_store=store,
        authority_verifier=FakeAuthorityVerifier(),
    )

    await broker.start()
    await broker.stop()

    assert store.load_thread != 0
    assert store.load_thread != main_thread


def test_body_projection_exposes_sensor_and_actuator_and_removes_both(monkeypatch) -> None:
    class Body:
        def __init__(self) -> None:
            self.limbs = {}

        def add_limb(self, limb) -> None:
            self.limbs[limb.name] = limb

        def remove_limb(self, name: str) -> None:
            self.limbs.pop(name, None)

    class TwoWayAdapter:
        adapter_id = "test.two_way"

        @staticmethod
        def declarations() -> tuple[ChannelDeclaration, ...]:
            return (
                _declaration("test.two_way.sensor"),
                _declaration("test.two_way.actuator", kind=ChannelKind.ACTUATOR),
            )

    import core.reality_reach.body_projection as projection_module

    body = Body()
    monkeypatch.setattr(projection_module, "get_body_schema", lambda: body)
    projection = project_adapter_to_body(
        TwoWayAdapter(),
        device_id="test.two_way",
        display_name="Two Way Device",
        transport="test",
        persistent_identity=True,
    )

    assert {limb.limb_type for limb in body.limbs.values()} == {
        LimbType.SENSOR,
        LimbType.ACTUATOR,
    }
    assert all(limb.source == "reality:test.two_way" for limb in body.limbs.values())
    remove_body_projection(projection)
    assert body.limbs == {}
