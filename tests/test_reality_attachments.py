from __future__ import annotations

import asyncio
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
from core.reality_reach.digital_twin import RealityDigitalTwinGraph
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


def _named_candidate(
    name: str,
    *,
    digest_character: str,
    salience: float,
) -> DeviceCandidate:
    base = _candidate(persistent=False, manifest=digest_character)
    return replace(
        base,
        candidate_id=f"test.candidate.{name}",
        device_id=f"test.device.{name}",
        display_name=f"Test Device {name.title()}",
        identity_fingerprint="sha256:" + digest_character * 64,
        proposal_salience=salience,
    )


class Connector:
    connector_id = "test.connector"

    def __init__(
        self,
        candidate: DeviceCandidate,
        *,
        fail_attach: bool = False,
        fail_detach: bool = False,
    ) -> None:
        self.candidate = candidate
        self.fail_attach = fail_attach
        self.fail_detach = fail_detach
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
        if self.fail_detach:
            raise RuntimeError("remote teardown unavailable")


class BlockingAttachConnector(Connector):
    def __init__(self, candidate: DeviceCandidate) -> None:
        super().__init__(candidate)
        self.attach_entered = asyncio.Event()
        self.attach_release = asyncio.Event()

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> Adapter:
        assert candidate.identity_fingerprint == self.candidate.identity_fingerprint
        assert access == (AttachmentAccess.OBSERVE,)
        self.attach_count += 1
        self.attach_entered.set()
        await self.attach_release.wait()
        return Adapter()


class MutableDiscoveryConnector(Connector):
    def __init__(self, candidates: tuple[DeviceCandidate, ...]) -> None:
        super().__init__(candidates[0])
        self.visible_candidates = candidates

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        now_ns = time.time_ns()
        return tuple(
            replace(
                candidate,
                discovered_at_ns=now_ns,
                expires_at_ns=now_ns + 60_000_000_000,
            )
            for candidate in self.visible_candidates
        )


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


class FakeMigrationAuthorityVerifier(FakeAuthorityVerifier):
    def __init__(self) -> None:
        super().__init__()
        self.migration_seen: set[str] = set()

    def verify_manifest_migration(self, capability, *, intent, persistent):
        receipt_id = str(capability.get("receipt_id") or "")
        if not receipt_id:
            raise AttachmentAuthorityError(
                "manifest_migration_authority_receipt_missing"
            )
        if receipt_id in self.migration_seen:
            raise AttachmentAuthorityError(
                "manifest_migration_authority_capability_replayed"
            )
        self.migration_seen.add(receipt_id)
        return {
            "capability": {"receipt_id": receipt_id},
            "intent": dict(intent),
            "persistent": bool(persistent),
            "verified_at_ns": time.time_ns(),
            "evidence_sha256": "sha256:" + "d" * 64,
        }

    def validate_persisted_manifest_migration(
        self,
        evidence,
        *,
        intent,
        persistent,
    ):
        value = dict(evidence)
        if value.get("intent") != dict(intent) or value.get("persistent") is not bool(
            persistent
        ) or value.get("evidence_sha256") != "sha256:" + "d" * 64:
            raise AttachmentAuthorityError(
                "manifest_migration_authority_evidence_invalid"
            )
        return value


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
    digital_twin: RealityDigitalTwinGraph | None = None,
    max_candidates: int = 2048,
    disappearance_quorum: int = 3,
) -> tuple[DeviceAttachmentBroker, RealityReachService]:
    service = RealityReachService(session_id="test.attachments")
    router = RealityObservationRouter(service, digital_twin=digital_twin)
    keychain = backend or FakeKeychain()
    return (
        DeviceAttachmentBroker(
            service,
            router,
            digital_twin=digital_twin,
            state_path=state_path,
            trust_store=KeychainAttachmentTrustStore(keychain, state_path),
            authority_verifier=authority or FakeAuthorityVerifier(),
            clock_ns=clock_ns,
            max_candidates=max_candidates,
            disappearance_quorum=disappearance_quorum,
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
async def test_cancelled_attach_completes_shielded_rollback_before_reraising(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    broker, service = _broker(tmp_path / "cancelled-attach.json")
    connector = BlockingAttachConnector(_candidate(persistent=False))
    broker.register_connector(connector)
    await broker.discover()
    request_id = broker.requests()[0].request_id

    attaching = asyncio.create_task(
        broker.authorize_and_attach(
            request_id,
            authority_capability=_authority("authority.test.cancelled"),
            persistent=False,
        )
    )
    await connector.attach_entered.wait()
    attaching.cancel()
    connector.attach_release.set()

    with pytest.raises(asyncio.CancelledError):
        await attaching

    request = next(item for item in broker.requests() if item.request_id == request_id)
    assert request.state == ConnectionState.ERROR
    assert request.error == "attachment_cancelled"
    assert connector.attach_count == 1
    assert connector.detach_count == 1
    assert service.status()["adapter_count"] == 0
    assert broker.status()["attached"] == 0
    assert broker.status()["trust_grants"] == 0


@pytest.mark.asyncio
async def test_manifest_migration_preflight_is_retryable_and_end_to_end_bound(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    graph = RealityDigitalTwinGraph(
        tmp_path / "migration" / "graph.sqlite3",
        session_id="test.attachments.migration",
    )
    authority = FakeMigrationAuthorityVerifier()
    broker, _service = _broker(
        tmp_path / "migration-trust.json",
        authority=authority,
        digital_twin=graph,
    )
    original = _candidate(persistent=False, manifest="b")
    connector = Connector(original)
    broker.register_connector(connector)
    await broker.discover()

    connector.candidate = replace(original, manifest_sha256="sha256:" + "c" * 64)
    await broker.discover()
    request = await broker.request_connection(
        connector.candidate.candidate_id,
        requested_access=(AttachmentAccess.OBSERVE,),
        initiated_by="aura",
        reason="admit an observed manifest replacement",
    )
    migration_intent = broker.manifest_migration_intent(request.request_id)
    assert migration_intent is not None

    with pytest.raises(
        AttachmentAuthorityError,
        match="manifest_migration_authority_capability_missing",
    ):
        await broker.authorize_and_attach(
            request.request_id,
            authority_capability=_authority("authority.test.migration.attach"),
            persistent=False,
        )
    assert authority.seen == set()

    attached = await broker.authorize_and_attach(
        request.request_id,
        authority_capability=_authority("authority.test.migration.attach"),
        manifest_migration_capability=_authority("authority.test.migration.cas"),
        persistent=False,
    )
    assert attached.state == ConnectionState.ATTACHED
    assert (
        graph.manifest_migration_intent(
            connector.candidate,
            request_id=request.request_id,
        )
        is None
    )
    assert authority.seen == {"authority.test.migration.attach"}
    assert authority.migration_seen == {"authority.test.migration.cas"}
    assert graph.snapshot()["twins"][0]["generation"] == 2


@pytest.mark.asyncio
async def test_disappearance_uses_uncapped_scan_and_bounded_quorum(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    candidate_a = _named_candidate("alpha", digest_character="c", salience=1.0)
    candidate_b = _named_candidate("beta", digest_character="d", salience=0.1)
    connector = MutableDiscoveryConnector((candidate_a,))
    broker, service = _broker(
        tmp_path / "disappearance-quorum.json",
        max_candidates=1,
        disappearance_quorum=3,
    )
    broker.register_connector(connector)
    await broker.discover()
    request_id = broker.requests()[0].request_id
    await broker.authorize_and_attach(
        request_id,
        authority_capability=_authority("authority.test.quorum"),
        persistent=False,
    )

    connector.visible_candidates = (
        replace(candidate_b, proposal_salience=1.0),
        replace(candidate_a, proposal_salience=0.0),
    )
    visible = await broker.discover()

    assert [item.candidate_id for item in visible] == [candidate_b.candidate_id]
    assert connector.detach_count == 0
    assert service.status()["adapter_count"] == 1

    connector.visible_candidates = (replace(candidate_b, proposal_salience=1.0),)
    await broker.discover()
    await broker.discover()
    assert connector.detach_count == 0
    assert broker.status()["candidate_absence_streaks"] == {
        candidate_a.candidate_id: 2
    }

    await broker.discover()

    request = next(item for item in broker.requests() if item.request_id == request_id)
    assert connector.detach_count == 1
    assert request.state == ConnectionState.LOST
    assert request.error == "candidate_disappearance_quorum_reached"
    assert service.status()["adapter_count"] == 0


@pytest.mark.asyncio
async def test_twin_detach_failure_never_blocks_local_fencing_and_reconciles(
    tmp_path: Path,
    no_body_projection: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del no_body_projection
    graph = RealityDigitalTwinGraph(
        tmp_path / "twin" / "graph.sqlite3",
        session_id="test.attachments",
    )
    connector = MutableDiscoveryConnector((_candidate(persistent=False),))
    broker, service = _broker(
        tmp_path / "twin-reconcile.json",
        digital_twin=graph,
        disappearance_quorum=2,
    )
    broker.register_connector(connector)
    await broker.discover()
    request_id = broker.requests()[0].request_id
    await broker.authorize_and_attach(
        request_id,
        authority_capability=_authority("authority.test.twin.reconcile"),
        persistent=False,
    )
    original_detach = graph.detach_adapter
    detach_attempts = 0

    def fail_first_detach(adapter_id: str, *, reason: str, lost: bool = False):
        nonlocal detach_attempts
        detach_attempts += 1
        if detach_attempts == 1:
            raise OSError("simulated twin store outage")
        return original_detach(adapter_id, reason=reason, lost=lost)

    monkeypatch.setattr(graph, "detach_adapter", fail_first_detach)
    connector.visible_candidates = ()
    await broker.discover()
    await broker.discover()

    request = next(item for item in broker.requests() if item.request_id == request_id)
    assert request.state == ConnectionState.LOST
    assert connector.detach_count == 1
    assert service.status()["adapter_count"] == 0
    assert broker.status()["twin_reconciliation"] == {
        "healthy": False,
        "pending": 1,
        "pending_detaches": 1,
        "pending_revocations": 0,
    }

    await broker.discover()

    assert detach_attempts == 2
    assert broker.status()["twin_reconciliation"]["healthy"] is True
    assert graph.snapshot()["twins"][0]["lifecycle"] == "lost"


@pytest.mark.asyncio
async def test_active_connector_requires_retirement_and_failed_teardown_keeps_owner(
    tmp_path: Path,
    no_body_projection: None,
) -> None:
    del no_body_projection
    broker, service = _broker(tmp_path / "connector-retirement.json")
    connector = Connector(_candidate(persistent=False), fail_detach=True)
    broker.register_connector(connector)
    await broker.discover()
    request_id = broker.requests()[0].request_id
    await broker.authorize_and_attach(
        request_id,
        authority_capability=_authority("authority.test.retirement"),
        persistent=False,
    )

    with pytest.raises(RuntimeError, match="retire_connector"):
        broker.unregister_connector(connector.connector_id)
    with pytest.raises(RuntimeError, match="could not fence"):
        await broker.retire_connector(connector.connector_id)

    assert broker.status()["connectors"] == 1
    assert broker.status()["attached"] == 1
    assert broker.status()["teardown_pending"] == 1
    assert service.status()["adapter_count"] == 0

    connector.fail_detach = False
    await broker.retire_connector(connector.connector_id)

    request = next(item for item in broker.requests() if item.request_id == request_id)
    assert broker.status()["connectors"] == 0
    assert broker.status()["attached"] == 0
    assert broker.status()["teardown_pending"] == 0
    assert connector.detach_count == 2
    assert request.state == ConnectionState.LOST
    assert service.status()["adapter_count"] == 0


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
