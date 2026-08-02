from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.reality_reach.attachments import (
    AttachmentAccess,
    ConnectionState,
    DeviceAttachmentBroker,
    DeviceCandidate,
)
from core.reality_reach.body_projection import PhysicalBodyProjection
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.digital_twin import (
    DigitalTwinConflictError,
    DigitalTwinCorruptionError,
    RealityDigitalTwinGraph,
    TwinDisposition,
)
from core.reality_reach.historian import RealityHistorian
from core.reality_reach.live import ChannelReading, ReadingStatus, RealityReachService
from core.reality_reach.observation_router import (
    RealityObservation,
    RealityObservationRouter,
    _digest,
    _observation_identifier,
)


class _OneUsePrivateVerifier:
    def __init__(self) -> None:
        self.consumed: set[str] = set()

    def verify(
        self,
        capability,
        *,
        expected_domain,
        expected_action_digest,
        consume,
    ):
        capability_id = str((capability or {}).get("capability_id") or "")
        scope = str((capability or {}).get("scope") or "")
        valid = (
            bool(capability_id)
            and capability_id not in self.consumed
            and expected_domain == "environment_action"
            and len(str(expected_action_digest)) == 64
            and all(character in "0123456789abcdef" for character in str(expected_action_digest))
            and consume is True
            and scope == "reality_reach.private_twin"
        )
        if valid:
            self.consumed.add(capability_id)
        return SimpleNamespace(
            ok=valid,
            capability=(SimpleNamespace(scope=scope) if valid else None),
            denial=(None if valid else SimpleNamespace(value="invalid_or_replayed")),
        )


class _MigrationVerifier:
    @staticmethod
    def evidence(intent: dict, *, receipt_id: str = "authority.receipt.manifest") -> dict:
        body = {
            "intent": dict(intent),
            "persistent": bool(intent["persistent"]),
            "capability": {"receipt_id": receipt_id},
        }
        return {**body, "evidence_sha256": _digest(body)}

    def validate_persisted_manifest_migration(
        self,
        evidence,
        *,
        intent,
        persistent,
    ):
        value = dict(evidence or {})
        body = {key: item for key, item in value.items() if key != "evidence_sha256"}
        if (
            value.get("intent") != dict(intent)
            or value.get("persistent") is not bool(persistent)
            or value.get("evidence_sha256") != _digest(body)
        ):
            raise PermissionError("migration evidence is not bound to the transition")
        return value


def _private_capability(capability_id: str) -> dict[str, str]:
    return {
        "capability_id": capability_id,
        "scope": "reality_reach.private_twin",
    }


def _private_snapshot(
    graph: RealityDigitalTwinGraph,
    capability_id: str,
    *,
    include_values: bool = False,
) -> dict:
    return graph.snapshot(
        include_property_values=include_values,
        include_private_topology=True,
        authority_capability=_private_capability(capability_id),
    )


def _declaration(channel_id: str = "test.device.temperature") -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id=channel_id,
        kind=ChannelKind.SENSOR,
        observable="temperature",
        unit="celsius",
        domain=NumericDomain(-40.0, 125.0),
        coupling=CouplingClass.NETWORK,
        reality_layers=(RealityLayer.EFFECTIVE,),
        evidence_level=EvidenceLevel.P2,
        owner="tests",
        resolution=0.1,
        sample_rate_hz=1.0,
        max_latency_s=1.0,
        stale_after_s=10.0,
        reference_id="test.reference.temperature",
        coupling_validated=True,
    )


class _Adapter:
    def __init__(
        self,
        adapter_id: str = "test.device_adapter",
        declaration: ChannelDeclaration | None = None,
    ) -> None:
        self.adapter_id = adapter_id
        self.declaration = declaration or _declaration()
        self.reading = _reading(self.declaration, value=21.5, source_sequence=1)

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (self.declaration,)

    def read(self) -> tuple[ChannelReading, ...]:
        return (self.reading,)

    async def refresh_readback(self) -> ChannelReading:
        return self.reading


def _candidate(
    *,
    identity: str = "a",
    manifest: str = "b",
    persistent: bool = True,
) -> DeviceCandidate:
    now = time.time_ns()
    return DeviceCandidate(
        candidate_id=f"test.candidate.{identity}",
        connector_id="test.connector",
        device_id=f"test.device.{identity}",
        display_name=f"Private Device {identity}",
        transport="test.transport",
        identity_fingerprint="sha256:" + identity * 64,
        manifest_sha256="sha256:" + manifest * 64,
        access=(AttachmentAccess.OBSERVE,),
        discovered_at_ns=now,
        expires_at_ns=now + 60_000_000_000,
        persistent_identity=persistent,
        privacy_sensitive=True,
        metadata={"secret_token": "never-persist-this"},
    )


def _reading(
    declaration: ChannelDeclaration,
    *,
    value: float,
    source_sequence: int,
    captured_at_ns: int | None = None,
) -> ChannelReading:
    return ChannelReading(
        channel_id=declaration.channel_id,
        value=value,
        unit=declaration.unit,
        captured_at_ns=captured_at_ns or time.time_ns(),
        status=ReadingStatus.AVAILABLE,
        source="test.device",
        uncertainty=0.1,
        source_epoch="test.epoch",
        source_sequence=source_sequence,
        source_event_id=f"test.event.{source_sequence}",
        source_quality="good",
    )


def _observation(
    graph: RealityDigitalTwinGraph,
    adapter: _Adapter,
    *,
    value: float,
    sequence: int,
    captured_at_ns: int | None = None,
    binding: dict | None = None,
) -> RealityObservation:
    fence = binding or graph.binding_context(adapter.adapter_id, adapter.declaration)
    reading = _reading(
        adapter.declaration,
        value=value,
        source_sequence=sequence,
        captured_at_ns=captured_at_ns,
    )
    received_monotonic_ns = max(1, time.monotonic_ns() + sequence)
    return RealityObservation(
        observation_id=_observation_identifier(
            adapter_id=adapter.adapter_id,
            channel_id=adapter.declaration.channel_id,
            reading_sha256=reading.sha256,
            received_monotonic_ns=received_monotonic_ns,
        ),
        adapter_id=adapter.adapter_id,
        declaration=adapter.declaration,
        reading=reading,
        salience=0.5,
        received_at_ns=time.time_ns(),
        received_monotonic_ns=received_monotonic_ns,
        subscription_id="reality.default",
        historian_record_id=f"history.record.{sequence}",
        historian_quality="good",
        historian_order_basis="source_sequence",
        twin_id=str(fence["twin_id"]),
        attachment_generation=int(fence["attachment_generation"]),
        attachment_bound_at_ns=int(fence["attachment_bound_at_ns"]),
        topology_revision=int(fence["topology_revision"]),
    )


def _attached_graph(
    tmp_path: Path,
) -> tuple[RealityDigitalTwinGraph, DeviceCandidate, _Adapter]:
    graph = RealityDigitalTwinGraph(
        tmp_path / "twin.sqlite3",
        session_id="test.twin.session",
        private_read_capability_verifier=_OneUsePrivateVerifier(),
    )
    candidate = _candidate()
    adapter = _Adapter()
    assert graph.observe_candidate(candidate).accepted is True
    receipt = graph.attach_adapter(
        candidate,
        adapter,
        body_projection=PhysicalBodyProjection(adapter.adapter_id, ("test_limb",)),
    )
    assert receipt.accepted is True
    return graph, candidate, adapter


def test_entity_component_topology_is_stable_bounded_and_private(tmp_path: Path) -> None:
    graph, candidate, adapter = _attached_graph(tmp_path)

    snapshot = graph.snapshot()
    duplicate = graph.attach_adapter(
        candidate,
        adapter,
        body_projection=PhysicalBodyProjection(adapter.adapter_id, ("test_limb",)),
    )
    private = _private_snapshot(graph, "private.topology")

    assert duplicate.disposition == TwinDisposition.DUPLICATE
    assert len(snapshot["twins"]) == 1
    assert snapshot["nodes"] == []
    assert snapshot["relationships"] == []
    assert "manifest_sha256" not in snapshot["twins"][0]
    assert {node["node_kind"] for node in private["nodes"]} == {
        "entity",
        "adapter",
        "channel",
        "body_limb",
    }
    assert {item["relationship_kind"] for item in private["relationships"]} == {
        "contains",
        "exposes",
        "projects_to",
    }
    serialized = str(snapshot)
    assert candidate.display_name not in serialized
    assert candidate.device_id not in serialized
    assert "never-persist-this" not in serialized
    assert graph.status()["actuation_authority"] is False
    assert graph.status()["lifecycle_drift"] == 0


def test_private_neighborhood_requires_fresh_one_use_authority(tmp_path: Path) -> None:
    graph, _candidate_value, _adapter = _attached_graph(tmp_path)
    private = _private_snapshot(graph, "private.neighborhood.seed")
    entity_node_id = next(
        node["node_id"] for node in private["nodes"] if node["node_kind"] == "entity"
    )

    with pytest.raises(PermissionError, match="one-use signed capability"):
        graph.neighbors(entity_node_id)

    capability = _private_capability("private.neighborhood.read")
    neighborhood = graph.neighbors(
        entity_node_id,
        include_private_topology=True,
        authority_capability=capability,
    )
    assert neighborhood["nodes"]
    assert neighborhood["relationships"]
    with pytest.raises(PermissionError, match="rejected"):
        graph.neighbors(
            entity_node_id,
            include_private_topology=True,
            authority_capability=capability,
        )


def test_manifest_drift_requires_explicit_authorized_migration(tmp_path: Path) -> None:
    migration_verifier = _MigrationVerifier()
    graph = RealityDigitalTwinGraph(
        tmp_path / "twin.sqlite3",
        session_id="test.twin.session",
        private_read_capability_verifier=_OneUsePrivateVerifier(),
        migration_authority_verifier=migration_verifier,
    )
    original = _candidate(manifest="b")
    changed = replace(original, manifest_sha256="sha256:" + "c" * 64)
    adapter = _Adapter()

    assert graph.observe_candidate(original).accepted is True
    conflict = graph.observe_candidate(changed)
    assert conflict.accepted is False
    assert conflict.disposition == TwinDisposition.MANIFEST_CONFLICT
    with pytest.raises((DigitalTwinConflictError, ValueError), match="migration"):
        graph.attach_adapter(changed, adapter)

    migration_intent = graph.manifest_migration_intent(
        changed,
        request_id="reality.connect.manifest",
    )
    assert migration_intent is not None
    migration_evidence = migration_verifier.evidence(migration_intent)
    migrated = graph.attach_adapter(
        changed,
        adapter,
        migration_request_id=migration_intent["request_id"],
        migration_authority_evidence=migration_evidence,
    )
    assert migrated.accepted is True
    private = _private_snapshot(graph, "private.manifest")
    assert private["twins"][0]["generation"] == 2
    assert private["twins"][0]["manifest_sha256"] == changed.manifest_sha256


def test_observation_order_and_attachment_epoch_prevent_state_resurrection(
    tmp_path: Path,
) -> None:
    graph, _candidate_value, adapter = _attached_graph(tmp_path)
    fence = graph.binding_context(adapter.adapter_id, adapter.declaration)
    newer = _observation(graph, adapter, value=23.0, sequence=2, binding=fence)
    older = _observation(
        graph,
        adapter,
        value=19.0,
        sequence=1,
        captured_at_ns=newer.reading.captured_at_ns - 1,
        binding=fence,
    )

    assert graph.observe_observation(newer).disposition == TwinDisposition.ACCEPTED
    assert graph.observe_observation(older).disposition == TwinDisposition.STALE_IGNORED
    private = _private_snapshot(graph, "private.observation.first", include_values=True)
    assert private["properties"][0]["value"] == 23.0
    assert graph.snapshot()["properties"] == []
    with pytest.raises(PermissionError, match="one-use signed capability"):
        graph.snapshot(include_property_values=True)

    graph.detach_adapter(adapter.adapter_id, reason="test detach")
    delayed = _observation(graph, adapter, value=99.0, sequence=3, binding=fence)
    assert graph.observe_observation(delayed).disposition == TwinDisposition.STALE_IGNORED
    assert (
        _private_snapshot(
            graph,
            "private.observation.second",
            include_values=True,
        )["properties"][0]["value"]
        == 23.0
    )
    with pytest.raises(PermissionError, match="rejected"):
        _private_snapshot(
            graph,
            "private.observation.second",
            include_values=True,
        )


def test_adapter_id_reuse_fences_delayed_observation_to_prior_twin(tmp_path: Path) -> None:
    graph, first_candidate, adapter = _attached_graph(tmp_path)
    old_fence = graph.binding_context(adapter.adapter_id, adapter.declaration)
    second = _candidate(identity="c", manifest="d")
    graph.observe_candidate(second)
    graph.attach_adapter(second, _Adapter(adapter.adapter_id))

    delayed = _observation(graph, adapter, value=88.0, sequence=4, binding=old_fence)
    receipt = graph.observe_observation(delayed)

    assert receipt.disposition == TwinDisposition.STALE_IGNORED
    lifecycles = {item["twin_id"]: item["lifecycle"] for item in graph.snapshot()["twins"]}
    assert lifecycles[str(old_fence["twin_id"])] == "lost"

    rebound = graph.attach_adapter(first_candidate, _Adapter(adapter.adapter_id))
    rebound_fence = graph.binding_context(adapter.adapter_id, adapter.declaration)
    assert rebound.disposition == TwinDisposition.ACCEPTED
    assert rebound_fence["twin_id"] == old_fence["twin_id"]
    assert rebound_fence["attachment_bound_at_ns"] != old_fence["attachment_bound_at_ns"]


def test_same_adapter_reattach_advances_epoch_and_remains_idempotent(tmp_path: Path) -> None:
    graph, candidate, adapter = _attached_graph(tmp_path)
    prior_fence = graph.binding_context(adapter.adapter_id, adapter.declaration)

    graph.detach_adapter(adapter.adapter_id, reason="temporary disconnect", lost=True)
    rediscovered_at_ns = candidate.discovered_at_ns + 1
    rediscovered = replace(
        candidate,
        discovered_at_ns=rediscovered_at_ns,
        expires_at_ns=rediscovered_at_ns + 60_000_000_000,
    )
    rediscovery = graph.observe_candidate(rediscovered)
    assert rediscovery.disposition == TwinDisposition.ACCEPTED
    assert graph.snapshot()["twins"][0]["lifecycle"] == "discovered"

    reattached = graph.attach_adapter(rediscovered, adapter)
    current_fence = graph.binding_context(adapter.adapter_id, adapter.declaration)

    assert reattached.disposition == TwinDisposition.ACCEPTED
    assert current_fence["attachment_generation"] == (
        prior_fence["attachment_generation"] + 1
    )
    assert current_fence["attachment_bound_at_ns"] >= prior_fence["attachment_bound_at_ns"]
    duplicate = graph.attach_adapter(rediscovered, adapter)
    assert duplicate.disposition == TwinDisposition.DUPLICATE
    assert graph.binding_context(adapter.adapter_id, adapter.declaration) == current_fence

    delayed = _observation(graph, adapter, value=99.0, sequence=1, binding=prior_fence)
    current = _observation(graph, adapter, value=22.0, sequence=2, binding=current_fence)
    assert graph.observe_observation(delayed).disposition == TwinDisposition.STALE_IGNORED
    assert graph.observe_observation(current).disposition == TwinDisposition.ACCEPTED


def test_discovery_heartbeats_are_prunable_but_lifecycle_events_survive(tmp_path: Path) -> None:
    path = tmp_path / "twin.sqlite3"
    graph = RealityDigitalTwinGraph(
        path,
        session_id="test.twin.discovery.capacity",
        max_events=256,
    )
    candidate = _candidate()
    graph.observe_candidate(candidate)

    for offset in range(1, 320):
        observed_at = candidate.discovered_at_ns + offset
        graph.observe_candidate(
            replace(
                candidate,
                discovered_at_ns=observed_at,
                expires_at_ns=observed_at + 60_000_000_000,
            )
        )

    status = graph.status()
    assert status["counts"]["events"] <= 256
    assert status["counts"]["twins"] == 1
    connection = sqlite3.connect(path)
    try:
        lifecycle_events = connection.execute(
            "SELECT event_kind, COUNT(*) FROM twin_events "
            "WHERE event_kind!='candidate_seen' GROUP BY event_kind"
        ).fetchall()
    finally:
        connection.close()
    assert lifecycle_events == [("candidate_discovered", 1)]


def test_restart_fences_process_bound_adapter_and_preserves_identity(tmp_path: Path) -> None:
    path = tmp_path / "twin.sqlite3"
    graph, candidate, adapter = _attached_graph(tmp_path)
    twin_id = graph.snapshot()["twins"][0]["twin_id"]
    graph.close()

    restored = RealityDigitalTwinGraph(path, session_id="test.twin.new_session")
    snapshot = restored.snapshot(twin_id)
    assert snapshot["twins"][0]["lifecycle"] == "lost"
    assert restored.status()["active_bindings"] == 0
    assert restored.observe_candidate(candidate).twin_id == twin_id
    rebound = _Adapter("test.device_adapter.after_restart")
    restored.attach_adapter(candidate, rebound)
    assert restored.snapshot(twin_id)["twins"][0]["lifecycle"] == "attached"


def test_row_digest_corruption_fails_health_and_queries(tmp_path: Path) -> None:
    graph, _candidate_value, _adapter = _attached_graph(tmp_path)
    graph.close()
    connection = sqlite3.connect(tmp_path / "twin.sqlite3")
    try:
        connection.execute("UPDATE twins SET health='degraded'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(DigitalTwinCorruptionError, match="row digest"):
        RealityDigitalTwinGraph(tmp_path / "twin.sqlite3", session_id="test.twin.reopen")


def test_live_row_corruption_cannot_be_laundered_by_later_upsert(tmp_path: Path) -> None:
    graph, candidate, adapter = _attached_graph(tmp_path)
    connection = sqlite3.connect(graph.db_path)
    try:
        connection.execute(
            "UPDATE twin_nodes SET model_json='{}' WHERE node_kind='adapter'"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(DigitalTwinCorruptionError, match="row digest"):
        graph.attach_adapter(candidate, adapter)
    assert graph.probe_health(full=True) is False


def _roll_lifecycle_archive(path: Path, *, session_id: str) -> None:
    graph = RealityDigitalTwinGraph(
        path,
        session_id=session_id,
        max_events=256,
        private_read_capability_verifier=_OneUsePrivateVerifier(),
    )
    candidate = _candidate()
    adapter = _Adapter()
    graph.observe_candidate(candidate)
    for cycle in range(140):
        graph.attach_adapter(candidate, adapter)
        graph.detach_adapter(adapter.adapter_id, reason=f"bounded rollover {cycle}")
    assert graph.status()["counts"]["event_segments"] >= 1
    assert graph.status()["counts"]["archived_lifecycle_events"] >= 1
    assert graph.probe_health(full=True) is True
    graph.close()


def test_lifecycle_archive_rollover_is_restart_safe_and_tamper_evident(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollover.sqlite3"
    _roll_lifecycle_archive(path, session_id="test.twin.rollover")

    restored = RealityDigitalTwinGraph(
        path,
        session_id="test.twin.rollover.restarted",
        max_events=256,
        private_read_capability_verifier=_OneUsePrivateVerifier(),
    )
    assert restored.probe_health(full=True) is True
    restored.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE twin_event_segments SET events_json='[]' WHERE segment_sequence=1"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(DigitalTwinCorruptionError, match="row digest"):
        RealityDigitalTwinGraph(
            path,
            session_id="test.twin.rollover.tampered",
            max_events=256,
        )


def test_lifecycle_archive_segment_deletion_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "rollover-deletion.sqlite3"
    _roll_lifecycle_archive(path, session_id="test.twin.rollover.deletion")
    connection = sqlite3.connect(path)
    try:
        connection.execute("DELETE FROM twin_event_segments WHERE segment_sequence=1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(DigitalTwinCorruptionError, match="archive head differs"):
        RealityDigitalTwinGraph(
            path,
            session_id="test.twin.rollover.deletion.reopen",
            max_events=256,
        )


def test_lifecycle_event_mutation_and_deletion_are_detected(tmp_path: Path) -> None:
    mutation_path = tmp_path / "mutation.sqlite3"
    mutation_graph = RealityDigitalTwinGraph(
        mutation_path,
        session_id="test.twin.event.mutation",
    )
    mutation_graph.observe_candidate(_candidate())
    mutation_graph.close()
    connection = sqlite3.connect(mutation_path)
    try:
        connection.execute(
            "UPDATE twin_events SET event_kind='candidate_rediscovered' "
            "WHERE lifecycle_sequence=1"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(DigitalTwinCorruptionError, match="row digest"):
        RealityDigitalTwinGraph(
            mutation_path,
            session_id="test.twin.event.mutation.reopen",
        )

    deletion_path = tmp_path / "deletion.sqlite3"
    deletion_graph = RealityDigitalTwinGraph(
        deletion_path,
        session_id="test.twin.event.deletion",
    )
    deletion_graph.observe_candidate(_candidate(identity="c"))
    deletion_graph.close()
    connection = sqlite3.connect(deletion_path)
    try:
        connection.execute("DELETE FROM twin_events WHERE lifecycle_sequence=1")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(DigitalTwinCorruptionError, match="chain head differs"):
        RealityDigitalTwinGraph(
            deletion_path,
            session_id="test.twin.event.deletion.reopen",
        )


def test_future_schema_and_symlink_storage_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "twin.sqlite3"
    graph = RealityDigitalTwinGraph(path, session_id="test.twin.schema")
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    graph.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE twin_meta SET value='999' WHERE key='schema_version'"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(DigitalTwinCorruptionError, match="unsupported.*schema"):
        RealityDigitalTwinGraph(path, session_id="test.twin.future")

    target = tmp_path / "not-a-twin.sqlite3"
    target.touch()
    link = tmp_path / "linked-twin.sqlite3"
    link.symlink_to(target)
    with pytest.raises(DigitalTwinCorruptionError, match="must not be a symlink"):
        RealityDigitalTwinGraph(link, session_id="test.twin.symlink")


def test_storage_permission_hardening_failure_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _deny_chmod(_path: object, _mode: int) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr("core.reality_reach.digital_twin.os.chmod", _deny_chmod)
    with pytest.raises(DigitalTwinCorruptionError, match="restricted to its owner"):
        RealityDigitalTwinGraph(
            tmp_path / "private" / "twin.sqlite3",
            session_id="test.twin.permission.failure",
        )


def test_topology_replacement_tombstones_removed_channel_without_corruption(
    tmp_path: Path,
) -> None:
    graph, candidate, adapter = _attached_graph(tmp_path)
    replacement = _Adapter(
        adapter.adapter_id,
        _declaration("test.device.replacement_temperature"),
    )

    receipt = graph.attach_adapter(candidate, replacement)

    assert receipt.accepted is True
    assert graph.probe_health(full=True) is True
    snapshot = _private_snapshot(graph, "private.replacement")
    active_channels = {
        node["model"]["channel_id"]
        for node in snapshot["nodes"]
        if node["node_kind"] == "channel"
    }
    assert active_channels == {"test.device.replacement_temperature"}
    connection = sqlite3.connect(graph.db_path)
    try:
        retired = connection.execute(
            "SELECT enabled FROM twin_nodes WHERE external_id_sha256=?",
            (_digest(adapter.declaration.channel_id),),
        ).fetchone()
    finally:
        connection.close()
    assert retired == (0,)


def test_forged_twin_binding_is_rejected_before_reconstruction(tmp_path: Path) -> None:
    graph, _candidate_value, adapter = _attached_graph(tmp_path)
    observation = _observation(graph, adapter, value=21.0, sequence=1)
    payload = observation.to_dict()
    twin_binding = payload["twin_binding"]
    twin_binding["topology_revision"] = int(twin_binding["topology_revision"]) + 1
    base_payload = dict(payload)
    base_payload.pop("historian", None)
    historian_evidence = dict(payload["historian"])
    historian_evidence.pop("binding_sha256", None)
    payload["historian"]["binding_sha256"] = _digest(
        {"observation": base_payload, "historian": historian_evidence}
    )

    with pytest.raises(ValueError, match="twin binding digest differs"):
        RealityObservation.from_dict(payload)

    forged = replace(
        observation,
        topology_revision=observation.topology_revision + 1,
    )
    assert graph.observe_observation(forged).disposition == TwinDisposition.STALE_IGNORED


@pytest.mark.asyncio
async def test_router_delivers_digital_twin_as_required_durable_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RealityReachService(session_id="test.router.twin")
    graph = RealityDigitalTwinGraph(
        tmp_path / "twin.sqlite3",
        session_id=service.session_id,
        private_read_capability_verifier=_OneUsePrivateVerifier(),
    )
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    events: list[object] = []
    cognitive: list[object] = []

    def _ingest(event):
        events.append(event)
        return SimpleNamespace(event_id=event.event_id, accepted=True, reason="accepted")

    def _observe(*args, **kwargs):
        cognitive.append((args, kwargs))
        return {"receipt_id": "advanced.receipt.twin"}

    services = {
        "multimodal_synchronizer": SimpleNamespace(ingest=_ingest),
        "advanced_cognition": SimpleNamespace(observe_state=_observe),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )
    router = RealityObservationRouter(
        service,
        historian=historian,
        digital_twin=graph,
        poll_interval_s=60.0,
        max_delivery_rate_hz=100.0,
    )
    declaration = _declaration("test.router.temperature")
    await router.start()
    try:
        receipt = await router.submit(
            declaration,
            _reading(declaration, value=25.0, source_sequence=1),
            adapter_id="test.router.adapter",
        )
        assert receipt.accepted is True
        for _ in range(100):
            if router.status()["delivered"] == 1:
                break
            await asyncio.sleep(0.01)
    finally:
        await router.stop()

    assert router.status()["delivered"] == 1
    assert (
        _private_snapshot(graph, "private.router", include_values=True)["properties"][0][
            "value"
        ]
        == 25.0
    )
    historian_status = historian.status()
    assert historian_status["delivery_counts"] == {"delivered": 1}
    assert historian_status["delivery_sink_status"]["digital_twin"]["delivered"] == 1
    assert historian_status["delivery_sink_status"]["digital_twin"]["pending"] == 0
    assert router.status()["required_sinks"] == [
        "digital_twin",
        "multimodal",
        "advanced_cognition",
    ]


class _VanishingConnector:
    connector_id = "test.connector"

    def __init__(self, candidate: DeviceCandidate) -> None:
        self.candidate = candidate
        self.visible = True
        self.failing = False
        self.detach_count = 0

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        if self.failing:
            raise TimeoutError("temporary connector outage")
        if not self.visible:
            return ()
        now = time.time_ns()
        return (
            replace(
                self.candidate,
                discovered_at_ns=now,
                expires_at_ns=now + 60_000_000_000,
            ),
        )

    async def attach(self, candidate, access):
        del candidate, access
        return _Adapter()

    async def detach(self, adapter):
        del adapter
        self.detach_count += 1


class _Authority:
    def verify(self, capability, *, intent, persistent):
        del persistent
        return {
            "capability": {"receipt_id": capability["receipt_id"]},
            "intent": dict(intent),
            "verified_at_ns": time.time_ns(),
        }

    def validate_persisted(self, evidence, *, intent, persistent):
        del intent, persistent
        return dict(evidence)


@pytest.mark.asyncio
async def test_disappeared_candidate_is_detached_and_transitions_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.reality_reach.attachments as attachment_module

    monkeypatch.setattr(
        attachment_module,
        "project_adapter_to_body",
        lambda adapter, **_kwargs: PhysicalBodyProjection(adapter.adapter_id, ()),
    )
    monkeypatch.setattr(attachment_module, "remove_body_projection", lambda _item: None)
    service = RealityReachService(session_id="test.broker.twin")
    graph = RealityDigitalTwinGraph(tmp_path / "twin.sqlite3", session_id=service.session_id)
    router = RealityObservationRouter(service, digital_twin=graph)
    broker = DeviceAttachmentBroker(
        service,
        router,
        digital_twin=graph,
        authority_verifier=_Authority(),
        trust_store_error="test_session_only",
        disappearance_quorum=2,
    )
    connector = _VanishingConnector(_candidate(persistent=False))
    broker.register_connector(connector)
    await broker.discover()
    request = broker.requests()[0]
    attached = await broker.authorize_and_attach(
        request.request_id,
        authority_capability={"receipt_id": "authority.receipt.session"},
        persistent=False,
    )
    assert attached.state == ConnectionState.ATTACHED

    connector.visible = False
    await broker.discover()

    assert connector.detach_count == 0
    assert broker.requests()[0].state == ConnectionState.ATTACHED
    await broker.discover()

    assert connector.detach_count == 1
    assert broker.requests()[0].state == ConnectionState.LOST
    assert service.status()["adapter_count"] == 0
    assert graph.snapshot()["twins"][0]["lifecycle"] == "lost"


@pytest.mark.asyncio
async def test_transient_discovery_failure_does_not_claim_attached_device_is_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.reality_reach.attachments as attachment_module

    monkeypatch.setattr(
        attachment_module,
        "project_adapter_to_body",
        lambda adapter, **_kwargs: PhysicalBodyProjection(adapter.adapter_id, ()),
    )
    monkeypatch.setattr(attachment_module, "remove_body_projection", lambda _item: None)
    service = RealityReachService(session_id="test.broker.transient")
    graph = RealityDigitalTwinGraph(tmp_path / "twin.sqlite3", session_id=service.session_id)
    router = RealityObservationRouter(service, digital_twin=graph)
    broker = DeviceAttachmentBroker(
        service,
        router,
        digital_twin=graph,
        authority_verifier=_Authority(),
        trust_store_error="test_session_only",
    )
    connector = _VanishingConnector(_candidate(persistent=False))
    broker.register_connector(connector)
    await broker.discover()
    request = broker.requests()[0]
    attached = await broker.authorize_and_attach(
        request.request_id,
        authority_capability={"receipt_id": "authority.receipt.session"},
        persistent=False,
    )
    assert attached.state == ConnectionState.ATTACHED

    connector.failing = True
    discovered = await broker.discover()

    assert discovered
    assert connector.detach_count == 0
    assert broker.requests()[0].state == ConnectionState.ATTACHED
    assert service.status()["adapter_count"] == 1
    assert graph.snapshot()["twins"][0]["lifecycle"] == "attached"
    assert broker.status()["connector_failure_streaks"] == {"test.connector": 1}
