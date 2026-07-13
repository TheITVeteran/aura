from __future__ import annotations

import hashlib
import json

import pytest

from core.social import presence_integration
from core.social.person_model import PersonModel
from core.social.relational_memory import RelationalMemoryAuthority
from core.social.relationship_graph import RelationshipGraph


def _authority(tmp_path, *, key: bytes = b"g" * 32) -> RelationalMemoryAuthority:
    return RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=key,
        legacy_paths=(),
        auto_provision_key=False,
    )


def _grant(
    authority: RelationalMemoryAuthority,
    agent_id: str,
    *,
    durable: bool = True,
    include_boundary: bool = True,
) -> None:
    operations = ["recall", "prompt"]
    if durable:
        operations.insert(0, "persist")
    kinds = ["shared_ground"]
    if include_boundary:
        kinds.append("boundary")
    authority.grant_consent(
        agent_id,
        kinds=kinds,
        operations=operations,
        receipt_id=f"relationship-consent-{agent_id}",
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_graph_refuses_unscoped_node_creation_without_exact_agent_consent(tmp_path):
    authority = _authority(tmp_path)
    graph = RelationshipGraph(authority=authority)

    with pytest.raises(PermissionError, match="exact-agent recall consent"):
        graph.get_or_create_node("bryan")
    with pytest.raises(ValueError, match="exact non-empty agent_id"):
        graph.get_or_create_node("")

    assert graph.nodes == {}
    assert authority.status()["record_count"] == 0


def test_topology_persists_encrypted_and_isolated_by_exact_agent(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan")
    _grant(authority, "alice")
    graph = RelationshipGraph(authority=authority)

    observed = graph.record_interaction(
        "bryan",
        digest=_digest("confirmed conversation event"),
    )
    restored_authority = _authority(tmp_path)
    restored = RelationshipGraph(authority=restored_authority)

    assert observed.interaction_count == 1
    assert observed.confidence == pytest.approx(0.25)
    assert restored.get_node("bryan").interaction_count == 1
    assert restored.get_node("alice") is None
    on_disk = (tmp_path / "relational.json").read_text(encoding="utf-8")
    assert "bryan" not in on_disk
    assert "confirmed conversation event" not in on_disk


def test_duplicate_evidence_is_idempotent_and_cannot_inflate_confidence(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan")
    graph = RelationshipGraph(authority=authority)
    evidence = _digest("one canonical event")

    first = graph.record_interaction("bryan", digest=evidence)
    replay = graph.record_interaction("bryan", digest=evidence)

    assert first.interaction_count == 1
    assert replay.interaction_count == 1
    assert replay.confidence == first.confidence


def test_session_topology_is_functional_without_durable_persistence(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan", durable=False)
    graph = RelationshipGraph(authority=authority)

    graph.record_interaction("bryan", digest=_digest("session event"))

    assert graph.get_node("bryan").interaction_count == 1
    assert authority.status()["durable_record_count"] == 0
    assert RelationshipGraph(authority=_authority(tmp_path)).get_node("bryan") is None


def test_unverified_sentiment_is_not_accepted_as_relationship_truth(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan")
    graph = RelationshipGraph(authority=authority)

    with pytest.raises(ValueError, match="cannot infer sentiment"):
        graph.record_interaction(
            "bryan",
            sentiment_delta=0.1,
            digest=_digest("praise text is not a verified outcome"),
        )

    assert graph.get_node("bryan") is None


def test_boundaries_require_separate_consent_evidence_and_authorization(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan", include_boundary=False)
    graph = RelationshipGraph(authority=authority)
    graph.record_interaction("bryan", digest=_digest("conversation"))

    with pytest.raises(PermissionError, match="boundary consent"):
        graph.set_boundary_flag(
            "bryan",
            "do_not_disturb_after_midnight",
            True,
            evidence_digest=_digest("explicit boundary statement"),
            authorization_receipt_id="boundary-receipt-1",
        )

    authority.grant_consent(
        "bryan",
        kinds=["boundary"],
        operations=["persist", "recall", "prompt"],
        receipt_id="boundary-consent-bryan",
    )
    with pytest.raises(ValueError, match="authorization receipt"):
        graph.set_boundary_flag(
            "bryan",
            "do_not_disturb_after_midnight",
            True,
            evidence_digest=_digest("explicit boundary statement"),
            authorization_receipt_id="",
        )
    graph.set_boundary_flag(
        "bryan",
        "do_not_disturb_after_midnight",
        True,
        evidence_digest=_digest("explicit boundary statement"),
        authorization_receipt_id="boundary-receipt-1",
    )

    node = graph.get_node("bryan")
    snapshot = authority.load_snapshot(
        "bryan",
        namespace="relationship_boundaries:v1",
        kind="boundary",
    )
    assert node.boundary_flags == {"do_not_disturb_after_midnight": True}
    assert snapshot is not None
    assert "boundary-receipt-1" not in json.dumps(snapshot)


def test_projects_and_connections_retain_digests_not_raw_identifiers(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan")
    graph = RelationshipGraph(authority=authority)
    graph.record_interaction("bryan", digest=_digest("conversation"))

    graph.link_project(
        "bryan",
        "private-project-aura",
        evidence_digest=_digest("explicit shared project statement"),
    )
    graph.associate(
        "bryan",
        "tatiana",
        evidence_digest=_digest("explicit connection statement"),
    )

    snapshot = authority.load_snapshot(
        "bryan",
        namespace="relationship_graph:v1",
        kind="shared_ground",
    )
    encoded = json.dumps(snapshot)
    assert "private-project-aura" not in encoded
    assert "tatiana" not in encoded
    assert graph.get_connections("bryan") == [_digest("tatiana")]


@pytest.mark.asyncio
async def test_live_registration_records_only_target_topology(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan")
    _grant(authority, "alice")
    graph = RelationshipGraph(authority=authority)

    node = await graph.register_interaction(
        "aura_self",
        "bryan",
        "conversation",
        "self",
        "person",
    )

    assert node.node_id == "bryan"
    assert node.relation_types == ["conversation"]
    assert len(node.evidence_digests) == 1
    assert graph.get_node("alice") is None


def test_targeted_forget_preserves_other_agent_memory_and_consent(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan")
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["persist", "recall", "prompt"],
        receipt_id="profile-consent-bryan",
    )
    authority.upsert_snapshot(
        "bryan",
        namespace="unrelated_profile:v1",
        kind="derived_profile",
        payload={"value": "preserve me"},
        confidence=1.0,
        provenance="test",
    )
    graph = RelationshipGraph(authority=authority)
    graph.record_interaction("bryan", digest=_digest("conversation"))
    graph.set_boundary_flag(
        "bryan",
        "block_social_post",
        True,
        evidence_digest=_digest("explicit boundary"),
        authorization_receipt_id="boundary-receipt",
    )

    assert graph.forget_node(
        "bryan",
        authorization_receipt_id="relationship-delete-receipt",
    )

    assert graph.get_node("bryan") is None
    assert authority.load_snapshot(
        "bryan",
        namespace="unrelated_profile:v1",
        kind="derived_profile",
    ) == {"value": "preserve me"}
    assert authority.allows("bryan", "shared_ground", "recall") is True


def test_unscoped_legacy_node_is_quarantined_not_auto_attributed(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan")
    legacy_dir = tmp_path / "legacy_graph"
    legacy_dir.mkdir()
    legacy_file = legacy_dir / "bryan.json"
    legacy_file.write_text(
        json.dumps(
            {
                "node_id": "bryan",
                "name": "Bryan",
                "sentiment_score": 0.99,
                "digests": ["raw private interaction"],
            }
        ),
        encoding="utf-8",
    )

    graph = RelationshipGraph(legacy_dir, authority=authority)

    assert legacy_file.exists() is False
    assert graph.get_node("bryan") is None
    assert authority.status()["legacy_quarantine_count"] == 1


def test_malformed_snapshot_is_bounded_and_prompt_safe(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan")
    authority.upsert_snapshot(
        "bryan",
        namespace="relationship_graph:v1",
        kind="shared_ground",
        payload={
            "node": {
                "node_id": "wrong-agent",
                "interaction_count": "not-an-int",
                "confidence": "infinity",
                "relation_types": ["CONVERSATION"] * 100,
                "evidence_digests": ["not-a-digest", "a" * 64],
                "shared_project_digests": "not-a-list",
            }
        },
        confidence=0.0,
        provenance="test.malformed",
    )
    graph = RelationshipGraph(authority=authority)

    node = graph.get_node("bryan")

    assert node.node_id == "bryan"
    assert node.interaction_count == 0
    assert node.confidence == 0.0
    assert node.relation_types == ["conversation"]
    assert node.evidence_digests == ["a" * 64]
    assert graph.get_context_block("bryan") == ""


def test_prompt_is_bounded_topology_not_relationship_overclaim(tmp_path):
    authority = _authority(tmp_path)
    _grant(authority, "bryan")
    graph = RelationshipGraph(authority=authority)
    graph.record_interaction("bryan", digest=_digest("conversation"))

    block = graph.get_context_block("bryan")

    assert "CONSENTED RELATIONSHIP TOPOLOGY" in block
    assert "not evidence of trust, intimacy, sentiment" in block
    assert '"interaction_evidence_count":1' in block


def test_person_model_is_fail_closed_view_not_parallel_store(tmp_path):
    authority = _authority(tmp_path)
    graph = RelationshipGraph(authority=authority)
    model = PersonModel("bryan", relationship_graph=graph)

    assert model.get_social_status()["authorized"] is False
    assert model.validate_action("social_post", {}) is False

    _grant(authority, "bryan")
    graph.record_interaction("bryan", digest=_digest("conversation"))
    graph.set_boundary_flag(
        "bryan",
        "block_social_post",
        True,
        evidence_digest=_digest("explicit boundary"),
        authorization_receipt_id="boundary-receipt",
    )

    assert model.get_social_status()["authorized"] is True
    assert model.validate_action("social_post", {}) is False
    assert not hasattr(model, "trust")
    assert not hasattr(model, "preferences")


def test_presence_registration_uses_one_graph_for_canonical_and_legacy_aliases(
    tmp_path,
    monkeypatch,
):
    authority = _authority(tmp_path)
    services = {}
    registrations = []

    def _get(name, default=None):
        return services.get(name, default)

    def _register(name, instance, **metadata):
        services[name] = instance
        registrations.append((name, instance, metadata))
        return True

    monkeypatch.setattr(presence_integration, "get_runtime_service", _get)
    monkeypatch.setattr(presence_integration, "register_runtime_service", _register)

    graph = presence_integration._register_relationship_graph(authority)

    assert services["relationship_graph"] is graph
    assert services["entity_graph"] is graph
    assert {name for name, _, _ in registrations} == {
        "relationship_graph",
        "entity_graph",
    }
    assert all(item[2]["owner"] == "relational_memory" for item in registrations)


def test_presence_registration_rejects_competing_legacy_graph(tmp_path, monkeypatch):
    authority = _authority(tmp_path)
    canonical = RelationshipGraph(authority=authority)
    services = {"relationship_graph": canonical, "entity_graph": object()}
    monkeypatch.setattr(
        presence_integration,
        "get_runtime_service",
        lambda name, default=None: services.get(name, default),
    )

    with pytest.raises(RuntimeError, match="competing relationship owner"):
        presence_integration._register_relationship_graph(authority)


def test_presence_registration_fails_when_runtime_registry_rejects_service(
    tmp_path,
    monkeypatch,
):
    authority = _authority(tmp_path)
    monkeypatch.setattr(
        presence_integration,
        "get_runtime_service",
        lambda _name, default=None: default,
    )
    monkeypatch.setattr(
        presence_integration,
        "register_runtime_service",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="rejected relationship_graph"):
        presence_integration._register_relationship_graph(authority)
