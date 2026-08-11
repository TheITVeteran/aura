"""Identity is anchored to a key, not to whichever uuid4 the state has now.

``IdentityAnchor.get_identity`` returned ``identity.name + "-" +
state_id[:8]``, and ``state_id`` is a fresh uuid4 on every derived state. The
object whose entire job is "this is the same entity across restarts and
evolutions" changed several times a second and changed completely on restart.
State lineage was strong and signed to nothing.

Every property below is one the old anchor failed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.identity.entity_key import EntityIdentity, entity_identity


@pytest.fixture
def identity(tmp_path) -> EntityIdentity:
    return entity_identity(tmp_path / "identity")


def test_entity_id_is_derived_from_the_key_not_from_a_state(identity) -> None:
    assert identity.entity_id.startswith("aura:")
    assert identity.entity_id.endswith(identity.report()["entity_id"].split(":")[-1])
    # Signing three states does not move the anchor. Under the old
    # implementation each of these would have produced a different id.
    before = identity.entity_id
    for index in range(3):
        identity.sign_state_link(state_id=f"s{index}", version=index)
    assert identity.entity_id == before


def test_identity_survives_a_restart(tmp_path) -> None:
    root = tmp_path / "identity"
    first = entity_identity(root)
    assert (root / "entity_key.json").exists()
    second = entity_identity(root)
    assert second.entity_id == first.entity_id
    assert second.public_key_hex == first.public_key_hex


def test_key_is_read_back_out_of_the_gateway_envelope(tmp_path, caplog) -> None:
    """The gateway wraps JSON as {schema, schema_version, payload}.

    Reading the file as though it were the payload found no private key, minted
    a fresh one and reported a critical degradation on every boot — an identity
    module breaking identity on the one boundary it exists to survive. Caught
    live, so it is held here.
    """

    root = tmp_path / "identity"
    first = entity_identity(root)
    reloaded = EntityIdentity(root)
    assert reloaded.entity_id == first.entity_id


def test_chain_links_each_state_to_its_parent(identity) -> None:
    first = identity.sign_state_link(state_id="s1", version=1, continuity_hash="h1")
    second = identity.sign_state_link(
        state_id="s2", version=2, parent_state_id="s1", continuity_hash="h2"
    )
    assert second.previous_signature == first.signature
    assert identity.verify_chain()["valid"] is True


def test_a_tampered_link_breaks_verification_at_its_index(identity) -> None:
    """Tamper-evidence, not merely signing.

    The previous signature is inside the payload, so editing a historic link
    invalidates it and everything after it rather than only itself.
    """

    first = identity.sign_state_link(state_id="s1", version=1, continuity_hash="h1")
    second = identity.sign_state_link(
        state_id="s2", version=2, parent_state_id="s1", continuity_hash="h2"
    )
    third = identity.sign_state_link(
        state_id="s3", version=3, parent_state_id="s2", continuity_hash="h3"
    )

    report = identity.verify_chain([first, replace(second, continuity_hash="X"), third])
    assert report["valid"] is False
    assert report["broken_at"] == 1


def test_a_forged_signature_does_not_verify(identity) -> None:
    link = identity.sign_state_link(state_id="s1", version=1)
    forged = replace(link, signature="00" * 64)
    assert identity.verify_link(forged) is False


def test_rotation_is_attested_by_the_outgoing_key(identity) -> None:
    """A key with no rotation story gets replaced by silently starting over."""

    original_public = identity.public_key_hex
    original_id = identity.entity_id

    record = identity.rotate(reason="scheduled probe")

    assert identity.entity_id != original_id
    assert record.predecessor_entity_id == original_id
    assert record.successor_entity_id == identity.entity_id
    # A verifier holding only the ORIGINAL public key can follow the chain.
    assert identity.verify_succession(record, original_public) is True
    # And cannot be fooled by the successor's own key.
    assert identity.verify_succession(record, identity.public_key_hex) is False


def test_rotation_persists(tmp_path) -> None:
    root = tmp_path / "identity"
    identity = entity_identity(root)
    identity.rotate(reason="probe")
    reloaded = EntityIdentity(root)
    assert reloaded.entity_id == identity.entity_id
    assert len(reloaded.successions()) == 1


def test_anchor_reports_unanchored_rather_than_inventing_one(monkeypatch) -> None:
    """A plausible-looking id would be the worst possible failure here."""

    import core.identity.identity_anchor as anchor_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no key")

    monkeypatch.setattr("core.identity.entity_key.entity_identity", _boom)
    anchor = anchor_module.IdentityAnchor()
    assert anchor.get_identity() == "Aura-Unanchored"


def test_anchor_does_not_consult_state_at_all() -> None:
    """The dependency that made it transient is gone, not merely reduced."""

    import inspect

    import core.identity.identity_anchor as anchor_module

    source = inspect.getsource(anchor_module.IdentityAnchor.get_identity)
    assert "state_id" not in source
    assert "state_repo" not in source
    assert "entity_identity" in source


def test_name_and_identity_are_separate_questions() -> None:
    import inspect

    import core.identity.identity_anchor as anchor_module

    identity_source = inspect.getsource(anchor_module.IdentityAnchor.get_identity)
    assert "display_name" not in identity_source
    assert hasattr(anchor_module.IdentityAnchor, "display_name")


def test_commit_path_signs_the_lineage() -> None:
    """The wiring, against the repository's own source."""

    import inspect

    import core.state.state_repository as repo

    commit = inspect.getsource(repo.StateRepository._process_commit_transaction)
    assert "_sign_state_lineage" in commit

    signer = inspect.getsource(repo._sign_state_lineage)
    assert "sign_state_link" in signer
    assert "get_continuity_hash" in signer


def test_aura_state_carries_the_signature_fields() -> None:
    from core.state.aura_state import AuraState

    state = AuraState()
    assert state.lineage_signature == ""
    assert state.lineage_entity_id == ""
