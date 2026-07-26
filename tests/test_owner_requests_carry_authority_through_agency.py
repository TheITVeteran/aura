"""An owner asking for something is not an unattributed autonomous drive.

LIVE DEFECT, 2026-07-25. Bryan asked Aura to build him a checkers game and
a clean-room 2048, and got back a governance sentence instead of a game:

    WILL REFUSED: agency_orchestrator/tool_execution --
    denied_by_default: tool_execution requires validated scoped authority
    (signed_standing_authority_lease_missing)

The Will was right to refuse what it was shown. The chat route had already
established every fact needed to authorize the turn — origin "user",
foreground_request, user_explicitly_authorized, user_requested_action, with
forgeable authority keys stripped — and placed them in
``proposal.payload["context"]``. ``AgencyOrchestrator._authorize`` then built
its Will context from drive, primitive, expected_outcome, state and
simulation, and dropped the provenance one frame later. It also never asked
standing authority for a lease, which is the very thing the Will's
tool_execution gate checks for.

So every proposal arriving at the Will looked identical: an unattributed
drive. An owner sitting at the keyboard asking for a game was
indistinguishable from the runtime deciding to run a tool by itself, and the
safe answer for that shape is no.

These tests pin both halves. The owner's request must get through. The
autonomous drive must still be refused, with the same message as before —
the fix restores a distinction, it does not lower a bar.
"""
from __future__ import annotations

import pytest

from core.agency.agency_orchestrator import AgencyOrchestrator, Proposal

OWNER_CONTEXT = {
    "origin": "user",
    "route": "chat.live_runtime_proof",
    "foreground_request": True,
    "user_explicitly_authorized": True,
    "user_requested_action": True,
    "requested_authority_scope": "foreground_user_requested:chat.live_runtime_proof:program_dna",
}

AUTONOMOUS_CONTEXT = {"origin": "autonomous_initiative"}


def _proposal(context, *, primitive="tool_execution"):
    """The proposal chat.py builds for a live skill request."""
    return Proposal(
        drive="live_runtime_proof",
        intent="execute live skill program_dna: build a checkers game",
        expected_outcome="program_dna completes under governed capability execution",
        primitive=primitive,
        payload={"skill_name": "program_dna", "params": {}, "context": dict(context)},
        priority=0.85,
    )


@pytest.fixture
def strict_will(monkeypatch):
    """The live runtime runs strict default-deny; the gate only exists there.

    The existential-stakes fixture is not decoration. The Will vetoes every
    heavy domain — tool_execution included — when threat exceeds 0.75, and
    that check runs BEFORE the authority gate these tests are about. Other
    suites leave a critical-threat stakes service in the shared
    ServiceContainer, which makes both the approve and the refuse case pass
    for the wrong reason. Pinning threat to zero keeps the assertions
    measuring authority.
    """
    monkeypatch.setenv("AURA_STRICT_WILL", "1")
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")

    from core.container import ServiceContainer
    from core.governance.will import _strict_default_deny_enabled

    class _CalmStakes:
        @staticmethod
        def get_existential_threat() -> float:
            return 0.0

    # Intercept the lookup rather than registering into the shared
    # container. Registering would leave the fixture behind when nothing was
    # registered before, and a test that quietly disables the survival veto
    # for every suite after it is the same disease being treated here.
    real_get = ServiceContainer.get.__func__

    def _get(cls, name, default="_SENTINEL"):
        if name == "existential_stakes":
            return _CalmStakes()
        return real_get(cls, name, default)

    monkeypatch.setattr(ServiceContainer, "get", classmethod(_get))
    assert _strict_default_deny_enabled() is True
    return True


class TestProvenanceSurvivesTheProposal:
    def test_the_owners_authority_reaches_the_will_context(self):
        carried = AgencyOrchestrator._proposal_authority_context(_proposal(OWNER_CONTEXT))
        assert carried["origin"] == "user"
        assert carried["foreground_request"] is True
        assert carried["user_explicitly_authorized"] is True
        assert carried["user_requested_action"] is True

    def test_an_autonomous_drive_carries_only_its_origin(self):
        carried = AgencyOrchestrator._proposal_authority_context(_proposal(AUTONOMOUS_CONTEXT))
        assert carried == {"origin": "autonomous_initiative"}

    def test_a_proposal_without_context_carries_nothing(self):
        bare = Proposal("d", "i", "o", "tool_execution", {}, 0.5)
        assert AgencyOrchestrator._proposal_authority_context(bare) == {}

    def test_a_non_mapping_context_is_ignored(self):
        hostile = Proposal("d", "i", "o", "tool_execution", {"context": "not-a-dict"}, 0.5)
        assert AgencyOrchestrator._proposal_authority_context(hostile) == {}

    @pytest.mark.parametrize(
        "forgeable",
        [
            "standing_authority_token",
            "capability_token",
            "capability_token_id",
            "scoped_authority",
            "standing_authority_grant_id",
            "standing_authority_receipt_id",
            "authority_args_digest",
        ],
    )
    def test_a_payload_may_never_assert_its_own_authority(self, forgeable):
        """Provenance may cross; a grant may not.

        Payload is reachable from places the chat route's own dict is not,
        so a forged token here would walk straight past the lease check it
        is supposed to satisfy.
        """
        context = dict(OWNER_CONTEXT) | {forgeable: "FORGED"}
        carried = AgencyOrchestrator._proposal_authority_context(_proposal(context))
        assert forgeable not in carried


class TestTheLeaseIsActuallyRequested:
    @pytest.mark.asyncio
    async def test_an_owner_request_mints_a_lease(self):
        minted = await AgencyOrchestrator()._mint_tool_authority(
            _proposal(OWNER_CONTEXT), dict(OWNER_CONTEXT),
        )
        assert minted.get("standing_authority_token")

    @pytest.mark.asyncio
    async def test_an_autonomous_drive_gets_no_lease(self):
        minted = await AgencyOrchestrator()._mint_tool_authority(
            _proposal(AUTONOMOUS_CONTEXT), dict(AUTONOMOUS_CONTEXT),
        )
        assert not minted.get("standing_authority_token")
        assert minted.get("standing_authority_denial_reason")


class TestTheDecisionSplitsCorrectly:
    @pytest.mark.asyncio
    async def test_the_owner_gets_their_checkers_game(self, strict_will):
        """The literal reproduction of what Bryan asked for."""
        decision = await AgencyOrchestrator()._authorize(
            _proposal(OWNER_CONTEXT), {}, {"ok": True},
        )
        assert decision["decision"] == "approved", decision.get("reason")

    @pytest.mark.asyncio
    async def test_an_unattended_autonomous_drive_is_still_refused(self, strict_will):
        """The half that must NOT change. Same domain, same tool, no owner."""
        decision = await AgencyOrchestrator()._authorize(
            _proposal(AUTONOMOUS_CONTEXT), {}, {"ok": True},
        )
        assert decision["decision"] == "blocked"
        assert "signed_standing_authority_lease_missing" in str(decision["reason"])

    @pytest.mark.asyncio
    async def test_a_forged_token_does_not_buy_approval(self, strict_will):
        """An autonomous drive that claims a lease it was never issued."""
        context = dict(AUTONOMOUS_CONTEXT) | {"standing_authority_token": "FORGED"}
        decision = await AgencyOrchestrator()._authorize(
            _proposal(context), {}, {"ok": True},
        )
        assert decision["decision"] == "blocked"

    @pytest.mark.asyncio
    async def test_claiming_owner_flags_from_a_non_user_origin_does_not_work(
        self, strict_will,
    ):
        """The flags describe an origin the route assigns; they do not create
        one. An autonomous drive asserting them stays autonomous."""
        context = {
            "origin": "autonomous_initiative",
            "foreground_request": True,
            "user_explicitly_authorized": True,
            "user_requested_action": True,
        }
        decision = await AgencyOrchestrator()._authorize(
            _proposal(context), {}, {"ok": True},
        )
        assert decision["decision"] == "blocked"


class TestNonToolProposalsAreUnaffected:
    @pytest.mark.asyncio
    async def test_a_memory_write_does_not_request_a_tool_lease(self, monkeypatch):
        """Only tool_execution consults standing authority; nothing else
        should acquire a lease as a side effect of this change."""
        called = False

        async def _spy(*_args, **_kwargs):
            nonlocal called
            called = True
            return {}

        orchestrator = AgencyOrchestrator()
        monkeypatch.setattr(orchestrator, "_mint_tool_authority", _spy)
        await orchestrator._authorize(
            _proposal(OWNER_CONTEXT, primitive="memory_write"), {}, {"ok": True},
        )
        assert called is False
