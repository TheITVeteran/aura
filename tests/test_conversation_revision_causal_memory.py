"""The reviewer's sharp bar, pinned: a remembered exchange must causally alter
a LATER decision in a way that quoting the transcript alone cannot reproduce.

These tests exercise the ablation proof directly — the with-revisions plan must
diverge from the transcript-blind baseline, each divergence must cite its turn,
and ungrounded / agreement turns must produce no revision at all (honesty)."""
from __future__ import annotations

from dataclasses import dataclass

from core.capabilities.conversation_revision import (
    build_self_test_plan,
    default_self_claims,
    extract_revisions,
    prove_causal_influence,
    revise_from_conversation,
)


@dataclass
class _Turn:
    index: int
    observed_reply: str


_MEMORY_CHALLENGE = (
    "Interlocutor: Memory can improve behavior, but stored continuity alone "
    "does not prove consciousness because a database can preserve facts "
    "without experience."
)
_INNER_LIFE_CHALLENGE = (
    "Interlocutor: A failure case for inner-life claims is when the system "
    "reports introspection but no downstream behavior, memory, or policy "
    "changes follow from it."
)


def test_challenge_produces_grounded_turn_cited_revision():
    turns = [_Turn(1, _MEMORY_CHALLENGE)]
    revisions = extract_revisions(turns)
    assert len(revisions) == 1
    rev = revisions[0]
    assert rev.claim_id == "episodic_memory_grounds_continuity"
    assert rev.challenge_turn == 1
    assert rev.verified is True
    # grounded: the cited quote is really in the cited turn's reply
    assert rev.challenge_quote.lower() in turns[0].observed_reply.lower()
    # real policy delta: status downgraded AND test upgraded to behavioral
    assert rev.prior_status == "ASSERTED"
    assert rev.revised_status == "PROVISIONAL"
    assert rev.revised_test != rev.prior_test


def test_causal_influence_proved_by_ablation():
    turns = [_Turn(1, _MEMORY_CHALLENGE), _Turn(2, _INNER_LIFE_CHALLENGE)]
    revisions, proof = revise_from_conversation(turns)
    assert proof.causal is True
    assert len(proof.changed_items) == 2
    # every changed later-decision cites a real turn (not 0)
    assert all(item["caused_by_turn"] > 0 for item in proof.changed_items)
    assert set(proof.attribution_by_turn) == {1, 2}

    # The crux: removing the revisions reverts the plan (transcript-blind
    # baseline). The with-plan and without-plan MUST differ on the touched
    # claims — quoting the transcript cannot manufacture that difference.
    claims = default_self_claims()
    with_plan = {i["claim_id"]: i for i in proof.plan_with_revisions}
    without_plan = {i["claim_id"]: i for i in proof.plan_without_revisions}
    for item in proof.changed_items:
        cid = item["claim_id"]
        assert with_plan[cid]["test_method"] != without_plan[cid]["test_method"]
        assert without_plan[cid]["caused_by_turn"] == 0


def test_no_challenge_is_honestly_non_causal():
    # An agreeable transcript changed no decision: causal influence must be
    # reported False, not faked.
    turns = [
        _Turn(1, "Interlocutor: I completely agree, that is exactly right and well argued."),
        _Turn(2, "Interlocutor: Yes, well put, nothing to add."),
    ]
    revisions, proof = revise_from_conversation(turns)
    assert revisions == []
    assert proof.causal is False
    assert "no_adjudicated_revisions" in proof.reason
    # the plans are identical when nothing was adjudicated
    assert proof.plan_with_revisions == proof.plan_without_revisions


def test_ungrounded_revision_is_rejected():
    # A turn with no substantive challenge text must not yield a revision even
    # though a downstream consumer might want one.
    turns = [_Turn(5, "Interlocutor: Sure.")]
    assert extract_revisions(turns) == []


def test_revision_quote_must_belong_to_the_cited_turn():
    # Guard against citing turn N with a quote that lives in a different turn:
    # the verifier requires the quote to be grounded in the cited turn's reply.
    from core.capabilities.conversation_revision import PositionRevision, _verify_revision

    claims = {c.claim_id: c for c in default_self_claims()}
    claim = claims["episodic_memory_grounds_continuity"]
    turns = [_Turn(1, _MEMORY_CHALLENGE), _Turn(2, "Interlocutor: unrelated remark.")]
    forged = PositionRevision(
        claim_id=claim.claim_id,
        statement=claim.statement,
        challenge_turn=2,  # cite turn 2 ...
        challenge_quote="stored continuity alone does not prove consciousness",  # ... but quote is turn 1's
        prior_status=claim.status,
        revised_status="PROVISIONAL",
        prior_test=claim.test_method,
        revised_test=claim.upgraded_test,
        self_model_delta="forged",
    )
    ok, note = _verify_revision(forged, turns, claim)
    assert ok is False
    assert "grounded" in note


def test_no_op_revision_is_rejected():
    # A "revision" that changes neither status nor test is not a policy delta.
    from core.capabilities.conversation_revision import PositionRevision, _verify_revision

    claims = {c.claim_id: c for c in default_self_claims()}
    claim = claims["affect_is_causal"]
    turns = [_Turn(1, "Interlocutor: emotion could still be roleplay, not real valence.")]
    noop = PositionRevision(
        claim_id=claim.claim_id,
        statement=claim.statement,
        challenge_turn=1,
        challenge_quote="emotion could still be roleplay",
        prior_status=claim.status,
        revised_status=claim.status,  # no downgrade
        prior_test=claim.test_method,
        revised_test=claim.test_method,  # no upgrade
        self_model_delta="noop",
    )
    ok, note = _verify_revision(noop, turns, claim)
    assert ok is False
    assert "no-op" in note or "policy delta" in note


def test_plan_items_are_test_specs_not_transcript_quotes():
    # The downstream decision is a policy (status + behavioral test), not an
    # echo of the challenge text — this is what makes it more than a transcript.
    turns = [_Turn(1, _INNER_LIFE_CHALLENGE)]
    revisions, _ = revise_from_conversation(turns)
    plan = build_self_test_plan(default_self_claims(), revisions)
    touched = [item for item in plan if item.caused_by_turn == 1]
    assert touched, "expected the inner-life claim to be revised"
    item = touched[0]
    assert item.test_method == "require_downstream_behavior_memory_or_policy_change"
    assert item.status == "PROVISIONAL"
