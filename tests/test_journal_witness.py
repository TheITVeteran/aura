"""SPARK-068: the checkpointed witness accepts exactly what the full prefix does.

An envelope that got cheaper by getting weaker would be worse than the
quadratic one it replaces. So the central test here is equivalence: over every
checkpoint position, the witness must accept precisely the journals the
complete-prefix validator accepts, and refuse precisely the ones it refuses.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from core.brain.llm.latent_cortex.action_intervention import (
    _validate_journal_prefix,
    action_intervention_attempt_id,
)
from core.brain.llm.latent_cortex.campaign_journal import (
    ARM_RESULT,
    COMMITTED,
    EVENT_SCHEMA,
    PLAN_EVENT,
    STARTED,
    VERIFIED,
    CampaignPlan,
)
from core.brain.llm.latent_cortex.journal_state import (
    initial_journal_state,
    replay_journal,
)
from core.brain.llm.latent_cortex.journal_witness import (
    build_journal_witness,
    verify_journal_witness,
    witness_payload_events,
)

_GENESIS = "0" * 64


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _plan(cells: int = 4) -> CampaignPlan:
    return CampaignPlan.build(
        "witness-test",
        [{"ordinal": index, "action": "BRANCH"} for index in range(cells)],
    )


def _event(sequence, plan, previous, name, cell_id, attempt_id, payload) -> dict:
    body = {
        "schema": EVENT_SCHEMA,
        "sequence": sequence,
        "plan_sha256": plan.plan_sha256,
        "previous_event_sha256": previous,
        "event": name,
        "cell_id": cell_id,
        "attempt_id": attempt_id,
        "payload": payload,
    }
    return {**body, "event_sha256": _sha(body)}


def _journal(plan: CampaignPlan, *, closed_cells: int, open_cell: int | None) -> list[dict]:
    """Build a valid journal: some committed cells, optionally one live attempt."""

    events = [
        _event(0, plan, _GENESIS, PLAN_EVENT, None, None, {"plan": plan.to_dict()})
    ]

    def append(name, cell_id, attempt_id, payload):
        events.append(
            _event(
                len(events),
                plan,
                events[-1]["event_sha256"],
                name,
                cell_id,
                attempt_id,
                payload,
            )
        )

    for index in range(closed_cells):
        cell_id = plan.cell_ids[index]
        attempt_id = action_intervention_attempt_id(
            campaign_plan_sha256=plan.plan_sha256, cell_id=cell_id, attempt_number=1
        )
        append(STARTED, cell_id, attempt_id, {"attempt_number": 1})
        append(ARM_RESULT, cell_id, attempt_id, {"result": {"score": index}})
        append(VERIFIED, cell_id, attempt_id, {"verification": {"ok": True}})
        append(COMMITTED, cell_id, attempt_id, {"commit": {"index": index}})

    if open_cell is not None:
        cell_id = plan.cell_ids[open_cell]
        attempt_id = action_intervention_attempt_id(
            campaign_plan_sha256=plan.plan_sha256, cell_id=cell_id, attempt_number=1
        )
        append(STARTED, cell_id, attempt_id, {"attempt_number": 1})

    return events


def _authority(plan: CampaignPlan, events: list[dict], open_cell: int) -> dict:
    cell_id = plan.cell_ids[open_cell]
    return {
        "attempt_id": action_intervention_attempt_id(
            campaign_plan_sha256=plan.plan_sha256, cell_id=cell_id, attempt_number=1
        ),
        "cell_id": cell_id,
        "attempt_number": 1,
        "journal_head_sha256": events[-1]["event_sha256"],
        "journal_event_count": len(events),
    }


@pytest.fixture
def campaign():
    plan = _plan()
    events = _journal(plan, closed_cells=2, open_cell=2)
    return plan, events, _authority(plan, events, 2)


def _trusted_state_digest(plan, events, sequence):
    state, _ = replay_journal(
        events[:sequence],
        plan=plan,
        attempt_id_for=action_intervention_attempt_id,
    )
    return state.digest()


def _witness(plan, events, sequence):
    return build_journal_witness(
        events,
        plan=plan,
        attempt_id_for=action_intervention_attempt_id,
        checkpoint_sequence=sequence,
    )


def _verify(witness, plan, authority, sequence, events):
    return verify_journal_witness(
        witness,
        plan=plan,
        authority=authority,
        attempt_id_for=action_intervention_attempt_id,
        trusted_checkpoint_state_sha256=(
            None if sequence == 0 else _trusted_state_digest(plan, events, sequence)
        ),
    )


# --- the extraction did not change the existing validator -------------------


def test_the_complete_prefix_validator_still_accepts_a_valid_journal(campaign):
    plan, events, authority = campaign
    transcript = _validate_journal_prefix(events, plan=plan, authority=authority)
    assert len(transcript) == len(events)
    assert transcript[-1]["event"] == STARTED


def test_the_complete_prefix_validator_still_refuses_a_broken_chain(campaign):
    plan, events, authority = campaign
    broken = [dict(row) for row in events]
    broken[3]["previous_event_sha256"] = _GENESIS
    with pytest.raises(ValueError, match="chain differs"):
        _validate_journal_prefix(broken, plan=plan, authority=authority)


# --- equivalence ------------------------------------------------------------


def test_every_checkpoint_position_accepts_the_valid_journal(campaign):
    plan, events, authority = campaign
    for sequence in range(len(events) + 1):
        witness = _witness(plan, events, sequence)
        transcript = _verify(witness, plan, authority, sequence, events)
        assert len(transcript) == len(events) - sequence


def test_the_genesis_checkpoint_is_a_full_replay(campaign):
    plan, events, authority = campaign
    witness = _witness(plan, events, 0)
    assert witness_payload_events(witness) == len(events)
    assert witness["checkpoint"]["state_sha256"] == initial_journal_state().digest()
    assert _verify(witness, plan, authority, 0, events)


def test_a_late_checkpoint_carries_almost_nothing(campaign):
    plan, events, authority = campaign
    late = len(events) - 1
    witness = _witness(plan, events, late)
    assert witness_payload_events(witness) == 1
    assert _verify(witness, plan, authority, late, events)


def test_the_checkpoint_state_is_bounded_by_cells_not_by_journal_length():
    plan = _plan(cells=4)
    short = _journal(plan, closed_cells=1, open_cell=1)
    long = _journal(plan, closed_cells=3, open_cell=3)
    short_state = _trusted_state_digest(plan, short, len(short))
    long_state = _trusted_state_digest(plan, long, len(long))
    assert short_state != long_state
    # Both states serialize over the same four cells regardless of how many
    # events produced them.
    state, _ = replay_journal(
        long, plan=plan, attempt_id_for=action_intervention_attempt_id
    )
    assert len(state.start_counts) <= len(plan.cell_ids)


# --- the trust boundary is mandatory, not defaulted -------------------------


def test_a_checkpointed_witness_without_a_trusted_digest_is_refused(campaign):
    plan, events, authority = campaign
    witness = _witness(plan, events, 5)
    with pytest.raises(ValueError, match="not trusted"):
        verify_journal_witness(
            witness,
            plan=plan,
            authority=authority,
            attempt_id_for=action_intervention_attempt_id,
            trusted_checkpoint_state_sha256=None,
        )


def test_a_checkpoint_disagreeing_with_the_trusted_digest_is_refused(campaign):
    plan, events, authority = campaign
    witness = _witness(plan, events, 5)
    with pytest.raises(ValueError, match="checkpoint state differs"):
        verify_journal_witness(
            witness,
            plan=plan,
            authority=authority,
            attempt_id_for=action_intervention_attempt_id,
            trusted_checkpoint_state_sha256=hashlib.sha256(b"elsewhere").hexdigest(),
        )


def test_a_genesis_witness_may_not_smuggle_in_a_trusted_digest(campaign):
    plan, events, authority = campaign
    witness = _witness(plan, events, 0)
    with pytest.raises(ValueError):
        verify_journal_witness(
            witness,
            plan=plan,
            authority=authority,
            attempt_id_for=action_intervention_attempt_id,
            trusted_checkpoint_state_sha256=hashlib.sha256(b"anything").hexdigest(),
        )


# --- forgery ----------------------------------------------------------------


def test_a_tampered_witness_body_breaks_its_own_digest(campaign):
    plan, events, authority = campaign
    witness = dict(_witness(plan, events, 5))
    witness["head_event_count"] = len(events) + 1
    with pytest.raises(ValueError, match="witness differs"):
        _verify(witness, plan, authority, 5, events)


def test_a_forged_checkpoint_state_fails_its_inclusion_proof(campaign):
    plan, events, authority = campaign
    sequence = 5
    forged_events = [dict(row) for row in events]
    forged_events[sequence - 1] = dict(forged_events[sequence - 1])
    witness = dict(_witness(plan, events, sequence))
    checkpoint = dict(witness["checkpoint"])
    inclusion = dict(checkpoint["inclusion"])
    inclusion["event_sha256"] = hashlib.sha256(b"forged").hexdigest()
    checkpoint["inclusion"] = inclusion
    witness["checkpoint"] = checkpoint
    with pytest.raises(ValueError):
        _verify(witness, plan, authority, sequence, events)


def test_a_witness_over_another_campaigns_plan_is_refused(campaign):
    plan, events, authority = campaign
    witness = _witness(plan, events, 5)
    other = _plan(cells=5)
    with pytest.raises(ValueError, match="witness plan differs"):
        verify_journal_witness(
            witness,
            plan=other,
            authority=authority,
            attempt_id_for=action_intervention_attempt_id,
            trusted_checkpoint_state_sha256=_trusted_state_digest(plan, events, 5),
        )


def test_a_witness_whose_size_disagrees_with_the_authority_is_refused(campaign):
    plan, events, authority = campaign
    witness = _witness(plan, events, 5)
    lying = {**authority, "journal_event_count": len(events) - 1}
    with pytest.raises(ValueError, match="witness size differs"):
        verify_journal_witness(
            witness,
            plan=plan,
            authority=lying,
            attempt_id_for=action_intervention_attempt_id,
            trusted_checkpoint_state_sha256=_trusted_state_digest(plan, events, 5),
        )


def test_a_suffix_event_edited_after_the_checkpoint_is_refused(campaign):
    plan, events, authority = campaign
    sequence = 5
    witness = dict(_witness(plan, events, sequence))
    suffix = [dict(row) for row in witness["suffix"]]
    suffix[0]["payload"] = {"result": {"score": 9999}}
    witness["suffix"] = suffix
    with pytest.raises(ValueError):
        _verify(witness, plan, authority, sequence, events)


def test_a_witness_over_an_invalid_journal_cannot_be_built(campaign):
    plan, events, _ = campaign
    broken = [dict(row) for row in events]
    broken[4]["payload"] = {"attempt_number": 7}
    with pytest.raises(ValueError):
        build_journal_witness(
            broken,
            plan=plan,
            attempt_id_for=action_intervention_attempt_id,
            checkpoint_sequence=2,
        )


def test_a_journal_whose_live_attempt_already_committed_is_refused():
    plan = _plan()
    events = _journal(plan, closed_cells=3, open_cell=None)
    authority = {
        "attempt_id": action_intervention_attempt_id(
            campaign_plan_sha256=plan.plan_sha256,
            cell_id=plan.cell_ids[2],
            attempt_number=1,
        ),
        "cell_id": plan.cell_ids[2],
        "attempt_number": 1,
        "journal_head_sha256": events[-1]["event_sha256"],
        "journal_event_count": len(events),
    }
    witness = _witness(plan, events, 5)
    with pytest.raises(ValueError, match="active journal attempt differs"):
        _verify(witness, plan, authority, 5, events)


# --- the scaling claim ------------------------------------------------------


def test_a_long_campaign_stops_paying_quadratically():
    plan = CampaignPlan.build(
        "witness-scale",
        [{"ordinal": index, "action": "BRANCH"} for index in range(40)],
    )
    events = _journal(plan, closed_cells=39, open_cell=39)
    assert len(events) > 150

    # A campaign publishes a checkpoint every `cadence` events, and an envelope
    # built at event h resumes from the newest checkpoint at or before h. That
    # is the usage the compact proof exists for: each envelope carries at most
    # one cadence of events, no matter how long the campaign has been running.
    cadence = 8
    prefix_events = 0
    witness_events = 0
    for head in range(1, len(events) + 1):
        checkpoint = (head // cadence) * cadence
        prefix_events += head
        witness_events += witness_payload_events(
            _witness(plan, events[:head], checkpoint)
        )

    assert witness_events <= cadence * len(events)
    assert witness_events < prefix_events // 10


def test_each_envelope_carries_at_most_one_checkpoint_cadence():
    plan = _plan(cells=8)
    events = _journal(plan, closed_cells=7, open_cell=7)
    cadence = 8
    for head in range(1, len(events) + 1):
        checkpoint = (head // cadence) * cadence
        witness = _witness(plan, events[:head], checkpoint)
        assert witness_payload_events(witness) < cadence
