"""Checkpointed replacement for the complete-prefix intervention envelope.

An action-intervention envelope today carries every journal event from genesis.
That is honest and quadratic. This is the compact form SPARK-068 asks for, and
the whole design turns on being precise about what it does and does not prove.

A witness carries three things:

1. An **accumulator root** over the journal's event digests, with the size
   bound in. This proves the log is exactly the ordered run it claims to be.
2. A **checkpoint**: the folded journal state at some sequence k, plus an
   O(log n) inclusion proof that the k-th event is really in that log. The
   state is bounded by the plan's cell count, not by journal length.
3. The **suffix**: events k..head, which the verifier folds itself.

Verification is then O(suffix + log n) instead of O(n), so a campaign's total
verification cost stops being quadratic in its own length.

**What a checkpoint is trusted for, stated plainly.** The witness cannot prove
its own checkpoint state. Someone had to replay genesis→k once to compute it,
and the verifier accepts that party's answer. So `verify_journal_witness`
*requires* the caller to pass the checkpoint state digest it independently
trusts, and refuses if the witness disagrees. There is no default, no "if
omitted, trust the witness" path — that argument being mandatory is the whole
safety property. A verifier that wants to trust nobody passes
`checkpoint_sequence=0`, which degenerates to a full replay and is exactly the
behavior the complete-prefix envelope has today.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from core.brain.llm.latent_cortex.campaign_journal import CampaignPlan
from core.brain.llm.latent_cortex.journal_accumulator import (
    accumulator_root,
    inclusion_proof,
    verify_inclusion,
)
from core.brain.llm.latent_cortex.journal_state import (
    GENESIS_PREVIOUS,
    assert_active_attempt,
    initial_journal_state,
    journal_state_from_dict,
    replay_journal,
)

JOURNAL_WITNESS_SCHEMA: Final = "aura.latent_cortex.campaign_journal_witness.v2"


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_journal_witness(
    events: Sequence[Mapping[str, Any]],
    *,
    plan: CampaignPlan,
    attempt_id_for: Callable[..., str],
    checkpoint_sequence: int,
) -> dict[str, Any]:
    """Compress a validated journal prefix into a checkpointed witness.

    Building requires the full events, because the builder is the party doing
    the replay. Verifying does not.
    """

    if (
        not isinstance(events, Sequence)
        or isinstance(events, (str, bytes))
        or not events
    ):
        raise ValueError("action intervention campaign journal prefix is invalid")
    if (
        type(checkpoint_sequence) is not int
        or not 0 <= checkpoint_sequence <= len(events)
    ):
        raise ValueError("action intervention campaign journal checkpoint is invalid")

    state, _ = replay_journal(
        events[:checkpoint_sequence],
        plan=plan,
        attempt_id_for=attempt_id_for,
    )
    # Fold the suffix too, so a witness is never built over a prefix its own
    # builder could not validate.
    replay_journal(
        events[checkpoint_sequence:],
        plan=plan,
        attempt_id_for=attempt_id_for,
        state=state,
    )

    digests = [str(event["event_sha256"]) for event in events]
    commitment = accumulator_root(digests)
    inclusion = (
        None
        if checkpoint_sequence == 0
        else inclusion_proof(digests, checkpoint_sequence - 1)
    )

    body = {
        "schema": JOURNAL_WITNESS_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "accumulator": {
            "size": commitment["size"],
            "root_sha256": commitment["root_sha256"],
        },
        "checkpoint": {
            "sequence": checkpoint_sequence,
            "state": state.to_dict(),
            "state_sha256": state.digest(),
            "inclusion": inclusion,
        },
        "suffix": [dict(event) for event in events[checkpoint_sequence:]],
        "head_event_sha256": digests[-1],
        "head_event_count": len(events),
    }
    return {**body, "witness_sha256": _sha256(body)}


def verify_journal_witness(
    witness: Any,
    *,
    plan: CampaignPlan,
    authority: Mapping[str, Any],
    attempt_id_for: Callable[..., str],
    trusted_checkpoint_state_sha256: str | None,
) -> list[dict[str, Any]]:
    """Verify a witness and return the suffix transcript it establishes.

    ``trusted_checkpoint_state_sha256`` is mandatory and has no default. Pass
    ``None`` only together with a genesis checkpoint, which trusts nothing and
    replays everything.
    """

    if not isinstance(witness, Mapping) or witness.get("schema") != JOURNAL_WITNESS_SCHEMA:
        raise ValueError("action intervention campaign journal witness is invalid")
    body = {name: witness[name] for name in witness if name != "witness_sha256"}
    if witness.get("witness_sha256") != _sha256(body):
        raise ValueError("action intervention campaign journal witness differs")
    if witness.get("plan_sha256") != plan.plan_sha256:
        raise ValueError("action intervention campaign journal witness plan differs")

    accumulator = witness.get("accumulator")
    checkpoint = witness.get("checkpoint")
    suffix = witness.get("suffix")
    if (
        not isinstance(accumulator, Mapping)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(suffix, Sequence)
        or isinstance(suffix, (str, bytes))
    ):
        raise ValueError("action intervention campaign journal witness is invalid")

    size = accumulator.get("size")
    if (
        type(size) is not int
        or size != witness.get("head_event_count")
        or size != authority["journal_event_count"]
    ):
        raise ValueError("action intervention campaign journal witness size differs")

    sequence = checkpoint.get("sequence")
    if type(sequence) is not int or not 0 <= sequence <= size:
        raise ValueError("action intervention campaign journal checkpoint is invalid")
    if len(suffix) != size - sequence:
        raise ValueError("action intervention campaign journal suffix differs")

    if sequence == 0:
        if (
            trusted_checkpoint_state_sha256 is not None
            or checkpoint.get("inclusion") is not None
        ):
            raise ValueError(
                "action intervention campaign journal checkpoint is invalid"
            )
        state = initial_journal_state()
        if checkpoint.get("state_sha256") != state.digest():
            raise ValueError(
                "action intervention campaign journal checkpoint state differs"
            )
    else:
        if not isinstance(trusted_checkpoint_state_sha256, str):
            raise ValueError(
                "action intervention campaign journal checkpoint is not trusted"
            )
        state = journal_state_from_dict(checkpoint.get("state"))
        if (
            state.digest() != checkpoint.get("state_sha256")
            or state.digest() != trusted_checkpoint_state_sha256
            or state.sequence != sequence
        ):
            raise ValueError(
                "action intervention campaign journal checkpoint state differs"
            )
        inclusion = checkpoint.get("inclusion")
        if (
            not isinstance(inclusion, Mapping)
            or inclusion.get("event_sha256") != state.previous_event_sha256
            or inclusion.get("index") != sequence - 1
            or not verify_inclusion(
                dict(inclusion),
                root_sha256=str(accumulator.get("root_sha256")),
                size=size,
            )
        ):
            raise ValueError(
                "action intervention campaign journal checkpoint is not included"
            )

    final, transcript = replay_journal(
        suffix,
        plan=plan,
        attempt_id_for=attempt_id_for,
        state=state,
    )
    if (
        final.previous_event_sha256 != witness.get("head_event_sha256")
        or final.sequence != size
    ):
        raise ValueError("action intervention campaign journal head differs")

    # At a genesis checkpoint the suffix *is* the whole log, so the root is
    # fully re-derivable and gets re-derived. Past a checkpoint it is not: the
    # verifier holds only the suffix. Re-comparing the witness's own root
    # against itself there would be a check that cannot fail, so it is not
    # performed. What binds the suffix instead is stronger than that would
    # have been: the checkpoint event's inclusion is proven against the root
    # above, and every suffix event chains to it by `previous_event_sha256`.
    # Substituting a suffix from another log therefore requires a SHA-256
    # collision, not merely a matching root.
    if sequence == 0:
        digests = [str(event["event_sha256"]) for event in transcript]
        if accumulator_root(digests)["root_sha256"] != accumulator.get("root_sha256"):
            raise ValueError("action intervention campaign journal witness root differs")

    assert_active_attempt(final, authority=authority)
    return transcript


def witness_payload_events(witness: Mapping[str, Any]) -> int:
    """How many events a witness actually carries -- the scaling claim."""

    return len(witness["suffix"])


__all__ = [
    "GENESIS_PREVIOUS",
    "JOURNAL_WITNESS_SCHEMA",
    "build_journal_witness",
    "verify_journal_witness",
    "witness_payload_events",
]
