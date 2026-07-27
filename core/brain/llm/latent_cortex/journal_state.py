"""The campaign journal's transition function, in one place.

`action_intervention` validated a journal by replaying every event from genesis
inside each envelope. The compact-proof work in `journal_accumulator` needs the
same replay, but starting from a checkpoint instead of from zero. Two copies of
a state machine drift, and a drifted copy is worse than no copy: it would let an
envelope pass one validator and fail the other, which is how a proof surface
loses its meaning.

So the fold lives here and both paths call it. Nothing about the rules changed
in the extraction -- the transitions, the ordering constraints, and the refusal
messages are the ones the full-prefix validator has always used.

The state is also digestible. A checkpoint that names its folded state digest
lets a verifier resume from there and replay only the suffix, which is what
turns an O(n)-per-envelope proof into an O(suffix + log n) one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from core.brain.llm.latent_cortex.campaign_journal import (
    ACTION_INTERVENTION_CLAIMED,
    ARM_RESULT,
    COMMITTED,
    EVENT_SCHEMA,
    FAILED,
    PLAN_EVENT,
    STARTED,
    VERIFIED,
    CampaignPlan,
)

JOURNAL_STATE_SCHEMA: Final = "aura.latent_cortex.campaign_journal_state.v1"
GENESIS_PREVIOUS: Final = "0" * 64

EVENT_FIELDS: Final = frozenset(
    {
        "schema",
        "sequence",
        "plan_sha256",
        "previous_event_sha256",
        "event",
        "cell_id",
        "attempt_id",
        "payload",
        "event_sha256",
    }
)


def _sha256(value: Any) -> str:
    # Byte-identical to the digest `action_intervention` has always taken over
    # these events. The extraction must not change one hash.
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise ValueError("action intervention is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class JournalState:
    """Everything the next event's admissibility depends on."""

    sequence: int = 0
    previous_event_sha256: str = GENESIS_PREVIOUS
    attempts: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_by_cell: dict[str, str] = field(default_factory=dict)
    start_counts: dict[str, int] = field(default_factory=dict)
    committed_cells: frozenset[str] = frozenset()

    def digest(self) -> str:
        """A deterministic commitment to the folded state.

        This is what a checkpoint carries so a verifier can resume from it
        rather than replaying the whole journal.
        """

        return _sha256(self.to_dict())


    def to_dict(self) -> dict[str, Any]:
        """Serialize the state so a checkpoint can carry it.

        This is bounded by the plan's cell count, not by journal length, which
        is what makes a checkpointed proof compact rather than merely smaller.
        """

        return {
            "schema": JOURNAL_STATE_SCHEMA,
            "sequence": self.sequence,
            "previous_event_sha256": self.previous_event_sha256,
            "attempts": {
                attempt_id: {
                    "cell_id": row["cell_id"],
                    "attempt_number": row["attempt_number"],
                    "state": row["state"],
                }
                for attempt_id, row in sorted(self.attempts.items())
            },
            "active_by_cell": dict(sorted(self.active_by_cell.items())),
            "start_counts": dict(sorted(self.start_counts.items())),
            "committed_cells": sorted(self.committed_cells),
        }


def journal_state_from_dict(value: Any) -> JournalState:
    """Rebuild a checkpoint state, refusing anything malformed."""

    if not isinstance(value, Mapping) or value.get("schema") != JOURNAL_STATE_SCHEMA:
        raise ValueError("action intervention campaign journal state is invalid")
    sequence = value.get("sequence")
    previous = value.get("previous_event_sha256")
    attempts = value.get("attempts")
    active = value.get("active_by_cell")
    counts = value.get("start_counts")
    committed = value.get("committed_cells")
    if (
        type(sequence) is not int
        or sequence < 0
        or not (previous == GENESIS_PREVIOUS or _is_sha256(previous))
        or not isinstance(attempts, Mapping)
        or not isinstance(active, Mapping)
        or not isinstance(counts, Mapping)
        or not isinstance(committed, Sequence)
        or isinstance(committed, (str, bytes))
    ):
        raise ValueError("action intervention campaign journal state is invalid")
    rebuilt: dict[str, dict[str, Any]] = {}
    for attempt_id, row in attempts.items():
        if (
            not isinstance(attempt_id, str)
            or not isinstance(row, Mapping)
            or set(row) != {"cell_id", "attempt_number", "state"}
        ):
            raise ValueError("action intervention campaign journal state is invalid")
        rebuilt[attempt_id] = dict(row)
    return JournalState(
        sequence=sequence,
        previous_event_sha256=str(previous),
        attempts=rebuilt,
        active_by_cell={str(key): str(item) for key, item in active.items()},
        start_counts={str(key): int(item) for key, item in counts.items()},
        committed_cells=frozenset(str(cell) for cell in committed),
    )


def initial_journal_state() -> JournalState:
    return JournalState()


def fold_event(
    state: JournalState,
    raw: Any,
    *,
    plan: CampaignPlan,
    attempt_id_for: Callable[..., str],
) -> tuple[JournalState, dict[str, Any]]:
    """Admit exactly one event, or refuse it by name.

    ``attempt_id_for`` is injected rather than imported: the deterministic
    attempt identity lives in `action_intervention`, which calls this fold, and
    a direct import would close that loop.

    Returns the state after the event and the normalized event itself.
    """

    if not isinstance(raw, Mapping) or set(raw) != EVENT_FIELDS:
        raise ValueError("action intervention campaign journal event differs")
    event = dict(raw)
    body = {name: event[name] for name in EVENT_FIELDS - {"event_sha256"}}
    sequence = state.sequence
    if (
        event.get("schema") != EVENT_SCHEMA
        or event.get("sequence") != sequence
        or event.get("plan_sha256") != plan.plan_sha256
        or event.get("previous_event_sha256") != state.previous_event_sha256
        or event.get("event_sha256") != _sha256(body)
    ):
        raise ValueError("action intervention campaign journal chain differs")

    event_name = event.get("event")
    cell_id = event.get("cell_id")
    attempt_id = event.get("attempt_id")
    payload = event.get("payload")

    attempts = {key: dict(value) for key, value in state.attempts.items()}
    active_by_cell = dict(state.active_by_cell)
    start_counts = dict(state.start_counts)
    committed_cells = set(state.committed_cells)

    if sequence == 0:
        if (
            event_name != PLAN_EVENT
            or cell_id is not None
            or attempt_id is not None
            or payload != {"plan": plan.to_dict()}
        ):
            raise ValueError("action intervention campaign journal genesis differs")
    else:
        if (
            not isinstance(cell_id, str)
            or cell_id not in plan.cell_ids
            or not isinstance(attempt_id, str)
            or not isinstance(payload, Mapping)
        ):
            raise ValueError("action intervention campaign journal identity differs")
        if event_name == STARTED:
            attempt_number = start_counts.get(cell_id, 0) + 1
            if (
                payload != {"attempt_number": attempt_number}
                or cell_id in active_by_cell
                or cell_id in committed_cells
                or attempt_id in attempts
                or attempt_id
                != attempt_id_for(
                    campaign_plan_sha256=plan.plan_sha256,
                    cell_id=cell_id,
                    attempt_number=attempt_number,
                )
            ):
                raise ValueError("action intervention campaign journal attempt differs")
            start_counts[cell_id] = attempt_number
            attempts[attempt_id] = {
                "cell_id": cell_id,
                "attempt_number": attempt_number,
                "state": STARTED,
            }
            active_by_cell[cell_id] = attempt_id
        else:
            attempt = attempts.get(attempt_id)
            if (
                attempt is None
                or attempt["cell_id"] != cell_id
                or active_by_cell.get(cell_id) != attempt_id
            ):
                raise ValueError(
                    "action intervention campaign journal attempt is inactive"
                )
            attempt_state = attempt["state"]
            if event_name == ACTION_INTERVENTION_CLAIMED:
                valid = (
                    attempt_state == STARTED
                    and set(payload)
                    == {
                        "intervention_sha256",
                        "request_payload_sha256",
                        "signed_journal_head_sha256",
                        "signed_journal_event_count",
                    }
                    and _is_sha256(payload.get("intervention_sha256"))
                    and _is_sha256(payload.get("request_payload_sha256"))
                    and payload.get("signed_journal_head_sha256")
                    == event.get("previous_event_sha256")
                    and payload.get("signed_journal_event_count") == sequence
                )
            elif event_name == ARM_RESULT:
                valid = attempt_state in {
                    STARTED,
                    ACTION_INTERVENTION_CLAIMED,
                } and set(payload) == {"result"}
            elif event_name == VERIFIED:
                valid = attempt_state == ARM_RESULT and set(payload) == {"verification"}
            elif event_name == COMMITTED:
                valid = attempt_state == VERIFIED and set(payload) == {"commit"}
            elif event_name == FAILED:
                valid = (
                    set(payload) == {"details", "reason"}
                    and isinstance(payload.get("details"), Mapping)
                    and isinstance(payload.get("reason"), str)
                    and bool(payload["reason"].strip())
                )
            else:
                valid = False
            if not valid:
                raise ValueError(
                    "action intervention campaign journal transition differs"
                )
            attempt["state"] = event_name
            if event_name in {COMMITTED, FAILED}:
                del active_by_cell[cell_id]
            if event_name == COMMITTED:
                committed_cells.add(cell_id)

    return (
        JournalState(
            sequence=sequence + 1,
            previous_event_sha256=str(event["event_sha256"]),
            attempts=attempts,
            active_by_cell=active_by_cell,
            start_counts=start_counts,
            committed_cells=frozenset(committed_cells),
        ),
        event,
    )


def replay_journal(
    events: Sequence[Mapping[str, Any]],
    *,
    plan: CampaignPlan,
    attempt_id_for: Callable[..., str],
    state: JournalState | None = None,
) -> tuple[JournalState, list[dict[str, Any]]]:
    """Fold a run of events, from genesis or from a checkpoint."""

    current = initial_journal_state() if state is None else state
    transcript: list[dict[str, Any]] = []
    for raw in events:
        current, event = fold_event(
            current, raw, plan=plan, attempt_id_for=attempt_id_for
        )
        transcript.append(event)
    return current, transcript


def assert_active_attempt(
    state: JournalState,
    *,
    authority: Mapping[str, Any],
) -> None:
    """Refuse unless the authority's attempt is the live one at this state."""

    target_attempt = state.attempts.get(str(authority["attempt_id"]))
    if (
        state.previous_event_sha256 != authority["journal_head_sha256"]
        or state.active_by_cell.get(authority["cell_id"]) != authority["attempt_id"]
        or target_attempt is None
        or target_attempt["state"] != STARTED
        or target_attempt["attempt_number"] != authority["attempt_number"]
    ):
        raise ValueError("action intervention active journal attempt differs")


__all__ = [
    "EVENT_FIELDS",
    "GENESIS_PREVIOUS",
    "JOURNAL_STATE_SCHEMA",
    "JournalState",
    "assert_active_attempt",
    "fold_event",
    "initial_journal_state",
    "journal_state_from_dict",
    "replay_journal",
]
