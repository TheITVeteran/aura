"""The person saw it, the record failed, and it was queued again.

`ObservationRouter._delivery_loop` does this:

    await self._deliver(...)                    # reaches the person
    await self._historian.mark_delivered(...)   # records that it did

Both sit inside one `try`. So a historian fault on the SECOND line — a
transient SQLite lock, a WAL checkpoint, a disk hiccup — landed in the same
handler as a sink that never delivered anything, and that handler calls
`mark_delivery_failed`, which requeues. The observation had already reached
the person; requeuing showed it to them a second time, and the recorded
error blamed the delivery for a bookkeeping fault.

The state machine had no way to say the third thing that actually happens.
`mark_delivered` means it worked; `mark_delivery_failed` means it did not,
so try again. "It worked and we could not write that down" had no
transition, so the router was forced to pick the one that redelivers.

`quarantine_delivery` is that transition: terminal, no redelivery, row kept
for the recovery audit.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────── the two failures are told apart


def test_the_loop_tracks_whether_the_person_actually_saw_it():
    """Without this flag both failures are the same exception in the same
    handler, and the handler has to guess."""
    source = (
        ROOT / "core" / "reality_reach" / "observation_router.py"
    ).read_text("utf-8")

    assert "reached_the_person = True" in source
    assert "reached_the_person = False" in source


def test_the_flag_is_set_after_delivery_not_before():
    """Set before `_deliver` and it lies about every failed send."""
    source = (
        ROOT / "core" / "reality_reach" / "observation_router.py"
    ).read_text("utf-8")

    deliver_at = source.index("await self._deliver(")
    set_true_at = source.index("reached_the_person = True")

    assert deliver_at < set_true_at, (
        "the flag is set before the delivery completes, so a failed send "
        "would be recorded as having reached the person"
    )


def test_the_flag_resets_for_every_observation():
    """A stale True from the previous iteration would quarantine an
    observation that was never sent."""
    source = (
        ROOT / "core" / "reality_reach" / "observation_router.py"
    ).read_text("utf-8")

    reset_at = source.index("reached_the_person = False")
    next_obs_at = source.index("= await self._next_observation()")

    assert next_obs_at < reset_at, (
        "the flag is not reset after claiming each observation"
    )


def test_a_delivered_observation_never_reaches_mark_delivery_failed():
    """`mark_delivery_failed` requeues. That is right for a send that did
    not happen and wrong for one that did."""
    from core.reality_reach import observation_router as router_mod

    source = inspect.getsource(router_mod)
    tree = ast.parse(source)

    guarded = False
    for node in ast.walk(tree):
        # Exactly `if reached_the_person:` — not any enclosing `if` whose
        # source happens to span both branches, which is what a substring
        # match over the rendered node would catch.
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Name) and node.test.id == "reached_the_person"):
            continue
        body = "\n".join(
            ast.get_source_segment(source, stmt) or "" for stmt in node.body
        )
        assert "mark_delivery_failed" not in body, (
            "the delivered branch still calls mark_delivery_failed, which "
            "requeues an observation the person has already seen"
        )
        assert "_settle_unconfirmed_delivery" in body
        guarded = True

    assert guarded, "no branch settles a delivered-but-unrecorded observation"


# ─────────────────── the transition that was missing


def test_the_historian_can_terminate_without_redelivering():
    from core.reality_reach.historian import RealityHistorian

    assert hasattr(RealityHistorian, "quarantine_delivery"), (
        "the state machine still cannot say 'delivered, record failed' — "
        "only 'worked' and 'did not work, try again'"
    )


def test_quarantine_is_a_terminal_state():
    from core.reality_reach.historian import _DELIVERY_STATES

    assert "quarantined" in _DELIVERY_STATES


def test_the_settle_path_retries_the_record_before_quarantining():
    """Most historian faults here are a transient lock or a checkpoint.
    Quarantining on the first one would throw away recoverable rows."""
    from core.reality_reach import observation_router as router_mod

    body = inspect.getsource(router_mod.RealityObservationRouter._settle_unconfirmed_delivery)

    assert body.index("mark_delivered") < body.index("quarantine_delivery"), (
        "quarantine happens before the record is retried"
    )


def test_an_unquarantinable_delivery_says_a_duplicate_is_possible():
    """If even quarantine fails, lease expiry requeues it. That must be
    stated, not left looking like an ordinary retry."""
    from core.reality_reach import observation_router as router_mod

    body = inspect.getsource(router_mod.RealityObservationRouter._settle_unconfirmed_delivery)

    assert "may see it twice" in body
    assert 'severity="critical"' in body


def test_the_ordinary_failure_path_still_retries():
    """The fix must not stop a genuinely failed delivery from being retried
    — that would silently drop observations."""
    from core.reality_reach import observation_router as router_mod

    source = inspect.getsource(router_mod)

    assert "mark_delivery_failed" in source, (
        "a delivery that never reached the person must still be requeued"
    )
