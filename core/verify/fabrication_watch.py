"""core/verify/fabrication_watch.py — the audit, on the turns people actually get.

:mod:`core.verify.fabrication_audit` can tell whether a sentence claims work
the turn's record does not support. :mod:`core.verify.work_ledger` records what
ran, and is live on both tool-execution paths. The two were never introduced.

``audit_text`` had exactly one non-test caller — the validation suite — so the
evidence was collected on every turn and checked on none of them. Aura's
confabulations are specific and always the same shape ("I checked the file",
"the correlation was r = 0.83") and the detector for that shape sat one
function call away from the reply, never invoked.

This module makes that call, on every finalized turn, and does nothing else.

**It cannot change a reply.** By the time this runs the turn is finalized and
the text has been served. That is deliberate and not a limitation: this
codebase's history includes gates that DECIDED on lexical evidence and
destroyed correct answers doing it — a refusal loop where 2 of 6 refusals were
correct answers killed by a lexical-overlap gate. A fabrication finding is a
lead, not a verdict, and a lead must not be wired to a weapon. It becomes a
signal Aura can perceive about herself and an operator can read in health.

**An unknown turn is never a finding.** ``audit_text`` already returns
UNKNOWN when the ledger has no record, and only UNSUPPORTED rows are counted
here. Eviction must not manufacture fabrication — that inversion is the exact
"absence of a check reported as a passed check" family this repository keeps
rediscovering, run backwards.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock
from core.verify.fabrication_audit import Support, audit_text

__all__ = [
    "observe_served_turn",
    "recent_findings",
    "fabrication_snapshot",
    "reset_fabrication_watch_for_test",
]

#: Bounded. A long session must not turn self-observation into a memory leak.
_MAX_ROWS = 256

_LOCK = checked_lock("fabrication_watch", rank=LockRank.LEAF)
_ROWS: deque[dict[str, Any]] = deque(maxlen=_MAX_ROWS)
_TURNS_SEEN = 0
_TURNS_WITH_FINDINGS = 0
_UNKNOWN_TURNS = 0


def observe_served_turn(turn_id: str, served_text: str) -> int:
    """Audit one served reply. Returns the number of unsupported claims.

    Never raises into the caller. This sits on the finalization path of every
    turn, so a defect here must not become a defect in answering — an audit
    that can take the runtime down is worse than no audit.
    """

    global _TURNS_SEEN, _TURNS_WITH_FINDINGS, _UNKNOWN_TURNS

    text = str(served_text or "").strip()
    turn = str(turn_id or "").strip()
    if not text or not turn:
        return 0
    try:
        findings = audit_text(text, turn)
    except Exception as exc:  # noqa: BLE001 — auditing may never break a turn
        record_degradation(
            "fabrication_watch",
            exc,
            severity="debug",
            action="served turn went unaudited",
        )
        return 0

    unsupported = [f for f in findings if f.support is Support.UNSUPPORTED]
    unknown = any(f.support is Support.UNKNOWN for f in findings)

    with _LOCK:
        _TURNS_SEEN += 1
        if unknown and not unsupported:
            _UNKNOWN_TURNS += 1
        if unsupported:
            _TURNS_WITH_FINDINGS += 1
            _ROWS.append(
                {
                    "turn_id": turn,
                    "at": time.time(),
                    "findings": [f.to_dict() for f in unsupported],
                }
            )
    return len(unsupported)


def recent_findings(limit: int = 32) -> list[dict[str, Any]]:
    with _LOCK:
        rows = list(_ROWS)
    return rows[-max(0, int(limit)) :]


def fabrication_snapshot() -> dict[str, Any]:
    """Telemetry view: how often served prose outran the record.

    ``turns_unknown`` is reported separately and is NOT a fabrication count.
    A turn the ledger never saw says nothing either way, and collapsing it
    into the rate would manufacture findings out of eviction.
    """

    with _LOCK:
        seen = _TURNS_SEEN
        flagged = _TURNS_WITH_FINDINGS
        unknown = _UNKNOWN_TURNS
        rows = list(_ROWS)
    return {
        "turns_audited": seen,
        "turns_with_unsupported_claims": flagged,
        "turns_unknown_to_the_ledger": unknown,
        "unsupported_rate": round(flagged / seen, 6) if seen else 0.0,
        "recent": rows[-8:],
    }


def reset_fabrication_watch_for_test() -> None:
    global _TURNS_SEEN, _TURNS_WITH_FINDINGS, _UNKNOWN_TURNS
    with _LOCK:
        _ROWS.clear()
        _TURNS_SEEN = 0
        _TURNS_WITH_FINDINGS = 0
        _UNKNOWN_TURNS = 0
