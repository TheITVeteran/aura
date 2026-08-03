"""One definition of what happened in a turn.

Aura's layers disagreed about what "it worked" means. Seven different
readings were live at once: the function returned; nothing raised; a
fallback eventually succeeded; the model generated text; a gate accepted
the text; the person received the text; the requested real-world effect
occurred. Those are not the same statement, and code that conflates them
produces the failure Aura kept producing — a correct answer discarded by a
gate and then reported to the person as an infrastructure failure.

Measured live: a complete, correct reply of 240 characters was rejected by
a truncation heuristic and the person was handed "I couldn't get to an
answer I'd stand behind on that one." The answer existed the whole time.
In another lane a preferred model was unavailable, the fallback completed
the task, and the turn was recorded as internal strain.

This module is the single vocabulary. Its rules:

* Nothing consequential returns a bare boolean. ``True`` cannot say which
  of the seven meanings it meant, so callers guessed, and they guessed
  differently.
* A gate ANNOTATES or TRANSFORMS a candidate. It does not destroy one.
  Every suppressed candidate stays recoverable until the turn ends, so a
  turn can never die holding an answer it could have served.
* Severity is decided at the end, after fallback and postcondition
  checking — never at the moment a component notices its own trouble.
* "Preferred lane failed, fallback succeeded" is SUCCESS WITH DEGRADATION,
  not a failed turn.
* "The tool returned" is not success. If a requested effect was declared,
  success requires an observed effect.
* ``UNKNOWN`` is the default. A turn that never said what happened is not
  a success, and this refuses to imply otherwise.
* There is exactly one terminal finalizer, it runs once, and it derives
  status from recorded evidence rather than accepting an assertion.

The status is COMPUTED, not assigned. ``finalize()`` reads the ledger. A
caller may state an intent, but cannot declare success for a turn whose
evidence does not support it — that is the whole point, because the
component that is wrong about a turn is usually the one most confident.
"""
from __future__ import annotations

import contextlib
import contextvars
import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

from core.runtime.errors import record_degradation
from core.security.structural_redaction import redact_structure, redact_text
from core.runtime.lockdep import checked_lock

__all__ = [
    "OutcomeStatus",
    "VerificationGrade",
    "UserVisibleState",
    "Candidate",
    "CandidateSuppressed",
    "EffectClaim",
    "TurnOutcome",
    "TurnOutcomeError",
    "AlreadyFinalized",
    "finalize_turn",
    "bind_turn",
    "current_turn",
    "note_candidate",
    "note_suppression",
    "recoverable_answer",
    "outcome_from_candidates",
]


class TurnOutcomeError(RuntimeError):
    """Base class for misuse of the turn-outcome contract."""


class AlreadyFinalized(TurnOutcomeError):
    """A turn was finalized twice.

    There is exactly one terminal finalizer. A second finalize means two
    code paths each believe they own the end of the turn, which is how a
    turn gets two different recorded results.
    """


class OutcomeStatus(str, enum.Enum):
    """What actually happened, in the only vocabulary the runtime uses."""

    #: The requested effect occurred and the person got the answer.
    SUCCEEDED = "succeeded"
    #: The person was served, but something was given up on the way: a
    #: fallback lane, a degraded verification grade, a partial effect. This
    #: is a SUCCESS. It exists so that "we had to work for it" stops being
    #: recorded as "we failed".
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    #: Aura declined deliberately. A refusal is a correct outcome and must
    #: never be counted as a malfunction — conflating them is what makes a
    #: system's health metrics punish its own good judgement.
    REFUSED = "refused"
    #: Failed, but the same request could succeed if tried again.
    RETRYABLE_FAILURE = "retryable_failure"
    #: Failed, and retrying this request will not help.
    TERMINAL_FAILURE = "terminal_failure"
    #: Nobody established what happened. The default, and never an alias
    #: for success.
    UNKNOWN = "unknown"

    @property
    def is_success(self) -> bool:
        return self in (OutcomeStatus.SUCCEEDED, OutcomeStatus.PARTIALLY_SUCCEEDED)

    @property
    def is_failure(self) -> bool:
        return self in (
            OutcomeStatus.RETRYABLE_FAILURE,
            OutcomeStatus.TERMINAL_FAILURE,
        )


class VerificationGrade(str, enum.Enum):
    """How well the claim about this turn is actually supported.

    Ordered. ``ASSERTED`` means a component said so about itself and is the
    weakest thing that can be called evidence at all.
    """

    NONE = "none"
    ASSERTED = "asserted"
    OBSERVED = "observed"
    POSTCONDITION_VERIFIED = "postcondition_verified"
    COUNTERFACTUALLY_VERIFIED = "counterfactually_verified"
    EXTERNALLY_VERIFIED = "externally_verified"

    @property
    def rank(self) -> int:
        return _GRADE_RANK[self]

    # All four comparisons, not just the two that were needed at the time.
    # This is a `str` enum, so any operator left undefined falls through to
    # STRING comparison — and the strings are not in rank order. With only
    # __lt__ and __le__ defined, `POSTCONDITION_VERIFIED >= COUNTERFACTUALLY_
    # VERIFIED` evaluated True because "p" sorts after "c", which silently
    # promoted a mid-tier grade past the bar meant to stop it.
    def __lt__(self, other: object) -> bool:  # noqa: D105 - ordering only
        if not isinstance(other, VerificationGrade):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, VerificationGrade):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, VerificationGrade):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, VerificationGrade):
            return NotImplemented
        return self.rank >= other.rank


_GRADE_RANK: dict[VerificationGrade, int] = {
    VerificationGrade.NONE: 0,
    VerificationGrade.ASSERTED: 1,
    VerificationGrade.OBSERVED: 2,
    VerificationGrade.POSTCONDITION_VERIFIED: 3,
    VerificationGrade.COUNTERFACTUALLY_VERIFIED: 4,
    VerificationGrade.EXTERNALLY_VERIFIED: 5,
}


class UserVisibleState(str, enum.Enum):
    """What the PERSON experienced. Distinct from what the system did.

    A turn can be an internal success the person never saw (delivery
    failed) and an internal failure the person experienced as a fine answer
    (fallback covered it). Recording only the internal view is how "the
    user received the text" got conflated with "the function returned".
    """

    NOT_YET_SERVED = "not_yet_served"
    ANSWER_SERVED = "answer_served"
    PARTIAL_ANSWER_SERVED = "partial_answer_served"
    REFUSAL_SERVED = "refusal_served"
    APOLOGY_SERVED = "apology_served"
    NOTHING_SERVED = "nothing_served"


#: Candidates and receipts are bounded: a turn that loops must not grow the
#: ledger without limit. Old entries are dropped from the FRONT, so the
#: most recent (and the served one) always survive.
_MAX_CANDIDATES = 32
_MAX_RECEIPTS = 64
#: Candidate text kept for recovery. Long enough for any real reply,
#: bounded so a runaway generation cannot pin memory per live turn.
_MAX_CANDIDATE_CHARS = 64_000


def _clip(text: Any, limit: int = _MAX_CANDIDATE_CHARS) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[:limit]


@dataclass(frozen=True)
class CandidateSuppressed:
    """The record of a gate rejecting a candidate.

    Kept alongside the candidate rather than replacing it. The suppression
    is a fact ABOUT the text, not a reason to lose the text.
    """

    gate: str
    reasons: tuple[str, ...]
    at: float
    hard: bool = False
    #: False only when the text genuinely must never be shown — a prompt
    #: leak, corrupted output, a policy refusal. A quality or formatting
    #: judgement is NOT unrecoverable, and this defaults accordingly.
    recoverable: bool = True


@dataclass
class Candidate:
    """An answer that existed at some point during the turn."""

    candidate_id: str
    text: str
    source: str
    created_at: float
    verification: VerificationGrade = VerificationGrade.NONE
    fallback: bool = False
    suppressed: CandidateSuppressed | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        """Not suppressed by any gate."""
        return self.suppressed is None

    @property
    def is_recoverable(self) -> bool:
        """Servable if nothing better exists.

        A live candidate is recoverable. A suppressed one is recoverable
        unless the gate marked it unrecoverable — which is reserved for
        text that must never reach a person, not text a heuristic disliked.
        """
        if self.suppressed is None:
            return bool(self.text.strip())
        return bool(self.text.strip()) and self.suppressed.recoverable

    def redacted_preview(self, limit: int = 160) -> str:
        preview, _ = redact_text(self.text[:limit])
        return preview


@dataclass(frozen=True)
class EffectClaim:
    """A real-world effect: what was asked for, and what was seen.

    ``observed`` is deliberately separate from ``requested``. "The tool
    returned" fills in neither.
    """

    name: str
    requested: str
    observed: str | None = None
    verification: VerificationGrade = VerificationGrade.NONE
    at: float = 0.0

    @property
    def is_confirmed(self) -> bool:
        """The effect was independently observed, not merely requested.

        ``ASSERTED`` does not confirm an effect: a component reporting its
        own success is the claim under test, not evidence for it.
        """
        return (
            self.observed is not None
            and self.verification >= VerificationGrade.OBSERVED
        )


class TurnOutcome:
    """The single mutable record of one turn, finalized exactly once.

    Thread-safe: phases run across threads and the event loop, and a
    ledger that races loses precisely the candidate a turn needed.

    Typical shape::

        outcome = TurnOutcome(turn_id, origin="user_chat")
        outcome.declare_effect("reply", "answer the question")
        cid = outcome.record_candidate(draft, source="cortex")
        # a gate dislikes it — annotate, never destroy
        outcome.suppress_candidate(cid, gate="reliability", reasons=("truncated_tail",))
        ...
        receipt = outcome.finalize()
        if receipt.status.is_success:
            serve(receipt.served_answer)
    """

    __slots__ = (
        "turn_id",
        "origin",
        "started_at",
        "_lock",
        "_candidates",
        "_effects",
        "_receipts",
        "_fallbacks",
        "_refusal",
        "_served",
        "_served_candidate_id",
        "_user_visible",
        "_finalized",
        "_receipt",
        "_terminal_errors",
        "_retryable_errors",
        "_dropped_candidates",
        "_dropped_receipts",
    )

    def __init__(self, turn_id: str | None = None, *, origin: str = "unknown") -> None:
        self.turn_id = str(turn_id or uuid.uuid4().hex)
        self.origin = str(origin or "unknown")
        self.started_at = time.time()
        self._lock = checked_lock("turn_outcome", reentrant=True)
        self._candidates: list[Candidate] = []
        self._effects: dict[str, EffectClaim] = {}
        self._receipts: list[dict[str, Any]] = []
        self._fallbacks: list[dict[str, Any]] = []
        self._refusal: dict[str, Any] | None = None
        self._served: str | None = None
        self._served_candidate_id: str | None = None
        self._user_visible = UserVisibleState.NOT_YET_SERVED
        self._finalized = False
        self._receipt: TurnReceipt | None = None
        self._terminal_errors: list[str] = []
        self._retryable_errors: list[str] = []
        self._dropped_candidates = 0
        self._dropped_receipts = 0

    # ---------------------------------------------------------------- candidates

    def record_candidate(
        self,
        text: Any,
        *,
        source: str,
        verification: VerificationGrade = VerificationGrade.NONE,
        fallback: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register an answer that exists. Returns its id.

        Call this the moment a draft exists, BEFORE any gate sees it. That
        ordering is the entire protection: a candidate the ledger never
        heard about is a candidate a gate can still make disappear.
        """
        candidate = Candidate(
            candidate_id=uuid.uuid4().hex,
            text=_clip(text),
            source=str(source or "unknown"),
            created_at=time.time(),
            verification=verification,
            fallback=bool(fallback),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._refuse_if_finalized("record_candidate")
            self._candidates.append(candidate)
            while len(self._candidates) > _MAX_CANDIDATES:
                dropped = self._candidates.pop(0)
                self._dropped_candidates += 1
                if dropped.candidate_id == self._served_candidate_id:
                    # The served text is held separately, so serving is
                    # unaffected; only its ledger entry aged out.
                    self._served_candidate_id = None
        return candidate.candidate_id

    def suppress_candidate(
        self,
        candidate_id: str,
        *,
        gate: str,
        reasons: Sequence[str] | None = None,
        hard: bool = False,
        recoverable: bool = True,
    ) -> bool:
        """Annotate a candidate as rejected by a gate. Never deletes it.

        ``recoverable=False`` is for text that must not reach a person
        under any circumstance — a prompt leak, corrupted output, a policy
        violation. Everything else stays recoverable, because a heuristic
        that dislikes a reply is not proof the reply was bad, and the
        alternative is what Aura did before: die holding a good answer.

        Returns False when the id is unknown, which means a gate is
        rejecting something that was never recorded.
        """
        note = CandidateSuppressed(
            gate=str(gate or "unknown"),
            reasons=tuple(str(r) for r in (reasons or ())),
            at=time.time(),
            hard=bool(hard),
            recoverable=bool(recoverable),
        )
        with self._lock:
            self._refuse_if_finalized("suppress_candidate")
            for candidate in self._candidates:
                if candidate.candidate_id == candidate_id:
                    candidate.suppressed = note
                    return True
        return False

    def candidates(self) -> tuple[Candidate, ...]:
        with self._lock:
            return tuple(self._candidates)

    def best_recoverable_candidate(self) -> Candidate | None:
        """The best answer still available, whatever the gates concluded.

        Preference order, strongest first:

        1. live (un-suppressed) over suppressed — a gate's objection is
           weak evidence, not no evidence;
        2. higher verification grade;
        3. non-fallback over fallback;
        4. most recent.

        Returns None only when the turn genuinely holds nothing servable.
        """
        with self._lock:
            usable = [c for c in self._candidates if c.is_recoverable]
        if not usable:
            return None
        return max(
            usable,
            key=lambda c: (
                c.is_live,
                c.verification.rank,
                not c.fallback,
                c.created_at,
            ),
        )

    # ------------------------------------------------------------------- effects

    def declare_effect(self, name: str, requested: str) -> None:
        """State what this turn is supposed to make true in the world.

        Declaring an effect raises the bar: the turn can no longer be
        SUCCEEDED on the strength of having produced text. Something must
        have observed the effect.
        """
        key = str(name or "effect")
        with self._lock:
            self._refuse_if_finalized("declare_effect")
            existing = self._effects.get(key)
            self._effects[key] = EffectClaim(
                name=key,
                requested=str(requested or ""),
                observed=existing.observed if existing else None,
                verification=existing.verification if existing else VerificationGrade.NONE,
                at=time.time(),
            )

    def observe_effect(
        self,
        name: str,
        observed: str,
        *,
        verification: VerificationGrade = VerificationGrade.OBSERVED,
    ) -> None:
        """Record what was actually seen to happen.

        Pass the grade honestly. ``ASSERTED`` here means "the component
        told us", and this treats that as unconfirmed on purpose.
        """
        key = str(name or "effect")
        with self._lock:
            self._refuse_if_finalized("observe_effect")
            existing = self._effects.get(key)
            self._effects[key] = EffectClaim(
                name=key,
                requested=existing.requested if existing else "",
                observed=str(observed),
                verification=verification,
                at=time.time(),
            )

    def effects(self) -> tuple[EffectClaim, ...]:
        with self._lock:
            return tuple(self._effects.values())

    # ------------------------------------------------------- degradation & errors

    def record_fallback(self, *, lane: str, reason: str, succeeded: bool) -> None:
        """A preferred path failed and something else was tried.

        A fallback that SUCCEEDS caps the turn at PARTIALLY_SUCCEEDED — it
        is a success, and recording it as a failure is exactly the bug
        that made a working fallback register as internal strain.
        """
        with self._lock:
            self._refuse_if_finalized("record_fallback")
            self._fallbacks.append(
                {
                    "lane": str(lane or "unknown"),
                    "reason": str(reason or ""),
                    "succeeded": bool(succeeded),
                    "at": time.time(),
                }
            )

    def record_refusal(self, *, reason: str, authority: str) -> None:
        """Aura declined on purpose. Not a malfunction."""
        with self._lock:
            self._refuse_if_finalized("record_refusal")
            self._refusal = {
                "reason": str(reason or ""),
                "authority": str(authority or "unknown"),
                "at": time.time(),
            }

    def record_error(self, detail: str, *, retryable: bool) -> None:
        """A real failure. Retryable and terminal are graded separately.

        Recording an error does NOT decide the turn. If a fallback later
        serves the person, this is history, not the verdict.
        """
        text, _ = redact_text(str(detail or ""))
        with self._lock:
            self._refuse_if_finalized("record_error")
            (self._retryable_errors if retryable else self._terminal_errors).append(text)

    def record_receipt(self, kind: str, payload: Mapping[str, Any] | None = None) -> None:
        """Attach a causal receipt: evidence of what this turn did and why."""
        redacted, _ = redact_structure(dict(payload or {}))
        with self._lock:
            self._refuse_if_finalized("record_receipt")
            self._receipts.append(
                {"kind": str(kind or "receipt"), "at": time.time(), "payload": redacted}
            )
            while len(self._receipts) > _MAX_RECEIPTS:
                self._receipts.pop(0)
                self._dropped_receipts += 1

    def receipts(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(self._receipts)

    # -------------------------------------------------------------------- serving

    def mark_served(
        self,
        text: Any,
        *,
        candidate_id: str | None = None,
        state: UserVisibleState = UserVisibleState.ANSWER_SERVED,
    ) -> None:
        """Record what the PERSON actually received.

        Until this is called the person got nothing, whatever the internal
        machinery believes about its own success.
        """
        with self._lock:
            self._refuse_if_finalized("mark_served")
            self._served = _clip(text)
            self._served_candidate_id = candidate_id
            self._user_visible = state

    # ------------------------------------------------------------------ finalizer

    def finalize(self, *, subsystem: str = "turn_outcome") -> "TurnReceipt":
        """THE terminal finalizer. Computes status from evidence. Runs once.

        Not "asks the caller how it went". The whole failure this module
        exists for came from components grading their own turns at the
        moment they noticed trouble, before the fallback that fixed it.
        """
        with self._lock:
            if self._finalized:
                raise AlreadyFinalized(
                    f"turn {self.turn_id} was already finalized; there is exactly "
                    "one terminal finalizer per turn"
                )
            status, rationale = self._compute_status()
            grade = self._effective_grade()
            receipt = TurnReceipt(
                turn_id=self.turn_id,
                origin=self.origin,
                status=status,
                rationale=rationale,
                started_at=self.started_at,
                finished_at=time.time(),
                requested_effects=tuple(
                    e.requested for e in self._effects.values() if e.requested
                ),
                observed_effects=tuple(
                    e.observed for e in self._effects.values() if e.observed
                ),
                answer_candidate=(
                    self.best_recoverable_candidate().text
                    if self.best_recoverable_candidate()
                    else None
                ),
                served_answer=self._served,
                fallback_used=any(f["succeeded"] for f in self._fallbacks),
                verification_grade=grade,
                user_visible_state=self._user_visible,
                causal_receipts=tuple(self._receipts),
                suppressed_candidates=tuple(
                    {
                        "source": c.source,
                        "gate": c.suppressed.gate,
                        "reasons": list(c.suppressed.reasons),
                        "recoverable": c.suppressed.recoverable,
                        "preview": c.redacted_preview(),
                    }
                    for c in self._candidates
                    if c.suppressed is not None
                ),
                errors=tuple(self._terminal_errors + self._retryable_errors),
                dropped_candidates=self._dropped_candidates,
            )
            self._receipt = receipt
            self._finalized = True

        _report(receipt, subsystem=subsystem)
        return receipt

    @property
    def is_finalized(self) -> bool:
        with self._lock:
            return self._finalized

    @property
    def receipt(self) -> "TurnReceipt | None":
        with self._lock:
            return self._receipt

    # ------------------------------------------------------------------ internals

    def _refuse_if_finalized(self, action: str) -> None:
        if self._finalized:
            raise AlreadyFinalized(
                f"turn {self.turn_id} is finalized; {action} would change a "
                "result that has already been reported"
            )

    def _effective_grade(self) -> VerificationGrade:
        """The grade of what was SERVED, not the best grade seen anywhere.

        A turn that verified a draft it then discarded has not verified its
        answer, and claiming the higher grade would be the same class of
        error this module exists to stop.
        """
        served_candidate = None
        if self._served_candidate_id:
            served_candidate = next(
                (c for c in self._candidates if c.candidate_id == self._served_candidate_id),
                None,
            )
        if served_candidate is not None:
            grade = served_candidate.verification
        elif self._served is not None:
            grade = next(
                (c.verification for c in self._candidates if c.text == self._served),
                VerificationGrade.NONE,
            )
        else:
            grade = VerificationGrade.NONE
        effect_grades = [e.verification for e in self._effects.values() if e.observed]
        if effect_grades:
            grade = max([grade, *effect_grades], key=lambda g: g.rank)
        return grade

    def _compute_status(self) -> tuple[OutcomeStatus, str]:
        """Derive the verdict. Order matters and encodes the policy."""
        # A deliberate refusal is a correct outcome and outranks the
        # machinery's opinion of itself.
        if self._refusal is not None:
            return OutcomeStatus.REFUSED, f"refused:{self._refusal['reason']}"

        declared = [e for e in self._effects.values() if e.requested]
        confirmed = [e for e in declared if e.is_confirmed]
        served = bool((self._served or "").strip())
        fallback_succeeded = any(f["succeeded"] for f in self._fallbacks)
        degraded = fallback_succeeded or self._user_visible in (
            UserVisibleState.PARTIAL_ANSWER_SERVED,
        )

        if declared:
            # A requested effect was declared, so text alone cannot pass.
            if confirmed and len(confirmed) == len(declared):
                if served or self._user_visible is not UserVisibleState.NOTHING_SERVED:
                    return (
                        (OutcomeStatus.PARTIALLY_SUCCEEDED, "effects_confirmed_with_degradation")
                        if degraded
                        else (OutcomeStatus.SUCCEEDED, "effects_confirmed")
                    )
                return (
                    OutcomeStatus.PARTIALLY_SUCCEEDED,
                    "effects_confirmed_but_person_was_not_served",
                )
            if confirmed:
                return (
                    OutcomeStatus.PARTIALLY_SUCCEEDED,
                    f"effects_confirmed:{len(confirmed)}/{len(declared)}",
                )
            # Nothing observed. "The tool returned" is not success.
            if self._terminal_errors:
                return OutcomeStatus.TERMINAL_FAILURE, "requested_effect_never_observed"
            if self._retryable_errors:
                return OutcomeStatus.RETRYABLE_FAILURE, "requested_effect_never_observed"
            return OutcomeStatus.UNKNOWN, "requested_effect_declared_but_never_observed"

        if served:
            return (
                (OutcomeStatus.PARTIALLY_SUCCEEDED, "served_after_fallback")
                if degraded
                else (OutcomeStatus.SUCCEEDED, "served")
            )

        # Nothing served. Now — and only now, after fallbacks have had
        # their chance — errors decide severity.
        if self._terminal_errors:
            return OutcomeStatus.TERMINAL_FAILURE, "terminal_error_and_nothing_served"
        if self._retryable_errors:
            return OutcomeStatus.RETRYABLE_FAILURE, "retryable_error_and_nothing_served"
        if any(c.is_recoverable for c in self._candidates):
            # The turn holds an answer it never served. This is the exact
            # shape of the live defect, and it is named rather than being
            # laundered into a generic infrastructure failure.
            return (
                OutcomeStatus.RETRYABLE_FAILURE,
                "answer_available_but_never_served",
            )
        return OutcomeStatus.UNKNOWN, "nothing_recorded"


@dataclass(frozen=True)
class TurnReceipt:
    """The immutable result of exactly one turn."""

    turn_id: str
    origin: str
    status: OutcomeStatus
    rationale: str
    started_at: float
    finished_at: float
    requested_effects: tuple[str, ...]
    observed_effects: tuple[str, ...]
    answer_candidate: str | None
    served_answer: str | None
    fallback_used: bool
    verification_grade: VerificationGrade
    user_visible_state: UserVisibleState
    causal_receipts: tuple[dict[str, Any], ...]
    suppressed_candidates: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]
    dropped_candidates: int = 0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def held_an_unserved_answer(self) -> bool:
        """The turn had something to say and said none of it.

        The signal worth alerting on: not "did something break", but "did a
        person go without an answer that existed".
        """
        return bool(self.answer_candidate) and not (self.served_answer or "").strip()

    def to_dict(self) -> dict[str, Any]:
        """Telemetry-safe view. Answer TEXT is never included.

        Lengths and previews only: this record travels to logs and health
        reports, and a turn's full reply is the person's content.
        """
        return {
            "turn_id": self.turn_id,
            "origin": self.origin,
            "status": self.status.value,
            "rationale": self.rationale,
            "duration_s": round(self.duration_s, 4),
            "requested_effects": list(self.requested_effects),
            "observed_effects": list(self.observed_effects),
            "answer_candidate_chars": len(self.answer_candidate or ""),
            "served_answer_chars": len(self.served_answer or ""),
            "held_an_unserved_answer": self.held_an_unserved_answer,
            "fallback_used": self.fallback_used,
            "verification_grade": self.verification_grade.value,
            "user_visible_state": self.user_visible_state.value,
            "suppressed_candidates": [
                {k: v for k, v in entry.items() if k != "preview"}
                for entry in self.suppressed_candidates
            ],
            "receipt_kinds": [r["kind"] for r in self.causal_receipts],
            "error_count": len(self.errors),
            "dropped_candidates": self.dropped_candidates,
        }


def _report(receipt: TurnReceipt, *, subsystem: str) -> None:
    """Emit the outcome once, at the severity the EVIDENCE supports.

    Deliberately quiet for successes and refusals. A fallback that worked
    is not an incident, and a refusal is not a defect; recording either as
    one is how health reports learned to cry wolf.
    """
    if receipt.status.is_success or receipt.status is OutcomeStatus.REFUSED:
        return

    if receipt.held_an_unserved_answer:
        severity = "critical"
        action = (
            "turn ended holding a servable answer that was never shown to the "
            "person; the gate that suppressed it is named in the receipt"
        )
    elif receipt.status is OutcomeStatus.TERMINAL_FAILURE:
        severity = "degraded"
        action = "turn failed terminally after fallbacks were exhausted"
    elif receipt.status is OutcomeStatus.RETRYABLE_FAILURE:
        severity = "warning"
        action = "turn failed in a way the same request could survive on retry"
    else:
        severity = "warning"
        action = "turn ended without establishing what happened"

    record_degradation(
        subsystem,
        TurnOutcomeError(f"{receipt.status.value}:{receipt.rationale}"),
        severity=severity,
        action=action,
        extra=receipt.to_dict(),
    )


def finalize_turn(
    outcome: TurnOutcome | None, *, subsystem: str = "turn_outcome"
) -> TurnReceipt | None:
    """Finalize once, tolerating a turn that another path already ended.

    For ``finally:`` blocks, where the turn may or may not have been
    finalized on the success path and raising there would replace a real
    error with a bookkeeping one.
    """
    if outcome is None:
        return None
    try:
        return outcome.finalize(subsystem=subsystem)
    except AlreadyFinalized:
        return outcome.receipt


def outcome_from_candidates(
    outcome: TurnOutcome,
    *,
    fallback_text: str | None = None,
) -> str | None:
    """The text to actually serve: best surviving candidate, else fallback.

    The recovery seam. When a gate has rejected everything, this asks the
    ledger whether a recoverable answer still exists before anyone reaches
    for an apology.
    """
    best = outcome.best_recoverable_candidate()
    if best is not None and best.text.strip():
        return best.text
    return fallback_text


# ---------------------------------------------------------------------------
# The bound turn.
#
# Aura's response path is thousands of lines across a dozen modules, and the
# gates that suppress candidates sit deep inside it. Threading an outcome
# object through every signature would be a refactor nobody finishes, and a
# half-finished one leaves exactly the gaps the ledger exists to close.
#
# So the turn binds to the context instead. A contextvar, not a global: it
# follows the async task and does not leak between concurrent turns, which a
# module-level singleton would do the moment two people talk at once.
# ---------------------------------------------------------------------------

_CURRENT_TURN: contextvars.ContextVar[TurnOutcome | None] = contextvars.ContextVar(
    "aura_current_turn_outcome", default=None
)


def current_turn() -> TurnOutcome | None:
    """The turn bound to this context, or None outside one.

    None is normal and never an error: background work, tests and tools run
    with no turn, and every recording helper degrades to a no-op there.
    """
    return _CURRENT_TURN.get()


@contextlib.contextmanager
def bind_turn(outcome: TurnOutcome) -> Iterator[TurnOutcome]:
    """Bind a turn for the duration of a block, restoring the previous one."""
    token = _CURRENT_TURN.set(outcome)
    try:
        yield outcome
    finally:
        _CURRENT_TURN.reset(token)


def note_candidate(
    text: Any,
    *,
    source: str,
    verification: VerificationGrade = VerificationGrade.NONE,
    fallback: bool = False,
) -> str | None:
    """Record a candidate on the bound turn if there is one.

    Safe to call anywhere. Returns the candidate id, or None when no turn
    is bound — never raises, because a ledger that can break a response
    path is worse than no ledger.
    """
    outcome = current_turn()
    if outcome is None:
        return None
    try:
        return outcome.record_candidate(
            text, source=source, verification=verification, fallback=fallback
        )
    except (AlreadyFinalized, TypeError, ValueError) as exc:
        record_degradation(
            "turn_outcome",
            exc,
            severity="warning",
            action="skipped one candidate ledger write; the response path is unaffected",
        )
        return None


def note_suppression(
    candidate_id: str | None,
    *,
    gate: str,
    reasons: Sequence[str] | None = None,
    hard: bool = False,
    recoverable: bool = True,
) -> None:
    """Record a gate's rejection on the bound turn. Never raises."""
    if not candidate_id:
        return
    outcome = current_turn()
    if outcome is None:
        return
    try:
        outcome.suppress_candidate(
            candidate_id,
            gate=gate,
            reasons=reasons,
            hard=hard,
            recoverable=recoverable,
        )
    except (AlreadyFinalized, TypeError, ValueError) as exc:
        record_degradation(
            "turn_outcome",
            exc,
            severity="warning",
            action="skipped one suppression ledger write; the response path is unaffected",
        )


def recoverable_answer() -> str | None:
    """The best answer the bound turn still holds, if any.

    Call this before serving an apology. If it returns text, the turn was
    about to tell a person it had nothing while holding something.
    """
    outcome = current_turn()
    if outcome is None:
        return None
    best = outcome.best_recoverable_candidate()
    if best is None:
        return None
    text = best.text.strip()
    return text or None


def merge_reasons(reasons: Iterable[Any]) -> tuple[str, ...]:
    """Normalize gate reasons to a stable, de-duplicated tuple."""
    seen: dict[str, None] = {}
    for reason in reasons or ():
        text = str(reason or "").strip()
        if text:
            seen.setdefault(text, None)
    return tuple(seen)
