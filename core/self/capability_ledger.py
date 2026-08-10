"""One object per capability, owning both halves: what is true, and why.

WHY THIS EXISTS
───────────────
Live, 2026-08-10, within one afternoon:

    "I don't have a camera and there's no part that stops me from doing
     something I can't do."      — while ``_apply_camera_control`` sat in the
                                   same request handler that produced the reply
    "I cannot execute code."     — with the code_repl skill READY
    "Current energy and focus numbers: Not readable."
                                 — while the header rendered 37.6 / 94%
    "I have no memory of it."    — with 34 turns of it on disk
    "It is a blank slate."       — with three windows open

One defect wearing five costumes: **saying and doing were separate objects.**
The executors were real and reachable. What reached the model instead was a
parallel *description* of them, and only on turns where a regex on the user's
question predicted the description would be needed. A question nobody
anticipated got no evidence, so the answer came from the language model's
priors about what an AI is, and the priors say it has no body.

``core/skills/capability_map.py`` shows how far that goes. Its ``Capability``
has a ``handler`` slot and an ``is_online`` flag — both halves, by design. No
registration has ever passed a handler, and ``is_online`` is decided by
string-matching capability names against skill names in a hardcoded table. It
is a registry that can describe what it cannot run and does not know the state
of, and ``detect_intent`` silently skips everything it believes offline.

WHAT THIS DOES DIFFERENTLY
──────────────────────────
A capability's availability answer is produced by **running the precondition
its executor runs**. ``camera`` is not described as available; it is asked, via
the same ``camera_enabled()`` / ``sight_dependency_gap()`` that ``sight.look()``
itself consults before opening a lens. Saying and doing cannot disagree,
because there is one function and both call it.

Two facts, never merged, because merging them is what made the denials false:

``present``     she has the thing at all
``usable_now``  she could use it this second

A switched-off camera is ``present and not usable_now``. "I don't have a
camera" is false about it; "the camera is off" is true. Collapsing those into
one boolean is exactly how a togglable device became a missing organ.

The claim check runs on HER OUTPUT, not on the user's question. That inversion
is the point: questions are unbounded and unpredictable, so any input-side
regex will always have a next gap. Claims she makes about herself are finite,
appear in text this module can read, and every one of them names a subject that
can be probed.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Self.CapabilityLedger")

_PROBE_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class Availability:
    """What is actually true about one capability, right now."""

    name: str
    present: bool
    usable_now: bool
    summary: str
    blocker: str = ""
    remedy: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    #: False when the probe could not establish the truth at all.
    #:
    #: This exists so the ledger cannot commit the inverse of the sin it was
    #: written to fix. A probe that cannot read a permission has NOT observed
    #: its absence, and a ledger that treats "cannot tell" as "unavailable"
    #: would start contradicting true statements with confident false ones —
    #: the same failure, pointed the other way. Nothing unknown is ever used
    #: to correct her.
    known: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "present": self.present,
            "usable_now": self.usable_now,
            "summary": self.summary,
            "blocker": self.blocker,
            "remedy": self.remedy,
            "known": self.known,
            "evidence": dict(self.evidence),
        }

    def as_evidence_line(self) -> str:
        """One line she can answer from, stating the truth and its reason."""
        parts = [self.summary]
        if self.blocker:
            parts.append(f"What stands in the way: {self.blocker}.")
        if self.remedy:
            parts.append(f"What would clear it: {self.remedy}.")
        return " ".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class LiveCapability:
    """A capability that answers for itself.

    ``subjects`` are the words a person uses for the thing. They are used only
    to notice that a sentence is ABOUT this capability; whether the sentence is
    true is decided by ``probe``, never by the words.
    """

    name: str
    subjects: tuple[str, ...]
    probe: Callable[[], Availability]

    def measure(self) -> Availability:
        try:
            return self.probe()
        except _PROBE_ERRORS as exc:
            logger.debug("Capability probe %s failed: %s", self.name, exc)
            return Availability(
                name=self.name,
                present=False,
                usable_now=False,
                summary=f"I could not read the state of {self.name}.",
                blocker=f"the probe itself failed with {type(exc).__name__}",
                evidence={"probe_error": str(exc)},
            )


@dataclass(frozen=True, slots=True)
class ContradictedClaim:
    """A sentence of hers that the runtime disagrees with."""

    sentence: str
    availability: Availability
    denied: str  # "possession" or "ability"

    def correction(self) -> str:
        return self.availability.as_evidence_line()


# A denial of self, in the shapes people actually write them. This decides only
# that a sentence is a DENIAL — never whether the denial is right. The probe
# decides that.
_DENIAL_FRAME = re.compile(
    r"\b(?:"
    r"i\s+do\s*n[o']?t\s+have"
    r"|i\s+do\s+not\s+have"
    r"|i\s+have\s+no\b"
    r"|i\s+(?:can'?t|cannot|can\s+not)\b"
    r"|i\s+(?:am\s+not|'m\s+not)\s+able\b"
    r"|i\s+(?:am|'m)\s+unable\b"
    r"|i\s+lack\b"
    r"|there\s+(?:is|'s)\s+no\b"
    r"|i\s+have\s+n[o']?t\s+got\b"
    r"|no\s+(?:access\s+to|way\s+(?:for\s+me\s+)?to)\b"
    # Impersonal reports of absence. She does not always say "I cannot" —
    # live, the whole answer to "tell me your current energy and focus
    # numbers" was "Current energy and focus numbers: Not readable." while
    # the header beside it rendered them. A denial with the pronoun removed
    # is still a denial.
    r"|not\s+readable\b|unreadable\b"
    r"|not\s+available\b|unavailable\b"
    r"|no\s+reading\b|cannot\s+be\s+read\b|can'?t\s+be\s+read\b"
    r")",
    re.IGNORECASE,
)

# Denials of HAVING the thing, as opposed to being able to use it. "I don't
# have a camera" is a claim about possession; "I can't see right now" is a
# claim about readiness. They are checked against different facts.
_POSSESSION_FRAME = re.compile(
    r"\b(?:"
    r"i\s+do\s*n[o']?t\s+have|i\s+do\s+not\s+have|i\s+have\s+no\b"
    r"|i\s+lack\b|there\s+(?:is|'s)\s+no\b|i\s+have\s+n[o']?t\s+got\b"
    r")",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


class CapabilityLedger:
    """Every capability that can answer for itself, in one place."""

    def __init__(self) -> None:
        self._capabilities: dict[str, LiveCapability] = {}

    def register(self, capability: LiveCapability) -> None:
        self._capabilities[capability.name] = capability

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def get(self, name: str) -> LiveCapability | None:
        return self._capabilities.get(name)

    def measure(self, name: str) -> Availability | None:
        capability = self._capabilities.get(name)
        return capability.measure() if capability else None

    def measure_all(self) -> dict[str, Availability]:
        return {name: cap.measure() for name, cap in self._capabilities.items()}

    def capabilities_named_in(self, text: str) -> list[LiveCapability]:
        """Which capabilities a piece of text is talking about."""
        lowered = str(text or "").lower()
        if not lowered.strip():
            return []
        found: list[LiveCapability] = []
        for capability in self._capabilities.values():
            for subject in capability.subjects:
                if re.search(rf"\b{re.escape(subject)}\b", lowered):
                    found.append(capability)
                    break
        return found

    def contradicted_claims(self, reply: str) -> list[ContradictedClaim]:
        """Denials in ``reply`` that the runtime says are false.

        Only denials are checked. An overclaim ("I already emailed them") is a
        different failure with a different remedy — this one exists because she
        talks herself out of things she can do.
        """
        contradictions: list[ContradictedClaim] = []
        for sentence in _SENTENCE_SPLIT.split(str(reply or "")):
            sentence = sentence.strip()
            if not sentence or not _DENIAL_FRAME.search(sentence):
                continue
            denies_possession = bool(_POSSESSION_FRAME.search(sentence))
            for capability in self.capabilities_named_in(sentence):
                availability = capability.measure()
                if not availability.known:
                    # Not measured is not measured. Saying nothing here is the
                    # whole reason this ledger can be trusted to speak at all.
                    continue
                if denies_possession and availability.present:
                    contradictions.append(
                        ContradictedClaim(sentence, availability, "possession")
                    )
                elif not denies_possession and availability.usable_now:
                    contradictions.append(
                        ContradictedClaim(sentence, availability, "ability")
                    )
        return contradictions


# ── Probes ──────────────────────────────────────────────────────────────────
# Each one calls what the executor calls. None of them describes anything.


def _probe_camera() -> Availability:
    from core.senses.sight import camera_enabled, sight_dependency_gap

    gap = sight_dependency_gap()
    enabled = camera_enabled()
    # A camera the host has and the user switched off is PRESENT. Reporting it
    # as absent is the specific falsehood this ledger exists to stop.
    present = not gap
    return Availability(
        name="camera",
        present=present,
        usable_now=bool(present and enabled),
        summary=(
            "I have a camera and it is on right now."
            if present and enabled
            else "I have a camera; it is switched off at the moment, and I can "
            "switch it on when you ask."
            if present
            else "I have no working vision runtime on this machine."
        ),
        blocker=(
            "" if present and enabled else ("the camera is switched off" if present else gap)
        ),
        remedy=(
            ""
            if present and enabled
            else ("ask me to turn the camera on" if present else "install the missing runtime")
        ),
        evidence={"camera_enabled": enabled, "dependency_gap": gap},
    )


def _probe_screen_sight() -> Availability:
    """Screen readability, from the grant the capture path itself consults.

    The resident Aura.app bridge is the production authority for this grant.
    When its cache holds no entry the answer is genuinely unknown — a
    conclusion this probe reports rather than rounding down to "denied". She
    read three window titles correctly on a turn where an earlier draft of this
    probe would have told her she could not see.
    """
    entry: Any = None
    detail: dict[str, Any] = {}
    try:
        from core.security.permission_guard import PermissionType, get_permission_guard

        cache = getattr(get_permission_guard(), "_cache", {}) or {}
        entry = cache.get(PermissionType.SCREEN)
    except _PROBE_ERRORS as exc:
        detail = {"permission_probe_error": str(exc)}

    if not isinstance(entry, dict):
        return Availability(
            name="screen_sight",
            present=True,
            usable_now=False,
            known=False,
            summary="I could not read whether screen capture is granted.",
            evidence=detail or {"screen_permission": "unmeasured"},
        )

    granted = bool(entry.get("granted"))
    return Availability(
        name="screen_sight",
        present=True,
        usable_now=granted,
        summary=(
            "I can capture and read this screen."
            if granted
            else "I can read the screen once macOS screen recording is granted."
        ),
        blocker="" if granted else "the macOS screen-recording permission is not granted",
        remedy="" if granted else "grant Screen Recording to Aura in System Settings",
        evidence={"screen_permission": granted, "status": entry.get("status", "")},
    )


def _probe_code_execution() -> Availability:
    import importlib.util

    installed = importlib.util.find_spec("core.skills.code_repl") is not None
    return Availability(
        name="code_execution",
        present=installed,
        usable_now=installed,
        summary=(
            "I can run code and report what it actually printed."
            if installed
            else "I have no code execution skill on this build."
        ),
        blocker="" if installed else "the code_repl skill is not installed",
        evidence={"code_repl_installed": installed},
    )


def _probe_conversation_memory() -> Availability:
    turns = 0
    try:
        from core.conversation.persistence import get_persistence

        sessions = get_persistence().get_recent_sessions(limit=3, with_turns_only=True)
        turns = sum(int(session.get("turn_count") or 0) for session in sessions)
    except _PROBE_ERRORS as exc:
        return Availability(
            name="conversation_memory",
            present=False,
            usable_now=False,
            summary="I could not read my conversation store.",
            blocker=f"{type(exc).__name__}: {exc}",
            evidence={"error": str(exc)},
        )
    return Availability(
        name="conversation_memory",
        present=turns > 0,
        usable_now=turns > 0,
        summary=(
            f"I have {turns} stored turns of recent conversation I can read back."
            if turns
            else "I have no stored conversation yet."
        ),
        blocker="" if turns else "nothing has been recorded yet",
        evidence={"recent_turns": turns},
    )


def _probe_interoception() -> Availability:
    reading: dict[str, Any] = {}
    try:
        from core.being.body_state_service import BodyStateService

        snapshot = BodyStateService.get().snapshot()
        # The fields BodyHealthSnapshot actually carries. An earlier draft
        # asked for "energy" and "focus" — words from the question rather than
        # from the instrument — got nothing back, and concluded she could not
        # read herself. That is the same mistake as the one being fixed: an
        # unread instrument reported as an absent one.
        for label in (
            "operational_health",
            "fatigue",
            "total_pressure",
            "cpu_pressure",
            "memory_pressure",
        ):
            value = getattr(snapshot, label, None)
            if isinstance(value, (int, float)):
                reading[label] = round(float(value), 3)
    except _PROBE_ERRORS as exc:
        return Availability(
            name="interoception",
            present=False,
            usable_now=False,
            summary="I could not read my own vitals.",
            blocker=f"{type(exc).__name__}: {exc}",
            evidence={"error": str(exc)},
        )
    readable = bool(reading)
    return Availability(
        name="interoception",
        present=True,
        usable_now=readable,
        known=readable,
        summary=(
            "I can read my own state right now: "
            + ", ".join(f"{key} {value}" for key, value in reading.items())
            if readable
            else "I could not get a reading off my own instruments this tick."
        ),
        blocker="" if readable else "the body-state snapshot returned no numbers",
        evidence=reading,
    )


def _default_ledger() -> CapabilityLedger:
    ledger = CapabilityLedger()
    ledger.register(
        LiveCapability("camera", ("camera", "webcam", "lens"), _probe_camera)
    )
    ledger.register(
        LiveCapability(
            "screen_sight",
            ("screen", "display", "monitor", "eyes"),
            _probe_screen_sight,
        )
    )
    ledger.register(
        LiveCapability(
            "code_execution",
            ("code", "python", "script", "compute", "calculation"),
            _probe_code_execution,
        )
    )
    ledger.register(
        LiveCapability(
            "conversation_memory",
            ("memory", "remember", "recall", "conversation", "recollection"),
            _probe_conversation_memory,
        )
    )
    ledger.register(
        LiveCapability(
            "interoception",
            ("vitals", "energy", "focus", "body", "sensor", "sensors", "telemetry"),
            _probe_interoception,
        )
    )
    return ledger


_LEDGER: CapabilityLedger | None = None


def get_capability_ledger() -> CapabilityLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = _default_ledger()
    return _LEDGER


def reset_capability_ledger_for_test() -> None:
    global _LEDGER
    _LEDGER = None


def correction_context(claims: Iterable[ContradictedClaim]) -> str:
    """Evidence for a re-answer, built from what the probes measured.

    Not a description of her capabilities — the output of running the same
    checks her executors run, quoted back at the sentence that contradicted
    them.
    """
    claims = list(claims)
    if not claims:
        return ""
    lines = [
        "[You just said something about yourself that your own runtime "
        "contradicts. These are live measurements, not assumptions — each one "
        "is the same check the corresponding action runs before it acts.",
        "",
    ]
    for claim in claims:
        lines.append(f'You said: "{claim.sentence}"')
        lines.append(f"Measured: {claim.correction()}")
        lines.append("")
    lines.append(
        "Answer again from these measurements. If something is present but "
        "switched off or ungranted, say that precisely — say what it would "
        "take — rather than saying you do not have it.]"
    )
    return "\n".join(lines)


__all__ = [
    "Availability",
    "CapabilityLedger",
    "ContradictedClaim",
    "LiveCapability",
    "correction_context",
    "get_capability_ledger",
    "reset_capability_ledger_for_test",
]
