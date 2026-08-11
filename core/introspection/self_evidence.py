"""Readings Aura actually has, and typed absences where she has none.

Asked on the live desktop, 2026-08-10, which of her subsystems were degraded
and whether any job had been failing repeatedly, she answered:

    "I couldn't get to an answer I'd stand behind on that one, and I won't send
    you a thinner one and pass it off as the real thing."

At that moment /api/health carried integrity=degraded, a stale CRSM manifest,
and overt_action_cycle with failures=13 and its exact TypeError. The answer was
structured, live, and hers. Asked instead what was on the screen — a sense that
health reports as granted, bridged and directly probed — she produced "a web
browser interface with multiple tabs", "no applications running in the
foreground" and "nothing displayed except generic desktop wallpaper", three
claims that cannot all be true, because nothing had handed her a reading and
nothing had told her that.

One fault under both. Evidence that exists in the runtime does not reach the
reply, and an absent reading is indistinguishable from an unremarkable one. A
generated answer then fills the space, agreeing with whatever the question
implied — confident where there was nothing, refusing where there was plenty.

So this module does not describe evidence, it fetches it. A Reading is either a
value with provenance or a typed absence naming which kind of nothing it is:

    READ                    a real value, with where it came from and when
    ABSENT_NEVER_SAMPLED    the channel exists and has never produced one
    ABSENT_UNAVAILABLE      the source is present but could not be read now
    ABSENT_NOT_INSTRUMENTED nothing measures this; it is not a failure

Those four are not the same fact, and collapsing them is what produced both
failures above. "Never sampled" is why the camera could not say whether anyone
else was in the room; "not instrumented" is why there was no median latency to
quote. Neither is "no".

The bundle is consumed, not narrated: render_self_health_answer builds an
answer out of the readings themselves, and the absent channels are the input a
verification gate needs to tell an unsupported claim from a supported one.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "EvidenceBundle",
    "Reading",
    "ReadingState",
    "asks_about_own_operational_state",
    "render_self_health_answer",
    "resolve_self_health",
    "self_health_answer",
]

#: Something of hers that can be in a state, paired below with a word for being
#: in a bad one. Both halves are required, so "how are the kids" and "my server
#: is degraded" do not resolve her health, and "is anything failing?" does.
_SELF_SUBJECT_RE = re.compile(
    r"\b(?:your|you'?re|yours|of\s+yours|are\s+you)\b|"
    r"\b(?:subsystem|substrate|runtime|internals?|faculties|organs?|"
    r"heartbeats?|degradations?|telemetry|jobs?|cycles?|loops?)\b",
    re.IGNORECASE,
)
_TROUBLE_RE = re.compile(
    r"\b(?:degraded?|degrading|failing|failed|failures?|broken|breaking|"
    r"unhealthy|down|erroring|errors?|faults?|wrong|off|struggling|"
    r"not\s+working|misbehaving|stuck|wedged|repeatedly)\b",
    re.IGNORECASE,
)
#: Asking about her state at all, even without a trouble word — "how are your
#: subsystems doing", "status of your internals".
_STATE_ENQUIRY_RE = re.compile(
    r"\b(?:status|health|healthy|state|doing|holding\s+up|nominal|ok(?:ay)?)\b",
    re.IGNORECASE,
)


def asks_about_own_operational_state(text: Any) -> bool:
    """True when the turn asks what is wrong with HER, not with something else.

    Deliberately narrow. This decides whether live readings are fetched and
    served, so a false positive answers a question nobody asked with a wall of
    telemetry. "Which of your subsystems is degraded right now?" qualifies;
    "my deploy is failing" does not.
    """

    raw = str(text or "").strip()
    if not raw:
        return False
    if not _SELF_SUBJECT_RE.search(raw):
        return False
    return bool(_TROUBLE_RE.search(raw) or _STATE_ENQUIRY_RE.search(raw))


def self_health_answer(message: Any) -> str:
    """The answer her own telemetry supports, or "" when this is not that turn.

    The whole point of the module in one function: a caller that is about to
    give up can ask whether the runtime already holds the answer. It returns
    text only when a channel actually produced a value, so it can never
    manufacture reassurance.
    """

    if not asks_about_own_operational_state(message):
        return ""
    bundle = resolve_self_health()
    if not bundle.grounded:
        return ""
    return render_self_health_answer(bundle)


class ReadingState(StrEnum):
    READ = "read"
    ABSENT_NEVER_SAMPLED = "absent_never_sampled"
    ABSENT_UNAVAILABLE = "absent_unavailable"
    ABSENT_NOT_INSTRUMENTED = "absent_not_instrumented"


@dataclass(frozen=True, slots=True)
class Reading:
    """One value Aura actually holds, or a named absence where she holds none."""

    channel: str
    state: ReadingState
    value: Any = None
    unit: str = ""
    provenance: str = ""
    detail: str = ""
    at: float = field(default_factory=time.time)

    @property
    def present(self) -> bool:
        return self.state is ReadingState.READ

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "state": str(self.state),
            "value": self.value,
            "unit": self.unit,
            "provenance": self.provenance,
            "detail": self.detail,
            "at": self.at,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Everything consulted for one question, present and absent alike."""

    demand: str
    readings: tuple[Reading, ...]

    @property
    def present(self) -> tuple[Reading, ...]:
        return tuple(r for r in self.readings if r.present)

    @property
    def absent(self) -> tuple[Reading, ...]:
        return tuple(r for r in self.readings if not r.present)

    @property
    def grounded(self) -> bool:
        """True when at least one channel produced a real value."""
        return bool(self.present)

    def to_dict(self) -> dict[str, Any]:
        return {
            "demand": self.demand,
            "grounded": self.grounded,
            "readings": [r.to_dict() for r in self.readings],
        }


def _degradation_readings() -> list[Reading]:
    try:
        from core.runtime.errors import recent_degradations
    except ImportError as exc:
        return [Reading(
            channel="degradations",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="core.runtime.errors",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    try:
        records = recent_degradations(limit=25)
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        return [Reading(
            channel="degradations",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="recent_degradations()",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    return [Reading(
        channel="degradations",
        state=ReadingState.READ,
        value=list(records),
        unit="records",
        provenance="core.runtime.errors.recent_degradations",
        detail=f"{len(records)} recent",
    )]


def _health_readings() -> list[Reading]:
    """Subsystem health and any repeatedly-failing conducted job."""

    try:
        from core.runtime.health_contract import runtime_health_report
    except ImportError as exc:
        return [Reading(
            channel="runtime_health",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="core.runtime.health_contract",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    try:
        report = runtime_health_report()
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        return [Reading(
            channel="runtime_health",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="runtime_health_report()",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    if not isinstance(report, dict):
        return [Reading(
            channel="runtime_health",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="runtime_health_report()",
            detail="report is not a mapping",
        )]

    readings = [Reading(
        channel="runtime_health",
        state=ReadingState.READ,
        value=str(report.get("status") or "unknown"),
        provenance="runtime_health_report().status",
    )]

    failing = _failing_jobs(report)
    readings.append(Reading(
        channel="failing_jobs",
        state=ReadingState.READ,
        value=failing,
        unit="jobs",
        provenance="runtime_health_report().full_runtime.components.autonomy_conductor.jobs",
        detail=f"{len(failing)} with failures",
    ))
    return readings


def _failing_jobs(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Conducted jobs with a nonzero failure count, newest error kept.

    The question "has a job of yours been failing repeatedly" has an exact
    answer in this structure — overt_action_cycle stood at failures=13 with its
    TypeError attached — and no path existed to reach it.
    """
    jobs = (
        report.get("full_runtime", {})
        .get("components", {})
        .get("autonomy_conductor", {})
        .get("jobs", {})
    )
    if not isinstance(jobs, dict):
        return []
    failing: list[dict[str, Any]] = []
    for name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        try:
            failures = int(job.get("failures") or 0)
        except (TypeError, ValueError):
            failures = 0
        if failures <= 0:
            continue
        last = job.get("last_result")
        error = ""
        if isinstance(last, dict):
            error = str(last.get("error") or "")
        failing.append({"job": str(name), "failures": failures, "error": error})
    failing.sort(key=lambda row: (-row["failures"], row["job"]))
    return failing


def resolve_self_health() -> EvidenceBundle:
    """Consult every channel that answers "what is wrong with you right now"."""

    readings: list[Reading] = []
    readings.extend(_health_readings())
    readings.extend(_degradation_readings())
    return EvidenceBundle(demand="self_health", readings=tuple(readings))


def render_self_health_answer(bundle: EvidenceBundle) -> str:
    """Build the answer out of the readings, or say exactly what was missing.

    Deterministic on purpose. This is the half that makes the bundle causal
    rather than decorative: the text is a function of the values, so it cannot
    drift from them, and when nothing was readable it says which channel failed
    instead of producing a fluent paragraph about being fine.
    """

    by_channel = {r.channel: r for r in bundle.readings}
    lines: list[str] = []

    status = by_channel.get("runtime_health")
    if status is not None and status.present:
        lines.append(f"Overall runtime status: {status.value}.")

    failing = by_channel.get("failing_jobs")
    if failing is not None and failing.present:
        rows = list(failing.value or [])
        if rows:
            lines.append("Jobs failing repeatedly:")
            for row in rows:
                error = str(row.get("error") or "").strip()
                suffix = f" — {error}" if error else ""
                lines.append(f"  • {row['job']}: {row['failures']} failures{suffix}")
        else:
            lines.append("No conducted job is currently recording failures.")

    degradations = by_channel.get("degradations")
    if degradations is not None and degradations.present:
        records = list(degradations.value or [])
        if records:
            lines.append(f"Recent degradations ({len(records)}):")
            for record in records[-6:]:
                subsystem = str(record.get("subsystem") or "?")
                message = str(record.get("error") or record.get("message") or "").strip()
                lines.append(f"  • {subsystem}: {message[:160]}" if message else f"  • {subsystem}")
        else:
            lines.append("No degradations recorded recently.")

    unreadable = [r for r in bundle.absent]
    for reading in unreadable:
        lines.append(
            f"{reading.channel}: not readable right now "
            f"({reading.state}{': ' + reading.detail if reading.detail else ''})."
        )

    if not lines:
        return ""
    return "\n".join(lines)
