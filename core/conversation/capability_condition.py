"""What she can do *right now*, in a form she can talk about.

Aura's capability engine already knows, per skill, whether it is available,
what state it is in, and why it is not — `_catalog_item_for_skill` computes
exactly that. None of it ever reached the part of her that speaks. So when a
skill was down she fell back to a string somebody wrote months earlier:

    I can't access external data right now, but based on what I know...

Three things are wrong with that. It is not her voice. It is fixed at a
moment that has nothing to do with this moment. And it collapses a
distinction that matters enormously to a person:

    "I can't search, there's no network"      — true now, false in a minute
    "I don't have a way to search at all"     — true until someone builds it

Bryan's framing, and it is the right one: awareness of a failed event is not
a catastrophe, it is communication. What she can do at one minute may not be
what she can do the next; she has to notice, say so in her own words, and
carry on.

This module is the evidence side of that. It reads the live catalog and
answers one question — for the capability this turn needs, what is true right
now — as a compact block the model can speak from. It deliberately produces
FACTS, not sentences: the moment this file starts writing prose, it becomes
the canned reply it exists to replace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

logger = logging.getLogger("Aura.CapabilityCondition")

__all__ = [
    "CapabilityStanding",
    "CapabilityCondition",
    "capability_condition_evidence",
    "condition_for",
    "needed_capabilities",
]


class CapabilityStanding(str, Enum):
    """The distinction a person actually cares about."""

    READY = "ready"
    #: Registered, but something about *now* is wrong. Recoverable.
    UNAVAILABLE_NOW = "unavailable_now"
    #: Nothing in the runtime provides this. Not a bad minute — a missing limb.
    ABSENT = "absent"
    #: She HAS the faculty and the world is not currently supplying what it
    #: needs. A person who knows how to search does not forget how when the
    #: wifi drops; they say "there's no internet". This is that state, and it
    #: is derived — capability AND preconditions — never asserted.
    BLOCKED_BY_PRECONDITION = "blocked_by_precondition"


@dataclass(frozen=True)
class CapabilityCondition:
    name: str
    standing: CapabilityStanding
    reason: str = ""
    detail: str = ""

    #: What the world is failing to supply, when that is the reason.
    missing_preconditions: tuple[str, ...] = ()

    @property
    def is_transient(self) -> bool:
        return self.standing in (
            CapabilityStanding.UNAVAILABLE_NOW,
            CapabilityStanding.BLOCKED_BY_PRECONDITION,
        )

    @property
    def faculty_intact(self) -> bool:
        """She has this capability, whatever the world is doing."""
        return self.standing is not CapabilityStanding.ABSENT


#: Which capability a turn is reaching for. Intentionally small: this is used
#: to decide whether to LOOK at the catalog, not to route anything, so a miss
#: costs nothing and a false positive only adds evidence nobody needed.
_CAPABILITY_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("web_search", (
        "search", "look up", "look it up", "google", "on the web", "online",
        "latest", "current price", "weather", "news", "who won", "how much is",
    )),
    ("file_operation", (
        "read the file", "open the file", "write a file", "save it to",
        "in my folder", "on disk", "the directory",
    )),
    ("code_execution", (
        "run this", "execute", "run the code", "calculate with python",
        "run it for real",
    )),
    ("computer_use", (
        "click", "open the app", "take a screenshot", "on my screen",
        "my desktop",
    )),
    ("email_adapter", ("email", "inbox", "send a message to")),
)


def needed_capabilities(user_message: Any) -> tuple[str, ...]:
    """Capabilities this turn plausibly reaches for, in cue order."""
    text = str(user_message or "").casefold()
    if not text:
        return ()
    found: list[str] = []
    for name, cues in _CAPABILITY_CUES:
        if any(cue in text for cue in cues) and name not in found:
            found.append(name)
    return tuple(found)


def _catalog_rows(capability_engine: Any) -> Iterable[dict[str, Any]]:
    if capability_engine is None:
        return ()
    for attr in ("iter_tool_catalog", "get_tool_catalog"):
        reader = getattr(capability_engine, attr, None)
        if not callable(reader):
            continue
        try:
            rows = reader(include_inactive=True)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Capability catalog read failed via %s: %s", attr, exc)
            continue
        if rows is None:
            continue
        return rows
    return ()


def condition_for(name: str, *, capability_engine: Any = None) -> CapabilityCondition:
    """The live standing of one capability.

    ABSENT is reserved for "nothing here provides this". A registry that could
    not be read is NOT absence — reporting a missing limb because a lookup
    failed would be a confident lie about herself, so that case reports
    UNAVAILABLE_NOW with the read failure as its reason.
    """
    wanted = str(name or "").strip()
    if not wanted:
        return CapabilityCondition("", CapabilityStanding.ABSENT, "no_capability_named")

    if capability_engine is None:
        try:
            from core.service_container import ServiceContainer

            capability_engine = ServiceContainer.get("capability_engine", default=None)
        except (ImportError, RuntimeError, AttributeError) as exc:
            logger.debug("Capability engine unavailable: %s", exc)
            capability_engine = None

    if capability_engine is None:
        return CapabilityCondition(
            wanted,
            CapabilityStanding.UNAVAILABLE_NOW,
            "capability_registry_unreadable",
        )

    # The world's side, computed first: a capability whose preconditions are
    # missing cannot work no matter how healthy the registry says it is.
    # Composed here rather than described in a prompt, so unplugging the
    # network changes the conclusion by itself.
    try:
        from core.conversation.capability_preconditions import failing_preconditions

        blocked_by = failing_preconditions(wanted)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug("Precondition probe unavailable for %s: %s", wanted, exc)
        blocked_by = ()

    seen_any = False
    for row in _catalog_rows(capability_engine):
        if not isinstance(row, dict):
            continue
        seen_any = True
        row_name = str(row.get("name") or row.get("skill") or "").strip()
        if row_name.casefold() != wanted.casefold():
            continue
        if bool(row.get("available")):
            if blocked_by:
                return CapabilityCondition(
                    wanted,
                    CapabilityStanding.BLOCKED_BY_PRECONDITION,
                    "; ".join(state.fact for state in blocked_by),
                    missing_preconditions=tuple(state.name for state in blocked_by),
                )
            return CapabilityCondition(wanted, CapabilityStanding.READY)
        return CapabilityCondition(
            wanted,
            CapabilityStanding.UNAVAILABLE_NOW,
            str(row.get("availability_reason") or row.get("state") or "unavailable"),
            str(row.get("policy_state") or ""),
        )

    if not seen_any:
        return CapabilityCondition(
            wanted,
            CapabilityStanding.UNAVAILABLE_NOW,
            "capability_registry_empty",
        )
    return CapabilityCondition(wanted, CapabilityStanding.ABSENT, "not_registered")


#: Reason codes rendered as the plain fact underneath them. The model reads
#: these and says it however it wants; nothing here is a sentence she must use.
_REASON_FACTS: dict[str, str] = {
    "capability_registry_unreadable": "her own capability registry could not be read this turn",
    "capability_registry_empty": "no capabilities are loaded yet",
    "not_registered": "nothing in this runtime provides it",
    "disabled_by_policy": "it is switched off by policy",
    "inactive_by_policy": "it is switched off by policy",
    "dependency_not_ready": "something it depends on has not finished loading",
    "network_unavailable": "there is no network right now",
    "offline": "there is no network right now",
    "memory_pressure": "memory is too tight to load it right now",
    "quarantined": "it was quarantined after failing",
    "ERROR": "it errored the last time it ran",
}


def _fact_for(condition: CapabilityCondition) -> str:
    raw = str(condition.reason or "").strip()
    for key, fact in _REASON_FACTS.items():
        if key.casefold() in raw.casefold():
            return fact
    return raw.replace("_", " ") or "it is not available"


def capability_condition_evidence(
    user_message: Any,
    *,
    capability_engine: Any = None,
    already_used: Iterable[str] = (),
) -> str:
    """A block of live capability facts, or "" when the turn needs none.

    Facts only. The instruction to speak in her own words lives with the other
    surface contracts; if this function ever returns a ready-made apology, the
    canned reply has simply moved house.
    """
    wanted = needed_capabilities(user_message)
    if not wanted:
        return ""

    # A capability that already produced evidence THIS TURN is working, full
    # stop. The registry can say whatever it likes; the turn has a receipt.
    # Without this, a search that just succeeded could still be announced as
    # "not available this moment" — the same contradiction as reporting a
    # scan blocked while its results sit in the prompt.
    proven = {str(name or "").casefold() for name in already_used if str(name or "").strip()}

    lines: list[str] = []
    for name in wanted:
        if name.casefold() in proven:
            continue
        condition = condition_for(name, capability_engine=capability_engine)
        if condition.standing is CapabilityStanding.READY:
            lines.append(f"- {name}: available right now")
        elif condition.standing is CapabilityStanding.BLOCKED_BY_PRECONDITION:
            lines.append(
                f"- {name}: YOU HAVE THIS, BUT IT CANNOT WORK RIGHT NOW — "
                f"{condition.reason}. The capability is intact; what it needs "
                f"is missing. Reason from that: it will work again when that "
                f"comes back."
            )
        elif condition.standing is CapabilityStanding.ABSENT:
            lines.append(
                f"- {name}: NOT SOMETHING YOU HAVE — {_fact_for(condition)}. "
                f"This is not a temporary outage."
            )
        else:
            lines.append(
                f"- {name}: NOT AVAILABLE THIS MOMENT — {_fact_for(condition)}. "
                f"You do have this capability; it may work again shortly."
            )
    if not lines:
        return ""

    return (
        "[LIVE CAPABILITY CONDITION]\n"
        + "\n".join(lines)
        + "\n"
        + "Say this in your own words, as part of your answer, the way you "
        "would mention any other fact about your situation. Do not apologise "
        "at length and do not treat it as a failure — it is information. Keep "
        "'not right now' and 'not something I can do' clearly different, and "
        "still answer whatever you can answer without it.\n"
        "[END LIVE CAPABILITY CONDITION]"
    )
