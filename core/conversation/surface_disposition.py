"""What to do with a user-facing draft that failed something.

Three independent gates decide whether a draft may reach a person: the MLX
worker before IPC success, the inference gate on the way back from the model,
and the response-generation phase. Each grew its own notion of "unsafe", each
could veto alone, and none of them could see what the others had. The
aggregate behaviour was "any veto anywhere ends the turn", which is how a
correct 735-character derivation became "I couldn't get to an answer I'd stand
behind on that one" on 2026-07-26 — after two of the three had already been
taught to pass it along.

The distinction that actually matters is not per-gate. It is per-REASON, and
there are only two kinds:

  INTEGRITY  — the text must not be spoken at all. A prompt artefact, a lane
               telemetry leak, corrupted language, a failure envelope, text
               with no grammar in it. No amount of repair makes it servable
               and serving it is worse than saying nothing.

  SHORTFALL  — the text is real and the turn wanted more of it. Truncated,
               thin, missing a requested count, a derivation that stopped one
               step before its answer. Repair should try; and if repair has
               nothing better, the draft is still what the person should get,
               because it beats an apology.

Anything unrecognised is treated as INTEGRITY. A new reason that nobody has
classified is not assumed safe to speak.
"""

from __future__ import annotations

import contextvars
from enum import Enum
from typing import Any, Iterable

__all__ = [
    "SurfaceDisposition",
    "SHORTFALL_REASONS",
    "clear_preserved_draft",
    "disposition_for",
    "draft_is_servable",
    "integrity_failures",
    "preserved_draft",
    "preserve_draft",
]

# The turn's best servable draft, from whichever layer last held one.
#
# A draft can be judged repairable deep in the stack — the inference gate and
# the response-generation phase both log that they are preserving one — and
# then be unreachable at the place that decides whether to refuse, because it
# was never threaded that far. Live 2026-07-26 all three gates preserved a
# 199-character draft and the route refused anyway, holding nothing.
#
# A context variable is the right shape: it is turn-scoped without changing any
# signature between here and there, and it cannot leak across requests.
_PRESERVED_DRAFT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_preserved_servable_draft", default=""
)


def preserve_draft(text: Any) -> None:
    """Record a draft this turn could serve if nothing better arrives."""
    body = str(text or "").strip()
    if body:
        _PRESERVED_DRAFT.set(body)


def preserved_draft() -> str:
    """The best draft preserved during this turn, or ""."""
    try:
        return _PRESERVED_DRAFT.get()
    except LookupError:
        return ""


def clear_preserved_draft() -> None:
    """Start a turn holding nothing."""
    _PRESERVED_DRAFT.set("")


class SurfaceDisposition(Enum):
    """What a gate should do with the draft it is holding."""

    SERVE = "serve"
    """Nothing objected, or nothing that matters. Hand it over."""

    REPAIR = "repair"
    """Real content that fell short. Try to improve it; serve it regardless."""

    DISCARD = "discard"
    """Must not be spoken. Another generation, or an honest refusal."""


#: Reasons that mean "the turn wanted more of this", not "this must not be
#: said". Keep this list additive and explicit: the default for an unknown
#: reason is DISCARD, so forgetting to classify one fails safe.
SHORTFALL_REASONS: frozenset[str] = frozenset(
    {
        # Content that exists and stopped short.
        #
        # THINNESS IS NOT HERE, deliberately. A truncated derivation has real
        # content the person can use; a thin one has none, and downstream
        # repair cannot invent the missing answer — "I don't know what caused
        # that timeout yet" clears every length floor and is still a
        # non-answer. Those need another generation, which is the existing
        # documented decision in inference_gate and stays that way.
        "truncated_tail",
        "final_answer_missing",
        "incomplete_code_response",
        # Requested-shape shortfalls
        "missing_requested_word_count",
        "missing_requested_sentence_count",
        "missing_requested_paragraph_count",
        "missing_requested_list_count",
        "missing_requested_followup_question",
        "missing_requested_self_process_coverage",
        "missing_requested_objective_facets",
        "missing_requested_memory_limit_coverage",
        # Tone and framing: worth improving, not worth withholding
        "generic_assistant_language",
        "persona_card_deflection",
        "detail_request_deflection",
        "vague_status_derailment",
        "status_page_self_reflection",
        "pseudo_internal_jargon",
        "off_topic_self_reflection_reply",
        "unsupported_operational_status_overclaim",
        "unsupported_runtime_telemetry_inference",
        "unsupported_tool_readiness_claim",
    }
)


def _reason_set(reasons: Any) -> set[str]:
    if reasons is None:
        return set()
    if isinstance(reasons, str):
        return {reasons} if reasons else set()
    if isinstance(reasons, Iterable):
        return {str(reason) for reason in reasons if str(reason)}
    return set()


def integrity_failures(reasons: Any) -> tuple[str, ...]:
    """The reasons in this set that make the text unspeakable."""
    return tuple(sorted(_reason_set(reasons) - SHORTFALL_REASONS))


def disposition_for(reasons: Any) -> SurfaceDisposition:
    """What should happen to a draft carrying these objections."""
    found = _reason_set(reasons)
    if not found:
        return SurfaceDisposition.SERVE
    if found - SHORTFALL_REASONS:
        return SurfaceDisposition.DISCARD
    return SurfaceDisposition.REPAIR


def draft_is_servable(reasons: Any) -> bool:
    """Whether this draft may still reach the person, repaired or as-is.

    The question every gate should be asking instead of "did anything fail".
    """
    return disposition_for(reasons) is not SurfaceDisposition.DISCARD
