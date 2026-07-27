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
the direction of the default is the whole design:

  UNSPEAKABLE — an ALLOWLIST. The text must not be said at all: a control
                token, a prompt artefact, an internal label, corrupted
                language, text with no grammar in it. Every entry is something
                positively identified in the text, never an estimate that
                quality is absent. Only these may destroy an answer.

  EVERYTHING ELSE — at most a request for repair. Truncated, thin, missing a
                requested count, one step short of its conclusion, or flagged
                by a detector written next week. The draft still reaches the
                person if nothing better arrives.

The default used to run the other way: anything unclassified was discarded, on
the theory that failing closed is safe. For a conversation it is not. A reply
withheld is a certain loss; a reply served with a flaw is a partial one — and
because a destroyed answer is indistinguishable from a model failure, the
runtime reported its own vetoes as "I couldn't get to an answer I'd stand
behind". Every incident therefore tightened the gates and nothing ever loosened
them. On 2026-07-26 that ratchet destroyed six consecutive correct answers to
the same question, each for a different heuristic.

A pipeline of gates that can only subtract has an output quality of min() over
all of them, which is why a system this large could answer worse than the bare
model it is built around. Hence the second half of this module: the raw model
draft is kept, and it is the floor nothing is allowed to fall below.
"""

from __future__ import annotations

import contextvars
from enum import Enum
from typing import Any, Iterable

__all__ = [
    "SurfaceDisposition",
    "SHORTFALL_REASONS",
    "UNSPEAKABLE_REASONS",
    "best_available_reply",
    "raw_model_draft",
    "record_raw_model_draft",
    "repair_is_an_improvement",
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
    _RAW_MODEL_DRAFT.set("")


# The bare model's own answer, before any of this runs.
#
# The vanilla floor: whatever the surrounding architecture does, the person
# should never receive LESS than the model alone would have given them. A
# dozen gates that can each only subtract make that guarantee impossible to
# hold by accident — it has to be an explicit fallback, holding the one thing
# every layer downstream is capable of losing.
_RAW_MODEL_DRAFT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_raw_model_draft", default=""
)


def record_raw_model_draft(text: Any) -> None:
    """Remember what the model actually said, before the pipeline touched it."""
    body = str(text or "").strip()
    if body:
        _RAW_MODEL_DRAFT.set(body)


def raw_model_draft() -> str:
    """What the model said this turn, or ""."""
    try:
        return _RAW_MODEL_DRAFT.get()
    except LookupError:
        return ""


def repair_is_an_improvement(before: Any, after: Any, question: Any = "") -> bool:
    """Whether a repair actually made the reply better, or merely different.

    Most of the gates in this pipeline were written when Aura was a weaker
    system, and they encode style expectations from that period — deflection,
    generic-assistant phrasing, status-page tone. Since the disposition
    inversion none of them can destroy an answer any more. What they could
    still do is trigger a repair that REPLACES a good answer with a blander
    one, which is the same loss arriving by a different route.

    So a repair has to earn the substitution: it may not introduce an objection
    the original did not have, and it may not be substantially shorter. If it
    does neither, it is an improvement and it wins; otherwise the original
    stands. A gate that cannot describe a better answer does not get to
    replace this one.
    """
    from core.conversation.response_reliability import assess_user_facing_reply

    original = str(before or "").strip()
    replacement = str(after or "").strip()
    if not replacement:
        return False
    if not original:
        return True
    try:
        original_reasons = set(assess_user_facing_reply(question, original).reasons)
        replacement_reasons = set(assess_user_facing_reply(question, replacement).reasons)
    except (RuntimeError, TypeError, ValueError):
        return False
    if replacement_reasons - original_reasons:
        return False
    if replacement_reasons & UNSPEAKABLE_REASONS:
        return False
    # Losing a third of the answer is a downgrade even when it silences a
    # complaint — unless the complaint was that the answer must be shorter.
    if len(replacement.split()) * 3 < len(original.split()) * 2:
        return bool(
            original_reasons
            and original_reasons - replacement_reasons
            and any("count" in reason or "brev" in reason for reason in original_reasons)
        )
    return True


def best_available_reply(*, minimum_words: int = 12) -> str:
    """The best thing this turn produced that is safe to say, or "".

    Checked in order of preference: a draft some layer deliberately preserved,
    then the model's own raw output. Either is returned only if nothing in it
    is unspeakable — a leak never becomes the fallback — and only if it is
    substantial enough to be worth more than an honest "ask me again".
    """
    from core.conversation.response_reliability import assess_user_facing_reply

    for candidate in (preserved_draft(), raw_model_draft()):
        body = str(candidate or "").strip()
        if not body or len(body.split()) < minimum_words:
            continue
        try:
            reasons = assess_user_facing_reply("", body).reasons
        except (RuntimeError, TypeError, ValueError):
            continue
        if not (set(reasons) & UNSPEAKABLE_REASONS):
            return body
    return ""


class SurfaceDisposition(Enum):
    """What a gate should do with the draft it is holding."""

    SERVE = "serve"
    """Nothing objected, or nothing that matters. Hand it over."""

    REPAIR = "repair"
    """Real content that fell short. Try to improve it; serve it regardless."""

    DISCARD = "discard"
    """Must not be spoken. Another generation, or an honest refusal."""


#: The only reasons that may DESTROY a reply.
#:
#: This list is an allowlist, and that direction is the whole point. Every
#: entry is something POSITIVELY IDENTIFIED in the text — a control token, a
#: prompt artefact, an internal label, text with no grammar in it. None of them
#: is an absence-of-quality judgement, because absence judgements are
#: heuristics and heuristics are wrong on a fraction of good answers.
#:
#: Everything else — every quality heuristic, present and future — can at most
#: ask for repair. A new detector therefore cannot silently start destroying
#: answers; to gain that power it has to be added here, deliberately, with the
#: evidence that it identifies rather than estimates.
#:
#: The old direction was the opposite: anything unclassified was DISCARD, on
#: the theory that failing closed is safe. For a conversation it is not. A
#: reply withheld is a certain loss; a reply served with a flaw is a partial
#: one. On 2026-07-26 a dozen gates each failing closed on their own heuristic
#: destroyed six consecutive correct answers to the same question, and every
#: one of them was reported to the person as though the model had failed.
UNSPEAKABLE_REASONS: frozenset[str] = frozenset(
    {
        "empty_reply",
        "empty_model_output",
        "escaped_control_artifact",
        "prompt_artifact",
        "prompt_echo_contamination",
        "protocol_artifact_leakage",
        "runtime_boilerplate",
        "raw_lane_telemetry",
        "raw_tool_result_fragment",
        "internal_live_gate_leak",
        "cognitive_engine_failure_envelope",
        "backend_symbolic_surface_leak",
        "raw_model_identity_leak",
        "corrupted_language",
        "unexpected_cjk_intrusion",
        # A measurement with a wide margin rather than a guess: English prose
        # runs 13-48% function words and these collapses run 0-5%. It is here
        # because it identifies text that is not language, without needing to
        # know the topic — the one judgement no other gate can make safely.
        "function_word_starvation",
        # The question demanded a quantity and the reply contains none, so it
        # answers a different question. Checkable, not estimated.
        "numeric_answer_missing",
        "arithmetic_answer_missing",
        "surface_validation_prompt_binding_invalid",
        "surface_quality_gate_unavailable",
        # ── Claims the runtime cannot support ────────────────────────────
        # These are not style. They are statements about reality that are
        # false: having spoken aloud, having a body, having run a tool,
        # remembering a conversation that did not happen, addressing a person
        # who is not there, reciting host telemetry as felt state. Each is
        # IDENTIFIED by the claim it makes rather than estimated from tone,
        # which is what earns them a place here — an answer that invents
        # something is worse than no answer, and saying so is the one thing a
        # person cannot check for themselves.
        "unfounded_voice_intrusion",
        "unsupported_embodiment_claim",
        "unsupported_affection_claim",
        "unsupported_self_telemetry_claim",
        "unsupported_external_provider_path_claim",
        "unsupported_context_continuation_claim",
        "unsupported_deployment_routing_claim",
        "unsupported_runtime_limits_claim",
        "host_telemetry_substituted_for_self_condition",
        "ungrounded_person_narrative",
        "ungrounded_person_address",
        "template_telemetry_greeting",
        "unfounded_alarm_derailment",
        "unrequested_pop_culture_intrusion",
        # An explicit instruction, checked by string comparison rather than
        # judged: the person said "answer exactly: yes" and the reply is
        # something else. Identification, not estimate.
        "missing_requested_exact_reply",
        # Internal task assignments and protocol tags spoken as though they
        # were speech: "<answer>…", "[SWARM PROTOCOL…", "To deconstruct and
        # comprehensively research the user preference…". Literal fragments of
        # the runtime's own machinery, not a judgement about quality.
        "internal_task_prompt_leak",
    }
)

#: Reasons that mean "the turn wanted more of this". Retained as the explicit
#: record of what has been triaged; the disposition no longer depends on a
#: reason appearing here, only on its absence from UNSPEAKABLE_REASONS.
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
    return tuple(sorted(_reason_set(reasons) & UNSPEAKABLE_REASONS))


def disposition_for(reasons: Any) -> SurfaceDisposition:
    """What should happen to a draft carrying these objections.

    A quality heuristic may ask for repair. Only positively identified
    unspeakable content may destroy the answer.
    """
    found = _reason_set(reasons)
    if not found:
        return SurfaceDisposition.SERVE
    if found & UNSPEAKABLE_REASONS:
        return SurfaceDisposition.DISCARD
    return SurfaceDisposition.REPAIR


def draft_is_servable(reasons: Any) -> bool:
    """Whether this draft may still reach the person, repaired or as-is.

    The question every gate should be asking instead of "did anything fail".
    """
    return disposition_for(reasons) is not SurfaceDisposition.DISCARD
