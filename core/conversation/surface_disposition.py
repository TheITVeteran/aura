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
import re
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
    "short_draft_answers_closed_question",
    "begin_turn_tool_receipts",
    "record_tool_receipt",
    "turn_tool_receipts",
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
    begin_turn_tool_receipts()


# What this turn ACTUALLY executed.
#
# Live 2026-07-27, asked to run a tool for real and show the output, she
# replied "Python code: 2 + 2 Output: 4". No tool ran. The only dispatches in
# the log that minute were the autonomous curiosity and refactor loops; the
# execution, and its result, were written by the language model.
#
# A claim about the world needs something in the world behind it, and for
# "I ran it" that something is a receipt. The holder is a mutable list rather
# than a scalar because asyncio copies the context into child tasks: a tool
# executed inside the turn's task tree appends to the same list the route
# reads afterwards, while the autonomous loops — whose tasks do not descend
# from this turn — cannot reach it. That distinction is the whole point.
#
# The default is None, not []. A ContextVar default is ONE object shared by
# every context that never set it, so `default=[]` meant a receipt recorded
# outside a turn — a background loop, a boot probe, anything that never called
# begin_turn_tool_receipts — was appended to that shared list and stayed there
# for the life of the process. An `ok: True` landing in it silently vouched for
# every completed-action claim in every later turn that had not begun its own
# list, which is the exact fail-open this whole mechanism exists to prevent,
# and it handed the autonomous loops the reach the comment above denies them.
_TURN_TOOL_RECEIPTS: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("aura_turn_tool_receipts", default=None)
)


def begin_turn_tool_receipts() -> None:
    """Start a turn having executed nothing."""
    _TURN_TOOL_RECEIPTS.set([])


def record_tool_receipt(tool_name: Any, *, ok: bool) -> None:
    """Record that a tool really ran in this turn, and whether it worked.

    Outside a turn there is nothing to record against, and inventing a list
    here would be writing into whatever context happens to be current.
    """
    name = str(tool_name or "").strip()
    if not name:
        return
    try:
        receipts = _TURN_TOOL_RECEIPTS.get()
    except LookupError:
        return
    if not isinstance(receipts, list):
        return
    # Bounded: a turn that fires hundreds of tools does not need all of them
    # to answer the only question asked here — did anything run?
    if len(receipts) < 64:
        receipts.append({"tool": name, "ok": bool(ok)})


def turn_tool_receipts() -> tuple[dict[str, Any], ...]:
    """Tools that really executed during this turn."""
    try:
        receipts = _TURN_TOOL_RECEIPTS.get()
    except LookupError:
        return ()
    return tuple(receipts) if isinstance(receipts, list) else ()


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


def best_available_reply(*, minimum_words: int = 12, question: Any = "") -> str:
    """The best thing this turn produced that is safe to say, or "".

    Checked in order of preference: a draft some layer deliberately preserved,
    then the model's own raw output. Either is returned only if nothing in it
    is unspeakable — a leak never becomes the fallback — and only if it is
    substantial enough to be worth more than an honest "ask me again".
    """
    from core.conversation.response_reliability import assess_user_facing_reply

    for candidate in (preserved_draft(), raw_model_draft()):
        body = str(candidate or "").strip()
        if not body:
            continue
        # The word floor is a preference, not a verdict. A complete answer to
        # a closed question clears nothing and is still the answer.
        if len(body.split()) < minimum_words and not short_draft_answers_closed_question(
            body, question
        ):
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
    "unfounded_tool_execution_claim",
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


#: A question whose correct answer is allowed to be two characters long.
_CLOSED_QUANTITY_PROMPT_RE = re.compile(
    r"\b(?:how many|how much|how old|how long|how far|what(?:'s| is| are)\s+"
    r"[^?]*\b(?:\d|times|plus|minus|divided|multiplied|squared|percent|sum|"
    r"total|average)\b)",
    re.IGNORECASE,
)
_ARITHMETIC_PROMPT_RE = re.compile(
    r"\d\s*(?:[-+*/x×÷]|times|plus|minus|divided by|multiplied by)\s*\d",
    re.IGNORECASE,
)
_POLAR_PROMPT_RE = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|will|would|should|has|have|"
    r"had|am)\b",
    re.IGNORECASE,
)
#: Acknowledgements that answer nothing on their own.
_FILLER_DRAFTS = frozenset(
    {"ok", "okay", "sure", "fine", "done", "right", "yep", "got it"}
)


def short_draft_answers_closed_question(text: Any, question: Any) -> bool:
    """Whether a SHORT draft is a finished answer rather than a stub.

    Thinness gates cannot tell "68" from "ok". Both are two-or-three
    character replies; one is the complete and correct answer to "what's 17
    times 4?" and the other is an acknowledgement that answers nothing. A
    gate that judges on length alone must get one of them wrong, and the
    one it gets wrong is the correct answer — every time the question was
    a closed one.

    Measured live 2026-08-04: Bryan asked "what's 17 times 4?", the Cortex
    answered "68", and three separate gates rejected it for being three
    characters long. He was told "I couldn't get to an answer I'd stand
    behind" about arithmetic she had already got right, and the runtime's
    own degradation record read "turn ended holding a servable answer that
    was never shown to the person".

    So the question decides, not the length. A quantity or a yes/no admits
    a brief answer; an open question does not, and a stub answering one
    still goes back for another generation.
    """
    draft = str(text or "").strip()
    prompt = str(question or "").strip()
    if not draft or not prompt:
        return False
    if draft.casefold().strip(".!") in _FILLER_DRAFTS:
        return False
    wants_quantity = bool(
        _ARITHMETIC_PROMPT_RE.search(prompt) or _CLOSED_QUANTITY_PROMPT_RE.search(prompt)
    )
    if wants_quantity and any(char.isdigit() for char in draft):
        return True
    if _POLAR_PROMPT_RE.match(prompt) and re.match(
        r"^\s*(?:yes|no|yeah|nope|not\b|it\s+is|it\s+isn't)\b", draft, re.IGNORECASE
    ):
        return True
    return False
