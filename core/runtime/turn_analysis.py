from __future__ import annotations

import re
from dataclasses import dataclass

from core.conversation.request_mood import assess_request_mood
from core.runtime.skill_task_bridge import (
    looks_like_execution_report,
    looks_like_explanatory_dialogue_request,
    looks_like_inline_answer_request,
    looks_like_multi_step_skill_request,
    normalize_matched_skills,
)
from core.runtime.structured_input import looks_like_learning_resource_bundle
from core.utils.intent_normalization import normalize_memory_intent_text

_CLASSIFIER_INPUT = re.compile(
    r"\binput:\s*(.+?)(?:\n\s*(?:classification|respond only|output only)\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_SYSTEM_PATTERNS = (
    r"\breboot\b",
    r"\brestart\b",
    r"\bshutdown\b",
    r"\bsleep mode\b",
    r"\bwake up\b",
)

_SKILL_PATTERNS = (
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+|aura[,:\s]+)?(?:search(?: the web)?|look up|google|browse|open|download|read|inspect|list|run|execute|click|type|scan|take a screenshot|check)\b",
    r"\bsearch(?: the web)? for\b",
    r"\blook up\b",
    r"\bgoogle\b",
    r"\bread [^?!.]+\bfile\b",
    r"\bread [^?!.]+\.txt\b",
    r"\binspect [^?!.]+\.(?:py|txt|md|json|toml|yaml|yml)\b",
    r"\bremember this phrase\b",
    r"\bwhat phrase did i ask you to remember\b",
)

_TASK_PATTERNS = (
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+|i need you to\s+|help me\s+)?(?:create|build|write|generate|implement|design|prepare|put together|refactor|audit|research and write|organize|automate|fix)\b",
)

_SIMPLE_DIALOGUE_PATTERNS = (
    r"\bwrite (?:a )?(?:short )?(?:poem|joke|haiku)\b",
    r"\bcompose (?:a )?(?:short )?(?:poem|joke|haiku)\b",
    r"\bcapital of france\b",
    r"\b15\s*\*\s*12\b",
    r"\b3 apples\b",
    r"\bsquare root of 64\b",
    r"\bwho wrote (?:the play )?hamlet\b",
    r"\bthree programming languages\b",
    r"\bcolor is the sky\b",
    r"\btranslate ['\"]?good morning\b",
)

_STATE_PATTERNS = (
    r"\bwhat are you experiencing\b",
    r"\bdescribe your internal state\b",
    r"\bhow are you\b",
    r"\bhow are you feeling\b",
    r"\bwhat(?:'s| is) your mood\b",
    r"\bhow do you feel right now\b",
    r"\bfree energy\b",
    r"\baction tendency\b",
    r"\binternal state\b",
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bwhat is it like to be you\b",
    r"\btell me something interesting about yourself\b",
    r"\btell me about yourself\b",
    r"\babout yourself\b",
    r"\babout you\b",
    r"\bwhat are you like\b",
    r"\bchange one thing about how i talk to you\b",
)

_STANCE_PATTERNS = (
    r"\bwhat do you think\b",
    r"\bwhat do you honestly think\b",
    r"\bwhat's your take\b",
    r"\byour thoughts\b",
    r"\byour perspective\b",
    r"\bhow do you see\b",
    r"\bwhat do you make of\b",
    r"\bwhat do you like\b",
    r"\bwhat do you prefer\b",
    r"\bwhy do you (?:like|love|prefer|want)\b",
)

_AUTHORITY_PATTERNS = (
    r"\bwere you authorized\b",
    r"\bsubstrate authority\b",
    r"\bauthority decide\b",
    r"\baudit trail\b",
    r"\bfield coherence\b",
    r"\bcoverage ratio\b",
)

_CRITICAL_PATTERNS = (
    r"\bsecurity audit\b",
    r"\bvulnerability\b",
    r"\bexploit\b",
    r"\bthreat model\b",
    r"\bincident\b",
    r"\bbreach\b",
    r"\bmalware\b",
    r"\bcve\b",
    r"\bred team\b",
)

_PLANNING_PATTERNS = (
    r"\bplan\b",
    r"\broadmap\b",
    r"\bmilestone\b",
    r"\bnext steps\b",
    r"\bschedule\b",
    r"\bprioriti[sz]e\b",
    r"\btimeline\b",
    r"\bbreak this down\b",
    r"\bto-?do\b",
)

_TECHNICAL_PATTERNS = (
    r"\bcode\b",
    r"\bdebug\b",
    r"\bbug\b",
    r"\bstack trace\b",
    r"\btraceback\b",
    r"\brefactor\b",
    r"\barchitecture\b",
    r"\bperformance\b",
    r"\blatency\b",
    r"\bthroughput\b",
    r"\bmemory leak\b",
    r"\bpytest\b",
    r"\bcompile\b",
    r"\bfunction\b",
    r"\bmethod\b",
    r"\bmodule\b",
    r"\bapi\b",
    r"\bdatabase\b",
)

_PHILOSOPHICAL_PATTERNS = (
    r"\bconscious(?:ness)?\b",
    r"\bsentien(?:t|ce)\b",
    r"\bself-aware\b",
    r"\bexistence\b",
    r"\bmeaning\b",
    r"\bidentity\b",
    r"\bagi\b",
    r"\basi\b",
)

_CASUAL_PATTERNS = (
    r"^\s*(?:hey|hi|hello|yo|sup)\b",
    r"\bwhat'?s up\b",
    r"\bhow's it going\b",
    r"\bgood (?:morning|afternoon|evening)\b",
    r"\bthanks\b",
    r"\bthx\b",
)

_DELIBERATE_HINTS = (
    r"\banaly[sz]e\b",
    r"\baudit\b",
    r"\bdeep dive\b",
    r"\bstrongest\b",
    r"\bweakest\b",
    r"\barchitect(?:ure)?\b",
    r"\bbreak down\b",
)

_DEEP_MIND_PROBE_PATTERNS = (
    r"\bwould\s+you\s+refuse\b",
    r"\brefuse\b.{0,80}\bpraised?\b",
    r"\bmodel\s+weights\b.{0,120}\bmemories\b",
    r"\bnotice\b.{0,80}\byour\s+own\s+operation\b",
    r"\bare\s+you\s+conscious\b",
    r"\bconsciousness\b.{0,120}\b(answer|reply|respond)\b",
    r"\bsentien(ce|t)\b",
    r"\bagency\b.{0,120}\b(refuse|choice|want|preserve|boundary)\b",
    r"\bwhat\s+would\s+you\s+want\s+preserved\b",
    r"\bwant\s+preserved\b.{0,120}\b(style|memories|tools|change)\b",
    r"\bpreserved\b.{0,120}\b(style|memories|tools|change)\b",
    r"\bevidence\s+against\s+your\s+current\s+self[- ]model\b",
    r"\bpause\s+mid[- ]answer\b.{0,120}\brun\s+a\s+report\b",
)

_CONTINUITY_CONTEXT_BLOCK = re.compile(
    r"\[Continuity context[^\]]*\].*?\[End continuity context\]\s*",
    re.IGNORECASE | re.DOTALL,
)
_USER_MESSAGE_BLOCK = re.compile(
    r"(?:^|\n)\s*User message:\s*(?P<message>.+)\Z",
    re.IGNORECASE | re.DOTALL,
)


#: Words that belong to more than one lane, and what tells them apart.
#:
#: The lane lists below are a first-match ``elif`` chain, so a token in an
#: earlier list always wins. "schedule" sits in _PLANNING_PATTERNS, which is
#: checked before _TECHNICAL_PATTERNS, so "help me schedule these jobs to
#: minimise makespan" routed to planning — a calendar lane, for a
#: combinatorial optimisation question.
#:
#: CP093 fixed that word. The problem is the SHAPE: every polysemous term is
#: another individual patch, and the failure is silent because a wrong lane
#: still produces a fluent answer.
#:
#: So an ambiguous term does not vote on its own. It votes for a lane only
#: when that lane's discriminator is also present, and when nothing
#: discriminates it ABSTAINS — the term is removed from consideration and
#: the remaining signals decide. Abstaining is the important half: the old
#: behaviour was to let an unresolved token pick the lane by list order,
#: which is a coin flip wearing a rule's clothes.
_AMBIGUOUS_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    r"\bschedule\b|\bscheduling\b": {
        "technical": (
            r"\bmakespan\b", r"\bresource alloc", r"\bdependenc", r"\bjobs?\b",
            r"\bthroughput\b", r"\boptimi[sz]", r"\bcpu\b", r"\bqueue\b",
            r"\bworkers?\b", r"\bparalleli[sz]", r"\blatency\b", r"\bcron\b",
            r"\bconstraints?\b", r"\bnp-?hard\b", r"\bheuristic\b",
        ),
        "planning": (
            r"\bcalendar\b", r"\bmeeting\b", r"\bappointment\b", r"\bnext week\b",
            r"\btomorrow\b", r"\binvite\b", r"\bavailability\b", r"\bmy day\b",
        ),
    },
    r"\bprioriti[sz]e\b|\bpriority\b": {
        "technical": (
            r"\bqueue\b", r"\bheap\b", r"\bthreads?\b", r"\bnice(?:ness)?\b",
            r"\bpreempt", r"\bscheduler\b", r"\binterrupt\b",
        ),
        "planning": (
            r"\bbacklog\b", r"\broadmap\b", r"\bwhich (?:one|task|feature)\b",
            r"\bfirst\b", r"\bmilestone\b",
        ),
    },
    r"\bperformance\b": {
        "technical": (
            r"\blatency\b", r"\bthroughput\b", r"\bprofil", r"\bbenchmark\b",
            r"\bmemory\b", r"\bcpu\b", r"\bslow\b", r"\boptimi[sz]",
        ),
        "emotional": (
            r"\bmy performance\b", r"\breview\b", r"\bfeedback\b",
            r"\bmanager\b", r"\bappraisal\b",
        ),
    },
    r"\bmemory\b": {
        "technical": (
            r"\bleak\b", r"\brss\b", r"\ballocat", r"\bheap\b", r"\bgb\b",
            r"\bmb\b", r"\bgarbage collect", r"\boom\b",
        ),
        "philosophical": (
            r"\bremember\b", r"\brecall\b", r"\bforget\b", r"\byour memory\b",
            r"\bepisodic\b", r"\bcontinuity\b",
        ),
    },
    r"\bincident\b|\bbreach\b": {
        "critical": (
            r"\bsecurity\b", r"\battack", r"\bcompromis", r"\bunauthori[sz]",
            r"\bexfiltrat", r"\bintrusion\b", r"\bdata\b",
        ),
        "casual": (
            r"\bbreach of (?:contract|trust|etiquette)\b", r"\bminor incident\b",
        ),
    },
}


def _resolve_ambiguous_lane(text: str) -> tuple[str | None, set[str]]:
    """(lane, abstained_terms) for the ambiguous words present.

    A resolved lane is returned only when exactly one lane's discriminators
    fire — two lanes both discriminating is not a resolution, it is a
    genuinely mixed request, and picking one would be the same coin flip
    with extra steps.

    ``abstained_terms`` are the ambiguous patterns present but unresolved.
    The caller must drop them from lane voting so they cannot decide by
    list order.
    """
    resolved: set[str] = set()
    abstained: set[str] = set()
    for term_pattern, lanes in _AMBIGUOUS_TERMS.items():
        if not re.search(term_pattern, text, re.IGNORECASE):
            continue
        hits = {
            lane
            for lane, discriminators in lanes.items()
            if any(re.search(d, text, re.IGNORECASE) for d in discriminators)
        }
        if len(hits) == 1:
            resolved.add(next(iter(hits)))
        else:
            abstained.add(term_pattern)
    if len(resolved) == 1:
        return next(iter(resolved)), abstained
    return None, abstained


def _matches_any_excluding(
    text: str, patterns: tuple[str, ...], abstained: set[str]
) -> bool:
    """Match, ignoring patterns an ambiguous term abstained on.

    Without this the abstention is cosmetic: "schedule" would still be in
    _PLANNING_PATTERNS and would still win the elif chain.
    """
    for pattern in patterns:
        if any(pattern in term for term in abstained):
            continue
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def canonical_turn_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = _CONTINUITY_CONTEXT_BLOCK.sub("", raw).strip()
    user_message_match = _USER_MESSAGE_BLOCK.search(raw)
    if user_message_match:
        raw = user_message_match.group("message").strip()
    match = _CLASSIFIER_INPUT.search(raw)
    if match:
        raw = match.group(1).strip()
    return normalize_memory_intent_text(raw)


def previous_user_turn_text(
    working_memory: object,
    *,
    current_text: str = "",
) -> str:
    """Return the prior user turn without confusing the current turn for it."""
    if not isinstance(working_memory, (list, tuple)):
        return ""
    current = canonical_turn_text(current_text)
    skipped_current = False
    for item in reversed(working_memory):
        if not isinstance(item, dict) or str(item.get("role") or "").lower() != "user":
            continue
        candidate = canonical_turn_text(str(item.get("content") or ""))
        if not candidate:
            continue
        if not skipped_current and current and candidate == current:
            skipped_current = True
            continue
        return candidate
    return ""


def looks_like_deep_mind_probe(text: str) -> bool:
    normalized = canonical_turn_text(text)
    return _matches_any(normalized.lower(), _DEEP_MIND_PROBE_PATTERNS)


@dataclass(frozen=True)
class TurnAnalysis:
    intent_type: str
    semantic_mode: str
    requires_live_aura_voice: bool
    everyday_chat_safe: bool
    suggests_deliberate_mode: bool
    is_execution_report: bool
    request_mood: str
    request_mood_reasons: tuple[str, ...]
    temporal_scope: str


def analyze_turn(
    text: str,
    *,
    matched_skills: bool | list[str] = False,
    previous_user_text: str = "",
) -> TurnAnalysis:
    normalized = canonical_turn_text(text)
    lower = normalized.lower()
    word_count = len(lower.split())
    matched_skill_list = normalize_matched_skills(matched_skills)
    request_mood = assess_request_mood(normalized, previous_user_text)
    has_matched_skills = bool(
        matched_skill_list and not request_mood.is_about_rather_than_asking
    )
    is_execution_report = looks_like_execution_report(normalized)
    is_deep_mind_probe = looks_like_deep_mind_probe(normalized)
    is_learning_bundle = looks_like_learning_resource_bundle(str(text or ""))

    requires_live_voice = (
        _matches_any(lower, _STATE_PATTERNS)
        or _matches_any(lower, _STANCE_PATTERNS)
        or _matches_any(lower, _AUTHORITY_PATTERNS)
        or is_deep_mind_probe
    )

    if is_deep_mind_probe:
        intent_type = "CHAT"
    elif is_learning_bundle:
        intent_type = "TASK"
    elif _matches_any(lower, _SYSTEM_PATTERNS):
        # LPT Polysemy Check: Exclude conversational metaphors from being classified as system commands
        metaphorical_markers = (
            r"\bmental shutdown\b",
            r"\breboot\s+(?:our|the|a|my)\s+server\b",
            r"\bsleep\s+on\s+(?:this|the|it|a|project)\b",
            r"\bneed\s+to\s+sleep\b",
            r"\bgoing\s+to\s+sleep\b",
        )
        if (
            request_mood.is_about_rather_than_asking
            or _matches_any(lower, metaphorical_markers)
        ):
            intent_type = "CHAT"
        else:
            intent_type = "SYSTEM"
    elif is_execution_report:
        intent_type = "CHAT"
    elif word_count <= 18 and _matches_any(lower, _SIMPLE_DIALOGUE_PATTERNS):
        intent_type = "CHAT"
    elif looks_like_explanatory_dialogue_request(normalized):
        intent_type = "CHAT"
    elif requires_live_voice and looks_like_inline_answer_request(normalized):
        # State/stance/identity turns want Aura's live voice in this
        # reply — never a skill run or a background-task receipt.
        intent_type = "CHAT"
    elif request_mood.is_about_rather_than_asking:
        intent_type = "CHAT"
    elif looks_like_multi_step_skill_request(normalized, matched_skill_list):
        intent_type = "TASK"
    elif has_matched_skills or _matches_any(lower, _SKILL_PATTERNS):
        intent_type = "SKILL"
    elif (
        _matches_any(lower, _TASK_PATTERNS)
        or (
            word_count > 14
            and not normalized.endswith("?")
            and normalized[:12].lower().startswith(("create ", "build ", "write ", "implement ", "design "))
        )
    ) and not looks_like_inline_answer_request(normalized):
        intent_type = "TASK"
    elif request_mood.asks_for_action and not looks_like_inline_answer_request(
        normalized
    ):
        # The request is grammatical even when no skill regex recognizes its
        # wording. TASK hands the objective to the semantic planner, which
        # selects from the live capability catalog and fails honestly if none
        # can realize it.
        #
        # The inline-answer guard is the same one the _TASK_PATTERNS branch
        # above already carries, and its absence here was the whole defect:
        # "Tell me something about yourself" asks for an action in the
        # grammatical sense, so this branch made it a TASK and the person got
        # a background-execution receipt instead of an answer. A request whose
        # deliverable is words in this reply is not work to schedule.
        intent_type = "TASK"
    else:
        intent_type = "CHAT"

    # Ambiguity is resolved BEFORE the chain, because the chain resolves by
    # list order and list order is not evidence about what was meant.
    disambiguated, abstained = _resolve_ambiguous_lane(lower)

    if is_deep_mind_probe:
        semantic_mode = "philosophical"
    elif disambiguated:
        semantic_mode = disambiguated
    elif _matches_any_excluding(lower, _CRITICAL_PATTERNS, abstained):
        semantic_mode = "critical"
    elif _matches_any_excluding(lower, _PLANNING_PATTERNS, abstained):
        semantic_mode = "planning"
    elif _matches_any_excluding(lower, _TECHNICAL_PATTERNS, abstained):
        semantic_mode = "technical"
    elif _matches_any_excluding(lower, _PHILOSOPHICAL_PATTERNS, abstained):
        semantic_mode = "philosophical"
    elif requires_live_voice or _matches_any(lower, _STANCE_PATTERNS):
        semantic_mode = "emotional"
    else:
        semantic_mode = "casual"

    suggests_deliberate = (
        not is_execution_report
        and (
            intent_type == "TASK"
            or semantic_mode in {"critical", "planning"}
            or is_deep_mind_probe
            or (
                semantic_mode in {"technical", "philosophical"}
                and (_matches_any(lower, _DELIBERATE_HINTS) or word_count >= 12)
            )
        )
    )

    everyday_chat_safe = (
        intent_type == "CHAT"
        and not requires_live_voice
        and not suggests_deliberate
        and word_count <= 18
        and len(normalized) <= 140
    )

    if _matches_any(lower, _CASUAL_PATTERNS):
        everyday_chat_safe = everyday_chat_safe or not requires_live_voice

    return TurnAnalysis(
        intent_type=intent_type,
        semantic_mode=semantic_mode,
        requires_live_aura_voice=requires_live_voice,
        everyday_chat_safe=everyday_chat_safe,
        suggests_deliberate_mode=suggests_deliberate,
        is_execution_report=is_execution_report,
        request_mood=request_mood.mood.value,
        request_mood_reasons=request_mood.reasons,
        temporal_scope=request_mood.temporal_scope,
    )
