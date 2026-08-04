"""Is this turn asking her to DO something, or talking ABOUT something?

The capability router matches skills by regex over the message, and several of
those patterns contain proper nouns::

    "web_interlocutor": [
        r"(?:ask|message) (?:gemini|chatgpt|claude|another ai)",
        ...
    ]

So naming a thing was enough to route a turn at it. "What do you think of
ChatGPT?" and "Go ask ChatGPT what it thinks" differ in exactly one way, and it
is not which words they contain — it is whether the named thing is the OBJECT
OF AN INSTRUCTION or the SUBJECT OF A REMARK. The same problem is already
visible in the router as a one-off: ``_looks_like_search_capability_question``
exists because "the search for a new apartment" was opening a browser.

Fixing it per-name would need a rule for every app that ever gets a pattern.
The distinction is grammatical, so this measures the grammar:

  DIRECTIVE  the clause is an imperative ("open …", "ask …"), or a
             second-person request ("can you …", "I want you to …").
  MENTION    the thing appears inside a frame that talks about it — a copular
             claim, a stance verb, a wh-question about it, a possessive, a
             report of what it said, or an explicit refusal to act on it.
  AMBIGUOUS  neither frame is present. The router should not manufacture an
             action out of nothing, but this is not evidence of a mention
             either, and it is reported separately so a caller can decide.

Nothing here is about which entity was named. A sentence about Gmail, a
sentence about a colleague and a sentence about ChatGPT are all handled by the
same rules, because the rules are about sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RequestMood(str, Enum):
    DIRECTIVE = "directive"
    MENTION = "mention"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class MoodVerdict:
    mood: RequestMood
    reasons: tuple[str, ...] = ()

    @property
    def asks_for_action(self) -> bool:
        return self.mood is RequestMood.DIRECTIVE

    @property
    def is_about_rather_than_asking(self) -> bool:
        return self.mood is RequestMood.MENTION

    def as_metrics(self) -> dict[str, object]:
        return {"request_mood": self.mood.value, "reasons": list(self.reasons)}


#: Verbs that, at the head of a clause, make it an instruction. These are
#: things one DOES to something, not things one thinks about it.
_ACTION_VERBS = (
    "open|go|goto|navigate|browse|visit|launch|start|run|execute|click|type|"
    "press|scroll|drag|search|look|find|fetch|download|upload|send|message|"
    "email|post|ask|tell|reply|respond|answer|show|pull|load|check|read|write|"
    "save|delete|move|copy|close|switch|log|sign|install|update|play|pause|"
    "take|make|create|build|generate|draw|compose|summarise|summarize|"
    "translate|compare|compile|deploy|test|call|ping|talk|chat|converse|"
    # Memory instructions are instructions. "Remember for future sessions that
    # my codename is glass orchard" asks her to STORE something; reading it as
    # a remark about remembering loses the fact.
    "remember|remind|note|store|save|forget|pin|log|track|record"
)

#: An imperative: optional politeness/discourse lead-in, then a bare verb.
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:(?:ok|okay|now|then|next|also|and|so|hey|please|quick|quickly|"
    r"first|finally|alright|right)[,\s]+)*"
    rf"(?:please\s+)?(?:{_ACTION_VERBS})\b",
    re.IGNORECASE,
)

#: A second-person request. "Can you open…", "I'd like you to ask…".
_SECOND_PERSON_REQUEST_RE = re.compile(
    r"\b(?:can|could|would|will|why don'?t)\s+you\b"
    r"|\bplease\s+(?:can\s+you|could\s+you)?\b"
    r"|\b(?:i|i'?d|i'?ll)\s+(?:want|need|like|would like)\s+you\s+to\b"
    r"|\bi\s+want\s+you\s+to\b"
    r"|\blet'?s\s+\w+"
    r"|\bgo\s+ahead\s+and\b"
    r"|\byour\s+(?:job|task)\s+is\s+to\b",
    re.IGNORECASE,
)

#: Frames that talk ABOUT a thing. Each one is a construction, not a topic.
_MENTION_FRAMES = (
    # stance / opinion / knowledge
    (r"\bwhat(?:'?s| is| do| did| does| are| would)\s+(?:you\s+)?(?:think|make|reckon|feel)\b", "stance_question"),
    (r"\b(?:do|did|have)\s+you\s+(?:know|hear|heard|see|seen|use|used|try|tried|like|remember)\b", "experience_question"),
    (r"\bhow\s+(?:do(?:es)?|did|is|are|was|were)\b.*\bwork\b", "explanation_question"),
    (r"\b(?:what|who|which|when|where|why)\s+(?:is|are|was|were|made|makes|built|owns)\b", "identification_question"),
    (r"\bwhat(?:'?s| is)\s+(?:the\s+)?(?:difference|relationship|comparison)\b", "comparison_question"),
    (r"\bhave\s+you\s+ever\b", "experience_question"),
    (r"\byour\s+(?:opinion|view|take|thoughts?)\b", "opinion_request"),
    # reported speech / recollection
    (r"\b(?:said|told|wrote|claimed|mentioned|answered|replied)\b", "reported_speech"),
    (r"\b(?:remember|recall)\s+(?:when|that|what|how)\b", "recollection"),
    (r"\blast\s+time\b|\bearlier\s+(?:you|we|i)\b|\byesterday\b", "recollection"),
    # Asking to be told about something ALREADY DONE is a recall request, not
    # an instruction to do it again. Measured live 2026-08-04: "Remind me what
    # you and ChatGPT discussed" opened a NEW browser session and held a second
    # conversation, because the router saw an imperative and a tool name and
    # had no way to see that the thing being named had already happened.
    (r"\bremind\s+me\s+(?:what|how|who|when|which|about)\b", "retrospective_request"),
    (r"\bwhat\s+did\s+(?:you|we|it|they)(?:\s+(?:two|both|and\s+\w+))?\s+"
     r"(?:discuss|talk\s+about|say|said|cover|conclude|decide|learn|find|end\s+up)\b",
     "retrospective_request"),
    (r"\bwhat\s+(?:you|we)\s+(?:discussed|talked\s+about|said|covered|concluded)\b",
     "retrospective_request"),
    (r"\bhow\s+did\s+(?:it|that|the\s+\w+)\s+go\b", "retrospective_request"),
    # explicitly NOT an instruction
    (r"\b(?:don'?t|do not|no need to|you don'?t have to|not asking you to|"
     r"without)\s+(?:go|open|visit|ask|message|search|use)\b", "refusal_to_act"),
    (r"\bif\s+you\s+(?:were|had|could)\s+to?\b|\bhypothetically\b", "hypothetical"),
    # copular claims about the thing
    (r"\b(?:is|are|was|were)\s+(?:a|an|the|just|basically|really|kind of|pretty)\b", "copular_claim"),
)


#: A sentence that opens with a noun phrase rather than a verb, addresses
#: nobody, and asks nothing. "the search for a new apartment has been
#: exhausting" names a capability keyword and requests nothing — the router
#: already carried a bespoke lookahead for exactly this one sentence shape.
#: `[A-Z][a-z]+` used to be an alternative here, and it matched the FIRST WORD
#: OF EVERY SENTENCE — including imperative verbs. "Remember for future sessions
#: that my codename is glass orchard" was read as a declarative statement and
#: stopped routing to memory_ops. A capitalised word at the start of a sentence
#: is evidence of nothing; a determiner or pronoun is.
_NOUN_PHRASE_OPENER_RE = re.compile(
    r"^\s*(?:the|a|an|this|that|these|those|my|our|his|her|their|its|i|we|they|he|she|it)\b",
    re.IGNORECASE,
)
_FINITE_STATEMENT_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|been|feels?|felt|seems?|seemed|"
    r"looks?|looked|means?|meant|gets?|got|becomes?|became)\b",
    re.IGNORECASE,
)


def _is_plain_declarative(text: str) -> bool:
    """A statement about something, addressed to no one, asking for nothing."""
    if "?" in text:
        return False
    if re.search(r"\byou\b|\byour\b|\byourself\b", text, re.IGNORECASE):
        return False
    if not _NOUN_PHRASE_OPENER_RE.match(text):
        return False
    return bool(_FINITE_STATEMENT_RE.search(text))


def assess_request_mood(message: str) -> MoodVerdict:
    """Decide whether the turn instructs an action or talks about something."""

    text = " ".join(str(message or "").split())
    if not text:
        return MoodVerdict(RequestMood.AMBIGUOUS, ("empty",))

    directive_reasons: list[str] = []
    if _IMPERATIVE_RE.search(text):
        directive_reasons.append("imperative_clause")
    if _SECOND_PERSON_REQUEST_RE.search(text):
        directive_reasons.append("second_person_request")

    mention_reasons = [
        name for pattern, name in _MENTION_FRAMES if re.search(pattern, text, re.IGNORECASE)
    ]
    if not mention_reasons and _is_plain_declarative(text):
        mention_reasons.append("declarative_statement")

    # An explicit instruction wins over incidental mention framing: "Ask ChatGPT
    # what it said yesterday" contains reported speech AND is an instruction.
    # The one exception is a frame that exists to CANCEL the action.
    # A retrospective frame cancels an imperative because it directs at what
    # ALREADY HAPPENED: "Remind me what you and ChatGPT discussed" is a request
    # to retrieve, and reading it as an instruction opened a second browser
    # session and held a whole new conversation (measured live 2026-08-04).
    # Plain recollection markers ("yesterday", "last time") do NOT cancel —
    # "Ask ChatGPT what it said yesterday" is still an instruction.
    cancelling = {"refusal_to_act", "hypothetical", "retrospective_request"}
    if directive_reasons and not (set(mention_reasons) & cancelling):
        return MoodVerdict(RequestMood.DIRECTIVE, tuple(directive_reasons))
    if mention_reasons:
        return MoodVerdict(RequestMood.MENTION, tuple(mention_reasons))
    if directive_reasons:
        return MoodVerdict(RequestMood.DIRECTIVE, tuple(directive_reasons))
    return MoodVerdict(RequestMood.AMBIGUOUS, ("no_frame_matched",))


def names_a_thing_without_asking_for_it(message: str) -> bool:
    """True when a turn talks about something rather than asking for it.

    The predicate the capability router needs: naming a tool is not a request
    to use it. Deliberately returns False for AMBIGUOUS — the router's other
    evidence decides those, and this only speaks when the grammar does.
    """
    return assess_request_mood(message).is_about_rather_than_asking


__all__ = [
    "MoodVerdict",
    "RequestMood",
    "assess_request_mood",
    "names_a_thing_without_asking_for_it",
]
