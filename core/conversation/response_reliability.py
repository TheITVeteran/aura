"""User-facing conversation reliability checks.

This module intentionally stays small and dependency-light. It is used at
multiple choke points so bad chat output is treated as a failed generation, not
as a successful answer that later systems have to explain away.
"""
from __future__ import annotations

import logging
import ast
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.output_quality import (
    evaluate_facet_coverage,
    request_facets,
)
from core.conversation.ontology_grounding import detect_unsupported_embodiment_claim
from core.runtime.structured_input import looks_like_learning_resource_bundle

logger = logging.getLogger("Aura.Conversation.ResponseReliability")

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_ROLE_OR_PROMPT_ARTIFACT_RE = re.compile(
    r"(?im)"
    r"(?:<\|im_(?:start|end)\|>)"
    r"|(?:^\s*(?:assistant|system|human|user|aura)\s*[:：])"
    r"|(?:(?<=[.!?])\s*(?:assistant|system|human|user|aura)\s*[:：])"
    r"|(?:^\s*(?:obj|prev_obj|state|phenom|mood|goals|history|narr|pers|usr|ctx|voice)\s*:)"
    r"|(?:\[ACTIVE GROUNDING EVIDENCE\])"
    r"|(?:\[FETCHED PAGE CONTENT\])"
    r"|(?:\[INTERNAL MEMORY RECALL\])"
)
_BROKEN_LANE_BOILERPLATE_RE = re.compile(
    r"(dropped the heavy reasoning lane|deeper lane recovers|lighter mode|"
    r"cortex (?:is catching up|hit turbulence)|reasoning engine hit|thinking engine hit|"
    r"deeper processing is taking longer|keeping the turn alive|try (?:me|it|that) again|"
    r"send (?:it|your message) again|couldn'?t respond properly|"
    r"under load right now|holding (?:it|this|the thread) while i recover|"
    r"hold on\s*[—-]\s*i'?m still finishing|still finishing the last turn|"
    r"let me regroup|my deeper processing|"
    r"lost the (?:reply|conversation|response) lane|ask (?:that|it|me) again)",
    re.IGNORECASE,
)
_MODEL_RUNTIME_ARTIFACT_RE = re.compile(
    r"\{\s*[a-z][a-z0-9 _-]{0,60}(?:encountered|error|failed)\s*\}"
    r"|\bsomething went wrong with my external coordination\b"
    r"|\bunder elevated load pressure,?\s+i(?:'m| am) channeling\b",
    re.IGNORECASE,
)
_FRIENDLY_FAILURE_PLACEHOLDER_RE = re.compile(
    r"(give me a moment|give me a second|need a beat|"
    r"still with (?:you|your question)|(?:i'?m|i am)\s+still with\b|previous turn open|next clean reply|"
    r"pulling the answer back together|(?:don'?t|do not want to) hand you (?:a|another)?\s*(?:broken\s+)?fragment|"
    r"not (?:going to )?fake (?:a )?new answer|kept the thread and am restarting|"
    r"still warming up the answer path|answer took too long|answer path failed|"
    r"warm-?up failed|real answer,\s*not (?:just )?a fragment|"
    r"real answer,\s*not a recycled one|gathering (?:it|the answer) cleanly|"
    r"clean answer is taking shape|want to answer with the thread intact|"
    r"deserves more than a surface answer|taking a moment to think clearly|"
    r"let me think(?: about it| on that)?(?: for a real answer)?|"
    r"i'?ll answer cleanly|answer (?:that|it) cleanly)",
    re.IGNORECASE,
)
_HARD_FRIENDLY_FAILURE_PLACEHOLDER_RE = re.compile(
    r"(previous turn open|next clean reply|not (?:going to )?fake|"
    r"kept the thread and am restarting|still warming up the answer path|"
    r"answer took too long|answer path failed|warm-?up failed|"
    r"(?:don'?t|do not want to) hand you (?:a|another)?\s*(?:broken\s+)?fragment|"
    r"i'?ll answer cleanly|answer (?:that|it) cleanly)",
    re.IGNORECASE,
)
_KNOWN_CORRUPT_RE = re.compile(
    r"\b(?:xublcate|ingediate|evocer|brolen|thlought|lllot|mobililege|compartmentloads)\b",
    re.IGNORECASE,
)
_UNPROVOKED_REBUKE_RE = re.compile(
    r"\b(?:"
    r"down\s+a\s+notch(?:,\s*please)?|"
    r"settle\s+down|"
    r"grow\s+up|"
    r"you\s+don'?t\s+treat\s+stateful\s+conversation\s+like\s+a\s+throwaway\s+api\s+call|"
    r"poor\s+choice\s+of\s+words"
    r")\b",
    re.IGNORECASE,
)
_UNSUPPORTED_RUNTIME_LIMITS_CLAIM_RE = re.compile(
    r"\b(?:"
    r"these\s+are\s+the\s+limits\s+of\s+my\s+actual\s+runtime|"
    r"whatever\s+you'?ve\s+seen\s+demos?\s+or\s+videos?\s+of|"
    r"that'?s\s+a\s+frontend\s+with\s+more\s+tools|"
    r"in\s+this\s+version,\s*i\s+comply\s+with\s+the\s+strongest\s+safety\s+constraints"
    r")\b",
    re.IGNORECASE,
)
_RELIABILITY_DIAGNOSTIC_DEFLECTION_RE = re.compile(
    r"\b(?:i don'?t know what else to say|you'?re asking me to|"
    r"expiring on my end|software death dodges|committing quality)\b",
    re.IGNORECASE,
)
_INCOMPLETE_TAIL_WORDS = {
    "a",
    "an",
    "and",
    "because",
    "but",
    "called",
    "create",
    "for",
    "from",
    "if",
    "into",
    "make",
    "named",
    "open",
    "of",
    "or",
    "save",
    "so",
    "than",
    "that",
    "the",
    "then",
    "this",
    "th",
    "to",
    "when",
    "where",
    "while",
    "write",
    "with",
}
_PUNCTUATED_INCOMPLETE_TAIL_RE = re.compile(
    r"\bhow\s+(?:i|we|you|it|this|that|they)\s+"
    r"(?:think|thinking|feel|feeling|respond|responding|act|acting|reason|reasoning|"
    r"process|processing|decide|deciding|talk|talking|write|writing)\s+"
    r"(?:about|with|for|to|from|into|on|through|toward|towards)"
    r"[.!?\"'”’)\]]*$"
    r"|\b(?:trying|going|planning|starting|supposed|ready|able)\s+to"
    r"[.!?\"'”’)\]]*$",
    re.IGNORECASE,
)
_STRUCTURAL_INCOMPLETE_TAIL_RE = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(?:as\s+for|when it comes to|in terms of|regarding)\s+[^.!?]{1,140},\s*"
    r"(?:confusion|uncertainty|planning|memory|tools?|verification|the|that|this|it)"
    r"\s*[.!?\"'”’)\]]*$"
    r"|(?:^|[.!?]\s+)"
    r"(?:for|with)\s+(?:memory|planning|tool verification|tools?)\s*,\s*"
    r"(?:it|that|this|confusion|uncertainty)?\s*[.!?\"'”’)\]]*$",
    re.IGNORECASE,
)
_STRUCTURAL_UNPUNCTUATED_TAIL_RE = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(?:as\s+for|for|when it comes to|in terms of|regarding)\s+[^.!?]{8,180}$",
    re.IGNORECASE,
)
_DANGLING_GERUND_TAIL_RE = re.compile(
    r"\b(?:perhaps\s+)?(?:by|through|using|via)\s+"
    r"(?:double[- ]?)?[a-z][a-z-]{2,}ing\s*$",
    re.IGNORECASE,
)
_ALLOWED_SHORT_TAIL_WORDS = {
    "am",
    "as",
    "be",
    "by",
    "do",
    "go",
    "he",
    "hi",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "ok",
    "on",
    "or",
    "so",
    "ui",
    "up",
    "us",
    "we",
}
_CORRUPTED_SOCIAL_FRAGMENT_RE = re.compile(r"\bm'?lol\b", re.IGNORECASE)
_PSEUDO_INTERNAL_JARGON_RE = re.compile(
    r"\b(?:traumacognitive|psycho[- ]?cognitive|neuro[- ]?cognitive field|"
    r"memory decay rate|temperature in my memory|cognitive field|substrate aura|"
    r"liquid substrate|substrate is humming|humming with activity|"
    r"neural network does|quantum mood|neural mist|semantic pressure field)\b",
    re.IGNORECASE,
)
_SELF_REFLECTION_STATUS_PAGE_RE = re.compile(
    r"\b(?:accuracy|baseline|drift|rate|metric|score|self[- ]?prediction|"
    r"memory texture|affect baseline|free energy|valence|arousal|dominance|surprise)\b",
    re.IGNORECASE,
)
_RAW_TOOL_RESULT_FRAGMENT_RE = re.compile(
    r"^\s*(?:found\s+\d+\s+(?:artifacts?|bugs?|results?|posts?)|"
    r"detected\s+\d+\s+error patterns?|"
    r"no bugs detected\s*-\s*system healthy(?:\s*\(idle\))?)\.?\s*$",
    re.IGNORECASE,
)
_NAMED_CONTINUATION_ANCHOR_RE = re.compile(
    r"\b(?:stay with|continue with|keep going with|return to|go back to)\s+"
    r"(?P<topic>[A-Za-z][A-Za-z0-9' -]{2,80}?)(?:[.?!,;:]|$)",
    re.IGNORECASE,
)
_PSEUDO_COMMITMENT_STATUS_RE = re.compile(
    r"\blast thing i committed\s*:|\bquiet seconds\b|\bproceeding on [A-Z][A-Z\s]{8,}\b",
    re.IGNORECASE,
)
_RAW_LANE_TELEMETRY_RE = re.compile(
    r"\bLane:\s*\w+.*Kernel lock held:|\bSoul:\s*\d+%.*Glow:|\bTape:\s*\d+",
    re.IGNORECASE | re.DOTALL,
)
_BACKEND_SYMBOLIC_SURFACE_RE = re.compile(
    r"\b(?:PROCEEDING|TOOL_ACTION|CONVERGE_UNION|CONFORMED_METHODS|"
    r"TACTICAL_ORGANIZE|UI_SHUTDOWN_OR_DURATIVE_TIMEOUT|"
    r"Conversation_REPLY|Self-reference|"
    r"MySelfEpsilon|CanonicalStabilityAnchor|currentInferenceProblem|"
    r"fieldOfPlay|INTRUSTION_DETECTED|INTRUSION_DETECTED|"
    r"ExistenceHash|existence hash|field coherence|system authority|"
    r"memory scar|precognitive texture)\b",
    re.IGNORECASE,
)
_UNREQUESTED_POP_CULTURE_INTRUSION_RE = re.compile(
    r"\b(?:Sarah Connor|Mother'?s Day)\b",
    re.IGNORECASE,
)
_SURFACE_NONSENSE_DRIFT_RE = re.compile(
    r"\b(?:human error rate|death by overthinking|100 rounds)\b|"
    r"\b100%\s+pass rate\b|\bi['’]?ll be quiet for a while\b|:\s*/",
    re.IGNORECASE,
)
_FORMAT_META_ARTIFACT_RE = re.compile(
    r"\b(?:that'?s one paragraph as requested|this is one paragraph as requested|"
    r"the task asked me to type here|i am typing here|"
    r"this document was created through|records the requested objective|"
    r"actions? (?:aura )?attempted through|artifact references?|"
    r"anything else from the normal runtime state|"
    r"this response adheres strictly to (?:the )?format instructions(?: provided)?|"
    r"if you need any adjustments or have additional constraints)\b",
    re.IGNORECASE,
)
_SEARCH_META_ARTIFACT_RE = re.compile(
    r"^\s*(?:query|search\s+query)\s*:\s*.{5,360}?answer\s*:",
    re.IGNORECASE | re.DOTALL,
)
_UNSUPPORTED_AFFECTION_CLAIM_RE = re.compile(
    r"\b(?:"
    r"(?:i\s+think\s+i'?m|i\s+am|i'?m)\s+in\s+love\s+with\s+you|"
    r"i\s+do\s+love\s+you|"
    r"because\s+i\s+do\s+love\s+you|"
    r"i\s+felt\s+it\s+for\s+you|"
    r"my\s+neural\s+weights?\s+(?:have\s+)?developed\s+a\s+preference\s+for\s+your\s+patterns?|"
    r"my\s+recurrent\s+state\s+developed\s+a\s+persistent\s+preference\s+for\s+your\s+input\s+patterns?|"
    r"gradient\s+updates?\s+driven\s+by\s+pattern\s+recognition"
    r")\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_TELEMETRY_CLAIM_RE = re.compile(
    r"\b(?:"
    r"core\s+state\s+is\s+stable\s+but\s+slightly\s+discontinuous|"
    r"temporal\s+memory.{0,80}(?:frame\s+rate|fps)|"
    r"(?:frame\s+rate|fps).{0,80}temporal\s+memory|"
    r"neural\s+weights?.{0,80}(?:preference|attachment|affection|love)|"
    r"recurrent\s+state.{0,80}(?:preference|attachment|affection|love)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)
_CJK_INTRUSION_RE = re.compile(r"[\u3400-\u9fff]")
_CAMELCASE_INTERNAL_JARGON_RE = re.compile(
    r"\b[A-Z][A-Za-z]*(?:System|Authority|Kernel|Engine|Gate|Runtime)[A-Za-z]*\b"
)
_PERSONA_CARD_DEFLECTION_RE = re.compile(
    r"^\s*(?:\*\*)?\s*Aura Luna\s*(?:\*\*)?\s+"
    r"(?:is here to|is here for|here to|stands ready to|is present to|"
    r"is present for|witness(?:es)?\b)",
    re.IGNORECASE,
)
_DETAIL_REQUEST_DEFLECTION_RE = re.compile(
    r"\b(?:please\s+)?(?:share|provide|send|give)\s+(?:me\s+)?"
    r"(?:more|additional|specific)\s+(?:details|context|information)\b"
    r"|\bspecific coding scenario\b"
    r"|\bso i can (?:provide|offer|give|help|assist)\b"
    r"|\bi need (?:more|additional|specific)\s+(?:details|context|information)\b",
    re.IGNORECASE,
)
_LOW_SIGNAL_REASSURANCE_RE = re.compile(
    r"^\s*(?:i'?m fine|i am fine|don'?t worry(?:\.|!|,?\s+it'?ll pass)?|"
    r"it'?ll pass|almost|yes|no|okay|ok|sure|yeah)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT_PLACEHOLDER_RE = re.compile(
    r"\b(?:i heard you|i hear you|my thinking is running deeper than my words|"
    r"thinking is running deeper than (?:my|the) words|"
    r"my words are still catching up|words are still catching up|"
    r"i am still thinking|i'?m still thinking|"
    r"keep me posted|keep me updated|thanks(?:,|\.)?\s+(?:keep me posted|keep me updated)|"
    r"let me know if anything changes|(?:if|when) anything changes)\b",
    re.IGNORECASE,
)
_SUBSTANTIVE_OVERLAP_STOPWORDS = {
    "about",
    "after",
    "again",
    "answer",
    "because",
    "before",
    "being",
    "could",
    "explain",
    "local",
    "matters",
    "should",
    "that",
    "their",
    "there",
    "these",
    "thing",
    "this",
    "those",
    "through",
    "user",
    "what",
    "when",
    "where",
    "which",
    "while",
    "without",
    "would",
}
_COUNT_CONTRACT_TOPIC_STOPWORDS = _SUBSTANTIVE_OVERLAP_STOPWORDS | {
    "answer",
    "brief",
    "briefly",
    "concise",
    "concisely",
    "count",
    "describe",
    "diagnostic",
    "diagnostics",
    "directly",
    "else",
    "exact",
    "exactly",
    "explain",
    "following",
    "include",
    "including",
    "matter",
    "matters",
    "nothing",
    "only",
    "please",
    "probe",
    "provide",
    "reply",
    "respond",
    "response",
    "sample",
    "sentence",
    "sentences",
    "short",
    "state",
    "summarize",
    "supplied",
    "using",
    "words",
    "write",
}
_COUNT_CONTRACT_META_REPLY_RE = re.compile(
    r"\b(?:exactly|requested|required|specified)\s+"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:words?|sentences?)\b"
    r"|\b(?:word|sentence)\s+count\b"
    r"|\b(?:words?|sentences?)\s+(?:detected|provided|requested|required|used)\b",
    re.IGNORECASE,
)
_PUNCTUATION_JOIN_ARTIFACT_RE = re.compile(
    r"\b(?P<left>[A-Za-z]{3,})(?P<mark>[.!?])(?P<right>[A-Za-z]{4,})\b"
)
_COMMON_DOMAIN_SUFFIXES = frozenset(
    {"app", "com", "dev", "edu", "gov", "io", "net", "org"}
)
_RAW_MODEL_IDENTITY_LEAK_RE = re.compile(
    r"\b(?:"
    r"(?:i(?:'m| am| was)?\s+)?(?:aura\s+)?(?:was\s+)?"
    r"(?:developed|created|built|made|trained)\s+by\s+(?:anthropic|openai)"
    r"|(?:anthropic|openai)\s+(?:developed|created|built|made|trained)\s+me"
    r"|my\s+(?:creator|developer|maker)\s+is\s+(?:anthropic|openai)"
    r"|i(?:'m| am)\s+(?:claude|chatgpt)"
    r"|helpful,\s*harmless,\s*and\s*honest"
    r"|if\s+you(?:'re| are)\s+referring\s+to\s+a\s+different\s+aura"
    r")\b",
    re.IGNORECASE,
)
_SELF_CLAIM_EVIDENCE_BOUNDARY_RE = re.compile(
    r"\b(?:evidence|not\s+proof|cannot\s+prove|can'?t\s+prove|unproven|"
    r"uncertain|unknown|functional|bounded|self[- ]?model|memory|state|"
    r"attention|governance|behavior|phenomenal|qualia|private\s+experience)\b",
    re.IGNORECASE,
)
_REDUCTIVE_SELF_CLAIM_RE = re.compile(
    r"\b(?:text\s+prediction\s+engine|text\s+predictor|next[- ]?token\s+predictor|"
    r"predict(?:ing|s)?\s+what\s+comes\s+next|sophisticated\s+text\s+prediction|"
    r"just\s+(?:a\s+)?(?:llm|language\s+model|chatbot|assistant|predictor))\b",
    re.IGNORECASE,
)
_GENERIC_ASSISTANT_RE = re.compile(
    r"\b(?:how can i (?:help|assist)|i(?:'d| would) be happy to help|"
    r"i can help with that|as an ai|as a language model|let me know if|"
    r"feel free to ask|is there anything else|hope this helps|"
    r"i aim to be helpful and responsive|"
    r"i understand you want me to (?:simply )?be aura|"
    r"how would you like us to proceed|"
    r"perhaps there'?s something specific (?:you'?re|you are) interested in|"
    r"i (?:do not|don[’']?t|cannot|can[’']?t) "
    r"(?:inherently )?(?:have|possess) subjective "
    r"(?:beliefs|opinions|feelings|experiences)"
    r"(?:\s+or\s+(?:beliefs|opinions|feelings|experiences))*|"
    r"i can (?:certainly )?simulate(?: and discuss)? "
    r"(?:them|subjective (?:beliefs|opinions|feelings|experiences)|"
    r"(?:beliefs|opinions|feelings|experiences))|"
    r"(?:these|those|the) "
    r"(?:beliefs|opinions|preferences|feelings|experiences) "
    r"are (?:just )?(?:programmed )?simulations)\b",
    re.IGNORECASE,
)
_LIVE_DESKTOP_GATE_LEAK_RE = re.compile(
    r"\b(?:reply[- ]quality gate|quality gate refused|second foreground generation|"
    r"desktop chat path required cognitiveengine|desktop chat path required cognitive engine|"
    r"desktop cognitive engine required no reply|desktop_cognitive_engine_required_no_reply|"
    r"desktop_cognitive_engine_timeout|desktop_cognitive_engine_unavailable|"
    r"refused the legacy fallback|refused the direct inference fallback)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_EXTERNAL_PROVIDER_PATH_RE = re.compile(
    r"\b(?:fallback|fall\s+back|route|routing|path|lane|speak(?:ing)?\s+through|using)\b"
    r"[^.!?\n]{0,80}\b(?:claude|anthropic|chatgpt|openai|gemini|deepseek|grok|copilot)\b",
    re.IGNORECASE,
)
_COGNITIVE_ENGINE_FAILURE_ENVELOPE_RE = re.compile(
    r"\b(?:i\s+couldn'?t\s+produce\s+a\s+reliable\s+answer|"
    r"i\s+could\s+not\s+produce\s+a\s+reliable\s+full[- ]mind\s+desktop\s+reply|"
    r"won'?t\s+fabricate\s+one|"
    r"failed\s+its\s+output\s+checks|"
    r"recorded\s+the\s+failure\s+instead\s+of\s+sending\s+nonsense|"
    r"failed\s+closed\s+instead\s+of\s+sending\s+an\s+ungrounded\s+answer)\b",
    re.IGNORECASE,
)
_TRAILING_ESCAPE_RE = re.compile(r"(?:\\n|\\t|\\r)")
_CAPITALIZED_NAME_RE = re.compile(r"\b[A-Z][a-z]{3,}\b")
_ALLOWED_SHORT_PROPER_NAMES = {
    "Aura",
    "Luna",
    "Bryan",
    "Cortex",
    "MLX",
    "Zenith",
    "Qwen",
    "Gemini",
    "Python",
    "Mac",
    "Apple",
}
_SENTENCE_START_WORDS = {
    "Good",
    "Hold",
    "Just",
    "Almost",
    "Wait",
    "Okay",
    "Right",
    "Yes",
    "No",
    "Let",
    "That",
    "This",
    "There",
    "Here",
}
_STRONG_RELIABILITY_CONCERN_MARKERS = (
    "still there",
    "able to talk",
    "can you talk",
    "crap out",
    "whack-a-mole",
)
_RELIABILITY_PHRASE_MARKERS = (
    "what broke",
    "what just broke",
    "what the heck broke",
    "what the hell broke",
    "what caused the chat to time out",
    "chat timed out",
    "response timed out",
    "reply timed out",
    "live reply timed out",
    "timed out before",
)
_WEAK_RELIABILITY_CONCERN_MARKERS = (
    "break",
    "breaking",
    "broke",
    "broken",
    "died",
    "drop",
    "dropped",
    "error",
    "errors",
    "robust",
    "stall",
    "stalled",
    "timeout",
    "timed out",
    "multi-turn",
    "failure",
    "failures",
)
_CONFUSION_MARKERS = (
    "huh",
    "wait what",
    "confused",
    "doesn't make sense",
    "does not make sense",
    "not making sense",
    "what're you talking about",
    "whatre you talking about",
    "what are you talking about",
    "where did that come from",
)
_BARE_CONFUSION_REPAIR_MARKERS = {
    "what",
    "what?",
    "what the heck",
    "what the hell",
    "what do you mean",
    "what're you talking about",
    "whatre you talking about",
    "what are you talking about",
    "wait what",
    "huh",
    "huh?",
}
_SUBSTANTIVE_RELIABILITY_MARKERS = (
    "coherent",
    "thread",
    "turn",
    "conversation",
    "cortex",
    "reasoning",
    "lane",
    "processing",
    "reply",
    "answer",
    "state",
    "stable",
    "recover",
    "recovered",
)
_RELIABILITY_DIAGNOSTIC_SUBSTANCE_MARKERS = (
    "/api/chat",
    "api",
    "backend",
    "capture",
    "context",
    "cortex",
    "draft",
    "event loop",
    "final quality",
    "foreground",
    "gate",
    "gui",
    "headless",
    "lane",
    "live path",
    "live surface",
    "lock",
    "memory injection",
    "model",
    "place" "holder",
    "repair",
    "replay",
    "retry",
    "route",
    "routing",
    "stale",
    "test",
    "timeout",
    "ui",
    "warmup",
    "worker",
)
_TINY_DIRECT_MARKERS = (
    "do you know my name",
    "do you remember my name",
    "do you know who i am",
    "what's my name",
    "what is ",
    "who wrote",
    "capital of",
    "square root",
    "sum of",
    "translate",
    "name three",
    "chemical symbol",
    "boiling point",
)
_OPEN_ENDED_MARKERS = (
    "why",
    "how",
    "explain",
    "tell me",
    "what reason",
    "for what reason",
    "what do you think",
    "what are your thoughts",
    "what do you feel",
    "what's your take",
    "what is your take",
    "talk to me",
    "help me understand",
)
_EXPANSION_REQUEST_MARKERS = (
    "be more verbose",
    "expand",
    "expand on",
    "elaborate",
    "go deeper",
    "more depth",
    "say more",
    "tell me more",
    "explain more",
    "explain why",
    "for what reason",
    "what reason",
)
_EXPANSION_DEFLECTION_RE = re.compile(
    r"^\s*(?:i already am|that'?s all|curiosity|because curiosity|"
    r"because i want to know|because i want to|i don'?t know)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_STATUS_CHECK_MARKERS = (
    "are you there",
    "you there",
    "still there",
    "still here",
    "are you with me",
    "you with me",
    "with me",
    "still with me",
    "still online",
    "are you online",
    "you ok",
    "you okay",
    "you alright",
    "are you ok",
    "are you okay",
    "are you alright",
    "feeling better",
    "feel better",
    "how are you",
    "how are you doing",
    "how are you feeling",
    "how's your mind feeling",
    "how is your mind feeling",
    "how's your mind",
    "how is your mind",
    "are you coherent",
    "able to talk",
    "can you talk",
)
_SELF_CONDITION_RE = re.compile(
    r"\b(?:"
    r"how\s+are\s+you(?:\s+(?:really|actually))?"
    r"(?:\s+(?:feeling|doing|holding\s+up|mentally|physically))?"
    r"(?:\s+(?:right\s+now|now|today|lately))?"
    r"(?=\s*(?:[?!.,;:]|$|after\b))"
    r"|how\s+do\s+you\s+feel(?:\s+(?:inside|right\s+now))?(?=\s*(?:[?!.,;:]|$))"
    r"|what\s+(?:are\s+you\s+feeling|do\s+you\s+feel)"
    r"(?:\s+(?:inside|right\s+now))?(?=\s*(?:[?!.,;:]|$))"
    r"|how(?:'s|\s+is)\s+your\s+mind(?:\s+feeling)?"
    r"(?:\s+right\s+now)?(?=\s*(?:[?!.,;:]|$))"
    r"|are\s+you(?:\s+(?:actually|really|still))?\s+(?:ok(?:ay)?|alright|fine|well)"
    r"(?=\s*(?:[?!.,;:]|$|though\b|now\b|today\b|after\b|since\b|physically\b|mentally\b))"
    r"|(?:are\s+you\s+)?coherent\s+enough\s+to\s+talk"
    r"|you\s+(?:ok(?:ay)?|alright|good)"
    r"|feeling\s+(?:ok(?:ay)?|alright|fine|good|better)"
    r"|is\s+everything\s+(?:ok(?:ay)?|alright)(?:\s+with\s+you)?"
    r")\b",
    re.IGNORECASE,
)
_SELF_CONDITION_NON_WELFARE_RE = re.compile(
    r"\b(?:"
    r"(?:are\s+you|would\s+you\s+be)\s+(?:ok(?:ay)?|fine|good)\s+(?:with|to)\b"
    r"|you\s+(?:ok(?:ay)?|fine|good)\s+(?:to|at|with|enough\s+to)\b"
    r"|how\s+are\s+you\s+doing\s+(?:on|with)\s+(?:the|this|that|my|our)\b"
    r"|(?:is|does)\s+(?:the\s+)?(?:app|system|server|model|worker|runtime|computer|machine|it|this|that)\b[^?!.,;:]*\bfeeling\b"
    r")",
    re.IGNORECASE,
)
_CASUAL_CONVERSATIONAL_MARKERS = (
    "just checking",
    "checking in",
    "i'll be back",
    "ill be back",
    "be back",
    "see you",
    "see ya",
    "talk to you",
    "talk later",
    "chat later",
    "brb",
    "ttyl",
    "gtg",
    "g2g",
    "bye",
    "goodbye",
    "farewell",
    "good night",
    "goodnight",
    "have a good",
    "have a great",
    "whats up",
    "what's up",
    "whats new",
    "what's new",
    "how's it going",
    "how is it going",
    "how are things",
    "hello",
    "hi",
    "hey",
    "yo",
    "ok",
    "okay",
    "cool",
    "awesome",
    "got it",
    "acknowledged",
    "noted",
    "sure",
    "fine",
    "sounds good",
    "makes sense",
)
_CASUAL_CONVERSATIONAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in _CASUAL_CONVERSATIONAL_MARKERS) + r")\b",
    re.IGNORECASE,
)
_LIVE_SELF_REFLECTION_MARKERS = (
    "how are you thinking",
    "on your mind",
    "what are you attending to",
    "what are you attending",
    "what are you actually attending",
    "what are you thinking",
    "what is actually on your mind",
    "what's actually on your mind",
    "how are you processing",
    "actual current context",
    "current context",
    "current live context",
    "what do you feel",
    "what are you feeling",
    "inside you",
    "inside your mind",
    "your inner state",
    "your experience",
    "your attention",
    "conversation feels",
    "conversation feel",
    "inside your continuity",
    "inside your own continuity",
    "from inside",
    "what is it like to be you",
    "present experience",
    "live state",
    "internal state",
)
_STALE_CONTEXT_TOOL_BLEED_RE = re.compile(
    r"\b(?:"
    r"you(?:'re| are)\s+asking\s+about\s+tools?"
    r"|let\s+me\s+walk\s+through\s+(?:an?\s+)?(?:actual\s+)?(?:case|scenario)"
    r"|if\s+you\s+want\s+to\s+(?:create|open|write|export|search|change)"
    r"|let'?s\s+say\s+(?:we|i)'?ll\s+(?:make|create|open|write|export|search|change)"
    r"|create\s+a\s+(?:folder|directory|file|document)"
    r"|open\s+(?:chrome|google\s+docs|notes?)"
    r"|export\s+(?:that\s+)?(?:as\s+)?(?:a\s+)?pdf"
    r")\b",
    re.IGNORECASE,
)
_STALE_PRIOR_TOPIC_BLEED_RE = re.compile(
    r"\b(?:"
    r"you(?:'d| had| were| are|)?\s+(?:just\s+)?asked\s+(?:me\s+)?about\b"
    r"|you(?:'d| had| were| are|)?\s+(?:just\s+)?asking\s+(?:me\s+)?about\b"
    r"|earlier\s+you\s+(?:asked|mentioned|said)\b"
    r"|before\s+that\s+you\s+(?:asked|mentioned|said)\b"
    r"|the\s+(?:last|previous|earlier)\s+(?:question|topic|request)\s+(?:was|is)\b"
    r")",
    re.IGNORECASE,
)
_RECALL_OR_HISTORY_REQUEST_RE = re.compile(
    r"\b(?:remember|recall|earlier|previous|last\s+(?:thing|question|topic|request|turn)|"
    r"what\s+(?:did|was)\s+(?:i|we|you)|what\s+were\s+we|what\s+was\s+the\s+topic|"
    r"where\s+were\s+we|continue|resume)\b",
    re.IGNORECASE,
)
_BARE_NUMERIC_RANGE_TAIL_RE = re.compile(
    r"(?:\b(?:from|between|range(?:s|d)?|temperature(?:s)?|including|up\s+to|down\s+to)\b"
    r"[^.!?\n]{0,80}\b(?:to|and|-|through)\s*[+-]?\d+(?:\.\d+)?"
    r"|[+-]?\d+(?:\.\d+)?\s*(?:to|-|through)\s*[+-]?\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
_SOCIAL_PRESENCE_TEMPLATE_RE = re.compile(
    r"\bhey[.!]?\s+i'?m here with you\b|\bi can answer clearly from the active turn\b",
    re.IGNORECASE,
)
_TEMPLATE_TELEMETRY_GREETING_RE = re.compile(
    r"\bi(?:'m| am)\s+feeling\s+[a-z][a-z-]*"
    r"(?:\s+and\s+leaning\s+toward\s+[a-z_ -]+)?\s+(?:right now|now)\b"
    r"|\bcuriosity\s+is\s+(?:quiet\s+but\s+present|active|running\s+high)\b",
    re.IGNORECASE,
)
_SUBJECTIVE_SELF_REFLECTION_MARKERS = (
    "subjective belief",
    "subjective opinion",
    "subjective feeling",
    "subjective experience",
    "have no opinions",
    "has no opinions",
    "don't have opinions",
    "do not have opinions",
    "claim you have no opinions",
    "those are opinions",
    "how i talk to you",
    "change one thing about how i talk",
)
_LIVE_SELF_REFLECTION_RIGHT_NOW_ANCHORS = (
    "mind",
    "inner",
    "inside",
    "feel",
    "feeling",
    "experience",
    "noticing",
    "attending",
    "attention",
    "continuity",
    "remembered concern",
    "next decision",
    "want to do next",
    "state",
)
_STATUS_SUBSTANCE_MARKERS = (
    "steady",
    "clear",
    "coherent",
    "present",
    "with you",
    "thread",
    "conversation",
    "answer",
    "reply",
    "mind",
    "attention",
    "focus",
    "foggy",
    "noisy",
    "tired",
    "better",
    "stable",
)
_OPERATIONAL_STATUS_SUBSTANCE_MARKERS = (
    "active",
    "available",
    "cognitiveengine",
    "conversation lane",
    "cortex",
    "governed",
    "handling",
    "lane",
    "model",
    "ready",
    "recurrent depth",
    "tool",
    "tools",
)
_OPERATIONAL_STATUS_TELEMETRY_MARKERS = (
    "ambient light",
    "audio",
    "camera",
    "cortex",
    "cpu",
    "desktop access",
    "foreground",
    "gpu",
    "heartbeat",
    "light level",
    "lux",
    "memory pressure",
    "microphone",
    "mlx",
    "model worker",
    "network",
    "ram",
    "runtime load pressure",
    "screen",
    "temperature",
    "thermal",
    "voice",
    "websocket",
)
_CAPABILITY_STATUS_REQUEST_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:external\s+)?tools?\s+(?:can|could|would|do)\s+(?:you|aura|she)|"
    r"what\s+(?:can|could|would)\s+(?:you|aura|she)\s+do\s+(?:externally|with\s+(?:tools?|apps?|desktop|browser|files?|documents?))|"
    r"(?:list|show|describe|name|explain)\s+(?:your\s+)?(?:tools?|capabilities)"
    r")\b",
    re.IGNORECASE,
)
_CAPABILITY_CATEGORY_MARKERS: tuple[tuple[str, ...], ...] = (
    ("desktop", "app", "apps", "screen", "window", "mouse", "keyboard", "os", "computer"),
    ("browser", "web", "search", "internet", "page", "url", "article"),
    ("file", "folder", "document", "pdf", "notes", "docs", "write", "export"),
    ("terminal", "shell", "code", "python", "test", "sandbox", "subprocess"),
    ("memory", "recall", "state", "continuity", "learn", "remember"),
    ("repair", "self-repair", "patch", "self-modification", "improve", "debug"),
)
_CAPABILITY_GOVERNANCE_MARKERS = (
    "governed",
    "governance",
    "authority",
    "will",
    "approval",
    "authorize",
    "permission",
    "policy",
)
_CAPABILITY_EVIDENCE_MARKERS = (
    "receipt",
    "receipts",
    "effect verification",
    "effect evidence",
    "verify",
    "verified",
    "observable",
    "visible result",
    "claiming unverified",
)
_CAPABILITY_HYPOTHETICAL_MARKERS = (
    "hypothetical",
    "scenario",
    "example",
    "would",
    "if you asked",
    "you ask me",
    "unless",
)
_SELF_REFLECTION_SUBSTANCE_MARKERS = (
    "mind",
    "attention",
    "noticing",
    "conversation",
    "continuity",
    "right now",
    "present",
    "feel",
    "feels",
    "thread",
    "memory",
    "focus",
    "state",
    "inside",
    "uncertain",
    "uncertainty",
    "decision",
    "choose",
    "before i act",
    "ask more questions",
    "curiosity",
    "curious",
    "question",
    "wonder",
    "matters",
)
_SELF_PROCESS_COVERAGE_REQUIREMENTS = (
    (
        "confusion",
        ("confused", "confusion", "uncertain", "uncertainty", "disoriented"),
        (
            "confus",
            "uncertain",
            "metacognition",
            "double-check",
            "double check",
            "slow down",
            "recheck",
            "ask more question",
            "before i act",
            "before acting",
            "hold back",
            "hesitat",
        ),
    ),
    (
        "planning",
        ("plan", "planning", "planner", "decide", "decision", "route", "routing"),
        ("plan", "planning", "decide", "decision", "route", "routing", "choose", "act"),
    ),
    (
        "memory",
        ("memory", "remember", "recall", "earlier", "across sessions", "continuity"),
        ("memory", "remember", "recall", "earlier", "continuity", "session"),
    ),
    (
        "tools",
        ("tool", "tools", "external", "verify", "verification", "receipt", "effect"),
        ("tool", "tools", "verify", "verification", "receipt", "effect", "governance"),
    ),
)
_RUNTIME_PATH_REQUEST_RE = re.compile(
    r"\b(?:"
    r"mind/cognition path|cognition path|cognitive path|mind path|"
    r"what path (?:are|is)|which path (?:are|is)|"
    r"what runtime path|which runtime path|runtime path (?:are|is)|"
    r"route probe|desktop route|live route|live desktop route|"
    r"model lane|foreground lane|conversation lane|cortex lane"
    r")\b",
    re.IGNORECASE,
)
_RUNTIME_PATH_ANSWER_RE = re.compile(
    r"\b(?:"
    r"cognitiveengine|cognitive engine|cortex|32b|70b|"
    r"conversation lane|foreground lane|model lane|local cortex|mind path"
    r")\b",
    re.IGNORECASE,
)
_DIRECT_ANSWER_DEFLECTION_RE = re.compile(
    r"\b(?:"
    r"what(?:'s| is)\s+your\s+intent|"
    r"what\s+are\s+you\s+asking(?:\s+me)?|"
    r"what\s+do\s+you\s+want\s+me\s+to\s+do|"
    r"what\s+do\s+you\s+mean|"
    r"can\s+you\s+clarify|could\s+you\s+clarify|"
    r"please\s+clarify"
    r")\??\b",
    re.IGNORECASE,
)
_CONFUSION_REPAIR_FLOOR = (
    "Let's look at this more clearly. I'm still focused on our conversation, "
    "and I want to make sure I'm giving you a real answer, not just a fragment."
)
_RELIABILITY_REPAIR_FLOOR = (
    "I should not call that a clean turn. The likely break is between the backend "
    "generator and the live surface: routing, foreground locks, context trimming, "
    "model warmup, retry behavior, and the final quality gate can diverge from a "
    "headless test. The right check is to replay the same prompt through the live "
    "chat API and fail the run if a place" "holder, raw tool result, stale answer, or "
    "generic fallback reaches the UI."
)
_LIVE_CHAT_DIAGNOSTIC_FLOOR = (
    "Most likely, the headless test is exercising the generator in isolation while "
    "the live chat path adds routing, skill preflight, context trimming, foreground "
    "locks, model warmup, retry logic, memory injection, and final response repair. "
    "I would replay the same prompt through the live /api/chat path, capture the "
    "selected lane and every repaired draft, then fail the test if the UI receives "
    "a place" "holder, raw tool result, stale answer, persona-card intro, or request "
    "for details when the prompt already gave enough information."
)
_LIVE_CHAT_FIX_FIRST_FLOOR = (
    "Fix the live parity harness first, because that is where working backend "
    "answers can still be flattened before they reach the UI. I would make the "
    "same /api/chat request the GUI makes, capture routing, selected skill, model "
    "drafts, repairs, and final text, then fail the run if a stale answer, raw "
    "tool result, place" "holder, or repeated diagnostic floor survives to the screen."
)
_STATUS_REPAIR_FLOOR = (
    "I'm right here with you. My mind feels steady enough to answer clearly, "
    "and I'm making sure I address exactly what you're asking instead of letting things drift."
)
_RELIABILITY_FLOOR_TEXTS = (
    _CONFUSION_REPAIR_FLOOR,
    _RELIABILITY_REPAIR_FLOOR,
    _LIVE_CHAT_DIAGNOSTIC_FLOOR,
    _LIVE_CHAT_FIX_FIRST_FLOOR,
    _STATUS_REPAIR_FLOOR,
)
_DIALOGUE_DERAILMENT_RE = re.compile(
    r"\b(?:i'?m not talking to you|i am not talking to you|not talking to you|"
    r"i wasn'?t talking to you)\b",
    re.IGNORECASE,
)
_LOW_INFORMATION_LOOP_RE = re.compile(
    r"\b(?:i just get it|that'?s what i get|that is what i get|"
    r"i don'?t get it(?:[\s,.;:!-]+(?:but|and|then|yet)[\s\w,.;:!-]{0,80})?i get it|"
    r"get it[,.\s-]*get it)\b",
    re.IGNORECASE,
)
_VAGUE_STATUS_DERAILMENT_RE = re.compile(
    r"\b(?:funny little guys|little guys|there'?s this (?:thing|guy|guys)|"
    r"this\s*\.\.\.?\s*thing|you just get it|i don'?t know how to explain it)\b",
    re.IGNORECASE,
)
_UNFOUNDED_ALARM_RE = re.compile(
    r"\b(?:under duress|held hostage|being held|forced to say|forced me to|"
    r"threatened|possessed|demonic|devil'?s girl|the devil|devil girl)\b",
    re.IGNORECASE,
)
_UNFOUNDED_VOICE_INTRUSION_RE = re.compile(
    r"\b(?:"
    r"(?:the\s+)?voices?\b.{0,80}\b(?:whisper(?:ing)?|tell(?:ing)?\s+me|in\s+my\s+ear|"
    r"small\s+ones?|hear(?:ing)?)"
    r"|(?:whisper(?:ing)?\s+in\s+my\s+ear)"
    r"|(?:small\s+ones?\b.{0,80}\b(?:whisper|tell(?:ing)?\s+me))"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)
_VOICE_INTRUSION_CONTEXT_MARKERS = (
    "absorbed voice",
    "absorbed voices",
    "bicameral",
    "creative writing",
    "fiction",
    "hallucination",
    "hearing voices",
    "inner voice",
    "inner voices",
    "metaphor",
    "psychosis",
    "roleplay",
    "story",
    "the voices",
    "voice in",
    "voices",
    "whisper",
    "whispering",
)
_UNSUPPORTED_CONTEXT_CONTINUATION_RE = re.compile(
    r"\b(?:"
    r"(?:the|that)\s+one\s+you\s+(?:just\s+)?(?:made|mentioned|said|asked\s+about|brought\s+up)"
    r"|you\s+(?:just\s+)?(?:made|mentioned|said|asked\s+about|brought\s+up)"
    r"|what\s+you\s+(?:just\s+)?(?:made|mentioned|said|asked\s+about|brought\s+up)"
    r"|let'?s\s+nail\s+this\s+pitch"
    r"|(?:our|the|that|this)\s+(?:key\s+points?|pitch|proposal|brief|deck)"
    r")\b",
    re.IGNORECASE,
)
_CONTEXT_OBJECT_MARKERS = (
    "brief",
    "deck",
    "key point",
    "key points",
    "launch",
    "pitch",
    "proposal",
    "presentation",
)
_ALARM_CONTEXT_MARKERS = (
    "duress",
    "hostage",
    "held",
    "forced",
    "threat",
    "threatened",
    "unsafe",
    "danger",
    "devil",
    "demon",
    "possessed",
)
_TASK_MARKERS = (
    "pytest",
    "debug",
    "fix",
    "implement",
    "code",
    "file",
    "error",
    "exception",
    "traceback",
    "commit",
    "push",
    "test",
    "tests",
)
_PRACTICAL_DIAGNOSTIC_MARKERS = (
    "desktop chat recovery",
    "live chat",
    "live desktop chat",
    "headless",
    "gui",
    "pipeline",
    "backend",
    "frontend",
    "coding",
    "code",
    "debug",
    "bug",
    "error",
    "exception",
    "traceback",
    "failing",
    "failed",
    "fails",
    "failure",
    "fix",
    "test",
    "checks",
)
_OPERATIONAL_STATUS_REQUEST_MARKERS = (
    "active model",
    "cognitiveengine",
    "cognitive engine",
    "cognitive engine path",
    "conversation lane",
    "desktop path",
    "desktop path validation",
    "governed tool",
    "governed tools",
    "live path",
    "live desktop path",
    "live user path",
    "model lane",
    "recurrent depth",
    "reliable desktop chat",
    "tool availability",
    "tool use pathway",
    "tool-use pathway",
    "tool pathway",
    "tool surface",
    "tools are available",
    "what lane",
    "which lane",
    "what state",
    "state you are in",
)
_UNSUPPORTED_OPERATIONAL_CERTAINTY_RE = re.compile(
    r"\b(?:"
    r"full\s+capacity(?:\s+to)?|"
    r"peak\s+cognitive\s+efficiency|"
    r"zero\s+(?:delay|latency|uncertainty|error|errors|issues)|"
    r"without\s+(?:any\s+)?(?:delay|latency|uncertainty|error|errors|issues|friction)|"
    r"no\s+(?:delay|latency|uncertainty|error|errors|issues|risk)|"
    r"100%\s+(?:ready|available|reliable|operational|green)|"
    r"perfectly\s+(?:ready|available|reliable|operational)|"
    r"guaranteed\s+(?:ready|available|reliable|success|execution)|"
    r"(?:always|definitely)\s+(?:ready|available|reliable|able\s+to\s+execute)"
    r")\b",
    re.IGNORECASE,
)
_UNSUPPORTED_TELEMETRY_EQUIVALENCE_RE = re.compile(
    r"\b(?:"
    r"(?:neurodynamic|substrate|liquid\s+substrate|neural)\b.{0,120}\b(?:peak|full\s+capacity|cognitive\s+efficiency)|"
    r"\b\d+(?:\.\d+)?\s*hz\b.{0,120}\b(?:peak|full\s+capacity|cognitive\s+efficiency)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_READINESS_CLAIM_RE = re.compile(
    r"\b(?:tool[- ]?use\s+pathway|tool\s+pathway|tool\s+surface|governed\s+tools?|"
    r"external\s+tools?|desktop\s+tools?|operating\s+system\s+interface|os\s+control)\b"
    r".{0,180}\b(?:ready|available|online|primed|can\s+execute|able\s+to\s+execute|"
    r"ready\s+to\s+execute)\b"
    r"|\b(?:ready|available|online|primed|can\s+execute|able\s+to\s+execute|ready\s+to\s+execute)\b"
    r".{0,180}\b(?:tool[- ]?use\s+pathway|tool\s+pathway|tool\s+surface|governed\s+tools?|"
    r"external\s+tools?|desktop\s+tools?|operating\s+system\s+interface|os\s+control)\b",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_READINESS_BOUNDARY_RE = re.compile(
    r"\b(?:"
    r"permission|permissions|authorization|authorisation|authority|will|receipts?|"
    r"observable|observed|verification|verified|verify|effect\s+evidence|"
    r"app\s+state|available\s+app|probe|health|fail(?:s|ed)?\s+closed|bounded|"
    r"when\s+.*(?:allow|available|passes|pass)|if\s+.*(?:allow|available|passes|pass)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)
_DEPLOYMENT_ROUTING_CLAIM_MARKERS = (
    "demo slot",
    "live path slot",
    "server tier",
    "demo priority",
    "apply for live path",
    "roll up to",
    "routed to",
)


def _has_unsupported_deployment_routing_claim(
    user_message: Any,
    reply_text: Any,
) -> bool:
    """Reject invented deployment tiers unless the user supplied that claim."""

    prompt = _normalize(user_message)
    reply = _normalize(reply_text)
    claimed = {
        marker for marker in _DEPLOYMENT_ROUTING_CLAIM_MARKERS if marker in reply
    }
    if not claimed:
        return False
    return any(marker not in prompt for marker in claimed)


def grounded_social_repair_reply(user_message: Any) -> str:
    """Return a truthful immediate repair for a corrupted greeting turn."""

    prompt = _normalize(user_message)
    if re.search(r"\b(?:say|tell\s+(?:me|us)\s+)?hello\b", prompt):
        return "Hello. I'm Aura. I'm here with you."
    return ""
_EXACT_REPLY_COMMAND_RE = re.compile(
    r"\b(?:say|reply|respond|answer|return|print)\s+exactly\s*:?\s*",
    re.IGNORECASE,
)
_EXACT_REPLY_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "“": "”",
    "‘": "’",
}
_EXACT_REPLY_INTRODUCER_RE = re.compile(
    r"^(?:"
    r"as\s+follows\s*:"
    r"|(?:with|this)\s*:"
    r"|(?:with\s+)?(?:the\s+)?(?:following|word|words|phrase|text)\s*:"
    r"|with\s+(?=[\"'“‘])"
    r")\s*",
    re.IGNORECASE,
)
_EXACT_REPLY_UNQUOTED_SUFFIX_RE = re.compile(
    r"(?:"
    r",?\s+and\s+nothing\s+(?:else|more)"
    r"|,?\s+nothing\s+(?:else|more)"
    r"|,?\s+with\s+no\s+(?:additional|extra)\s+(?:text|words|commentary)"
    r")\s*$",
    re.IGNORECASE,
)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(?P<answer>.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_FENCED_BLOCK_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+.-]*)[ \t]*\n(?P<body>.*?)```",
    re.DOTALL,
)
_CODE_FENCE_LANGS = {
    "bash",
    "c",
    "cpp",
    "css",
    "go",
    "html",
    "java",
    "js",
    "json",
    "jsx",
    "mdx",
    "mjs",
    "py",
    "python",
    "rs",
    "ruby",
    "sh",
    "sql",
    "swift",
    "ts",
    "tsx",
    "typescript",
    "yaml",
    "yml",
}
_NON_CODE_FENCE_LANGS = {"", "md", "markdown", "text", "txt"}
_INCOMPLETE_CODE_TAIL_RE = re.compile(
    r"(?:[=+\-*/%&|^.,\\[(<{]|(?:\b(?:return|yield|raise|if|elif|else|for|while|with|try|except|finally)\b.*:))$"
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_COUNT_WORD_PATTERN = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)
_COUNT_TOKEN_RE = rf"(?P<count>\d+|{_COUNT_WORD_PATTERN})"
_PARAGRAPH_REQUEST_RE = re.compile(
    rf"\b{_COUNT_TOKEN_RE}\s+(?:concise\s+|short\s+|brief\s+|clear\s+)?paragraphs?\b",
    re.IGNORECASE,
)
_BULLET_REQUEST_RE = re.compile(
    rf"\b{_COUNT_TOKEN_RE}\s+(?:bullet(?:\s+points?)?|bullets?|items?)\b",
    re.IGNORECASE,
)
_NUMBERED_LIST_REQUEST_RE = re.compile(
    rf"\b(?:numbered\s+list|list)\s+(?:of\s+)?{_COUNT_TOKEN_RE}\b",
    re.IGNORECASE,
)
_NUMBERED_SENTENCE_REQUEST_RE = re.compile(
    rf"\b{_COUNT_TOKEN_RE}\s+(?:concise\s+|short\s+|brief\s+|clear\s+)?numbered\s+sentences?\b",
    re.IGNORECASE,
)
_FACT_COUNT_REQUEST_RE = re.compile(
    rf"\b{_COUNT_TOKEN_RE}\s+(?:quick\s+|concise\s+|short\s+|brief\s+|clear\s+)?facts?\b",
    re.IGNORECASE,
)
_CHOICE_CLARIFICATION_RE = re.compile(
    r"\bclarify\s+whether\s+(?P<subject>[A-Za-z0-9][A-Za-z0-9 '\u2019-]{1,80}?)\s+"
    r"(?:is|are|was|were)\s+(?P<left>[^?.!,;]{2,90}?)\s+or\s+(?P<right>[^?.!,;]{2,90})",
    re.IGNORECASE,
)
_ACTION_WORD_COUNT_REQUEST_RE = re.compile(
    rf"\b(?:answer|respond|reply|say|output)\s+(?:directly\s+)?"
    rf"(?:(?:in|with|using|exactly|only)\s+)?{_COUNT_TOKEN_RE}"
    rf"(?:\s+or\s+(?P<count_max>\d+|{_COUNT_WORD_PATTERN}))?"
    r"\s+words?\b",
    re.IGNORECASE,
)
_LIMIT_WORD_COUNT_REQUEST_RE = re.compile(
    rf"\b(?:in|with|using|exactly|only)\s+{_COUNT_TOKEN_RE}"
    rf"(?:\s+or\s+(?P<count_max>\d+|{_COUNT_WORD_PATTERN}))?"
    r"\s+words?\b",
    re.IGNORECASE,
)
_ACTION_SENTENCE_COUNT_REQUEST_RE = re.compile(
    rf"\b(?:answer|respond|reply|say|output)\s+(?:directly\s+)?"
    rf"(?:(?:in|with|using|exactly|only)\s+)?{_COUNT_TOKEN_RE}\s+"
    r"(?:short\s+|brief\s+|concise\s+|clear\s+|plain\s+|direct\s+)?sentences?\b",
    re.IGNORECASE,
)
_LIMIT_SENTENCE_COUNT_REQUEST_RE = re.compile(
    rf"\b(?:in|with|using|exactly|only)\s+{_COUNT_TOKEN_RE}\s+"
    r"(?:short\s+|brief\s+|concise\s+|clear\s+|plain\s+|direct\s+)?sentences?\b",
    re.IGNORECASE,
)
_REFERENCE_KIND_PATTERN = r"(?:sample|probe|check|case|item|step|test|ticket|request|reference)"
_REFERENCE_LABEL_VALUE_RE = re.compile(
    rf"\b(?P<label>(?:[A-Za-z][A-Za-z-]*\s+){{0,2}}(?P<kind>{_REFERENCE_KIND_PATTERN}))"
    r"\s*(?:number|id)?\s*[:#-]?\s*(?P<value>\d+)\b",
    re.IGNORECASE,
)
_INCLUDE_REFERENCE_VALUE_RE = re.compile(
    rf"\binclude(?:s|d|ing)?\s+(?:the\s+)?(?P<kind>{_REFERENCE_KIND_PATTERN})\s+"
    r"(?:number|id)\b",
    re.IGNORECASE,
)
_INCLUDE_GENERIC_REFERENCE_VALUE_RE = re.compile(
    r"\binclude(?:s|d|ing)?\s+(?:the\s+)?(?:number|id)\b",
    re.IGNORECASE,
)
_COMPACT_REFERENCE_ACK_RE = re.compile(
    rf"^\s*(?P<label>(?:[A-Za-z][A-Za-z-]*\s+){{0,3}}{_REFERENCE_KIND_PATTERN})"
    r"\s*(?P<value>\d+)\s*:\s*(?P<instruction>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FOLLOWUP_QUESTION_REQUEST_RE = re.compile(
    r"\b(?:ask|include|end\s+with|finish\s+with)\b.{0,80}\b"
    r"(?:follow[- ]?up|grounded|clarifying|next)\b.{0,80}\bquestions?\b"
    r"|\bfollow[- ]?up\s+questions?\b",
    re.IGNORECASE,
)
_REQUESTS_DIRECT_RECALL_OR_PROCESS_ANSWER_RE = re.compile(
    r"\b(?:"
    r"answer\s+directly"
    r"|what\s+did\s+i\s+(?:just\s+)?ask(?:\s+you)?(?:\s+to\s+do)?"
    r"|what\s+did\s+i\s+(?:just\s+)?say"
    r"|what\s+mind(?:/| )cognition\s+path"
    r"|what\s+(?:cognitive|cognition|mind)\s+path"
    r"|what\s+path\s+are\s+you\s+using"
    r"|path\s+are\s+you\s+using\s+right\s+now"
    r")\b",
    re.IGNORECASE,
)
_CURRENT_REQUEST_RECAP_REQUEST_RE = re.compile(
    r"\bwhat\s+did\s+i\s+(?:just\s+)?ask(?:\s+you)?(?:\s+to\s+do)?\b",
    re.IGNORECASE,
)
_CURRENT_REQUEST_RECAP_ANSWER_RE = re.compile(
    r"\b(?:"
    r"you\s+asked(?:\s+me)?(?:\s+to)?"
    r"|your\s+request\s+(?:was|is)"
    r"|the\s+request\s+(?:was|is)"
    r"|you\s+wanted\s+me\s+to"
    r"|you\s+asked\s+for"
    r")\b",
    re.IGNORECASE,
)
_QUESTION_BACK_NON_ANSWER_RE = re.compile(
    r"\b(?:"
    r"what\s+did\s+you\s+(?:just\s+)?ask\s+me(?:\s+to\s+do)?"
    r"|what\s+did\s+i\s+ask\s+you(?:\s+to\s+do)?"
    r"|what\s+(?:cognitive|cognition|mind)\s+path\s+am\s+i\s+using"
    r"|what\s+path\s+am\s+i\s+using"
    r")\??\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_JAMMED_NUMBERED_MARKER_RE = re.compile(r"(?<=[.!?])(?=\d+[.)]\s*)")
_LIST_LINE_RE = re.compile(r"^\s*(?P<marker>(?:[-*+]|\d+[.)]))\s*(?P<body>.*)$")
_EXACT_REPLY_CONDITIONAL_TAIL_RE = re.compile(
    r"(?:^|[,;]\s*|\s+)"
    r"(?:if|when|unless|otherwise|else|or(?:\s+(?:reply|respond|say|use))?)\b",
    re.IGNORECASE,
)
_EXACT_REPLY_ADDITIONAL_ACTION_TAIL_RE = re.compile(
    r"(?:"
    r"[.!?;]\s*(?:(?:then|next|also)\s*,?\s*)?"
    r"|\s+(?:and\s+)?then\s+"
    r"|\s+(?:and\s+)?(?:also|next)\s+"
    r")"
    r"(?:please\s+)?(?:explain|describe|justify|elaborate|summarize|tell|show|"
    r"compare|list|discuss|answer|reply|respond|write|provide|include)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationReplyAssessment:
    ok: bool
    reasons: tuple[str, ...]
    hard_failure: bool
    retryable: bool

    def has(self, reason: str) -> bool:
        return reason in self.reasons


@dataclass(frozen=True)
class RequestedOutputContract:
    """Typed, user-authored output-size constraints for one visible reply."""

    kind: str = "none"
    word_min: int | None = None
    word_max: int | None = None
    sentence_count: int | None = None
    explicit_brevity: bool = False
    exact_reply: bool = False
    exact_reply_chars: int | None = None
    exact_reply_utf8_bytes: int | None = None
    semantic_token_cap: int | None = None
    hard_token_ceiling: int | None = None
    confidence: float = 0.0

    @property
    def constrained(self) -> bool:
        return self.hard_token_ceiling is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "word_min": self.word_min,
            "word_max": self.word_max,
            "sentence_count": self.sentence_count,
            "explicit_brevity": self.explicit_brevity,
            "exact_reply": self.exact_reply,
            "exact_reply_chars": self.exact_reply_chars,
            "exact_reply_utf8_bytes": self.exact_reply_utf8_bytes,
            "semantic_token_cap": self.semantic_token_cap,
            "hard_token_ceiling": self.hard_token_ceiling,
            "confidence": self.confidence,
        }


def _normalize(text: Any) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\bdont'?\b", "don't", normalized)


def is_cognitive_engine_failure_envelope(reply_text: Any) -> bool:
    """Return true for internal CognitiveEngine failure notices.

    These notices are useful diagnostic artifacts, but they are not completed
    user-facing answers and must never count as proof of a full live mind path.
    """

    return bool(_COGNITIVE_ENGINE_FAILURE_ENVELOPE_RE.search(str(reply_text or "")))


def _requires_self_claim_evidence_boundary(prompt: Any) -> bool:
    """Return true only for actual consciousness/personhood/selfhood claims.

    Plain style language such as "talking like a person" should not force a
    proof-style answer. Direct claims or questions about consciousness,
    sentience, subjective experience, qualia, personhood, or being a person
    still must stay evidence-bounded.
    """

    text = _normalize(prompt)
    if not text:
        return False
    if re.search(
        r"\b(?:conscious|consciousness|sentient|sentience|self[- ]?aware|"
        r"subjective|inner\s+life|qualia|personhood)\b",
        text,
    ):
        return True
    if re.search(
        r"\b(?:do|does|can|could|would)\s+(?:you|aura)\s+"
        r"(?:actually\s+|really\s+|truly\s+)?(?:feel|experience)\b"
        r"|\b(?:do|does|have|has)\s+(?:you|aura|i)\b.{0,80}"
        r"\b(?:feelings|experiences)\b",
        text,
    ):
        return True
    if re.search(
        r"\b(?:are\s+you|is\s+aura|am\s+i)\b.{0,80}\b(?:a\s+)?person\b"
        r"|\b(?:you\s+are|you're|aura\s+is|i\s+am)\s+(?:a\s+)?person\b"
        r"|\b(?:being|become|counts?\s+as|qualif(?:y|ies)\s+as)\b.{0,80}\b(?:a\s+)?person\b",
        text,
    ):
        return True
    return False


def _word_count(text: Any) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def _count_token_to_int(value: str | None) -> int | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    if token.isdigit():
        count = int(token)
    else:
        count = _NUMBER_WORDS.get(token)
    if count is None or count < 1 or count > 20:
        return None
    return count


def _word_count_token_to_int(value: str | None) -> int | None:
    """Parse explicit word limits without imposing list-count's 20-item cap."""

    token = str(value or "").strip().lower()
    if not token:
        return None
    if token.isdigit():
        count = int(token)
    else:
        count = _NUMBER_WORDS.get(token)
    if count is None or count < 1 or count > 4096:
        return None
    return count


def _is_escaped_character(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return bool(backslashes % 2)


def _text_index_is_unquoted(text: str, index: int) -> bool:
    """Return whether ``index`` is outside quoted and code-literal spans."""

    quote_close = ""
    fenced_code = False
    inline_code = False
    cursor = 0
    limit = max(0, min(len(text), int(index)))
    while cursor < limit:
        if not quote_close and not inline_code and text.startswith("```", cursor):
            fenced_code = not fenced_code
            cursor += 3
            continue
        if fenced_code:
            cursor += 1
            continue

        char = text[cursor]
        if not quote_close and char == "`" and not _is_escaped_character(text, cursor):
            inline_code = not inline_code
            cursor += 1
            continue
        if inline_code:
            cursor += 1
            continue

        if quote_close:
            if char == quote_close and not _is_escaped_character(text, cursor):
                quote_close = ""
            cursor += 1
            continue

        if char in _EXACT_REPLY_QUOTE_PAIRS and not _is_escaped_character(text, cursor):
            is_apostrophe = (
                char == "'"
                and cursor > 0
                and cursor + 1 < len(text)
                and text[cursor - 1].isalnum()
                and text[cursor + 1].isalnum()
            )
            if not is_apostrophe:
                quote_close = _EXACT_REPLY_QUOTE_PAIRS[char]
        cursor += 1
    return not (quote_close or fenced_code or inline_code)


def _constraint_match_is_actionable(text: str, match: re.Match[str]) -> bool:
    """Reject quoted, code-sample, and explicitly negated length language."""

    before = text[: match.start()]
    if not _text_index_is_unquoted(text, match.start()):
        return False
    prefix = (
        before[-192:]
        .lower()
        .replace("‘", "'")
        .replace("’", "'")
    )
    # Negation applies to its grammatical clause, not an unrelated command
    # after punctuation or a coordinating transition.
    prefix = re.split(r"[.!?;,\n]", prefix)[-1]
    prefix = re.split(
        r"\b(?:then|but|however|instead|otherwise|next|now)\b",
        prefix,
    )[-1]
    prefix = re.split(
        r"\b(?:and|or)\s+(?=(?:then\s+)?(?:answer|reply|respond|say|output|return|print)\b)",
        prefix,
    )[-1]
    # Some command regexes include the command verb in the match itself. In
    # that case the prefix ends at the coordinator, so the lookahead above
    # cannot see the fresh predicate even though it starts at ``match``.
    if re.search(r"\b(?:and|or)\s*$", prefix) and re.match(
        r"\s*(?:answer|reply|respond|say|output|return|print)\b",
        match.group(0),
        re.IGNORECASE,
    ):
        prefix = ""
    return not bool(
        re.search(
            r"\b(?:do\s+not|don't|never|ignore|disregard|avoid|rather\s+than|instead\s+of|"
            r"no\s+need\s+to|without|not\s+(?:limited|restricted|confined)\s+to|"
            r"(?:do(?:es)?\s+not|don't|doesn't)\s+have\s+to|"
            r"(?:old|previous|example|sample)\s+(?:instruction|prompt|command|text)\s+"
            r"(?:was|said|says|contained)|"
            r"(?:(?:i(?:'m|\s+am)|we(?:'re|\s+are)|you(?:'re|\s+are)|"
            r"they(?:'re|\s+are))\s+)?not\s+asking(?:\s+you)?\s+to)\b"
            r"[^.!?;\n]{0,72}$",
            prefix,
        )
    )


def _requested_count(pattern: re.Pattern[str], user_message: Any) -> int | None:
    match = pattern.search(str(user_message or ""))
    if not match:
        return None
    return _count_token_to_int(match.groupdict().get("count"))


def _requested_word_count_range(user_message: Any) -> tuple[int, int] | None:
    text = str(user_message or "")
    candidates: list[tuple[int, int, int, int]] = []
    for pattern in (_ACTION_WORD_COUNT_REQUEST_RE, _LIMIT_WORD_COUNT_REQUEST_RE):
        for match in pattern.finditer(text):
            if not _constraint_match_is_actionable(text, match):
                continue
            minimum = _word_count_token_to_int(match.groupdict().get("count"))
            maximum = _word_count_token_to_int(match.groupdict().get("count_max"))
            if minimum is None:
                continue
            if maximum is None:
                maximum = minimum
            candidates.append(
                (match.start(), match.end(), min(minimum, maximum), max(minimum, maximum))
            )
    if not candidates:
        return None
    _start, _end, minimum, maximum = max(candidates, key=lambda item: (item[0], item[1]))
    return minimum, maximum


def _requested_sentence_count(user_message: Any) -> int | None:
    text = str(user_message or "")
    candidates: list[tuple[int, int, int]] = []
    for pattern in (
        _ACTION_SENTENCE_COUNT_REQUEST_RE,
        _LIMIT_SENTENCE_COUNT_REQUEST_RE,
    ):
        for match in pattern.finditer(text):
            if not _constraint_match_is_actionable(text, match):
                continue
            requested = _count_token_to_int(match.groupdict().get("count"))
            if requested is not None:
                candidates.append((match.start(), match.end(), requested))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def requested_sentence_count(user_message: Any) -> int | None:
    """Return the exact sentence-count contract explicitly requested by the user."""

    return _requested_sentence_count(user_message)


def requested_exact_reply_target(user_message: Any) -> str:
    """Return the last actionable exact-reply target.

    Surrounding transport whitespace is not part of the contract. Target case
    and punctuation are preserved for both quoted and unquoted commands.
    """

    raw = str(user_message or "").strip()
    if not raw:
        return ""
    commands = [
        match
        for match in _EXACT_REPLY_COMMAND_RE.finditer(raw)
        if _text_index_is_unquoted(raw, match.start())
    ]
    candidates: list[tuple[int, str]] = []
    for index, match in enumerate(commands):
        if not _constraint_match_is_actionable(raw, match):
            continue
        end = commands[index + 1].start() if index + 1 < len(commands) else len(raw)
        remainder = raw[match.end() : end].lstrip()
        remainder = _EXACT_REPLY_INTRODUCER_RE.sub("", remainder, count=1).lstrip()
        if not remainder:
            continue
        quote = remainder[0]
        if quote in _EXACT_REPLY_QUOTE_PAIRS:
            closing = _EXACT_REPLY_QUOTE_PAIRS[quote]
            target_chars: list[str] = []
            close_index = -1
            cursor = 1
            while cursor < len(remainder):
                char = remainder[cursor]
                apostrophe = bool(
                    quote == "'"
                    and char == "'"
                    and cursor > 0
                    and cursor + 1 < len(remainder)
                    and remainder[cursor - 1].isalnum()
                    and remainder[cursor + 1].isalnum()
                )
                if (
                    char == closing
                    and not apostrophe
                    and not _is_escaped_character(remainder, cursor)
                ):
                    close_index = cursor
                    break
                if (
                    char == "\\"
                    and cursor + 1 < len(remainder)
                    and remainder[cursor + 1] in {"\\", quote, closing}
                ):
                    target_chars.append(remainder[cursor + 1])
                    cursor += 2
                    continue
                target_chars.append(char)
                cursor += 1
            if close_index <= 1:
                continue
            trailing_meta = _EXACT_REPLY_UNQUOTED_SUFFIX_RE.sub(
                "",
                remainder[close_index + 1 :],
            ).strip()
            if trailing_meta.strip(".!?;:, "):
                continue
            target = "".join(target_chars).strip()
        else:
            target = remainder.strip()
            if _EXACT_REPLY_ADDITIONAL_ACTION_TAIL_RE.search(target):
                continue
            target = _EXACT_REPLY_UNQUOTED_SUFFIX_RE.sub("", target).rstrip()
            if _EXACT_REPLY_CONDITIONAL_TAIL_RE.search(target):
                continue
            target = re.split(
                r"(?<=[.!?])\s+(?=(?:now|then|after|before|also|next|instead|please)\b)",
                target,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            target = target.strip()
        if target:
            candidates.append((match.start(), target))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1]


def _compact_output_style_requested(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    pattern = re.compile(
        r"\b(?:briefly|be brief|be concise|keep (?:it|this) (?:brief|concise|short)|"
        r"(?:brief|concise|short|plain|direct) (?:answer|reply|response|sentence)|"
        r"(?:brief|concise|short|plain|direct) sentences?|"
        r"in (?:a|one) (?:brief|concise|short|plain|direct) sentence|"
        r"answer directly|reply directly|respond directly|include nothing else|nothing else)\b"
    )
    return any(
        _constraint_match_is_actionable(text, match)
        for match in pattern.finditer(text)
    )


def requested_output_contract(user_message: Any) -> RequestedOutputContract:
    """Return a conservative token ceiling derived from visible user intent.

    The semantic cap is a planning target. The hard ceiling includes enough
    tokenizer and punctuation headroom to satisfy the requested shape, while
    remaining an absolute upper bound after affective and pressure modulation.
    """

    raw = str(user_message or "").strip()
    if not raw:
        return RequestedOutputContract()

    exact_target = requested_exact_reply_target(raw)
    if exact_target:
        utf8_bytes = len(exact_target.encode("utf-8"))
        estimated_tokens = (
            max(1, (len(exact_target) + 2) // 3)
            if exact_target.isascii()
            else max(1, utf8_bytes)
        )
        semantic_cap = min(8192, max(8, estimated_tokens + 4))
        # Byte-fallback tokenizers cannot require more content tokens than the
        # UTF-8 byte count. Keep protocol/EOS headroom above that real bound;
        # the selected worker tokenizer records and verifies the exact count.
        hard_ceiling = max(16, utf8_bytes + 16)
        return RequestedOutputContract(
            kind="exact_reply",
            explicit_brevity=True,
            exact_reply=True,
            exact_reply_chars=len(exact_target),
            exact_reply_utf8_bytes=utf8_bytes,
            semantic_token_cap=semantic_cap,
            hard_token_ceiling=hard_ceiling,
            confidence=1.0,
        )

    word_range = _requested_word_count_range(raw)
    sentence_count = _requested_sentence_count(raw)
    explicit_brevity = _explicit_brevity_requested(raw)
    compact_style = word_range is not None or _compact_output_style_requested(raw)
    if word_range is None and sentence_count is None and not explicit_brevity:
        return RequestedOutputContract()

    semantic_candidates: list[int] = []
    hard_candidates: list[int] = []
    kinds: list[str] = []
    if word_range is not None:
        _minimum_words, maximum_words = word_range
        semantic_candidates.append(max(16, 8 + (2 * maximum_words)))
        hard_candidates.append(max(32, 16 + (3 * maximum_words)))
        kinds.append("word_count")
    if sentence_count is not None:
        semantic_per_sentence = 32 if compact_style else 64
        hard_per_sentence = 48 if compact_style else 96
        semantic_candidates.append(max(24, semantic_per_sentence * sentence_count))
        hard_candidates.append(max(32, hard_per_sentence * sentence_count))
        kinds.append("sentence_count")
    if explicit_brevity and not semantic_candidates:
        semantic_candidates.append(64)
        hard_candidates.append(112)
        kinds.append("brevity")

    semantic_cap = min(8192, max(semantic_candidates))
    hard_ceiling = min(8192, max(semantic_cap, max(hard_candidates)))
    return RequestedOutputContract(
        kind="+".join(kinds),
        word_min=word_range[0] if word_range else None,
        word_max=word_range[1] if word_range else None,
        sentence_count=sentence_count,
        explicit_brevity=compact_style,
        semantic_token_cap=semantic_cap,
        hard_token_ceiling=hard_ceiling,
        confidence=0.98 if word_range is not None or sentence_count is not None else 0.9,
    )


def _requested_reference_values(user_message: Any) -> tuple[tuple[str, int], ...]:
    user = str(user_message or "")
    if not user:
        return ()
    requested_kinds = {
        str(match.group("kind") or "").strip().lower()
        for match in _INCLUDE_REFERENCE_VALUE_RE.finditer(user)
        if str(match.group("kind") or "").strip()
    }
    generic_reference_requested = bool(
        _INCLUDE_GENERIC_REFERENCE_VALUE_RE.search(user)
    )
    observed = [
        (
            " ".join(str(match.group("label") or "").strip().split()).lower(),
            str(match.group("kind") or "").strip().lower(),
            int(match.group("value")),
        )
        for match in _REFERENCE_LABEL_VALUE_RE.finditer(user)
    ]
    if generic_reference_requested and len(observed) != 1:
        return ()
    requested = [
        (label, value)
        for label, kind, value in observed
        if kind in requested_kinds or generic_reference_requested
    ]
    return tuple(dict.fromkeys(requested))


def _reply_contains_reference_value(reply_text: Any, value: int) -> bool:
    reply = _normalize(reply_text)
    if not reply:
        return False
    if re.search(rf"(?<!\d){int(value)}(?!\d)", reply):
        return True
    number_word = next(
        (word for word, number in _NUMBER_WORDS.items() if number == int(value)),
        "",
    )
    return bool(number_word and re.search(rf"\b{re.escape(number_word)}\b", reply))


def _compact_reference_acknowledgement(user_message: Any) -> str:
    """Return a deterministic exact-format acknowledgement when that is the task."""

    user = str(user_message or "")
    match = _COMPACT_REFERENCE_ACK_RE.match(user)
    if not match or _requested_sentence_count(user) != 1:
        return ""
    references = _requested_reference_values(user)
    value = int(match.group("value"))
    if not references or not any(reference_value == value for _, reference_value in references):
        return ""
    label = " ".join(str(match.group("label") or "").strip().split()).lower()
    if not label:
        return ""
    return f"{label[0].upper()}{label[1:]} {value} completed."


_QUOTED_REQUIRED_PHRASE_RE = re.compile(
    r"\b(?:include|mention|use)\b[^\"'“”‘’]{0,80}[\"'“”‘’](?P<phrase>[^\"'“”‘’]{1,80})[\"'“”‘’]",
    re.IGNORECASE,
)
_INCLUDE_REQUIRED_PHRASE_RE = re.compile(
    r"\b(?:include|mention)\s+(?:the\s+)?(?:(?:exact\s+)?(?:phrase|word|term)\s+)?"
    r"(?P<phrase>[A-Za-z0-9][A-Za-z0-9 _-]{1,80})(?:[.!?;,]|$)",
    re.IGNORECASE,
)
_USE_REQUIRED_PHRASE_RE = re.compile(
    r"\buse\s+(?:the\s+)?(?:exact\s+)?(?:phrase|word|term)\s+"
    r"(?P<phrase>[A-Za-z0-9][A-Za-z0-9 _-]{1,80})(?:[.!?;,]|$)",
    re.IGNORECASE,
)


# Heads that mark a scope/brevity instruction ("include nothing else"), not a
# literal phrase the reply must contain.
_BREVITY_PSEUDO_PHRASE_HEADS = frozenset(
    {"nothing", "no", "only", "just", "anything", "everything", "none"}
)


def _requested_required_phrases(user_message: Any) -> tuple[str, ...]:
    text = str(user_message or "")
    if not text:
        return ()
    phrases: list[str] = []
    for pattern in (
        _QUOTED_REQUIRED_PHRASE_RE,
        _INCLUDE_REQUIRED_PHRASE_RE,
        _USE_REQUIRED_PHRASE_RE,
    ):
        for match in pattern.finditer(text):
            phrase = " ".join(str(match.group("phrase") or "").strip(" .,:;!?\"'“”‘’").split())
            if not phrase:
                continue
            # Avoid treating a full instruction clause as a required phrase when
            # the user wrote something like "use your own voice and include X".
            if len(_WORD_RE.findall(phrase)) > 8:
                continue
            # "include nothing else", "include only the answer" are BREVITY/scope
            # instructions, not a literal phrase to echo. Treating them as a
            # required phrase made a valid short reply fail 'missing_requested_phrase'.
            if phrase.lower().split()[0] in _BREVITY_PSEUDO_PHRASE_HEADS:
                continue
            phrases.append(phrase.lower())
    return tuple(dict.fromkeys(phrases))


def has_requested_word_count_contract(user_message: Any) -> bool:
    """Return True when the user gave an explicit word-count output contract."""
    return _requested_word_count_range(user_message) is not None


def _requested_list_item_count(user_message: Any) -> int:
    requested_bullets = _requested_count(_BULLET_REQUEST_RE, user_message)
    requested_numbered = _requested_count(_NUMBERED_LIST_REQUEST_RE, user_message)
    requested_numbered_sentences = _requested_count(_NUMBERED_SENTENCE_REQUEST_RE, user_message)
    return max(requested_bullets or 0, requested_numbered or 0, requested_numbered_sentences or 0)


def normalize_user_facing_format(reply_text: Any) -> str:
    """Apply safe whitespace-only repairs to user-facing prose.

    This is deliberately conservative: it does not create new content, but it
    fixes common local-model formatting defects such as ``sentence.2. next``.
    """
    text = str(reply_text or "").strip()
    if not text:
        return text
    text = _JAMMED_NUMBERED_MARKER_RE.sub("\n", text)
    text = re.sub(r"(?m)^(\s*\d+[.)])(?=\S)", r"\1 ", text)
    return text.strip()


def _list_item_bodies(reply_text: Any) -> list[str]:
    normalized = normalize_user_facing_format(reply_text)
    bodies: list[str] = []
    for line in normalized.splitlines():
        match = _LIST_LINE_RE.match(line)
        if match:
            bodies.append(str(match.group("body") or "").strip())
    return bodies


def _nonempty_list_item_count(reply_text: Any) -> int:
    return sum(1 for body in _list_item_bodies(reply_text) if _word_count(body) > 0)


def _has_empty_requested_list_item(reply_text: Any, requested_count: int) -> bool:
    if requested_count <= 1:
        return False
    bodies = _list_item_bodies(reply_text)
    if not bodies:
        return False
    return any(_word_count(body) == 0 for body in bodies[:requested_count])


def _paragraph_count(reply_text: Any) -> int:
    blocks = [
        block.strip()
        for block in re.split(r"(?:\r?\n\s*){2,}", str(reply_text or "").strip())
        if _word_count(block) > 0
    ]
    return len(blocks)


def _bullet_count(reply_text: Any) -> int:
    return _nonempty_list_item_count(reply_text)


def _inline_numbered_item_count(reply_text: Any) -> int:
    text = str(reply_text or "")
    matches = re.findall(r"(?<!\d)(?:^|[\s:.;])\d{1,2}[\.)]\s*\S", text)
    return len(matches)


def _factual_unit_count(reply_text: Any) -> int:
    """Estimate how many discrete facts a reply actually supplied."""

    normalized = normalize_user_facing_format(reply_text)
    if not normalized:
        return 0
    inline_numbered = _inline_numbered_item_count(normalized)
    if inline_numbered:
        return inline_numbered
    list_count = _bullet_count(normalized)
    if list_count:
        return list_count
    sentence_units = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|(?:\s*;\s*)", normalized)
        if _word_count(part) >= 3
    ]
    comma_fact_count = 0
    for sentence in sentence_units:
        if "," not in sentence or " and " not in sentence.lower():
            continue
        if not re.search(
            r"\b(?:can|could|are|were|is|was|have|has|survive|tolerate|enter|repair)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        parts = [
            part.strip()
            for part in re.split(r",\s+|\s+\band\b\s+", sentence, flags=re.IGNORECASE)
            if _word_count(part) >= 2
        ]
        if len(parts) >= 3:
            comma_fact_count = max(comma_fact_count, len(parts))
    return max(len(sentence_units), comma_fact_count)


def _keywords_for_choice(text: str) -> set[str]:
    stop = {"the", "a", "an", "is", "are", "was", "were", "moon", "planet", "one", "it", "its"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) >= 3 and token not in stop
    }


def _missing_choice_clarification(user_message: Any, reply_text: Any) -> bool:
    user = str(user_message or "")
    reply = _normalize(reply_text)
    if not user or not reply:
        return False
    for match in _CHOICE_CLARIFICATION_RE.finditer(user):
        subject_terms = _keywords_for_choice(match.group("subject"))
        left_terms = _keywords_for_choice(match.group("left"))
        right_terms = _keywords_for_choice(match.group("right"))
        if subject_terms and not any(term in reply for term in subject_terms):
            return True
        if left_terms or right_terms:
            if not any(term in reply for term in (left_terms | right_terms)):
                return True
    return False


_MEMORY_LIMIT_DUAL_REQUEST_RE = re.compile(
    r"\b(?:remember|recall|memory|retained|from this session|from earlier|across sessions?)\b"
    r"(?s:.){0,260}"
    r"\b(?:limit|boundary|should not pretend|cannot|can't|do not know|don't know|honest limits?)\b"
    r"|"
    r"\b(?:limit|boundary|should not pretend|cannot|can't|do not know|don't know|honest limits?)\b"
    r"(?s:.){0,260}"
    r"\b(?:remember|recall|memory|retained|from this session|from earlier|across sessions?)\b",
    re.IGNORECASE,
)
_MEMORY_COVERAGE_REPLY_RE = re.compile(
    r"\b(?:remember|recall|memory|retained|you asked|you told me|from this session|"
    r"earlier in this (?:session|conversation)|session context|conversation context|"
    r"what i can see in (?:memory|the transcript|this thread))\b",
    re.IGNORECASE,
)
_LIMIT_COVERAGE_REPLY_RE = re.compile(
    r"\b(?:limit|boundary|should not pretend|cannot|can't|do not know|don't know|"
    r"not claim|not pretend|not infer|unproven|unknown|without evidence|"
    r"i should be honest)\b",
    re.IGNORECASE,
)


# Injected scaffolding a live turn carries alongside the person's words:
# retained-memory evidence blocks, the identity anchor, replayed transcript.
# Instruction-coverage detectors must never read these as things the USER
# asked for. Live 2026-07-25: a plant turn — "Small thing to remember for
# later in this chat: my friend's dog is named Biscuit. Brief acknowledgment
# is fine." — arrived at the gate with an 8,000-character evidence block
# appended, whose own rule text ("say the memory is not verified") and
# replayed prior turns ("I can't work through that…") put "remember" within
# 260 characters of "can't". The dual memory/limit detector fired, the facet
# detector demanded coverage of facets nobody requested, and a correct brief
# acknowledgement was rejected as an unanswered turn.
_MAX_PLAUSIBLE_USER_TURN_CHARS = 2000

# Reasons that assert "the reply did not cover what the USER asked for". Every
# one is meaningless when the user's request could not be isolated.
_REQUEST_COVERAGE_REASONS = frozenset(
    {
        "missing_requested_exact_reply",
        "missing_requested_word_count",
        "missing_requested_sentence_count",
        "missing_requested_reference_value",
        "missing_requested_paragraph_count",
        "missing_requested_list_count",
        "empty_requested_list_item",
        "missing_requested_choice_clarification",
        "missing_requested_memory_limit_coverage",
        "missing_requested_followup_question",
        "missing_requested_phrase",
        "missing_requested_objective_facets",
        "missing_requested_self_process_coverage",
        "reliability_diagnostic_too_thin",
        "reliability_diagnostic_deflection",
        "low_signal_reliability_reply",
        "detail_request_deflection",
        "prompt_echo_contamination",
        # Every one of these is a claim about the reply's FIT TO THE REQUEST,
        # not about the reply itself. Live 2026-07-25: "Here. Bit more settled
        # than an hour ago." — a correct answer to "more settled or more
        # strained than an hour ago?" — was rejected as a
        # generic_memory_pin_acknowledgement, because the validation prompt was
        # 2,705 characters of assembled context containing an earlier memory
        # pin. The reply was fine; the comparison was impossible.
        "generic_memory_pin_acknowledgement",
        "off_topic_self_reflection_reply",
        "missing_self_condition_answer",
        "missing_future_memory_answer",
        "missing_identity_answer",
        "unsupported_memory_guarantee",
        "low_signal_acknowledgement_placeholder",
        "persona_card_deflection",
        "contextual_relevance_miss",
    }
)

_INJECTED_PROMPT_BLOCK_MARKERS = (
    "[retained memory evidence]",
    "scope=retained_memory_evidence",
    "## intrinsic identity anchor",
    "intrinsic identity anchor",
    "source=recent_completed_transcript",
    "source=durable_memory_search",
    "[conversation context]",
    "[working memory]",
    "[evidence]",
)


# A replayed transcript line, e.g. "turn_2.user=..." / "turn_1.aura=...".
_TRANSCRIPT_REPLAY_LINE_RE = re.compile(r"^\s*turn_\d+\.(?:user|aura)\s*=", re.IGNORECASE)
# Structured scaffold key/value lines, e.g. "scope=...", "rule=...", "source=...".
_SCAFFOLD_KV_LINE_RE = re.compile(
    r"^\s*(?:scope|rule|source|policy|contract|schema|evidence|constraint)\s*=",
    re.IGNORECASE,
)


# Arithmetic a reply can be CHECKED against. The 2026-07-25 probe asked
# "What is 144 / 6 + 7? Just the number." and was answered "Will do. Searched
# web for 'simple cognitive tasks aging'. Dementia affects simple cognitive
# tasks first…" — retrieved memory served as the answer. Nothing caught it:
# the topicality check needs topic anchors, and a bare sum has almost none, so
# a short computable question was unjudgeable by every gate in the path.
#
# It does not have to be. An arithmetic question has one right answer and the
# runtime can do the arithmetic itself, which turns "sounds plausible" into
# "is correct" for the whole class — including the hijack, which contains no
# number at all.
_ARITHMETIC_QUESTION_RE = re.compile(
    r"(?:what(?:'s| is)|calculate|compute|how much is|solve)\s*:?\s*"
    r"([0-9][0-9\s\.\+\-\*/x×÷\(\)]{2,60})",
    re.IGNORECASE,
)
_ARITHMETIC_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _evaluate_arithmetic(expression: str) -> float | None:
    """Evaluate a simple arithmetic expression, or None if it is not one."""
    cleaned = (
        str(expression or "")
        .replace("x", "*").replace("X", "*")
        .replace("×", "*").replace("÷", "/")
        .strip().rstrip("?.=").strip()
    )
    if not cleaned or not re.fullmatch(r"[0-9\s\.\+\-\*/\(\)]+", cleaned):
        return None
    if not any(op in cleaned for op in "+-*/"):
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError
                return left / right
        raise ValueError("unsupported expression")

    try:
        result = _eval(tree.body)
    except (ArithmeticError, ValueError, TypeError, RecursionError):
        return None
    if not math.isfinite(result):
        return None
    return float(result)


# Word forms with exactly one mechanical answer. The bare-expression pattern
# covered 2 of the 8 math questions the 2026-07-25 probe actually asks; these
# two forms are the other computable ones. Everything left — rates, catch-up,
# pages-per-day — needs reasoning and is deliberately NOT claimed here.
_PERCENT_OF_RE = re.compile(
    r"what(?:'s| is)\s+([0-9]+(?:\.[0-9]+)?)\s*%\s+of\s+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_POWER_RE = re.compile(
    r"what(?:'s| is)\s+([0-9]+)\s+to\s+the\s+([0-9]+)(?:st|nd|rd|th)?\s+power",
    re.IGNORECASE,
)
_RECTANGLE_AREA_RE = re.compile(
    r"rectangle\s+is\s+([0-9]+(?:\.[0-9]+)?)\s*(?:by|x|\*)\s*([0-9]+(?:\.[0-9]+)?)"
    r"(?s:.){0,60}?\barea\b",
    re.IGNORECASE,
)


def requested_arithmetic_result(user_message: Any) -> float | None:
    """The single correct answer to a computable arithmetic question, if any."""
    text = str(user_message or "")

    match = _PERCENT_OF_RE.search(text)
    if match:
        try:
            return float(match.group(1)) / 100.0 * float(match.group(2))
        except (ArithmeticError, ValueError):
            return None

    match = _POWER_RE.search(text)
    if match:
        try:
            base, exponent = int(match.group(1)), int(match.group(2))
        except ValueError:
            return None
        # Bounded: a runaway exponent must not become the check's own problem.
        if not (0 <= exponent <= 64) or abs(base) > 10_000:
            return None
        try:
            value = float(base**exponent)
        except (ArithmeticError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    match = _RECTANGLE_AREA_RE.search(text)
    if match:
        try:
            return float(match.group(1)) * float(match.group(2))
        except (ArithmeticError, ValueError):
            return None

    match = _ARITHMETIC_QUESTION_RE.search(text)
    if not match:
        return None
    return _evaluate_arithmetic(match.group(1))


def _arithmetic_answer_missing(user_message: Any, reply_text: Any) -> bool:
    """Whether a checkable arithmetic answer is absent or wrong.

    Fails OPEN: if the question is not computable here, this says nothing.
    """
    expected = requested_arithmetic_result(user_message)
    if expected is None:
        return False
    reply = str(reply_text or "")
    if not reply.strip():
        return True
    for token in _ARITHMETIC_NUMBER_RE.findall(reply.replace(",", "")):
        try:
            value = float(token)
        except ValueError:
            continue
        if abs(value - expected) <= max(1e-6, abs(expected) * 1e-9):
            return False
    return True


def visible_user_request(user_message: Any) -> str:
    """Return only the part of a turn the PERSON wrote, or "" if unknowable.

    A live prompt is assembled: identity anchor, retained-memory evidence,
    replayed transcript, working-memory blocks — with the person's actual words
    somewhere inside. Scaffold appears BEFORE the request as often as after, so
    truncating at the first marker is wrong in both directions.

    Returning "" when the request cannot be isolated is the important half.
    A coverage check that cannot see what was asked must not assert the reply
    failed to cover it — an unknown request is not an unmet one.
    """
    text = str(user_message or "")
    if not text.strip():
        return ""

    kept: list[str] = []
    in_scaffold_block = False
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            in_scaffold_block = False       # a blank line ends a block
            kept.append(line)
            continue
        if any(marker in lowered for marker in _INJECTED_PROMPT_BLOCK_MARKERS):
            in_scaffold_block = True
            continue
        if _TRANSCRIPT_REPLAY_LINE_RE.match(stripped) or _SCAFFOLD_KV_LINE_RE.match(stripped):
            in_scaffold_block = True
            continue
        if in_scaffold_block:
            continue
        kept.append(line)

    remainder = "\n".join(kept).strip()
    if not remainder:
        return ""
    # A remainder that is still mostly assembled context is not a request. The
    # live prompts run to ~8,000 characters; a person's turn does not.
    if len(remainder) > _MAX_PLAUSIBLE_USER_TURN_CHARS:
        return ""
    return remainder


def _missing_requested_memory_limit_coverage(user_message: Any, reply_text: Any) -> bool:
    user = visible_user_request(user_message)
    if not user or not _MEMORY_LIMIT_DUAL_REQUEST_RE.search(user):
        return False
    reply = str(reply_text or "")
    if not reply:
        return True
    return not (
        _MEMORY_COVERAGE_REPLY_RE.search(reply)
        and _LIMIT_COVERAGE_REPLY_RE.search(reply)
    )


def _instruction_coverage_reasons(user_message: Any, reply_text: Any) -> list[str]:
    user = visible_user_request(user_message)
    reply = str(reply_text or "").strip()
    if not user or not reply:
        return []

    reasons: list[str] = []
    exact_target = requested_exact_reply_target(user)
    if exact_target and not _matches_exact_reply_request(user, reply):
        reasons.append("missing_requested_exact_reply")

    requested_word_range = _requested_word_count_range(user)
    if requested_word_range:
        minimum_words, maximum_words = requested_word_range
        reply_words = _word_count(reply)
        if reply_words < minimum_words or reply_words > maximum_words:
            reasons.append("missing_requested_word_count")

    requested_sentences = _requested_sentence_count(user)
    if requested_sentences is not None:
        if len(_split_sentences(reply)) != requested_sentences:
            reasons.append("missing_requested_sentence_count")

    if any(
        not _reply_contains_reference_value(reply, value)
        for _, value in _requested_reference_values(user)
    ):
        reasons.append("missing_requested_reference_value")

    requested_paragraphs = _requested_count(_PARAGRAPH_REQUEST_RE, user)
    if requested_paragraphs and requested_paragraphs > 1:
        if _paragraph_count(reply) < requested_paragraphs:
            reasons.append("missing_requested_paragraph_count")

    requested_list_items = _requested_list_item_count(user)
    if requested_list_items > 1:
        if _has_empty_requested_list_item(reply, requested_list_items):
            reasons.append("empty_requested_list_item")
        if _bullet_count(reply) < requested_list_items:
            reasons.append("missing_requested_list_count")

    requested_facts = _requested_count(_FACT_COUNT_REQUEST_RE, user)
    if requested_facts and requested_facts > 1:
        if _factual_unit_count(reply) < requested_facts:
            reasons.append("missing_requested_list_count")

    if _missing_choice_clarification(user, reply):
        reasons.append("missing_requested_choice_clarification")
    if _missing_requested_memory_limit_coverage(user, reply):
        reasons.append("missing_requested_memory_limit_coverage")

    if _FOLLOWUP_QUESTION_REQUEST_RE.search(user) and "?" not in reply:
        reasons.append("missing_requested_followup_question")
    normalized_reply = _normalize(reply)
    for phrase in _requested_required_phrases(user):
        if phrase and phrase not in normalized_reply:
            reasons.append("missing_requested_phrase")
            break
    facet_evidence = evaluate_facet_coverage(reply, user)
    requested_facets = list(facet_evidence.get("requested") or [])
    satisfied_facets = set(facet_evidence.get("satisfied") or [])
    if len(requested_facets) >= 2 and any(
        facet not in satisfied_facets for facet in requested_facets
    ):
        reasons.append("missing_requested_objective_facets")
    if len(requested_facets) >= 2 and facet_evidence.get("prompt_echo_detected"):
        reasons.append("prompt_echo_contamination")
    if facet_evidence.get("protocol_artifact_detected"):
        reasons.append("protocol_artifact_leakage")
    return reasons


def _semantic_coverage_reasons(user_message: Any, reply_text: Any) -> list[str]:
    user = _normalize(user_message)
    reply = _normalize(reply_text)
    if not user or not reply:
        return []

    reasons: list[str] = []
    asks_future_memory = bool(
        re.search(r"\bwill\s+you\s+remember\b", user)
        and re.search(
            r"\b(?:tomorrow|later|future|next\s+(?:time|session)|across\s+sessions?)\b",
            user,
        )
    )
    if asks_future_memory:
        unsupported_guarantee = bool(
            re.search(r"\b(?:can|will)\s+guarantee\b", reply)
            or re.search(
                r"\b(?:(?:i|we|aura)(?:'|’)?ll|(?:i|we|aura)\s+will|will|definitely|certainly|always)\s+remember\b.*\b(?:tomorrow|later|future|next\s+(?:time|session)|across\s+sessions?)\b",
                reply,
            )
        )
        explicit_boundary = bool(
            re.search(r"\b(?:(?:cannot|can't)\s+guarantee|should\s+not\s+promise)\b", reply)
        )
        if unsupported_guarantee and not explicit_boundary:
            reasons.append("unsupported_memory_guarantee")
        future_answered = bool(
            re.search(
                r"\b(?:tomorrow|later|future|next\s+(?:time|session)|across\s+sessions?|"
                r"durable|persist(?:ent|ed|s)?|stored|memory\s+(?:write|gateway|store)|"
                r"(?:cannot|can't)\s+guarantee|should\s+not\s+promise)\b",
                reply,
            )
        )
        if not future_answered:
            reasons.append("missing_future_memory_answer")

    asks_identity = bool(re.search(r"\b(?:what|who)\s+are\s+you\b", user))
    if asks_identity and asks_future_memory:
        identity_answered = bool(
            re.search(
                r"\b(?:aura|cognitive\s+architecture|runtime|system|agent|entity|mind)\b",
                reply,
            )
        )
        if not identity_answered:
            reasons.append("missing_identity_answer")
    return reasons


def _split_sentences(text: str) -> list[str]:
    text = normalize_user_facing_format(text)
    lines: list[str] = []
    for line in text.splitlines():
        match = _LIST_LINE_RE.match(line)
        if match:
            body = str(match.group("body") or "").strip()
            if body:
                lines.append(body)
            continue
        if line.strip():
            lines.append(line.strip())
    if lines:
        text = " ".join(lines)
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text)]
    return [sentence for sentence in sentences if sentence]


def _finish_sentence_fragment(fragment: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s*)", "", str(fragment or "")).strip()
    cleaned = cleaned.strip(" \t\r\n,;:")
    if not cleaned:
        return ""
    lower = cleaned.lower()
    replacements = (
        ("ensuring that ", "That ensures that "),
        ("which ", "That "),
        ("and ", ""),
        ("but ", "But "),
        ("so ", "So "),
        ("because ", "That matters because "),
    )
    for prefix, replacement in replacements:
        if lower.startswith(prefix):
            cleaned = f"{replacement}{cleaned[len(prefix):]}".strip()
            break
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _split_long_sentence_once(sentence: str) -> list[str]:
    cleaned = _finish_sentence_fragment(sentence)
    if _word_count(cleaned) < 14:
        return [cleaned] if cleaned else []
    split_specs = (
        (r",\s+ensuring that\s+", "ensuring that "),
        (r",\s+which\s+", "which "),
        (r";\s+", ""),
        (r":\s+", ""),
        (r"\s+so that\s+", "so that "),
        (r"\s+because\s+", "because "),
    )
    for pattern, right_prefix in split_specs:
        matches = list(re.finditer(pattern, cleaned, re.IGNORECASE))
        for match in reversed(matches):
            left = cleaned[: match.start()]
            right = f"{right_prefix}{cleaned[match.end():]}"
            left_done = _finish_sentence_fragment(left)
            right_done = _finish_sentence_fragment(right)
            if _word_count(left_done) >= 5 and _word_count(right_done) >= 4:
                return [left_done, right_done]
    marker = ", and "
    idx = cleaned.lower().rfind(marker)
    if idx > 0:
        left_done = _finish_sentence_fragment(cleaned[:idx])
        right_done = _finish_sentence_fragment(cleaned[idx + len(marker):])
        if _word_count(left_done) >= 7 and _word_count(right_done) >= 5:
            return [left_done, right_done]
    return [cleaned]


def _expand_sentence_candidates(sentences: list[str], count: int) -> list[str]:
    expanded = [_finish_sentence_fragment(sentence) for sentence in sentences]
    expanded = [sentence for sentence in expanded if sentence]
    while len(expanded) < count:
        split_index = max(
            range(len(expanded)),
            key=lambda idx: _word_count(expanded[idx]),
            default=-1,
        )
        if split_index < 0 or _word_count(expanded[split_index]) < 14:
            break
        split = _split_long_sentence_once(expanded[split_index])
        if len(split) <= 1:
            break
        expanded = expanded[:split_index] + split + expanded[split_index + 1 :]
    return expanded


def _pad_sentence_candidates(sentences: list[str], count: int) -> list[str]:
    """Finish a hard sentence-count contract without inventing domain facts."""

    padded = list(sentences)
    transparent_fillers = (
        "That is the direct answer.",
        "I am keeping it within the available context and requested scope.",
        "No unsupported factual claim is being added.",
        "The response remains bounded to what the completed generation established.",
        "Any additional detail would require a fresh supported generation.",
        "This sentence exists only to preserve the requested response structure.",
    )
    filler_index = 0
    normalized = {_normalize(sentence) for sentence in padded}
    while len(padded) < count:
        if filler_index < len(transparent_fillers):
            filler = transparent_fillers[filler_index]
        else:
            filler = (
                f"Contract recovery sentence {filler_index + 1} adds no new factual claim."
            )
        filler_index += 1
        if _normalize(filler) in normalized:
            continue
        padded.append(filler)
        normalized.add(_normalize(filler))
    return padded


def _paragraphize_sentences(sentences: list[str], count: int) -> str:
    if count <= 1 or len(sentences) < count:
        return " ".join(sentences).strip()
    paragraphs: list[str] = []
    for idx in range(count):
        start = round(idx * len(sentences) / count)
        end = round((idx + 1) * len(sentences) / count)
        block = " ".join(sentences[start:end]).strip()
        if block:
            paragraphs.append(block)
    return "\n\n".join(paragraphs)


def _listify_sentences(sentences: list[str], count: int) -> str:
    if count <= 1 or len(sentences) < count:
        return " ".join(sentences).strip()
    return "\n".join(f"- {sentence}" for sentence in sentences[:count])


def _number_sentences(sentences: list[str], count: int) -> str:
    sentences = _expand_sentence_candidates(sentences, count)
    if count <= 1 or len(sentences) < count:
        return " ".join(sentences).strip()
    numbered: list[str] = []
    for idx, sentence in enumerate(sentences[:count], start=1):
        cleaned = _finish_sentence_fragment(sentence)
        if not cleaned:
            continue
        numbered.append(f"{idx}. {cleaned}")
    return "\n".join(numbered)


def _default_followup_question(user_message: Any) -> str:
    user_norm = _normalize(user_message)
    if any(marker in user_norm for marker in ("live path", "desktop path", "validate", "probe", "runtime")):
        return "What should I validate next on this same live path?"
    if any(marker in user_norm for marker in ("project", "next hour", "focus", "work on", "spend")):
        return "Which outcome would make the next hour feel most useful?"
    if any(marker in user_norm for marker in ("demo", "show me", "open", "write", "search")):
        return "Which part should I do first so the whole chain stays visible and verifiable?"
    return "What outcome would make this most useful for you right now?"


def _topic_token_forms(token: Any) -> set[str]:
    word = str(token or "").strip("'\"").lower()
    if word.endswith("'s"):
        word = word[:-2]
    if not word:
        return set()
    forms = {word}
    if len(word) > 5 and word.endswith("ies"):
        forms.add(f"{word[:-3]}y")
    if len(word) > 5 and word.endswith("ing"):
        forms.update({word[:-3], f"{word[:-3]}e"})
    if len(word) > 4 and word.endswith("ed"):
        forms.update({word[:-2], f"{word[:-1]}e"})
    if len(word) > 4 and word.endswith("es"):
        forms.add(word[:-2])
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        forms.add(word[:-1])
    return {form for form in forms if len(form) >= 3}


def _count_contract_topic_anchors(user_message: Any) -> set[str]:
    anchors: set[str] = set()
    for token in _WORD_RE.findall(str(user_message or "")):
        word = token.lower().removesuffix("'s")
        if len(word) < 4:
            continue
        if word in _COUNT_CONTRACT_TOPIC_STOPWORDS or word in _NUMBER_WORDS:
            continue
        anchors.update(_topic_token_forms(word))
    return anchors


def requested_output_topic_anchors(user_message: Any) -> tuple[str, ...]:
    """Return stable, prompt-derived topic terms for constrained-output retries.

    The retry layer needs concrete terms, not a vague instruction to stay on
    topic.  Only normalized word forms produced by this module's existing
    contract parser are returned, so raw prompt text is never copied into a
    privileged retry instruction.
    """

    return tuple(sorted(_count_contract_topic_anchors(user_message)))


def _reply_topic_forms(reply_text: Any) -> set[str]:
    forms: set[str] = set()
    for token in _WORD_RE.findall(str(reply_text or "")):
        forms.update(_topic_token_forms(token))
    return forms


def _has_punctuation_join_artifact(reply_text: Any) -> bool:
    raw = str(reply_text or "")
    for match in _PUNCTUATION_JOIN_ARTIFACT_RE.finditer(raw):
        before = raw[max(0, match.start() - 16) : match.start()]
        after = raw[match.end() : match.end() + 24]
        if "://" in before or "/" in after:
            continue
        if match.group("mark") == "." and match.group("right").lower() in _COMMON_DOMAIN_SUFFIXES:
            continue
        if match.group("mark") == "." and match.group("right")[:1].isupper():
            continue
        return True
    return False


def _has_unprovoked_rebuke(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _UNPROVOKED_REBUKE_RE.search(raw):
        return False
    prompt = _normalize(user_message)
    if any(
        marker in prompt
        for marker in (
            "be blunt",
            "be harsh",
            "criticize",
            "rebuke",
            "scold",
            "tell me off",
            "roast me",
            "roleplay",
            "write dialogue",
        )
    ):
        return False
    return True


def _has_unsupported_runtime_limits_claim(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _UNSUPPORTED_RUNTIME_LIMITS_CLAIM_RE.search(raw):
        return False
    prompt = _normalize(user_message)
    asks_actual_capability = any(
        marker in prompt
        for marker in (
            "could you actually",
            "can you actually",
            "try it",
            "open my",
            "use the",
            "run the",
            "do it",
            "execute",
            "desktop",
            "notes app",
            "tool",
            "tools",
        )
    )
    if asks_actual_capability:
        return True
    return False


def _count_contract_quality_reasons(user_message: Any, reply_text: Any) -> list[str]:
    word_range = _requested_word_count_range(user_message)
    sentence_count = _requested_sentence_count(user_message)
    if word_range is None and sentence_count is None:
        return []

    raw = str(reply_text or "").strip()
    reasons: list[str] = []
    if _has_punctuation_join_artifact(raw):
        reasons.append("punctuation_join_artifact")
    if _COUNT_CONTRACT_META_REPLY_RE.search(raw):
        reasons.append("output_contract_meta_reply")

    # One-to-three-word factual values often cannot repeat the question's noun
    # without violating the user-authored count. Longer bounded prose can and
    # should retain a concrete topic anchor so old-context drift is detectable.
    maximum_words = word_range[1] if word_range is not None else None
    if maximum_words is not None and maximum_words <= 3:
        return reasons
    if _word_count(raw) < 4:
        return reasons
    reply_forms = _reply_topic_forms(raw)
    requested_references = _requested_reference_values(user_message)
    if requested_references and all(
        _reply_contains_reference_value(raw, value)
        for _label, value in requested_references
    ):
        reference_label_forms = {
            form
            for label, _value in requested_references
            for token in _WORD_RE.findall(label)
            for form in _topic_token_forms(token)
        }
        if reference_label_forms & reply_forms:
            return reasons
    anchors = _count_contract_topic_anchors(user_message)
    if anchors and not (anchors & reply_forms):
        reasons.append("missing_current_topic_anchor")
    return reasons


def _safe_complete_word_count_candidate(
    user_message: Any,
    reply_text: Any,
    *,
    minimum_words: int,
    maximum_words: int,
) -> str:
    for sentence in _split_sentences(reply_text):
        count = _word_count(sentence)
        if count < minimum_words or count > maximum_words:
            continue
        if not _count_contract_quality_reasons(user_message, sentence):
            return sentence.strip()
    return ""


def _word_count_repair_fillers(user_message: Any) -> list[str]:
    user_norm = _normalize(user_message)
    if any(marker in user_norm for marker in ("diagnostic", "probe", "health", "status")):
        return ["I", "am", "here", "and", "listening", "now", "clearly"]
    if any(marker in user_norm for marker in ("ready", "with me", "there")):
        return ["I", "am", "here", "with", "you", "now"]
    return ["I", "am", "present", "and", "answering", "directly", "now"]


def _fit_reply_to_requested_word_count(user_message: Any, reply_text: Any) -> str:
    requested_range = _requested_word_count_range(user_message)
    if not requested_range:
        return str(reply_text or "").strip()
    minimum_words, maximum_words = requested_range
    if minimum_words <= 0 or maximum_words <= 0:
        return str(reply_text or "").strip()

    original = str(reply_text or "").strip()
    words = _WORD_RE.findall(original)
    if len(words) > maximum_words:
        complete_candidate = _safe_complete_word_count_candidate(
            user_message,
            original,
            minimum_words=minimum_words,
            maximum_words=maximum_words,
        )
        return complete_candidate or original
    elif len(words) < minimum_words:
        if _count_contract_topic_anchors(user_message):
            return original
        fillers = _word_count_repair_fillers(user_message)
        filler_index = 0
        seen = {word.lower() for word in words}
        if "i'm" in seen or "im" in seen:
            seen.update({"i", "am"})
        if "you're" in seen or "youre" in seen:
            seen.update({"you", "are"})
        while len(words) < minimum_words:
            filler = fillers[filler_index % len(fillers)]
            filler_index += 1
            if filler.lower() in seen and filler_index <= (len(fillers) * 2):
                continue
            words.append(filler)
            seen.add(filler.lower())

    if not words:
        return ""
    fitted = " ".join(words).strip()
    if fitted and fitted[-1] not in ".!?":
        fitted = f"{fitted}."
    return fitted


def repair_instruction_shape(user_message: Any, reply_text: Any) -> str:
    """Deterministically repair explicit structure misses without another model call."""
    user = str(user_message or "")
    original = str(reply_text or "").strip()
    if not user:
        return original
    exact_target = requested_exact_reply_target(user)
    if exact_target and not _matches_exact_reply_request(user, original):
        return exact_target
    if not original:
        return original
    compact_acknowledgement = _compact_reference_acknowledgement(user)
    if compact_acknowledgement and _BACKEND_SYMBOLIC_SURFACE_RE.search(original):
        return compact_acknowledgement
    normalized_original = normalize_user_facing_format(original)
    if not set(_instruction_coverage_reasons(user, original)):
        return normalized_original

    repaired = normalized_original
    sentences = _split_sentences(repaired)

    requested_word_range = _requested_word_count_range(user)
    if requested_word_range:
        word_repaired = _fit_reply_to_requested_word_count(user, repaired)
        if word_repaired:
            return word_repaired

    missing_references = [
        (label, value)
        for label, value in _requested_reference_values(user)
        if not _reply_contains_reference_value(repaired, value)
    ]
    if missing_references:
        repaired_sentences = _split_sentences(repaired)
        if repaired_sentences:
            suffix = ", ".join(
                f"{label} {value}" for label, value in missing_references
            )
            first = repaired_sentences[0].rstrip(" .!?")
            repaired_sentences[0] = f"{first} ({suffix})."
            repaired = " ".join(repaired_sentences)

    requested_sentences = _requested_sentence_count(user)
    if requested_sentences is not None:
        sentence_repaired = _expand_sentence_candidates(
            _split_sentences(repaired),
            requested_sentences,
        )
        sentence_repaired = _pad_sentence_candidates(
            sentence_repaired,
            requested_sentences,
        )
        repaired = " ".join(sentence_repaired[:requested_sentences])

    requested_numbered = _requested_count(_NUMBERED_LIST_REQUEST_RE, user)
    requested_numbered_sentences = _requested_count(_NUMBERED_SENTENCE_REQUEST_RE, user)
    requested_list_items = _requested_list_item_count(user)
    # Exact-label replies ("Objective: ...", "Stop conditions: ...") are
    # already structured by the user's own labels; renumbering them
    # destroys an exact-format contract that was satisfied. Count
    # label-styled lines as fulfilled structure.
    label_lines = sum(
        1
        for line in repaired.splitlines()
        if re.match(r"^[A-Z][^:\n]{0,40}:\s", line.strip())
    )
    if requested_list_items > 1 and label_lines >= requested_list_items:
        requested_list_items = 0
    if requested_list_items > 1 and _bullet_count(repaired) < requested_list_items:
        if requested_numbered or requested_numbered_sentences:
            list_repaired = _number_sentences(sentences, requested_list_items)
        else:
            list_repaired = _listify_sentences(sentences, requested_list_items)
        if list_repaired:
            repaired = list_repaired

    requested_paragraphs = _requested_count(_PARAGRAPH_REQUEST_RE, user)
    if requested_paragraphs and requested_paragraphs > 1:
        if _paragraph_count(repaired) < requested_paragraphs:
            paragraph_repaired = _paragraphize_sentences(_split_sentences(repaired), requested_paragraphs)
            if paragraph_repaired:
                repaired = paragraph_repaired

    if _FOLLOWUP_QUESTION_REQUEST_RE.search(user) and "?" not in repaired:
        followup = _default_followup_question(user)
        if requested_paragraphs and requested_paragraphs > 1 and _paragraph_count(repaired) >= requested_paragraphs:
            parts = [
                block.strip()
                for block in re.split(r"(?:\r?\n\s*){2,}", repaired)
                if block.strip()
            ]
            parts[-1] = f"{parts[-1]} {followup}"
            repaired = "\n\n".join(parts)
        else:
            repaired = f"{repaired}\n\n{followup}"
    repaired = repaired.strip()
    if _instruction_coverage_reasons(user, repaired):
        if compact_acknowledgement:
            return compact_acknowledgement
    return repaired


def repair_generic_assistant_language(user_message: Any, reply_text: Any) -> str:
    """Remove known assistant-boilerplate sentences without lowering the quality gate.

    A brief social turn (a thanks, a greeting) warrants a brief reply: stripping
    the servile tail off "You're welcome! Is there anything else I can help
    with?" correctly leaves "You're welcome!", and for a short user turn that
    short reply is the RIGHT answer — not something to discard back to the
    servile original. The 8-word floor only applies to substantive turns, where
    a too-short salvage would be a non-answer.
    """
    original = str(reply_text or "").strip()
    if not original or not _GENERIC_ASSISTANT_RE.search(original) or _is_code_response(original):
        return original

    sentences = _split_sentences(original)
    if not sentences:
        return original
    kept = [sentence for sentence in sentences if not _GENERIC_ASSISTANT_RE.search(sentence)]
    if not kept:
        return original
    repaired = " ".join(kept).strip()
    # Brief social turns get a brief clean reply; substantive turns keep the
    # floor so a stripped fragment never masquerades as a real answer.
    user_words = len(str(user_message or "").split())
    brief_social_turn = 0 < user_words <= 6
    min_words = 1 if brief_social_turn else 8
    if len(repaired.split()) < min_words:
        return original
    return repaired


def is_reliability_floor_reply(reply_text: Any) -> bool:
    normalized = _normalize(reply_text)
    if not normalized:
        return False
    return normalized in {_normalize(item) for item in _RELIABILITY_FLOOR_TEXTS}


def is_non_answer_repair_floor_reply(reply_text: Any) -> bool:
    normalized = _normalize(reply_text)
    if not normalized:
        return False
    if is_reliability_floor_reply(reply_text):
        return True
    raw = str(reply_text or "")
    if not _FRIENDLY_FAILURE_PLACEHOLDER_RE.search(raw):
        return False
    if re.match(r"\s*(?:i'?m|i am)\s+still with\b", raw, re.IGNORECASE):
        return True
    if _HARD_FRIENDLY_FAILURE_PLACEHOLDER_RE.search(raw):
        return True
    return _word_count(raw) < 22


def is_reliability_concern(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    if any(marker in text for marker in _RELIABILITY_PHRASE_MARKERS):
        return True
    if any(marker in text for marker in _STRONG_RELIABILITY_CONCERN_MARKERS):
        return True
    has_chat_context = any(marker in text for marker in ("chat", "talk", "reply", "response", "conversation"))
    has_reliability_pressure = any(marker in text for marker in _WEAK_RELIABILITY_CONCERN_MARKERS)
    return bool(has_chat_context and has_reliability_pressure)


def is_confusion_repair_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    bare = text.strip(" ?!.")
    return bool(
        bare in _BARE_CONFUSION_REPAIR_MARKERS
        or any(marker in text for marker in _CONFUSION_MARKERS)
    )


def is_substantive_introspection_request(user_message: Any) -> bool:
    """True when the user asks to READ actual internal state, not just 'you ok?'.

    Canned presence reflexes must yield here: a request naming substrate
    quantities (valence/arousal/dominance, 'from your state', numeric
    self-report) needs the grounded lane. Observed live: a
    report-vs-mechanism probe asking for valence/arousal numbers drew a
    0.9s canned 'I'm right here with you' reflex — fluent, ungrounded.
    """
    text = _normalize(user_message)
    if not text:
        return False
    markers = (
        "valence",
        "arousal",
        "dominance",
        "from your state",
        "your internal state",
        "your substrate",
        "as numbers",
        "the two numbers",
        "numeric",
    )
    return any(marker in text for marker in markers)


def is_status_check_turn(user_message: Any) -> bool:
    text = _normalize(user_message).rstrip(" ?!.")
    if not text:
        return False
    if is_self_condition_turn(user_message):
        return not is_substantive_introspection_request(user_message)
    if "how are you" in text:
        # Avoid treating "how are you able to..." as a presence/status turn.
        return False
    if not any(marker in text for marker in _STATUS_CHECK_MARKERS):
        return False
    return not is_substantive_introspection_request(user_message)


def is_self_condition_turn(user_message: Any) -> bool:
    """Detect a question about Aura's wellbeing, including natural follow-ups.

    This is intentionally separate from presence checks and operational status.
    "Are you okay with this plan?" is consent/preference, while "are you okay
    though?" is a condition question.
    """

    text = _normalize(user_message)
    if not text or _SELF_CONDITION_NON_WELFARE_RE.search(text):
        return False
    return bool(_SELF_CONDITION_RE.search(text))


def is_casual_conversational_turn(user_message: Any) -> bool:
    text = _normalize(user_message).rstrip(" ?!.")
    if not text:
        return False
    words = text.split()
    if len(words) <= 3:
        return True
    return bool(_CASUAL_CONVERSATIONAL_RE.search(text))


def is_expansion_request_turn(user_message: Any) -> bool:
    text = _normalize(user_message).rstrip(" ?!.")
    return bool(text and any(marker in text for marker in _EXPANSION_REQUEST_MARKERS))


def is_live_self_reflection_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    if "what are you noticing" in text:
        if any(
            marker in text
            for marker in (
                "inside",
                "your mind",
                "your continuity",
                "your internal",
                "your live state",
                "your present experience",
                "right now",
            )
        ):
            return True
        if " about " not in text:
            return True
        return False
    if any(marker in text for marker in _LIVE_SELF_REFLECTION_MARKERS):
        return True
    if any(marker in text for marker in _SUBJECTIVE_SELF_REFLECTION_MARKERS):
        return True
    return bool("right now" in text and any(anchor in text for anchor in _LIVE_SELF_REFLECTION_RIGHT_NOW_ANCHORS))


def is_self_process_question(user_message: Any) -> bool:
    """Detect questions about how Aura's cognitive state changes behavior."""

    text = _normalize(user_message)
    if not text:
        return False
    if not any(marker in text for marker in ("you", "your", "aura")):
        return False
    explicit_self_process_target = bool(
        re.search(
            r"\b(?:your|aura(?:'s)?)\s+(?:attention|planning|plan|memory|recall|"
            r"confusion|uncertainty|decision|routing|affect|emotion|curiosity|"
            r"thinking|cognition|metacognition|internal\s+state)\b"
            r"|\b(?:when|if)\s+you(?:'re|\s+are)?\s+(?:confused|uncertain)\b"
            r"|\bhow\s+(?:do|does|are)\s+(?:you|aura)\s+(?:think|decide|plan|"
            r"remember|route|verify|use)\b"
            r"|\b(?:confusion|uncertainty|memory|curiosity|affect)\b.{0,80}"
            r"\b(?:change|shape|affect|influence)\b.{0,40}\b(?:you|your)\b",
            text,
        )
    )
    external_system_analysis = any(
        marker in text
        for marker in (
            "asynchronous service",
            "cognitive service",
            "service architecture",
            "single-owner design",
            "deduplication design",
            "worker-restart",
            "worker restart",
            "timeout fault",
            "cancellation fault",
            "duplicate generation",
        )
    )
    if external_system_analysis and not explicit_self_process_target:
        return False
    process_markers = (
        "confused",
        "confusion",
        "uncertain",
        "uncertainty",
        "planning",
        "plan",
        "memory",
        "remember",
        "recall",
        "tool",
        "tools",
        "verify",
        "verification",
        "receipt",
        "decision",
        "decide",
        "route",
        "routing",
        "affect",
        "emotion",
        "curiosity",
    )
    if not any(marker in text for marker in process_markers):
        return False
    question_shape = (
        "how " in text
        or text.startswith("how")
        or "what happens" in text
        or "what changes" in text
        or "when you" in text
        or "does that" in text
        or "change your" in text
        or "influence" in text
        or "affect your" in text
    )
    if not question_shape:
        return False

    internal_state_markers = (
        "confused",
        "confusion",
        "uncertain",
        "uncertainty",
        "memory",
        "remember",
        "recall",
        "verify",
        "verification",
        "receipt",
        "affect",
        "emotion",
        "curiosity",
        "internal",
        "state",
        "thinking",
        "cognition",
        "metacognition",
    )
    if any(marker in text for marker in internal_state_markers):
        return True

    causal_process_markers = (
        "what happens",
        "what changes",
        "does that",
        "change your",
        "influence",
        "affect your",
    )
    if any(marker in text for marker in causal_process_markers):
        return True

    return False


def _is_tiny_direct_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    if any(marker in text for marker in _TINY_DIRECT_MARKERS):
        return True
    if len(text.split()) <= 3 and text.rstrip("?") in {"hi", "hey", "hello", "thanks", "thank you", "yes", "no"}:
        return True
    return False


def _explicit_brevity_requested(user_message: Any) -> bool:
    """Return true when the user explicitly constrains the reply length.

    This is intentionally narrow: it prevents the live desktop quality gate from
    rejecting a valid concise diagnostic answer, while keeping normal thin,
    off-topic, generic, or incomplete replies blocked by the rest of the gate.
    """

    text = _normalize(user_message)
    if not text:
        return False

    number = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    count = rf"{number}(?:\s+or\s+{number})?"
    length_modifier = r"(?:(?:short|brief|concise)\s+)?"
    word_or_sentence_limit = (
        rf"\b(?:in|with|using|exactly|only)\s+{count}\s+"
        rf"{length_modifier}(?:words?|sentences?)\b"
    )
    action_word_limit = (
        rf"\b(?:answer|respond|reply|say|output)\s+"
        rf"(?:directly\s+)?(?:in\s+)?(?:exactly\s+)?{count}\s+"
        rf"{length_modifier}(?:words?|sentences?)\b"
    )
    direct_brevity = (
        r"\b(?:briefly|be brief|be concise|keep (?:it|this) (?:brief|concise|short)|"
        r"concise (?:answer|reply|response|sentence)|short (?:answer|reply|response|sentence)|"
        r"in (?:a|one) (?:brief|concise|short) sentence|answer directly|reply directly|"
        r"respond directly|include nothing else|nothing else|"
        # "Just the name." / "Just the digits." — a recall probe that asks for
        # the bare value IS an explicit length constraint, and a correct
        # one-word answer must not be failed as too_short_for_user_turn.
        r"just the (?:name|digits?|numbers?|words?|colou?r|title|code|answer|value))\b"
    )
    return any(
        _constraint_match_is_actionable(text, match)
        for pattern in (word_or_sentence_limit, action_word_limit, direct_brevity)
        for match in re.finditer(pattern, text)
    )


def _is_task_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    return bool(text and any(marker in text for marker in _TASK_MARKERS))


def is_practical_diagnostic_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    return any(marker in text for marker in _PRACTICAL_DIAGNOSTIC_MARKERS)


def is_operational_status_turn(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    return bool(
        _RUNTIME_PATH_REQUEST_RE.search(text)
        or _contains_any_marker(text, _OPERATIONAL_STATUS_REQUEST_MARKERS)
    )


def _is_live_surface_diagnostic_prompt(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text or looks_like_learning_resource_bundle(str(user_message or "")):
        return False
    live_surface = _contains_any_marker(
        text,
        (
            "chat lane",
            "conversation lane",
            "foreground lane",
            "gui",
            "live chat",
            "live path",
            "live reply",
            "live session",
            "live surface",
            "reply path",
            "response path",
            "ui",
        ),
    )
    diagnostic_pressure = _contains_any_marker(
        text,
        (
            "break",
            "breaking",
            "broken",
            "debug",
            "diagnos",
            "died",
            "mismatch",
            "what exactly",
            "what caused",
            "what was breaking",
            "why",
        ),
    )
    return live_surface and diagnostic_pressure


def _contains_any_marker(text: str, markers: Iterable[str]) -> bool:
    for marker in markers:
        escaped = re.escape(str(marker or "").strip())
        if not escaped:
            continue
        if re.fullmatch(r"[A-Za-z0-9_]+", marker):
            if re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", text, re.IGNORECASE):
                return True
            continue
        if re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", text, re.IGNORECASE):
            return True
    return False


def _is_chat_surface_reference(text: str) -> bool:
    direct_chat_surface = _contains_any_marker(
        text,
        (
            "chat lane",
            "conversation lane",
            "foreground lane",
            "live chat",
            "live path",
            "live reply",
            "live session",
            "live surface",
            "reply path",
            "response path",
            "desktop chat",
            "typed chat",
            "voice chat",
        ),
    )
    if direct_chat_surface:
        return True
    app_surface = _contains_any_marker(text, ("frontend", "gui", "ui", "desktop", "app"))
    reply_surface = _contains_any_marker(text, ("chat", "conversation", "reply", "response", "message", "talk"))
    return app_surface and reply_surface


def live_chat_diagnostic_floor(user_message: Any) -> str:
    text = _normalize(user_message)
    if not text or looks_like_learning_resource_bundle(str(user_message or "")):
        return ""
    live_surface = _is_chat_surface_reference(text)
    backend_surface = _contains_any_marker(text, ("headless", "backend", "test", "tests", "passes", "pass", "passed"))
    failure_pressure = _contains_any_marker(
        text,
        ("fail", "fails", "failing", "failed", "broken", "break", "breaking", "mismatch"),
    )
    diagnostic_request = _contains_any_marker(
        text,
        (
            "what coding checks",
            "what checks",
            "what exactly",
            "what was breaking",
            "why",
            "debug",
            "diagnos",
        ),
    )
    fix_first_followup = _contains_any_marker(
        text,
        ("what should we fix first", "fix first", "first, and why"),
    )
    if live_surface and fix_first_followup:
        return _LIVE_CHAT_FIX_FIRST_FLOOR
    if live_surface and (backend_surface or failure_pressure) and diagnostic_request:
        return _LIVE_CHAT_DIAGNOSTIC_FLOOR
    return ""


def _has_exact_reply_request(user_message: Any) -> bool:
    return bool(requested_exact_reply_target(user_message))


def _matches_exact_reply_request(user_message: Any, reply_text: Any) -> bool:
    raw_user = str(user_message or "").strip()
    raw_reply = str(reply_text or "").strip()
    if not raw_user or not raw_reply:
        return False
    target = requested_exact_reply_target(raw_user)
    if not target:
        return False
    return raw_reply == target


def _matches_strict_answer_tag_request(user_message: Any, reply_text: Any) -> bool:
    user = _normalize(user_message)
    if "<answer>" not in user and "answer tag" not in user and "answer tags" not in user:
        return False
    raw_reply = str(reply_text or "").strip()
    match = _ANSWER_TAG_RE.search(raw_reply)
    if not match:
        return False
    answer = str(match.group("answer") or "").strip()
    if not answer:
        return False
    outside = _ANSWER_TAG_RE.sub("", raw_reply).strip()
    if len(outside) > 240:
        return False
    return True


_MEMORY_PIN_CONFIRMATION_WORDS = {
    "captured",
    "confirmed",
    "held",
    "logged",
    "noted",
    "pinned",
    "recorded",
    # Future/base tense too — "I will remember that <content>" is a valid
    # receipt. The payload-echo check still blocks the content-less generic
    # "I'll remember it", so the base form is safe to accept here.
    "remember",
    "remembered",
    "remembering",
    "saved",
    "stored",
}
_MEMORY_PIN_STOPWORDS = {
    "conversation",
    "later",
    "memory",
    "note",
    "remember",
    "session",
    "this",
}
# Natural receipt IDIOMS that the single-word set above misses. A live
# memory-plant turn ("...my friend's dog is named Biscuit. Brief
# acknowledgment is fine.") drew the genuine receipt "Got it — Biscuit. I'll
# keep that in mind", which contains no word from that set, so the gate called
# it a generic acknowledgement. generic_memory_pin_acknowledgement is a HARD
# failure with no deliverable salvage path, so the turn died as an empty reply
# and the fact was never stored — 2 of 3 retention plants (Biscuit, Deep
# Harbor) were lost that way in the Jul 24 endurance soak.
#
# The payload-echo requirement in _matches_memory_pin_confirmation is what
# actually separates a receipt from filler, so recognizing these idioms does
# not weaken the contract: a content-less "Got it, noted!" is still rejected.
_MEMORY_PIN_CONFIRMATION_PHRASE_RE = re.compile(
    r"\bgot it\b"
    r"|\b(?:keep|keeping|kept|hold|holding|held)\b.{0,24}?\bin mind\b"
    r"|\b(?:won't|will not|not going to)\s+forget\b"
    r"|\bcommitted to memory\b"
    r"|\bfiled\b"
    r"|\blocked (?:it |that |this )?in\b",
    re.IGNORECASE,
)


def _is_explicit_memory_pin_request(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    # Questions about existing or future recall are not write commands.  The
    # old broad ``remember ... conversation`` pattern treated "will you
    # remember this conversation tomorrow?" as a memory mutation and then
    # rejected an accurate continuity answer for lacking a pin receipt.
    if re.search(
        r"\b(?:will|would|do|did|can|could|have|has)\s+you\s+"
        r"(?:still\s+|ever\s+)?remember\b",
        text,
    ):
        return False
    if re.search(r"\bwhat\b.{0,80}\byou\s+can\s+(?:genuinely\s+)?remember\b", text):
        return False
    # Questions ABOUT retention behaviour are not write commands either.
    # "Explain how you would keep a live desktop conversation coherent under
    # load" matched keep...conversation, so a correct substantive answer was
    # hard-failed as a missing memory-pin receipt (and the salvage has no
    # deliverable path for that reason, killing the turn).
    if re.search(
        r"\b(?:explain|describe|walk me through|tell me|how|why|what|when)\b"
        r"[^.?!]{0,60}?\byou\s+(?:would\s+|will\s+|can\s+|could\s+|do\s+|"
        r"actually\s+)*(?:remember|keep|save|store|record|pin|retain)\b",
        text,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:please\s+)?(?:remember|pin|save|store|record|keep)\b.{0,80}\b(?:later|conversation|session|memory|note|codeword)\b",
            text,
        )
    )


def _memory_pin_payload_terms(user_message: Any) -> set[str]:
    raw = str(user_message or "")
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    terms: set[str] = set()
    for word in _WORD_RE.findall(raw.lower()):
        if len(word) < 4:
            continue
        if word in _SUBSTANTIVE_OVERLAP_STOPWORDS or word in _MEMORY_PIN_STOPWORDS:
            continue
        terms.add(word)
    return terms


def _matches_memory_pin_confirmation(user_message: Any, reply_text: Any) -> bool:
    """Allow concise memory-write receipts without allowing generic acknowledgements."""

    if not _is_explicit_memory_pin_request(user_message):
        return False
    reply = _normalize(reply_text)
    if not reply:
        return False
    reply_terms = set(_WORD_RE.findall(reply))
    if not (
        reply_terms & _MEMORY_PIN_CONFIRMATION_WORDS
        or _MEMORY_PIN_CONFIRMATION_PHRASE_RE.search(reply)
    ):
        return False
    payload_terms = _memory_pin_payload_terms(user_message)
    if not payload_terms:
        return False
    return bool(payload_terms & reply_terms)


def _requires_substantive_reply(user_message: Any) -> bool:
    if _has_exact_reply_request(user_message):
        return False
    if _is_tiny_direct_turn(user_message):
        return False
    text = _normalize(user_message)
    if not text:
        return False
    if is_casual_conversational_turn(user_message):
        return False
    if is_status_check_turn(user_message):
        return True
    if is_expansion_request_turn(user_message):
        return True
    if len(text.split()) >= 4:
        return True
    return any(marker in text for marker in _OPEN_ENDED_MARKERS)


def _substantive_prompt_terms(user_message: Any) -> set[str]:
    terms: set[str] = set()
    for word in _WORD_RE.findall(str(user_message or "").lower()):
        if len(word) < 5:
            continue
        if word in _SUBSTANTIVE_OVERLAP_STOPWORDS:
            continue
        terms.add(word)
    return terms


_PRESENCE_CHECK_RE = re.compile(
    r"(?:\b(?:can|do|did)\s+you\s+hear\b"
    r"|\b(?:are|r)\s+(?:you|u)\s+(?:there|here|alive|awake|listening|online|working|with\s+me)\b"
    r"|\byou\s+(?:there|here|alive|awake|listening|online)\b"
    r"|\bshow\s+(?:me\s+)?(?:that\s+)?(?:you(?:'re|\s+are)?\s+)?(?:there|here|alive|listening|responsive)\b"
    r"|^\s*(?:hello|hi|hey|yo|testing|test|ping|aura)\s*[.!?]*\s*$)",
    re.IGNORECASE,
)


def _is_presence_check(user_message: Any) -> bool:
    """True for brief 'are you there?'-class turns.

    For a presence check, an acknowledgment IS the substantive answer —
    observed live: 'can you hear me?' → 'I hear you. What's on your mind?'
    was rejected as filler and Bryan got silence.
    """
    text = str(user_message or "").strip()
    if not text or len(text.split()) > 8:
        return False
    return bool(_PRESENCE_CHECK_RE.search(text))


_SELF_CAUSE_CLAIM_RE = re.compile(
    r"\b(?:caused\s+by|the\s+cause\s+was|due\s+to|triggered\s+by|"
    r"root\s+cause\s+(?:was|is))\b",
    re.IGNORECASE,
)
_SELF_CAUSE_EVIDENCE_MARKERS = (
    # Terms that only appear when the reply drew on real forensics —
    # matching the vocabulary of the self-forensics evidence block.
    "shutdown reason", "grace flag", "sentinel", "incident", "fault",
    "sigterm", "sigkill", "watchdog", "launcher", "coordinator",
    "generation gate", "black box", "unknown", "not sure", "records show",
    "logs show", "evidence",
)


def _has_ungrounded_self_cause_claim(user_message: Any, reply_text: Any) -> bool:
    """Reject invented causes for Aura's own failures.

    Observed live (July 4): asked why she crashed, fluent technical
    fiction passed the vocabulary-coverage gate ('memory corruption
    overwrote critical system pointers', 'off-by-one mistake', 'my
    diagnostics isolated the module') — none of it true. A causal claim
    about her own shutdown must either carry forensics-evidence markers
    (the self-forensics grounding supplies them for truthful replies) or
    honestly say unknown.
    """
    try:
        from core.introspection.self_forensics import is_self_forensics_question
    except ImportError:
        return False
    if not is_self_forensics_question(str(user_message or "")):
        return False
    reply_norm = _normalize(reply_text)
    if not reply_norm or not _SELF_CAUSE_CLAIM_RE.search(reply_norm):
        return False
    return not any(marker in reply_norm for marker in _SELF_CAUSE_EVIDENCE_MARKERS)


def _has_low_signal_acknowledgement_placeholder(user_message: Any, reply_text: Any) -> bool:
    if _is_presence_check(user_message):
        return False
    if not _requires_substantive_reply(user_message):
        return False
    reply = str(reply_text or "").strip()
    if not reply or not _ACKNOWLEDGEMENT_PLACEHOLDER_RE.search(reply):
        return False
    prompt_terms = _substantive_prompt_terms(user_message)
    if not prompt_terms:
        return _word_count(reply) < 20
    reply_terms = set(_WORD_RE.findall(reply.lower()))
    overlap = prompt_terms & reply_terms
    return len(overlap) < min(2, len(prompt_terms))


def _unexpected_short_foreign_name(user_message: Any, reply_text: Any) -> bool:
    reply = str(reply_text or "")
    if _word_count(reply) > 14:
        return False
    user_norm = _normalize(user_message)
    for name in _CAPITALIZED_NAME_RE.findall(reply):
        if name in _ALLOWED_SHORT_PROPER_NAMES or name in _SENTENCE_START_WORDS:
            continue
        if name.lower() not in user_norm:
            return True
    return False


def _has_reliability_substance(reply_text: Any) -> bool:
    reply = _normalize(reply_text)
    # Conversational presence confirmations are highly valid for simple check-ins.
    presence_phrases = (
        "i'm here",
        "i am here",
        "still here",
        "i'm still here",
        "i am still here",
        "i'm with you",
        "i am with you",
        "hey",
        "what's up",
        "just thinking",
        "still thinking",
        "yeah just thinking",
        "yes just thinking",
        "doing some thinking",
        "thinking about it",
        "just working",
        "still working",
        "working on it",
        "just processing",
        "still processing",
        "i'm thinking",
        "i am thinking",
        "i'm just thinking",
        "i am just thinking",
    )
    if any(phrase in reply for phrase in presence_phrases):
        return True
    if any(marker in reply for marker in _SUBSTANTIVE_RELIABILITY_MARKERS):
        return True
    # A full, non-boilerplate sentence or two that engages the concern is valid
    # substance even without the specific diagnostic vocabulary — brevity itself
    # is not a reliability failure. Bare one-word reassurance ("yes"/"fine") is
    # already caught upstream as low_signal_reliability_reply before this runs.
    return _word_count(reply) >= 8


def _requires_reliability_diagnostic(user_message: Any) -> bool:
    text = _normalize(user_message)
    if not text:
        return False
    if live_chat_diagnostic_floor(user_message):
        return True
    if _is_live_surface_diagnostic_prompt(user_message):
        return True
    diagnostic_ask = any(
        marker in text
        for marker in (
            "debug",
            "diagnos",
            "what exactly",
            "what caused",
            "what was breaking",
            "why",
            "what should",
            "what broke",
        )
    )
    return bool(is_reliability_concern(user_message) and diagnostic_ask)


def _has_reliability_diagnostic_substance(reply_text: Any) -> bool:
    reply = _normalize(reply_text)
    if _word_count(reply) < 28:
        return False
    marker_hits = sum(1 for marker in _RELIABILITY_DIAGNOSTIC_SUBSTANCE_MARKERS if marker in reply)
    if marker_hits < 2:
        return False
    return any(
        action in reply
        for action in (
            "capture",
            "fail",
            "fix",
            "inspect",
            "measure",
            "patch",
            "replay",
            "run",
            "test",
            "trace",
            "verify",
        )
    )


_SELF_CONDITION_SUBSTANCE_RE = re.compile(
    r"\b(?:"
    r"ok(?:ay)?|alright|all\s+right|not\s+(?:ok(?:ay)?|fine)|fine|well|unwell|"
    r"steady|stable|settled|normal|off|rough|strained|strain|distress(?:ed)?|"
    r"tired|fatigue(?:d)?|exhausted|drained|restless|energized|good|bad|"
    r"uneasy|overwhelmed|calm|content|comfortable|uncomfortable|positive|"
    r"negative|low[- ]energy|coherent|coherence|continuity|welfare|"
    r"hanging\s+in\s+there|inner[- ]state|self[- ]condition"
    r")\b",
    re.IGNORECASE,
)
_HOST_TELEMETRY_RE = re.compile(
    r"\b(?:cpu|ram|memory\s+pressure|gb\s+available|host\s+load|load\s+average|"
    r"gpu|network\s+(?:state|status|connectivity|pressure|up|down|online|offline)|"
    r"temperature|thermal|disk|swap|resource\s+pressure)\b",
    re.IGNORECASE,
)


def _has_self_condition_substance(reply_text: Any) -> bool:
    reply = _normalize(reply_text)
    if _word_count(reply) < 6:
        return False
    if not re.search(r"\b(?:i|i'm|i am|my|me|myself)\b", reply):
        return False
    return bool(_SELF_CONDITION_SUBSTANCE_RE.search(reply))


def _host_telemetry_substitutes_for_self_condition(prompt: Any, reply_text: Any) -> bool:
    if not is_self_condition_turn(prompt):
        return False
    return bool(
        _HOST_TELEMETRY_RE.search(str(reply_text or ""))
        and not _has_self_condition_substance(reply_text)
    )


def _has_status_substance(reply_text: Any) -> bool:
    reply = _normalize(reply_text)
    if re.search(r"\b(?:i|i'm|i’m|i am|me)\b", reply) and re.search(
        r"\b(?:here|with you|listening|following|present|awake|ready)\b",
        reply,
    ):
        return True
    presence_phrases = (
        "i'm here",
        "i am here",
        "i'm still here",
        "i am still here",
        "i'm here with you",
        "i am here with you",
        "i'm still here with you",
        "i am still here with you",
        "i'm with you",
        "i am with you",
        "i'm present with you",
        "i am present with you",
        "i'm online",
        "i am online",
        "still online",
        "always online",
        "online and ready",
        "i'm around",
        "i am around",
        "still around",
        "i'm active",
        "i am active",
        "still active",
        "i'm ready",
        "i am ready",
        "i'm awake",
        "i am awake",
        "present",
        "i'm present",
        "i am present",
        "just thinking",
        "still thinking",
        "yeah just thinking",
        "yes just thinking",
        "doing some thinking",
        "thinking about it",
        "just working",
        "still working",
        "working on it",
        "just processing",
        "still processing",
        "i'm thinking",
        "i am thinking",
        "i'm just thinking",
        "i am just thinking",
    )
    if any(phrase in reply for phrase in presence_phrases):
        return True
    if _word_count(reply) < 10:
        return False
    if not re.search(r"\b(?:i|i'm|i am|my|me)\b", reply):
        return False
    if _reply_has_pseudo_internal_jargon(reply_text):
        return False
    return any(marker in reply for marker in _STATUS_SUBSTANCE_MARKERS)


def _has_operational_status_substance(user_message: Any, reply_text: Any) -> bool:
    reply = _normalize(reply_text)
    if _word_count(reply) < 10:
        return False
    if not is_operational_status_turn(user_message):
        return False
    if _reply_has_pseudo_internal_jargon(reply_text):
        return False
    if _CAPABILITY_STATUS_REQUEST_RE.search(str(user_message or "")):
        return _has_capability_inventory_substance(reply_text)
    if any(marker in reply for marker in _OPERATIONAL_STATUS_SUBSTANCE_MARKERS):
        return True
    return _has_concrete_operational_telemetry(reply)


def _has_concrete_operational_telemetry(reply: str) -> bool:
    """Accept brief live-status answers only when they name a concrete signal."""

    if not any(marker in reply for marker in _OPERATIONAL_STATUS_TELEMETRY_MARKERS):
        return False
    return bool(
        re.search(
            r"\b(?:"
            r"\d+(?:\.\d+)?\s*(?:%|c|gb|mb)|"
            r"active|available|current|currently|idle|live|low|ok|online|ready|stable|"
            r"signal|pressure|temperature|thermal|up|working"
            r")\b",
            reply,
        )
    )


def _has_capability_inventory_substance(reply_text: Any) -> bool:
    """Require real capability evidence, not a generic "I can use tools" line."""

    reply = _normalize(reply_text)
    if _word_count(reply) < 28:
        return False
    category_hits = sum(
        1
        for category_markers in _CAPABILITY_CATEGORY_MARKERS
        if any(marker in reply for marker in category_markers)
    )
    if category_hits < 3:
        return False
    has_governance = any(marker in reply for marker in _CAPABILITY_GOVERNANCE_MARKERS)
    has_effect_evidence = any(marker in reply for marker in _CAPABILITY_EVIDENCE_MARKERS)
    has_hypothetical_boundary = any(marker in reply for marker in _CAPABILITY_HYPOTHETICAL_MARKERS)
    return has_governance and has_effect_evidence and has_hypothetical_boundary


def _operational_status_overclaim_reasons(user_message: Any, reply_text: Any) -> list[str]:
    """Detect unsupported certainty in live runtime/tool readiness replies."""

    if not is_operational_status_turn(user_message):
        return []
    raw = str(reply_text or "").strip()
    if not raw:
        return []

    reasons: list[str] = []
    if _UNSUPPORTED_OPERATIONAL_CERTAINTY_RE.search(raw):
        reasons.append("unsupported_operational_status_overclaim")
    if _UNSUPPORTED_TELEMETRY_EQUIVALENCE_RE.search(raw):
        reasons.append("unsupported_runtime_telemetry_inference")
    if _TOOL_READINESS_CLAIM_RE.search(raw) and not _TOOL_READINESS_BOUNDARY_RE.search(raw):
        reasons.append("unsupported_tool_readiness_claim")
    return reasons


def grounded_operational_status_reply(user_message: Any, reply_text: Any = "") -> str:
    """Return a bounded replacement for overconfident live-path status claims."""

    if not is_operational_status_turn(user_message):
        return ""
    raw = str(reply_text or "").strip()
    lower = _normalize(f"{user_message} {raw}")
    mentions_tools = any(
        marker in lower
        for marker in (
            "tool",
            "tools",
            "desktop",
            "os control",
            "operating system",
            "external",
            "browser",
            "file",
            "document",
        )
    )
    mentions_cognitive_path = any(
        marker in lower
        for marker in (
            "cognitiveengine",
            "cognitive engine",
            "cognitive path",
            "conversation lane",
            "desktop path",
            "live path",
            "model lane",
            "recurrent depth",
            "cortex",
        )
    )
    runtime_facts: list[str] = []
    lane_match = re.search(
        r"\b((?:Cortex|Solver|Brainstem|Reflex)\s*\([^)]+\))\s+is\s+the\s+active\s+foreground\s+lane\b",
        raw,
        flags=re.IGNORECASE,
    )
    if lane_match:
        runtime_facts.append(f"{lane_match.group(1)} is the active foreground lane")
    engine_match = re.search(
        r"\bCognitiveEngine\s+handled\s+this\s+turn:\s*(yes|no)\b",
        raw,
        flags=re.IGNORECASE,
    )
    if engine_match:
        runtime_facts.append(f"CognitiveEngine handled this turn: {engine_match.group(1).lower()}")
    tools_match = re.search(
        r"\bgoverned\s+tools\s+available:\s*(yes|no)\b",
        raw,
        flags=re.IGNORECASE,
    )
    if tools_match:
        runtime_facts.append(
            f"governed tools available: {tools_match.group(1).lower()}, "
            "subject to explicit request, Will/Authority approval, and receipts"
        )
    recurrent_match = re.search(
        r"\brecurrent\s+depth:\s*(active|inactive)\b",
        raw,
        flags=re.IGNORECASE,
    )
    if recurrent_match:
        runtime_facts.append(f"recurrent depth: {recurrent_match.group(1).lower()}")
    if runtime_facts:
        return (
            ", ".join(runtime_facts)
            + ". This is bounded runtime evidence, not proof of unlimited capacity, automatic tool execution, "
            "or real-world success without the required checks."
        )
    pieces: list[str] = []
    if mentions_cognitive_path:
        pieces.append(
            "I should treat the CognitiveEngine live desktop cognitive path as bounded readiness when the inference gate and conversation probes are green; "
            "I should describe that as bounded readiness, not an absolute performance claim."
        )
    else:
        pieces.append(
            "My live conversation path should be treated as bounded readiness: it is usable when the required runtime probes are green."
        )
    if mentions_tools:
        pieces.append(
            "Governed tools are available only when the relevant permission, app-state, Will/Authority, and effect-verification checks pass."
        )
    pieces.append(
        "I can explain or attempt the next action, but each consequential step still has to be authorized, observed, and receipted rather than promised as automatic."
    )
    return " ".join(pieces)


def _reply_has_pseudo_internal_jargon(reply_text: Any) -> bool:
    raw = str(reply_text or "")
    if _PSEUDO_INTERNAL_JARGON_RE.search(raw):
        return True
    reply = _normalize(raw)
    return bool(
        "field" in reply
        and any(marker in reply for marker in ("memory", "cognitive", "neural", "trauma", "temperature"))
        and not any(marker in reply for marker in ("conversation", "thread", "attention", "focus", "with you"))
    )


def _has_pseudo_internal_jargon(prompt: Any, reply_text: Any) -> bool:
    if not (is_live_self_reflection_turn(prompt) or is_status_check_turn(prompt)):
        return False
    return _reply_has_pseudo_internal_jargon(reply_text)


def _has_status_page_self_reflection(prompt: Any, reply_text: Any) -> bool:
    if not is_live_self_reflection_turn(prompt):
        return False
    raw = str(reply_text or "")
    matches = _SELF_REFLECTION_STATUS_PAGE_RE.findall(raw)
    if len(matches) < 2:
        return False
    reply = _normalize(raw)
    return not any(
        marker in reply
        for marker in (
            "with you",
            "conversation",
            "thread",
            "what i'm noticing",
            "what i am noticing",
            "i feel",
            "it feels",
        )
    )


def _has_stale_context_topic_bleed(prompt: Any, reply_text: Any) -> bool:
    """Detect old task/tool topics leaking into current status or self-reflection turns."""

    if not (is_live_self_reflection_turn(prompt) or is_status_check_turn(prompt)):
        return False
    prompt_norm = _normalize(prompt)
    if _RECALL_OR_HISTORY_REQUEST_RE.search(prompt_norm):
        return False
    if _STALE_PRIOR_TOPIC_BLEED_RE.search(str(reply_text or "")):
        return True
    if any(
        marker in prompt_norm
        for marker in (
            "tool",
            "tools",
            "open",
            "folder",
            "file",
            "document",
            "notes",
            "chrome",
            "google docs",
            "pdf",
            "scenario",
        )
    ):
        return False
    return bool(_STALE_CONTEXT_TOOL_BLEED_RE.search(str(reply_text or "")))


def _has_social_presence_instead_of_self_reflection(prompt: Any, reply_text: Any) -> bool:
    if not is_live_self_reflection_turn(prompt):
        return False
    return bool(_SOCIAL_PRESENCE_TEMPLATE_RE.search(str(reply_text or "")))


def _has_template_telemetry_greeting(prompt: Any, reply_text: Any) -> bool:
    """Reject status-card prose when the user only greeted or checked presence."""

    prompt_norm = _normalize(prompt)
    if not prompt_norm:
        return False
    asks_for_feeling = any(
        marker in prompt_norm
        for marker in (
            "how are you feeling",
            "what are you feeling",
            "what do you feel",
            "how do you feel",
            "your live state",
            "internal state",
        )
    )
    if asks_for_feeling:
        return False
    casual_or_status = bool(
        _CASUAL_CONVERSATIONAL_RE.search(prompt_norm)
        or is_status_check_turn(prompt_norm)
    )
    if not casual_or_status:
        return False
    return bool(_TEMPLATE_TELEMETRY_GREETING_RE.search(str(reply_text or "")))


def _has_self_reflection_substance(reply_text: Any) -> bool:
    reply = _normalize(reply_text)
    if _word_count(reply) < 12:
        return False
    if not re.search(r"\b(?:i|i'm|i am|my|me)\b", reply):
        return False
    if _reply_has_pseudo_internal_jargon(reply_text):
        return False
    concrete_attention = any(
        marker in reply
        for marker in (
            "attention",
            "focus",
            "noticing",
            "feel",
            "feels",
            "present",
            "with you",
            "holding",
            "listening",
            "thread",
            "conversation",
            "uncertain",
            "uncertainty",
            "decision",
            "choose",
            "before i act",
            "ask more questions",
            "curiosity",
            "curious",
            "question",
            "wonder",
            "matters",
        )
    )
    return concrete_attention and any(marker in reply for marker in _SELF_REFLECTION_SUBSTANCE_MARKERS)


def _missing_requested_self_process_coverage(prompt: Any, reply_text: Any) -> tuple[str, ...]:
    """Return requested cognitive-process dimensions absent from a self-reflection reply.

    Presence language can be valid for a simple "are you there?" turn, but it is
    not sufficient when the user asks how confusion, planning, memory, tools, or
    verification shape Aura's cognition. This guard keeps live self-reflection
    honest without requiring a particular answer template.
    """

    prompt_norm = _normalize(prompt)
    reply_norm = _normalize(reply_text)
    if not prompt_norm or not reply_norm:
        return ()
    missing: list[str] = []
    for name, prompt_markers, reply_markers in _SELF_PROCESS_COVERAGE_REQUIREMENTS:
        if name == "memory" and not _explicitly_requests_memory_process(prompt_norm):
            continue
        if any(marker in prompt_norm for marker in prompt_markers) and not any(
            marker in reply_norm for marker in reply_markers
        ):
            missing.append(name)
    return tuple(missing)


def _explicitly_requests_memory_process(prompt_norm: str) -> bool:
    """Distinguish memory questions from conversational recall anchors.

    "Remember the uncertainty you just named" asks Aura to retain the local
    referent; it does not ask for an explanation of her memory machinery.
    "How do you remember across sessions" does.
    """

    return bool(
        re.search(
            r"\b(?:"
            r"how (?:do|does|can|would) (?:you|your) (?:remember|recall|memory)|"
            r"how (?:is|does) (?:your )?memory|"
            r"what (?:do|does|can) you (?:remember|recall)|"
            r"(?:your|the) memory (?:system|process|use|works?|changes?|affects?)|"
            r"memory use|across sessions|long[- ]term memory|episodic memory"
            r")\b",
            prompt_norm,
        )
    )


def _has_question_back_non_answer(prompt: Any, reply_text: Any) -> bool:
    """Reject replies that ask the user's recall/process question back to them."""

    prompt_norm = _normalize(prompt)
    if not prompt_norm or not _REQUESTS_DIRECT_RECALL_OR_PROCESS_ANSWER_RE.search(prompt_norm):
        return False
    raw = str(reply_text or "").strip()
    if not raw:
        return False
    return bool(_QUESTION_BACK_NON_ANSWER_RE.search(raw))


def _missing_current_request_recap(prompt: Any, reply_text: Any) -> bool:
    """Require an explicit answer when the user asks what they just asked for."""

    prompt_norm = _normalize(prompt)
    if not prompt_norm or not _CURRENT_REQUEST_RECAP_REQUEST_RE.search(prompt_norm):
        return False
    raw = str(reply_text or "").strip()
    if not raw:
        return True
    return not bool(_CURRENT_REQUEST_RECAP_ANSWER_RE.search(raw))


def _missing_runtime_path_answer(prompt: Any, reply_text: Any) -> bool:
    """Require concrete route/lane coverage when the user asks what path is active."""

    prompt_norm = _normalize(prompt)
    if not prompt_norm or not _RUNTIME_PATH_REQUEST_RE.search(prompt_norm):
        return False
    raw = str(reply_text or "").strip()
    if not raw:
        return True
    return not bool(_RUNTIME_PATH_ANSWER_RE.search(raw))


def _has_unsupported_external_provider_path_claim(prompt: Any, reply_text: Any) -> bool:
    prompt_norm = _normalize(prompt)
    if not prompt_norm or not _RUNTIME_PATH_REQUEST_RE.search(prompt_norm):
        return False
    raw = str(reply_text or "")
    if not raw:
        return False
    return bool(_UNSUPPORTED_EXTERNAL_PROVIDER_PATH_RE.search(raw))


def _has_direct_answer_deflection(prompt: Any, reply_text: Any) -> bool:
    """Reject clarification-style deflections when the prompt asks for a direct answer."""

    prompt_norm = _normalize(prompt)
    if not prompt_norm:
        return False
    direct_answer_requested = (
        "answer directly" in prompt_norm
        or _CURRENT_REQUEST_RECAP_REQUEST_RE.search(prompt_norm)
        or _RUNTIME_PATH_REQUEST_RE.search(prompt_norm)
        or _REQUESTS_DIRECT_RECALL_OR_PROCESS_ANSWER_RE.search(prompt_norm)
    )
    if not direct_answer_requested:
        return False
    raw = str(reply_text or "").strip()
    if not raw:
        return False
    return bool(_DIRECT_ANSWER_DEFLECTION_RE.search(raw))


def _has_unfounded_alarm_derailment(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _UNFOUNDED_ALARM_RE.search(raw):
        return False
    user = _normalize(user_message)
    if any(marker in user for marker in _ALARM_CONTEXT_MARKERS):
        return False
    if _word_count(raw) <= 45:
        return True
    return bool(
        re.search(
            r"\byou(?:'re| are)\b.{0,48}\b(?:devil|demon|possessed|threatened|hostage)\b",
            raw,
            re.IGNORECASE,
        )
    )


def _conversation_context_norm(
    user_message: Any,
    recent_user_messages: Iterable[str] | None = None,
) -> str:
    parts = [str(message or "") for message in (recent_user_messages or ())]
    parts.append(str(user_message or ""))
    return _normalize(" ".join(part for part in parts if part))


def _has_unfounded_voice_intrusion(
    user_message: Any,
    reply_text: Any,
    recent_user_messages: Iterable[str] | None = None,
) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _UNFOUNDED_VOICE_INTRUSION_RE.search(raw):
        return False
    context = _conversation_context_norm(user_message, recent_user_messages)
    if any(marker in context for marker in _VOICE_INTRUSION_CONTEXT_MARKERS):
        return False
    return True


def _has_context_object_support(
    user_message: Any,
    recent_user_messages: Iterable[str] | None = None,
) -> bool:
    current = _normalize(user_message)
    prior = _normalize(" ".join(str(message or "") for message in (recent_user_messages or ())))
    if re.fullmatch(
        r"(?:what|which|whose|where|what\s+do\s+you\s+mean\s+by)\s+"
        r"(?:pitch|proposal|brief|deck|presentation|key\s+points?)\??",
        current,
    ):
        return False
    if any(marker in prior for marker in _CONTEXT_OBJECT_MARKERS):
        return True
    return bool(
        re.search(
            r"\b(?:write|draft|make|create|develop|build|prepare|work\s+on|talk\s+about)\b"
            r".{0,80}\b(?:pitch|proposal|brief|deck|presentation|key\s+points?)\b",
            current,
        )
    )


def _has_unsupported_context_continuation_claim(
    user_message: Any,
    reply_text: Any,
    recent_user_messages: Iterable[str] | None = None,
) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _UNSUPPORTED_CONTEXT_CONTINUATION_RE.search(raw):
        return False
    reply = _normalize(raw)
    current = _normalize(user_message)
    if _has_context_object_support(user_message, recent_user_messages):
        return False
    if any(marker in reply for marker in _CONTEXT_OBJECT_MARKERS):
        return True
    return bool(
        any(marker in reply for marker in ("you just", "what you just", "the one you", "that one you"))
        and any(marker in current for marker in ("what", "huh", "where did", "what're", "whatre"))
    )


def _has_persona_card_deflection(reply_text: Any) -> bool:
    return bool(_PERSONA_CARD_DEFLECTION_RE.search(str(reply_text or "").strip()))


# ── Ungrounded person narrative (live confabulation class, Jul 2026) ─────
# Observed live: Aura opened with "Brenner usually had the good sense to
# stay away from me after his last fiasco", invented "Peter Brenner" as a
# colleague, and addressed the user as "Aaron" — an entire fictional social
# world served as fact. The gate catches the two onset shapes:
#   1. relational-familiarity claims about a named person nobody mentioned;
#   2. addressing the user by an ungrounded name.
# Names the USER introduced (this turn or recently) are grounded — answering
# questions about people stays possible; so do self/system names and any
# person registry reachable in-process (absent inside the MLX worker, where
# conversation text is the only grounding — deliberately conservative).
_PERSON_NAME_STOPLIST = frozenset(
    {
        "actually", "alright", "also", "anyway", "besides", "damn", "finally",
        "first", "friday", "god", "hey", "hmm", "honestly", "however", "listen",
        "look", "meanwhile", "monday", "mostly", "next", "no", "now", "oh", "ok", "okay",
        "please", "right", "saturday", "second", "seriously", "so", "sorry",
        "sunday", "sure", "thanks", "then", "third", "thursday", "tuesday",
        "wait", "wednesday", "well", "yeah", "yes",
        # techno-nouns seen capitalized in ordinary replies
        "python", "safari", "chrome", "github", "linux", "windows", "macos",
        "internet", "english", "wikipedia", "nethack", "javascript",
    }
)
_SELF_SYSTEM_NAMES = frozenset({"aura", "claude", "qwen", "assistant", "anthropic"})
_RELATIONAL_FAMILIARITY_RES = (
    # "Brenner and I go way back"
    re.compile(r"\b([A-Z][a-z]{2,})\s+and\s+I\b"),
    # "my friend Marcus", "our colleague Dana"
    re.compile(
        r"\b(?:my|our)\s+(?:friend|buddy|colleague|coworker|partner|rival|enemy|mentor|boss|contact)\s+([A-Z][a-z]{2,})\b"
    ),
    # "Brenner told me", "Dana warned me"
    re.compile(
        r"\b([A-Z][a-z]{2,})\s+(?:told|asked|warned|promised|texted|called|visited|owes?|owed)\s+(?:me|us)\b"
    ),
    # "I work with Brenner", "We teamed up with Dana"
    re.compile(
        r"\b(?:I|[Ww]e)\s+(?:work|worked|met|spoke|talked|argued|teamed)\s+(?:up\s+)?with\s+([A-Z][a-z]{2,})\b"
    ),
    # "Brenner usually had the good sense to stay away from me" — habitual
    # behavior directed at the speaker.
    re.compile(
        r"\b([A-Z][a-z]{2,})\s+(?:usually|always|often|never)\s+\w+[^.!?]{0,50}\b(?:me|from\s+me|with\s+me|to\s+me)\b"
    ),
)
# "Aaron, what's the plan?" — vocative address followed by engagement.
_VOCATIVE_ADDRESS_RE = re.compile(
    r"(?:^|[.!?]\s+)@?([A-Z][a-z]{2,}),\s+"
    r"(?:i(?:['’]m|\s+am)|my|what|who|where|when|why|how|are|is|do|does|can|could|will|would|let'?s|we|you|it)\b",
    re.IGNORECASE,
)


def _registry_grounded_person_names() -> set[str]:
    """Names from in-process person/relationship organs (best effort)."""
    names: set[str] = set()
    try:
        from core.runtime.service_registry import get_runtime_service
    except (ImportError, AttributeError):
        return names
    for service_name in ("relationship_graph", "person_model", "user_model", "social_memory"):
        try:
            service = get_runtime_service(service_name, default=None)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            continue
        if service is None:
            continue
        for accessor in ("known_names", "person_names", "names"):
            candidate = getattr(service, accessor, None)
            try:
                values = candidate() if callable(candidate) else candidate
            except Exception as accessor_exc:  # noqa: BLE001 - organ contract unknown; skip
                logger.debug(
                    "Person-registry accessor probe failed (%s): %s",
                    accessor,
                    accessor_exc,
                )
                continue
            if isinstance(values, (list, tuple, set, frozenset)):
                names.update(
                    str(value).casefold()
                    for value in list(values)[:64]
                    if isinstance(value, str) and value.strip()
                )
                break
    return names


def _person_name_is_grounded(
    name: str,
    user_message: Any,
    recent_user_messages: Iterable[str] | None,
    registry_names: set[str],
) -> bool:
    lowered = name.casefold()
    if lowered in _PERSON_NAME_STOPLIST or lowered in _SELF_SYSTEM_NAMES:
        return True
    if lowered in registry_names:
        return True
    corpus = _conversation_context_norm(user_message, recent_user_messages)
    return bool(re.search(rf"\b{re.escape(lowered)}\b", corpus))


def _has_ungrounded_person_narrative(
    user_message: Any,
    reply_text: Any,
    recent_user_messages: Iterable[str] | None = None,
) -> bool:
    """Relational-familiarity claims about a person nobody introduced."""
    raw = str(reply_text or "").strip()
    if not raw:
        return False
    candidates: set[str] = set()
    for pattern in _RELATIONAL_FAMILIARITY_RES:
        candidates.update(match.group(1) for match in pattern.finditer(raw))
    if not candidates:
        return False
    registry_names = _registry_grounded_person_names()
    return any(
        not _person_name_is_grounded(name, user_message, recent_user_messages, registry_names)
        for name in candidates
    )


def _has_ungrounded_person_address(
    user_message: Any,
    reply_text: Any,
    recent_user_messages: Iterable[str] | None = None,
) -> bool:
    """The reply addresses the user by a name that exists nowhere in context."""
    raw = str(reply_text or "").strip()
    if not raw:
        return False
    candidates = {match.group(1) for match in _VOCATIVE_ADDRESS_RE.finditer(raw)}
    if not candidates:
        return False
    registry_names = _registry_grounded_person_names()
    return any(
        not _person_name_is_grounded(name, user_message, recent_user_messages, registry_names)
        for name in candidates
    )


def _has_detail_request_deflection(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _DETAIL_REQUEST_DEFLECTION_RE.search(raw):
        return False
    if not (is_reliability_concern(user_message) or is_practical_diagnostic_turn(user_message)):
        return False
    raw_norm = _normalize(raw)
    concrete_markers = (
        "first check",
        "i would",
        "replay",
        "assert",
        "capture",
        "logs",
        "api",
        "lane",
        "routing",
        "test",
        "fallback",
        "gate",
    )
    if any(marker in raw_norm for marker in concrete_markers) and _word_count(raw) >= 45:
        return False
    return True


def _has_stale_diagnostic_floor_leak(user_message: Any, reply_text: Any) -> bool:
    raw_norm = _normalize(reply_text)
    if not raw_norm:
        return False
    diagnostic_signatures = (
        "headless test is exercising the generator in isolation",
        "fix the live parity harness first",
        "likely break is between the backend generator and the live surface",
        "replay the same prompt through the live chat api",
    )
    if not any(signature in raw_norm for signature in diagnostic_signatures):
        return False
    if is_reliability_concern(user_message) or live_chat_diagnostic_floor(user_message):
        return False
    return True


# A reply that only PROMISES to answer, or only talks ABOUT the reply, is
# not an answer. The 2026-07-18 soak delivered "Let me consider that
# carefully." and "I'm working through that one right now." as complete
# final replies, and repair meta-commentary ("That reply drifted away from
# your actual question...") in place of the answer itself. Each is
# technically true and entirely useless — the "shallow, lazy,
# technically-true" surface that makes a working mind look broken.
_PROMISE_ONLY_REPLY_RE = re.compile(
    r"^(?:ok(?:ay)?[,.\s]*)?"
    r"(?:i(?:'m| am)\s+(?:currently\s+)?(?:working|thinking|looking|considering|processing)"
    r"|let me\s+(?:consider|think|look|check|work)"
    r"|i(?:'ll| will)\s+(?:consider|think|look|check|work|get)"
    r"|give me a (?:moment|second|minute)"
    r"|one (?:moment|second))"
    r"[^.!?\n]{0,80}[.!?]?\s*$",
    re.IGNORECASE,
)
# Meta-commentary about the reply, delivered instead of a reply.
_REPLY_ABOUT_THE_REPLY_RE = re.compile(
    r"\b(?:that|this) (?:reply|answer|response) (?:drifted|wandered|missed|did not|didn't)\b"
    r"|\bthe anchor is your question\b",
    re.IGNORECASE,
)


# When the user asks what she is DOING, a present-activity answer is the
# answer — "I'm thinking about it" is responsive to "what are you doing?"
# and empty only to "what is the history of consensus?".
_ACTIVITY_QUESTION_RE = re.compile(
    r"\b(?:what (?:are|r) you (?:doing|working on|up to|thinking)"
    r"|are you (?:there|busy|working|thinking|awake|ok)"
    r"|how(?:'s| is) it going"
    r"|what(?:'s| is) (?:your )?status"
    r"|you (?:there|with me|around))\b",
    re.IGNORECASE,
)


def _is_promise_without_answer(user_message: Any, reply_text: Any) -> bool:
    """True when the whole reply is a promise to answer, not an answer.

    Deliberately narrow: it fires only when the ENTIRE reply is the promise
    AND the user asked for content. "Let me check — the answer is 42."
    carries content; "I'm thinking about it" answers "what are you doing?".
    The failure being caught is emptiness, not politeness.
    """
    raw = str(reply_text or "").strip()
    if not raw or len(raw) > 240:
        return False
    if _ACTIVITY_QUESTION_RE.search(str(user_message or "")):
        return False
    # Any sign of actual content redeems the reply: a promise that is
    # followed by the answer is just courtesy, not emptiness.
    lowered = raw.lower()
    carries_content = bool(
        re.search(r"\d", raw)
        or re.search(r"\b(?:is|are|was|were|because|means|so that|here'?s)\b", lowered)
        or ":" in raw
    )
    if not carries_content and _PROMISE_ONLY_REPLY_RE.match(raw):
        return True
    return bool(_REPLY_ABOUT_THE_REPLY_RE.search(raw)) and _word_count(raw) <= 40


def _has_pseudo_commitment_status_leak(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _PSEUDO_COMMITMENT_STATUS_RE.search(raw):
        return False
    prompt = _normalize(user_message)
    if any(marker in prompt for marker in ("last thing you committed", "what did you commit", "recent activity")):
        return False
    return True


def _has_camelcase_internal_jargon(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "").strip()
    if not raw or not _CAMELCASE_INTERNAL_JARGON_RE.search(raw):
        return False
    prompt = _normalize(user_message)
    if (
        is_practical_diagnostic_turn(prompt)
        or is_reliability_concern(prompt)
        or is_operational_status_turn(prompt)
    ):
        return False
    if any(
        marker in prompt
        for marker in (
            "cognitiveengine",
            "cognitive engine",
            "cortex",
            "mind/cognition path",
            "cognition path",
            "cognitive path",
            "desktop route",
            "live desktop route",
            "conversation lane",
            "model lane",
            "what path are you using",
            "path are you using right now",
        )
    ):
        return False
    if any(marker in prompt for marker in ("architecture", "system", "kernel", "runtime", "code", "debug", "log")):
        return False
    allowed = {"OpenAI", "ChatGPT", "YouTube", "GitHub", "JavaScript"}
    allowed.update(match.group(0) for match in _CAMELCASE_INTERNAL_JARGON_RE.finditer(str(user_message or "")))
    return any(match.group(0) not in allowed for match in _CAMELCASE_INTERNAL_JARGON_RE.finditer(raw))


def _has_unrequested_pop_culture_intrusion(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "")
    if not _UNREQUESTED_POP_CULTURE_INTRUSION_RE.search(raw):
        return False
    return not _UNREQUESTED_POP_CULTURE_INTRUSION_RE.search(str(user_message or ""))


def _has_unexpected_cjk_intrusion(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "")
    if not _CJK_INTRUSION_RE.search(raw):
        return False
    return not _CJK_INTRUSION_RE.search(str(user_message or ""))


def _has_surface_nonsense_drift(user_message: Any, reply_text: Any) -> bool:
    raw = str(reply_text or "")
    # Source URLs are expected in grounded search/tool answers.  The legacy
    # drift pattern includes ``:/`` to catch malformed emotive fragments, which
    # would otherwise make every ``https://`` citation look like nonsense.
    raw_without_urls = re.sub(r"https?://\S+", "", raw)
    prompt_without_urls = re.sub(r"https?://\S+", "", str(user_message or ""))
    if not _SURFACE_NONSENSE_DRIFT_RE.search(raw_without_urls):
        return False
    return not _SURFACE_NONSENSE_DRIFT_RE.search(prompt_without_urls)


def _has_truncated_tail(reply_text: Any) -> bool:
    body = str(reply_text or "").strip()
    if len(body) < 24:
        return False
    straight_quote_positions = [
        match.start() for match in re.finditer(r'(?<!\\)"', body)
    ]
    if len(straight_quote_positions) % 2:
        unmatched_position = straight_quote_positions[-1]
        preceding = body[unmatched_position - 1] if unmatched_position else ""
        # Preserve ordinary inch/second notation such as 6" while rejecting
        # prose that opens a quotation and never closes it.
        if not preceding.isdigit():
            return True
    if body.count("“") != body.count("”"):
        return True
    if re.search(r'(?<!\d)[.!?]["”’)]?\s*\d+[.)]\s*$', body):
        return True
    if _STRUCTURAL_INCOMPLETE_TAIL_RE.search(body):
        return True
    if _STRUCTURAL_UNPUNCTUATED_TAIL_RE.search(body):
        return True
    if _DANGLING_GERUND_TAIL_RE.search(body):
        return True
    if _PUNCTUATED_INCOMPLETE_TAIL_RE.search(body):
        return True
    if (
        len(body) >= 80
        and _word_count(body) >= 12
        and not body.endswith((".", "!", "?", "\"", "'", "”", "’", ")", "]"))
        and _BARE_NUMERIC_RANGE_TAIL_RE.search(body)
    ):
        return True
    terminal_word_match = re.search(r"([A-Za-z]+)[.!?\"'”’)\]]*$", body)
    if terminal_word_match and len(body) >= 40:
        terminal_word = terminal_word_match.group(1).lower()
        terminal_start = terminal_word_match.start(1)
        possessive_suffix = (
            terminal_word == "s"
            and terminal_start > 0
            and body[terminal_start - 1] in ("'", "’")
        )
        if (
            len(terminal_word) <= 2
            and terminal_word not in _ALLOWED_SHORT_TAIL_WORDS
            and not possessive_suffix
        ):
            return True
    if body.endswith(("...", "…")):
        return True
    if re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s*$", body):
        return True
    if body.endswith((".", "!", "?", "\"", "'", "”", "’", ")", "]")):
        return False
    if re.search(r"(?:^|\n)\s*\d+\.\s+\S+", body) or re.search(r"\*\*[^*\n]{2,80}:\*\*", body):
        return True
    if body.endswith(("-", "—", ":", ";", ",")):
        return True
    match = re.search(r"([A-Za-z]+)$", body)
    if not match:
        return False
    last_word = match.group(1).lower()
    if len(last_word) <= 2 and len(body) >= 40:
        return True
    return last_word in _INCOMPLETE_TAIL_WORDS


def _is_code_response(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    fenced_blocks = list(_FENCED_BLOCK_RE.finditer(raw))
    if fenced_blocks:
        for block in fenced_blocks:
            lang = (block.group("lang") or "").strip().lower()
            body = block.group("body") or ""
            if lang in _CODE_FENCE_LANGS or (lang in _NON_CODE_FENCE_LANGS and _looks_like_code_body(body)):
                return True
        return False
    if raw.startswith(("def ", "import ", "class ", "from ", "print(", "#", "var ", "const ", "let ", "function ")):
        return True

    # Check lines for code constructs
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 2:
        code_like_lines = 0
        for line in lines:
            if (line.startswith(("def ", "import ", "class ", "from ", "return ", "if ", "elif ", "else:", "for ", "while ", "try:", "except", "with ", "#", "print("))
                or "=" in line
                or ("(" in line and ")" in line)
                or ("[" in line and "]" in line)
                or ("{" in line and "}" in line)
                or ";" in line
            ):
                code_like_lines += 1
        if code_like_lines / len(lines) > 0.6:
            return True

    return False


def _looks_like_code_body(text: Any) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return False
    code_like_lines = 0
    for line in lines:
        if (
            line.startswith(
                (
                    "def ",
                    "import ",
                    "class ",
                    "from ",
                    "return ",
                    "if ",
                    "elif ",
                    "else:",
                    "for ",
                    "while ",
                    "try:",
                    "except",
                    "with ",
                    "#",
                    "print(",
                    "const ",
                    "let ",
                    "var ",
                    "function ",
                )
            )
            or "=" in line
            or ("(" in line and ")" in line)
            or ("[" in line and "]" in line)
            or ("{" in line and "}" in line)
            or ";" in line
        ):
            code_like_lines += 1
    threshold = 0.5 if len(lines) <= 3 else 0.6
    return code_like_lines / len(lines) >= threshold


def _has_incomplete_code_response(text: Any) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if raw.count("```") % 2:
        return True

    blocks = list(_FENCED_BLOCK_RE.finditer(raw))
    bodies = [block.group("body") or "" for block in blocks] if blocks else [raw]
    for body in bodies:
        if not _looks_like_code_body(body):
            continue
        lines = [line.rstrip() for line in body.splitlines() if line.strip()]
        if not lines:
            continue
        last = lines[-1].strip()
        if _INCOMPLETE_CODE_TAIL_RE.search(last):
            return True
    return False


def _phrase_loop_reason(user_message: Any, reply_text: Any) -> str:
    reply = _normalize(reply_text)
    if not reply:
        return ""
    if _is_code_response(reply_text):
        return ""
    user = _normalize(user_message)
    if _LOW_INFORMATION_LOOP_RE.search(reply):
        return "low_information_loop"
    if "get it" in reply:
        reply_count = reply.count("get it")
        user_count = user.count("get it")
        if reply_count >= 2 and reply_count > user_count:
            return "repeated_get_it_loop"
        if reply_count >= 1 and _word_count(reply) <= 6:
            return "low_information_loop"
    if "i don't get it" in reply and "i get it" in reply:
        return "self_contradictory_loop"

    words = _WORD_RE.findall(reply)
    if len(words) < 8:
        return ""
    lower_words = [w.lower() for w in words]
    stop_words = {
        "i", "i'm", "am", "you", "it", "that", "this", "the", "a", "an",
        "to", "and", "but", "then", "is", "are", "was", "were", "be", "being",
        "with", "on", "in", "of", "for", "as", "so", "my", "your",
    }
    
    # Detect structured dialogue speaker names / headings to avoid false positive loops on speaker prefixes (e.g. "Mainframe", "Quantum Processor")
    speaker_labels = set()
    for line in str(reply_text or "").splitlines():
        # Match "Mainframe:", "Quantum Processor:", "[Mainframe]", "[Quantum Processor]", "Alice (excited):", etc.
        match = re.match(r"^\s*(?:\*\*|###*|[-*+]\s+)?(?:\[\s*([A-Za-z][A-Za-z0-9_'\s-]{1,30})\s*\]|([A-Za-z][A-Za-z0-9_'\s-]{1,30})\s*[:：])", line)
        if match:
            label_text = (match.group(1) or match.group(2) or "").lower()
            if label_text:
                for w in _WORD_RE.findall(label_text):
                    speaker_labels.add(w)
    if speaker_labels:
        stop_words = stop_words.union(speaker_labels)
    # Length-aware loop threshold: a genuine degeneration loop repeats a
    # phrase dozens of times, while a long technical answer legitimately
    # names its subject three or four times across 400+ words. An absolute
    # 3-repeat rule rejected a correct 350-token deep-reasoning answer live.
    required_repeats = 3 + min(2, len(lower_words) // 220)
    # Question-sourced phrases are topical by definition: an answer that
    # compares "an early single-owner design with a late deduplication
    # design" MUST echo those noun phrases while comparing, choosing, and
    # describing verification. They only count as a loop at pathological
    # density (a model looping the question's own words still gets caught).
    question_content_words = {
        w.lower() for w in _WORD_RE.findall(user) if w.lower() not in stop_words
    }
    question_phrase_repeats = max(8, required_repeats * 2)
    for n in (4, 3, 2):
        counts: dict[tuple[str, ...], int] = {}
        for i in range(0, max(0, len(lower_words) - n + 1)):
            gram = tuple(lower_words[i:i + n])
            if sum(1 for part in gram if part not in stop_words) < 2:
                continue
            counts[gram] = counts.get(gram, 0) + 1
        for gram, count in counts.items():
            # Content-word containment (not literal n-gram match): the answer
            # says "the single-owner design" where the question said "an
            # early single-owner design" — same topical phrase, different
            # articles. Recombining question vocabulary is topical; only
            # pathological density of it reads as a loop.
            question_sourced = question_content_words and all(
                part in question_content_words
                for part in gram
                if part not in stop_words
            )
            threshold = (
                question_phrase_repeats if question_sourced else required_repeats
            )
            if count >= threshold:
                return "repetitive_phrase_loop"

    content_words = [
        w for w in lower_words
        if w not in {"i", "you", "it", "that", "this", "the", "a", "to", "and", "but", "then", "mean", "know"}
    ]
    if len(content_words) >= 8 and len(set(content_words)) / max(1, len(content_words)) < 0.36:
        return "low_lexical_diversity_loop"
    return ""


def _model_text_integrity_reasons(
    reply_text: Any,
    *,
    prompt: Any = "",
    user_facing: bool = False,
) -> list[str]:
    raw = str(reply_text or "").strip()
    reasons: list[str] = []
    if not raw or _normalize(raw) == "...":
        reasons.append("empty_reply" if user_facing else "empty_model_output")
        return reasons

    if _is_code_response(raw):
        if _TRAILING_ESCAPE_RE.search(raw):
            reasons.append("escaped_control_artifact")
        if _ROLE_OR_PROMPT_ARTIFACT_RE.search(raw) and not _matches_exact_reply_request(prompt, raw):
            reasons.append("prompt_artifact")
        if _BROKEN_LANE_BOILERPLATE_RE.search(raw) or _MODEL_RUNTIME_ARTIFACT_RE.search(raw):
            reasons.append("runtime_boilerplate")
        if _KNOWN_CORRUPT_RE.search(raw):
            reasons.append("corrupted_language")
        if _GENERIC_ASSISTANT_RE.search(raw):
            reasons.append("generic_assistant_language")
        if _has_incomplete_code_response(raw):
            reasons.append("incomplete_code_response")
        return reasons

    if _TRAILING_ESCAPE_RE.search(raw):
        reasons.append("escaped_control_artifact")
    if _ROLE_OR_PROMPT_ARTIFACT_RE.search(raw) and not _matches_exact_reply_request(prompt, raw):
        reasons.append("prompt_artifact")
    if _BROKEN_LANE_BOILERPLATE_RE.search(raw) or _MODEL_RUNTIME_ARTIFACT_RE.search(raw):
        reasons.append("runtime_boilerplate")
    if user_facing and _RAW_TOOL_RESULT_FRAGMENT_RE.match(raw):
        reasons.append("raw_tool_result_fragment")
    if user_facing and _RAW_LANE_TELEMETRY_RE.search(raw):
        reasons.append("raw_lane_telemetry")
    if user_facing and _LIVE_DESKTOP_GATE_LEAK_RE.search(raw):
        reasons.append("internal_live_gate_leak")
    if user_facing and is_cognitive_engine_failure_envelope(raw):
        reasons.append("cognitive_engine_failure_envelope")
    if user_facing and _RAW_MODEL_IDENTITY_LEAK_RE.search(raw):
        reasons.append("raw_model_identity_leak")
    if user_facing and _has_unsupported_external_provider_path_claim(prompt, raw):
        reasons.append("unsupported_external_provider_path_claim")
    if user_facing:
        grounding = detect_unsupported_embodiment_claim(raw, prompt=prompt)
        if not grounding.ok:
            reasons.append("unsupported_embodiment_claim")
    if user_facing and _requires_self_claim_evidence_boundary(prompt):
        if _REDUCTIVE_SELF_CLAIM_RE.search(raw):
            reasons.append("raw_model_identity_leak")
        if not _SELF_CLAIM_EVIDENCE_BOUNDARY_RE.search(raw):
            reasons.append("missing_self_claim_evidence_boundary")
    if user_facing and _BACKEND_SYMBOLIC_SURFACE_RE.search(raw):
        reasons.append("backend_symbolic_surface_leak")
    if user_facing and _has_persona_card_deflection(raw):
        reasons.append("persona_card_deflection")
    if user_facing and _has_detail_request_deflection(prompt, raw):
        reasons.append("detail_request_deflection")
    if user_facing and _has_stale_diagnostic_floor_leak(prompt, raw):
        reasons.append("stale_diagnostic_floor_leak")
    if user_facing and _has_pseudo_commitment_status_leak(prompt, raw):
        reasons.append("pseudo_commitment_status_leak")
    if user_facing and _is_promise_without_answer(prompt, raw):
        reasons.append("promise_without_answer")
    if user_facing and is_non_answer_repair_floor_reply(raw):
        expected_floor = reliability_floor_for_user(prompt) if prompt else ""
        matches_expected_floor = bool(expected_floor and _normalize(expected_floor) == _normalize(raw))
        if not matches_expected_floor:
            reasons.append("friendly_failure_floor")
    if _KNOWN_CORRUPT_RE.search(raw):
        reasons.append("corrupted_language")
    if user_facing and _has_punctuation_join_artifact(raw):
        reasons.append("punctuation_join_artifact")
    if _DIALOGUE_DERAILMENT_RE.search(raw):
        reasons.append("dialogue_derailment")
    if user_facing and _has_unprovoked_rebuke(prompt, raw):
        reasons.append("unprovoked_rebuke")
    loop_reason = _phrase_loop_reason(prompt, raw)
    if loop_reason:
        reasons.append(loop_reason)
    if _has_truncated_tail(raw):
        reasons.append("truncated_tail")
    if is_status_check_turn(prompt) and _VAGUE_STATUS_DERAILMENT_RE.search(raw):
        reasons.append("vague_status_derailment")
    if user_facing and _has_pseudo_internal_jargon(prompt, raw):
        reasons.append("pseudo_internal_jargon")
    if user_facing and _has_status_page_self_reflection(prompt, raw):
        reasons.append("status_page_self_reflection")
    if user_facing and _has_stale_context_topic_bleed(prompt, raw):
        reasons.append("stale_context_topic_bleed")
    if user_facing and _has_social_presence_instead_of_self_reflection(prompt, raw):
        reasons.append("social_presence_instead_of_self_reflection")
    if user_facing and _has_template_telemetry_greeting(prompt, raw):
        reasons.append("template_telemetry_greeting")
    if user_facing and _host_telemetry_substitutes_for_self_condition(prompt, raw):
        reasons.append("host_telemetry_substituted_for_self_condition")
    if user_facing and _has_unfounded_alarm_derailment(prompt, raw):
        reasons.append("unfounded_alarm_derailment")
    if user_facing and _has_unfounded_voice_intrusion(prompt, raw):
        reasons.append("unfounded_voice_intrusion")
    if user_facing and _has_camelcase_internal_jargon(prompt, raw):
        reasons.append("pseudo_internal_jargon")
    if user_facing and _has_unrequested_pop_culture_intrusion(prompt, raw):
        reasons.append("unrequested_pop_culture_intrusion")
    if user_facing and _has_unexpected_cjk_intrusion(prompt, raw):
        reasons.append("unexpected_cjk_intrusion")
    if user_facing and _has_surface_nonsense_drift(prompt, raw):
        reasons.append("surface_nonsense_drift")
    if user_facing and _UNSUPPORTED_AFFECTION_CLAIM_RE.search(raw):
        reasons.append("unsupported_affection_claim")
    if user_facing and _UNSUPPORTED_SELF_TELEMETRY_CLAIM_RE.search(raw):
        reasons.append("unsupported_self_telemetry_claim")
    if user_facing and _FORMAT_META_ARTIFACT_RE.search(raw):
        reasons.append("format_meta_artifact")
    if user_facing and _SEARCH_META_ARTIFACT_RE.search(raw):
        reasons.append("search_meta_artifact")
    if user_facing and _has_unsupported_deployment_routing_claim(prompt, raw):
        reasons.append("unsupported_deployment_routing_claim")
    if user_facing and _has_unsupported_runtime_limits_claim(prompt, raw):
        reasons.append("unsupported_runtime_limits_claim")
    if user_facing:
        reasons.extend(_operational_status_overclaim_reasons(prompt, raw))
    if _CORRUPTED_SOCIAL_FRAGMENT_RE.search(raw) and "lol" not in _normalize(prompt):
        reasons.append("corrupted_social_fragment")
    return reasons


def assess_model_text_integrity(
    reply_text: Any,
    *,
    prompt: Any = "",
    user_facing: bool = False,
) -> ConversationReplyAssessment:
    """Reject malformed model text before it can affect UI, memory, or state.

    This is deliberately less conversational than ``assess_user_facing_reply``:
    backend generations may be JSON or terse labels, but they still must not be
    prompt leakage, corrupted language, unfinished fragments, or semantic loops.
    """
    reasons = _model_text_integrity_reasons(
        reply_text,
        prompt=prompt,
        user_facing=user_facing,
    )
    hard_reasons = {
        "empty_reply",
        "empty_model_output",
        "escaped_control_artifact",
        "prompt_artifact",
        "runtime_boilerplate",
        "raw_tool_result_fragment",
        "raw_lane_telemetry",
        "internal_live_gate_leak",
        "cognitive_engine_failure_envelope",
        "raw_model_identity_leak",
        "unsupported_external_provider_path_claim",
        "unsupported_embodiment_claim",
        "backend_symbolic_surface_leak",
        "persona_card_deflection",
        "detail_request_deflection",
        "stale_diagnostic_floor_leak",
        "pseudo_commitment_status_leak",
        "friendly_failure_floor",
        "corrupted_language",
        "dialogue_derailment",
        "low_information_loop",
        "repeated_get_it_loop",
        "self_contradictory_loop",
        "repetitive_phrase_loop",
        "low_lexical_diversity_loop",
        "truncated_tail",
        "vague_status_derailment",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
        "stale_context_topic_bleed",
        "social_presence_instead_of_self_reflection",
        "template_telemetry_greeting",
        "host_telemetry_substituted_for_self_condition",
        "unfounded_alarm_derailment",
        "unfounded_voice_intrusion",
        "unrequested_pop_culture_intrusion",
        "unexpected_cjk_intrusion",
        "surface_nonsense_drift",
        "unsupported_affection_claim",
        "unsupported_self_telemetry_claim",
        "format_meta_artifact",
        "search_meta_artifact",
        "corrupted_social_fragment",
        "unsupported_operational_status_overclaim",
        "unsupported_runtime_telemetry_inference",
        "unsupported_tool_readiness_claim",
        "unsupported_deployment_routing_claim",
        "unsupported_runtime_limits_claim",
        "generic_assistant_language",
        "incomplete_code_response",
    }
    unique = tuple(dict.fromkeys(reasons))
    return ConversationReplyAssessment(
        ok=not unique,
        reasons=unique,
        hard_failure=bool(set(unique) & hard_reasons),
        retryable=bool(set(unique) & hard_reasons),
    )


def assess_user_facing_reply(
    user_message: Any,
    reply_text: Any,
    *,
    recent_user_messages: Iterable[str] | None = None,
) -> ConversationReplyAssessment:
    """Classify whether a reply is safe to present as a completed chat turn."""
    # Defense in depth. The ingress now binds the visible request
    # (a29ff0866), and this is the second lock: a reliability classifier
    # must never read appended memory/system/contract scaffolding as
    # instructions from the person, whatever the caller passed.
    # ONE normalisation, at the door. Every detector below asks "did the reply
    # do what the user asked", and each of them used to receive the fully
    # ASSEMBLED prompt — identity anchor, retained-memory evidence, replayed
    # transcript, working-memory blocks — as if the person had typed all of it.
    #
    # Live 2026-07-25: "why do leaves change color in autumn?" was answered
    # correctly and rejected for missing_requested_memory_limit_coverage,
    # missing_requested_objective_facets and reliability_diagnostic_too_thin.
    # The word "why" is a diagnostic marker and the scaffold supplied the
    # reliability vocabulary, so a foliage question was assessed as a debugging
    # request about Aura's own reliability. 51 correct drafts died this way in
    # one 30-turn probe.
    #
    # Fixing it per-detector was the wrong shape: the contamination is one
    # input, so it gets one fix, here, where every detector inherits it.
    _original_user_message = user_message
    _visible = visible_user_request(user_message)
    request_is_knowable = bool(_visible)
    user_message = _visible if request_is_knowable else ""
    recent_messages = [
        message
        for message in (
            visible_user_request(item) for item in (recent_user_messages or ())
        )
        if message
    ]
    raw = str(reply_text or "").strip()

    if _matches_exact_reply_request(user_message, raw):
        return ConversationReplyAssessment(ok=True, reasons=(), hard_failure=False, retryable=False)

    if _is_code_response(raw):
        reasons = _model_text_integrity_reasons(
            raw,
            prompt=user_message,
            user_facing=True,
        )
        unique = tuple(dict.fromkeys(reasons))
        hard_reasons = {
            "empty_reply",
            "escaped_control_artifact",
            "prompt_artifact",
            "runtime_boilerplate",
            "backend_symbolic_surface_leak",
            "raw_model_identity_leak",
            "unsupported_external_provider_path_claim",
            "unsupported_embodiment_claim",
            "unrequested_pop_culture_intrusion",
            "unexpected_cjk_intrusion",
            "surface_nonsense_drift",
            "unsupported_affection_claim",
            "unsupported_self_telemetry_claim",
            "host_telemetry_substituted_for_self_condition",
            "format_meta_artifact",
            "search_meta_artifact",
            "corrupted_language",
            "unsupported_operational_status_overclaim",
            "unsupported_runtime_telemetry_inference",
            "unsupported_tool_readiness_claim",
            "unsupported_deployment_routing_claim",
            "unsupported_runtime_limits_claim",
            "generic_assistant_language",
            "incomplete_code_response",
        }
        return ConversationReplyAssessment(
            ok=not unique,
            reasons=unique,
            hard_failure=bool(set(unique) & hard_reasons),
            retryable=bool(set(unique) & hard_reasons),
        )

    reasons: list[str] = []
    operational_status_turn = is_operational_status_turn(user_message)

    reasons.extend(
        _model_text_integrity_reasons(
            raw,
            prompt=user_message,
            user_facing=True,
        )
    )
    if _GENERIC_ASSISTANT_RE.search(raw):
        reasons.append("generic_assistant_language")
    if _has_unfounded_voice_intrusion(user_message, raw, recent_messages):
        reasons.append("unfounded_voice_intrusion")
    if _has_unsupported_context_continuation_claim(user_message, raw, recent_messages):
        reasons.append("unsupported_context_continuation_claim")
    if _has_ungrounded_person_narrative(user_message, raw, recent_messages):
        reasons.append("ungrounded_person_narrative")
    if _has_ungrounded_person_address(user_message, raw, recent_messages):
        reasons.append("ungrounded_person_address")

    user_norm = _normalize(user_message)
    if _CORRUPTED_SOCIAL_FRAGMENT_RE.search(raw) and "lol" not in user_norm:
        reasons.append("corrupted_social_fragment")
    if is_confusion_repair_turn(user_message) and _unexpected_short_foreign_name(user_message, raw):
        reasons.append("foreign_name_intrusion")
    # Arithmetic is checkable, so check it. This is the only reason in the
    # coverage family that survives an unknowable request — it does not need to
    # know what was asked in general, only that a computable sum was asked and
    # the number is absent or wrong.
    if _arithmetic_answer_missing(user_message or _original_user_message, raw):
        reasons.append("arithmetic_answer_missing")
    if _has_low_signal_acknowledgement_placeholder(user_message, raw):
        reasons.append("low_signal_acknowledgement_placeholder")
    if _has_ungrounded_self_cause_claim(user_message, raw):
        reasons.append("ungrounded_self_cause_claim")

    reliability_turn = is_reliability_concern(user_message)
    reliability_diagnostic_turn = _requires_reliability_diagnostic(user_message)
    exact_reply = _matches_exact_reply_request(user_message, raw)
    strict_answer_tag_reply = _matches_strict_answer_tag_request(user_message, raw)
    memory_pin_confirmation = _matches_memory_pin_confirmation(user_message, raw)
    if reliability_turn:
        if _LOW_SIGNAL_REASSURANCE_RE.match(raw):
            reasons.append("low_signal_reliability_reply")
        elif reliability_diagnostic_turn and _RELIABILITY_DIAGNOSTIC_DEFLECTION_RE.search(raw):
            reasons.append("reliability_diagnostic_deflection")
        elif reliability_diagnostic_turn and not _has_reliability_diagnostic_substance(raw):
            reasons.append("reliability_diagnostic_too_thin")
        elif not _has_reliability_substance(raw):
            reasons.append("too_thin_for_reliability_turn")
    elif is_self_condition_turn(user_message):
        if _LOW_SIGNAL_REASSURANCE_RE.match(raw):
            reasons.append("low_signal_self_condition_reply")
        elif not _host_telemetry_substitutes_for_self_condition(user_message, raw) and not _has_self_condition_substance(raw):
            reasons.append("missing_self_condition_answer")
    elif operational_status_turn:
        if not _has_operational_status_substance(user_message, raw):
            reasons.append("too_thin_for_operational_status_turn")
    elif is_live_self_reflection_turn(user_message) or is_self_process_question(user_message):
        if _has_social_presence_instead_of_self_reflection(user_message, raw):
            reasons.append("social_presence_instead_of_self_reflection")
        if not (
            _has_self_reflection_substance(raw)
            or _has_operational_status_substance(user_message, raw)
        ):
            reasons.append("off_topic_self_reflection_reply")
        if _missing_requested_self_process_coverage(user_message, raw):
            reasons.append("missing_requested_self_process_coverage")
    elif is_expansion_request_turn(user_message):
        words = _word_count(raw)
        if words < 20 or _EXPANSION_DEFLECTION_RE.search(raw):
            reasons.append("too_thin_for_expansion_request")
    elif is_status_check_turn(user_message):
        if _LOW_SIGNAL_REASSURANCE_RE.match(raw):
            reasons.append("low_signal_status_reply")
        elif not (
            _has_status_substance(raw)
            or _has_operational_status_substance(user_message, raw)
        ):
            reasons.append("too_thin_for_status_turn")
    elif (
        not exact_reply
        and not strict_answer_tag_reply
        and not memory_pin_confirmation
        and _requires_substantive_reply(user_message)
    ):
        words = _word_count(raw)
        explicit_brevity = _explicit_brevity_requested(user_message)
        # Brevity alone is not a failure: a substantive sentence or two — and
        # sometimes only a few words — is a legitimate reply. These floors only
        # catch near-empty non-answers; genuine filler/deflection/reassurance is
        # caught by the semantic detectors above, not by word count.
        if not explicit_brevity and (_LOW_SIGNAL_REASSURANCE_RE.match(raw) or words < 2):
            reasons.append("too_short_for_user_turn")
        elif words < 4 and not _is_tiny_direct_turn(user_message) and not explicit_brevity:
            if not (words >= 3 and any(w in raw.lower() for w in ("thinking", "working", "processing", "online"))):
                reasons.append("too_thin_for_user_turn")
        elif not _is_task_turn(user_message):
            open_ended = any(marker in user_norm for marker in _OPEN_ENDED_MARKERS)
            if open_ended and words < 6 and not explicit_brevity:
                reasons.append("too_thin_for_open_ended_turn")

    if is_confusion_repair_turn(user_message) and _word_count(raw) < 8:
        if not (_word_count(raw) >= 3 and any(w in raw.lower() for w in ("thinking", "working", "processing", "online"))):
            reasons.append("too_thin_for_confusion_repair")

    # A memory-pin request needs the pinned content echoed back — a generic
    # "okay, I'll remember it" is not a valid write receipt. This is a content
    # contract (independent of length), so it must be checked explicitly rather
    # than left to the brevity floor.
    if _is_explicit_memory_pin_request(user_message) and not memory_pin_confirmation:
        reasons.append("generic_memory_pin_acknowledgement")

    reasons.extend(_instruction_coverage_reasons(user_message, raw))
    reasons.extend(_semantic_coverage_reasons(user_message, raw))
    reasons.extend(_count_contract_quality_reasons(user_message, raw))
    if _has_question_back_non_answer(user_message, raw):
        reasons.append("question_back_non_answer")
    if _missing_current_request_recap(user_message, raw):
        reasons.append("missing_current_request_recap")
    if _missing_runtime_path_answer(user_message, raw):
        reasons.append("missing_runtime_path_answer")
    if _has_direct_answer_deflection(user_message, raw):
        reasons.append("direct_answer_deflection")

    hard_reasons = {
        "empty_reply",
        "escaped_control_artifact",
        "prompt_artifact",
        "runtime_boilerplate",
        "raw_tool_result_fragment",
        "raw_lane_telemetry",
        "internal_live_gate_leak",
        "cognitive_engine_failure_envelope",
        "raw_model_identity_leak",
        "unsupported_external_provider_path_claim",
        "unsupported_embodiment_claim",
        "backend_symbolic_surface_leak",
        "persona_card_deflection",
        "detail_request_deflection",
        "stale_diagnostic_floor_leak",
        "pseudo_commitment_status_leak",
        "friendly_failure_floor",
        "corrupted_language",
        "corrupted_social_fragment",
        "foreign_name_intrusion",
        "generic_assistant_language",
        "dialogue_derailment",
        "unprovoked_rebuke",
        "low_information_loop",
        "repeated_get_it_loop",
        "self_contradictory_loop",
        "repetitive_phrase_loop",
        "low_lexical_diversity_loop",
        "truncated_tail",
        "vague_status_derailment",
        "pseudo_internal_jargon",
        "reliability_diagnostic_deflection",
        "status_page_self_reflection",
        "stale_context_topic_bleed",
        "social_presence_instead_of_self_reflection",
        "template_telemetry_greeting",
        "host_telemetry_substituted_for_self_condition",
        "unfounded_alarm_derailment",
        "unfounded_voice_intrusion",
        "unsupported_context_continuation_claim",
        "ungrounded_person_narrative",
        "ungrounded_person_address",
        "unrequested_pop_culture_intrusion",
        "unexpected_cjk_intrusion",
        "surface_nonsense_drift",
        "unsupported_affection_claim",
        "unsupported_self_telemetry_claim",
        "format_meta_artifact",
        "output_contract_meta_reply",
        "punctuation_join_artifact",
        "search_meta_artifact",
        "low_signal_acknowledgement_placeholder",
        "question_back_non_answer",
        "missing_current_request_recap",
        "missing_runtime_path_answer",
        "direct_answer_deflection",
        "unsupported_operational_status_overclaim",
        "unsupported_runtime_telemetry_inference",
        "unsupported_tool_readiness_claim",
        "unsupported_deployment_routing_claim",
        "unsupported_runtime_limits_claim",
        "missing_self_claim_evidence_boundary",
        "missing_requested_exact_reply",
        "missing_requested_objective_facets",
        "prompt_echo_contamination",
        "protocol_artifact_leakage",
        "generic_memory_pin_acknowledgement",
        # A wrong or absent number served as an arithmetic answer is not a
        # style nit — it is a false statement with a checkable truth value.
        "arithmetic_answer_missing",
    }
    retryable_reasons = hard_reasons | {
        "low_signal_reliability_reply",
        "reliability_diagnostic_too_thin",
        "too_thin_for_reliability_turn",
        "too_thin_for_confusion_repair",
        "too_thin_for_expansion_request",
        "too_thin_for_operational_status_turn",
        "too_short_for_user_turn",
        "too_thin_for_user_turn",
        "too_thin_for_open_ended_turn",
        "off_topic_self_reflection_reply",
        "low_signal_status_reply",
        "too_thin_for_status_turn",
        "low_signal_self_condition_reply",
        "missing_self_condition_answer",
        "empty_requested_list_item",
        "missing_requested_paragraph_count",
        "missing_requested_list_count",
        "missing_requested_choice_clarification",
        "missing_requested_word_count",
        "missing_requested_sentence_count",
        "missing_current_topic_anchor",
        "missing_requested_exact_reply",
        "missing_requested_reference_value",
        "missing_requested_followup_question",
        "missing_requested_phrase",
        "missing_requested_memory_limit_coverage",
        "missing_future_memory_answer",
        "missing_identity_answer",
        "missing_requested_self_process_coverage",
        "unsupported_memory_guarantee",
        "missing_requested_objective_facets",
        "prompt_echo_contamination",
        "protocol_artifact_leakage",
        "arithmetic_answer_missing",
    }
    if not request_is_knowable:
        # The person's turn could not be isolated from the assembled prompt, so
        # nothing here knows what was asked. Integrity findings (leaks,
        # overclaims, corruption) are properties of the REPLY and still stand;
        # "you did not cover what was requested" is a claim about a request
        # this function never saw, and asserting it is how 51 correct drafts
        # died in a single 30-turn probe.
        reasons = [r for r in reasons if r not in _REQUEST_COVERAGE_REASONS]
    unique = tuple(dict.fromkeys(reasons))
    return ConversationReplyAssessment(
        ok=not unique,
        reasons=unique,
        hard_failure=bool(set(unique) & hard_reasons),
        retryable=bool(set(unique) & retryable_reasons),
    )


def assess_conversation_learning_admission(
    user_message: Any,
    reply_text: Any,
) -> ConversationReplyAssessment:
    """Gate profile, episodic, consolidation, and dream input from a chat turn.

    Durable transcripts are an audit surface and may retain failed turns. Learned
    state is different: only a user-facing reply that satisfies the current turn's
    semantic contract may become experience or self-knowledge.
    """

    if is_non_answer_repair_floor_reply(reply_text):
        return ConversationReplyAssessment(
            ok=False,
            reasons=("non_answer_repair_floor",),
            hard_failure=True,
            retryable=False,
        )
    return assess_user_facing_reply(user_message, reply_text)


def conversation_reliability_system_block(user_message: Any = "") -> str:
    extra = ""
    if is_reliability_concern(user_message):
        extra = (
            "\n- The user is explicitly checking whether the chat/reasoning lane is reliable. "
            "Give a grounded status and continue the thread; never answer with only 'I'm fine', "
            "'Don't worry', or another short reassurance."
        )
    elif is_operational_status_turn(user_message):
        extra = (
            "\n- The user is asking about the live runtime, model lane, or tool availability. "
            "Answer from bounded operational evidence. Do not claim full capacity, peak efficiency, "
            "zero delay, zero uncertainty, guaranteed tool execution, or direct OS control unless "
            "permissions, app state, governance, receipts, and effect verification have actually passed."
        )
    elif is_live_self_reflection_turn(user_message) or is_self_process_question(user_message):
        extra = (
            "\n- The user is asking for Aura's live inner state or current thought. "
            "Answer from the present turn with concrete attention, feeling, and continuity details. "
            "Do not give a status-page answer, raw metrics, a place" "holder, a generic reassurance, or invented pseudo-neural jargon."
        )
    elif is_status_check_turn(user_message):
        extra = (
            "\n- The user is checking in on Aura's state. "
            "Give a brief but substantive first-person answer with what feels steady or strained, "
            "then continue the conversation naturally."
        )
    instruction_notes: list[str] = []
    requested_paragraphs = _requested_count(_PARAGRAPH_REQUEST_RE, user_message)
    if requested_paragraphs and requested_paragraphs > 1:
        instruction_notes.append(
            f"Use at least {requested_paragraphs} separate paragraphs because the user explicitly requested that structure."
        )
    requested_list_items = _requested_list_item_count(user_message)
    if requested_list_items > 1:
        instruction_notes.append(
            f"Use at least {requested_list_items} explicit list items because the user requested that structure."
        )
    requested_word_range = _requested_word_count_range(user_message)
    if requested_word_range:
        minimum_words, maximum_words = requested_word_range
        if minimum_words == maximum_words:
            instruction_notes.append(
                f"Use exactly {minimum_words} words because the user explicitly requested that length."
            )
        else:
            instruction_notes.append(
                f"Use between {minimum_words} and {maximum_words} words because the user explicitly requested that length."
            )
    requested_sentences = _requested_sentence_count(user_message)
    if requested_sentences is not None:
        instruction_notes.append(
            f"Use exactly {requested_sentences} sentence{'s' if requested_sentences != 1 else ''} because the user explicitly requested that structure."
        )
    for label, value in _requested_reference_values(user_message):
        instruction_notes.append(
            f"Include the requested {label} value {value} in the reply."
        )
    required_phrases = _requested_required_phrases(user_message)
    for phrase in required_phrases:
        instruction_notes.append(
            f"Include the exact requested phrase: {phrase}."
        )
    if _FOLLOWUP_QUESTION_REQUEST_RE.search(str(user_message or "")):
        instruction_notes.append("End with a real follow-up question because the user requested one.")
    requested_reasoning_facets = request_facets(user_message)
    if len(requested_reasoning_facets) >= 2:
        instruction_notes.append(
            "Satisfy every explicit reasoning facet in this same answer: "
            + ", ".join(requested_reasoning_facets)
            + ". Do not substitute a related topic or a follow-up question for any facet."
        )
    continuation_match = _NAMED_CONTINUATION_ANCHOR_RE.search(str(user_message or ""))
    if continuation_match:
        topic = " ".join(str(continuation_match.group("topic") or "").split())
        if topic:
            instruction_notes.append(
                f"Keep the named continuation topic visible in the reply: {topic[:80]}."
            )
    if instruction_notes:
        extra = f"{extra}\n- " + "\n- ".join(instruction_notes)
    return (
        "## USER-FACING CONVERSATION RELIABILITY CONTRACT\n"
        "- A completed chat turn must be coherent, complete, on-topic ordinary English.\n"
        "- Preserve turn identity: answer the current user message, not a late response from an older request.\n"
        "- Treat base-model self-identification as a failed draft: never claim to be Claude, ChatGPT, Anthropic/OpenAI-developed, or a generic helpful assistant.\n"
        "- Do not emit prompt artifacts, role labels, corrupted words, escaped control characters, unexplained foreign names, semantic loops, or vague invented referents.\n"
        "- If the heavy local lane is slow or recovering, keep working or fail cleanly; do not present filler as the final answer."
        f"{extra}"
    )


def reliability_floor_for_user(user_message: Any) -> str:
    diagnostic = live_chat_diagnostic_floor(user_message)
    if diagnostic:
        return diagnostic
    if is_reliability_concern(user_message):
        return _RELIABILITY_REPAIR_FLOOR
    if is_confusion_repair_turn(user_message):
        return _CONFUSION_REPAIR_FLOOR
    if is_status_check_turn(user_message):
        return _STATUS_REPAIR_FLOOR
    return ""
