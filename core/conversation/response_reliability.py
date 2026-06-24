"""User-facing conversation reliability checks.

This module intentionally stays small and dependency-light. It is used at
multiple choke points so bad chat output is treated as a failed generation, not
as a successful answer that later systems have to explain away.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.runtime.structured_input import looks_like_learning_resource_bundle

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
    r"\b(?:xublcate|ingediate|evocer|brolen|thlought|lllot)\b",
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
    r"i am still thinking|i'?m still thinking)\b",
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
        ),
    ),
    (
        "planning",
        ("plan", "planning", "planner", "decide", "decision", "route", "routing"),
        ("plan", "planning", "decide", "decision", "route", "routing", "choose"),
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
    r"route probe|desktop route|live desktop route|"
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
_EXACT_REPLY_RE = re.compile(
    r"(?:say|reply|respond|answer|return|print)\s+exactly\s*:?\s*[\"'“”‘’]*(?P<target>.+?)\s*[\"'“”‘’]*\s*$",
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
}
_COUNT_TOKEN_RE = r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
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


@dataclass(frozen=True)
class ConversationReplyAssessment:
    ok: bool
    reasons: tuple[str, ...]
    hard_failure: bool
    retryable: bool

    def has(self, reason: str) -> bool:
        return reason in self.reasons


def _normalize(text: Any) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\bdont'?\b", "don't", normalized)


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


def _requested_count(pattern: re.Pattern[str], user_message: Any) -> int | None:
    match = pattern.search(str(user_message or ""))
    if not match:
        return None
    return _count_token_to_int(match.groupdict().get("count"))


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


def _instruction_coverage_reasons(user_message: Any, reply_text: Any) -> list[str]:
    user = str(user_message or "")
    reply = str(reply_text or "").strip()
    if not user or not reply:
        return []

    reasons: list[str] = []
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

    if _FOLLOWUP_QUESTION_REQUEST_RE.search(user) and "?" not in reply:
        reasons.append("missing_requested_followup_question")
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
        grounded_persistence = bool(
            re.search(
                r"\b(?:durable|persist(?:ent|ed|s)?|stored|memory\s+(?:write|gateway|store))\b",
                reply,
            )
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


def repair_instruction_shape(user_message: Any, reply_text: Any) -> str:
    """Deterministically repair explicit structure misses without another model call."""
    user = str(user_message or "")
    original = str(reply_text or "").strip()
    if not user or not original:
        return original
    normalized_original = normalize_user_facing_format(original)
    if not set(_instruction_coverage_reasons(user, original)):
        return normalized_original

    repaired = normalized_original
    sentences = _split_sentences(repaired)

    requested_bullets = _requested_count(_BULLET_REQUEST_RE, user)
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
    return repaired.strip()


def repair_generic_assistant_language(user_message: Any, reply_text: Any) -> str:
    """Remove known assistant-boilerplate sentences without lowering the quality gate."""
    del user_message  # reserved for future context-aware live-voice repair
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
    if len(repaired.split()) < 8:
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


def is_status_check_turn(user_message: Any) -> bool:
    text = _normalize(user_message).rstrip(" ?!.")
    return bool(text and any(marker in text for marker in _STATUS_CHECK_MARKERS))


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
    return bool(_EXACT_REPLY_RE.search(str(user_message or "").strip()))


def _matches_exact_reply_request(user_message: Any, reply_text: Any) -> bool:
    raw_user = str(user_message or "").strip()
    raw_reply = str(reply_text or "").strip()
    if not raw_user or not raw_reply:
        return False
    match = _EXACT_REPLY_RE.search(raw_user)
    if not match:
        return False
    target = match.group("target").strip(" .!?\t\r\n\"'“”‘’")
    reply = raw_reply.strip(" .!?\t\r\n\"'“”‘’")
    return bool(target and _normalize(target) == _normalize(reply))


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


def _has_low_signal_acknowledgement_placeholder(user_message: Any, reply_text: Any) -> bool:
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

    if _word_count(reply) < 8:
        return False
    return any(marker in reply for marker in _SUBSTANTIVE_RELIABILITY_MARKERS)


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
    return any(marker in reply for marker in _OPERATIONAL_STATUS_SUBSTANCE_MARKERS)


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
            "I am on the live desktop cognitive path when the inference gate and conversation probes are green; "
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
    if any(
        marker in prompt_norm
        for marker in (
            "tool",
            "tools",
            "desktop",
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
        if any(marker in prompt_norm for marker in prompt_markers) and not any(
            marker in reply_norm for marker in reply_markers
        ):
            missing.append(name)
    return tuple(missing)


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
    if not _SURFACE_NONSENSE_DRIFT_RE.search(raw):
        return False
    return not _SURFACE_NONSENSE_DRIFT_RE.search(str(user_message or ""))


def _has_truncated_tail(reply_text: Any) -> bool:
    body = str(reply_text or "").strip()
    if len(body) < 24:
        return False
    if _STRUCTURAL_INCOMPLETE_TAIL_RE.search(body):
        return True
    if _STRUCTURAL_UNPUNCTUATED_TAIL_RE.search(body):
        return True
    if _DANGLING_GERUND_TAIL_RE.search(body):
        return True
    if _PUNCTUATED_INCOMPLETE_TAIL_RE.search(body):
        return True
    terminal_word_match = re.search(r"([A-Za-z]+)[.!?\"'”’)\]]*$", body)
    if terminal_word_match and len(body) >= 40:
        terminal_word = terminal_word_match.group(1).lower()
        if len(terminal_word) <= 2 and terminal_word not in _ALLOWED_SHORT_TAIL_WORDS:
            return True
    if body.endswith(("...", "…")):
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
    for n in (4, 3, 2):
        counts: dict[tuple[str, ...], int] = {}
        for i in range(0, max(0, len(lower_words) - n + 1)):
            gram = tuple(lower_words[i:i + n])
            if sum(1 for part in gram if part not in stop_words) < 2:
                continue
            counts[gram] = counts.get(gram, 0) + 1
        if any(count >= 3 for count in counts.values()):
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
        if _BROKEN_LANE_BOILERPLATE_RE.search(raw):
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
    if _BROKEN_LANE_BOILERPLATE_RE.search(raw):
        reasons.append("runtime_boilerplate")
    if user_facing and _RAW_TOOL_RESULT_FRAGMENT_RE.match(raw):
        reasons.append("raw_tool_result_fragment")
    if user_facing and _RAW_LANE_TELEMETRY_RE.search(raw):
        reasons.append("raw_lane_telemetry")
    if user_facing and _LIVE_DESKTOP_GATE_LEAK_RE.search(raw):
        reasons.append("internal_live_gate_leak")
    if user_facing and _RAW_MODEL_IDENTITY_LEAK_RE.search(raw):
        reasons.append("raw_model_identity_leak")
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
    if user_facing and is_non_answer_repair_floor_reply(raw):
        expected_floor = reliability_floor_for_user(prompt) if prompt else ""
        matches_expected_floor = bool(expected_floor and _normalize(expected_floor) == _normalize(raw))
        if not matches_expected_floor:
            reasons.append("friendly_failure_floor")
    if _KNOWN_CORRUPT_RE.search(raw):
        reasons.append("corrupted_language")
    if _DIALOGUE_DERAILMENT_RE.search(raw):
        reasons.append("dialogue_derailment")
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
        "raw_model_identity_leak",
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
        "unfounded_alarm_derailment",
        "unfounded_voice_intrusion",
        "unrequested_pop_culture_intrusion",
        "unexpected_cjk_intrusion",
        "surface_nonsense_drift",
        "unsupported_affection_claim",
        "unsupported_self_telemetry_claim",
        "format_meta_artifact",
        "corrupted_social_fragment",
        "unsupported_operational_status_overclaim",
        "unsupported_runtime_telemetry_inference",
        "unsupported_tool_readiness_claim",
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
    recent_messages = [str(message or "") for message in (recent_user_messages or ())]
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
            "unrequested_pop_culture_intrusion",
            "unexpected_cjk_intrusion",
            "surface_nonsense_drift",
            "unsupported_affection_claim",
            "unsupported_self_telemetry_claim",
            "format_meta_artifact",
            "corrupted_language",
            "unsupported_operational_status_overclaim",
            "unsupported_runtime_telemetry_inference",
            "unsupported_tool_readiness_claim",
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

    user_norm = _normalize(user_message)
    if _CORRUPTED_SOCIAL_FRAGMENT_RE.search(raw) and "lol" not in user_norm:
        reasons.append("corrupted_social_fragment")
    if is_confusion_repair_turn(user_message) and _unexpected_short_foreign_name(user_message, raw):
        reasons.append("foreign_name_intrusion")
    if _has_low_signal_acknowledgement_placeholder(user_message, raw):
        reasons.append("low_signal_acknowledgement_placeholder")

    reliability_turn = is_reliability_concern(user_message)
    reliability_diagnostic_turn = _requires_reliability_diagnostic(user_message)
    exact_reply = _matches_exact_reply_request(user_message, raw)
    strict_answer_tag_reply = _matches_strict_answer_tag_request(user_message, raw)
    if reliability_turn:
        if _LOW_SIGNAL_REASSURANCE_RE.match(raw):
            reasons.append("low_signal_reliability_reply")
        elif reliability_diagnostic_turn and _RELIABILITY_DIAGNOSTIC_DEFLECTION_RE.search(raw):
            reasons.append("reliability_diagnostic_deflection")
        elif reliability_diagnostic_turn and not _has_reliability_diagnostic_substance(raw):
            reasons.append("reliability_diagnostic_too_thin")
        elif not _has_reliability_substance(raw):
            reasons.append("too_thin_for_reliability_turn")
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
    elif not exact_reply and not strict_answer_tag_reply and _requires_substantive_reply(user_message):
        words = _word_count(raw)
        if _LOW_SIGNAL_REASSURANCE_RE.match(raw) or words < 2:
            reasons.append("too_short_for_user_turn")
        elif words < 6 and not _is_tiny_direct_turn(user_message):
            if not (words >= 3 and any(w in raw.lower() for w in ("thinking", "working", "processing", "online"))):
                reasons.append("too_thin_for_user_turn")
        elif not _is_task_turn(user_message):
            open_ended = any(marker in user_norm for marker in _OPEN_ENDED_MARKERS)
            if open_ended and words < 12:
                reasons.append("too_thin_for_open_ended_turn")

    if is_confusion_repair_turn(user_message) and _word_count(raw) < 8:
        if not (_word_count(raw) >= 3 and any(w in raw.lower() for w in ("thinking", "working", "processing", "online"))):
            reasons.append("too_thin_for_confusion_repair")

    reasons.extend(_instruction_coverage_reasons(user_message, raw))
    reasons.extend(_semantic_coverage_reasons(user_message, raw))
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
        "raw_model_identity_leak",
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
        "unfounded_alarm_derailment",
        "unfounded_voice_intrusion",
        "unsupported_context_continuation_claim",
        "unrequested_pop_culture_intrusion",
        "unexpected_cjk_intrusion",
        "surface_nonsense_drift",
        "unsupported_affection_claim",
        "unsupported_self_telemetry_claim",
        "format_meta_artifact",
        "low_signal_acknowledgement_placeholder",
        "question_back_non_answer",
        "missing_current_request_recap",
        "missing_runtime_path_answer",
        "direct_answer_deflection",
        "unsupported_operational_status_overclaim",
        "unsupported_runtime_telemetry_inference",
        "unsupported_tool_readiness_claim",
        "missing_self_claim_evidence_boundary",
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
        "empty_requested_list_item",
        "missing_requested_paragraph_count",
        "missing_requested_list_count",
        "missing_requested_followup_question",
        "missing_future_memory_answer",
        "missing_identity_answer",
        "missing_requested_self_process_coverage",
        "unsupported_memory_guarantee",
    }
    unique = tuple(dict.fromkeys(reasons))
    return ConversationReplyAssessment(
        ok=not unique,
        reasons=unique,
        hard_failure=bool(set(unique) & hard_reasons),
        retryable=bool(set(unique) & retryable_reasons),
    )


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
    if _FOLLOWUP_QUESTION_REQUEST_RE.search(str(user_message or "")):
        instruction_notes.append("End with a real follow-up question because the user requested one.")
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
