"""Is this turn ordinary conversation, or is it work?

Extracted so the two routing phases cannot answer it differently. Both
core/phases/cognitive_routing.py and core/phases/cognitive_routing_unitary.py
carried their own copy of a LITERAL PHRASE LIST — "capital of france",
"square root of 64", "who wrote hamlet" — assembled from whatever inputs
earlier tests happened to use. Anything not on the list took the heavy
DELIBERATE lane, which measured at 13 of 18 ordinary conversational turns:
"Good morning!", "Hey Aura", "thanks, that helped", "what did you do today?".

Two copies of a judgement is one copy too many, and the way it fails is that
one phase answers a question the other was going to answer differently.
"""

from __future__ import annotations

import re

from core.runtime.skill_task_bridge import (
    _DIRECT_EXECUTION_PREFIX_RE,
    looks_like_explanatory_dialogue_request,
)

#: The historical literals. Kept so the turns they encode keep working; they
#: are no longer the only way to be recognised as conversation.
_SIMPLE_DIALOGUE_RE = re.compile(
    r"\b("
    r"capital of france|15\s*\*\s*12|square root of 64|3 apples|"
    r"who wrote (?:the play )?hamlet|three programming languages|"
    r"color is the sky|translate ['\"]?good morning|"
    r"continuity check|what did we just verify|live chat path|"
    r"conversation lane|reply path|response path|"
    r"you ok|you okay|are you ok|are you okay|"
    r"what feels most important|what should you do differently|"
    r"write (?:a )?(?:short )?(?:poem|joke|haiku)|"
    r"compose (?:a )?(?:short )?(?:poem|joke|haiku)"
    r")\b",
    re.IGNORECASE,
)

#: Openers that are conversation by construction — a greeting or an
#: acknowledgement is never a request to go do something.
_CONVERSATIONAL_OPENER_RE = re.compile(
    r"^\s*(?:"
    r"hi|hey|hello|yo|howdy|"
    r"good\s+(?:morning|afternoon|evening|night)|morning|goodnight|"
    r"thanks|thank\s+you|ty|ok|okay|k|cool|nice|great|awesome|lovely|"
    r"got\s+it|sounds\s+good|fair\s+enough|makes\s+sense|i\s+see|"
    r"yeah|yep|yes|no\s+worries|np|sorry|oops"
    r")\b",
    re.IGNORECASE,
)

#: Questions ABOUT HER — her state, her day, her preferences, her opinion.
#: Deliberately excludes the request forms ("can you", "could you", "would
#: you"), because those are how a person asks for an ACTION and the skill
#: path below has to keep them.
_ABOUT_HER_RE = re.compile(
    r"\b(?:"
    r"(?:do|did|does|are|was|were|have|has)\s+you\b"
    r"|(?:what|how|why|when|where|which)\s+"
    r"(?:do|did|does|are|is|was|were|have|has)\s+you\b"
    r"|what'?s\s+(?:your|on\s+your\s+mind)"
    r"|how'?s\s+(?:your|it\s+going)"
    r"|tell\s+me\s+(?:about\s+yourself|something|a\s+story|more)"
    r"|your\s+(?:favou?rite|opinion|take|thoughts?|day|morning|mood|"
    r"feelings?|experience)"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_simple_dialogue_request(text: str) -> bool:
    """Whether this turn is ordinary conversation, not work.

    This used to be a LITERAL PHRASE LIST — "capital of france",
    "square root of 64", "who wrote hamlet" — assembled from whatever
    specific inputs earlier tests happened to use. Everything else fell
    through to the DELIBERATE lane, so "Good morning!", "Hey Aura",
    "thanks, that helped" and "what did you do today?" all took the heavy
    32B deliberate path. Measured before this change: 13 of 18 ordinary
    conversational turns. That is not a routing policy, it is the absence
    of one, and the person on the other end pays for it in latency on
    every sentence that is not on the list.

    Judged by SHAPE now. The literals stay so the turns they encode keep
    working, but a greeting is recognised as a greeting.

    The exclusion is what keeps this safe: an execution prefix ("open ...",
    "can you search ...") returns False here and falls through to skill
    detection below, so widening the conversational lane cannot swallow a
    request to actually do something.
    """
    body = str(text or "").strip()
    if not body:
        return False
    if looks_like_explanatory_dialogue_request(body):
        return True
    if len(body.split()) > 28:
        return False
    if _SIMPLE_DIALOGUE_RE.search(body):
        return True
    if _DIRECT_EXECUTION_PREFIX_RE.search(body):
        # A request to DO something. The skill/task path owns it.
        return False
    if _CONVERSATIONAL_OPENER_RE.match(body):
        return True
    return bool(_ABOUT_HER_RE.search(body))




__all__ = ["looks_like_simple_dialogue_request"]


#: Public name. The underscore version is kept as an alias because both
#: routing phases already call it that.
looks_like_simple_dialogue_request = _looks_like_simple_dialogue_request
