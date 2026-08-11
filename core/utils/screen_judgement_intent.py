"""Is this person asking for an OPINION about the screen, not an action on it?

One definition, because two layers need the answer.

The conversational floor answers it from the window layout and perception
evidence — it can name what is open and say which one it would close and why.
The desktop-objective router uses it to decline, because a judgement has no
observable effect for os_automation to verify.

LIVE DEFECT, 2026-08-10. Asked:

    "look at my screen again, then give me an opinion rather than a
     description. of everything you can see open right now, which window would
     you close first if you were me, and why that one? I want your actual
     judgement, not a list."

the router matched its screen-observation branch and sent the turn to the
desktop lane, which answered:

    "os_automation failed: OS automation refused to act because the objective
     has no complete observable acceptance contract … Completed 0/1 steps. I am
     not claiming the desktop action finished."

os_automation was right to refuse — nothing was asked to happen. Nobody asked
her to close a window; they asked which one she WOULD close. The request needs
screen data as evidence and a judgement as the answer, and only the
conversational lane can produce the second half.

Her own agency rules already say this: "Hypotheticals, quoted requests, negated
actions, and recalled evidence are not execution requests merely because they
name a tool." That was written in the identity contract and never expressed in
the router, which is why the rule did not bind. This module is the router's half
of it.

Kept in core/utils deliberately. core/runtime may not import cognition, and a
second copy of this predicate over there is how the two answers drift apart —
the same lesson as core/utils/own_source_intent.py and
core/utils/occluded_view_intent.py.
"""
from __future__ import annotations

import re
from typing import Any

#: Asking for a view, a preference or a recommendation rather than an effect.
JUDGEMENT_REQUEST_RE = re.compile(
    r"\b(?:"
    r"opinion|your\s+take|your\s+view|your\s+judg(?:e)?ment|"
    r"what\s+do\s+you\s+think|which\s+would\s+you|what\s+would\s+you|"
    r"would\s+you\s+(?:close|open|quit|kill|keep|pick|choose|drop|move|"
    r"rearrange|prioriti[sz]e)|"
    r"if\s+you\s+were\s+me|in\s+my\s+(?:position|shoes)|"
    r"recommend|suggest|advise|worth\s+(?:closing|keeping)|"
    r"should\s+i\s+(?:close|open|quit|keep|drop)"
    r")\b",
    re.IGNORECASE,
)

#: A conditional frame. "would you close X" is a question; "close X" is a task.
SUBJUNCTIVE_RE = re.compile(
    r"\b(?:would|could|should|might)\s+(?:you|i|we)\b"
    r"|\bif\s+(?:you|i|we)\s+(?:were|had|wanted)\b",
    re.IGNORECASE,
)

#: An explicit refusal of the list-shaped answer, which is what a capture gives.
WANTS_JUDGEMENT_NOT_LISTING_RE = re.compile(
    r"\b(?:rather\s+than|instead\s+of|not)\s+a?\s*"
    r"(?:description|list|listing|inventory|dump|readout|summary)\b"
    r"|\bnot\s+(?:just\s+)?(?:describe|listing|a\s+list)\b",
    re.IGNORECASE,
)

#: …and it has to be about the screen, or it is not this question at all.
SCREEN_SUBJECT_RE = re.compile(
    r"\b(?:screen|window|windows|display|desktop|monitor|app|apps|"
    r"application|tab|tabs|open)\b",
    re.IGNORECASE,
)

#: An either/or question. "should I close chrome or keep it" names no screen
#: noun at all, and a choice put to her is never an instruction to act.
DELIBERATION_RE = re.compile(
    r"\b(?:should|would|could)\s+(?:i|we|you)\b[^.?!]{0,80}\bor\b"
    r"|\bwhich\s+(?:one|of\s+(?:these|them|those))\b"
    r"|\b(?:better|worth\s+it)\s+to\b",
    re.IGNORECASE,
)

#: A mutating imperative at the start of a clause. This is the one signal that
#: outranks an opinion request: "close the window and give me your opinion on
#: the article" is real desktop work that also wants a view, and declining it
#: would break the work half.
#:
#: Observation verbs are deliberately absent. "look at my screen", "read", "show
#: me" and "tell me" change nothing, so they must not make a judgement question
#: look like an effect — which is exactly the shape of the request that started
#: this: "look at my screen again, then give me an opinion…".
EFFECT_IMPERATIVE_RE = re.compile(
    r"(?:\A|[.;!?]\s+|,\s+(?:then|and|also)\s+|\b(?:then|and)\s+)"
    r"(?:please\s+)?"
    r"(?:close|open|quit|kill|launch|start|write|save|delete|rename|move|"
    r"drag|click|type|paste|minimi[sz]e|maximi[sz]e|resize|switch)\b",
    re.IGNORECASE,
)


def asks_for_screen_judgement(user_message: Any) -> bool:
    """True when the answer wanted is an opinion about the screen.

    Deliberately conservative in one direction: a plain imperative ("close
    Chrome", "open Notes and write a paragraph") must never match, because
    declining those would break real desktop work. The signal required is an
    opinion request or a subjunctive frame — a question about what she would
    do — not merely the presence of an action word.
    """

    raw = str(user_message or "")
    if not raw.strip():
        return False
    # A real instruction to change something outranks any opinion request in
    # the same message. "close the window and give me your opinion on the
    # article" is desktop work that also wants a view; declining it would drop
    # the work.
    if EFFECT_IMPERATIVE_RE.search(raw):
        return False
    # A choice put to her is never an instruction, even with no screen noun in
    # it: "should I close chrome or keep it".
    if DELIBERATION_RE.search(raw):
        return True
    if not SCREEN_SUBJECT_RE.search(raw):
        return False
    if WANTS_JUDGEMENT_NOT_LISTING_RE.search(raw):
        return True
    if not JUDGEMENT_REQUEST_RE.search(raw):
        return False
    # An opinion word alone is not enough to call it a judgement question;
    # require either the conditional frame, or an unambiguous ask for a view.
    return bool(SUBJUNCTIVE_RE.search(raw)) or bool(
        re.search(
            r"\b(?:opinion|your\s+take|your\s+view|your\s+judg(?:e)?ment|"
            r"recommend|suggest|advise|what\s+do\s+you\s+think)\b",
            raw,
            re.IGNORECASE,
        )
    )


__all__ = [
    "JUDGEMENT_REQUEST_RE",
    "SCREEN_SUBJECT_RE",
    "SUBJUNCTIVE_RE",
    "WANTS_JUDGEMENT_NOT_LISTING_RE",
    "asks_for_screen_judgement",
]
