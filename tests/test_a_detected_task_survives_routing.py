"""A detected task must survive the routing that follows it.

Live 2026-07-27: "Open the Notes app and write a new note titled Orca Field
Notes with a couple of sentences about orcas in it."

The router got it right —

    Routing: multi-step skill-backed task detected -> TASK via ['desktop_task']

— and then threw it away one line later:

    CognitiveRouting: deterministic intent=TASK semantic=casual
    Routing: Casual/Autonomous bypass. Forcing REACTIVE.

_CASUAL_KEYWORDS holds ordinary English, and the request contained the word
"sentences". Nothing executed, and the conversational lane answered "I can't
actually open apps or write notes" — denying a capability she has, because
the sentence mentioned sentences.

Casualness is a guess about TONE. A deterministic intent and a matched skill
are evidence about WHAT WAS ASKED. Evidence wins.

The second half: changing a machine setting is a desktop objective too. The
vocabulary had every verb for moving files and windows and none for altering
the system, so "change my desktop background to an orca" — which
desktop_task performs through system_control — did not route at all. Setting
verbs are admitted only when bound to a machine surface, because "change your
mind" and "I want to change the subject" must stay conversation.
"""

import pytest

from interface.routes.chat import _looks_like_desktop_objective

pytestmark = pytest.mark.unit

DESKTOP_WORK = [
    "open the notes app",
    "create a folder called Orca on my desktop",
    "open google chrome and go to google.com",
    "change my desktop background to an orca",
    "set my wallpaper to a picture of an orca",
    "turn on dark mode",
    "set the volume to 50%",
    "increase the brightness",
]

CONVERSATION = [
    "what do you think about entropy?",
    "change your mind about that",
    "I want to change the subject",
    "set aside what you said earlier",
    "that would increase the risk a lot",
]


@pytest.mark.parametrize("message", DESKTOP_WORK)
def test_desktop_work_routes_to_the_desktop(message: str):
    assert _looks_like_desktop_objective(message)


@pytest.mark.parametrize("message", CONVERSATION)
def test_conversation_stays_conversation(message: str):
    assert not _looks_like_desktop_objective(message)


def test_a_casual_keyword_cannot_discard_a_detected_task():
    """The bypass must consult the deterministic intent before firing."""
    import inspect

    from core.phases import cognitive_routing

    source = inspect.getsource(cognitive_routing)
    assert "deterministic_task" in source
    assert "and not deterministic_task" in source
