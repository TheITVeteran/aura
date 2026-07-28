"""An execution directive does not have to open the sentence.

Asked live: "Do this for real now: open Chrome, take a screenshot of what is
on my screen, and tell me what you actually see. Use your desktop control."

The turn never reached the desktop lane. looks_like_capability_inventory_
dialogue_request classified it as a QUESTION about her capabilities, and the
desktop router bails on that basis — so the request fell through to ordinary
chat, where she narrated a screen she had never looked at, complete with
Notepad++ and File Explorer on a Mac.

The guard against exactly this existed. It only looked at position zero, so
any preamble — "Do this for real now:", "When you get a chance, please",
"First" — hid the instruction from it.

The second half of the corpus is what keeps the fix honest: "what tools can
you use to open apps" contains an execution verb and an app noun and is
still a question, because the verb is the object of the question rather than
an instruction.
"""

import pytest

from core.runtime.skill_task_bridge import (
    looks_like_capability_inventory_dialogue_request,
)
from interface.routes.chat import _looks_like_desktop_objective

pytestmark = pytest.mark.unit

INSTRUCTIONS = [
    "Do this for real now: open Chrome, take a screenshot of what is on my "
    "screen, and tell me what you actually see.",
    "open chrome and take a screenshot",
    "take a screenshot of my screen",
    "When you get a chance, please open a blank doc and write a paragraph",
    "First open Finder, then create a folder called reports",
]

QUESTIONS = [
    "what tools can you use",
    "what can you actually do on this computer right now?",
    "what tools can you use to open apps",
    "what are your capabilities",
]


@pytest.mark.parametrize("message", INSTRUCTIONS)
def test_an_instruction_reaches_the_desktop_lane(message: str):
    assert _looks_like_desktop_objective(message), (
        "an instruction that never routes is an instruction she will narrate "
        "instead of perform"
    )


@pytest.mark.parametrize("message", QUESTIONS)
def test_a_question_is_still_a_question(message: str):
    assert looks_like_capability_inventory_dialogue_request(message)
