"""A question about what she can do must reach the registry, not her memory.

Asked live, 2026-07-27, through the same API the desktop UI uses:

    What can you actually do on this computer right now? Not what you could
    in principle — what is genuinely wired up and working this minute.

She answered from the model's own idea of herself: listed code_repl and
execute_nethack_action, and stated flatly

    I cannot: - Access the internet or perform web searches.

web_search is registered, and had executed successfully minutes earlier in
the same runtime. computer_use, file_operation and desktop_task — the things
a person on a desktop actually wants — went unmentioned.

The trigger was a list of ~40 literal phrases ("what tools can you use").
Nothing in that list matches how the question was asked, so the registry was
never consulted. A phrase list cannot cover how people ask; the shape can.

The negative half is what keeps it honest: "can you do the marble problem?"
also contains her, a capability verb and a question mark, and is a request
for work rather than a question about herself.
"""

import pytest

from interface.routes.chat import _is_explicit_capability_inventory_request

pytestmark = pytest.mark.unit

ASKING_ABOUT_HERSELF = [
    "What can you actually do on this computer right now?",
    "what tools can you use",
    "what are you capable of?",
    "what skills do you have available to you?",
    "What can you do on my computer?",
    "what's actually wired up for you right now?",
    "what capabilities do you have",
    "list your tools",
]

#: Process questions name her, a capability word and a question mark, and want
#: none of the registry. "How" and "why" are asking how something works.
ASKING_ABOUT_PROCESS = [
    "When you are confused, how does that change your planning, memory use, "
    "and tool verification?",
    "how do you decide which tool to reach for?",
    "why does your memory use change under load?",
]

#: Statements. An apology can name her, a capability word and a question word
#: and still be asking nothing — live, one was answered with all 75 skills.
NOT_ASKING_AT_ALL = [
    "Aura, Bryan sent me and I owe you an apology. I could not find a tool "
    "dispatch in the logs.",
    "you were right about the screen and I was wrong about your tools",
    "thanks for listing your skills earlier, that helped",
]

ASKING_FOR_WORK = [
    "can you do the marble problem?",
    "what do you think about entropy?",
    "what did I have for breakfast?",
    "can you open the door for me?",
    "what time is it?",
    "what do you make of the second law?",
]


@pytest.mark.parametrize("message", ASKING_ABOUT_HERSELF)
def test_a_capability_question_is_recognised(message: str):
    assert _is_explicit_capability_inventory_request(message), (
        "this question must be answered from the registry, not from memory"
    )


@pytest.mark.parametrize("message", ASKING_FOR_WORK)
def test_a_request_for_work_is_not_an_inventory_question(message: str):
    assert not _is_explicit_capability_inventory_request(message)


@pytest.mark.parametrize("message", ASKING_ABOUT_PROCESS)
def test_a_process_question_is_not_an_inventory_question(message: str):
    """Asking HOW something works is not asking WHAT is available."""
    assert not _is_explicit_capability_inventory_request(message)


@pytest.mark.parametrize("message", NOT_ASKING_AT_ALL)
def test_a_statement_is_not_an_inventory_question(message: str):
    """An inventory question has to be a question."""
    assert not _is_explicit_capability_inventory_request(message)
