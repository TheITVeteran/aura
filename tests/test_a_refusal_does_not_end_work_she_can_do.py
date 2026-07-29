"""When cognition declines work the executor owns, the executor does it.

Live 2026-07-28, demo rehearsal:

    Open the Notes app and write a new note with three sentences about
    humpback whales. Actually do it.

The route classified it correctly — desktop_execution_contract=True — and the
planner produced a complete, executable plan: open_app Notes, set_clipboard,
cmd+n, cmd+v. Then, because free-form prose is better when the model writes
it, the turn deferred to cognition for the body. Cognition answered:

    I can't physically interact with your device or open apps. I'm just this
    text interface — no hands, no screen to tap on.

and THAT was served. The hands were right there, planned and waiting.

Deferring for better prose is correct. Letting the refusal end a turn she can
perform is not. The detector stays narrow because it only decides whether to
run an executor that was already going to run, on a turn already classified
as a desktop objective.
"""

import pytest

from interface.routes.chat import _looks_like_capability_refusal

pytestmark = pytest.mark.unit

REFUSALS = [
    "I can't physically interact with your device or open apps. I'm just this "
    "text interface — no hands, no screen to tap on.",
    "I can't open apps",
    "I'm not able to do that",
    "I don't have a body",
    "I cannot directly access your files",
]

NOT_REFUSALS = [
    "I opened Notes and wrote the note.",
    "Humpback whales migrate thousands of miles each year.",
    "I can do that — give me a moment.",
    "The result is 19/66.",
    "I don't have that in this conversation's turns, so I won't guess.",
]


@pytest.mark.parametrize("text", REFUSALS)
def test_a_capability_refusal_is_recognised(text: str):
    assert _looks_like_capability_refusal(text)


@pytest.mark.parametrize("text", NOT_REFUSALS)
def test_ordinary_replies_are_not_refusals(text: str):
    assert not _looks_like_capability_refusal(text)


def test_the_route_runs_the_executor_instead_of_serving_the_refusal():
    import inspect

    from interface.routes import chat as chat_routes

    source = inspect.getsource(chat_routes)
    assert "_looks_like_capability_refusal(salvaged_no_reply)" in source
    assert "executed_after_refusal" in source


def test_the_executor_is_called_with_a_valid_signature():
    """This path only fires on a live refusal, so no unit test exercised it —
    and it shipped calling _execute_desktop_objective_from_chat without its
    required cognitive_reply, turning every refusal into status 'error'.
    Binding the real signature catches that without needing the live path.
    """
    import inspect

    from interface.routes.chat import _execute_desktop_objective_from_chat

    signature = inspect.signature(_execute_desktop_objective_from_chat)
    # Must bind exactly as the call site invokes it.
    signature.bind("open notes and write a note", cognitive_reply="")
