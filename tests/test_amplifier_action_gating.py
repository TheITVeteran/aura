"""The amplifier must not hijack imperative actions/tool commands."""
from __future__ import annotations

import pytest

from core.brain.reasoning_amplifier_v2 import is_action_request, is_amplifiable


@pytest.mark.parametrize("text", [
    "Open 3 tabs",
    "open three tabs in chrome",
    "click the submit button",
    "send the email to Sarah",
    "play the next song",
    "go to github.com",
    "set a timer for 5 minutes",
    "download the report and save it",
    "delete the temp files",
    "schedule a meeting for 3pm",
    "navigate to the settings page",
    "turn off notifications",
    "please open the file manager",
    "can you click on the link",
])
def test_actions_are_not_amplified(text):
    assert is_action_request(text) is True
    assert is_amplifiable(text) is None


@pytest.mark.parametrize("text,expected", [
    ("write a function that adds two numbers", "code"),
    ("compute the product of 12 and 12", "math"),
    ("where is the inference gate implemented", "repo_audit"),
    ("describe the architecture of the subprocess gateway", "repo_audit"),
])
def test_reasoning_questions_still_amplify(text, expected):
    assert is_action_request(text) is False
    assert is_amplifiable(text) == expected


@pytest.mark.parametrize("text", [
    "how are you today",
    "tell me a story",
    "what do you think about jazz",
])
def test_casual_chat_not_amplified(text):
    assert is_amplifiable(text) is None


def test_action_with_number_not_treated_as_math():
    # The specific case raised: a digit in an action must not route to math.
    assert is_amplifiable("Open 3 tabs") is None
    assert is_amplifiable("click button 2 times") is None
    assert is_amplifiable("send 5 emails to the team") is None


@pytest.mark.asyncio
async def test_phase_skips_action_turn():
    import types

    from core.phases.response_generation_unitary import UnitaryResponsePhase

    class _LLM:
        calls = 0

        async def think(self, prompt, **kw):
            type(self).calls += 1
            return "x"

    llm = _LLM()
    out = await UnitaryResponsePhase._maybe_amplify_response(
        types.SimpleNamespace(_last_reasoning_receipt=None),
        objective="open 3 tabs and click the first result",
        draft="Opening the tabs now.",
        llm=llm,
        state=types.SimpleNamespace(metadata={}),
        request_timeout=20.0,
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )
    assert out == "Opening the tabs now."
    assert llm.calls == 0
