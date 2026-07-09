from types import SimpleNamespace

import pytest

from core.introspection.inner_monologue import ThoughtPacket
from core.brain.language_center import LanguageCenter


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result


@pytest.mark.asyncio
async def test_language_center_dispatches_expression_as_messages():
    generate = AsyncCallRecorder(result="Sharp answer.")
    router = SimpleNamespace(generate=generate)

    center = LanguageCenter()
    center._router = router

    thought = ThoughtPacket(
        stance="Here's the point.",
        primary_points=["Say the point clearly."],
        model_tier="primary",
        tone="direct",
        length_target="brief",
        llm_briefing="SYSTEM BRIEF",
    )

    result = await center.express(
        thought,
        "What do you think?",
        history=[
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ],
    )

    assert result == "Sharp answer."
    assert len(generate.calls) == 1
    kwargs = generate.calls[0].kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "SYSTEM BRIEF"},
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "What do you think?"},
    ]
    assert kwargs["prefer_tier"] == "primary"
    assert kwargs["purpose"] == "expression"
    assert kwargs["origin"] == "user"
    assert kwargs["is_background"] is False


@pytest.mark.asyncio
async def test_language_center_can_mark_autonomous_expression_as_background():
    generate = AsyncCallRecorder(result="Quiet reflection.")
    router = SimpleNamespace(generate=generate)

    center = LanguageCenter()
    center._router = router

    thought = ThoughtPacket(
        stance="Reflect quietly.",
        primary_points=["Stay internal."],
        model_tier="primary",
        tone="thoughtful",
        length_target="brief",
        llm_briefing="SYSTEM BRIEF",
    )

    result = await center.express(
        thought,
        "What should I explore next?",
        origin="autonomous",
    )

    assert result == "Quiet reflection."
    assert len(generate.calls) == 1
    kwargs = generate.calls[0].kwargs
    assert kwargs["origin"] == "autonomous"
    assert kwargs["is_background"] is True
