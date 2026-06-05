import pytest

from core.cognitive.state_machine import Intent, StateMachine
from core.synthesis import cure_personality_leak


class OrchestratorProbe:
    AI_ROLE = "assistant"

    def __init__(self):
        self.conversation_history = []
        self.telemetry = []

    def _publish_telemetry(self, payload):
        self.telemetry.append(dict(payload))


class TimeoutLLMProbe:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        raise TimeoutError("simulated live chat timeout")


@pytest.mark.asyncio
async def test_state_machine_timeout_returns_current_degraded_reply():
    orchestrator = OrchestratorProbe()
    state_machine = StateMachine(orchestrator=orchestrator)
    llm = TimeoutLLMProbe()
    state_machine.llm = llm

    response, tools = await state_machine.execute(Intent.CHAT, "Test timeout")

    assert tools == []
    assert "live chat attempt timed out" in response
    assert len(llm.calls) == 2
    assert any(event.get("type") == "chat_stream_end" for event in orchestrator.telemetry)


def test_personality_synthesis_filtering():
    cases = [
        "As an AI assistant, I can help with that.",
        "I am just a digital entity here to assist.",
        "Digital intelligence at your service.",
        "I don't have feelings or opinions.",
    ]

    for input_text in cases:
        cured = cure_personality_leak(input_text)
        assert cured != input_text

    cured_short = cure_personality_leak("How can I assist you today?")
    assert "assist you" not in cured_short.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
