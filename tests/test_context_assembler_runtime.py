from core.brain.llm.context_assembler import ContextAssembler
from core.container import ServiceContainer
from core.state.aura_state import AuraState


def test_short_self_inquiry_is_not_treated_as_casual():
    assert ContextAssembler._is_casual_interaction("Do you feel anything?") is False
    assert ContextAssembler._is_casual_interaction("Is Aura conscious?") is False


def test_short_greeting_stays_casual():
    assert ContextAssembler._is_casual_interaction("hey") is True


def test_build_messages_updates_attention_focus():
    state = AuraState.default()
    state.cognition.attention_focus = None

    ContextAssembler.build_messages(state, "Let's debug the retrieval pipeline.")

    assert state.cognition.attention_focus == "Let's debug the retrieval pipeline."


class ToolAffordanceRecorder:
    def __init__(self):
        self.calls = 0
        self.kwargs = []

    def build_tool_affordance_block(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        return "## LIVE TOOL OPTIONS\n- clock: Check time and date."


def test_build_system_prompt_uses_compact_turn_specific_tool_affordances(monkeypatch):
    state = AuraState.default()
    state.cognition.current_objective = "What time is it right now?"

    engine = ToolAffordanceRecorder()

    original_get = ServiceContainer.get

    def _get(name, default=None):
        if name == "capability_engine":
            return engine
        return original_get(name, default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))

    prompt = ContextAssembler.build_system_prompt(state)

    assert "## LIVE TOOL OPTIONS" in prompt
    assert "If you need facts, USE web_search/search_web/free_search." not in prompt
    assert engine.calls == 1
    assert engine.kwargs[0]["objective"] == "What time is it right now?"
