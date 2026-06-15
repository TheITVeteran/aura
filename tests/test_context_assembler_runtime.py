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


def test_context_assembler_methods_are_not_replaced_at_import_time():
    assert ContextAssembler._is_casual_interaction.__module__ == "core.brain.llm.context_assembler"
    assert ContextAssembler.build_system_prompt.__module__ == "core.brain.llm.context_assembler"
    assert ContextAssembler.build_messages.__module__ == "core.brain.llm.context_assembler"
    assert not getattr(ContextAssembler, "_patched_v1", False)


def test_deep_conversation_keeps_compact_continuity():
    state = AuraState.default()
    state.cognition.current_objective = "Continue the architecture review."
    state.cognition.rolling_summary = "We are preserving the canonical desktop path."
    state.cognition.modifiers["continuity_obligations"] = {
        "identity_mismatch": False,
        "current_objective": "Harden live context assembly",
        "active_commitments": ["keep the live path coherent"],
        "subject_thread": "desktop reliability",
    }
    state.cognition.working_memory = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"}
        for index in range(32)
    ]

    prompt = ContextAssembler.build_system_prompt(state)

    assert "## CONTINUITY SUMMARY" in prompt
    assert "canonical desktop path" in prompt
    assert "## TEMPORAL OBLIGATIONS" in prompt
    assert "Harden live context assembly" in prompt


def test_build_messages_preserves_current_input_under_tight_budget(monkeypatch):
    state = AuraState.default()
    state.response_modifiers["black_box_steering"] = True
    objective = "BEGIN-" + ("x" * 9000) + "-END"
    monkeypatch.setattr(
        ContextAssembler,
        "build_system_prompt",
        staticmethod(lambda _state: "SYSTEM-HEAD\n" + ("s" * 20000) + "\nSYSTEM-TAIL"),
    )

    messages = ContextAssembler.build_messages(state, objective, max_tokens=2048)

    assert len(messages[0]["content"]) <= 8192
    assert messages[0]["content"].startswith("SYSTEM-HEAD")
    assert messages[0]["content"].endswith("SYSTEM-TAIL")
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("BEGIN-")
    assert messages[-1]["content"].endswith("-END")
    assert sum(len(message["content"]) for message in messages) <= 8192


def test_build_messages_counts_dropped_history_without_negative_slice(monkeypatch):
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "z" * 500}
        for index in range(12)
    ]
    monkeypatch.setattr(
        ContextAssembler,
        "build_system_prompt",
        staticmethod(lambda _state: "s" * 6500),
    )

    messages = ContextAssembler.build_messages(state, "current", max_tokens=2048)

    notices = [
        message["content"]
        for message in messages
        if message["role"] == "system" and "older conversational messages were omitted" in message["content"]
    ]
    assert notices
    assert "10 older conversational messages" in notices[0]
