from core.brain.llm.context_assembler import ContextAssembler
from core.container import ServiceContainer
from core.runtime.principal_context import relational_principal_scope
from core.social.relational_memory import RelationalMemoryAuthority
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


def test_context_assembler_uses_exact_active_social_agent_without_intimacy_claims():
    class ExactAgentEstimator:
        active_agent_id = "bryan"

        def __init__(self):
            self.requested_agents = []

        def context_injection(self, agent_id):
            self.requested_agents.append(agent_id)
            return f"SOCIAL_MARKER agent={agent_id} hypothesis_only=true"

    estimator = ExactAgentEstimator()
    ServiceContainer.clear()
    ServiceContainer.register_instance(
        "other_agent_model",
        estimator,
        required=False,
    )
    state = AuraState.default()
    state.cognition.current_objective = "Continue the architecture review."

    try:
        prompt = ContextAssembler.build_system_prompt(state)
    finally:
        ServiceContainer.clear()

    assert estimator.requested_agents == ["bryan"]
    assert "SOCIAL_MARKER agent=bryan hypothesis_only=true" in prompt
    lowered = prompt.lower()
    assert "## who i'm talking to" not in lowered
    assert "deep bond" not in lowered
    assert "be more personal" not in lowered
    assert "high rapport → lean in" not in lowered
    assert "relational register: intimate" not in lowered


def test_context_assembler_excludes_unscoped_legacy_relationship_memory():
    class LegacySocialMemory:
        @staticmethod
        def get_social_context():
            return "PRIVATE_OTHER_USER_RELATIONSHIP"

    ServiceContainer.clear()
    ServiceContainer.register_instance(
        "social_memory",
        LegacySocialMemory(),
        required=False,
    )

    try:
        prompt = ContextAssembler.build_system_prompt(AuraState.default())
    finally:
        ServiceContainer.clear()

    assert "PRIVATE_OTHER_USER_RELATIONSHIP" not in prompt


def test_context_assembler_injects_only_consented_exact_agent_memory(tmp_path):
    authority = RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"k" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    authority.grant_consent(
        "bryan",
        kinds=["boundary"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    authority.record(
        "bryan",
        kind="boundary",
        content="Keep the project codename private.",
    )
    authority.grant_consent(
        "alice",
        kinds=["boundary"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-2",
    )
    authority.record(
        "alice",
        kind="boundary",
        content="ALICE_PRIVATE_BOUNDARY",
    )

    estimator = type(
        "Estimator",
        (),
        {
            "active_agent_id": "bryan",
            "context_injection": lambda self, agent_id: f"agent={agent_id}",
        },
    )()
    ServiceContainer.clear()
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    ServiceContainer.register_instance("relational_memory", authority, required=False)

    try:
        prompt = ContextAssembler.build_system_prompt(AuraState.default())
    finally:
        ServiceContainer.clear()

    assert "Keep the project codename private." in prompt
    assert "ALICE_PRIVATE_BOUNDARY" not in prompt


def test_request_scoped_principal_overrides_process_global_active_agent(tmp_path):
    authority = RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"k" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    for agent_id, content in (
        ("bryan", "BRYAN_PRIVATE_BOUNDARY"),
        ("alice", "ALICE_PRIVATE_BOUNDARY"),
    ):
        authority.grant_consent(
            agent_id,
            kinds=["boundary"],
            operations=["persist", "recall", "prompt"],
            receipt_id=f"grant-{agent_id}",
        )
        authority.record(agent_id, kind="boundary", content=content)

    class Estimator:
        active_agent_id = "bryan"

        def __init__(self):
            self.requested_agents = []

        def context_injection(self, agent_id):
            self.requested_agents.append(agent_id)
            return f"agent={agent_id}"

    estimator = Estimator()
    ServiceContainer.clear()
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    ServiceContainer.register_instance("relational_memory", authority, required=False)

    try:
        with relational_principal_scope("alice"):
            prompt = ContextAssembler.build_system_prompt(AuraState.default())
    finally:
        ServiceContainer.clear()

    assert estimator.requested_agents == ["alice"]
    assert "agent=alice" in prompt
    assert "ALICE_PRIVATE_BOUNDARY" in prompt
    assert "BRYAN_PRIVATE_BOUNDARY" not in prompt


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


def test_context_assembler_excludes_proof_fixture_from_lived_continuity():
    state = AuraState.default()
    state.cognition.rolling_summary = (
        "Mode=reactive | Commitments=A long-running microservice periodically "
        "crashes with OSError: too many open files. A code review reveals a resource leak"
    )

    prompt = ContextAssembler.build_system_prompt(state)

    assert "long-running microservice" not in prompt
    assert "code review reveals" not in prompt
    assert "## CONTINUITY SUMMARY" not in prompt


def test_context_assembler_does_not_promote_evaluation_objective_to_attention():
    state = AuraState.default()
    state.cognition.attention_focus = "Bryan's current conversation"
    fixture = (
        "A long-running microservice periodically crashes with OSError; "
        "code review reveals a resource leak."
    )

    messages = ContextAssembler.build_messages(state, fixture, max_tokens=2048)

    assert state.cognition.attention_focus == "Bryan's current conversation"
    assert messages[-1]["content"] == fixture


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
