from __future__ import annotations

import asyncio

import pytest

from core.phases.cognitive_routing import CognitiveRoutingPhase
from core.runtime.turn_analysis import analyze_turn, previous_user_turn_text
from core.state.aura_state import AuraState, CognitiveMode


@pytest.mark.parametrize(
    "text",
    (
        "It would be useful if you compared the latest three incident reports.",
        "I was wondering whether you could preserve the verified findings.",
        "The next logical move is to inspect the connected device before acting.",
        "Maybe you should create a concise report from those measurements.",
    ),
)
def test_indirect_action_requests_reach_semantic_task_planning(text):
    analysis = analyze_turn(text)

    assert analysis.intent_type == "TASK"
    assert analysis.request_mood == "directive"


@pytest.mark.parametrize(
    "text",
    (
        "If I asked you to open Notes, what would you do?",
        "Could you explain how you would set the wallpaper?",
        "I don't want you to restart Aura; describe the process instead.",
        '"Download the image and save it to Desktop."',
    ),
)
def test_nonexecuting_action_language_stays_conversational_even_with_skill_hits(text):
    analysis = analyze_turn(text, matched_skills=["computer_use", "web_fetch"])

    assert analysis.intent_type == "CHAT"
    assert analysis.request_mood == "mention"


def test_future_action_and_retrospective_question_keep_distinct_time_direction():
    future = analyze_turn("Tomorrow, create a reminder to inspect the training receipt.")
    past = analyze_turn("Why did you open Notes yesterday?")

    assert future.intent_type == "TASK"
    assert future.temporal_scope == "scheduled"
    assert past.intent_type == "CHAT"
    assert past.temporal_scope == "retrospective"


def test_followup_consent_uses_prior_turn_without_requiring_repeated_words():
    analysis = analyze_turn(
        "Go ahead.",
        previous_user_text="Could you open Notes and write the verified result?",
    )

    assert analysis.intent_type == "TASK"
    assert analysis.request_mood_reasons == ("contextual_action_followup",)


def test_previous_user_turn_skips_the_current_message():
    memory = [
        {"role": "user", "content": "Could you inspect the runtime report?"},
        {"role": "assistant", "content": "I can do that."},
        {"role": "user", "content": "Go ahead."},
    ]

    assert previous_user_turn_text(memory, current_text="Go ahead.") == (
        "could you inspect the runtime report?"
    )


class _RouteContainer:
    def __init__(self, capability_engine=None):
        self.capability_engine = capability_engine

    def get(self, name, default=None):
        if name == "capability_engine":
            return self.capability_engine if self.capability_engine is not None else default
        return default


class _AlwaysMatchesTool:
    def __init__(self):
        self.calls = []

    def detect_intent(self, text):
        self.calls.append(text)
        return ["sovereign_browser"]


def _state(*messages: tuple[str, str]) -> AuraState:
    state = AuraState()
    for role, content in messages:
        state.cognition.working_memory.append(
            {"role": role, "origin": "user" if role == "user" else "aura", "content": content}
        )
    return state


def test_live_router_suppresses_skill_and_url_activation_for_hypothetical():
    async def scenario():
        capability_engine = _AlwaysMatchesTool()
        phase = CognitiveRoutingPhase(_RouteContainer(capability_engine))
        state = _state(
            (
                "user",
                "If I asked you to open https://example.com, how would you decide whether to do it?",
            )
        )

        result = await phase.execute(state)

        assert capability_engine.calls == []
        assert result.response_modifiers["intent_type"] == "CHAT"
        assert result.response_modifiers["request_mood"] == "mention"
        assert "matched_skills" not in result.response_modifiers
        assert "auto_browse_urls" not in result.response_modifiers

    asyncio.run(scenario())


def test_live_router_sends_indirect_request_to_semantic_task_lane():
    async def scenario():
        phase = CognitiveRoutingPhase(_RouteContainer())
        phase._spawn_parallel_branch = lambda *_args, **_kwargs: None
        state = _state(
            (
                "user",
                "It would help if you compared the latest runtime incidents and saved the result.",
            )
        )

        result = await phase.execute(state)

        assert result.cognition.current_mode == CognitiveMode.DELIBERATE
        assert result.response_modifiers["intent_type"] == "TASK"
        assert result.response_modifiers["request_mood"] == "directive"

    asyncio.run(scenario())


def test_live_router_resolves_contextual_go_ahead_as_action():
    async def scenario():
        phase = CognitiveRoutingPhase(_RouteContainer())
        phase._spawn_parallel_branch = lambda *_args, **_kwargs: None
        state = _state(
            ("user", "Could you inspect the runtime report and save the findings?"),
            ("assistant", "I can do that."),
            ("user", "Go ahead."),
        )

        result = await phase.execute(state)

        assert result.response_modifiers["intent_type"] == "TASK"
        assert result.response_modifiers["request_mood_reasons"] == [
            "contextual_action_followup"
        ]

    asyncio.run(scenario())
