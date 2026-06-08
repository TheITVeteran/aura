from __future__ import annotations

from types import SimpleNamespace

import pytest


INVENTORY_PROMPT = (
    "What tools can you hypothetically use externally on my computer? "
    "Explain the categories, name one realistic multi-step scenario, "
    "and do not actually open apps or execute tools yet."
)


def test_capability_inventory_prompt_is_not_desktop_execution() -> None:
    from core.phases.action_intent import detect_action_intent
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective
    from core.runtime.skill_task_bridge import (
        looks_like_capability_inventory_dialogue_request,
        looks_like_multi_step_skill_request,
    )
    from core.runtime.turn_analysis import analyze_turn

    assert looks_like_capability_inventory_dialogue_request(INVENTORY_PROMPT) is True
    assert looks_like_desktop_objective(INVENTORY_PROMPT) is False
    assert looks_like_multi_step_skill_request(
        INVENTORY_PROMPT,
        ["desktop_task", "computer_use"],
    ) is False
    action_intent = detect_action_intent(INVENTORY_PROMPT)
    assert action_intent.should_execute is False
    assert analyze_turn(INVENTORY_PROMPT, matched_skills=["desktop_task"]).intent_type == "CHAT"


def test_desktop_command_with_negative_constraint_still_executes() -> None:
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    prompt = "Open Notes and write a timestamped sentence, but do not use Chrome."

    assert looks_like_desktop_objective(prompt) is True


@pytest.mark.asyncio
async def test_cognitive_routing_keeps_inventory_prompt_off_skill_fastpath(monkeypatch) -> None:
    from core.phases.cognitive_routing import CognitiveRoutingPhase
    from core.state.aura_state import AuraState

    detect_calls = 0

    class Capability:
        def detect_intent(self, *_args, **_kwargs):
            nonlocal detect_calls
            detect_calls += 1
            return ["desktop_task", "computer_use"]

    container = SimpleNamespace(
        get=lambda name, default=None: Capability() if name == "capability_engine" else default
    )
    phase = CognitiveRoutingPhase(container)
    state = AuraState()
    state.cognition.current_objective = INVENTORY_PROMPT
    state.cognition.current_origin = "desktop_ui"

    routed = await phase.execute(state, INVENTORY_PROMPT)

    assert detect_calls == 0
    assert routed.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in routed.response_modifiers


def test_capability_inventory_reply_repairs_false_tool_limitation(monkeypatch: pytest.MonkeyPatch) -> None:
    from interface.routes import chat as chat_routes

    class FakeCapabilityEngine:
        def get_tool_catalog(self, *, include_inactive: bool = True) -> list[dict]:
            return [
                {
                    "name": "computer_use",
                    "available": True,
                    "description": "Control desktop apps with governed screen, mouse, and keyboard actions.",
                    "route_class": "desktop",
                    "risk_class": "critical",
                    "effect_scope": "external_io",
                },
                {
                    "name": "web_search",
                    "available": True,
                    "description": "Search and inspect live web sources.",
                    "route_class": "external_io",
                    "risk_class": "medium",
                    "effect_scope": "external_io",
                },
                {
                    "name": "file_operation",
                    "available": True,
                    "description": "Read and write local files and documents.",
                    "route_class": "stateful",
                    "risk_class": "medium",
                    "effect_scope": "file_system",
                },
                {
                    "name": "sovereign_terminal",
                    "available": True,
                    "description": "Run governed terminal commands.",
                    "route_class": "subprocess",
                    "risk_class": "critical",
                    "effect_scope": "subprocess",
                },
                {
                    "name": "clock",
                    "available": True,
                    "description": "Return the current local time.",
                    "route_class": "read_only",
                    "risk_class": "low",
                    "effect_scope": "read_only",
                },
            ]

    def fake_get(name: str, default=None):
        if name == "capability_engine":
            return FakeCapabilityEngine()
        return default

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", fake_get)
    monkeypatch.setattr(chat_routes, "_runtime_tool_governance_available", lambda: True)

    assert chat_routes._capability_inventory_reply_is_inadequate(
        INVENTORY_PROMPT,
        "I can't actually open apps or execute tools on your computer.",
    )
    repaired = chat_routes._build_grounded_capability_inventory_reply(INVENTORY_PROMPT)

    lowered = repaired.lower()
    assert "computer_use" in repaired
    assert "web_search" in repaired
    assert "file_operation" in repaired
    assert "sovereign_terminal" in repaired
    assert "specialized governed skills (clock)" in lowered
    assert "files, documents, and workspace operations (file_operation, clock)" not in lowered
    assert "will/authority" in lowered
    assert "not opening apps" in lowered
    assert "i can't" not in lowered


@pytest.mark.asyncio
async def test_owner_name_recall_repairs_thin_desktop_cognitive_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_owner_session_is_verified", lambda **_kwargs: True)
    monkeypatch.setattr(chat_routes, "_resolve_primary_operator_name", lambda: "Bryan")

    repaired, stale, same_diff, off_topic, reason, did_repair = (
        await chat_routes._repair_final_degraded_reply(
            "Do you know my name?",
            "Yes.",
            stale=False,
            same_diff=False,
            off_topic=False,
        )
    )

    assert did_repair is True
    assert stale is False
    assert same_diff is False
    assert off_topic is False
    assert reason == ""
    assert "Bryan" in repaired
    assert "verified owner session" in repaired


def test_owner_name_recall_does_not_disclose_without_owner_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "_owner_session_is_verified", lambda **_kwargs: False)

    reply = chat_routes._build_owner_name_recall_reply("What's my name?")

    assert reply
    assert "owner-verified" in reply
    assert "Bryan" not in reply
