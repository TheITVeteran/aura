import json

import pytest

from core.skills.desktop_task import DesktopTaskSkill


@pytest.mark.asyncio
async def test_desktop_task_executes_bounded_steps_through_capability_engine(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return {"ok": True, "summary": f"{params['action']} ok"}

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Create a note from the clipboard.",
            "steps": [
                {"action": "open_app", "target": "Notes", "reason": "Open Notes"},
                {"action": "set_clipboard", "target": "hello", "reason": "Copy text"},
                {
                    "action": "write_text_file",
                    "target": {"path": "Aura Proof/receipt.txt", "content": "done"},
                    "reason": "Write receipt",
                },
            ],
        },
        {"origin": "user"},
    )

    assert result["ok"] is True
    assert result["steps_completed"] == 3
    assert [call[0] for call in calls] == ["computer_use", "computer_use", "computer_use"]
    assert calls[2][1]["action"] == "write_text_file"
    assert json.loads(calls[2][1]["target"])["content"] == "done"
    assert calls[0][2]["route"] == "desktop_task.computer_use"
    assert calls[0][2]["user_requested_action"] is True


@pytest.mark.asyncio
async def test_desktop_task_stops_on_first_failed_step(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return {"ok": params["action"] != "click", "error": "click failed"}

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Click and type.",
            "steps": [
                {"action": "click", "x": 10, "y": 10},
                {"action": "type", "target": "should not run"},
            ],
        },
        {"origin": "user"},
    )

    assert result["ok"] is False
    assert result["steps_completed"] == 0
    assert len(calls) == 1
    assert result["failures"][0]["action"] == "click"


@pytest.mark.asyncio
async def test_desktop_task_derives_general_plan_from_desktop_objective(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return {"ok": True, "summary": f"{params['action']} ok"}

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": (
                "Open Notes, write a timestamped summary, save it as a PDF in a new "
                "folder titled Aura's Journal, and search for an image of a robot."
            ),
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_task_document_body": "Aura summary body from CognitiveEngine.",
        },
    )

    assert result["ok"] is True
    assert result["steps_requested"] >= 5
    actions = [call[1]["action"] for call in calls]
    assert actions[:2] == ["create_folder", "open_app"]
    assert "open_url" in actions
    assert "write_text_file" in actions
    assert "render_text_pdf" in actions
    folder_payload = json.loads(calls[0][1]["target"])
    assert folder_payload["path"] == "Aura's Journal"
    pdf_payload = json.loads(calls[-1][1]["target"])
    assert pdf_payload["path"].endswith(".pdf")
    assert "Aura summary body from CognitiveEngine." in pdf_payload["body"]
    assert calls[0][2]["route"] == "desktop_task.computer_use"
    assert calls[0][2]["origin"] == "desktop_ui"


def test_desktop_task_discovered_and_ranked_for_chained_desktop_prompt(monkeypatch):
    from core.capability_engine import CapabilityEngine, SkillMetadata

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {
        "desktop_task": SkillMetadata(
            name="desktop_task",
            description="desktop task",
            metabolic_cost=2,
            effect_scope="foreground_desktop_control",
            trigger_patterns=[
                r"use (?:my )?computer",
                r"(?:multi[- ]?step|chained|chain) .* (?:desktop|computer|app|screen)",
            ],
        ),
        "computer_use": SkillMetadata(
            name="computer_use",
            description="computer use",
            metabolic_cost=2,
            effect_scope="foreground_desktop_control",
            trigger_patterns=[r"click (?:on|the)"],
        ),
    }
    engine.active_skills = set(engine.skills)
    engine.skill_states = {name: "READY" for name in engine.skills}
    engine.skill_last_errors = {}
    engine.resolve_skill_name = lambda name: name
    engine._explicitly_deactivated_skills = set()

    prompt = "Use my computer to open Calculator, copy a result, paste it in Notes, export a PDF, and move it."

    assert "desktop_task" in engine.detect_intent(prompt)
    assert engine._rank_tool_candidates(objective=prompt, max_tools=3)[0] == "desktop_task"
    assert engine.get_tool_catalog(include_inactive=True)[0]["risk_class"] == "critical"
