import json

import pytest

from core.skills.desktop_task import DesktopTaskSkill


def _fake_computer_use_result(params):
    action = params["action"]
    target = params.get("target") or ""
    try:
        payload = json.loads(target)
    except (TypeError, json.JSONDecodeError):
        payload = {}
    if action == "create_folder":
        return {"ok": True, "action": action, "path": payload.get("path", "Aura Proof")}
    if action == "open_app":
        return {"ok": True, "opened": target, "returncode": 0}
    if action == "open_url":
        return {"ok": True, "action": action, "url": target}
    if action == "write_text_file":
        content = str(payload.get("content") or "")
        return {
            "ok": True,
            "action": action,
            "path": payload.get("path", "Aura Proof/receipt.txt"),
            "bytes": len(content.encode("utf-8")),
        }
    if action == "render_text_pdf":
        body = str(payload.get("body") or "")
        return {
            "ok": True,
            "action": action,
            "path": payload.get("path", "Aura Proof/receipt.pdf"),
            "bytes": max(128, len(body.encode("utf-8"))),
            "pages": 1,
            "chars": len(body),
        }
    if action == "move_file":
        return {
            "ok": True,
            "action": action,
            "destination": payload.get("destination", "Aura Proof/moved.txt"),
            "bytes": 12,
        }
    if action == "set_clipboard":
        return {"ok": True, "action": action, "chars": len(str(target))}
    if action == "hotkey":
        return {"ok": True, "action": action, "hotkey": target}
    if action == "wait":
        return {"ok": True, "action": action, "seconds": float(target or 1.0)}
    if action == "type":
        return {"ok": True, "action": action, "typed": str(target)[:50], "verification": "state shifted"}
    return {"ok": True, "action": action, "summary": f"{action} ok"}


@pytest.mark.asyncio
async def test_desktop_task_executes_bounded_steps_through_capability_engine(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

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
async def test_desktop_task_rejects_child_ok_without_required_effect_evidence(monkeypatch):
    from core.container import ServiceContainer

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            return {"ok": True, "summary": "claimed success without a file receipt"}

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Write a durable receipt file.",
            "steps": [
                {
                    "action": "write_text_file",
                    "target": {"path": "Aura Proof/receipt.txt", "content": "done"},
                    "reason": "Write receipt",
                }
            ],
        },
        {"origin": "user"},
    )

    assert result["ok"] is False
    assert result["steps_completed"] == 0
    assert result["failures"][0]["effect_verified"] is False
    assert result["failures"][0]["effect_evidence"] == "missing written file path"


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
            return _fake_computer_use_result(params)

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
    open_urls = [call[1]["target"] for call in calls if call[1]["action"] == "open_url"]
    assert any(url.startswith("https://duckduckgo.com/?q=") for url in open_urls)
    assert any("iax=images" in url for url in open_urls)
    pdf_payload = json.loads(calls[-1][1]["target"])
    assert pdf_payload["path"].endswith(".pdf")
    assert "Aura summary body from CognitiveEngine." in pdf_payload["body"]
    assert "Image search opened:" in pdf_payload["body"]
    assert "No local image insertion is claimed" in pdf_payload["body"]
    assert calls[0][2]["route"] == "desktop_task.computer_use"
    assert calls[0][2]["origin"] == "desktop_ui"


@pytest.mark.asyncio
async def test_desktop_task_uses_cognitive_engine_structured_plan_before_fallback(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    cognitive_plan = {
        "steps": [
            {"action": "open_app", "target": "TextEdit", "reason": "Use the requested writing app."},
            {
                "action": "write_text_file",
                "target": {"path": "Aura Drafts/general_plan.txt", "content": "planned body"},
            },
        ]
    }

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Use the desktop to create an arbitrary local draft artifact.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "cognitive_reply": f"Plan:\n```json\n{json.dumps(cognitive_plan)}\n```",
        },
    )

    assert result["ok"] is True
    assert result["steps_requested"] == 2
    assert [call[1]["action"] for call in calls] == ["open_app", "write_text_file"]
    assert calls[0][1]["target"] == "TextEdit"
    assert json.loads(calls[1][1]["target"])["content"] == "planned body"


def test_desktop_task_extracts_generic_named_app_mentions():
    assert DesktopTaskSkill._generic_open_app_mentions("Open TextEdit application and create a draft.") == [
        "TextEdit"
    ]


def test_desktop_task_does_not_invent_aura_journal_folder_name():
    folder = DesktopTaskSkill._extract_folder_name("Write a private journal entry.")

    assert folder != "Aura's Journal"
    assert folder.startswith("Aura Desktop Task ")


@pytest.mark.asyncio
async def test_desktop_task_derives_generic_web_document_plan_without_demo_shortcuts(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Open a tab for Google Docs and start typing a coherent essay about climate adaptation.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_task_document_body": "Essay body from CognitiveEngine.",
        },
    )

    assert result["ok"] is True
    actions = [call[1]["action"] for call in calls]
    assert actions == ["open_url", "set_clipboard", "wait", "hotkey"]
    assert "create_folder" not in actions
    assert "write_text_file" not in actions
    assert "render_text_pdf" not in actions
    open_urls = [call[1]["target"] for call in calls if call[1]["action"] == "open_url"]
    assert open_urls == ["https://docs.google.com/document/u/0/create"]
    assert not any("duckduckgo.com" in url for url in open_urls)
    clipboard_payload = calls[1][1]["target"]
    assert "Essay body from CognitiveEngine." in clipboard_payload
    assert calls[-1][1]["target"] == "command+v"


@pytest.mark.asyncio
async def test_desktop_task_escalates_unrepresented_desktop_workflow_to_os_automation(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if skill_name == "os_automation":
                return {
                    "ok": True,
                    "result": "arranged visible browser window",
                    "receipt_id": "receipt-os-1",
                    "adapter": "applescript",
                }
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": (
                "Use my computer to resize the current browser window and arrange it "
                "on the left side of the screen."
            ),
            "steps": [],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is True
    assert result["planner"] == "os_automation_fallback"
    assert result["steps_requested"] == 1
    assert result["steps_completed"] == 1
    assert calls == [
        (
            "os_automation",
            {
                "goal": (
                    "Use my computer to resize the current browser window and arrange it "
                    "on the left side of the screen."
                ),
                "script_type": "applescript",
                "execute": True,
            },
            {
                "origin": "desktop_ui",
                "route": "desktop_task.os_automation",
                "objective": (
                    "Use my computer to resize the current browser window and arrange it "
                    "on the left side of the screen."
                ),
                "foreground_request": True,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "desktop_task_reason": (
                    "Primitive desktop actions were not sufficient for this objective; "
                    "escalating to governed OS automation."
                ),
                "desktop_task_expect": "OS automation receipt proves the visible desktop action ran.",
            },
        )
    ]
    receipt = result["receipts"][0]
    assert receipt["action"] == "os_automation"
    assert receipt["effect_verified"] is True
    assert receipt["effect_evidence"] == "receipt_id=receipt-os-1"


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
