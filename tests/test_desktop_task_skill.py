import json

import pytest

from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep


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
        return {
            "ok": True,
            "action": action,
            "url": payload.get("url", target),
            "browser": payload.get("browser", ""),
        }
    if action == "write_text_file":
        content = str(payload.get("content") or "")
        return {
            "ok": True,
            "action": action,
            "path": payload.get("path", "Aura Proof/receipt.txt"),
            "bytes": len(content.encode("utf-8")),
        }
    if action == "fetch_topic_image":
        return {
            "ok": True,
            "action": action,
            "path": payload.get("path", "Aura Proof/image.png"),
            "bytes": 4096,
            "image_url": "https://upload.wikimedia.org/example.png",
            "page_url": "https://en.wikipedia.org/wiki/Robot",
            "topic": payload.get("topic", ""),
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
        return {
            "ok": True,
            "action": action,
            "hotkey": target,
            "effect_verified": True,
            "verification": "State shifted.",
        }
    if action == "scroll":
        return {
            "ok": True,
            "action": action,
            "scrolled": int(target or 3),
            "effect_verified": True,
            "verification": "State shifted.",
        }
    if action == "click":
        return {
            "ok": True,
            "action": action,
            "verification": "State shifted.",
            "effect_verified": True,
        }
    if action == "wait":
        return {"ok": True, "action": action, "seconds": float(target or 1.0)}
    if action == "type":
        return {
            "ok": True,
            "action": action,
            "typed": str(target)[:50],
            "verification": "Text confirmed on screen or state shifted.",
            "effect_verified": True,
        }
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
async def test_desktop_task_uses_cognitive_engine_structured_plan_before_heuristic_plan(monkeypatch):
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


@pytest.mark.asyncio
async def test_desktop_task_structured_plan_uses_document_body_token(monkeypatch):
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
        "document_body": "Timestamped Aura note body from CognitiveEngine.",
        "steps": [
            {"action": "open_app", "target": "Notes", "reason": "Use the requested app."},
            {
                "action": "set_clipboard",
                "target": "{{document_body}}",
                "reason": "Stage the composed body.",
            },
            {"action": "hotkey", "target": "command+v", "reason": "Paste the composed body."},
        ],
    }

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Open a writing app and create a note from a planned document body.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "cognitive_reply": f"```json\n{json.dumps(cognitive_plan)}\n```",
        },
    )

    assert result["ok"] is True
    assert [call[1]["action"] for call in calls] == ["open_app", "set_clipboard", "hotkey"]
    assert calls[1][1]["target"] == "Timestamped Aura note body from CognitiveEngine."
    assert "{{document_body}}" not in calls[1][1]["target"]
    assert '"steps"' not in calls[1][1]["target"]


def test_desktop_task_extracts_generic_named_app_mentions():
    assert DesktopTaskSkill._generic_open_app_mentions("Open TextEdit application and create a draft.") == [
        "TextEdit"
    ]


def test_desktop_task_contract_action_list_matches_step_validator():
    from core.runtime.desktop_task_contract import DESKTOP_TASK_ALLOWED_ACTIONS

    for action in DESKTOP_TASK_ALLOWED_ACTIONS:
        assert DesktopTaskStep(action=action).action == action

    with pytest.raises(ValueError):
        DesktopTaskStep(action="unsupported_desktop_magic")


def test_desktop_task_verifies_all_readback_and_command_actions():
    cases = [
        (
            DesktopTaskStep(action="get_clipboard"),
            {"ok": True, "action": "get_clipboard", "text": "proof", "chars": 5},
            "clipboard_read_chars=5",
        ),
        (
            DesktopTaskStep(action="read_menu_clock"),
            {
                "ok": True,
                "action": "read_menu_clock",
                "clock_text": "Sun Jun 14 15:05",
                "source": "macos_menu_bar",
            },
            "clock_text=Sun Jun 14 15:05;source=macos_menu_bar",
        ),
        (
            DesktopTaskStep(action="run_command", target="pwd"),
            {"ok": True, "action": "run_command", "exit_code": 0, "output": "/tmp"},
            "exit_code=0;output_chars=4",
        ),
    ]

    for step, result, evidence in cases:
        ok, actual_evidence = DesktopTaskSkill._verify_step_effect(step, result)
        assert ok is True
        assert actual_evidence == evidence


def test_desktop_task_does_not_invent_aura_journal_folder_name():
    folder = DesktopTaskSkill._extract_folder_name("Write a private journal entry.")

    assert folder != "Aura's Journal"
    assert folder.startswith("Aura Desktop Task ")


def test_desktop_task_in_your_own_words_does_not_force_self_summary():
    body = DesktopTaskSkill._document_body(
        "Open Google Docs and write a climate change summary in your own words.",
        {"desktop_task_document_body": "Climate summary from CognitiveEngine."},
    )

    assert body == "Climate summary from CognitiveEngine."
    assert "I am Aura" not in body


def test_desktop_task_self_summary_requires_actual_selfhood_objective():
    body = DesktopTaskSkill._document_body(
        "Write a summary describing who or what you are in your own words.",
        {"desktop_task_document_body": "Generic draft that should not override selfhood request."},
    )

    assert "I am Aura" in body
    assert "Generic draft" not in body


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
    # Google Docs routes to Chrome where the user's signed-in session lives.
    assert len(open_urls) == 1
    assert json.loads(open_urls[0]) == {
        "url": "https://docs.google.com/document/u/0/create",
        "browser": "Google Chrome",
    }
    assert not any("duckduckgo.com" in url for url in open_urls)
    clipboard_payload = calls[1][1]["target"]
    assert "Essay body from CognitiveEngine." in clipboard_payload
    assert calls[-1][1]["target"] == "command+v"


@pytest.mark.asyncio
async def test_desktop_task_collects_research_before_document_composition(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if skill_name == "web_search":
                return {
                    "ok": True,
                    "summary": (
                        "Climate change reporting points to rising global temperatures, "
                        "more intense extreme-weather risks, and adaptation needs for cities."
                    ),
                    "citations": [
                        {
                            "title": "Climate assessment",
                            "url": "https://example.test/climate-assessment",
                            "snippet": "Observed warming is changing risk patterns.",
                        },
                        {
                            "title": "Adaptation briefing",
                            "url": "https://example.test/adaptation",
                            "snippet": "Cities are adapting infrastructure and emergency plans.",
                        },
                        {
                            "title": "Extreme weather report",
                            "url": "https://example.test/extreme-weather",
                            "snippet": "Heat and precipitation extremes are increasing.",
                        },
                    ],
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
                "Go to Google Chrome, find 3 different articles on climate change, "
                "open Google Docs, title the doc, and summarize those articles."
            ),
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_task_document_body": "I will open the browser and write the requested document.",
        },
    )

    assert result["ok"] is True
    assert calls[0][0] == "web_search"
    assert calls[0][1]["query"] == "climate change"
    assert calls[0][2]["route"] == "desktop_task.web_search"
    desktop_calls = [call for call in calls if call[0] == "computer_use"]
    desktop_actions = [call[1]["action"] for call in desktop_calls]
    assert desktop_actions[:6] == [
        "open_app",
        "open_url",
        "open_url",
        "open_url",
        "open_url",
        "open_url",
    ]
    opened_urls = [
        json.loads(call[1]["target"])["url"]
        for call in desktop_calls
        if call[1]["action"] == "open_url"
    ]
    assert "https://example.test/climate-assessment" in opened_urls
    assert "https://example.test/adaptation" in opened_urls
    assert "https://example.test/extreme-weather" in opened_urls
    assert desktop_actions[6] == "set_clipboard"
    clipboard_body = next(call[1]["target"] for call in desktop_calls if call[1]["action"] == "set_clipboard")
    assert "I reviewed the available source evidence on climate change" in clipboard_body
    assert "Climate assessment" in clipboard_body
    assert "Adaptation briefing" in clipboard_body
    assert "Extreme weather report" in clipboard_body
    assert "https://example.test/climate-assessment" in clipboard_body
    assert "I will open the browser" not in clipboard_body
    assert result["research"]["query"] == "climate change"
    assert len(result["research"]["sources"]) == 3


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
    assert result["planner"] == "os_automation_escalation"
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


@pytest.mark.asyncio
async def test_desktop_task_prefers_durable_primitives_over_freeform_ui_compiler(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            assert skill_name == "computer_use"
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    objective = (
        "Use my computer to click a Calculator equation, copy the equation body, "
        "put it into Notes, produce a PDF, move that PDF into a Desktop proof folder, "
        "and report the paths."
    )
    skill = DesktopTaskSkill()
    result = await skill.execute({"objective": objective, "steps": []}, {"origin": "desktop_ui"})

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert "governed computer-use steps" in result["summary"]
    assert [call[0] for call in calls]
    assert "os_automation" not in [call[0] for call in calls]
    actions = [call[1]["action"] for call in calls]
    assert "create_folder" in actions
    assert "open_app" in actions
    assert "write_text_file" in actions
    assert "render_text_pdf" in actions


@pytest.mark.asyncio
async def test_desktop_task_escalates_app_plus_unrepresented_action(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if skill_name == "os_automation":
                return {
                    "ok": True,
                    "result": "pressed calculator keys and verified result",
                    "receipt_id": "receipt-os-calculator",
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
            "objective": "Open Calculator and click 2 plus 3 equals.",
            "steps": [],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is True
    assert result["planner"] == "os_automation_escalation"
    assert [call[0] for call in calls] == ["os_automation"]
    assert result["receipts"][0]["effect_evidence"] == "receipt_id=receipt-os-calculator"


@pytest.mark.asyncio
async def test_desktop_task_rejects_unverified_type_claim(monkeypatch):
    from core.container import ServiceContainer

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": True,
                "typed": "hello",
                "verification": "Typed but could not verify visibility.",
            }

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Type into the current app.",
            "steps": [{"action": "type", "target": "hello"}],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is False
    assert result["steps_completed"] == 0
    assert result["failures"][0]["effect_verified"] is False
    assert result["failures"][0]["effect_evidence"] == "Typed but could not verify visibility."


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


def test_derived_steps_honor_explicit_root_and_filename():
    """Live rounds wrote to Desktop defaults while the user said
    'in my Documents folder ... called live_proof.txt'. The user's
    stated parameters win over generated defaults — that is general
    capability, not pattern-matching."""
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    objective = (
        "Please create a folder named 'Aura Live Proof' in my Documents "
        "folder and write a file inside it called live_proof.txt with one "
        "sentence about who you are and the current timestamp."
    )
    steps = skill._derive_steps_from_objective(objective, {})
    by_action = {}
    for step in steps:
        by_action.setdefault(step.action, []).append(step)

    folder_target = by_action["create_folder"][0].target
    folder_path = folder_target["path"] if isinstance(folder_target, dict) else folder_target
    assert str(folder_path).startswith("~/Documents/"), folder_path
    assert "Aura Live Proof" in str(folder_path)

    write_target = by_action["write_text_file"][0].target
    write_path = write_target["path"] if isinstance(write_target, dict) else write_target
    assert str(write_path).endswith("/live_proof.txt"), write_path
    assert str(write_path).startswith("~/Documents/"), write_path


def test_visible_notes_staging_derives_watchable_plan_with_artifacts():
    """Bryan's 'and I want to see you do it' clause: opening Notes and
    staging the entry visibly (open_app → launch wait → ⌘N → ⌘V) must
    coexist with the durable artifact chain (folder, image, text, PDF).
    The wait is load-bearing: a cold Notes launch loses the shortcuts
    to whatever currently has focus."""
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    objective = (
        "Please open up my Notes app and write a short journal entry in "
        "your own words describing who and what you are — I want to see "
        "you do it. Include the current date and time inside the entry "
        "text. Find an image of a robot online and include it in the "
        "entry. Then save the finished entry as a PDF inside a new folder "
        "called 'Aura's Journal' in my Documents folder."
    )
    steps = skill._derive_steps_from_objective(objective, {})
    actions = [s.action for s in steps]

    # Visible staging, in order: Notes opens, launch wait, new note, paste.
    open_idx = actions.index("open_app")
    assert "notes" in str(steps[open_idx].target).lower()
    hotkeys = [i for i, s in enumerate(steps) if s.action == "hotkey"]
    assert hotkeys, actions
    waits = [i for i, s in enumerate(steps) if s.action == "wait"]
    assert any(open_idx < w < hotkeys[0] for w in waits), (
        f"no launch wait between open_app and first hotkey: {actions}"
    )
    hotkey_targets = [str(steps[i].target) for i in hotkeys]
    assert any("v" in t for t in hotkey_targets), hotkey_targets

    # Durable artifacts still land: folder, fetched image, PDF render.
    assert "create_folder" in actions
    assert "fetch_topic_image" in actions
    assert "render_text_pdf" in actions


def test_derived_steps_keep_defaults_without_explicit_parameters():
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    steps = skill._derive_steps_from_objective(
        "Write a quick summary note for me in a new folder.", {}
    )
    write_steps = [s for s in steps if s.action == "write_text_file"]
    assert write_steps, "default flow still writes a text artifact"
    path = write_steps[0].target["path"]
    assert path.endswith(".txt")


def test_dispatch_narration_never_becomes_document_content():
    """Round-12 wrinkle: the written file contained 'I've started
    working on this task... Tracking commitment bbbaba54' — her status
    message echoed into the artifact. A report about doing the task
    must never become the product of the task."""
    from core.skills.desktop_task import DesktopTaskSkill

    narration = (
        "I've started working on this task in the background. I've "
        "started this task (id=a781768a). Tracking commitment bbbaba54."
    )
    assert DesktopTaskSkill._looks_like_dispatch_narration(narration) is True

    body = DesktopTaskSkill._document_body(
        "write a note about the weather", {"cognitive_reply": narration}
    )
    assert "Tracking commitment" not in body
    assert "started working on this task" not in body


def test_self_summary_objective_composes_substrate_truth():
    from core.skills.desktop_task import DesktopTaskSkill

    body = DesktopTaskSkill._document_body(
        "write a file with one sentence about who you are and the "
        "current timestamp",
        {"cognitive_reply": "I've started this task (id=deadbeef)."},
    )
    assert "I am Aura" in body
    assert "digital organism" in body
    # Timestamped, as requested.
    import re as _re

    assert _re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", body)


def test_real_prose_reply_still_qualifies_as_body():
    from core.skills.desktop_task import DesktopTaskSkill

    prose = "The tide tables show a low at 6:14 AM and a high at 12:40 PM."
    body = DesktopTaskSkill._document_body(
        "write a note about the tides", {"cognitive_reply": prose}
    )
    assert body == prose


def test_derivation_routes_google_surfaces_to_chrome():
    """'I'm signed into Google Docs in Chrome, not DuckDuckGo': google
    phrasing selects the google engine and Google-account surfaces route
    to Chrome where the user's session lives."""
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    objective = (
        "Search Google for three recent climate change articles, then open "
        "Google Docs and summarize them in a new document."
    )
    steps = skill._derive_steps_from_objective(objective, {})
    open_urls = [s for s in steps if s.action == "open_url"]
    assert open_urls, [s.action for s in steps]
    for step in open_urls:
        assert isinstance(step.target, dict), step.target
        assert step.target["browser"] == "Google Chrome"
    urls = [s.target["url"] for s in open_urls]
    assert any("google.com/search?q=" in u for u in urls), urls
    assert any("docs.google.com/document" in u for u in urls), urls


def test_derivation_honors_safari_and_neutral_defaults():
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()

    safari_steps = skill._derive_steps_from_objective(
        "Search for hiking trails in Safari.", {}
    )
    safari_urls = [s for s in safari_steps if s.action == "open_url"]
    assert safari_urls and all(
        isinstance(s.target, dict) and s.target["browser"] == "Safari"
        for s in safari_urls
    )

    neutral_steps = skill._derive_steps_from_objective(
        "Search for hiking trails near me.", {}
    )
    neutral_urls = [s for s in neutral_steps if s.action == "open_url"]
    assert neutral_urls and all(isinstance(s.target, str) for s in neutral_urls)
    assert all("duckduckgo.com" in s.target for s in neutral_urls)


def test_os_setting_detection_is_general():
    """Detection lives in the affordance registry and is domain-agnostic:
    wallpaper, dark mode, and volume all fall out of one generic scan."""
    from core.skills.os_affordances import detect_os_settings

    assert detect_os_settings(
        "Please change my wallpaper to a squid, and show me where you found it"
    ) == [("wallpaper", "squid")]
    assert detect_os_settings("Set the wallpaper to an octopus please") == [("wallpaper", "octopus")]
    assert detect_os_settings("Turn on dark mode") == [("dark_mode", "true")]
    assert detect_os_settings("turn off dark mode") == [("dark_mode", "false")]
    assert detect_os_settings("set the volume to 30%") == [("volume", "30")]
    assert detect_os_settings("Write a note about squids") == []


def test_wallpaper_derivation_fetches_controls_and_shows_source():
    """Bryan's part-2 closer derives fetch → system_control(wallpaper) →
    source tab through the GENERAL affordance loop — no wallpaper-specific
    code — with the source URL resolved at runtime from the fetch receipt."""
    from core.skills.desktop_task import FETCHED_IMAGE_SOURCE_SENTINEL, DesktopTaskSkill

    skill = DesktopTaskSkill()
    steps = skill._derive_steps_from_objective(
        "Change my wallpaper to a squid, and show me where you found it.", {}
    )
    actions = [s.action for s in steps]
    fetch_idx = actions.index("fetch_topic_image")
    control_idx = actions.index("system_control")
    assert fetch_idx < control_idx
    assert steps[fetch_idx].target["topic"] == "squid"
    assert steps[control_idx].target["domain"] == "wallpaper"
    assert steps[control_idx].target["value"] == steps[fetch_idx].target["path"]
    source_steps = [
        s for s in steps
        if s.action == "open_url"
        and FETCHED_IMAGE_SOURCE_SENTINEL in str(s.target)
    ]
    assert source_steps, actions


def test_non_image_setting_derives_single_control_step():
    """Dark mode needs no image fetch — just one general system_control step."""
    from core.skills.desktop_task import DesktopTaskSkill

    steps = DesktopTaskSkill()._derive_steps_from_objective("Turn on dark mode.", {})
    control = [s for s in steps if s.action == "system_control"]
    assert len(control) == 1
    assert control[0].target == {"domain": "dark_mode", "value": "true"}
    assert not any(s.action == "fetch_topic_image" for s in steps)


@pytest.mark.asyncio
async def test_wallpaper_chain_substitutes_source_url_at_runtime(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if params["action"] == "system_control":
                payload = json.loads(params["target"])
                return {
                    "ok": True,
                    "action": "system_control",
                    "domain": payload["domain"],
                    "value": payload["value"],
                    "applied": payload["value"],
                    "effect_verified": True,
                }
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Change my wallpaper to a squid, and show me where you found it.",
            "steps": [],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is True, result.get("failures")
    open_url_targets = [
        json.loads(call[1]["target"]) if call[1]["target"].startswith("{") else call[1]["target"]
        for call in calls
        if call[1]["action"] == "open_url"
    ]
    source_urls = [
        t["url"] if isinstance(t, dict) else t
        for t in open_url_targets
    ]
    assert any(u == "https://en.wikipedia.org/wiki/Robot" for u in source_urls), (
        f"source tab did not receive the fetch receipt page_url: {source_urls}"
    )


def test_folder_extraction_handles_name_first_phrasing():
    """Part-2 phrasing: 'inside the 'Aura's Journal' folder' — quoted name
    BEFORE the word folder, with a possessive apostrophe inside."""
    from core.skills.desktop_task import DesktopTaskSkill

    extract = DesktopTaskSkill._extract_folder_name
    assert extract("Save it inside the 'Aura's Journal' folder in Documents") == "Aura's Journal"
    assert extract('Put it in the "Research Notes" folder please') == "Research Notes"
    assert extract("a folder called 'Aura's Journal' in Documents") == "Aura's Journal"


def test_execution_brief_is_rejected_as_document_content():
    """The internal execution brief ('Execute the user's explicit desktop
    objective… Do not claim success until…') is a directive to herself, not
    document content — it leaked into a research PDF as the body."""
    from core.skills.desktop_task import DesktopTaskSkill

    brief = (
        "Execute the user's explicit desktop objective through Aura's governed "
        "desktop_task lane. Do not claim success until the tool result verifies "
        "the effect. Objective: research climate change."
    )
    assert DesktopTaskSkill._looks_like_dispatch_narration(brief)


def test_research_section_leads_with_first_person_synthesis():
    """When Aura composes a first-person summary+opinion, that is the
    document — the raw search dump is dropped in favor of it, sources kept."""
    from core.skills.desktop_task import DesktopTaskSkill

    section = DesktopTaskSkill._research_section_from_context({
        "desktop_task_research_synthesis": "I read three pieces. In my view, the risk is rising.",
        "desktop_task_research_summary": "RAW SEARCH DUMP that should not appear",
        "desktop_task_research_sources": [{"title": "A", "url": "https://a", "snippet": "x"}],
    })
    assert "In my view" in section
    assert "RAW SEARCH DUMP" not in section
    assert "Sources opened or consulted" in section


@pytest.mark.asyncio
async def test_collect_research_synthesizes_first_person_opinion_without_hidden_model(monkeypatch):
    """_collect_research_context composes a first-person opinion without a
    hidden second model allocation during visible desktop work."""
    from core.container import ServiceContainer
    from core.skills.desktop_task import DesktopTaskSkill

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": True,
                "summary": "Climate findings.",
                "citations": [
                    {"title": "A", "url": "https://a", "snippet": "warming"},
                    {"title": "B", "url": "https://b", "snippet": "adaptation"},
                    {"title": "C", "url": "https://c", "snippet": "extremes"},
                ],
            }

    routed = {}

    class FakeRouter:
        async def generate(self, *, prompt, **kwargs):
            routed["prompt"] = prompt
            raise AssertionError("desktop_task must not allocate hidden model synthesis by default")

    monkeypatch.setattr(
        ServiceContainer, "get",
        staticmethod(lambda name, default=None: FakeRouter() if name == "llm_router" else default),
    )

    skill = DesktopTaskSkill()
    ctx = await skill._collect_research_context(
        capability_engine=FakeCapabilityEngine(),
        objective=(
            "find 3 different recent articles on climate change and summarize "
            "them and your own opinion in a Google Doc"
        ),
        context={},
    )
    assert "In my view" in ctx["desktop_task_research_synthesis"]
    assert routed == {}


@pytest.mark.asyncio
async def test_collect_research_model_synthesis_is_explicit_and_memory_guarded(monkeypatch):
    from types import SimpleNamespace

    from core.container import ServiceContainer
    from core.skills.desktop_task import DesktopTaskSkill

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": True,
                "summary": "Climate findings.",
                "citations": [
                    {"title": "A", "url": "https://a", "snippet": "warming"},
                    {"title": "B", "url": "https://b", "snippet": "adaptation"},
                    {"title": "C", "url": "https://c", "snippet": "extremes"},
                ],
            }

    routed = {}

    class FakeRouter:
        async def generate(self, *, prompt, **kwargs):
            routed["prompt"] = prompt
            return "Three articles converge on rising risk. In my view, the evidence is compelling."

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: FakeRouter() if name == "llm_router" else default),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(warning=False, refuse_heavy_local_generation=False),
    )

    skill = DesktopTaskSkill()
    ctx = await skill._collect_research_context(
        capability_engine=FakeCapabilityEngine(),
        objective=(
            "find 3 different recent articles on climate change and summarize "
            "them and your own opinion in a Google Doc"
        ),
        context={"allow_desktop_task_model_synthesis": True},
    )

    assert "In my view" in ctx["desktop_task_research_synthesis"]
    assert "first-person opinion" in routed["prompt"]
