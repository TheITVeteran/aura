import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

from core.runtime.errors import get_degradation_tracker
from core.skills.computer_use import ComputerUseSkill


@pytest.mark.asyncio
async def test_computer_use_read_screen_text_fallback_on_permission_block(monkeypatch):
    skill = ComputerUseSkill()

    # Mock permissions to return blocked for ACCESSIBILITY/AUTOMATION
    async def mock_require_permissions(capability, *permission_names):
        return {"ok": False, "status": "denied", "error": "mock block"}

    called_tree = False

    def mock_query_window_tree():
        nonlocal called_tree
        called_tree = True
        return "Process: Finder\n  Window: Desktop\n    Element [AXButton]: Close"

    monkeypatch.setattr(skill, "_require_permissions", mock_require_permissions)
    monkeypatch.setattr(skill, "_query_system_events_window_tree", mock_query_window_tree)

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})
    assert result["ok"] is True
    assert result["source"] == "applescript_window_tree_fallback"
    assert "Finder" in result["text"]
    assert called_tree is True


@pytest.mark.asyncio
async def test_computer_use_read_screen_text_fallback_on_unavailable(monkeypatch):
    skill = ComputerUseSkill()

    # Mock permissions to pass
    async def mock_require_permissions(capability, *permission_names):
        return None

    # Mock screen text to return unavailable error string
    def mock_read_screen_text_macos():
        return "[accessibility error or ui unresponsive]"

    called_tree = False

    def mock_query_window_tree():
        nonlocal called_tree
        called_tree = True
        return "Fallback Process tree"

    monkeypatch.setattr(skill, "_require_permissions", mock_require_permissions)
    monkeypatch.setattr(skill, "_read_screen_text_macos", mock_read_screen_text_macos)
    monkeypatch.setattr(skill, "_query_system_events_window_tree", mock_query_window_tree)

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})
    assert result["ok"] is True
    assert result["source"] == "applescript_window_tree_fallback"
    assert "Fallback Process tree" in result["text"]
    assert called_tree is True


@pytest.mark.asyncio
async def test_computer_use_click_retry_success(monkeypatch):
    skill = ComputerUseSkill()

    # Mock get_pyautogui
    class MockPyAutoGUI:
        def __init__(self):
            self.clicks = 0

        def click(self, x, y):
            self.clicks += 1

    mock_pyautogui = MockPyAutoGUI()
    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (mock_pyautogui, None))

    # Mock permissions
    async def mock_require_permissions(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", mock_require_permissions)

    # Mock _read_screen_text_macos to simulate state change only on 2nd attempt
    state_counter = 0

    def mock_read_screen():
        nonlocal state_counter
        state_counter += 1
        if state_counter <= 2:
            return "State A"
        return "State B"

    monkeypatch.setattr(skill, "_read_screen_text_macos", mock_read_screen)

    # Fast forward sleep
    async def mock_sleep(secs):
        pass

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    result = await skill.execute({"action": "click", "x": 100, "y": 200}, {})
    assert result["ok"] is True
    assert result["attempts"] == 2
    assert result["verification"] == "State shifted."
    assert mock_pyautogui.clicks == 2


@pytest.mark.asyncio
async def test_computer_use_type_pre_clicks_and_retries(monkeypatch):
    skill = ComputerUseSkill()

    # Mock get_pyautogui
    class MockPyAutoGUI:
        def __init__(self):
            self.clicks = 0
            self.typed = ""

        def click(self, x, y):
            self.clicks += 1

        def typewrite(self, text, interval):
            self.typed = text

    mock_pyautogui = MockPyAutoGUI()
    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (mock_pyautogui, None))

    # Mock permissions
    async def mock_require_permissions(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", mock_require_permissions)

    # Mock _read_screen_text_macos to contain the typed text
    def mock_read_screen():
        return "Hello World! output"

    monkeypatch.setattr(skill, "_read_screen_text_macos", mock_read_screen)

    # Fast forward sleep
    async def mock_sleep(secs):
        pass

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    result = await skill.execute({"action": "type", "target": "Hello World!", "x": 50, "y": 60}, {})
    assert result["ok"] is True
    assert result["attempts"] == 1
    assert result["verification"] == "Text confirmed on screen or state shifted."
    assert mock_pyautogui.clicks == 1
    assert mock_pyautogui.typed == "Hello World!"


@pytest.mark.asyncio
async def test_computer_use_run_command_intercepts(monkeypatch, tmp_path):
    skill = ComputerUseSkill()

    # Let's create a couple of files to list
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file1.txt").write_text("hello")
    (tmp_path / "file2.py").write_text("print(1)")

    # 1. Test tree command intercept
    result = await skill.execute({"action": "run_command", "target": f"tree {tmp_path}"}, {})
    assert result["ok"] is True
    assert "subdir/" in result["output"]
    assert "file2.py" in result["output"]
    assert "file1.txt" in result["output"]

    # 2. Test recursive ls command intercept
    result = await skill.execute({"action": "run_command", "target": f"ls -R {tmp_path}"}, {})
    assert result["ok"] is True
    assert "subdir/" in result["output"]
    assert "file2.py" in result["output"]

    # 3. Test find command auto-constraining depth
    run_args = None

    def mock_run(args, capture_output, text, timeout):
        nonlocal run_args
        run_args = args
        return SimpleNamespace(returncode=0, stdout="find output", stderr="")

    monkeypatch.setattr("core.skills.computer_use.subprocess.run", mock_run)

    result = await skill.execute({"action": "run_command", "target": "find . -name '*.py'"}, {})
    assert result["ok"] is True
    assert "-maxdepth" in run_args
    assert "4" in run_args


@pytest.mark.asyncio
async def test_computer_use_missing_permission_guard_fails_closed(monkeypatch):
    from core.container import ServiceContainer

    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()
    monkeypatch.setattr(ServiceContainer, "get", lambda *_args, **_kwargs: None)

    result = await skill._require_permissions("desktop control", "ACCESSIBILITY")

    assert result["ok"] is False
    assert result["permission"] == "guard"
    assert any(
        "permission guard was not registered" in record.action
        for record in tracker.recent(subsystem="computer_use")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_computer_use_clock_falls_back_when_permission_probe_times_out(monkeypatch):
    from core.container import ServiceContainer

    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()
    skill.PERMISSION_CHECK_TIMEOUT_S = 0.01

    class SlowPermissionGuard:
        async def check_permission(self, *_args, **_kwargs):
            await asyncio.sleep(0.1)
            return {"granted": True, "status": "active", "guidance": ""}

        def get_guidance(self, *_args, **_kwargs):
            return "permission guidance"

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: SlowPermissionGuard()
        if name == "permission_guard"
        else default,
    )

    result = await skill.execute({"action": "read_menu_clock", "target": ""}, context={})

    assert result["ok"] is True
    assert result["status"] == "limited"
    assert result["source"] == "system_clock_permission_fallback"
    assert result["permission_result"]["status"] == "timeout"
    assert any(
        "bounded permission timeout" in record.action
        for record in tracker.recent(subsystem="computer_use")
    )
    tracker.reset()


def test_computer_use_applescript_runner_uses_bounded_subprocess_by_default(monkeypatch):
    skill = ComputerUseSkill()
    run_call = {}

    def fake_run(args, capture_output, text, timeout):
        run_call.update(
            {
                "args": args,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
            }
        )
        return SimpleNamespace(returncode=0, stdout="menu clock", stderr="")

    monkeypatch.delenv("AURA_COMPUTER_USE_NATIVE_APPLESCRIPT", raising=False)
    monkeypatch.setattr("core.skills.computer_use.subprocess.run", fake_run)

    assert skill._run_applescript('return "menu clock"', timeout=6) == "menu clock"
    assert run_call["args"][0] == "osascript"
    assert run_call["timeout"] == 6


@pytest.mark.asyncio
async def test_computer_use_clipboard_actions_use_system_clipboard(monkeypatch):
    skill = ComputerUseSkill()
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "pbpaste":
            return SimpleNamespace(returncode=0, stdout="copied text", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("core.skills.computer_use.subprocess.run", fake_run)

    set_result = await skill.execute({"action": "set_clipboard", "target": "copied text"}, {})
    get_result = await skill.execute({"action": "get_clipboard", "target": ""}, {})

    assert set_result == {"ok": True, "action": "set_clipboard", "chars": 11}
    assert get_result["ok"] is True
    assert get_result["text"] == "copied text"
    assert calls[0][0] == ["pbcopy"]
    assert calls[1][0] == ["pbpaste"]


@pytest.mark.asyncio
async def test_computer_use_run_applescript_requires_permissions_and_blocks_shell(monkeypatch):
    skill = ComputerUseSkill()

    async def allow_permissions(*_args, **_kwargs):
        return None

    monkeypatch.setattr(skill, "_require_permissions", allow_permissions)
    monkeypatch.setattr(skill, "_run_applescript", lambda *_args, **_kwargs: "done")

    ok = await skill.execute({"action": "run_applescript", "target": 'return "done"'}, {})
    blocked_target = 'do shell script "rm -rf ' + "/".join(["", "tmp", "demo"]) + '"'
    blocked = await skill.execute({"action": "run_applescript", "target": blocked_target}, {})

    assert ok["ok"] is True
    assert ok["output"] == "done"
    assert blocked["ok"] is False
    assert "blocked desktop operation" in blocked["error"]


@pytest.mark.asyncio
async def test_computer_use_desktop_file_pdf_and_move_receipts(monkeypatch, tmp_path):
    skill = ComputerUseSkill()
    monkeypatch.setattr(skill, "_allowed_desktop_roots", lambda: [tmp_path])

    source_pdf = tmp_path / "note.pdf"
    moved_pdf = tmp_path / "proof" / "moved-note.pdf"
    receipt_file = tmp_path / "proof" / "receipt.txt"

    pdf_payload = {
        "path": str(source_pdf),
        "title": "Aura Desktop Proof",
        "body": "Equation: 2 + 3 = 5\nCreated by Aura's governed desktop skill.",
    }
    move_payload = {"source": str(source_pdf), "destination": str(moved_pdf)}
    text_payload = {"path": str(receipt_file), "content": "moved PDF into proof folder"}

    rendered = await skill.execute(
        {"action": "render_text_pdf", "target": json.dumps(pdf_payload)},
        {},
    )
    moved = await skill.execute(
        {"action": "move_file", "target": json.dumps(move_payload)},
        {},
    )
    written = await skill.execute(
        {"action": "write_text_file", "target": json.dumps(text_payload)},
        {},
    )

    assert rendered["ok"] is True
    assert rendered["bytes"] > 100
    assert not source_pdf.exists()
    assert moved["ok"] is True
    assert moved_pdf.exists()
    assert moved_pdf.read_bytes().startswith(b"%PDF")
    assert written["ok"] is True
    assert receipt_file.read_text() == "moved PDF into proof folder"


@pytest.mark.asyncio
async def test_computer_use_clock_falls_back_when_applescript_times_out(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()

    async def allow_permissions(*_args, **_kwargs):
        return None

    def fake_run(*_args, timeout, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["osascript"], timeout=timeout)

    monkeypatch.setattr(skill, "_require_permissions", allow_permissions)
    monkeypatch.setattr("core.skills.computer_use.subprocess.run", fake_run)

    result = await skill.execute({"action": "read_menu_clock", "target": ""}, context={})

    assert result["ok"] is True
    assert result["status"] == "limited"
    assert result["source"] == "system_clock_fallback"
    assert "AppleScript timed out" in result["error"]
    assert any(
        "clock fallback" in record.action for record in tracker.recent(subsystem="computer_use")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_computer_use_click_failure_returns_payload_and_receipt(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()

    class DesktopController:
        def __init__(self):
            self.clicked = False

        def click(self, x, y):
            self.clicked = True
            raise RuntimeError(f"desktop rejected click at {x},{y}")

    controller = DesktopController()
    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (controller, None))
    monkeypatch.setattr(skill, "_read_screen_text_macos", lambda: "before")

    async def permissions_available(*_args, **_kwargs):
        return None

    monkeypatch.setattr(skill, "_require_permissions", permissions_available)

    result = await skill.execute({"action": "click", "x": 10, "y": 20}, {})

    assert result["ok"] is False
    assert controller.clicked is True
    assert "desktop rejected click" in result["error"]
    assert any(
        "explicit computer-use failure payload" in record.action
        for record in tracker.recent(subsystem="computer_use")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_computer_use_mycelial_pulse_failure_does_not_block_action(monkeypatch):
    import core.skills.computer_use as computer_use
    from core.container import ServiceContainer

    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()
    container_failures = []
    original_container_get = ServiceContainer.get

    def unavailable_container(*_args, **_kwargs):
        if _args and _args[0] == "mycelial_network":
            container_failures.append("called")
            raise RuntimeError("container unavailable")
        return original_container_get(*_args, **_kwargs)

    monkeypatch.setattr(ServiceContainer, "get", unavailable_container)

    def run_echo(args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="hello\n", stderr="")

    monkeypatch.setattr(computer_use.subprocess, "run", run_echo)

    result = await skill.execute({"action": "run_command", "target": "echo hello"}, {})

    assert result["ok"] is True
    assert result["output"] == "hello"
    assert container_failures == ["called"]
    assert any(
        "mycelial telemetry pulse failed" in record.action
        for record in tracker.recent(subsystem="computer_use")
    )
    tracker.reset()
