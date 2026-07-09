import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import external_chat, terminal_chat
from core.runtime.errors import get_degradation_tracker


def test_terminal_chat_rejects_unsafe_window_id():
    with pytest.raises(ValueError):
        external_chat.TerminalChatWindow("chat;bad", SimpleNamespace())


def test_terminal_script_stores_initial_message_outside_shell_body():
    window = external_chat.TerminalChatWindow("chat_safe", SimpleNamespace())
    initial = 'hello; touch "not-executed"'

    script_path = window._create_chat_script(initial)
    script = script_path.read_text(encoding="utf-8")
    message_file = script_path.with_name("chat_safe_initial.txt")

    assert initial not in script
    assert message_file.read_text(encoding="utf-8") == initial
    assert "INITIAL_MESSAGE_FILE=" in script

    window.close()


def test_linux_terminal_launch_uses_argument_vector(monkeypatch):
    launched = []
    window = external_chat.TerminalChatWindow("chat_launch", SimpleNamespace())

    monkeypatch.setattr(external_chat.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        external_chat.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "xterm" else None
    )
    monkeypatch.setattr(
        external_chat,
        "_spawn_detached",
        lambda command, **kwargs: launched.append((command, kwargs)) or 1234,
    )
    monkeypatch.setattr(window, "_start_message_handler", lambda: None)

    window.open("hello")

    assert window.active is True
    assert window.process == 1234
    assert launched[0][0][0:3] == ["xterm", "-e", "bash"]
    assert launched[0][0][3].endswith("chat_launch.sh")
    assert launched[0][1] == {"source": "core.conversation.external_chat.terminal_linux_launch"}

    window.close()


def test_terminal_script_uses_file_write_gateway(tmp_path, monkeypatch):
    calls = []

    class FakeFileWriteGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown"):
            calls.append((Path(path).name, text, encoding, source))
            Path(path).write_text(text, encoding=encoding)

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def write_text_async(self, *args, **kwargs):
            return self.write_text(*args, **kwargs)

    monkeypatch.setattr(
        external_chat.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        external_chat,
        "get_file_write_gateway",
        lambda: FakeFileWriteGateway(),
    )

    window = external_chat.TerminalChatWindow("chat_gateway", SimpleNamespace())
    script_path = window._create_chat_script("hello")

    assert script_path.name == "chat_gateway.sh"
    assert [call[3] for call in calls] == [
        "core.conversation.external_chat.initial_message",
        "core.conversation.external_chat.chat_script",
    ]
    assert script_path.stat().st_mode & 0o700 == 0o700


def test_pipe_helpers_read_and_write_regular_files(tmp_path):
    inbound = tmp_path / "in"
    inbound.write_text("first\n\nsecond\n", encoding="utf-8")

    assert external_chat.TerminalChatWindow._read_pipe_messages(inbound) == ["first", "second"]

    outbound = tmp_path / "out"
    outbound.touch()
    external_chat.TerminalChatWindow._write_pipe_message(outbound, "aura says hi")

    assert outbound.read_text(encoding="utf-8") == "aura says hi\n"


def test_terminal_user_message_reaches_orchestrator_history():
    calls = []
    orchestrator = SimpleNamespace(
        conversation_history=[],
        enqueue_from_thread=lambda message, origin: calls.append((message, origin)),
    )
    window = external_chat.TerminalChatWindow("chat_history", orchestrator)

    window._process_user_message("please inspect status")

    assert calls == [("please inspect status", "external_window_chat_history")]
    assert orchestrator.conversation_history[-1]["message"] == "please inspect status"


def test_terminal_shell_fails_closed_when_safety_controller_unavailable(monkeypatch):
    import core.autonomy.behavior_controller as behavior_controller

    get_degradation_tracker().reset()
    outputs = []
    launched = []
    terminal = terminal_chat.TerminalFallbackChat()
    terminal._write_output = outputs.append

    class BrokenController:
        def __init__(self):
            self.reason = "safety controller offline"
            raise RuntimeError("safety controller offline")

    async def fail_if_launched(*args, **kwargs):
        launched.append((args, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(terminal_chat, "MIN_OUTPUT_INTERVAL", 0)
    monkeypatch.setattr(
        behavior_controller,
        "AutonomousBehaviorController",
        BrokenController,
    )
    monkeypatch.setattr(
        terminal_chat.asyncio,
        "create_subprocess_shell",
        fail_if_launched,
    )

    asyncio.run(terminal._run_shell_command("echo should-not-run"))

    assert launched == []
    assert any("fail-closed" in message for message in outputs)
    assert any(
        "blocked terminal shell command" in record.action
        for record in get_degradation_tracker().recent(subsystem="terminal_chat", limit=5)
    )


def test_terminal_shell_timeout_kills_and_reaps_process(monkeypatch):
    import core.autonomy.behavior_controller as behavior_controller

    get_degradation_tracker().reset()
    outputs = []
    process = SimpleNamespace(killed=False)
    terminal = terminal_chat.TerminalFallbackChat()
    terminal._write_output = outputs.append

    class ApprovingController:
        def validate_action(self, _action):
            return True

    async def communicate():
        await asyncio.sleep(1.0)
        return b"", None

    def kill():
        process.killed = True

    async def wait():
        return 0

    async def spawn(*args, **kwargs):
        process.communicate = communicate
        process.kill = kill
        process.wait = wait
        return process

    monkeypatch.setattr(terminal_chat, "MIN_OUTPUT_INTERVAL", 0)
    monkeypatch.setattr(terminal_chat, "SHELL_COMMAND_TIMEOUT_SECS", 0.01)
    monkeypatch.setattr(
        behavior_controller,
        "AutonomousBehaviorController",
        ApprovingController,
    )
    monkeypatch.setattr(terminal_chat.asyncio, "create_subprocess_shell", spawn)

    asyncio.run(terminal._run_shell_command("sleep 60"))

    assert process.killed is True
    assert any("timed out" in message for message in outputs)
    assert any(
        "killed timed-out terminal shell command" in record.action
        for record in get_degradation_tracker().recent(subsystem="terminal_chat", limit=5)
    )


def test_terminal_shell_launch_error_is_reported(monkeypatch):
    import core.autonomy.behavior_controller as behavior_controller

    get_degradation_tracker().reset()
    outputs = []
    terminal = terminal_chat.TerminalFallbackChat()
    terminal._write_output = outputs.append

    class ApprovingController:
        def validate_action(self, _action):
            return True

    async def spawn(*args, **kwargs):
        outputs.append("spawn attempted")
        raise OSError("spawn denied")

    monkeypatch.setattr(terminal_chat, "MIN_OUTPUT_INTERVAL", 0)
    monkeypatch.setattr(
        behavior_controller,
        "AutonomousBehaviorController",
        ApprovingController,
    )
    monkeypatch.setattr(terminal_chat.asyncio, "create_subprocess_shell", spawn)

    asyncio.run(terminal._run_shell_command("echo hi"))

    assert any("Shell error: spawn denied" in message for message in outputs)
    assert any(
        "returned shell error" in record.action
        for record in get_degradation_tracker().recent(subsystem="terminal_chat", limit=5)
    )
