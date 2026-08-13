import asyncio
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.conversation import external_chat, terminal_chat
from core.conversation.external_chat_ipc import (
    DurableChannelSpool,
    ExternalChatAuthenticationError,
    ExternalChatLaunchError,
    FrameCodec,
    create_private_channel_directory,
)
from core.conversation.session_scope import current_conversation_session
from core.runtime.errors import get_degradation_tracker


def test_terminal_chat_rejects_unsafe_window_id(tmp_path):
    with pytest.raises(ValueError):
        external_chat.TerminalChatWindow(
            "chat;bad",
            SimpleNamespace(),
            runtime_root=tmp_path,
        )


def test_terminal_script_keeps_channel_secret_out_of_process_arguments(tmp_path):
    window = external_chat.TerminalChatWindow(
        "chat_safe",
        SimpleNamespace(),
        runtime_root=tmp_path,
    )
    initial = 'hello; touch "not-executed"'

    script_path = window._create_chat_script()
    script = script_path.read_text(encoding="utf-8")

    assert initial not in script
    assert script_path.name == "client.py"
    assert script_path.stat().st_mode & 0o777 == 0o700
    assert str(window.channel_dir) in script
    assert window._secret.hex() in script

    window.close()


def test_linux_terminal_launch_uses_argument_vector(tmp_path, monkeypatch):
    launched = []
    window = external_chat.TerminalChatWindow(
        "chat_launch",
        SimpleNamespace(),
        runtime_root=tmp_path,
    )

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
    monkeypatch.setattr(window, "_wait_for_launch_ack", lambda **_kwargs: True)

    window.open("hello")

    assert window.active is True
    assert window.process == 1234
    assert launched[0][0][0:2] == ["xterm", "-e"]
    assert launched[0][0][2] == external_chat.sys.executable
    assert launched[0][0][3].endswith("client.py")
    assert window._secret.hex() not in " ".join(launched[0][0])
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
        external_chat,
        "get_file_write_gateway",
        lambda: FakeFileWriteGateway(),
    )

    window = external_chat.TerminalChatWindow(
        "chat_gateway",
        SimpleNamespace(),
        runtime_root=tmp_path,
    )
    script_path = window._create_chat_script()

    assert script_path.name == "client.py"
    assert [call[3] for call in calls] == ["core.conversation.external_chat.chat_script"]
    assert script_path.stat().st_mode & 0o700 == 0o700
    window.close()


def _spool(tmp_path, channel_id="chat_frame"):
    channel_dir = create_private_channel_directory(tmp_path)
    codec = FrameCodec(channel_id=channel_id, secret=b"s" * 32)
    return DurableChannelSpool(channel_dir, codec), codec


def test_multiline_outbound_remains_one_authenticated_frame_until_ack(tmp_path):
    spool, codec = _spool(tmp_path)
    text = "first line\nsecond line\n\nfinal paragraph"

    message_id = spool.enqueue_outbound(text)

    pending = spool.pending_outbound()
    assert len(pending) == 1
    assert pending[0].message_id == message_id
    assert pending[0].text == text
    assert spool.pending_outbound()[0].text == text

    spool.write_frame(spool.acks_to_server, kind="ack", message_id=message_id)
    assert spool.acknowledge_outbound() == (message_id,)
    assert spool.pending_outbound() == ()


def test_outbound_delivery_order_follows_enqueue_time_not_random_filename(tmp_path):
    spool, _codec = _spool(tmp_path)

    spool.enqueue_outbound("first", message_id="f" * 32)
    spool.enqueue_outbound("second", message_id="b" * 32)
    spool.enqueue_outbound("third", message_id="a" * 32)

    assert [frame.text for frame in spool.pending_outbound()] == ["first", "second", "third"]


def test_authenticated_frame_with_invalid_message_identity_is_rejected(tmp_path):
    _spool_instance, codec = _spool(tmp_path)
    encoded = codec.encode(kind="user_message", text="hello", message_id="bad")

    with pytest.raises(ExternalChatAuthenticationError, match="identity"):
        codec.decode(encoded)


def test_tampered_frame_is_rejected_instead_of_entering_chat(tmp_path):
    spool, _codec = _spool(tmp_path)
    message_id = spool.enqueue_outbound("authentic")
    path = spool.to_client / f"{message_id}.frame"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["text"] = "tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ExternalChatAuthenticationError, match="MAC"):
        spool.pending_outbound()


def test_inbound_frame_is_retained_until_orchestrator_ack(tmp_path):
    spool, _codec = _spool(tmp_path)
    message_id = spool.write_frame(
        spool.to_server,
        kind="user_message",
        text="do not lose this",
    )

    pending = spool.pending_inbound()
    assert [(frame.message_id, frame.text) for frame in pending] == [
        (message_id, "do not lose this")
    ]
    assert (spool.to_server / f"{message_id}.frame").exists()

    spool.acknowledge_inbound(message_id)
    assert not (spool.to_server / f"{message_id}.frame").exists()
    ack = spool.read_frame(spool.acks_to_client / f"{message_id}.frame")
    assert ack.kind == "ack"


def test_private_channel_namespace_is_random_owner_only_and_not_a_symlink(tmp_path):
    first = create_private_channel_directory(tmp_path)
    second = create_private_channel_directory(tmp_path)

    assert first != second
    assert first.parent == second.parent == tmp_path
    assert not first.is_symlink()
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    if hasattr(os, "getuid"):
        assert first.stat().st_uid == os.getuid()


def test_terminal_launch_requires_authenticated_client_ack(tmp_path, monkeypatch):
    window = external_chat.TerminalChatWindow(
        "chat_no_ack",
        SimpleNamespace(),
        runtime_root=tmp_path,
    )
    monkeypatch.setattr(external_chat.platform, "system", lambda: "Linux")
    monkeypatch.setattr(external_chat.shutil, "which", lambda _name: "/usr/bin/xterm")
    monkeypatch.setattr(external_chat, "_spawn_detached", lambda *_args, **_kwargs: 123)
    monkeypatch.setattr(window, "_wait_for_launch_ack", lambda **_kwargs: False)

    with pytest.raises(ExternalChatLaunchError, match="did not acknowledge"):
        window.open("not delivered")

    assert window.active is False
    assert not window.channel_dir.exists()


def test_terminal_launch_fails_when_server_handler_is_not_admitted(tmp_path, monkeypatch):
    window = external_chat.TerminalChatWindow(
        "chat_no_handler",
        SimpleNamespace(),
        runtime_root=tmp_path,
    )
    monkeypatch.setattr(external_chat.platform, "system", lambda: "Linux")
    monkeypatch.setattr(external_chat.shutil, "which", lambda _name: "/usr/bin/xterm")
    monkeypatch.setattr(external_chat, "_spawn_detached", lambda *_args, **_kwargs: 123)
    monkeypatch.setattr(window, "_wait_for_launch_ack", lambda **_kwargs: True)
    monkeypatch.setattr(
        external_chat,
        "get_task_tracker",
        lambda: SimpleNamespace(
            create_task=lambda _handler, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("task admission denied")
            )
        ),
    )

    with pytest.raises(ExternalChatLaunchError, match="task admission denied"):
        window.open("not delivered")

    assert window.active is False
    assert not window.channel_dir.exists()


def test_external_turn_uses_canonical_foreground_origin_and_local_conversation():
    calls = []

    async def process(message, *, origin):
        calls.append((message, origin, current_conversation_session()))
        return "status is ready"

    result = asyncio.run(
        external_chat._run_external_turn(
            SimpleNamespace(process_user_input_priority=process),
            "please inspect status",
        )
    )

    assert result == "status is ready"
    assert calls == [("please inspect status", "external", "127.0.0.1")]


def test_external_turn_fails_when_foreground_processor_is_missing():
    with pytest.raises(ExternalChatLaunchError, match="no foreground processing"):
        asyncio.run(external_chat._run_external_turn(SimpleNamespace(), "hello"))


def test_inbound_completion_commits_correlated_reply_before_ack(tmp_path):
    spool, _codec = _spool(tmp_path)
    inbound_id = spool.write_frame(
        spool.to_server,
        kind="user_message",
        text="hello",
    )

    response_id = spool.complete_inbound(inbound_id, response_text="hello back")

    assert response_id == inbound_id
    assert spool.inbound_completed(inbound_id) is True
    assert spool.pending_inbound() == ()
    assert [(frame.message_id, frame.text) for frame in spool.pending_outbound()] == [
        (inbound_id, "hello back")
    ]
    assert spool.read_frame(spool.acks_to_client / f"{inbound_id}.frame").kind == "ack"


def test_completed_inbound_replay_does_not_require_response_to_remain(tmp_path):
    spool, _codec = _spool(tmp_path)
    inbound_id = spool.write_frame(
        spool.to_server,
        kind="user_message",
        text="hello",
    )
    spool.complete_inbound(inbound_id, response_text="hello back")
    spool.write_frame(spool.acks_to_server, kind="ack", message_id=inbound_id)
    spool.acknowledge_outbound()

    assert spool.pending_outbound() == ()
    assert spool.inbound_completed(inbound_id) is True


@pytest.mark.asyncio
async def test_terminal_handler_routes_exact_reply_before_input_ack(tmp_path, monkeypatch):
    calls = []
    completed = asyncio.Event()
    owner_loop = asyncio.get_running_loop()

    async def process(message, *, origin):
        calls.append((message, origin, current_conversation_session()))
        return "A multiline reply.\n\nStill one turn."

    window = external_chat.TerminalChatWindow(
        "chat_round_trip",
        SimpleNamespace(
            process_user_input_priority=process,
            conversation_history=[],
        ),
        runtime_root=tmp_path,
    )
    inbound_id = window._spool.write_frame(
        window._spool.to_server,
        kind="user_message",
        text="please answer",
    )
    original_complete = window._spool.complete_inbound

    def complete_inbound(*args, **kwargs):
        result = original_complete(*args, **kwargs)
        owner_loop.call_soon_threadsafe(completed.set)
        return result

    monkeypatch.setattr(window._spool, "complete_inbound", complete_inbound)
    window.active = True
    task = asyncio.create_task(window._message_handler_loop())
    try:
        async with asyncio.timeout(2.0):
            await completed.wait()
    finally:
        window.active = False
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls == [("please answer", "external", "127.0.0.1")]
    assert window._spool.pending_inbound() == ()
    assert [(frame.message_id, frame.text) for frame in window._spool.pending_outbound()] == [
        (inbound_id, "A multiline reply.\n\nStill one turn.")
    ]
    window.close()


def test_terminal_outbox_survives_absent_reader_and_reconnect(tmp_path):
    orchestrator = SimpleNamespace(conversation_history=[])
    window = external_chat.TerminalChatWindow(
        "chat_reconnect",
        orchestrator,
        runtime_root=tmp_path,
    )
    window.active = True

    message_id = window.send_message("paragraph one\n\nparagraph two")

    assert [frame.message_id for frame in window._spool.pending_outbound()] == [message_id]
    assert orchestrator.conversation_history[-1]["delivery_state"] == "pending_ack"
    reconnected = DurableChannelSpool(window.channel_dir, window._codec)
    assert reconnected.pending_outbound()[0].text == "paragraph one\n\nparagraph two"

    reconnected.write_frame(
        reconnected.acks_to_server,
        kind="ack",
        message_id=message_id,
    )
    assert reconnected.acknowledge_outbound() == (message_id,)
    assert reconnected.pending_outbound() == ()
    window.close()


def test_manager_does_not_publish_window_after_launch_failure(tmp_path, monkeypatch):
    manager = external_chat.ExternalChatManager(
        SimpleNamespace(conversation_history=[]),
        runtime_root=tmp_path,
    )

    def fail_open(self, _message=None):
        raise ExternalChatLaunchError("terminal unavailable")

    monkeypatch.setattr(external_chat.TerminalChatWindow, "open", fail_open)

    with pytest.raises(ExternalChatLaunchError, match="terminal unavailable"):
        manager.open_chat_window("hello", window_type="terminal")

    assert manager.windows == {}
    assert manager.get_active_windows() == []


def test_manager_publishes_only_active_random_channel(tmp_path, monkeypatch):
    manager = external_chat.ExternalChatManager(
        SimpleNamespace(conversation_history=[]),
        runtime_root=tmp_path,
    )

    def open_active(self, _message=None):
        self.active = True

    monkeypatch.setattr(external_chat.TerminalChatWindow, "open", open_active)
    first = manager.open_chat_window("one", window_type="terminal")
    second = manager.open_chat_window("two", window_type="terminal")

    assert first != second
    assert first.startswith("chat_") and len(first) > len("chat_1")
    assert set(manager.get_active_windows()) == {first, second}
    manager.close_all_windows()


def test_manager_executes_terminal_fallback_after_gui_launch_failure(tmp_path, monkeypatch):
    manager = external_chat.ExternalChatManager(
        SimpleNamespace(conversation_history=[]),
        runtime_root=tmp_path,
    )

    def gui_fails(self, _message=None):
        raise ExternalChatLaunchError("tk unavailable")

    def terminal_opens(self, _message=None):
        self.active = True

    monkeypatch.setattr(external_chat.GUIChatWindow, "open", gui_fails)
    monkeypatch.setattr(external_chat.TerminalChatWindow, "open", terminal_opens)

    window_id = manager.open_chat_window("hello", window_type="gui")

    assert isinstance(manager.windows[window_id], external_chat.TerminalChatWindow)
    assert manager.windows[window_id].active is True
    manager.close_all_windows()


def test_gui_launch_fails_without_live_owner_loop():
    window = external_chat.GUIChatWindow("chat_gui", SimpleNamespace())

    with pytest.raises(ExternalChatLaunchError, match="no live orchestrator event loop"):
        window.open("hello")

    assert window.active is False


def test_manager_shutdown_closes_windows_and_removes_runtime_root(tmp_path, monkeypatch):
    manager = external_chat.ExternalChatManager(
        SimpleNamespace(conversation_history=[]),
        runtime_root=tmp_path,
    )

    def open_active(self, _message=None):
        self.active = True

    monkeypatch.setattr(external_chat.TerminalChatWindow, "open", open_active)
    manager.open_chat_window("hello", window_type="terminal")
    runtime_root = manager.runtime_root

    manager.shutdown()

    assert manager.windows == {}
    assert not runtime_root.exists()


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
    # Patch the GATEWAY, not asyncio. Shell launches route through
    # get_subprocess_gateway().spawn_shell_async — governed, with a declared
    # accelerator capability — so patching asyncio.create_subprocess_shell
    # replaced something the code no longer calls, and these tests were
    # asserting against a fake process the runtime never touched.
    async def _gateway_spawn(command, **kwargs):
        return await fail_if_launched(command)

    monkeypatch.setattr(
        terminal_chat,
        "get_subprocess_gateway",
        lambda: SimpleNamespace(spawn_shell_async=_gateway_spawn),
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
    # Patch the GATEWAY, not asyncio. Shell launches route through
    # get_subprocess_gateway().spawn_shell_async — governed, with a declared
    # accelerator capability — so patching asyncio.create_subprocess_shell
    # replaced something the code no longer calls, and these tests were
    # asserting against a fake process the runtime never touched.
    async def _gateway_spawn(command, **kwargs):
        return await spawn(command)

    monkeypatch.setattr(
        terminal_chat,
        "get_subprocess_gateway",
        lambda: SimpleNamespace(spawn_shell_async=_gateway_spawn),
    )

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
    # Patch the GATEWAY, not asyncio. Shell launches route through
    # get_subprocess_gateway().spawn_shell_async — governed, with a declared
    # accelerator capability — so patching asyncio.create_subprocess_shell
    # replaced something the code no longer calls, and these tests were
    # asserting against a fake process the runtime never touched.
    async def _gateway_spawn(command, **kwargs):
        return await spawn(command)

    monkeypatch.setattr(
        terminal_chat,
        "get_subprocess_gateway",
        lambda: SimpleNamespace(spawn_shell_async=_gateway_spawn),
    )

    asyncio.run(terminal._run_shell_command("echo hi"))

    assert any("Shell error: spawn denied" in message for message in outputs)
    assert any(
        "returned shell error" in record.action
        for record in get_degradation_tracker().recent(subsystem="terminal_chat", limit=5)
    )
