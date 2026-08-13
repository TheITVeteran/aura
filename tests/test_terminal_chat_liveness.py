from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from types import SimpleNamespace

from core.conversation import terminal_chat


def test_idle_timeouts_keep_one_owned_stdin_reader(monkeypatch):
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8", buffering=1)
    monkeypatch.setattr(terminal_chat.sys, "stdin", reader)
    chat = terminal_chat.TerminalFallbackChat()

    async def scenario() -> None:
        assert await chat._read_stdin_line(wait_seconds=0.01) is None
        loop = chat._stdin_loop
        fd = chat._stdin_fd
        assert chat._stdin_reader_installed

        assert await chat._read_stdin_line(wait_seconds=0.01) is None
        assert chat._stdin_loop is loop
        assert chat._stdin_fd == fd

        os.write(write_fd, b"hello\n")
        assert await chat._read_stdin_line(wait_seconds=0.5) == "hello\n"
        chat._stop_stdin_reader()
        assert not chat._stdin_reader_installed

    try:
        asyncio.run(scenario())
    finally:
        os.close(write_fd)
        reader.close()


def test_output_rate_limit_yields_instead_of_blocking(monkeypatch):
    monkeypatch.setattr(terminal_chat, "MIN_OUTPUT_INTERVAL", 0.05)
    chat = terminal_chat.TerminalFallbackChat()
    written: list[str] = []
    chat._write_output = written.append

    async def scenario() -> None:
        chat._last_output_at = terminal_chat.time.monotonic()
        output_task = asyncio.create_task(chat._write_output_scheduled("second"))
        await asyncio.sleep(0)
        assert not output_task.done()
        marker = []
        await asyncio.sleep(0)
        marker.append("event-loop-progressed")
        assert marker == ["event-loop-progressed"]
        await output_task

    asyncio.run(scenario())
    assert written == ["second"]


def test_deactivate_joins_pending_stdin_wait_and_removes_reader(monkeypatch):
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8", buffering=1)
    monkeypatch.setattr(terminal_chat.sys, "stdin", reader)
    chat = terminal_chat.TerminalFallbackChat()

    async def scenario() -> None:
        chat._active = True
        task = asyncio.create_task(chat._read_stdin_line(wait_seconds=10.0))
        chat._chat_task = task
        await asyncio.sleep(0)
        assert chat._stdin_reader_installed

        await chat.deactivate("test")

        assert task.cancelled()
        assert chat._chat_task is None
        assert not chat._stdin_reader_installed

    try:
        asyncio.run(scenario())
    finally:
        os.close(write_fd)
        reader.close()


def test_stdin_eof_removes_readiness_callback_before_reporting_eof(monkeypatch):
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8", buffering=1)
    monkeypatch.setattr(terminal_chat.sys, "stdin", reader)
    chat = terminal_chat.TerminalFallbackChat()

    async def scenario() -> None:
        task = asyncio.create_task(chat._read_stdin_line(wait_seconds=0.5))
        await asyncio.sleep(0)
        os.close(write_fd)
        try:
            await task
        except EOFError:
            pass
        else:
            raise AssertionError("EOF must be reported to the terminal loop")
        assert not chat._stdin_reader_installed
        assert chat._stdin_lines.empty()
        chat._stop_stdin_reader()

    try:
        asyncio.run(scenario())
    finally:
        with suppress(OSError):
            os.close(write_fd)
        reader.close()


def test_partial_pipe_input_never_blocks_and_is_reassembled(monkeypatch):
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8", buffering=1)
    monkeypatch.setattr(terminal_chat.sys, "stdin", reader)
    chat = terminal_chat.TerminalFallbackChat()

    async def scenario() -> None:
        task = asyncio.create_task(chat._read_stdin_line(wait_seconds=0.5))
        await asyncio.sleep(0)
        os.write(write_fd, b"caf")
        await asyncio.sleep(0.01)
        assert not task.done()
        suffix = "é\n".encode()
        os.write(write_fd, suffix[:1])
        await asyncio.sleep(0.01)
        assert not task.done()
        os.write(write_fd, suffix[1:])
        assert await task == "café\n"
        chat._stop_stdin_reader()

    try:
        asyncio.run(scenario())
    finally:
        os.close(write_fd)
        reader.close()


def test_ui_liveness_uses_owner_clients_not_runtime_processes(monkeypatch):
    from interface import websocket_manager

    chat = terminal_chat.TerminalFallbackChat()
    manager = SimpleNamespace(owner_count=lambda: 0)
    monkeypatch.setattr(websocket_manager, "ws_manager", manager)
    assert chat._is_main_ui_open() is False

    manager.owner_count = lambda: 1
    assert chat._is_main_ui_open() is True


def test_paired_conversation_socket_is_not_an_owner_ui():
    from interface.websocket_manager import WebSocketManager

    owner = object()
    paired = object()
    manager = WebSocketManager()
    manager.active_connections = {owner: asyncio.Queue(), paired: asyncio.Queue()}
    manager._connection_scopes = {owner: "owner", paired: "conversation"}

    assert manager.count() == 2
    assert manager.owner_count() == 1
