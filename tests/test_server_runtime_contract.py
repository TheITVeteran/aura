import asyncio
import inspect

import pytest

from interface import server


@pytest.mark.asyncio
async def test_ws_broadcaster_unsubscribes_when_shutdown_is_requested(monkeypatch):
    events = []

    class Bus:
        async def subscribe(self):
            events.append("subscribed")
            return asyncio.Queue()

        async def unsubscribe(self, queue):
            events.append(("unsubscribed", queue.empty()))

    monkeypatch.setattr(server, "broadcast_bus", Bus())
    monkeypatch.setattr(server, "is_shutdown_requested", lambda: True)

    await server._ws_broadcaster()

    assert events == ["subscribed", ("unsubscribed", True)]


def test_websocket_timeout_path_does_not_direct_generate_raw_fallback():
    source = inspect.getsource(server.websocket_endpoint)

    assert "gate.generate(" not in source
    assert "instead of fabricating a recovered answer" in source


def test_websocket_chat_uses_desktop_cognitive_engine_trace_metadata():
    source = inspect.getsource(server.websocket_endpoint)

    assert 'origin="desktop-ui"' in source
    assert 'source="desktop_websocket"' in source
    assert "require_engine=True" in source
    assert "desktop WebSocket chat path requires CognitiveEngine" in source


def test_websocket_desktop_path_has_no_legacy_kernel_or_orchestrator_fallback():
    source = inspect.getsource(server.websocket_endpoint)

    assert "KernelInterface" not in source
    assert "process_user_input_priority" not in source
    assert "recovered WebSocket reply" not in source
    assert "refusing legacy fallback" in source


def test_cognitive_engine_turn_required_contract_has_no_kernel_fallback_language():
    from interface.routes import chat

    signature = inspect.signature(chat._run_cognitive_engine_chat_turn)
    source = inspect.getsource(chat._run_cognitive_engine_chat_turn)

    assert "require_engine" in signature.parameters
    assert "cognitive_engine_required" in source
    assert "required caller must fail closed" in source
    assert "falling back to kernel lane" not in source


def test_desktop_cognitive_turn_carries_generic_execution_planning_contract():
    from interface.routes import chat
    from core.phases.response_generation import ResponseGenerationPhase
    from core.runtime.desktop_task_contract import (
        DESKTOP_TASK_ALLOWED_ACTIONS,
        desktop_task_action_sentence,
        desktop_task_planning_schema,
    )

    chat_source = inspect.getsource(chat._run_cognitive_engine_chat_turn)
    response_source = inspect.getsource(ResponseGenerationPhase.execute)

    assert "desktop_execution_contract" in chat_source
    assert "desktop_task_planning_schema" in chat_source
    schema = desktop_task_planning_schema()
    assert "{{document_body}}" in schema["steps"][0]["target"]
    assert "{{steps.1.result.path}}" in schema["steps"][0]["target"]
    assert "critical" in schema["steps"][0]
    assert "DESKTOP_TASK_ALLOWED_ACTIONS" in chat_source
    assert "LIVE DESKTOP EXECUTION PLANNING CONTRACT" in response_source
    assert "Do not claim completion" in response_source
    assert "inside this draft" in response_source
    action_sentence = desktop_task_action_sentence()
    for action in DESKTOP_TASK_ALLOWED_ACTIONS:
        assert action in action_sentence
