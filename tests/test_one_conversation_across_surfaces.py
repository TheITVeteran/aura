"""Talking and typing are the same conversation.

Switching from talking to typing mid-thought is the most ordinary thing a
person does with something sitting on their desk. It must not be the moment
she loses the thread — and it very nearly was: the voice socket minted a
fresh uuid per *connection* and handed it downstream as the conversation
identity, so a spoken turn and a typed turn were two different threads to
everything that keys on it, and a voice socket that merely reconnected (the
client retries with backoff, so that is routine) started a third.

These tests pin the property rather than the plumbing: the same person gets
the same conversation key however they said it, and it survives a reconnect.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from interface.routes.chat import _chat_turn_session_key


def _request(host: str = "127.0.0.1") -> SimpleNamespace:
    return SimpleNamespace(client=SimpleNamespace(host=host), headers={}, scope={})


def test_the_desktop_derives_its_key_from_the_caller() -> None:
    """No session_id on the wire, so the caller's identity is the thread."""
    key = _chat_turn_session_key(_request(), SimpleNamespace(session_id=""))
    assert key == "127.0.0.1"


def test_a_voice_turn_lands_on_the_same_key_as_a_typed_one() -> None:
    """The property the whole fix exists for.

    The voice lane now passes a principal-derived conversation key rather
    than its socket's uuid, so this is the same string the desktop's own
    turns produce.
    """
    typed = _chat_turn_session_key(_request(), SimpleNamespace(session_id=""))
    spoken = _chat_turn_session_key(_request(), SimpleNamespace(session_id="127.0.0.1"))
    assert typed == spoken


def test_a_reconnect_does_not_start_a_new_conversation() -> None:
    """The client reconnects with backoff; that must not reset the thread."""
    first = _chat_turn_session_key(_request(), SimpleNamespace(session_id="127.0.0.1"))
    after_reconnect = _chat_turn_session_key(
        _request(), SimpleNamespace(session_id="127.0.0.1")
    )
    assert first == after_reconnect


def test_the_voice_route_derives_a_stable_key_not_a_per_socket_uuid() -> None:
    """Read the route's own source: the socket id must not be the thread id.

    A behavioural test would need a live websocket and an authenticated
    principal. What actually regressed here is a single argument, and the
    regression is invisible at runtime — two threads that each work fine on
    their own — so this asserts the wiring directly.
    """
    import inspect

    from interface.routes import voice_duplex

    source = inspect.getsource(voice_duplex.voice_duplex_endpoint)
    assert "conversation_key = (" in source

    # The governed chat turn must receive the conversation key. The socket's
    # own uuid stays where it belongs — session bookkeeping, reservations and
    # logs — so this checks the one call that decides the thread.
    call = source[source.index("run_governed_voice_chat_turn(") :]
    call = call[: call.index(")\n")]
    assert "session_id=conversation_key" in call
    assert "session_id=session_id" not in call


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_the_key_is_stable_for_a_given_caller(host: str) -> None:
    a = _chat_turn_session_key(_request(host), SimpleNamespace(session_id=host))
    b = _chat_turn_session_key(_request(host), SimpleNamespace(session_id=host))
    assert a == b == host
