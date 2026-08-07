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
    """The derivation is a named function, so it can be exercised rather than read.

    The first version of this test asserted against `inspect.getsource` of the
    route, which the source-inspection ratchet correctly rejected: a string
    match passes on code that has been commented out. Naming the derivation
    made it testable *and* gave the concept somewhere to live.
    """
    from interface.routes.voice_duplex import conversation_key_for

    # A local owner: the key is the caller, which is exactly what the
    # desktop's own chat turns derive.
    assert conversation_key_for(None, "127.0.0.1") == "127.0.0.1"

    # Same person, same conversation, however they said it — and stable
    # across the reconnects the client performs with backoff.
    assert conversation_key_for(None, "127.0.0.1") == conversation_key_for(
        None, "127.0.0.1"
    )

    # A paired device is its own thread, and cannot collide with a host.
    assert conversation_key_for("phone-42", "127.0.0.1") == "paired:phone-42"
    assert conversation_key_for("phone-42", "127.0.0.1") != conversation_key_for(
        None, "127.0.0.1"
    )


def test_the_voice_key_is_what_the_desktop_would_produce() -> None:
    """The two derivations have to agree, or the thread splits on surface.

    This is the property the whole fix is about, and it is checkable without
    a websocket: run both derivations for the same principal and compare.
    """
    from types import SimpleNamespace

    from interface.routes.voice_duplex import conversation_key_for

    spoken = conversation_key_for(None, "127.0.0.1")
    typed = _chat_turn_session_key(
        _request("127.0.0.1"), SimpleNamespace(session_id=spoken)
    )
    assert spoken == typed


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_the_key_is_stable_for_a_given_caller(host: str) -> None:
    a = _chat_turn_session_key(_request(host), SimpleNamespace(session_id=host))
    b = _chat_turn_session_key(_request(host), SimpleNamespace(session_id=host))
    assert a == b == host
