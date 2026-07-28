"""Ending voice mode is a thing people do, not a failure.

Measured live. The person asked "Can you hear what I'm saying?", ended voice
mode, and the runtime produced:

    FAULT RUNTIME-VOICE_DUPLEX-ROUTE [MARGINAL] ... RuntimeError: Cannot call
    "send" once a close message has been sent.
    Exception in callback DuplexVoiceSession._task_done(...)
    ... ConnectionClosedOK -> ClientDisconnected -> WebSocketDisconnect

and she said "My response was cut short."

Two defects: `_closed` only tracked OUR close, so a client-initiated close left
it False and the next send raised; and `_task_done` caught a tuple that did not
include the disconnect types, so it escaped as an unhandled callback exception
with a full traceback in the neural feed.
"""

from __future__ import annotations

from core.voice.duplex.session import _is_peer_disconnect


class WebSocketDisconnect(Exception):
    pass


class ClientDisconnected(Exception):
    pass


class ConnectionClosedOK(Exception):
    pass


def _live_chain() -> BaseException:
    """The exact wrapping observed on the live socket."""
    inner = ConnectionClosedOK("received 1000 (OK) user ended voice mode")
    middle = ClientDisconnected()
    middle.__cause__ = inner
    outer = WebSocketDisconnect()
    outer.__cause__ = middle
    return outer


def test_the_live_disconnect_chain_is_recognised():
    assert _is_peer_disconnect(_live_chain()) is True


def test_each_disconnect_type_alone_is_recognised():
    for exc in (WebSocketDisconnect(), ClientDisconnected(), ConnectionClosedOK()):
        assert _is_peer_disconnect(exc) is True, type(exc).__name__


def test_the_close_message_runtimeerror_is_recognised():
    """Raised by the websocket layer without a disconnect type attached."""
    assert _is_peer_disconnect(
        RuntimeError('Cannot call "send" once a close message has been sent.')
    ) is True


def test_a_real_failure_is_not_swallowed():
    """The guard must not become a blanket except."""
    for exc in (ValueError("bad state"), RuntimeError("model exploded"), KeyError("x")):
        assert _is_peer_disconnect(exc) is False, repr(exc)
    assert _is_peer_disconnect(None) is False


def test_a_disconnect_chain_cannot_loop_forever():
    """Self-referential __context__ must not hang the classifier."""
    exc = RuntimeError("boom")
    exc.__context__ = exc
    assert _is_peer_disconnect(exc) is False


def test_the_session_marks_itself_closed_on_a_peer_disconnect():
    """A client-initiated close must stop further sends, not raise on each one."""
    import inspect

    from core.voice.duplex import session as session_module

    source = inspect.getsource(session_module.DuplexVoiceSession.__init__)
    assert "_is_peer_disconnect" in source, (
        "guarded sends must classify a vanished peer instead of raising"
    )
    assert "self._closed = True" in source

    done = inspect.getsource(session_module.DuplexVoiceSession._task_done)
    assert "_is_peer_disconnect" in done, (
        "a disconnect reached the done-callback as an unhandled exception"
    )
