"""The ambient client's promises, checked against its source.

Following this repo's existing convention for front-end contracts (see
`test_desktop_shell_contracts.py`): the properties below are the ones a
refactor could quietly drop, and every one of them is either a privacy
promise or the reason the feature is worth having at all. None of them are
visible in a screenshot, and the failure mode of each is silence rather than
an error.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICE_MODE_JS = ROOT / "interface/static/voice_mode.js"
AURA_JS = ROOT / "interface/static/aura.js"
INDEX_HTML = ROOT / "interface/static/index.html"


def _voice() -> str:
    return VOICE_MODE_JS.read_text(encoding="utf-8")


def _aura() -> str:
    return AURA_JS.read_text(encoding="utf-8")


# ── the privacy promise ──────────────────────────────────────────────────


def test_ambient_never_asks_for_the_microphone_on_load() -> None:
    """A page that pops a permission prompt on load has decided for the user.

    Ambient listening resumes only where the browser *already* holds a grant
    from a previous deliberate act. Pressing VOICE is where that decision
    belongs, because pressing it is the request.
    """
    source = _voice()
    assert "navigator.permissions.query({ name: 'microphone' })" in source
    assert "status.state === 'granted'" in source
    # The startup path must consult the permission before touching capture.
    startup = source[source.index("async function maybeStartAmbient") :]
    startup = startup[: startup.index("\n    }")]
    assert "if (!granted) return false;" in startup
    # Feature-detecting `getUserMedia` is fine; *calling* it is the prompt.
    assert "getUserMedia({" not in startup


def test_the_local_preference_defaults_to_off() -> None:
    """An ambient microphone is not something to enable by accident.

    The persisted `voice.auto_listen` setting is the authority; the local
    copy exists only so a reload resumes in the same frame. If it cannot be
    read, the answer is no.
    """
    source = _voice()
    assert "window.localStorage.getItem(AMBIENT_PREF_KEY) === 'on'" in source
    enabled = source[source.index("enabledByUser()") :]
    enabled = enabled[: enabled.index("},")]
    assert "return false;" in enabled


def test_a_denied_microphone_stops_ambient_silently() -> None:
    """Ambient listening must never nag. It stops; the button remains."""
    source = _voice()
    assert "if (state.ambient)" in source
    assert "ambient.setEnabled(false)" in source


# ── the reason it is worth having ────────────────────────────────────────


def test_spoken_turns_go_into_the_chat_thread() -> None:
    """One conversation, not two that agree afterwards.

    The old surface kept its own transcript and folded it back on exit, so
    the visible history disagreed with what she remembered until the modal
    was closed — which reads as her confabulating.
    """
    voice = _voice()
    aura = _aura()
    for hook in ("auraAppendVoiceTurn", "auraStreamVoiceReply", "auraFinishVoiceReply"):
        assert f"window.{hook}" in aura, f"{hook} is not defined in the chat surface"
        assert hook in voice, f"{hook} is never called from the voice surface"
    # The reply is streamed into the same bubble machinery a typed reply uses.
    assert "startStreamMsg('aura')" in aura
    assert "appendStreamChunk(" in aura


def test_the_focused_surface_tells_the_server_the_floor_is_open() -> None:
    """Otherwise focused voice mode still second-guesses whether it is being
    spoken to, which is the one place that judgement is not wanted."""
    source = _voice()
    assert "sendCommand('set_floor'" in source
    # Re-asserted on connect: a reconnect gets a fresh session that defaults
    # to ambient, and a user sitting in focused mode would find the gate
    # silently back in front of them.
    connect = source[source.index("ws.onopen = () =>") :]
    connect = connect[: connect.index("};")]
    assert "set_floor" in connect


def test_the_overheard_line_is_transient_and_never_enters_the_thread() -> None:
    """An ambient microphone that transcribed the room into the chat log
    would be its own kind of hostile; showing nothing would make the gate
    invisible."""
    source = _voice()
    assert "showOverheard(" in source
    assert "ambient-overheard-show" in source
    overheard = source[source.index("showOverheard(text, why)") :]
    overheard = overheard[: overheard.index("},")]
    assert "setTimeout(" in overheard
    assert "auraAppendVoiceTurn" not in overheard


def test_ambient_does_not_steal_the_keyboard() -> None:
    """Space is a page scroll and Escape closes what the user was looking at.

    Taking either because a microphone happens to be open is what makes a
    background feature feel like a foreground one.
    """
    source = _voice()
    handler = source[source.index("function onKeyDown(e)") :]
    handler = handler[: handler.index("\n    }")]
    assert "if (state.ambient)" in handler
    assert "return;" in handler


def test_leaving_the_focused_surface_does_not_switch_the_microphone_off() -> None:
    """"End" closes the window. It is not a setting the user never touched."""
    source = _voice()
    assert "async function leaveFocusedMode()" in source
    assert "leaveFocusedMode()" in source
    leave = source[source.index("async function leaveFocusedMode()") :]
    leave = leave[: leave.index("\n    }")]
    assert "ambient.enabledByUser()" in leave


# ── one control, not two ─────────────────────────────────────────────────


def test_auto_listen_prefers_the_duplex_lane() -> None:
    """The duplex lane is the one with barge-in, backchannels, clause
    streaming and the addressivity gate. The legacy half-duplex path stays
    only so that auto-listen is never simply dead."""
    source = _aura()
    reconcile = source[source.index("async function reconcileAutoListenFromSettings") :]
    reconcile = reconcile[: reconcile.index("\n}")]
    assert "duplex.setAmbient(true)" in reconcile
    assert "toggleVoice(true" in reconcile  # the fallback survives
    assert reconcile.index("duplex.setAmbient(true)") < reconcile.index("toggleVoice(true")


def test_both_voice_assets_are_actually_loaded() -> None:
    """None of the above runs if the bundle is not on the page."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert '<script src="/static/voice_mode.js"></script>' in html
    assert '<link rel="stylesheet" href="/static/voice_mode.css">' in html
    # aura.js defines the chat hooks the voice surface calls, so it has to be
    # installed first.
    assert html.index("/static/aura.js") < html.index("/static/voice_mode.js")
