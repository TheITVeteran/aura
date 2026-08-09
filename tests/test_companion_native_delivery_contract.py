"""The companion surface is one durable Aura, not a decorative second client."""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.perception.ambient_presence import AmbientPresence, PresenceMode
from core.perception.desktop_overlay import BubbleOverlay
from core.skills.computer_use import ComputerUseParams, ComputerUseSkill

ROOT = Path(__file__).resolve().parents[1]
BUBBLE_JS = ROOT / "interface/static/bubble.js"
COMPANION_JS = ROOT / "interface/static/companion_chat.js"
LAUNCHER = ROOT / "scripts/AuraLauncher.swift"


def _native_presence() -> AmbientPresence:
    presence = AmbientPresence()
    presence.set_mode(PresenceMode.BUBBLE)
    presence.note_surface_poll("native-bubble")
    return presence


def test_only_the_native_bubble_consumes_host_commands() -> None:
    presence = _native_presence()
    assert presence.request_highlight(1, 2, 30, 40, 2.0)
    assert presence.request_bubble_move(120, 240)

    preview = presence.state(surface="browser-preview")
    assert preview["highlight"] is None
    assert preview["bubble_move"] is None

    native = presence.state(surface="native-bubble")
    assert native["highlight"]["width"] == 30
    assert native["bubble_move"]["x"] == 120
    assert presence.state(surface="native-bubble")["bubble_move"] is None


def test_movement_refuses_without_a_visible_native_host() -> None:
    presence = AmbientPresence()
    presence.set_mode(PresenceMode.BUBBLE)
    assert presence.request_bubble_move(10, 20) is None

    presence.note_surface_poll("native-bubble")
    presence.hide()
    assert presence.request_bubble_move(10, 20) is None


def test_movement_rejects_non_finite_coordinates() -> None:
    presence = _native_presence()
    assert presence.request_bubble_move(float("nan"), 10) is None
    assert presence.request_bubble_move(10, float("inf")) is None


def test_overlay_exposes_honest_native_movement(monkeypatch) -> None:
    presence = _native_presence()
    monkeypatch.setattr(
        "core.perception.ambient_presence.get_ambient_presence", lambda: presence
    )

    sequence = BubbleOverlay().move_to(x=300, y=200)
    assert sequence == 1
    assert presence.state(surface="native-bubble")["bubble_move"]["y"] == 200


def test_only_the_native_measured_origin_acknowledges_a_move() -> None:
    presence = _native_presence()
    sequence = presence.request_bubble_move(300, 200)
    assert sequence == 1
    assert presence.acknowledge_bubble_move(sequence=2, x=300, y=200) is False
    assert presence.acknowledge_bubble_move(sequence=1, x=288, y=196) is True
    assert asyncio.run(presence.wait_for_bubble_move(1, timeout_s=0.1)) == (
        288.0,
        196.0,
    )


def test_unacknowledged_move_expires_instead_of_claiming_success() -> None:
    presence = _native_presence()
    sequence = presence.request_bubble_move(300, 200)
    assert sequence == 1
    assert asyncio.run(presence.wait_for_bubble_move(1, timeout_s=0.05)) is None


def test_a_later_acknowledgement_cannot_close_an_earlier_receipt() -> None:
    presence = _native_presence()
    first = presence.request_bubble_move(100, 100)
    second = presence.request_bubble_move(200, 200)
    assert (first, second) == (1, 2)
    assert presence.acknowledge_bubble_move(sequence=2, x=200, y=200)
    assert asyncio.run(presence.wait_for_bubble_move(1, timeout_s=0.05)) is None
    assert asyncio.run(presence.wait_for_bubble_move(2, timeout_s=0.05)) == (
        200.0,
        200.0,
    )


async def _acknowledged_position(_sequence: int, *, timeout_s: float):
    assert timeout_s == 6.0
    return (412.0, 318.0)


def test_computer_use_waits_for_native_movement_evidence(monkeypatch) -> None:
    class _Overlay:
        @staticmethod
        def move_to(*, x: float, y: float) -> int:
            assert (x, y) == (400, 300)
            return 17

    class _Presence:
        wait_for_bubble_move = staticmethod(_acknowledged_position)

    monkeypatch.setattr(
        "core.perception.desktop_overlay.get_desktop_overlay", lambda: _Overlay()
    )
    monkeypatch.setattr(
        "core.perception.ambient_presence.get_ambient_presence", lambda: _Presence()
    )

    result = asyncio.run(
        ComputerUseSkill()._execute_action(
            ComputerUseParams(action="move_aura_bubble", x=400, y=300), {}
        )
    )

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["position"] == [412.0, 318.0]
    assert result["sequence"] == 17


def test_computer_use_refuses_when_no_native_bubble_is_alive(monkeypatch) -> None:
    class _Overlay:
        @staticmethod
        def move_to(*, x: float, y: float):
            return None

    monkeypatch.setattr(
        "core.perception.desktop_overlay.get_desktop_overlay", lambda: _Overlay()
    )
    result = asyncio.run(
        ComputerUseSkill()._execute_action(
            ComputerUseParams(action="move_aura_bubble", x=1, y=2), {}
        )
    )
    assert result["ok"] is False
    assert result["effect_verified"] is False
    assert result["status"] == "companion_surface_unavailable"


def test_bubble_preserves_messages_when_state_is_unavailable() -> None:
    source = BUBBLE_JS.read_text(encoding="utf-8")
    assert "state.available === false" in source
    assert "throw new Error(\"ambient state unavailable\")" in source


def test_native_panel_tracks_visible_content_instead_of_intercepting_520_points() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "let size = NSSize(width: 56, height: 56)" in source
    assert 'case "resize":' in source
    assert 'case "move":' in source
    assert "pendingBubbleMoveSequence" in source
    assert "private func resizeBubblePanel" in source
    assert "max(48, min(520, width))" in source
    assert 'surface=restore' in source


def test_companion_uses_the_fenced_desktop_delivery_contract() -> None:
    source = COMPANION_JS.read_text(encoding="utf-8")
    assert '"X-Aura-Surface": "desktop-ui"' in source
    assert '"X-Aura-Require-CognitiveEngine": "true"' in source
    assert '"X-Idempotency-Key"' in source
    assert "/api/chat/delivery/" in source
    assert "AbortController" in source
    assert "PENDING_KEY" in source
    assert "DELIVERY_TIMEOUT_MS" not in source
    assert "while (true)" in source
    assert "updateProgress(payload" in source
    assert "missingPolls >= 3" in source
    assert "postChat(item)" in source
    assert "session_id" not in source, (
        "the companion would fork a disposable conversation instead of sharing "
        "the authenticated owner's canonical desktop session"
    )


def test_companion_exposes_durable_progress_as_status_not_fake_thought() -> None:
    source = COMPANION_JS.read_text(encoding="utf-8")
    html = (ROOT / "interface/static/companion_chat.html").read_text(encoding="utf-8")

    assert 'progress?.message' in source
    assert 'id="thinking-label"' in html
    assert 'role="status"' in html
    assert "chain-of-thought" in html


def test_native_move_sequence_returns_through_the_position_route() -> None:
    bubble = BUBBLE_JS.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "sequence: command.sequence" in bubble
    assert "sequence: Number.isInteger(detail.sequence)" in bubble
    assert 'body["sequence"] as? Int' in launcher
    assert "reportBubbleOrigin(panel.frame.origin, sequence: sequence)" in launcher


def test_reopening_companion_does_not_reload_away_its_transcript() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "if companionWebView?.url == nil" in source
    assert "hideCompanionChat(restoringBubble: false)" in source
    assert "if restoringBubble && !desktopWindowIsVisible()" in source
