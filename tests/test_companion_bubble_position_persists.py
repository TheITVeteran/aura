""""She can move it, position persists" described a POST with no reader.

Dragging the bubble worked. The page debounced the move and posted it, and
``move_bubble`` stored the coordinates — in memory, on an object that dies
with the process, in a field the launcher never read back. So she reappeared
in the bottom-left corner after every restart regardless of where she had
been left, including the restarts nobody chose.

Two halves had to be true and neither was: the runtime has to remember across
a restart, and the launcher has to place her there before showing her.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.perception.ambient_presence import AmbientPresence, PresenceMode

_LAUNCHER = Path("scripts/AuraLauncher.swift")


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def presence(data_dir):
    instance = AmbientPresence()
    instance.set_mode(PresenceMode.BUBBLE)
    return instance


# ────────────────────────────────────────────── it survives the process


def test_a_parked_position_is_written_to_disk(presence, data_dir):
    presence.move_bubble(1200.0, 840.0)

    assert asyncio.run(presence.persist_bubble_position()) is True

    written = json.loads(
        (data_dir / "companion" / "bubble_position.json").read_text(encoding="utf-8")
    )
    # Through the gateway, so it carries the schema envelope rather than being
    # a bare dict — and the loader has to read it back out of that envelope.
    assert written["schema_version"] == 1
    assert written["payload"]["x"] == 1200.0
    assert written["payload"]["y"] == 840.0


def test_a_restart_puts_her_back_where_she_was_left(presence, data_dir):
    """The property the feature claimed and did not have."""
    presence.move_bubble(1200.0, 840.0)
    asyncio.run(presence.persist_bubble_position())

    reborn = AmbientPresence()

    assert reborn.state()["bubble_position"] == [1200.0, 840.0]


def test_a_first_run_reports_the_unset_sentinel(data_dir):
    """(0, 0) means "never parked", which the launcher reads as its default."""
    assert AmbientPresence().state()["bubble_position"] == [0.0, 0.0]


def test_a_corrupt_position_file_degrades_to_first_run(data_dir):
    """A bad file must not strand her somewhere with no pixels."""
    target = data_dir / "companion" / "bubble_position.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json at all", encoding="utf-8")

    assert AmbientPresence().state()["bubble_position"] == [0.0, 0.0]


def test_persistence_failure_never_breaks_the_drag(presence, monkeypatch):
    """Dragging her must keep working when the disk does not.

    The position is a convenience; refusing the move because it could not be
    written would make a broken disk look like a frozen bubble.
    """
    monkeypatch.setattr(
        "core.runtime.file_write_gateway.get_file_write_gateway",
        lambda: (_ for _ in ()).throw(RuntimeError("no gateway")),
    )

    assert presence.move_bubble(10.0, 20.0) == (10.0, 20.0)
    assert asyncio.run(presence.persist_bubble_position()) is False
    assert presence.state()["bubble_position"] == [10.0, 20.0]


# ────────────────────────────── the write does not run on the event loop

def test_the_disk_write_is_on_the_async_lane():
    """A sync fsync in a request handler once froze the live runtime 20 min.

    ``move_bubble`` is called from an async route, so the durable write is a
    separate coroutine the route awaits rather than something it does inline.
    """
    import inspect

    assert inspect.iscoroutinefunction(AmbientPresence.persist_bubble_position)
    assert not inspect.iscoroutinefunction(AmbientPresence.move_bubble)

    source = inspect.getsource(AmbientPresence.persist_bubble_position)
    assert "write_json_async" in source
    assert "ensure_directory_async" in source


# ─────────────────────────────────────── the launcher actually reads it back


def test_the_launcher_restores_the_position_before_showing_her():
    """The reader that never existed.

    Placing her and THEN moving her would slide her across the screen on every
    window close, so the fetch has to complete before orderFront.
    """
    source = _LAUNCHER.read_text(encoding="utf-8")

    assert "restoreBubbleOrigin" in source, (
        "the persisted position is still written and never read"
    )
    assert "bubble_position" in source

    showing = source.split("private func showBubble()", 1)[1][:3000]
    restore_at = showing.find("restoreBubbleOrigin")
    order_at = showing.find("orderFront")
    assert restore_at != -1 and order_at != -1
    assert restore_at < order_at, "she would appear in the corner, then jump"


def test_a_position_from_a_display_that_is_gone_is_clamped():
    """A saved spot on a detached monitor is somewhere with no pixels.

    Which is indistinguishable, to the person, from the bubble being broken.
    """
    source = _LAUNCHER.read_text(encoding="utf-8")

    assert "clampToScreen" in source


def test_an_unreachable_runtime_still_shows_the_bubble():
    """A lookup that fails must not mean no bubble at all."""
    source = _LAUNCHER.read_text(encoding="utf-8")
    restore = source.split("private func restoreBubbleOrigin", 1)[1][:1600]

    # Always calls back, including on the guard path, or showBubble never
    # reaches orderFront and she silently never appears.
    assert restore.count("done(nil)") >= 1
    assert "DispatchQueue.main.async { done(origin) }" in restore
