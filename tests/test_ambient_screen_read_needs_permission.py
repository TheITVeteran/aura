"""Reading unprompted needs the same permission as speaking unprompted.

CP126 16e9dc54. The Spatial Empathy watcher captured the person's screen
whenever a global-workspace payload carried two matching strings
(`intent="seek_connection"`, `action="read_ambient_screen"`). The
workspace has no publisher identity, so nothing established who asked,
whether it was allowed, or that the capture was governed.

Reading the screen is the most invasive thing this process does, and
unlike a read the person asked for, nobody is present to see this one
happen. A quiet window means she is not observing either.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "core" / "agency" / "agency_core.py").read_text("utf-8")


def _handler_body() -> str:
    """The watcher's handler, bounded by the AST rather than a line window."""
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_spatial_empathy":
            return ast.get_source_segment(SOURCE, node) or ""
    raise AssertionError("the spatial empathy handler is gone")


def test_the_ambient_read_checks_the_quiet_window():
    body = _handler_body()
    assert "_proactivity_suppressed_now()" in body, (
        "an unprompted screen capture no longer asks whether she is allowed "
        "to act unprompted at all"
    )


def test_the_quiet_window_check_precedes_the_capture():
    body = _handler_body()
    assert body.index("_proactivity_suppressed_now()") < body.index("read_screen_text"), (
        "the screen is read before permission is checked"
    )


def test_the_capture_runs_inside_a_governed_scope():
    body = _handler_body()
    assert "local_internal_governed_scope(" in body, (
        "the capture reaches for the skill directly and leaves no receipt"
    )
    assert "agency_core.spatial_empathy_screen_read" in body


def test_the_governed_scope_wraps_the_read_itself():
    body = _handler_body()
    scope = body.index("local_internal_governed_scope(")
    read = body.index("read_screen_text")
    assert scope < read, "the scope is opened after the capture already happened"


def test_the_suppressed_path_reads_nothing():
    """Suppression must return, not merely log and continue."""
    body = _handler_body()
    suppressed = body.index("_proactivity_suppressed_now()")
    read = body.index("read_screen_text")
    between = body[suppressed:read]
    assert "return" in between, (
        "the quiet window is checked and then the screen is read anyway"
    )


def test_the_proactivity_gate_it_relies_on_still_fails_closed():
    """This reuses the speaking gate; that gate must stay fail-closed."""
    initiative = (ROOT / "core" / "brain" / "initiative_engine.py").read_text("utf-8")
    assert "FAILS CLOSED" in initiative
