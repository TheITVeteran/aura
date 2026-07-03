"""Deterministic artifact builder — always-openable outputs, no app dependency.

Pins Bryan's requirement: when Aura wants to SHOW something she builds a real,
openable artifact in a portable format even when no spreadsheet app exists and
the smart (task-engine) path is unavailable.
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path

from core.actuators import artifact_builder as ab


def _redirect_output(monkeypatch, tmp_path):
    monkeypatch.setattr(ab, "_output_dir", lambda: tmp_path)


def test_build_table_produces_openable_csv_and_html(monkeypatch, tmp_path):
    _redirect_output(monkeypatch, tmp_path)
    built = ab.build_table(
        [["Mon", "9am", "Standup"], ["Tue", "2pm", "Review"]],
        headers=["Day", "Time", "Task"],
        title="Work Schedule",
    )
    assert built.ok
    csv_path = Path(built.primary)
    assert csv_path.suffix == ".csv" and csv_path.exists()
    # Real, parseable CSV with header + data.
    with csv_path.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["Day", "Time", "Task"]
    assert rows[1] == ["Mon", "9am", "Standup"]
    # HTML companion opens in any browser.
    html_path = next(p for p in built.paths if p.endswith(".html"))
    html = Path(html_path).read_text()
    assert "<table>" in html and "Standup" in html


def test_build_table_escapes_special_characters(monkeypatch, tmp_path):
    _redirect_output(monkeypatch, tmp_path)
    built = ab.build_table([['a,b', 'quote"x', "line\nbreak"]], headers=["c1", "c2", "c3"])
    with Path(built.primary).open() as fh:
        rows = list(csv.reader(fh))
    # csv round-trips the escaped values intact.
    assert rows[1] == ['a,b', 'quote"x', "line\nbreak"]


def test_build_doc_and_program(monkeypatch, tmp_path):
    _redirect_output(monkeypatch, tmp_path)
    doc = ab.build_doc("## Heading\n- point one\n- point two", title="Plan")
    assert doc.ok and Path(doc.primary).suffix == ".html"
    assert "<h2>Heading</h2>" in Path(doc.primary).read_text()

    prog = ab.build_program("print('hi')", language="python")
    assert prog.ok and Path(prog.primary).suffix == ".py"
    assert "print('hi')" in Path(prog.primary).read_text()


def test_demonstrate_artifact_falls_back_to_deterministic_builder(monkeypatch, tmp_path):
    """When the task engine can't decompose, a real openable artifact is still
    produced and 'opened' — the always-succeeds floor."""
    _redirect_output(monkeypatch, tmp_path)
    monkeypatch.setattr(ab, "open_artifact", _fake_open := _make_recorder())

    from core.cognition.affordance_realizers import realize_demonstrate_artifact

    class _DeadEngine:
        calls = []

        async def execute_goal(self, *_a, **_k):
            _DeadEngine.calls.append(1)
            raise RuntimeError("no model available")

    monkeypatch.setattr(
        "core.agency.autonomous_task_engine.get_task_engine", lambda: _DeadEngine()
    )

    out = asyncio.run(
        realize_demonstrate_artifact(
            {"kind": "table", "spec": "weekly work schedule"},
            {"last_user_message": "I need to make a table for work"},
        )
    )
    assert out["ok"] is True
    assert out["artifact_kind"] == "table"
    assert Path(out["path"]).exists()
    assert "something like this" in out["spoken"].lower()


def _make_recorder():
    calls = []

    async def _recorder(path):
        calls.append(path)
        return True

    _recorder.calls = calls
    return _recorder
