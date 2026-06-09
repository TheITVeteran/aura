"""Planning generality: the desktop pipeline is plan-agnostic.

The journal-demo rehearsal proves one chain. These tests prove the
property Bryan actually asked for: any valid plan over the general
action vocabulary — tasks the executor has never seen, with different
shapes, lengths, and action mixes — executes through the same governed
engine with the same per-step effect verification. No notes-specific
or journal-specific pathway is involved.

Also pinned: when a structured model-emitted plan exists, the
pattern-based fallback derivation is NOT consulted. The heuristics are
a safety net, never the headline act.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.container import ServiceContainer  # noqa: E402
from core.skills.desktop_task import DesktopTaskParams, DesktopTaskSkill  # noqa: E402


class _GovernedEngineDouble:
    """Evidence-shaped responses for every action in the vocabulary."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def execute(self, capability: str, payload: dict, *, context: dict):
        assert capability == "computer_use"
        self.calls.append(dict(payload))
        action = payload.get("action")
        target = payload.get("target") or ""
        try:
            parsed = json.loads(target)
        except (TypeError, ValueError):
            parsed = {}
        if action == "open_app":
            return {"ok": True, "opened": target}
        if action == "open_url":
            return {"ok": True, "url": target}
        if action == "create_folder":
            return {"ok": True, "path": parsed.get("path") or target}
        if action == "write_text_file":
            content = str(parsed.get("content") or "")
            return {"ok": True, "path": parsed.get("path"), "bytes": len(content)}
        if action == "render_text_pdf":
            content = str(parsed.get("content") or "")
            return {
                "ok": True,
                "path": parsed.get("path"),
                "bytes": max(1, len(content)),
                "pages": 1 + len(content) // 1800,
                "chars": max(1, len(content)),
            }
        if action == "move_file":
            return {"ok": True, "destination": parsed.get("destination"), "bytes": 1024}
        if action == "set_clipboard":
            return {"ok": True, "chars": len(str(parsed.get("text") or target))}
        if action == "hotkey":
            return {"ok": True, "hotkey": target}
        if action == "wait":
            return {"ok": True, "seconds": float(parsed.get("seconds") or 1)}
        if action == "read_screen_text":
            return {"ok": True, "text": "Quarterly Report — draft v3"}
        if action == "run_applescript":
            return {"ok": True, "result": "done"}
        return {"ok": True}


def _execute(steps: list[dict[str, Any]], objective: str) -> tuple[dict, _GovernedEngineDouble]:
    engine = _GovernedEngineDouble()
    ServiceContainer.register_instance("capability_engine", engine, required=False)
    try:
        skill = DesktopTaskSkill()
        params = DesktopTaskParams(objective=objective, steps=steps, stop_on_error=True)
        result = asyncio.run(skill.execute(params, context={"origin": "generality_battery"}))
        return result, engine
    finally:
        ServiceContainer.clear()


def _assert_all_verified(result: dict, expected_steps: int) -> None:
    assert result["ok"] is True, result
    assert result["steps_requested"] == expected_steps
    assert result["steps_completed"] == expected_steps
    for receipt in result["receipts"]:
        assert receipt["effect_verified"] is True
        assert receipt["effect_evidence"]


def test_novel_task_research_dossier():
    """Browse three sources, capture notes, render a dossier PDF."""
    steps = [
        {"action": "open_url", "target": "https://example.org/fusion-news",
         "reason": "first source", "expect": "page open"},
        {"action": "open_url", "target": "https://example.org/iter-update",
         "reason": "second source", "expect": "page open"},
        {"action": "read_screen_text", "target": "",
         "reason": "capture visible findings", "expect": "text returned"},
        {"action": "create_folder",
         "target": json.dumps({"path": "~/Documents/Research/Fusion"}),
         "reason": "dossier home", "expect": "folder exists"},
        {"action": "write_text_file",
         "target": json.dumps({"path": "~/Documents/Research/Fusion/notes.txt",
                                "content": "Source notes: tokamak milestones."}),
         "reason": "persist notes", "expect": "file written"},
        {"action": "render_text_pdf",
         "target": json.dumps({"path": "~/Documents/Research/Fusion/dossier.pdf",
                                "content": "Fusion dossier summary." * 40}),
         "reason": "deliverable", "expect": "pdf rendered"},
    ]
    result, engine = _execute(steps, "Build me a fusion research dossier")
    _assert_all_verified(result, 6)
    assert [c["action"] for c in engine.calls] == [s["action"] for s in steps]


def test_novel_task_inbox_triage_shape():
    """Different shape: app + clipboard + hotkey automation mix."""
    steps = [
        {"action": "open_app", "target": "Mail",
         "reason": "user asked for inbox triage", "expect": "Mail frontmost"},
        {"action": "set_clipboard",
         "target": json.dumps({"text": "Following up — see attached summary."}),
         "reason": "stage the reply text", "expect": "clipboard set"},
        {"action": "hotkey", "target": "cmd+n",
         "reason": "new message", "expect": "compose window"},
        {"action": "wait", "target": json.dumps({"seconds": 1}),
         "reason": "let compose settle", "expect": "window ready"},
        {"action": "run_applescript",
         "target": 'tell application "Mail" to activate',
         "reason": "ensure focus", "expect": "Mail active"},
    ]
    result, engine = _execute(steps, "Help me triage my inbox")
    _assert_all_verified(result, 5)
    assert [c["action"] for c in engine.calls] == [s["action"] for s in steps]


def test_novel_task_long_archive_chain():
    """Length stress: a 12-step archive/reorganize chain, order preserved."""
    steps = []
    for quarter in ("Q1", "Q2", "Q3", "Q4"):
        steps.append(
            {"action": "create_folder",
             "target": json.dumps({"path": f"~/Documents/Archive/{quarter}"}),
             "reason": f"{quarter} archive home", "expect": "folder exists"}
        )
        steps.append(
            {"action": "move_file",
             "target": json.dumps({"source": f"~/Desktop/report-{quarter}.pdf",
                                    "destination": f"~/Documents/Archive/{quarter}/report.pdf"}),
             "reason": f"file {quarter} report", "expect": "file moved"}
        )
        steps.append(
            {"action": "write_text_file",
             "target": json.dumps({"path": f"~/Documents/Archive/{quarter}/manifest.txt",
                                    "content": f"{quarter} archived."}),
             "reason": "manifest", "expect": "file written"}
        )
    result, engine = _execute(steps, "Archive my quarterly reports")
    _assert_all_verified(result, 12)
    assert [c["action"] for c in engine.calls] == [s["action"] for s in steps]


def test_structured_plan_bypasses_pattern_fallbacks(monkeypatch):
    """When a model-emitted plan exists, heuristics must not be consulted."""
    skill = DesktopTaskSkill()
    consulted = {"derive": False}

    original_derive = skill._derive_steps_from_objective

    def _tracking_derive(*args, **kwargs):
        consulted["derive"] = True
        return original_derive(*args, **kwargs)

    monkeypatch.setattr(skill, "_derive_steps_from_objective", _tracking_derive)

    engine = _GovernedEngineDouble()
    ServiceContainer.register_instance("capability_engine", engine, required=False)
    try:
        params = DesktopTaskParams(
            objective="Take notes about robots in a new folder",
            steps=[
                {"action": "create_folder",
                 "target": json.dumps({"path": "~/Documents/Robots"}),
                 "reason": "explicit plan", "expect": "folder exists"},
            ],
            stop_on_error=True,
        )
        result = asyncio.run(skill.execute(params, context={}))
    finally:
        ServiceContainer.clear()

    assert result["ok"] is True
    assert consulted["derive"] is False, (
        "pattern-based fallback ran despite a structured plan — "
        "heuristics must stay a safety net"
    )
