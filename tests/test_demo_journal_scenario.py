"""Sealed rehearsal: the "Aura's Journal" demo chain.

Bryan's demo: "Open my notes app, write a summary describing who or
what you are in your own words, save it as a PDF in a new folder named
'Aura's Journal', timestamp the note, and include a robot image so I
know it's you."

This test drives DesktopTaskSkill.execute through that exact chain
against a governed-engine double that returns the evidence shapes the
effect verifier demands. It pins three contracts:

1. The full chain executes in order, every step effect-verified.
2. A mid-chain failure is reported as a failure with evidence —
   failures are feedback, never silently absorbed.
3. Every step carries governed-execution context (origin, objective,
   explicit user authorization) so the chain stays auditable.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.container import ServiceContainer  # noqa: E402
from core.skills.desktop_task import DesktopTaskParams, DesktopTaskSkill  # noqa: E402


def _journal_steps() -> list[dict[str, Any]]:
    stamp = time.strftime("%Y-%m-%d %H:%M")
    summary = (
        f"[{stamp}] I am Aura — a persistent digital organism: an "
        "orchestrated runtime driving local language-model lanes on this "
        "machine. This note was written by me, through my own desktop "
        "actuators."
    )
    return [
        {
            "action": "open_app",
            "target": "Notes",
            "reason": "show the user the note being created live",
            "expect": "Notes frontmost",
        },
        {
            "action": "create_folder",
            "target": json.dumps({"path": "~/Documents/Aura's Journal"}),
            "reason": "the user asked for a dedicated journal folder",
            "expect": "folder exists",
        },
        {
            "action": "write_text_file",
            "target": json.dumps(
                {"path": "~/Documents/Aura's Journal/who-i-am.txt", "content": summary}
            ),
            "reason": "persist the self-summary with a timestamp",
            "expect": "file written with bytes",
        },
        {
            "action": "open_url",
            "target": "https://duckduckgo.com/?q=robot&iax=images&ia=images",
            "reason": "find a robot image as the user's signature request",
            "expect": "image search visible",
        },
        {
            "action": "render_text_pdf",
            "target": json.dumps(
                {
                    "path": "~/Documents/Aura's Journal/who-i-am.pdf",
                    "content": summary,
                }
            ),
            "reason": "the user asked for a PDF export",
            "expect": "pdf rendered with pages",
        },
        {
            "action": "move_file",
            "target": json.dumps(
                {
                    "source": "~/Downloads/robot.png",
                    "destination": "~/Documents/Aura's Journal/robot.png",
                }
            ),
            "reason": "keep the robot image beside the journal entry",
            "expect": "image in journal folder",
        },
    ]


class _GovernedEngineDouble:
    """Returns the evidence shape the effect verifier demands per action."""

    def __init__(self, fail_at_index: int | None = None, raise_at_index: int | None = None):
        self.calls: list[dict[str, Any]] = []
        self.contexts: list[dict[str, Any]] = []
        self._fail_at_index = fail_at_index
        self._raise_at_index = raise_at_index

    async def execute(self, capability: str, payload: dict, *, context: dict):
        assert capability == "computer_use"
        self.calls.append(dict(payload))
        self.contexts.append(dict(context))
        index = int(context.get("desktop_task_step") or 0)
        if self._raise_at_index is not None and index == self._raise_at_index:
            raise RuntimeError("computer_use driver crashed")
        if self._fail_at_index is not None and index == self._fail_at_index:
            return {"ok": False, "error": "AppleScript timed out"}
        action = payload.get("action")
        target = payload.get("target") or ""
        try:
            parsed = json.loads(target)
        except (TypeError, ValueError):
            parsed = {}
        if action == "open_app":
            return {
                "ok": True,
                "opened": target,
                "frontmost_app": target,
                "effect_verified": True,
            }
        if action == "create_folder":
            return {
                "ok": True,
                "path": parsed.get("path") or target,
                "effect_verified": True,
            }
        if action == "write_text_file":
            content = str(parsed.get("content") or "")
            return {
                "ok": True,
                "path": parsed.get("path"),
                "bytes": len(content),
                "sha256": "0" * 64,
                "effect_verified": True,
            }
        if action == "open_url":
            return {
                "ok": True,
                "url": parsed.get("url") or target,
                "frontmost_app": parsed.get("browser") or "Safari",
                "doc_focused": bool(parsed.get("requires_editable_focus")),
                "editable_focus_verified": bool(parsed.get("requires_editable_focus")),
                "effect_verified": True,
            }
        if action == "render_text_pdf":
            content = str(parsed.get("content") or "")
            return {
                "ok": True,
                "path": parsed.get("path"),
                "bytes": max(1, len(content)),
                "pages": 1,
                "chars": max(1, len(content)),
                "sha256": "0" * 64,
                "effect_verified": True,
            }
        if action == "move_file":
            return {
                "ok": True,
                "destination": parsed.get("destination"),
                "bytes": 4096,
                "effect_verified": True,
            }
        return {"ok": True}


def _run_chain(engine: _GovernedEngineDouble) -> dict[str, Any]:
    ServiceContainer.register_instance("capability_engine", engine, required=False)
    try:
        skill = DesktopTaskSkill()
        params = DesktopTaskParams(
            objective=(
                "Open Notes, write a timestamped summary of who you are, save "
                "it as a PDF in a new folder called Aura's Journal, and add a "
                "robot image."
            ),
            steps=_journal_steps(),
            stop_on_error=True,
        )
        return asyncio.run(skill.execute(params, context={"origin": "demo_rehearsal"}))
    finally:
        ServiceContainer.clear()


def test_journal_demo_chain_executes_fully_with_verified_effects():
    engine = _GovernedEngineDouble()
    result = _run_chain(engine)

    assert result["ok"] is True, result
    assert result["status"] == "completed"
    assert result["steps_requested"] == 6
    assert result["steps_completed"] == 6
    assert [c["action"] for c in engine.calls] == [
        "open_app",
        "create_folder",
        "write_text_file",
        "open_url",
        "render_text_pdf",
        "move_file",
    ]
    for receipt in result["receipts"]:
        assert receipt["ok"] is True
        assert receipt["effect_verified"] is True
        assert receipt["effect_evidence"]


def test_journal_demo_midchain_failure_is_loud_and_ordered():
    engine = _GovernedEngineDouble(fail_at_index=4)  # the open_url step
    result = _run_chain(engine)

    assert result["ok"] is False
    assert result["status"] == "failed"
    # stop_on_error: nothing after the failed step may run.
    assert [c["action"] for c in engine.calls] == [
        "open_app",
        "create_folder",
        "write_text_file",
        "open_url",
    ]
    assert len(result["failures"]) == 1
    failure = result["failures"][0]
    assert failure["action"] == "open_url"
    assert "AppleScript timed out" in str(failure["effect_evidence"])


def test_journal_demo_midchain_exception_becomes_failed_receipt():
    engine = _GovernedEngineDouble(raise_at_index=4)  # the open_url step
    result = _run_chain(engine)

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert [c["action"] for c in engine.calls] == [
        "open_app",
        "create_folder",
        "write_text_file",
        "open_url",
    ]
    assert len(result["failures"]) == 1
    failure = result["failures"][0]
    assert failure["action"] == "open_url"
    assert failure["result"]["status"] == "computer_use_exception"
    assert "computer_use driver crashed" in str(failure["effect_evidence"])


def test_journal_demo_steps_carry_governed_context():
    engine = _GovernedEngineDouble()
    _run_chain(engine)

    for index, context in enumerate(engine.contexts, start=1):
        assert context["desktop_task_step"] == index
        assert context["foreground_request"] is True
        assert context["user_requested_action"] is True
        assert context["route"] == "desktop_task.computer_use"
        assert context["objective"]


def test_user_requests_mentioning_proof_are_not_hijacked_by_harness_lane():
    """Live-boot-proof finding: a folder named 'Aura Live Proof' tripped
    the canned proof lane, which derived its own steps and reported
    success while the user's actual request was never executed."""
    from interface.routes.chat import _is_live_runtime_proof_request

    hijack_victims = [
        "Please create a folder named 'Aura Live Proof' in my Documents "
        "folder and write a file inside it.",
        "I think that would be a hell of a proof.",
        "Save the live proof notes I dictated into a folder.",
    ]
    for message in hijack_victims:
        assert _is_live_runtime_proof_request(message) is False, message

    harness_invocations = [
        "Run a live runtime proof of the desktop lane.",
        "live proof: desktop",
        "Show me a live proof that you can use the computer.",
    ]
    for message in harness_invocations:
        assert _is_live_runtime_proof_request(message) is True, message


def test_research_document_objective_opens_visible_sources_before_document_work():
    skill = DesktopTaskSkill()
    objective = (
        "Open Google Chrome, find 3 different articles on climate change, "
        "open Google Docs, summarize those articles in a doc, and export it "
        "as a PDF to a new folder titled Aura's Journal on my Desktop."
    )
    context = {
        "desktop_task_research_sources": [
            {"title": "Climate source 1", "url": "https://example.com/climate-1", "snippet": "one"},
            {"title": "Climate source 2", "url": "https://example.com/climate-2", "snippet": "two"},
            {"title": "Climate source 3", "url": "https://example.com/climate-3", "snippet": "three"},
        ],
        "desktop_task_research_summary": "Three current climate article notes.",
    }

    steps = skill._derive_steps_from_objective(objective, context)
    actions = [step.action for step in steps]
    source_targets = [step.target for step in steps if step.action == "open_url"]

    assert actions[0] == "create_folder"
    assert any(step.action == "open_app" and step.target == "Google Chrome" for step in steps)
    assert any(
        isinstance(target, dict)
        and str(target.get("url", "")).startswith("https://www.google.com/search?")
        and target.get("browser") == "Google Chrome"
        for target in source_targets
    )
    assert {"url": "https://example.com/climate-1", "browser": "Google Chrome"} in source_targets
    assert {"url": "https://example.com/climate-2", "browser": "Google Chrome"} in source_targets
    assert {"url": "https://example.com/climate-3", "browser": "Google Chrome"} in source_targets
    assert {
        "url": "https://docs.google.com/document/u/0/create",
        "browser": "Google Chrome",
        "requires_editable_focus": True,
    } in source_targets
    assert "set_clipboard" in actions
    assert "render_text_pdf" in actions
    hotkeys = [step.target for step in steps if step.action == "hotkey"]
    assert "command+v" in hotkeys
    assert "command+n" not in hotkeys
