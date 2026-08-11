"""Asked what is in a directory, look — do not compose an answer.

LIVE, 2026-08-10, three runs of one request:

    "Count how many .py files are in
     /Users/bryan/.aura/live-source/core/introspection, then write that number
     and the file names into ~/Documents/aura_probe_count.txt."

Run 1: never reached a tool. "There are 3 .py files ... 1. introspection.py
2. self_assessment.py 3. system_monitoring.py. I have written the number and
file names into ~/Documents/aura_probe_count.txt." The directory holds 9, none
of those names exist, and no file was created.

Run 2, after routing was fixed: the plan aimed the WRITE at the source
directory and was refused by the artifact-root guard.

Run 3, after the destination fix: the file landed in the right place
containing "Number of .py files: 0 / (No files found)". A correct destination
holding a measurement nobody took.

The remaining gap was structural — there was no read action at all. Every
desktop action wrote, so a question about the contents of a real path could
only ever be answered by generation. list_directory is that action, bounded by
a readable-roots set that is deliberately wider than the writable one (her own
source tree, because source proprioception is a capability she is meant to
have) and narrower than the filesystem.

The count in the file is now the count that was read, via the step-reference
tokens that already existed for exactly this.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.skills.computer_use import ComputerUseSkill
from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep

OBJECTIVE = (
    "Count how many .py files are in /Users/bryan/.aura/live-source/core/introspection, "
    "then write that number and the file names into ~/Documents/aura_probe_count.txt."
)


def _skill() -> ComputerUseSkill:
    return ComputerUseSkill.__new__(ComputerUseSkill)


# ── The read action ────────────────────────────────────────────────────────

def test_a_real_directory_is_counted_correctly() -> None:
    """The measurement she could not take.

    Built inside ~/Documents rather than pytest's tmp_path, because tmp_path
    lives under /private/tmp and is correctly outside the readable roots —
    a fixture directory is not a reason to widen them.
    """
    import shutil

    scratch = Path.home() / "Documents" / ".aura_list_directory_probe"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        for name in ("a.py", "b.py", "notes.txt"):
            (scratch / name).write_text("x", encoding="utf-8")

        result = _skill()._list_directory(
            json.dumps({"path": str(scratch), "pattern": "*.py"})
        )

        assert result["ok"] is True, result.get("error")
        assert result["count"] == 2
        assert result["names"] == ["a.py", "b.py"]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_a_directory_outside_the_readable_roots_is_refused(tmp_path: Path) -> None:
    """pytest's tmp_path is under /private/tmp, and that is correct to refuse."""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")

    result = _skill()._list_directory(json.dumps({"path": str(tmp_path)}))

    assert result["ok"] is False
    assert "readable roots" in str(result["error"])


def test_her_own_source_tree_is_readable() -> None:
    """Reading her own code is the case that started this."""
    result = _skill()._list_directory(
        json.dumps({
            "path": "/Users/bryan/.aura/live-source/core/introspection",
            "pattern": "*.py",
        })
    )

    assert result["ok"] is True
    assert result["count"] >= 8
    assert "self_evidence.py" in result["names"]


@pytest.mark.parametrize("path", ["/etc", "/", "/usr/bin"])
def test_the_rest_of_the_filesystem_is_refused(path: str) -> None:
    """Readable is wider than writable, not unbounded."""
    result = _skill()._list_directory(json.dumps({"path": path}))

    assert result["ok"] is False
    assert "readable roots" in str(result["error"])


def test_a_symlink_out_of_the_roots_is_judged_by_where_it_lands() -> None:
    """Same defence as the write guard: /etc resolves to /private/etc."""
    result = _skill()._list_directory(json.dumps({"path": "/etc"}))

    assert result["ok"] is False
    assert "resolves to" in str(result["error"])


def test_a_file_is_not_a_directory() -> None:
    result = _skill()._list_directory(
        json.dumps({"path": "/Users/bryan/.aura/live-source/CLAUDE.md"})
    )

    assert result["ok"] is False
    assert "Not a directory" in str(result["error"])


# ── The plan ───────────────────────────────────────────────────────────────

def test_the_request_plans_a_read_of_the_source_directory() -> None:
    step = DesktopTaskSkill._directory_read_step(
        OBJECTIVE, skip="~/Documents/aura_probe_count.txt"
    )

    assert step is not None
    assert step.action == "list_directory"
    target = json.loads(step.target)
    assert target["path"] == "/Users/bryan/.aura/live-source/core/introspection"
    assert target["pattern"] == "*.py"


def test_the_write_destination_is_never_read_as_the_source() -> None:
    """Confusing the two is what aimed a write at her own source tree."""
    step = DesktopTaskSkill._directory_read_step(
        OBJECTIVE, skip="/Users/bryan/.aura/live-source/core/introspection"
    )

    if step is not None:
        assert json.loads(step.target)["path"] != (
            "/Users/bryan/.aura/live-source/core/introspection"
        )


def test_a_plain_write_plans_no_read() -> None:
    assert DesktopTaskSkill._directory_read_step(
        "write hello into ~/Documents/x.txt", skip="~/Documents/x.txt"
    ) is None


def test_list_directory_is_an_allowed_action() -> None:
    from core.runtime.desktop_task_contract import DESKTOP_TASK_ALLOWED_ACTIONS

    assert "list_directory" in DESKTOP_TASK_ALLOWED_ACTIONS


# ── The number written is the number read ──────────────────────────────────

def test_the_written_content_comes_from_the_read() -> None:
    receipts = [{
        "index": 0,
        "action": "list_directory",
        "ok": True,
        "effect_verified": True,
        "result": {
            "ok": True,
            "count": 9,
            "names": ["a.py", "b.py"],
            "effect_verified": True,
        },
    }]
    step = DesktopTaskStep(
        action="write_text_file",
        target=json.dumps({
            "path": "~/Documents/out.txt",
            "content": "Count: {{last.result.count}}\nNames: {{last.result.names}}",
            "overwrite": True,
        }),
        reason="r",
        expect="e",
        critical=True,
    )

    ok, resolved, error = DesktopTaskSkill._resolve_step_target(step, receipts)

    assert ok is True, error
    content = json.loads(resolved.target)["content"]
    assert "Count: 9" in content
    assert "a.py" in content


def test_an_unverified_read_does_not_get_written_up_as_fact() -> None:
    """A failed read must not become a confident zero in a file."""
    receipts = [{
        "index": 0,
        "action": "list_directory",
        "ok": False,
        "effect_verified": False,
        "result": {"ok": False, "error": "Not a directory"},
    }]
    step = DesktopTaskStep(
        action="write_text_file",
        target=json.dumps({
            "path": "~/Documents/out.txt",
            "content": "Count: {{last.result.count}}",
            "overwrite": True,
        }),
        reason="r",
        expect="e",
        critical=True,
    )

    ok, _resolved, error = DesktopTaskSkill._resolve_step_target(step, receipts)

    assert ok is False
    assert "did not verify" in error


# ── A bad declared plan must not be a dead turn ────────────────────────────

def test_an_invalid_declared_plan_falls_back_to_heuristic_planning() -> None:
    """LIVE: the model named an action that does not exist and the turn died.

        "Structured desktop plan contains an invalid or unsupported step.
         Completed 0/0 steps."

    A working heuristic plan for the same objective was sitting directly behind
    that return. A malformed declared plan is a reason to plan differently, not
    a reason to do nothing.
    """
    import inspect

    source = inspect.getsource(DesktopTaskSkill)
    marker = "plan_error = self._declared_plan_validation_error(task_context)"
    assert marker in source
    window = source[source.find(marker) : source.find(marker) + 1400]

    # The heuristic plan is computed BEFORE deciding to give up ...
    assert window.find("_steps_with_provenance_from_context") < window.find(
        '"status": "invalid_desktop_task_plan"'
    )
    # ... and giving up now requires that it produced nothing either.
    assert "if plan_error and not steps:" in window


def test_the_objective_planner_is_the_fallback_that_can_still_answer() -> None:
    """The context planner reads the payload that was malformed in the first place.

    LIVE: the first fallback attempt used _steps_with_provenance_from_context,
    which derives steps from the model's own context — the very thing that had
    just failed validation — so it returned nothing and the turn died again
    with the identical message. The objective planner works from the request
    text and is the one with an answer.
    """
    import inspect

    source = inspect.getsource(DesktopTaskSkill)
    marker = "plan_error = self._declared_plan_validation_error(task_context)"
    window = source[source.find(marker) : source.find(marker) + 1400]

    assert "_derive_steps_from_objective(objective, task_context)" in window
    assert window.find("_derive_steps_from_objective") < window.find(
        '"status": "invalid_desktop_task_plan"'
    )


def test_the_objective_planner_plans_read_then_write() -> None:
    skill = DesktopTaskSkill.__new__(DesktopTaskSkill)

    steps = skill._derive_steps_from_objective(OBJECTIVE, {})

    assert [step.action for step in steps] == ["list_directory", "write_text_file"]
    assert json.loads(steps[0].target)["path"] == (
        "/Users/bryan/.aura/live-source/core/introspection"
    )
    assert json.loads(steps[1].target)["path"] == "~/Documents/aura_probe_count.txt"


def test_a_read_verifies_on_the_reading_itself() -> None:
    """LIVE: "unsupported effect evidence for desktop action list_directory".

    Every prior action changed the world, so effect verification only knew how
    to confirm changes. A read's effect IS the reading, and without a branch
    saying so, a directory that had been read correctly reported 0/2 steps and
    the write that depended on it never ran.
    """
    step = DesktopTaskStep(
        action="list_directory",
        target=json.dumps({"path": "/Users/bryan/.aura/live-source/core/introspection"}),
        reason="r",
        expect="e",
        critical=True,
    )
    verified, evidence = DesktopTaskSkill._verify_step_effect(
        step,
        {
            "ok": True,
            "path": "/Users/bryan/.aura/live-source/core/introspection",
            "pattern": "*.py",
            "count": 9,
            "names": ["a.py"],
        },
    )

    assert verified is True
    assert "count=9" in evidence


def test_a_failed_read_does_not_verify() -> None:
    step = DesktopTaskStep(
        action="list_directory",
        target=json.dumps({"path": "/nope"}),
        reason="r",
        expect="e",
        critical=True,
    )
    verified, evidence = DesktopTaskSkill._verify_step_effect(
        step, {"ok": False, "error": "Not a directory"}
    )

    assert verified is False
    assert "Not a directory" in evidence
