"""A failure with no cause is barely better than a silent one.

Asked twice, on two different days, to create a file on the Desktop, what
reached Bryan was:

    "I routed this through CognitiveEngine and the governed desktop task lane,
     but it did not complete: desktop_task reported failure without a cause
     (status=failed). Completed 0/1 steps."

The cause existed the whole time. Every failing receipt carries the step's
action, what it expected, the effect evidence, and the child result's own
error. None of it was lifted into the skill's ``error`` field, so BaseSkill
substituted its "reported failure without a cause" placeholder — which is a
correct description of the payload and useless to the person reading it. He
cannot act on it, she cannot explain it, and the surprise engine banks a
maximal-surprise learning signal carrying no information about what to do
differently.

Both failure returns that omitted it — the os_automation escalation and the
main step loop — now say which step failed and why.
"""
from __future__ import annotations

from pathlib import Path

from core.skills.desktop_task import DesktopTaskSkill

SOURCE = Path("core/skills/desktop_task.py")


def _cause(failures, objective: str = "") -> str:
    return DesktopTaskSkill._failure_cause(failures, objective=objective)


def test_a_child_errors_own_words_are_used() -> None:
    cause = _cause(
        [
            {
                "action": "write_text_file",
                "expect": "~/Desktop/aura_hello.txt exists",
                "result": {"ok": False, "error": "permission denied writing ~/Desktop"},
            }
        ]
    )
    assert "write_text_file failed" in cause
    assert "permission denied" in cause
    assert "~/Desktop/aura_hello.txt exists" in cause


def test_a_status_stands_in_when_there_is_no_error_string() -> None:
    cause = _cause([{"action": "open_app", "result": {"ok": False, "status": "app_not_found"}}])
    assert "open_app failed" in cause
    assert "app_not_found" in cause


def test_effect_evidence_is_used_when_the_child_said_nothing() -> None:
    cause = _cause(
        [
            {
                "action": "run_applescript",
                "effect_evidence": "window never became frontmost",
                "result": {"ok": False},
            }
        ]
    )
    assert "window never became frontmost" in cause


def test_an_unverified_effect_names_what_was_expected() -> None:
    cause = _cause([{"action": "move_file", "expect": "the file is on the Desktop"}])
    assert "move_file" in cause
    assert "the file is on the Desktop" in cause


def test_a_receipt_that_says_nothing_at_all_still_names_the_step() -> None:
    assert "click" in _cause([{"action": "click"}])


def test_no_failing_receipt_still_produces_something_actionable() -> None:
    cause = _cause([], objective="create a file on my Desktop called aura_hello.txt")
    assert "aura_hello.txt" in cause
    assert cause.strip()


def test_the_cause_is_bounded() -> None:
    cause = _cause([{"action": "x", "result": {"ok": False, "error": "y" * 5000}}])
    assert len(cause) <= 400


def test_both_failure_returns_carry_the_cause() -> None:
    """The two returns that produced "0/1 steps" with no error field."""
    src = SOURCE.read_text(encoding="utf-8")
    assert src.count('"error": self._failure_cause(') >= 1
    assert '"error": self._failure_cause(\n                        critical_failures or failures' in src


def test_a_successful_run_carries_no_error_key() -> None:
    """An `error` on a success is its own kind of lie."""
    src = SOURCE.read_text(encoding="utf-8")
    assert '**({} if ok else {"error": self._failure_cause([receipt], objective=objective)}),' in src
