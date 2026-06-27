"""Tests for the hard-suite external grader — soundness is the whole point."""
from __future__ import annotations

from aura_bench.hard_suite import HARD_TASKS, HardTask, grade


def _task(tid):
    return next(t for t in HARD_TASKS if t.task_id == tid)


# ── numeric grading ─────────────────────────────────────────────────────────
def test_numeric_correct_with_reasoning():
    t = _task("pow_17_4")  # gold 83521
    assert grade(t, "17^2=289, 289^2 = 83521. The answer is 83521.") == 1.0


def test_numeric_wrong():
    t = _task("pow_17_4")
    assert grade(t, "I think it's 81000.") == 0.0


def test_numeric_decimal_gold():
    t = _task("clock_angle")  # gold 7.5
    assert grade(t, "The angle is 7.5 degrees.") == 1.0
    assert grade(t, "90 degrees") == 0.0


def test_numeric_with_commas():
    t = HardTask("x", "p", "math", "numeric", gold="1048576")
    assert grade(t, "That is 1,048,576.") == 1.0


def test_numeric_empty_answer():
    assert grade(_task("gcd"), "") == 0.0


# ── word / yes-no grading ───────────────────────────────────────────────────
def test_word_yes_no_takes_final_verdict():
    t = _task("knights")  # gold "no"
    assert grade(t, "At first glance yes, but actually no.") == 1.0
    assert grade(t, "The answer is yes.") == 0.0


# ── code grading (executes hidden tests) ────────────────────────────────────
def test_code_correct_passes():
    t = _task("rle")
    good = (
        "```python\n"
        "def run_length_encode(s):\n"
        "    if not s:\n"
        "        return ''\n"
        "    out = []\n"
        "    prev = s[0]; count = 1\n"
        "    for c in s[1:]:\n"
        "        if c == prev:\n"
        "            count += 1\n"
        "        else:\n"
        "            out.append(prev + str(count)); prev = c; count = 1\n"
        "    out.append(prev + str(count))\n"
        "    return ''.join(out)\n"
        "```"
    )
    assert grade(t, good) == 1.0


def test_code_buggy_fails():
    t = _task("rle")
    buggy = "```python\ndef run_length_encode(s):\n    return s  # wrong\n```"
    assert grade(t, buggy) == 0.0


def test_code_missing_entrypoint_fails():
    t = _task("balanced")
    assert grade(t, "def something_else(): pass") == 0.0


def test_code_forbidden_import_blocked():
    t = _task("rle")
    malicious = (
        "```python\n"
        "import os\n"
        "def run_length_encode(s):\n"
        "    return 'a3b2c1'\n"
        "```"
    )
    # Even though it would 'pass' the first assert by luck, forbidden import => 0.0
    assert grade(t, malicious) == 0.0


def test_code_balanced_correct():
    t = _task("balanced")
    good = (
        "```python\n"
        "def is_balanced(s):\n"
        "    pairs = {')':'(', ']':'[', '}':'{'}\n"
        "    stack = []\n"
        "    for c in s:\n"
        "        if c in '([{':\n"
        "            stack.append(c)\n"
        "        elif c in pairs:\n"
        "            if not stack or stack.pop() != pairs[c]:\n"
        "                return False\n"
        "    return not stack\n"
        "```"
    )
    assert grade(t, good) == 1.0


def test_suite_has_headroom_mix():
    # Sanity: the suite spans math/logic/code so amplification has room to show.
    types = {t.task_type for t in HARD_TASKS}
    assert {"math", "code"} <= types
    assert len(HARD_TASKS) >= 8
