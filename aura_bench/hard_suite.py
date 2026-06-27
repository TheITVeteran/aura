"""Hard reasoning suite + a SOUND external grader.

The delta only means something if (a) the base model genuinely fails some tasks
single-pass (headroom for amplification to show), and (b) grading is objective and
independent of the amplifier's own verifiers (no rubber-stamping).

So grading here does NOT call the in-amplifier verifier registry. It uses:
  * numeric/exact : extract the model's final answer and exact-match the gold.
  * code          : execute the model's function against hidden asserts in an
                    isolated, AST-screened, timeout-bounded subprocess.

Tasks are chosen to be error-prone for a 7B/32B single pass (big powers, trailing
zeros, modular arithmetic, edge-case code) but trivially checkable.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# Imports a grader-run script must never contain (defense-in-depth; tasks are pure).
_FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes", "requests"}


@dataclass
class HardTask:
    task_id: str
    prompt: str
    task_type: str          # "math" | "code" | "logic"
    grader: str             # "numeric" | "code"
    gold: str = ""          # for numeric/exact
    tests: list[str] = field(default_factory=list)   # for code (assert strings)
    entrypoint: str = ""    # for code (function name)


HARD_TASKS: tuple[HardTask, ...] = (
    # ── numeric: error-prone for small models, exact answers ──
    HardTask("pow_17_4", "What is 17 to the power of 4? Give only the final integer.", "math", "numeric", gold="83521"),
    HardTask("trailing_zeros_100", "How many trailing zeros does 100 factorial (100!) have? Give only the integer.", "math", "numeric", gold="24"),
    HardTask("gcd", "What is the greatest common divisor (GCD) of 1071 and 462? Give only the integer.", "math", "numeric", gold="21"),
    HardTask("mod", "What is 123456 mod 7? Give only the integer.", "math", "numeric", gold="4"),
    HardTask("clock_angle", "A clock shows 3:15. What is the angle in degrees between the hour and minute hands? Give only the number.", "math", "numeric", gold="7.5"),
    HardTask("primes_sum", "What is the sum of all prime numbers strictly below 20? Give only the integer.", "math", "numeric", gold="77"),
    # ── logic / word: unique exact answer ──
    HardTask("knights", "Knights always tell the truth, knaves always lie. A says 'I am a knave'. Is that possible? Answer with exactly one word: yes or no.", "logic", "numeric", gold="no"),
    HardTask("trains", "Two trains 300 km apart head toward each other at 70 km/h and 80 km/h. After how many hours do they meet? Give only the number.", "math", "numeric", gold="2"),
    # ── code: graded by executing hidden tests ──
    HardTask(
        "rle", "Write a Python function run_length_encode(s) that returns the run-length encoding as a string, e.g. 'aaabbc' -> 'a3b2c1'. Single chars still get a count.",
        "code", "code", entrypoint="run_length_encode",
        tests=[
            "assert run_length_encode('aaabbc') == 'a3b2c1'",
            "assert run_length_encode('') == ''",
            "assert run_length_encode('x') == 'x1'",
            "assert run_length_encode('aabbaa') == 'a2b2a2'",
        ],
    ),
    HardTask(
        "balanced", "Write a Python function is_balanced(s) that returns True iff the brackets in s — (), [], {} — are correctly matched and nested. Ignore other characters.",
        "code", "code", entrypoint="is_balanced",
        tests=[
            "assert is_balanced('(a[b]{c})') is True",
            "assert is_balanced('([)]') is False",
            "assert is_balanced('') is True",
            "assert is_balanced('(((') is False",
            "assert is_balanced('{[()]}') is True",
        ],
    ),
    HardTask(
        "roman", "Write a Python function int_to_roman(n) converting an integer 1..3999 to a Roman numeral string.",
        "code", "code", entrypoint="int_to_roman",
        tests=[
            "assert int_to_roman(4) == 'IV'",
            "assert int_to_roman(9) == 'IX'",
            "assert int_to_roman(58) == 'LVIII'",
            "assert int_to_roman(1994) == 'MCMXCIV'",
            "assert int_to_roman(3888) == 'MMMDCCCLXXXVIII'",
        ],
    ),
)


def _extract_code_block(answer: str) -> str:
    fences = re.findall(r"```(?:python|py)?\s*(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    if fences:
        # Prefer the longest fenced block (usually the full function).
        return max(fences, key=len).strip()
    return answer.strip()


def _ast_is_safe(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] in _FORBIDDEN_IMPORTS for a in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _FORBIDDEN_IMPORTS:
                return False
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "__import__", "compile", "open"}:
                return False
    return True


def _grade_code(answer: str, task: HardTask, *, timeout: float = 10.0) -> float:
    code = _extract_code_block(answer)
    if task.entrypoint and task.entrypoint not in code:
        return 0.0
    if not _ast_is_safe(code):
        return 0.0
    script = code + "\n\n" + "\n".join(task.tests) + "\nprint('ALL_TESTS_PASSED')\n"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "grade.py"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(path)],
                capture_output=True, text=True, timeout=timeout, cwd=d,
            )
        except (subprocess.TimeoutExpired, OSError):
            return 0.0
    return 1.0 if proc.returncode == 0 and "ALL_TESTS_PASSED" in proc.stdout else 0.0


def _grade_numeric(answer: str, gold: str) -> float:
    gold = gold.strip()
    ans = str(answer or "").strip()
    if not ans:
        return 0.0
    # Word answers (yes/no): require the gold word, and not its negation right after.
    if not _NUM_RE.search(gold):
        low = ans.lower()
        glow = gold.lower()
        # Take the last yes/no token as the model's final verdict.
        verdicts = re.findall(r"\b(yes|no)\b", low)
        if verdicts:
            return 1.0 if verdicts[-1] == glow else 0.0
        return 1.0 if re.search(rf"\b{re.escape(glow)}\b", low) else 0.0
    # Numeric: compare the model's final number to gold (tolerant of commas/decimals).
    nums = [n.replace(",", "") for n in _NUM_RE.findall(ans)]
    if not nums:
        return 0.0
    try:
        gold_v = float(gold.replace(",", ""))
    except ValueError:
        return 0.0
    for n in nums:
        try:
            if abs(float(n) - gold_v) < 1e-6:
                return 1.0
        except ValueError:
            continue
    return 0.0


def grade(task: HardTask, answer: str) -> float:
    """Objective score in [0,1] — independent of the amplifier's own verifiers."""
    if task.grader == "code":
        return _grade_code(answer, task)
    return _grade_numeric(answer, task.gold)
