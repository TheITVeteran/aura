"""Deterministic LLM lane for capability-delta harness self-tests.

The callable returned by ``make_deterministic_llm`` returns different answers
depending on which subsystems the profile reports as enabled. ``base_llm_only``
produces a deliberately weaker answer while ``full`` chains the profile signal
into a correct answer. This proves the harness math without claiming that a
local deterministic lane is external benchmark evidence.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from aura_bench.capability_delta.profiles import profile_by_name


_ARITH_PATTERN = re.compile(r"(-?\d+)\s*([+\-*])\s*(-?\d+)")


def _solve_arith(prompt: str) -> str:
    """Solve the simple ``a + b`` / ``a - b`` / ``a * b`` problems."""
    match = _ARITH_PATTERN.search(prompt)
    if match is None:
        return ""
    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    return ""


def make_deterministic_llm(*, base_accuracy: float = 0.3) -> Callable[[str, str], str]:
    """Return a deterministic lane whose answer quality depends on the profile."""

    def llm(prompt: str, profile_name: str) -> str:
        profile = profile_by_name(profile_name)
        truth = _solve_arith(prompt)
        if not truth:
            return "I don't know"
        if profile.name == "full":
            return truth

        task_bucket = abs(hash((prompt, profile_name))) % 1000 / 1000.0
        if profile.name == "base_llm_only":
            return truth if task_bucket < base_accuracy else str(int(truth) + 1)

        disabled_count = max(0, 8 - len(profile.enabled_subsystems))
        accuracy = max(base_accuracy, 1.0 - 0.1 * disabled_count)
        return truth if task_bucket < accuracy else str(int(truth) + 1)

    return llm
