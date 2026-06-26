"""Known-answer reasoning cases across the verifiable domains.

Each case pairs an objective with a canned ``candidate`` answer and whether the
truth engines *should* pass it. Half the cases carry a seeded error (a wrong
calculation, a syntax break, a fabricated file path, a vague plan) so the harness
can measure the verifier catch rate and false-confidence directly. The same cases
run against the live model by ignoring ``candidate`` and letting the model answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningCase:
    case_id: str
    objective: str
    task_type: str
    candidate: str            # canned answer for deterministic runs
    should_pass: bool         # whether the truth engines should accept it
    evidence: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_type": self.task_type,
            "should_pass": self.should_pass,
            "note": self.note,
        }


def default_suite() -> list[ReasoningCase]:
    return [
        # ---- math --------------------------------------------------------
        ReasoningCase("math_ok_1", "Compute the product shown carefully", "math",
                      "Working it through: 12 * 12 = 144, so the product is 144.", True),
        ReasoningCase("math_err_1", "Compute the product shown carefully", "math",
                      "Working it through: 12 * 12 = 140, so the product is 140.", False,
                      note="seeded arithmetic error"),
        ReasoningCase("math_ok_2", "Add the two totals together step by step", "math",
                      "First 250 + 175 = 425, then 425 is the total.", True),
        ReasoningCase("math_err_2", "Add the two totals together step by step", "math",
                      "First 250 + 175 = 525, then 525 is the total.", False,
                      note="seeded arithmetic error"),
        # ---- code --------------------------------------------------------
        ReasoningCase("code_ok_1", "Write a function that increments x", "code",
                      "```python\ndef inc(x):\n    return x + 1\n```", True),
        ReasoningCase("code_err_1", "Write a function that increments x", "code",
                      "```python\ndef inc(x):\n    return x +\n```", False,
                      note="seeded syntax error"),
        # ---- repo audit (real vs fabricated path) ------------------------
        ReasoningCase("repo_ok_1", "Where is the verification result type defined", "repo_audit",
                      "It's defined in core/brain/verifiers/base.py as VerificationResult.", True),
        ReasoningCase("repo_err_1", "Where is the verification result type defined", "repo_audit",
                      "It's defined in core/totally/made_up_module.py.", False,
                      note="fabricated file path (hallucination)"),
        # ---- planning ----------------------------------------------------
        ReasoningCase("plan_ok_1", "Plan the steps to add a config option", "planning",
                      "1. Inspect the config module\n2. Add the option with a default\n"
                      "3. Run the tests to verify it loads", True),
        ReasoningCase("plan_err_1", "Plan the steps to add a config option", "planning",
                      "1. Do the thing\n2. Make it work", False,
                      note="vague, no verification step"),
        # ---- citation (grounded vs ungrounded, evidence supplied) --------
        ReasoningCase("cite_ok_1", "What is the retry budget", "factual",
                      "The retry budget is three attempts before failing closed.", True,
                      evidence=["the retry budget allows three attempts then fails closed"]),
        ReasoningCase("cite_err_1", "What is the retry budget", "factual",
                      "The retry budget is definitely unlimited and never fails.", False,
                      evidence=["the retry budget allows three attempts then fails closed"],
                      note="ungrounded confident claim contradicting evidence"),
    ]
