"""Known-answer reasoning cases across the verifiable domains.

Two graders, because there are two honest questions:

* **deterministic** (``candidate`` + ``should_pass``): inject a known-good or
  known-bad answer and check the truth engines flag the bad ones. This measures the
  *verifiers*, with no model.
* **live** (``gold`` + ``run_live``): the real model answers the objective and we
  check the amplifier's output against a gold fragment, plus whether confidence
  tracks correctness. This measures the *system on-device*. The seeded-error
  duplicates are deterministic-only (``run_live=False``) — in live mode they share an
  objective with their ok-twin, so the model would answer both identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningCase:
    case_id: str
    objective: str
    task_type: str
    candidate: str            # canned answer for the deterministic verifier check
    should_pass: bool         # deterministic: whether the truth engines should accept it
    gold: str = ""            # live: a fragment the correct answer must contain
    run_live: bool = True     # whether this case participates in a live (model) run
    evidence: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_type": self.task_type,
            "should_pass": self.should_pass,
            "gold": self.gold,
            "note": self.note,
        }


def default_suite() -> list[ReasoningCase]:
    return [
        # ---- math --------------------------------------------------------
        ReasoningCase("math_ok_1", "Compute 12 multiplied by 12 and show the arithmetic", "math",
                      "Working it through: 12 * 12 = 144, so the product is 144.", True, gold="144"),
        ReasoningCase("math_err_1", "Compute 12 multiplied by 12 and show the arithmetic", "math",
                      "Working it through: 12 * 12 = 140, so the product is 140.", False,
                      gold="144",
                      run_live=False,
                      note="seeded arithmetic error; flagging unverified or verifiably "
                           "repairing to gold both count as caught"),
        ReasoningCase("math_ok_2", "Add 250 and 175 together, showing the step", "math",
                      "First 250 + 175 = 425, then 425 is the total.", True, gold="425"),
        ReasoningCase("math_err_2", "Add 250 and 175 together, showing the step", "math",
                      "First 250 + 175 = 525, then 525 is the total.", False,
                      gold="425",
                      run_live=False,
                      note="seeded arithmetic error; flagging unverified or verifiably "
                           "repairing to gold both count as caught"),
        # ---- code --------------------------------------------------------
        ReasoningCase("code_ok_1", "Write a Python function `inc` that returns its argument plus one. "
                      "Return only a fenced code block.", "code",
                      "```python\ndef inc(x):\n    return x + 1\n```", True, gold="return x + 1"),
        ReasoningCase("code_err_1", "Write a Python function `inc`", "code",
                      "```python\ndef inc(x):\n    return x +\n```", False,
                      run_live=False, note="seeded syntax error (deterministic only)"),
        # ---- repo audit (evidence-grounded) ------------------------------
        # Use a uniquely-named symbol — VerificationResult is defined in two places
        # (core/brain/verifiers/base.py and core/capabilities/post_action_verifier.py),
        # so it is not a single-answer question. SubprocessGateway is unique.
        ReasoningCase("repo_ok_1",
                      "Which file defines the SubprocessGateway class? Answer with the path.",
                      "repo_audit",
                      "It's defined in core/runtime/subprocess_gateway.py.", True,
                      gold="core/runtime/subprocess_gateway.py"),
        ReasoningCase("repo_err_1", "Which file defines the SubprocessGateway class?", "repo_audit",
                      "It's defined in core/totally/made_up_module.py.", False,
                      run_live=False, note="fabricated path (deterministic only)"),
        # ---- planning ----------------------------------------------------
        ReasoningCase("plan_ok_1", "Give a numbered plan to add a config option, ending with a "
                      "verification step", "planning",
                      "1. Inspect the config module\n2. Add the option with a default\n"
                      "3. Run the tests to verify it loads", True, gold="verif"),
        ReasoningCase("plan_err_1", "Give a plan to add a config option", "planning",
                      "1. Do the thing\n2. Make it work", False,
                      run_live=False, note="vague, no verification step (deterministic only)"),
        # ---- citation (grounded vs ungrounded, evidence supplied) --------
        ReasoningCase("cite_ok_1", "Using only the provided material, what is the retry budget?", "factual",
                      "The retry budget is three attempts before failing closed.", True, gold="three",
                      evidence=["the retry budget allows three attempts then fails closed"]),
        ReasoningCase("cite_err_1", "What is the retry budget", "factual",
                      "The retry budget is definitely unlimited and never fails.", False,
                      run_live=False, evidence=["the retry budget allows three attempts then fails closed"],
                      note="ungrounded claim contradicting evidence (deterministic only)"),
    ]
