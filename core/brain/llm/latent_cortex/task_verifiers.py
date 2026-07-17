"""Task-typed correctness verification INSIDE the live episode.

Closes the spec's central inference-time gap: "more stable internal thought
can still be confidently wrong." Until now the live route selected branches
by convergence quality and accepted latent-opt steps by proxy loss — internal
signals that measure stability, not truth. This module gives the episode a
deterministic, worker-safe verifier so branch selection and hill-climbing are
guided by CHECKED properties of candidate answers:

- **Arithmetic claims** — every "a op b = c" the candidate asserts is
  recomputed exactly; confidently wrong arithmetic is penalized per claim.
- **Code blocks** — fenced Python must compile (syntax truth, no execution:
  running model-authored code stays outside the worker, in the sandboxed
  service-side engines); other fenced blocks must be structurally balanced.
- **Facet coverage** — the SAME request_facets definition the product gate
  judges by scores whether the candidate addresses what was asked.
- **Objective grounding** — lexical overlap with the request's key terms,
  so a fluent non-answer scores below a grounded one.

Scores are deterministic, bounded [0, 1], and receipted with per-check
evidence, so a winning branch carries WHY it won ("passed 3/3 arithmetic
claims; python compiles") — not "converged prettier". No network, no
subprocess, no model calls: safe at any point inside the worker episode.
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Any

from core.brain.llm.latent_cortex.output_quality import request_facets

logger = logging.getLogger("Aura.LatentCortex.TaskVerifiers")

TASK_VERIFIER_SCHEMA = "aura.latent_task_verifier.v1"

# The trailing guard rejects decimal continuations ("= 40.5") without
# rejecting sentence-final claims ("= 40."). The leading guard keeps the
# first operand from starting mid-number or mid-decimal.
_ARITH_CLAIM_RE = re.compile(
    r"(?<![\d.])(-?\d{1,12})\s*([+\-*/x×])\s*(-?\d{1,12})\s*=\s*(-?\d{1,12})(?!\d)(?!\.\d)"
)
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
_ANSWER_FACET_HINTS = {
    "compare": re.compile(
        r"\b(?:whereas|while|unlike|versus|compared|by\s+contrast|in\s+contrast)\b", re.I
    ),
    "select": re.compile(
        r"\b(?:choose|recommend|prefer|stronger|best|should\s+(?:use|choose|adopt)|the\s+winner)\b",
        re.I,
    ),
    "verify": re.compile(
        r"\b(?:verify|test|assert|inject|simulate|fault|cancel|timeout|restart|invariant|receipt)\w*\b",
        re.I,
    ),
    "explain": re.compile(
        r"\b(?:because|therefore|thus|so\s+that|leads?\s+to|prevents?|causes?|ensures?)\b", re.I
    ),
    "enumerate": re.compile(r"(?:^\s*(?:[-*+]|\d+[.)])\s+\S)", re.M),
}


def check_arithmetic_claims(text: str) -> dict[str, Any]:
    """Recompute every explicit arithmetic claim in the candidate."""
    checked = passed = 0
    failures: list[str] = []
    for match in _ARITH_CLAIM_RE.finditer(text or ""):
        a, op, b, claimed = match.groups()
        try:
            a_v, b_v, claimed_v = int(a), int(b), int(claimed)
        except ValueError:
            continue
        if op in {"x", "×"}:
            op = "*"
        if op == "/":
            if b_v == 0 or a_v % b_v != 0:
                continue  # non-integer division claims are not judged here
            actual = a_v // b_v
        elif op == "+":
            actual = a_v + b_v
        elif op == "-":
            actual = a_v - b_v
        else:
            actual = a_v * b_v
        checked += 1
        if actual == claimed_v:
            passed += 1
        elif len(failures) < 8:
            failures.append(f"{a_v}{op}{b_v}={claimed_v} (actual {actual})")
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "score": (passed / checked) if checked else None,
    }


def check_code_blocks(text: str) -> dict[str, Any]:
    """Syntax-verify fenced code. Python must parse; others must balance."""
    checked = passed = 0
    failures: list[str] = []
    for match in _FENCE_RE.finditer(text or ""):
        language = (match.group(1) or "").strip().lower()
        body = match.group(2)
        if not body.strip():
            continue
        checked += 1
        if language in {"python", "py", ""} and not language.startswith("json"):
            try:
                ast.parse(body)
                passed += 1
            except SyntaxError as exc:
                if len(failures) < 8:
                    failures.append(f"python_syntax:{exc.lineno}:{exc.msg}")
        else:
            balanced = all(
                body.count(open_ch) == body.count(close_ch)
                for open_ch, close_ch in (("{", "}"), ("(", ")"), ("[", "]"))
            )
            if balanced:
                passed += 1
            elif len(failures) < 8:
                failures.append(f"{language or 'unknown'}:unbalanced_brackets")
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "score": (passed / checked) if checked else None,
    }


def check_facet_coverage(text: str, objective: str) -> dict[str, Any]:
    requested = request_facets(objective)
    satisfied = [
        name for name in requested if _ANSWER_FACET_HINTS[name].search(text or "")
    ]
    return {
        "requested": requested,
        "satisfied": satisfied,
        "score": (len(satisfied) / len(requested)) if requested else None,
    }


def check_objective_grounding(text: str, objective: str) -> dict[str, Any]:
    objective_terms = {word.lower() for word in _WORD_RE.findall(objective or "")}
    if not objective_terms:
        return {"matched": 0, "of": 0, "score": None}
    answer_terms = {word.lower() for word in _WORD_RE.findall(text or "")}
    matched = len(objective_terms & answer_terms)
    return {
        "matched": matched,
        "of": len(objective_terms),
        "score": min(1.0, matched / max(4, min(len(objective_terms), 16))),
    }


class EpisodeTaskVerifier:
    """Deterministic candidate scorer for one episode's objective.

    Instances are callables suitable for ``LatentCortexEngine.reason``'s
    ``verifier`` argument. Every scored candidate leaves an evidence row so
    the receipt can prove WHY the winner won. Weights renormalize over the
    checks that were actually applicable (no code in the answer ⇒ the code
    check neither helps nor hurts).
    """

    _WEIGHTS = {
        "arithmetic": 0.35,
        "code": 0.25,
        "facets": 0.25,
        "grounding": 0.15,
    }

    def __init__(self, objective: str) -> None:
        self.objective = str(objective or "")
        self.evaluations: list[dict[str, Any]] = []

    def evaluate(self, text: str) -> dict[str, Any]:
        checks = {
            "arithmetic": check_arithmetic_claims(text),
            "code": check_code_blocks(text),
            "facets": check_facet_coverage(text, self.objective),
            "grounding": check_objective_grounding(text, self.objective),
        }
        weighted = total_weight = 0.0
        for name, result in checks.items():
            score = result.get("score")
            if score is None:
                continue
            weighted += self._WEIGHTS[name] * float(score)
            total_weight += self._WEIGHTS[name]
        # A candidate exercising no verifiable surface scores a neutral 0.5:
        # verifiability itself must not be punished, but it earns nothing.
        score = (weighted / total_weight) if total_weight > 0 else 0.5
        row = {
            "schema": TASK_VERIFIER_SCHEMA,
            "score": round(score, 6),
            "applicable_checks": [
                name for name, result in checks.items() if result.get("score") is not None
            ],
            "checks": checks,
            "text_chars": len(text or ""),
        }
        self.evaluations.append(row)
        return row

    def __call__(self, text: str) -> float:
        return float(self.evaluate(text)["score"])

    def to_receipt(self) -> dict[str, Any]:
        """Bounded evidence: every evaluation's score + the best row's why."""
        if not self.evaluations:
            return {"schema": TASK_VERIFIER_SCHEMA, "evaluations": 0}
        best = max(self.evaluations, key=lambda row: row["score"])
        return {
            "schema": TASK_VERIFIER_SCHEMA,
            "evaluations": len(self.evaluations),
            "score_trail": [row["score"] for row in self.evaluations[:32]],
            "best_score": best["score"],
            "best_applicable_checks": list(best["applicable_checks"]),
            "best_failures": {
                name: list(result.get("failures") or [])
                for name, result in best["checks"].items()
                if result.get("failures")
            },
        }


__all__ = [
    "EpisodeTaskVerifier",
    "TASK_VERIFIER_SCHEMA",
    "check_arithmetic_claims",
    "check_code_blocks",
    "check_facet_coverage",
    "check_objective_grounding",
]
