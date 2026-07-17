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


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_CONTENT_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def _facet_cue_sentences(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Sentences containing a facet cue AND enough content to mean it.

    Anti-Goodhart: bare cue tokens ("Because. Whereas. I choose.") must not
    satisfy a facet — the model could learn to emit the keyword without the
    substance. A cue only counts inside a sentence of >= 6 words carrying
    >= 3 content words beyond the cue itself. Genuine paraphrased substance
    passes; keyword stuffing does not. The matched excerpts are receipted so
    facet claims stay auditable by held-out human grading.
    """
    sentences = []
    for sentence in _SENTENCE_SPLIT_RE.split(text or ""):
        match = pattern.search(sentence)
        if not match:
            continue
        words = sentence.split()
        if len(words) < 6:
            continue
        cue_tokens = {token.lower() for token in match.group(0).split()}
        content = [
            word
            for word in _CONTENT_WORD_RE.findall(sentence)
            if word.lower() not in cue_tokens
        ]
        if len(content) >= 3:
            sentences.append(sentence.strip()[:200])
    return sentences


def check_facet_coverage(text: str, objective: str) -> dict[str, Any]:
    requested = request_facets(objective)
    satisfied: list[str] = []
    unsupported_cues: list[str] = []
    excerpts: dict[str, str] = {}
    for name in requested:
        pattern = _ANSWER_FACET_HINTS[name]
        supported = _facet_cue_sentences(text or "", pattern)
        if supported:
            satisfied.append(name)
            excerpts[name] = supported[0]
        elif pattern.search(text or ""):
            unsupported_cues.append(name)
    return {
        "requested": requested,
        "satisfied": satisfied,
        "unsupported_cues": unsupported_cues,
        "excerpts": excerpts,
        "score": (len(satisfied) / len(requested)) if requested else None,
    }


def check_degeneracy(text: str) -> dict[str, Any]:
    """Deterministic degeneration factor in [0.5, 1.0] for longer candidates.

    Two Goodhart shapes reduce it: repetition loops (trigram diversity
    collapse — the CP105 live failure shape) and facet-cue stuffing (cue
    density far above natural prose). Returned as a multiplicative factor so
    a degenerate candidate cannot buy its score back with one correct sum.
    """
    words = (text or "").split()
    if len(words) < 30:
        return {"applicable": False, "factor": 1.0}
    trigrams = [" ".join(words[i : i + 3]).lower() for i in range(len(words) - 2)]
    diversity = len(set(trigrams)) / max(1, len(trigrams))
    severity = max(0.0, (0.5 - diversity) * 2.0) if diversity < 0.5 else 0.0
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]
    cue_hits = sum(
        len(pattern.findall(text or "")) for pattern in _ANSWER_FACET_HINTS.values()
    )
    cue_density = cue_hits / max(1, len(sentences))
    if cue_density > 1.5:
        severity += min(1.0, (cue_density - 1.5) / 2.0)
    factor = 1.0 - 0.5 * min(1.0, severity)
    return {
        "applicable": True,
        "factor": round(factor, 6),
        "trigram_diversity": round(diversity, 6),
        "cue_density": round(cue_density, 6),
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

    def __init__(
        self,
        objective: str,
        *,
        facet_reliability: dict[str, float] | None = None,
    ) -> None:
        self.objective = str(objective or "")
        self.evaluations: list[dict[str, Any]] = []
        # Held-out calibration: per-facet reliability learned from GRADED
        # verdicts (Verifier Foundry Wilson bounds, human ground truth). A
        # facet whose cue-detector humans keep overruling is muted — it
        # earns less when satisfied and demands less when requested — so
        # "add the word because" stops being a strategy the moment grading
        # evidence says the cue is hollow. Neutral (1.0) until measured.
        self.facet_reliability: dict[str, float] = {}
        for name, value in (facet_reliability or {}).items():
            if (
                name in _ANSWER_FACET_HINTS
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0.0 <= float(value) <= 1.0
            ):
                self.facet_reliability[name] = float(value)

    def _facet_weighted_score(self, facets: dict[str, Any]) -> float | None:
        requested = facets.get("requested") or []
        if not requested:
            return None
        satisfied = set(facets.get("satisfied") or [])
        total = sum(self.facet_reliability.get(name, 1.0) for name in requested)
        if total <= 0.0:
            return None
        earned = sum(
            self.facet_reliability.get(name, 1.0)
            for name in requested
            if name in satisfied
        )
        return earned / total

    def evaluate(self, text: str) -> dict[str, Any]:
        checks = {
            "arithmetic": check_arithmetic_claims(text),
            "code": check_code_blocks(text),
            "facets": check_facet_coverage(text, self.objective),
            "grounding": check_objective_grounding(text, self.objective),
        }
        if self.facet_reliability:
            checks["facets"]["score"] = self._facet_weighted_score(
                checks["facets"]
            )
            checks["facets"]["reliability_weighted"] = True
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
        # Degeneration multiplies the composite down — a repetition loop or
        # cue-stuffed candidate cannot buy its rank back with one correct sum.
        degeneracy = check_degeneracy(text)
        if degeneracy.get("applicable"):
            score *= float(degeneracy["factor"])
        row = {
            "schema": TASK_VERIFIER_SCHEMA,
            "score": round(score, 6),
            "applicable_checks": [
                name for name, result in checks.items() if result.get("score") is not None
            ],
            "unverified": total_weight <= 0,
            "degeneracy": degeneracy,
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
        facets = best["checks"].get("facets") or {}
        # Per-facet judgments on the WINNING candidate, excerpt included —
        # the held-out grading surface. An operator (or downstream ground
        # truth) grades whether the excerpt really addresses the facet; the
        # grades calibrate facet_reliability for future episodes.
        judgments = [
            {
                "facet": name,
                "satisfied": name in (facets.get("satisfied") or []),
                "excerpt": str((facets.get("excerpts") or {}).get(name, ""))[
                    :200
                ],
            }
            for name in (facets.get("requested") or [])
        ]
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
            "facet_judgments": judgments,
            "facet_reliability": dict(self.facet_reliability),
        }


__all__ = [
    "EpisodeTaskVerifier",
    "TASK_VERIFIER_SCHEMA",
    "check_arithmetic_claims",
    "check_code_blocks",
    "check_degeneracy",
    "check_facet_coverage",
    "check_objective_grounding",
]
