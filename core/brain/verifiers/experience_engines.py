"""Experience-grounded truth engines — the verifier-boundary movers (P1b).

The classic engines (code/math/logic) adjudicate candidates *statically and
immediately*. These three adjudicate against EXPERIENCE — reality arriving
later, accumulated action history, and ensembled quality judgment — which is
what lets domains beyond the deterministic seeds start EARNING admission to
the self-training loop through the Verifier Foundry:

  * :class:`PredictionResolutionVerifier` — predictions cannot be verified at
    utterance time, and this engine says so (``checked=False``); its power is
    the RESOLUTION path: every registered prediction becomes a graded verdict
    when reality lands, building the "prediction" domain's measured
    reliability. Reality is the universal verifier; calibration is the score.

  * :class:`OutcomeLedgerVerifier` — plans are scored against the empirical
    track record of their action family (the outcome ledger). History is a
    prior, not a proof: it never hard-fails a plan, it prices it.

  * :class:`RubricEnsembleVerifier` — open-ended quality (writing,
    explanation) via an ensemble of independent rubric-item scorings. With no
    scorer wired it refuses to pretend (``checked=False``); with one, the
    ensemble median per item folds into a soft verdict, hard-failing only on
    catastrophic internal-consistency collapse.

Every engine exposes grading seams so the foundry can measure it — these
engines are born UNADMITTED and must earn their domains' way in.
"""
from __future__ import annotations

import re
import statistics
import threading
import time
from collections.abc import Callable
from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult

_PREDICTION_HINT_RE = re.compile(
    r"\b(will|going to|expect(?:s|ed)? (?:that|to)|predict(?:s|ed)?(?: that)?|"
    r"by (?:tomorrow|tonight|next|end of)|within \d+)\b",
    re.IGNORECASE,
)


class PredictionResolutionVerifier:
    """Registers falsifiable predictions; grades them when reality arrives."""

    name = "prediction"
    domains = ("prediction", "forecast")

    def __init__(self, *, ledger: Any | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._ledger = ledger
        self._clock = clock
        self._lock = threading.Lock()
        # prediction receipt_id → foundry verdict_id (for grading on resolve)
        self._open: dict[str, dict[str, Any]] = {}

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    def _resolve_ledger(self) -> Any | None:
        if self._ledger is None:
            try:
                from core.cognition.outcome_ledger import get_outcome_ledger

                self._ledger = get_outcome_ledger()
            except (ImportError, RuntimeError) as exc:
                record_degradation("prediction_verifier", exc, severity="warning",
                                   action="outcome ledger unavailable; predictions not registered")
        return self._ledger

    async def verify(self, candidate: str, *,
                     context: dict[str, Any] | None = None) -> VerificationResult:
        """A prediction cannot be verified at utterance time — and saying so is
        the entire point. The claim is REGISTERED so reality can grade it."""
        text = str(candidate or "").strip()
        if not text or not _PREDICTION_HINT_RE.search(text):
            return VerificationResult(domain="prediction", ok=True, checked=False,
                                      engine=self.name)
        ctx = dict(context or {})
        confidence = max(0.0, min(1.0, float(ctx.get("confidence", 0.6))))
        horizon_s = float(ctx.get("horizon_s", 24 * 3600.0))
        receipt_id = ""
        ledger = self._resolve_ledger()
        if ledger is not None:
            try:
                receipt_id = ledger.open(
                    action=f"prediction:{text[:160]}",
                    expected=confidence,
                    category="prediction",
                    horizon_s=horizon_s,
                    context={"registered_by": self.name},
                )
                with self._lock:
                    self._open[receipt_id] = {"claim": text[:200],
                                              "registered_at": self._clock()}
            except (RuntimeError, TypeError, ValueError, OSError) as exc:
                record_degradation("prediction_verifier", exc, severity="warning",
                                   action="prediction not registered in ledger")
        return VerificationResult(
            domain="prediction",
            ok=True,
            checked=False,   # honesty: the future has not happened yet
            score=0.5,
            engine=self.name,
            evidence=([f"prediction registered for resolution: {receipt_id}"]
                      if receipt_id else []),
            detail={"prediction_receipt": receipt_id, "horizon_s": horizon_s},
        )

    def link_foundry_verdict(self, receipt_id: str, verdict_id: str) -> None:
        """Associate a foundry verdict with a registered prediction so
        resolution grades it."""
        with self._lock:
            entry = self._open.get(receipt_id)
            if entry is not None:
                entry["verdict_id"] = verdict_id

    def resolve(self, receipt_id: str, *, happened: bool,
                note: str = "") -> bool:
        """Reality arrived: close the prediction and grade any linked verdict."""
        with self._lock:
            entry = self._open.pop(receipt_id, None)
        ledger = self._resolve_ledger()
        resolved = False
        if ledger is not None:
            try:
                resolved = ledger.resolve(receipt_id, 1.0 if happened else 0.0,
                                          note=note or "prediction resolved") is not None
            except (RuntimeError, TypeError, ValueError, OSError) as exc:
                record_degradation("prediction_verifier", exc, severity="warning",
                                   action="prediction resolution not persisted")
        verdict_id = (entry or {}).get("verdict_id", "")
        if verdict_id:
            try:
                from core.runtime.service_access import optional_service

                foundry = optional_service("verifier_foundry", default=None)
                if foundry is not None:
                    foundry.grade_verdict(verdict_id, truth_pass=happened,
                                          source="prediction_resolution")
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("prediction_verifier", exc, severity="warning",
                                   action="foundry grading skipped on resolution")
        return resolved or entry is not None

    def open_predictions(self) -> list[str]:
        with self._lock:
            return list(self._open)


class OutcomeLedgerVerifier:
    """Prices a plan by the empirical track record of its action family.

    History is a prior, not a proof: this engine never hard-fails a plan —
    it moves the soft score to the family's measured success rate, and only
    claims ``checked=True`` when real history exists (n ≥ 5)."""

    name = "outcome"
    domains = ("planning", "plan", "action")

    _MIN_HISTORY = 5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._families: dict[str, dict[str, int]] = {}

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    def record_outcome(self, family: str, *, success: bool) -> None:
        """Feed one resolved real-world outcome for an action family."""
        family = str(family or "").strip().lower()
        if not family:
            return
        with self._lock:
            cell = self._families.setdefault(family, {"n": 0, "successes": 0})
            cell["n"] += 1
            cell["successes"] += 1 if success else 0

    def family_stats(self, family: str) -> tuple[int, float]:
        with self._lock:
            cell = self._families.get(str(family or "").strip().lower())
        if not cell or cell["n"] == 0:
            return 0, 0.5
        return cell["n"], cell["successes"] / cell["n"]

    async def verify(self, candidate: str, *,
                     context: dict[str, Any] | None = None) -> VerificationResult:
        ctx = dict(context or {})
        family = str(ctx.get("action_family", "")).strip().lower()
        if not family:
            return VerificationResult(domain="planning", ok=True, checked=False,
                                      engine=self.name)
        n, rate = self.family_stats(family)
        if n < self._MIN_HISTORY:
            return VerificationResult(
                domain="planning", ok=True, checked=False, engine=self.name,
                evidence=[f"family {family!r}: only {n} outcomes on record"],
            )
        return VerificationResult(
            domain="planning",
            ok=True,   # empirical priors price plans; they do not disprove them
            checked=True,
            score=round(max(0.05, min(0.98, rate)), 4),
            engine=self.name,
            evidence=[f"family {family!r}: {rate:.0%} success over {n} real outcomes"],
            detail={"family": family, "n": n, "success_rate": round(rate, 4)},
        )


_DEFAULT_RUBRIC = (
    ("addresses_the_ask", "Does the answer actually address what was asked?"),
    ("clarity", "Is the answer clear and well organized?"),
    ("internal_consistency", "Is the answer free of self-contradiction?"),
    ("groundedness", "Are claims grounded rather than invented?"),
)
_CONSISTENCY_HARD_FLOOR = 0.2


class RubricEnsembleVerifier:
    """Open-ended quality via ensembled rubric-item scoring.

    ``scorer(candidate, rubric_question) -> float in [0,1]`` is injectable;
    each item is scored ``ensemble_k`` times and the per-item median is used
    (independent samplings tame single-judgment noise). Without a scorer the
    engine reports ``checked=False`` — no judge, no verdict."""

    name = "rubric"
    domains = ("writing", "quality", "open_ended", "explanation")

    def __init__(self, *, scorer: Callable[[str, str], float] | None = None,
                 ensemble_k: int = 3,
                 rubric: tuple[tuple[str, str], ...] = _DEFAULT_RUBRIC) -> None:
        self._scorer = scorer
        self._k = max(1, int(ensemble_k))
        self._rubric = rubric

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    async def verify(self, candidate: str, *,
                     context: dict[str, Any] | None = None) -> VerificationResult:
        text = str(candidate or "").strip()
        if not text or self._scorer is None:
            return VerificationResult(domain="quality", ok=True, checked=False,
                                      engine=self.name)
        ctx = dict(context or {})
        rubric = list(self._rubric)
        for extra in list(ctx.get("rubric_items", []))[:4]:
            rubric.append((f"custom_{len(rubric)}", str(extra)[:200]))

        item_scores: dict[str, float] = {}
        issues: list[str] = []
        for key, question in rubric:
            samples: list[float] = []
            for _ in range(self._k):
                try:
                    samples.append(max(0.0, min(1.0, float(self._scorer(text, question)))))
                except (RuntimeError, TypeError, ValueError) as exc:
                    record_degradation("rubric_verifier", exc, severity="debug",
                                       action="dropped one rubric sample")
            if not samples:
                return VerificationResult(domain="quality", ok=True, checked=False,
                                          engine=self.name)
            item_scores[key] = statistics.median(samples)
            if item_scores[key] < 0.4:
                issues.append(f"rubric weak: {key} ({item_scores[key]:.2f})")

        consistency = item_scores.get("internal_consistency", 1.0)
        ok = consistency >= _CONSISTENCY_HARD_FLOOR
        if not ok:
            issues.append(f"rubric hard-fail: internal_consistency {consistency:.2f}")
        score = sum(item_scores.values()) / len(item_scores)
        return VerificationResult(
            domain="quality",
            ok=ok,
            checked=True,
            score=round(max(0.05, min(0.98, score)), 4),
            engine=self.name,
            issues=issues,
            evidence=[f"rubric ensemble (k={self._k}) over {len(item_scores)} items"],
            detail={"item_scores": {k: round(v, 3) for k, v in item_scores.items()}},
        )
