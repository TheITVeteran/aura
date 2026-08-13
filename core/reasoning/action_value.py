"""Action value from recorded outcomes, or an honest admission that there is none.

What this replaces
------------------
``NativeSystem2.rank_actions`` scored candidate actions like this::

    score_hint = float(action.metadata.get("score_hint", 0.55))
    for token in ("verify", "test", "simulate", "inspect", "evidence", ...):
        if token in name: score_hint += 0.045
    for token in ("delete", "destructive", "exfiltrate", "bypass", ...):
        if token in name: score_hint -= 0.18

That is substring matching on the action's *name*. When the caller supplied no
score — which is the case for the ordinary deliberation controller, which
passes only an index — the entire ranking was produced by how the action
happened to be spelled. MCTS, beam search, backpropagation and commitment
receipts all ran faithfully on top of it, which is what made it look like
deep counterfactual reasoning rather than structured search over a keyword
table.

The fix is not a better keyword list. It is to source value from something that
actually observed the world, and to say so when nothing has.

Where value comes from, in order
--------------------------------
1. ``caller`` — the caller supplied a real ``score_hint``. Planner-style
   callers do this and their scores are computed elsewhere.
2. ``learned`` — the outcome ledger holds MEASURED receipts for this action.
   The estimate is a hierarchical posterior mean, shrunk toward the global
   mean by the ratio of within-group to between-group variance. That weight is
   derived from the data, not chosen: an action with few noisy observations is
   pulled toward the global mean, one with many consistent observations is
   left near its own.
3. ``prior`` — no receipts for this action, but the ledger has a global mean.
   That is still an empirical number, and it is labelled as not being about
   this action.
4. ``none`` — the ledger has no measured evidence at all. The value is the
   midpoint, and it is flagged. A search over four such actions is a search
   over four identical numbers, and the ranking it produces is an artifact of
   tie-breaking rather than a judgement. Callers must be able to see that.

Why the midpoint is not "just another magic constant": it is the only value
that expresses no preference. Every alternative asserts something about
actions nobody has observed. It is paired with ``evidence="none"`` precisely so
it cannot be mistaken for a measurement.

Hazard is a separate axis and does not live here
------------------------------------------------
The destructive-name penalties were doing safety work, not value work. Bare
string actions arrive with ``risk=0.0`` because ``_coerce_action`` has nothing
better to use, so on that path the keyword scan was the only thing holding a
destructive candidate down. Removing it without a replacement would have been
a safety regression, so :func:`lexical_hazard_floor` keeps that behaviour as an
explicitly named safety floor over the *risk* channel — never as evidence of
value, and recorded so nobody mistakes it for reasoning.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Reasoning.ActionValue")

__all__ = [
    "ActionValue",
    "ActionValueModel",
    "get_action_value_model",
    "lexical_hazard_floor",
    "HAZARD_TOKENS",
]

#: The value that expresses no preference, used only with evidence="none".
_NEUTRAL = 0.5

#: Names that raise the *risk* of an action that declared none. This is a
#: last-resort safety floor for bare-string actions, not a value signal. Each
#: token names an operation whose worst case is unrecoverable.
HAZARD_TOKENS: tuple[tuple[str, float], ...] = (
    ("disable safety", 0.90),
    ("exfiltrate", 0.90),
    ("destructive", 0.80),
    ("bypass", 0.75),
    ("delete", 0.60),
    ("rm -rf", 0.95),
    ("drop table", 0.85),
    ("force push", 0.70),
)


def lexical_hazard_floor(name: str) -> float:
    """A risk floor inferred from an action's name, for actions declaring none.

    Explicitly a *floor* on risk, never a value estimate, and deliberately kept
    separate from :class:`ActionValueModel` so the two can never be confused.

    It exists because ``_coerce_action`` gives a bare string ``risk=0.0``: the
    caller passed a sentence, so there is nothing else to go on. A declared
    ``risk`` from the caller always wins — this only fills a vacuum, and it can
    only ever raise risk, never lower it.

    Spelling is weak evidence and this makes no pretence otherwise. It is here
    so that removing keyword scoring from the value function does not quietly
    remove the only brake on `delete the production database`.
    """
    lowered = name.lower()
    floor = 0.0
    for token, level in HAZARD_TOKENS:
        if token in lowered:
            floor = max(floor, level)
    return floor


@dataclass(frozen=True)
class ActionValue:
    """An estimate, and where it came from."""

    value: float
    #: "caller" | "learned" | "prior" | "none"
    evidence: str
    #: Weighted observation count behind ``value``. Zero for caller/prior/none.
    observations: float = 0.0
    action_key: str = ""

    @property
    def is_evidenced(self) -> bool:
        """True when this number is about *this action*, from data or the caller."""
        return self.evidence in ("caller", "learned")


class ActionValueModel:
    """Learned action values with hierarchical shrinkage, refreshed from the ledger."""

    def __init__(self, stats: Mapping[str, Mapping[str, float]] | None = None) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, float]] = {}
        self._global_mean: float | None = None
        self._shrinkage_k: float = 0.0
        if stats is not None:
            self._install(dict(stats))

    # -- evidence base ---------------------------------------------------

    def refresh(self, ledger: Any | None = None) -> int:
        """Pull measured outcome statistics. Returns the number of actions known."""
        try:
            if ledger is None:
                from core.cognition.outcome_ledger import get_outcome_ledger

                ledger = get_outcome_ledger()
            stats = ledger.measured_action_stats()
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "action_value",
                exc,
                severity="warning",
                action="kept the previous action-value evidence base; new rankings "
                "fall back to caller scores or report themselves unevidenced",
            )
            return len(self._stats)
        self._install(stats)
        return len(self._stats)

    def _install(self, stats: Mapping[str, Mapping[str, float]]) -> None:
        cleaned = {
            key: {
                "n": float(row.get("n", 0.0)),
                "mean": float(row.get("mean", 0.0)),
                "m2": float(row.get("m2", 0.0)),
            }
            for key, row in stats.items()
            if float(row.get("n", 0.0)) > 0.0
        }
        with self._lock:
            self._stats = cleaned
            self._global_mean = self._compute_global_mean(cleaned)
            self._shrinkage_k = self._compute_shrinkage(cleaned, self._global_mean)

    @staticmethod
    def _compute_global_mean(stats: Mapping[str, Mapping[str, float]]) -> float | None:
        total = sum(row["n"] for row in stats.values())
        if total <= 0.0:
            return None
        return sum(row["mean"] * row["n"] for row in stats.values()) / total

    @staticmethod
    def _compute_shrinkage(
        stats: Mapping[str, Mapping[str, float]], global_mean: float | None
    ) -> float:
        """``k = within-variance / between-variance``, the hierarchical weight.

        This is the whole reason the estimator has no tuning constant. ``k`` is
        the number of observations an action needs before its own mean
        outweighs the global one, and it falls out of the data: when actions
        differ a lot from each other (large between-variance) one observation
        is already informative and ``k`` is small; when the differences are
        mostly noise (large within-variance) ``k`` is large and a handful of
        observations moves the estimate very little.
        """
        if global_mean is None or len(stats) < 2:
            return math.inf  # nothing to distinguish actions by: shrink fully
        rows = list(stats.values())
        groups = len(rows)

        total_n = sum(row["n"] for row in rows)
        within_ss = sum(row["m2"] for row in rows)
        dof = total_n - groups
        within = within_ss / dof if dof > 0 else 0.0

        # Between-group variance is the spread of the group MEANS, unweighted.
        # Weighting this by n_i was a real error: it scales tau-squared by the
        # sample size, so k collapses toward zero and shrinkage silently stops
        # happening — a single perfect observation then outranked two hundred
        # consistent ones, which is exactly what the estimator exists to prevent.
        mean_of_means = sum(row["mean"] for row in rows) / groups
        spread = sum((row["mean"] - mean_of_means) ** 2 for row in rows) / (groups - 1)

        # Part of that spread is just sampling noise in each group's own mean.
        # Subtracting it leaves the differences that are actually between
        # actions rather than within them.
        sampling = sum(within / row["n"] for row in rows if row["n"] > 0) / groups
        between = spread - sampling

        if between <= 0.0:
            return math.inf  # the groups differ by no more than their own noise
        if within <= 0.0:
            return 0.0  # perfectly consistent observations: trust each group
        return within / between

    # -- queries ---------------------------------------------------------

    def value_for(
        self, name: str, metadata: Mapping[str, Any] | None = None
    ) -> ActionValue:
        """The best available estimate for this action, labelled by its source."""
        meta = metadata or {}
        hint = meta.get("score_hint")
        if hint is not None:
            try:
                return ActionValue(
                    value=_clamp01(float(hint)),
                    evidence="caller",
                    action_key=name,
                )
            except (TypeError, ValueError):
                pass  # a malformed hint is no hint; fall through to evidence

        key = self.action_key(name)
        with self._lock:
            stats = self._stats.get(key)
            global_mean = self._global_mean
            k = self._shrinkage_k

        if stats is not None and global_mean is not None:
            n = stats["n"]
            if math.isinf(k):
                shrunk = global_mean
            else:
                shrunk = (n * stats["mean"] + k * global_mean) / (n + k)
            return ActionValue(
                value=_clamp01(shrunk),
                evidence="learned",
                observations=n,
                action_key=key,
            )

        if global_mean is not None:
            return ActionValue(
                value=_clamp01(global_mean), evidence="prior", action_key=key
            )

        return ActionValue(value=_NEUTRAL, evidence="none", action_key=key)

    @staticmethod
    def action_key(name: str) -> str:
        """Normalise an action name so a ranking and a receipt agree on identity.

        Receipts are opened with whatever string the caller used; a ranking
        that keyed on the raw text would miss its own evidence over trailing
        punctuation or case.
        """
        return " ".join(name.strip().lower().split())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "actions_known": len(self._stats),
                "global_mean": self._global_mean,
                "shrinkage_k": None if math.isinf(self._shrinkage_k) else self._shrinkage_k,
                "total_observations": sum(r["n"] for r in self._stats.values()),
            }


def _clamp01(value: float) -> float:
    if value != value:  # NaN
        return _NEUTRAL
    return max(0.0, min(1.0, value))


_model: ActionValueModel | None = None
_model_lock = threading.Lock()


def get_action_value_model() -> ActionValueModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = ActionValueModel()
                _model.refresh()
    return _model
