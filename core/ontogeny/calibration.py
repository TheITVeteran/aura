"""L4 — anti-collapse: keeping a head honest, and turning her record into grounded doubt.

Two things live here, and they share a question: *how much should a claim from
experience be trusted?*

**Calibration** watches whether a head's stated probabilities match reality.
This matters more than accuracy. A head that is right 70% of the time and says
so is useful; a head that is right 70% of the time and says 95% is a liability,
because everything downstream sizes its caution by the number. Brier score and
expected calibration error are tracked on a rolling window, and a head whose
calibration degrades materially against its grant-time baseline has its
authority revoked automatically. That revocation is the anti-collapse
guarantee: once a head decides, it makes its own training data, and drift is
the expected failure mode rather than a surprising one.

**The track record** is the part that is causal on day one and needs nobody's
permission. It is not a model and makes no predictions — it is arithmetic over
what actually happened: in situations like this one, how often did it go well,
and how sure can she be of that given how few times she has been here? A Wilson
interval on her own history is a fact about her, available from the first
handful of episodes, and it is the honest source for how confident she sounds.

The distinction matters. A head's probability is a claim that must earn trust.
A track record is a count. Aura can act on the count immediately.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from core.ontogeny.experience import Episode, OutcomeKind

logger = logging.getLogger("Aura.Ontogeny.Calibration")

_Z95 = 1.959963984540054

#: Rolling window per control point. Long enough for a stable estimate, short
#: enough that a head going bad this week is visible this week.
_WINDOW = 500

#: Calibration bins for ECE.
_BINS = 10

#: A head whose ECE exceeds its grant-time baseline by this much has stopped
#: being honest about itself and loses authority.
ECE_DRIFT_LIMIT = 0.12

#: Below this many graded episodes a track record states its ignorance rather
#: than a rate. Three coin flips are not a base rate.
MIN_TRACK_RECORD = 12


def wilson(successes: float, total: float, *, upper: bool, z: float = _Z95) -> float:
    """Wilson score bound — the conservative reading of a small sample.

    Used in both directions on purpose: a challenger is judged by its
    pessimistic bound and the incumbent by its optimistic one, so promotion
    requires genuine separation rather than a lucky streak.
    """
    if total <= 0:
        return 0.0 if upper is False else 1.0
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))) / denom
    value = centre + margin if upper else centre - margin
    return float(min(1.0, max(0.0, value)))


@dataclass(frozen=True)
class CalibrationReport:
    """How honest a head's confidence has been lately."""

    control_point: str
    samples: int
    accuracy: float
    brier: float
    ece: float
    mean_confidence: float
    #: mean_confidence - accuracy. Positive means overconfident, which is the
    #: direction that hurts.
    overconfidence: float
    reliability: tuple[tuple[float, float, int], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_point": self.control_point,
            "samples": self.samples,
            "accuracy": round(self.accuracy, 4),
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "overconfidence": round(self.overconfidence, 4),
            "reliability": [
                {"confidence": round(c, 3), "accuracy": round(a, 3), "n": n}
                for c, a, n in self.reliability
            ],
        }


class CalibrationMonitor:
    """Rolling calibration per control point, with a drift verdict."""

    def __init__(self, window: int = _WINDOW) -> None:
        self._window = int(window)
        self._samples: dict[str, deque[tuple[float, bool, float]]] = {}
        self._baselines: dict[str, float] = {}

    def observe(self, control_point: str, *, confidence: float, correct: bool) -> None:
        series = self._samples.setdefault(control_point, deque(maxlen=self._window))
        series.append((float(confidence), bool(correct), time.time()))

    def report(self, control_point: str) -> CalibrationReport | None:
        series = self._samples.get(control_point)
        if not series:
            return None
        confidences = [c for c, _, _ in series]
        corrects = [1.0 if ok else 0.0 for _, ok, _ in series]
        n = len(series)
        accuracy = sum(corrects) / n
        brier = sum((c - ok) ** 2 for c, ok in zip(confidences, corrects, strict=True)) / n
        mean_conf = sum(confidences) / n

        buckets: list[tuple[float, float, int]] = []
        ece = 0.0
        for b in range(_BINS):
            low, high = b / _BINS, (b + 1) / _BINS
            members = [
                (c, ok) for c, ok in zip(confidences, corrects, strict=True)
                if (low < c <= high) or (b == 0 and c <= high)
            ]
            if not members:
                continue
            bin_conf = sum(c for c, _ in members) / len(members)
            bin_acc = sum(ok for _, ok in members) / len(members)
            buckets.append((bin_conf, bin_acc, len(members)))
            ece += (len(members) / n) * abs(bin_conf - bin_acc)

        return CalibrationReport(
            control_point=control_point,
            samples=n,
            accuracy=accuracy,
            brier=brier,
            ece=ece,
            mean_confidence=mean_conf,
            overconfidence=mean_conf - accuracy,
            reliability=tuple(buckets),
        )

    def set_baseline(self, control_point: str) -> float | None:
        """Freeze the calibration a head had when it was granted authority."""
        report = self.report(control_point)
        if report is None:
            return None
        self._baselines[control_point] = report.ece
        return report.ece

    def drifted(self, control_point: str) -> tuple[bool, str]:
        """Has this head stopped being honest since it was trusted?"""
        baseline = self._baselines.get(control_point)
        report = self.report(control_point)
        if baseline is None or report is None or report.samples < 50:
            return False, "insufficient evidence"
        if report.ece > baseline + ECE_DRIFT_LIMIT:
            return True, (
                f"calibration error {report.ece:.3f} exceeds grant-time "
                f"{baseline:.3f} by more than {ECE_DRIFT_LIMIT}"
            )
        return False, f"ece {report.ece:.3f} within {ECE_DRIFT_LIMIT} of baseline {baseline:.3f}"

    def all_reports(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for control_point in self._samples:
            report = self.report(control_point)
            if report is not None:
                out[control_point] = report.as_dict()
        return out


@dataclass(frozen=True)
class TrackRecord:
    """What actually happened, in situations like this one.

    Not a prediction. A count, with an interval that is wide when she has not
    been here often. This is the number that should shape how confidently she
    speaks — and it is available from her twelfth episode, not from a promotion.
    """

    control_point: str
    bucket: str
    successes: int
    failures: int
    #: Episodes in this bucket whose outcome was never observed. Reported
    #: because "I acted forty times and only ever saw how eight of them went"
    #: is itself important, and hiding it would inflate her sense of feedback.
    unobserved: int = 0

    @property
    def graded(self) -> int:
        return self.successes + self.failures

    @property
    def rate(self) -> float | None:
        return self.successes / self.graded if self.graded else None

    @property
    def interval(self) -> tuple[float, float] | None:
        if self.graded < MIN_TRACK_RECORD:
            return None
        return (
            wilson(self.successes, self.graded, upper=False),
            wilson(self.successes, self.graded, upper=True),
        )

    @property
    def is_grounded(self) -> bool:
        """Enough lived evidence to say anything at all."""
        return self.graded >= MIN_TRACK_RECORD

    def phrase(self) -> str:
        """How this reads when it reaches language.

        Deliberately plain, and deliberately willing to say the unflattering
        version. A track record that only speaks when it is good is decoration.
        """
        if not self.is_grounded:
            return (
                f"I have only {self.graded} graded outcome"
                f"{'' if self.graded == 1 else 's'} to go on here, so I am going by "
                "reasoning rather than track record."
            )
        low, high = self.interval or (0.0, 1.0)
        rate = self.rate or 0.0
        span = f"{low:.0%}–{high:.0%}"
        if rate >= 0.8:
            return f"This has gone well {rate:.0%} of the time for me ({span}, n={self.graded})."
        if rate <= 0.45:
            return (
                f"I have a poor record here — {rate:.0%} ({span}, n={self.graded}). "
                "Worth more caution than this feels like it needs."
            )
        return f"My record here is mixed: {rate:.0%} ({span}, n={self.graded})."

    def as_dict(self) -> dict[str, Any]:
        interval = self.interval
        return {
            "control_point": self.control_point,
            "bucket": self.bucket,
            "successes": self.successes,
            "failures": self.failures,
            "unobserved": self.unobserved,
            "graded": self.graded,
            "rate": round(self.rate, 4) if self.rate is not None else None,
            "interval": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
            "grounded": self.is_grounded,
        }


class TrackRecordIndex:
    """Live tallies per (control point, bucket), maintained as outcomes land.

    The track record is consulted on the decision path, so it cannot be a
    query. Counting is incremental — an outcome moves one integer — and the
    index is rehydrated from the corpus on a slow cadence so a restart or an
    eviction cannot let it drift away from what the ledger actually says.
    """

    def __init__(self) -> None:
        self._cells: dict[tuple[str, str], dict[str, int]] = {}
        self._hydrated_at = 0.0

    def observe(self, control_point: str, bucket: str, kind: OutcomeKind, *, weight: int = 1) -> None:
        cell = self._cells.setdefault((control_point, bucket), {"s": 0, "f": 0, "u": 0})
        if kind is OutcomeKind.SUCCESS:
            cell["s"] += weight
        elif kind is OutcomeKind.FAILURE:
            cell["f"] += weight
        else:
            cell["u"] += weight

    def get(self, control_point: str, bucket: str) -> TrackRecord | None:
        cell = self._cells.get((control_point, bucket))
        if cell is None:
            return None
        return TrackRecord(
            control_point=control_point, bucket=bucket,
            successes=cell["s"], failures=cell["f"], unobserved=cell["u"],
        )

    def hydrate(self, control_point: str, episodes: Iterable[Episode], *, keys: Sequence[str] = ()) -> int:
        """Rebuild one control point's tallies from the corpus, replacing them."""
        fresh = track_records(episodes, keys=keys)
        for key in [k for k in self._cells if k[0] == control_point]:
            self._cells.pop(key, None)
        for bucket, record in fresh.items():
            self._cells[(control_point, bucket)] = {
                "s": record.successes, "f": record.failures, "u": record.unobserved,
            }
        self._hydrated_at = time.time()
        return len(fresh)

    def report(self) -> dict[str, Any]:
        return {
            "buckets": len(self._cells),
            "hydrated_age_s": round(time.time() - self._hydrated_at, 1) if self._hydrated_at else None,
            "records": {
                f"{cp}|{bucket}": dict(cell)
                for (cp, bucket), cell in sorted(self._cells.items())[:40]
            },
        }


def bucket_of(episode: Episode, *, keys: Sequence[str] = ()) -> str:
    """Coarse context bucket for a track record.

    Coarse on purpose. Fine buckets give beautiful, meaningless rates over
    n=2. The default groups by what was decided, which answers the question
    that actually gets asked: "when I do this, how does it usually go?"
    """
    if not keys:
        return episode.decision
    parts = [episode.decision]
    for key in keys:
        value = episode.features.get(key)
        if value is None:
            parts.append(f"{key}=?")
        else:
            parts.append(f"{key}={'hi' if float(value) >= 0.5 else 'lo'}")
    return "|".join(parts)


def track_records(
    episodes: Iterable[Episode], *, keys: Sequence[str] = ()
) -> dict[str, TrackRecord]:
    """Aggregate lived episodes into per-bucket records."""
    tally: dict[str, dict[str, int]] = {}
    control_point = ""
    for episode in episodes:
        control_point = control_point or episode.control_point
        bucket = bucket_of(episode, keys=keys)
        cell = tally.setdefault(bucket, {"s": 0, "f": 0, "u": 0})
        if episode.outcome is None:
            continue
        weight = max(1, int(episode.repeat_count))
        if episode.outcome.kind is OutcomeKind.SUCCESS:
            cell["s"] += weight
        elif episode.outcome.kind is OutcomeKind.FAILURE:
            cell["f"] += weight
        else:
            cell["u"] += weight
    return {
        bucket: TrackRecord(
            control_point=control_point,
            bucket=bucket,
            successes=cell["s"],
            failures=cell["f"],
            unobserved=cell["u"],
        )
        for bucket, cell in tally.items()
    }


__all__ = [
    "ECE_DRIFT_LIMIT",
    "MIN_TRACK_RECORD",
    "CalibrationMonitor",
    "CalibrationReport",
    "TrackRecord",
    "TrackRecordIndex",
    "bucket_of",
    "track_records",
    "wilson",
]
