"""core/verify/causal_influence.py

Did this faculty change the output?

A subsystem that reports it ran is not evidence that it mattered. Aura is full
of receipts asserting influence — ``live_mind_controls_bound: True``, a system
prompt that tells the model "this is causal grounding for the reply", a
``response_path`` string the code writes about itself. None of them answer the
only question that separates an integrated architecture from a decorated one:
if this faculty's contribution had been neutral, would the output have differed?

This module holds the answer, and it holds it as a measurement rather than a
claim. A channel is INFLUENTIAL only when paired trials — the same input run
with the faculty intact and with it lesioned — diverge by more than the system's
own nondeterminism. That noise floor is not assumed. It is measured, per
channel, from null pairs where nothing was lesioned at all.

Requiring the null is the whole design. An A/B in this repository once scored
d=17.3, p=0.0005 on its own null arm: the pipeline manufactured that separation
out of nothing, and the only reason anyone found out was that somebody ran the
null. A verdict here cannot be reached without one. ``verdict()`` returns
UNMEASURED for a channel with a thousand treatment trials and no null.

Layering: this module measures faculties it must never import. Callers push
observations in; nothing here reaches back out. See DEPS.
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Sequence

__all__ = [
    "Verdict",
    "Divergence",
    "ChannelVerdict",
    "InfluenceLedger",
    "get_influence_ledger",
    "reset_influence_ledger_for_test",
    "text_divergence",
    "vector_divergence",
]


class Verdict(StrEnum):
    """What the measurements support saying about a channel."""

    #: No paired trials, no null, or too few of either to resolve an effect
    #: from noise. The honest default, and the one most channels sit at.
    UNMEASURED = "unmeasured"

    #: Measured with enough power to have seen an effect the size of the
    #: channel's own noise, and none was there. The faculty ran; it did not
    #: change the output.
    INERT = "inert"

    #: Lesioning the faculty moved the output further than nondeterminism
    #: alone moves it, with the confidence interval clear of zero.
    INFLUENTIAL = "influential"


class Arm(StrEnum):
    TREATMENT = "treatment"
    NULL = "null"


@dataclass(frozen=True)
class Divergence:
    """One paired observation.

    ``magnitude`` is how far the two outputs of a pair sat from each other, on
    a 0..1 scale where 0 is byte-identical. For the treatment arm the pair is
    (intact, lesioned); for the null arm it is (intact, intact-again) and the
    magnitude is pure nondeterminism.
    """

    channel: str
    arm: Arm
    magnitude: float
    turn_id: str
    observed_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "arm": str(self.arm),
            "magnitude": round(self.magnitude, 6),
            "turn_id": self.turn_id,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True)
class ChannelVerdict:
    """The full measurement state of one channel — never just a boolean.

    ``verdict`` is the summary, but the numbers behind it travel with it so a
    consumer can never report the conclusion without the evidence that produced
    it. A receipt that carries this cannot claim influence it did not measure.
    """

    channel: str
    verdict: Verdict
    n_treatment: int
    n_null: int
    mean_treatment: float
    mean_null: float
    #: mean(treatment) - mean(null): how much of the divergence the lesion
    #: explains, over and above the noise the null already accounts for.
    effect: float
    ci_low: float
    ci_high: float
    #: Nonparametric effect size. Divergences are bounded and skewed, so a
    #: standardized mean difference would flatter them.
    cliffs_delta: float
    #: The measurement's own resolution: the null arm's spread. An effect
    #: smaller than this is not distinguishable from the system's noise.
    noise_floor: float
    reason: str

    @property
    def is_influential(self) -> bool:
        return self.verdict is Verdict.INFLUENTIAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "verdict": str(self.verdict),
            "n_treatment": self.n_treatment,
            "n_null": self.n_null,
            "mean_treatment": round(self.mean_treatment, 6),
            "mean_null": round(self.mean_null, 6),
            "effect": round(self.effect, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "cliffs_delta": round(self.cliffs_delta, 6),
            "noise_floor": round(self.noise_floor, 6),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return str(text or "").split()


def text_divergence(a: str, b: str) -> float:
    """Normalized token-level edit distance between two generations, 0..1.

    Token-level rather than character-level: a model that rewords a clause has
    diverged, and a model that changes one letter of one word has essentially
    not. Character distance scores those the same way round.
    """

    left = _tokens(a)
    right = _tokens(b)
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    if left == right:
        return 0.0

    # Levenshtein over tokens, two rows at a time.
    previous = list(range(len(right) + 1))
    for i, lhs in enumerate(left, start=1):
        current = [i]
        for j, rhs in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if lhs == rhs else 1),
                )
            )
        previous = current
    return min(1.0, previous[-1] / max(len(left), len(right)))


def vector_divergence(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance between two numeric states, clamped to 0..1.

    Used for channels whose output is a state vector rather than text — a
    logit distribution, a substrate slice, an embedding.
    """

    left = [float(x) for x in a]
    right = [float(x) for x in b]
    if len(left) != len(right) or not left:
        return 1.0 if left != right else 0.0
    norm_left = math.sqrt(sum(x * x for x in left))
    norm_right = math.sqrt(sum(x * x for x in right))
    if norm_left < 1e-12 or norm_right < 1e-12:
        # A zero vector has no direction; calling that "identical" would let a
        # dead channel score a perfect null.
        return 0.0 if norm_left == norm_right else 1.0
    dot = sum(x * y for x, y in zip(left, right))
    return max(0.0, min(1.0, 1.0 - (dot / (norm_left * norm_right))))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _cliffs_delta(treatment: Sequence[float], null: Sequence[float]) -> float:
    """P(treatment > null) - P(treatment < null), in [-1, 1]."""

    if not treatment or not null:
        return 0.0
    greater = 0
    lesser = 0
    for t in treatment:
        for n in null:
            if t > n:
                greater += 1
            elif t < n:
                lesser += 1
    total = len(treatment) * len(null)
    return (greater - lesser) / total


def _bootstrap_difference_ci(
    treatment: Sequence[float],
    null: Sequence[float],
    *,
    resamples: int,
    confidence: float,
    rng: random.Random,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(treatment) - mean(null).

    Bootstrap rather than a t-interval because divergences are bounded at 0 and
    1 and pile up against the floor; the normal approximation puts mass below
    zero where no observation can land.
    """

    if len(treatment) < 2 or len(null) < 2:
        return (float("-inf"), float("inf"))
    differences: list[float] = []
    n_t = len(treatment)
    n_n = len(null)
    for _ in range(resamples):
        resampled_t = _mean([treatment[rng.randrange(n_t)] for _ in range(n_t)])
        resampled_n = _mean([null[rng.randrange(n_n)] for _ in range(n_n)])
        differences.append(resampled_t - resampled_n)
    differences.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = int(math.floor(tail * (len(differences) - 1)))
    high_index = int(math.ceil((1.0 - tail) * (len(differences) - 1)))
    return (differences[low_index], differences[high_index])


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass
class _ChannelSamples:
    treatment: list[float] = field(default_factory=list)
    null: list[float] = field(default_factory=list)
    #: Bound so a long-lived process cannot grow this without limit. Oldest
    #: observations fall off; a channel's verdict tracks how it behaves now,
    #: not how it behaved ten thousand turns ago.
    limit: int = 512

    def add(self, arm: Arm, magnitude: float) -> None:
        bucket = self.treatment if arm is Arm.TREATMENT else self.null
        bucket.append(magnitude)
        if len(bucket) > self.limit:
            del bucket[: len(bucket) - self.limit]


class InfluenceLedger:
    """Accumulates paired divergences and reports what they support.

    Thread-safe: the live runtime records from the generation lane while
    health reporting reads verdicts off the same instance.
    """

    def __init__(
        self,
        *,
        confidence: float = 0.95,
        bootstrap_resamples: int = 2000,
        seed: int | None = None,
    ) -> None:
        self._confidence = confidence
        self._resamples = bootstrap_resamples
        self._channels: dict[str, _ChannelSamples] = {}
        self._recent: list[Divergence] = []
        self._lock = threading.Lock()
        # Seeded so a verdict is reproducible from the same observations. An
        # unseeded bootstrap makes the same data yield different verdicts on
        # either side of a boundary, which is how a measurement turns into a
        # coin flip nobody notices.
        self._rng = random.Random(seed if seed is not None else 0xA11CE)

    # -- recording ---------------------------------------------------------

    def record_treatment(
        self,
        channel: str,
        *,
        intact: Any,
        lesioned: Any,
        turn_id: str = "",
        metric: str = "text",
    ) -> Divergence:
        """Record one (intact vs lesioned) pair for ``channel``."""

        return self._record(
            channel,
            Arm.TREATMENT,
            intact,
            lesioned,
            turn_id=turn_id,
            metric=metric,
        )

    def record_null(
        self,
        channel: str,
        *,
        first: Any,
        second: Any,
        turn_id: str = "",
        metric: str = "text",
    ) -> Divergence:
        """Record one (intact vs intact) pair — the channel's own noise.

        Without these, ``verdict()`` cannot conclude anything for this channel,
        no matter how many treatment pairs it has.
        """

        return self._record(
            channel,
            Arm.NULL,
            first,
            second,
            turn_id=turn_id,
            metric=metric,
        )

    def _record(
        self,
        channel: str,
        arm: Arm,
        left: Any,
        right: Any,
        *,
        turn_id: str,
        metric: str,
    ) -> Divergence:
        name = str(channel or "").strip()
        if not name:
            raise ValueError("causal influence observations require a channel name")
        magnitude = self._measure(left, right, metric)
        observation = Divergence(
            channel=name,
            arm=arm,
            magnitude=magnitude,
            turn_id=str(turn_id or ""),
            observed_at=time.time(),
        )
        with self._lock:
            self._channels.setdefault(name, _ChannelSamples()).add(arm, magnitude)
            self._recent.append(observation)
            if len(self._recent) > 256:
                del self._recent[: len(self._recent) - 256]
        return observation

    @staticmethod
    def _measure(left: Any, right: Any, metric: str) -> float:
        if metric == "vector":
            return vector_divergence(left, right)
        if metric == "text":
            return text_divergence(str(left), str(right))
        raise ValueError(f"unknown divergence metric: {metric!r}")

    # -- reading -----------------------------------------------------------

    def verdict(self, channel: str) -> ChannelVerdict:
        """What the accumulated observations support saying about ``channel``."""

        name = str(channel or "").strip()
        with self._lock:
            samples = self._channels.get(name)
            treatment = list(samples.treatment) if samples else []
            null = list(samples.null) if samples else []

        mean_treatment = _mean(treatment)
        mean_null = _mean(null)
        effect = mean_treatment - mean_null
        noise_floor = _stdev(null)
        delta = _cliffs_delta(treatment, null)

        if not null:
            # The refusal that matters. Treatment divergence on its own is
            # indistinguishable from a nondeterministic decoder, and reading it
            # as influence is exactly the error that produced a d=17.3 null.
            return ChannelVerdict(
                channel=name,
                verdict=Verdict.UNMEASURED,
                n_treatment=len(treatment),
                n_null=0,
                mean_treatment=mean_treatment,
                mean_null=0.0,
                effect=0.0,
                ci_low=float("-inf"),
                ci_high=float("inf"),
                cliffs_delta=0.0,
                noise_floor=0.0,
                reason=(
                    "no null pairs: divergence under lesion cannot be separated "
                    "from the channel's own nondeterminism"
                ),
            )

        ci_low, ci_high = _bootstrap_difference_ci(
            treatment,
            null,
            resamples=self._resamples,
            confidence=self._confidence,
            rng=random.Random(self._rng.randint(0, 2**31 - 1)),
        )

        if ci_low > 0.0:
            return ChannelVerdict(
                channel=name,
                verdict=Verdict.INFLUENTIAL,
                n_treatment=len(treatment),
                n_null=len(null),
                mean_treatment=mean_treatment,
                mean_null=mean_null,
                effect=effect,
                ci_low=ci_low,
                ci_high=ci_high,
                cliffs_delta=delta,
                noise_floor=noise_floor,
                reason=(
                    f"lesioning moved the output {effect:.4f} beyond nondeterminism "
                    f"(95% CI [{ci_low:.4f}, {ci_high:.4f}] excludes zero)"
                ),
            )

        # INERT has to be earned too. Calling a channel inert on two
        # observations asserts an absence nobody looked for. The claim is only
        # available once the interval is tight enough that an effect the size
        # of the measurement's own noise would have shown up.
        if math.isfinite(ci_high) and noise_floor > 0.0 and ci_high < noise_floor:
            return ChannelVerdict(
                channel=name,
                verdict=Verdict.INERT,
                n_treatment=len(treatment),
                n_null=len(null),
                mean_treatment=mean_treatment,
                mean_null=mean_null,
                effect=effect,
                ci_low=ci_low,
                ci_high=ci_high,
                cliffs_delta=delta,
                noise_floor=noise_floor,
                reason=(
                    f"lesioning the channel changed nothing detectable: the whole "
                    f"interval [{ci_low:.4f}, {ci_high:.4f}] sits below the "
                    f"{noise_floor:.4f} noise floor"
                ),
            )

        return ChannelVerdict(
            channel=name,
            verdict=Verdict.UNMEASURED,
            n_treatment=len(treatment),
            n_null=len(null),
            mean_treatment=mean_treatment,
            mean_null=mean_null,
            effect=effect,
            ci_low=ci_low,
            ci_high=ci_high,
            cliffs_delta=delta,
            noise_floor=noise_floor,
            reason=(
                f"underpowered: interval [{ci_low:.4f}, {ci_high:.4f}] spans zero "
                f"and is wider than the {noise_floor:.4f} noise floor"
            ),
        )

    def channels(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._channels))

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view: every channel's verdict with its evidence."""

        verdicts = {name: self.verdict(name).as_dict() for name in self.channels()}
        counts: dict[str, int] = {str(v): 0 for v in Verdict}
        for entry in verdicts.values():
            counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
        return {
            "channels": verdicts,
            "verdict_counts": counts,
            "measured_channels": sum(
                1 for e in verdicts.values() if e["verdict"] != str(Verdict.UNMEASURED)
            ),
        }

    def recent(self, limit: int = 32) -> list[dict[str, Any]]:
        with self._lock:
            return [d.as_dict() for d in self._recent[-max(0, limit) :]]

    # -- persistence -------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "channels": {
                    name: {"treatment": list(s.treatment), "null": list(s.null)}
                    for name, s in self._channels.items()
                },
            }

    def load(self, payload: dict[str, Any]) -> None:
        """Restore accumulated observations.

        A ledger that resets every boot never reaches a verdict, so the samples
        have to outlive the process that took them.
        """

        if not isinstance(payload, dict) or int(payload.get("version", 0)) != 1:
            return
        channels = payload.get("channels")
        if not isinstance(channels, dict):
            return
        with self._lock:
            for name, arms in channels.items():
                if not isinstance(arms, dict):
                    continue
                samples = self._channels.setdefault(str(name), _ChannelSamples())
                for arm_name, bucket in (
                    ("treatment", samples.treatment),
                    ("null", samples.null),
                ):
                    values = arms.get(arm_name)
                    if not isinstance(values, Iterable):
                        continue
                    for value in values:
                        try:
                            bucket.append(max(0.0, min(1.0, float(value))))
                        except (TypeError, ValueError):
                            continue
                    if len(bucket) > samples.limit:
                        del bucket[: len(bucket) - samples.limit]


_LEDGER: InfluenceLedger | None = None
_LEDGER_LOCK = threading.Lock()


def get_influence_ledger() -> InfluenceLedger:
    global _LEDGER
    with _LEDGER_LOCK:
        if _LEDGER is None:
            _LEDGER = InfluenceLedger()
        return _LEDGER


def reset_influence_ledger_for_test() -> None:
    global _LEDGER
    with _LEDGER_LOCK:
        _LEDGER = None
