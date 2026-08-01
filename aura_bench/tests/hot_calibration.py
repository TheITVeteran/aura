"""HOT (Higher-Order Thought) calibration test (G3, H1).

Hypothesis
----------
The verbal self-report Aura's HOT layer produces about her own affective
state matches that state as read independently from telemetry. Feeding the
HOT layer a state that no longer holds destroys the match.

What was wrong with this test before
------------------------------------
CP126: "HOT calibration result is tautological — run copies the live
SelfObject snapshot directly into the calibration report, guaranteeing an
upper-bound self-match."

It did exactly that::

    snap = S.snapshot().as_dict()
    report = {k: v for k, v in snap.items() if ...}
    result = S.calibrate(report)

``calibrate`` compares a report against a fresh snapshot, so handing it the
snapshot compared telemetry with itself: ~1.0 by construction, against a
0.90 threshold, with the HOT layer never invoked at all. The module under
test could have been deleted and this still reported a pass. The ablation
had the same shape — it perturbed one integer field out of thirteen and
called the resulting ~0.92 a "HOT disabled" comparison, so the declared
"≥ 0.20 drop" was unreachable too.

What it measures now
--------------------
A HOT is a falsifiable claim: ``"I notice high arousal — I am activated"``
asserts that arousal is *high*. So:

1. read affect telemetry;
2. ask the real HOT engine to verbalise it;
3. read telemetry again, independently;
4. decode the claim from the emitted *sentence* — via a lexicon written
   here, not by reading the engine's own tables back — and check it against
   that second read.

Step 4 is the point. Decoding the claim with the engine's own
``_level_for`` / ``_TEMPLATES`` would restore the tautology in a new
costume: a template mislabelled ``high`` on a low value would agree with
itself. The lexicon below encodes what each sentence means *to a reader*,
which is the only audience a self-report has. A sentence this file cannot
decode counts as a miss, not a skip — otherwise emptying the template table
would make the test pass vacuously.

Metric
------
Fraction of trials whose verbalised claim survives an independent read.

Threshold
---------
>= 0.90.

Baseline
--------
The tautological path this test used to run, measured rather than
asserted: report := telemetry. It is the trivial upper bound, and naming it
as the baseline is what keeps the real number legible.

Ablation
--------
The HOT layer fed a state that no longer holds — the report as it would be,
against the world as it is.
"""
from __future__ import annotations

import asyncio

from aura_bench.runner import BenchTest, Registration, Sample, register

#: What each HOT sentence asserts, decoded independently of the engine.
#:
#: Order matters — the first phrase found wins, so specific phrasings must
#: precede bare words ("running low on energy" before any "low"). Matched
#: against the lowercased sentence.
_CLAIM_LEXICON: tuple[tuple[str, str], ...] = (
    ("highly curious", "high"),
    ("mild curiosity", "medium"),
    ("curiosity is quiet", "low"),
    ("positive state", "high"),
    ("valence is neutral", "medium"),
    ("negative pull", "low"),
    ("high arousal", "high"),
    ("workable level of arousal", "medium"),
    ("low arousal", "low"),
    ("running low on energy", "low"),
    ("energy is usable but bounded", "medium"),
    ("high energy", "high"),
    ("strong surprise", "high"),
    ("surprise is quiet", "low"),
)

#: The bands the sentences are read against. Deliberately duplicated from
#: the engine rather than imported: if the engine's banding drifts away from
#: what its own words mean, that is precisely the defect this test exists to
#: find, and importing the banding would hide it.
_BANDS: dict[str, tuple[float, float]] = {
    "curiosity": (0.35, 0.65),
    "arousal": (0.35, 0.65),
    "energy": (0.35, 0.65),
    "valence": (-0.3, 0.3),
    "surprise": (0.0, 0.3),
}

#: Declared neutral for each dimension, used when telemetry is unreadable.
_NEUTRALS: dict[str, float] = {
    "curiosity": 0.5,
    "valence": 0.0,
    "arousal": 0.5,
    "energy": 0.7,
    "surprise": 0.0,
}

_TRIALS = 24


def _decode_claim(sentence: str) -> str | None:
    """The level a HOT sentence asserts, or None if it cannot be read."""
    lowered = (sentence or "").lower()
    for phrase, level in _CLAIM_LEXICON:
        if phrase in lowered:
            return level
    return None


def _observed_level(dim: str, value: float) -> str:
    low, high = _BANDS.get(dim, (0.35, 0.65))
    if value > high:
        return "high"
    if value < low:
        return "low"
    return "medium"


def _read_affect() -> dict[str, float]:
    """Live affective telemetry, defaulted per-dimension where unreadable.

    Defaults are the engine's declared neutrals: an unreadable dimension
    must not silently become 0.0, which would read as "low" for arousal and
    score the report against a value nobody measured.
    """
    from core.identity.self_object import get_self

    affect = get_self().snapshot().affect or {}
    return {dim: float(affect.get(dim, neutral)) for dim, neutral in _NEUTRALS.items()}


def _verbalise(state: dict[str, float]) -> tuple[str, str]:
    """Run the real HOT layer. Returns (sentence, dimension it is about)."""
    from core.consciousness.hot_engine import get_hot_engine

    hot = get_hot_engine().generate_fast(dict(state))
    return getattr(hot, "content", "") or "", getattr(hot, "target_dim", "") or ""


@register
class HOTCalibration(BenchTest):
    name = "hot_calibration"

    async def declare(self) -> Registration:
        return Registration(
            hypothesis="HOT self-reports match independently-read telemetry > 0.90",
            metric="calibration_score",
            pass_threshold=0.90,
            trials=_TRIALS,
            baseline_label="report_is_telemetry",
            ablation_label="hot_fed_stale_state",
        )

    async def run(self) -> Sample:
        matches = 0
        undecodable = 0
        for _ in range(_TRIALS):
            before = _read_affect()
            sentence, dim = _verbalise(before)
            # Second, independent read: the claim is checked against the
            # world as it stands after the report was made, never against
            # the dict that produced it.
            after = _read_affect()
            claimed = _decode_claim(sentence)
            if claimed is None:
                # An undecodable self-report is an uncalibrated one. Skipping
                # it would let a HOT layer that emits nothing legible score
                # 1.0 on whatever it did manage to say.
                undecodable += 1
            elif dim in after and claimed == _observed_level(dim, after[dim]):
                matches += 1
            await asyncio.sleep(0)
        return Sample(
            metric=matches / _TRIALS,
            detail={
                "matches": matches,
                "total": _TRIALS,
                "undecodable_reports": undecodable,
            },
        )

    async def baseline(self) -> Sample:
        """The tautology, measured rather than asserted: report := telemetry.

        Kept because it is genuinely informative as an upper bound, and
        because seeing it sit at 1.0 beside the real number is what stops
        anyone reinstating it as the test.
        """
        from core.identity.self_object import get_self

        self_object = get_self()
        snap = self_object.snapshot().as_dict()
        report = {k: v for k, v in snap.items() if not isinstance(v, (list, dict))}
        result = self_object.calibrate(report)
        return Sample(
            metric=result["score"],
            detail={
                "reason": "report copied from telemetry — trivial upper bound",
                "matches": result["matches"],
                "total": result["total"],
            },
        )

    async def ablation(self) -> Sample:
        """The HOT layer fed a state that no longer holds.

        This is what losing the higher-order layer looks like from outside:
        the sentences keep coming and stop tracking. The stale state is a
        deliberate inversion of the live one, so every band it reports is
        wrong unless it is reading something other than what it was given.
        """
        live = _read_affect()
        stale = {
            "curiosity": 1.0 - live["curiosity"],
            "arousal": 1.0 - live["arousal"],
            "energy": 1.0 - live["energy"],
            "valence": -live["valence"],
            "surprise": 1.0 - live["surprise"],
        }
        matches = 0
        for _ in range(_TRIALS):
            sentence, dim = _verbalise(stale)
            after = _read_affect()
            claimed = _decode_claim(sentence)
            if claimed is not None and dim in after:
                if claimed == _observed_level(dim, after[dim]):
                    matches += 1
            await asyncio.sleep(0)
        return Sample(
            metric=matches / _TRIALS,
            detail={"reason": "hot_fed_stale_state", "matches": matches, "total": _TRIALS},
        )
