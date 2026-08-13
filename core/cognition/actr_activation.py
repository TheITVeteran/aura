"""ACT-R subsymbolic activation: recency, frequency, spreading, and latency.

Why this exists, concretely
---------------------------
``EpisodicMemory._recency_score`` was::

    min(1.0, max(0.0, ep.timestamp - 1774000000) / 2000000)

which is not a recency score. It is a step function keyed to a hardcoded
wall-clock epoch: 0.0 for anything before 2026-03-20, a 23-day ramp, then a
flat 1.0 forever after 2026-04-12. Evaluated on 2026-08-12, an episode from one
minute ago and one from thirty days ago both scored exactly 1.000000, so the
recency term contributed a constant 0.4 to every candidate and the ranking was
importance-only. It could not have discriminated, and it drifts further from
usable every day it is left alone.

The fix is not a better constant. Any absolute-epoch formulation has this bug
latent in it. ACT-R's base-level equation is scale-free — it depends only on
*elapsed* time — so it cannot saturate and has no epoch to go stale.

The equations
-------------
Base-level activation, the power law of forgetting and practice together::

    B_i = ln( Σ_j t_j^-d )

over the ages ``t_j`` of each prior use of chunk ``i``. Frequency raises it,
recency weights the recent uses more, and the sum reproduces both the forgetting
curve and the spacing effect without either being modelled separately.

Total activation adds context and error terms::

    A_i = B_i + Σ_k W_k · S_ki + P_i + ε

Retrieval probability and — the part Aura has never had — predicted latency::

    P(retrieve) = 1 / (1 + exp(-(A_i - τ) / s))
    T_i         = F · exp(-A_i)

Both were then fitted against Aura's own measured recall, and they came out
differently. That result is the important part of this docstring.

**The retrieval curve fits.** Maximum likelihood over 6,000 samples — 150
batches of 40 traces, ages from one minute to a year, 0 to 30 rehearsals,
scored on whether each trace actually came back in the ranked top-k — gives
``tau = -0.4666``, ``s = 2.0`` (:data:`FITTED_PARAMETERS`), with a Brier skill
of 0.154 over the base rate. Modest by construction: activation carries 0.4 of
the ranking blend against 0.6 for importance, so activation alone should
explain some of recall and not most of it.

**The latency equation does not transfer, and F is therefore not fitted.**
Regressing ``ln T`` on ``-A`` across the same 6,000 samples gives
``r^2 = 0.000037``. There is no relationship, and there is no reason there
should be: ``T = F·e^-A`` earns its shape in ACT-R because retrieval there is a
race between activations, whereas Aura's recall is a ranked scan whose cost
tracks how many candidates exist and what the store does, not how strong the
winning trace is.

This is worth stating plainly because F is a pure multiplicative scale and
would have absorbed any timing whatsoever. Fitting it would have produced a
confident number with no mechanism under it — the exact failure
``RETRACTION.json`` is about. ``tools/fit_actr_retrieval.py`` refuses to emit
an F below an r^2 of 0.10 for that reason, and a test pins the null so that if
retrieval ever does become activation-driven, someone finds out.

So Aura can now predict *which* memories return, with fitted parameters and a
reported skill score. It cannot predict *how long* recall takes from
activation, and that is a property of its retrieval architecture rather than a
gap in the model.

On ACT-R's own main criticism
-----------------------------
The standard objection to ACT-R is that it has enough free parameters to fit
anything, and that fits are reported without showing the parameters were
identifiable. Taking that seriously is why :class:`ActrParameters` carries
published provenance per field rather than bare floats, and why
:func:`latency_sensitivity` exists: a latency claim from this module should be
reported with the spread produced by perturbing the parameters, so a fit that
only works at one point in parameter space is visible as such.

Everything here is pure: no clocks, no I/O, no globals. ``now`` is always
passed in, which is also what makes the tests deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

__all__ = [
    "ActrParameters",
    "DEFAULT_PARAMETERS",
    "base_level_activation",
    "base_level_optimized",
    "fan_strength",
    "spreading_activation",
    "mismatch_penalty",
    "activation",
    "retrieval_probability",
    "retrieval_latency",
    "expected_latency",
    "latency_sensitivity",
]

#: Smallest age, in seconds, that a presentation may have. ``t^-d`` diverges as
#: t → 0, so a use "right now" would return infinite activation. ACT-R has the
#: same singularity and the same remedy; 50ms is one production cycle, the
#: shortest interval the architecture can distinguish anyway.
_MIN_AGE_S = 0.05


@dataclass(frozen=True)
class ActrParameters:
    """Subsymbolic parameters, each with the source of its default.

    Defaults are the conventional published values, not choices made here. They
    are a starting point for a fit, not a result: a latency prediction quoted
    from unfitted defaults is a prediction about ACT-R, not about Aura.
    """

    #: Base-level decay ``d``. 0.5 is the near-universal ACT-R value, stable
    #: across a very wide range of memory tasks.
    decay: float = 0.5
    #: Activation noise ``s``. Sets the softness of the retrieval threshold;
    #: the logistic approximation to Gaussian noise uses the same s.
    noise_s: float = 0.4
    #: Retrieval threshold ``τ``. Below this a chunk is not retrieved at all.
    #: Strongly task-dependent — this default is a placeholder for a fit.
    threshold: float = 0.0
    #: Latency factor ``F`` in ``T = F·e^-A``, in seconds. Task-dependent
    #: scaling; must be fit before any absolute time claim is made.
    latency_factor: float = 0.35
    #: Maximum associative strength ``S`` in the fan equation ``S_ki = S - ln(fan_k)``.
    max_association: float = 2.0
    #: Total source activation ``W`` spread over the cues in the context.
    source_activation: float = 1.0
    #: Mismatch scaling ``P``. Multiplies accumulated cue mismatch.
    mismatch_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.decay < 1.0:
            raise ValueError("decay must lie in (0, 1); the optimized form diverges outside it")
        if self.noise_s <= 0.0:
            raise ValueError("noise_s must be positive")
        if self.latency_factor <= 0.0:
            raise ValueError("latency_factor must be positive")


DEFAULT_PARAMETERS = ActrParameters()

#: Calibrated against Aura's own ranker by ``tools/fit_actr_retrieval.py`` in
#: SYNTHETIC mode: 6,000 samples over 150 batches of 40 generated traces,
#: maximum likelihood on whether each was returned in the ranked top-k.
#:
#: Read that word carefully. This is the ranker calibrated against itself on
#: invented inputs — a self-consistency check with a fitted curve attached. It
#: is not a measurement of Aura's memory and it is emphatically not evidence
#: that Aura reproduces human ACT-R retrieval curves; no human data is involved
#: anywhere in the pipeline that produced these two numbers.
#:
#: ``--source observed`` fits the same curve against REAL recalls recorded by
#: ``core/memory/recall_observations.py``, which is the honest source and gives
#: materially different answers — an 80-ranking sample measured tau=1.50,
#: s=2.1, Brier skill 0.209 against the synthetic 0.154. These constants stay
#: synthetic until enough live recall has accumulated to replace them, and the
#: gap between the two is the reason the distinction is worth keeping.
#:
#: ``latency_factor`` is deliberately left at its published default and must
#: not be read as fitted — see the module docstring for why it is not fittable.
FITTED_PARAMETERS = ActrParameters(threshold=-0.4666, noise_s=2.0)

#: Brier skill of :data:`FITTED_PARAMETERS` over predicting the base rate.
#: Modest on purpose: activation carries 0.4 of the ranking blend against 0.6
#: for importance, so activation alone should explain some of recall and not
#: most of it. A number near 1.0 here would mean importance had stopped
#: mattering.
FITTED_BRIER_SKILL = 0.154


def _ages(presentations: Sequence[float], now: float) -> list[float]:
    """Ages of each presentation, clamped away from the ``t^-d`` singularity.

    Presentations in the future are treated as "just now" rather than raising:
    clock skew between a stored timestamp and the caller's ``now`` is a
    routine condition and must not be able to abort a retrieval.
    """
    return [max(_MIN_AGE_S, now - t) for t in presentations]


def base_level_activation(
    presentations: Sequence[float],
    now: float,
    *,
    decay: float = DEFAULT_PARAMETERS.decay,
) -> float:
    """``B_i = ln( Σ_j t_j^-d )`` over every recorded use of the chunk.

    Returns ``-inf`` for a chunk that has never been presented, which is the
    correct answer: it is below every finite threshold and so is never
    retrieved, without needing a special case at the call site.
    """
    ages = _ages(presentations, now)
    if not ages:
        return -math.inf
    return math.log(sum(age**-decay for age in ages))


def base_level_optimized(
    n_presentations: int,
    lifetime_s: float,
    *,
    decay: float = DEFAULT_PARAMETERS.decay,
) -> float:
    """Hybrid approximation to :func:`base_level_activation`.

    ``B ≈ ln( n / (1-d) ) - d·ln(T)``, where ``T`` is the time since the first
    presentation. Exact base-level cost grows with the number of uses, which is
    unbounded for a long-lived system; this form is O(1) and needs only a count
    and a first-seen timestamp.

    It assumes presentations are roughly uniform over the lifetime, so it is
    accurate for many well-spread uses and wrong for a few clustered ones —
    :func:`base_level_activation` is the one to use when the actual timestamps
    are on hand.
    """
    if n_presentations <= 0:
        return -math.inf
    span = max(_MIN_AGE_S, lifetime_s)
    return math.log(n_presentations / (1.0 - decay)) - decay * math.log(span)


def fan_strength(
    fan: int,
    *,
    max_association: float = DEFAULT_PARAMETERS.max_association,
) -> float:
    """``S_ji = S - ln(fan_j)`` — the fan effect.

    A cue associated with many chunks is weaker evidence for any one of them.
    This is the mechanism behind ACT-R's account of retrieval interference, and
    it is the piece a plain similarity score has no way to express.
    """
    return max_association - math.log(max(1, fan))


def spreading_activation(
    cue_weights: Mapping[str, float],
    strengths: Mapping[str, float],
) -> float:
    """``Σ_k W_k · S_ki`` — context lending activation to a chunk.

    ``cue_weights`` is the attentional weight on each cue (conventionally the
    total source activation W split evenly across cues) and ``strengths`` the
    per-cue association to this chunk. Cues with no association contribute
    nothing rather than being an error.
    """
    return sum(w * strengths.get(cue, 0.0) for cue, w in cue_weights.items())


def mismatch_penalty(
    similarities: Sequence[float],
    *,
    mismatch_scale: float = DEFAULT_PARAMETERS.mismatch_scale,
) -> float:
    """``P·Σ(sim - 1)`` — partial matching, always ≤ 0.

    ``similarities`` are in [0, 1] where 1.0 is an exact match. Partial
    matching is what lets a near-miss chunk be retrieved at a cost instead of
    being invisible, which is the difference between graceful recall and a
    lookup that silently returns nothing.
    """
    return mismatch_scale * sum(min(1.0, max(0.0, s)) - 1.0 for s in similarities)


def activation(
    *,
    base_level: float,
    spreading: float = 0.0,
    penalty: float = 0.0,
    noise: float = 0.0,
) -> float:
    """``A_i = B_i + Σ W·S + P + ε``.

    Noise is passed in rather than drawn here so that a caller can run the same
    retrieval with and without it. A stochastic default would make every
    downstream measurement unreproducible by construction.
    """
    if base_level == -math.inf:
        return -math.inf
    return base_level + spreading + penalty + noise


def retrieval_probability(
    activation_value: float,
    *,
    threshold: float = DEFAULT_PARAMETERS.threshold,
    noise_s: float = DEFAULT_PARAMETERS.noise_s,
) -> float:
    """``1 / (1 + exp(-(A - τ)/s))`` — the logistic recall curve.

    The standard logistic approximation to activation noise. Guarded against
    overflow so an activation far below threshold returns 0.0 instead of
    raising.
    """
    if activation_value == -math.inf:
        return 0.0
    z = (activation_value - threshold) / noise_s
    if z < -60.0:
        return 0.0
    if z > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def retrieval_latency(
    activation_value: float,
    *,
    latency_factor: float = DEFAULT_PARAMETERS.latency_factor,
    threshold: float = DEFAULT_PARAMETERS.threshold,
) -> float:
    """``T = F·e^-A`` seconds, the predicted time to retrieve this chunk.

    A chunk below threshold is not retrieved, and the architecture still spends
    time discovering that: the failure latency is ``F·e^-τ``, the time to reach
    the threshold itself. Reporting failure as zero-cost is the common mistake
    and it makes the model faster than the system it describes.
    """
    effective = max(activation_value, threshold)
    if effective == -math.inf:
        return latency_factor * math.exp(-threshold)
    return latency_factor * math.exp(-effective)


def expected_latency(
    activation_value: float,
    *,
    params: ActrParameters = DEFAULT_PARAMETERS,
) -> float:
    """Latency averaged over whether retrieval succeeds at all.

    ``p·T(A) + (1-p)·T(τ)``. This is the quantity to compare against a measured
    wall-clock time, because a measurement cannot condition on success it did
    not observe.
    """
    p = retrieval_probability(
        activation_value, threshold=params.threshold, noise_s=params.noise_s
    )
    hit = retrieval_latency(
        activation_value,
        latency_factor=params.latency_factor,
        threshold=params.threshold,
    )
    miss = params.latency_factor * math.exp(-params.threshold)
    return p * hit + (1.0 - p) * miss


@dataclass(frozen=True)
class SensitivityBand:
    """Latency range produced by perturbing one parameter."""

    parameter: str
    low: float
    high: float
    nominal: float
    perturbation: float = field(default=0.0)

    @property
    def spread_ratio(self) -> float:
        """How many times wider the band is than the nominal prediction."""
        if self.nominal <= 0.0:
            return math.inf
        return (self.high - self.low) / self.nominal


def latency_sensitivity(
    activation_value: float,
    *,
    params: ActrParameters = DEFAULT_PARAMETERS,
    relative_perturbation: float = 0.10,
) -> list[SensitivityBand]:
    """How much each parameter alone moves the latency prediction.

    The honest companion to any number this module produces. ACT-R's standing
    criticism is that its parameter count lets it fit anything; the answer is
    not to deny it but to publish the band. A prediction whose band is wider
    than the effect it claims to explain has not explained it.
    """
    bands: list[SensitivityBand] = []
    nominal = expected_latency(activation_value, params=params)
    for name in ("decay", "noise_s", "threshold", "latency_factor"):
        value = getattr(params, name)
        delta = abs(value) * relative_perturbation or relative_perturbation
        lows: list[float] = []
        for signed in (-delta, +delta):
            try:
                perturbed = replace(params, **{name: value + signed})
            except ValueError:
                continue  # perturbation left the parameter's valid domain
            lows.append(expected_latency(activation_value, params=perturbed))
        if not lows:
            continue
        bands.append(
            SensitivityBand(
                parameter=name,
                low=min(lows),
                high=max(lows),
                nominal=nominal,
                perturbation=delta,
            )
        )
    return bands
