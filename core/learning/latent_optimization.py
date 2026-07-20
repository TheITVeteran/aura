"""Gradient descent over thoughts, not weights (CP231).

Anima Rationis calls this the most radical part (line 173): treat the
latent workspace as an optimizable object and improve it directly while
every weight stays frozen.

    Z <- Z + eta * grad_Z S(Z)
    S(Z) = lv*V(Z) + lc*C(Z) + lr*R(Z) - ld*D(Z, Z_0)

    V  verified correctness / constraint satisfaction
    C  agreement across independently derived consequences
    R  reconstruction of the problem and its assumptions
    D  distance from the model's normal activation manifold

Two warnings from the source document are load-bearing here and are
enforced in code rather than left to discipline:

* **Line 220 -- never optimize confidence.** "Merely pushing the model
  toward high confidence would often strengthen confident mistakes." A
  score built from the model's own certainty makes wrong answers more
  confident and reports the improvement as progress. So an objective with
  no verifier and no consistency term is REFUSED, not merely discouraged.
* **Line 453 -- the random perturbation control is essential.** Moving a
  latent state by a meaningful magnitude changes the output. If a matched
  random perturbation produces the same gain, the optimizer learned
  nothing and the whole mechanism is a noise generator. The control ships
  in this module rather than in a test, because it has to run on every
  real claim.

The manifold term D is what keeps this from being adversarial-example
generation against the model's own coda: it is easy to find a Z that
maximizes any score and lies far outside anything the decoder was trained
to read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

LATENT_OPTIMIZATION_SCHEMA = "aura.latent_optimization.v1"


@dataclass(frozen=True)
class LatentObjective:
    """Weights on the score terms, with the honesty constraints enforced."""

    verifier_weight: float = 1.0
    consistency_weight: float = 0.5
    reconstruction_weight: float = 0.25
    # Distance from the starting activation manifold. Without this, the
    # optimizer happily produces states the coda has never seen and cannot
    # decode -- adversarial examples against the model's own decoder.
    manifold_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "verifier_weight",
            "consistency_weight",
            "reconstruction_weight",
            "manifold_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 100.0
            ):
                raise ValueError(f"{name} must be a non-negative finite number")
        if self.verifier_weight <= 0.0 and self.consistency_weight <= 0.0:
            raise ValueError(
                "an objective with neither a verifier nor a consistency term "
                "optimizes model confidence, which strengthens confident "
                "mistakes rather than correcting them"
            )
        if self.manifold_weight <= 0.0:
            raise ValueError(
                "manifold_weight must be positive: without it the optimizer "
                "produces states outside anything the decoder was trained to "
                "read, which is adversarial-example generation"
            )


def manifold_distance(state: Any, anchor: Any) -> Any:
    """How far the optimized thought has drifted from where it started."""
    import mlx.core as mx

    if state.shape != anchor.shape:
        raise ValueError("state and anchor must share a shape")
    wide_state = state.astype(mx.float32)
    wide_anchor = anchor.astype(mx.float32)
    # Epsilon INSIDE the sqrt. d/dx sqrt(x) is infinite at x=0, so a state
    # that starts exactly at the anchor -- which is every optimization's
    # first step -- produced a NaN gradient and the run silently made no
    # progress at all.
    anchor_energy = mx.mean(mx.square(wide_anchor))
    # A near-zero anchor would otherwise divide by ~1e-6 and report an
    # enormous drift for any movement whatsoever.
    scale = mx.sqrt(anchor_energy + 1e-6)
    displacement = mx.sqrt(mx.mean(mx.square(wide_state - wide_anchor)) + 1e-12)
    return displacement / scale


def optimize_latent(
    state: Any,
    score_fn: Callable[[Any], Any],
    *,
    objective: LatentObjective,
    steps: int = 8,
    learning_rate: float = 0.05,
    max_drift: float = 0.5,
) -> dict[str, Any]:
    """Ascend ``score_fn`` in latent space with the weights frozen.

    ``score_fn`` must be verifier-grounded; this function adds only the
    manifold penalty. Returns the optimized state plus the trajectory,
    because an optimization nobody can inspect is indistinguishable from
    noise with a good story.
    """
    import mlx.core as mx

    if type(steps) is not int or not 1 <= steps <= 256:
        raise ValueError("steps must be inside [1, 256]")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not 0.0 < float(learning_rate) <= 1.0
    ):
        raise ValueError("learning_rate must be inside (0, 1]")
    if not 0.0 < float(max_drift) <= 5.0:
        raise ValueError("max_drift must be inside (0, 5]")

    anchor = mx.stop_gradient(state)

    def penalized(current: Any) -> Any:
        return score_fn(current) - objective.manifold_weight * manifold_distance(
            current, anchor
        )

    grad_fn = mx.grad(penalized)
    current = state
    trajectory: list[float] = [float(score_fn(current))]
    stopped_early = False
    for _ in range(steps):
        gradient = grad_fn(current)
        candidate = current + learning_rate * gradient
        drift = float(manifold_distance(candidate, anchor))
        if drift > max_drift:
            # Refuse the step rather than clamp it silently: a run that hit
            # the wall should say so, since the remaining budget was not
            # actually spent.
            stopped_early = True
            break
        current = candidate
        trajectory.append(float(score_fn(current)))
    return {
        "schema": LATENT_OPTIMIZATION_SCHEMA,
        "state": current,
        "score_trajectory": [round(v, 6) for v in trajectory],
        "initial_score": round(trajectory[0], 6),
        "final_score": round(trajectory[-1], 6),
        "improved": bool(trajectory[-1] > trajectory[0]),
        "drift": round(float(manifold_distance(current, anchor)), 6),
        "steps_taken": len(trajectory) - 1,
        "stopped_early": stopped_early,
    }


def matched_random_control(
    state: Any,
    score_fn: Callable[[Any], Any],
    *,
    drift: float,
    trials: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """The control that decides whether optimization did anything.

    Perturb the state by the SAME magnitude the optimizer moved it, in
    random directions, and score those. Any latent movement of meaningful
    size changes the output; the question is whether the DIRECTION was
    earned. Anima Rationis line 453 calls this control essential, and it is
    the difference between a result and a coincidence.
    """
    import mlx.core as mx

    if not 0.0 <= float(drift) <= 10.0:
        raise ValueError("drift must be a finite non-negative magnitude")
    if type(trials) is not int or not 1 <= trials <= 128:
        raise ValueError("trials must be inside [1, 128]")

    anchor = mx.stop_gradient(state)
    scale = float(mx.sqrt(mx.mean(mx.square(anchor.astype(mx.float32))))) + 1e-6
    scores: list[float] = []
    for trial in range(trials):
        noise = mx.random.normal(anchor.shape, key=mx.random.key(seed + trial))
        noise_rms = float(mx.sqrt(mx.mean(mx.square(noise)))) + 1e-9
        # Match the optimizer's displacement magnitude exactly.
        perturbed = anchor + noise * (drift * scale / noise_rms)
        scores.append(float(score_fn(perturbed)))
    mean = sum(scores) / len(scores)
    return {
        "schema": LATENT_OPTIMIZATION_SCHEMA,
        "trials": trials,
        "mean_score": round(mean, 6),
        "best_score": round(max(scores), 6),
        "matched_drift": round(float(drift), 6),
    }


def optimization_verdict(
    optimized: dict[str, Any],
    control: dict[str, Any],
    *,
    margin: float = 0.05,
) -> dict[str, Any]:
    """Did directed optimization beat matched random movement?

    Compared against the control's BEST trial, not its mean. Beating the
    average of random noise is a low bar that a lucky direction clears; the
    honest question is whether the optimizer found something random search
    of the same magnitude did not.
    """
    gain = optimized["final_score"] - optimized["initial_score"]
    control_gain = control["best_score"] - optimized["initial_score"]
    beats = gain > control_gain + margin
    return {
        "schema": LATENT_OPTIMIZATION_SCHEMA,
        "optimized_gain": round(gain, 6),
        "control_best_gain": round(control_gain, 6),
        "margin": margin,
        "beats_random": bool(beats),
        "verdict": (
            "directed latent optimization"
            if beats
            else "indistinguishable from matched random perturbation"
        ),
    }


def consistency_score(answers: list[Any]) -> float:
    """Agreement across independently derived consequences (the C term).

    A verifier-free signal that is still not confidence: it asks whether
    several derivations AGREE, which a confidently wrong model fails when
    its derivations are genuinely independent.
    """
    if len(answers) < 2:
        raise ValueError("consistency needs at least two derivations")
    keyed = [str(a) for a in answers]
    counts: dict[str, int] = {}
    for value in keyed:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.values()) / len(keyed)


__all__ = [
    "LATENT_OPTIMIZATION_SCHEMA",
    "LatentObjective",
    "consistency_score",
    "manifold_distance",
    "matched_random_control",
    "optimization_verdict",
    "optimize_latent",
]
