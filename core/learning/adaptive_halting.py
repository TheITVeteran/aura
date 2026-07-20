"""Spend thought where it is needed (CP230).

Anima Rationis makes the compute-scaling curve the central success
criterion -- accuracy should rise with recurrent steps on problems that
need them (line 403), and the Will should allocate depth by expected
difficulty, uncertainty, stakes and expected value of further computation
(line 765). Neither is possible with a fixed T: a constant depth is not an
allocation policy, and a flat curve measured at constant depth says nothing
about whether depth COULD have helped.

The document also names the failure this must avoid: latent "overthinking"
(line 425), where excessive recurrence degrades results. So halting is not
merely an efficiency optimization. Running a contracting loop past its
fixed point burns compute and can make answers worse, and CP226 measured
exactly that shape -- deltas 0.55, 0.50, 0.32 with accuracy falling.

The head is zero-initialized, so an untrained model never halts early and
behaves exactly as it did at fixed depth. Allocation is something the model
earns, not something imposed on it at attach time.

The claim this module makes falsifiable: **allocated depth should
correlate with required depth.** A halting policy that spends uniformly, or
spends by prompt length, is not allocating thought -- and that is
measurable rather than assertable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

ADAPTIVE_HALTING_SCHEMA = "aura.adaptive_halting.v1"


class HaltingHead:
    """Reads the latent state, emits P(this is enough thinking).

    Deliberately tiny. A large halting network would be a second model
    whose own errors are indistinguishable from the cortex's, and it would
    have to be trained on the same scarce verified outcomes.
    """

    def __init__(self, hidden_size: int, *, threshold: float = 0.5) -> None:
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 < float(threshold) < 1.0
        ):
            raise ValueError("threshold must be inside (0, 1)")
        import mlx.core as mx

        self.hidden_size = hidden_size
        self.threshold = float(threshold)
        # Zero weight => logit 0 => p = 0.5 everywhere. Combined with the
        # minimum-steps floor, an untrained head never changes behaviour.
        self.weight = mx.zeros((hidden_size, 1))
        self.bias = mx.zeros((1,))

    def is_identity(self) -> bool:
        import mlx.core as mx

        return bool(mx.all(self.weight == 0) and mx.all(self.bias == 0))

    def halt_probability(self, state: Any) -> Any:
        """P(stop) from the pooled latent state."""
        import mlx.core as mx

        if state.shape[-1] != self.hidden_size:
            raise ValueError("state width does not match the halting head")
        pooled = mx.mean(state.astype(mx.float32), axis=tuple(range(state.ndim - 1)))
        logit = mx.reshape(pooled, (1, -1)) @ self.weight + self.bias
        return mx.sigmoid(mx.reshape(logit, ()))

    def parameters(self) -> dict[str, Any]:
        return {"weight": self.weight, "bias": self.bias}

    def parameter_count(self) -> int:
        return int(sum(p.size for p in self.parameters().values()))


@dataclass(frozen=True)
class HaltingPolicy:
    """Bounds on how much thought may be spent."""

    min_steps: int = 1
    max_steps: int = 8
    # Price per step, in units of loss. Anima Rationis line 765 wants
    # allocation by expected value of computation; a price is how that
    # trade-off becomes explicit instead of implicit.
    ponder_cost: float = 0.01

    def __post_init__(self) -> None:
        for name in ("min_steps", "max_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        if (
            isinstance(self.ponder_cost, bool)
            or not isinstance(self.ponder_cost, (int, float))
            or not 0.0 <= float(self.ponder_cost) <= 1.0
        ):
            raise ValueError("ponder_cost must be inside [0, 1]")


def decide_steps(
    head: HaltingHead,
    states: Sequence[Any],
    policy: HaltingPolicy,
) -> dict[str, Any]:
    """Walk the trajectory and stop when the head says enough.

    Returns the chosen depth plus the per-step probabilities, because a
    halting decision nobody can inspect is indistinguishable from a
    constant.
    """
    if not states:
        raise ValueError("no states to evaluate")
    probabilities: list[float] = []
    chosen = min(len(states), policy.max_steps)
    halted_early = False
    for index, state in enumerate(states[: policy.max_steps], start=1):
        probability = float(head.halt_probability(state))
        probabilities.append(round(probability, 6))
        if index >= policy.min_steps and probability >= head.threshold:
            chosen = index
            halted_early = index < min(len(states), policy.max_steps)
            break
    return {
        "schema": ADAPTIVE_HALTING_SCHEMA,
        "steps": int(chosen),
        "halt_probabilities": probabilities,
        "halted_early": bool(halted_early),
        "min_steps": policy.min_steps,
        "max_steps": policy.max_steps,
    }


def ponder_loss(
    step_losses: Sequence[Any],
    halt_probabilities: Sequence[Any],
    policy: HaltingPolicy,
) -> tuple[Any, dict[str, Any]]:
    """Expected loss under the halting distribution, plus a price on depth.

    The head learns to stop where stopping is cheap in ACCURACY, not where
    stopping is cheap in compute -- the ponder cost is a tiebreaker, not the
    objective. An objective dominated by the compute term produces a model
    that always halts immediately and reports excellent efficiency.
    """
    import mlx.core as mx

    if len(step_losses) != len(halt_probabilities):
        raise ValueError("step losses and halt probabilities must align")
    if not step_losses:
        raise ValueError("nothing to score")

    # Probability of stopping AT step i = p_i * prod(1 - p_j) for j < i.
    remaining = mx.array(1.0)
    weights: list[Any] = []
    for index, probability in enumerate(halt_probabilities):
        last = index == len(halt_probabilities) - 1
        stop_here = remaining if last else remaining * probability
        weights.append(stop_here)
        remaining = remaining * (1.0 - probability)

    expected = mx.stack(
        [weight * loss for weight, loss in zip(weights, step_losses)]
    ).sum()
    expected_steps = mx.stack(
        [weight * float(index + 1) for index, weight in enumerate(weights)]
    ).sum()
    total = expected + policy.ponder_cost * expected_steps
    return total, {
        "schema": ADAPTIVE_HALTING_SCHEMA,
        "expected_loss": float(expected),
        "expected_steps": round(float(expected_steps), 4),
        "ponder_cost": policy.ponder_cost,
        "total": float(total),
    }


def allocation_report(
    allocations: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Does spent depth track REQUIRED depth?

    ``allocations`` is (required_depth, allocated_steps). This is the
    falsifiable core of the whole halting claim: a policy that spends
    uniformly, or spends by prompt length, is not allocating thought. The
    correlation says which one is happening, and a flat allocation is
    reported as such rather than defended.
    """
    if len(allocations) < 3:
        raise ValueError("need at least three allocations to judge a policy")
    required = [float(r) for r, _ in allocations]
    spent = [float(s) for _, s in allocations]
    n = len(allocations)
    mean_r, mean_s = sum(required) / n, sum(spent) / n
    cov = sum((r - mean_r) * (s - mean_s) for r, s in zip(required, spent)) / n
    var_r = sum((r - mean_r) ** 2 for r in required) / n
    var_s = sum((s - mean_s) ** 2 for s in spent) / n
    if var_s < 1e-12:
        return {
            "schema": ADAPTIVE_HALTING_SCHEMA,
            "correlation": 0.0,
            "allocates_by_difficulty": False,
            "reason": "constant allocation is not a policy",
            "mean_steps": round(mean_s, 4),
        }
    correlation = cov / math.sqrt(max(var_r * var_s, 1e-12))
    return {
        "schema": ADAPTIVE_HALTING_SCHEMA,
        "correlation": round(correlation, 4),
        "mean_steps": round(mean_s, 4),
        "mean_required": round(mean_r, 4),
        # A weak positive correlation is not allocation; it is noise with a
        # sign. The bar is deliberately not "greater than zero".
        "allocates_by_difficulty": bool(correlation >= 0.3),
    }


def overthinking_report(
    step_losses: Sequence[float],
) -> dict[str, Any]:
    """Did extra steps make it worse?

    Anima Rationis line 425 names latent overthinking as a real observed
    failure. CP226 measured its shape: the loop kept moving (deltas 0.55,
    0.50, 0.32) while accuracy fell. Reported explicitly so 'the state is
    still changing' is never mistaken for 'it is still thinking'.
    """
    values = [float(v) for v in step_losses]
    if len(values) < 2:
        raise ValueError("need at least two steps to detect overthinking")
    best = min(values)
    best_step = values.index(best) + 1
    return {
        "schema": ADAPTIVE_HALTING_SCHEMA,
        "best_step": best_step,
        "best_loss": round(best, 6),
        "final_loss": round(values[-1], 6),
        "overthinks": bool(values[-1] > best + 1e-9),
        "wasted_steps": len(values) - best_step,
    }


__all__ = [
    "ADAPTIVE_HALTING_SCHEMA",
    "HaltingHead",
    "HaltingPolicy",
    "allocation_report",
    "decide_steps",
    "overthinking_report",
    "ponder_loss",
]
