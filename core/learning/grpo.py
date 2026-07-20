"""Verifier-driven RL over reasoning trajectories (CP229).

Anima Rationis line 511 records the existence proof this module chases:
QwQ-32B reached DeepSeek-R1-comparable reasoning through RL over a 32B
foundation, using correctness verifiers for mathematics and execution
feedback for code. Same parameter class as Aura's resident cortex. The
document lists verifier-driven RL among the components that make "the real
leap" possible, and it is the one that was never built.

Group Relative Policy Optimization: sample K completions per prompt, grade
each with a program, and use the group's own mean as the baseline. No value
network -- which matters here because a critic is another thing that can be
wrong, and this project has already been bitten repeatedly by machinery
that appeared to work while measuring nothing.

    advantage_i = (r_i - mean(r)) / std(r)
    loss        = -mean(advantage_i * logprob_i) + beta * KL(policy || ref)

Three disciplines, each answering a specific way RL produces fake progress:

* **Degenerate groups are detected, not silently trained on.** If all K
  completions are correct (or all wrong), every advantage is zero and the
  step teaches nothing. Averaged into a loss curve that looks like smooth
  convergence; it is actually a run with no signal.
* **Reward comes only from the grader.** Never from model confidence,
  never from a learned reward head. Optimizing confidence strengthens
  confident mistakes -- Anima Rationis line 220 warns of exactly this.
* **KL to the reference policy is enforced.** Unconstrained RL on a narrow
  verifier collapses diversity and trades general ability for the graded
  metric, which is the interference failure the document devotes section
  6 to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

GRPO_SCHEMA = "aura.grpo.v1"

# Below this spread, the group's rewards are effectively identical and the
# normalized advantage is numerical noise rather than learning signal.
MIN_REWARD_STD = 1e-4


@dataclass(frozen=True)
class GRPOConfig:
    """Group size, KL strength, and the guards on both."""

    group_size: int = 8
    kl_coefficient: float = 0.04
    # Clipping keeps one lucky trajectory from dominating a group.
    advantage_clip: float = 4.0
    # Fraction of degenerate groups above which a run is not learning and
    # should say so rather than continue producing a tidy loss curve.
    max_degenerate_fraction: float = 0.7

    def __post_init__(self) -> None:
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError(
                "group_size must be at least 2: a group of one has no "
                "baseline, which is the entire mechanism"
            )
        for name in ("kl_coefficient", "advantage_clip", "max_degenerate_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 100.0
            ):
                raise ValueError(f"{name} must be a non-negative finite number")
        if self.advantage_clip <= 0.0:
            raise ValueError("advantage_clip must be positive")

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": GRPO_SCHEMA,
            "group_size": self.group_size,
            "kl_coefficient": self.kl_coefficient,
            "advantage_clip": self.advantage_clip,
            "max_degenerate_fraction": self.max_degenerate_fraction,
        }


def group_advantages(
    rewards: Sequence[float], *, clip: float = 4.0
) -> dict[str, Any]:
    """Group-relative advantages, with degeneracy reported explicitly.

    Returns ``advantages`` plus the diagnosis. A caller that ignores
    ``degenerate`` will train on a vector of zeros and see a loss curve
    that looks like convergence.
    """
    if len(rewards) < 2:
        raise ValueError("a group needs at least two completions")
    values = [float(r) for r in rewards]
    if any(math.isnan(v) or math.isinf(v) for v in values):
        raise ValueError("rewards must be finite; a grader returned garbage")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    degenerate = std < MIN_REWARD_STD
    if degenerate:
        # Every completion earned the same grade. There is no preference to
        # learn from, and manufacturing one from noise would be worse.
        advantages = [0.0] * len(values)
    else:
        advantages = [
            max(-clip, min(clip, (v - mean) / std)) for v in values
        ]
    return {
        "schema": GRPO_SCHEMA,
        "advantages": advantages,
        "mean_reward": round(mean, 6),
        "reward_std": round(std, 6),
        "degenerate": bool(degenerate),
        "all_correct": bool(degenerate and mean >= 1.0 - 1e-9),
        "all_wrong": bool(degenerate and mean <= 1e-9),
    }


def step_scores_from_ce(ce_trail: Sequence[float]) -> list[float]:
    """Map a per-step answer-CE trail to bounded higher-is-better scores."""
    scores: list[float] = []
    for value in ce_trail:
        ce = float(value)
        if math.isnan(ce) or math.isinf(ce) or ce < 0.0:
            raise ValueError("CE trail must contain finite non-negative values")
        scores.append(math.exp(-ce))
    return scores


def trajectory_shaped_rewards(
    final_rewards: Sequence[float],
    step_score_trails: Sequence[Sequence[float]],
    *,
    shaping_weight: float = 0.25,
) -> dict[str, Any]:
    """Blend verifier outcomes with latent-trajectory credit (RLTT-style).

    Final-outcome GRPO gives poor credit assignment to internal latent
    steps: a group where every completion fails is a zero-advantage group,
    even when some trajectories moved decisively TOWARD the answer before
    missing it. Distributing bounded credit over the internal trajectory —
    scored by how much each window pass improved the answer's decodability
    (``latent_step_answer_ce``) — turns those wasted groups into signal.

    Honesty contract:
    - the verifier stays the last word: shaping is bounded to
      ±``shaping_weight`` and never manufactured from a learned reward
      head, only from measured per-step answer scores;
    - whenever shaping REORDERS two completions relative to their final
      rewards, the receipt says so (``shaping_reordered``) — silent
      reordering would be reward hacking's front door;
    - a trail too short to have increments contributes zero shaping.
    """
    if len(final_rewards) != len(step_score_trails):
        raise ValueError("final rewards and step trails must align")
    if not final_rewards:
        raise ValueError("nothing to shape")
    if (
        isinstance(shaping_weight, bool)
        or not isinstance(shaping_weight, (int, float))
        or not 0.0 <= float(shaping_weight) <= 0.49
    ):
        raise ValueError(
            "shaping_weight must be inside [0, 0.49]: at 0.5+ the shaping "
            "could outweigh a full verifier grade gap"
        )
    rows: list[dict[str, Any]] = []
    shaped: list[float] = []
    for reward, trail in zip(final_rewards, step_score_trails, strict=True):
        final = float(reward)
        if math.isnan(final) or math.isinf(final):
            raise ValueError("final rewards must be finite")
        scores = [float(score) for score in trail]
        if any(math.isnan(s) or math.isinf(s) for s in scores):
            raise ValueError("step scores must be finite")
        if any(not 0.0 <= s <= 1.0 for s in scores):
            raise ValueError("step scores must be inside [0, 1]")
        improvements = [
            after - before for before, after in zip(scores, scores[1:])
        ]
        shaping = (
            float(shaping_weight) * (sum(improvements) / len(improvements))
            if improvements
            else 0.0
        )
        shaped.append(final + shaping)
        rows.append(
            {
                "final_reward": round(final, 6),
                "shaping": round(shaping, 6),
                "shaped_reward": round(final + shaping, 6),
                "steps": len(scores),
            }
        )
    order_by_final = sorted(
        range(len(final_rewards)), key=lambda i: (float(final_rewards[i]), i)
    )
    order_by_shaped = sorted(
        range(len(shaped)), key=lambda i: (shaped[i], i)
    )
    return {
        "schema": GRPO_SCHEMA,
        "shaped_rewards": shaped,
        "rows": rows,
        "shaping_weight": float(shaping_weight),
        "shaping_reordered": order_by_final != order_by_shaped,
    }


def sequence_logprob(logits: Any, tokens: Any) -> Any:
    """Sum of token log-probabilities for one completion.

    Computed in float32: the same fp16 reduction overflow that made the
    recurrence dynamics read as a dead loop would here silently distort
    every advantage weighting.
    """
    import mlx.core as mx
    import mlx.nn as nn

    losses = nn.losses.cross_entropy(
        logits.astype(mx.float32), tokens, reduction="none"
    )
    return -mx.sum(losses)


def grpo_loss(
    policy_logprobs: Sequence[Any],
    advantages: Sequence[float],
    *,
    reference_logprobs: Sequence[Any] | None = None,
    kl_coefficient: float = 0.04,
) -> tuple[Any, dict[str, Any]]:
    """Advantage-weighted policy gradient with a KL leash.

    ``reference_logprobs`` are the frozen pre-RL policy's. Without the
    leash, RL against a narrow verifier will happily trade everything the
    model could already do for the one thing being graded.
    """
    import mlx.core as mx

    if len(policy_logprobs) != len(advantages):
        raise ValueError("logprobs and advantages must align")
    if not policy_logprobs:
        raise ValueError("nothing to optimize")

    weighted = [
        -float(advantage) * logprob
        for logprob, advantage in zip(policy_logprobs, advantages)
    ]
    policy_loss = mx.stack(weighted).mean()

    kl_value = 0.0
    total = policy_loss
    if reference_logprobs is not None:
        if len(reference_logprobs) != len(policy_logprobs):
            raise ValueError("reference logprobs must align with policy")
        # k3 estimator: exp(d) - d - 1, non-negative and lower variance
        # than the naive difference.
        deltas = [
            reference - policy
            for policy, reference in zip(policy_logprobs, reference_logprobs)
        ]
        kl_terms = [mx.exp(d) - d - 1.0 for d in deltas]
        kl = mx.stack(kl_terms).mean()
        total = policy_loss + float(kl_coefficient) * kl
        kl_value = float(kl)

    return total, {
        "schema": GRPO_SCHEMA,
        "policy_loss": float(policy_loss),
        "kl": round(kl_value, 6),
        "total": float(total),
        "group_size": len(policy_logprobs),
    }


@dataclass
class GRPOTelemetry:
    """Is this run learning, or producing a tidy curve over no signal?"""

    groups: int = 0
    degenerate: int = 0
    all_correct: int = 0
    all_wrong: int = 0
    reward_sum: float = 0.0

    def observe(self, report: dict[str, Any]) -> None:
        self.groups += 1
        self.reward_sum += float(report["mean_reward"])
        if report["degenerate"]:
            self.degenerate += 1
            self.all_correct += int(report["all_correct"])
            self.all_wrong += int(report["all_wrong"])

    def verdict(self, config: GRPOConfig) -> dict[str, Any]:
        if not self.groups:
            return {
                "schema": GRPO_SCHEMA,
                "groups": 0,
                "learning_signal": False,
                "reason": "no groups observed",
            }
        degenerate_fraction = self.degenerate / self.groups
        usable = self.groups - self.degenerate
        diagnosis = "healthy"
        if degenerate_fraction > config.max_degenerate_fraction:
            # Both failure modes look identical in the loss curve and need
            # opposite fixes: harder tasks vs. easier ones.
            if self.all_wrong > self.all_correct:
                diagnosis = "tasks_too_hard: the model never succeeds, so "
                "there is nothing to reinforce"
            else:
                diagnosis = "tasks_too_easy: the model always succeeds, so "
                "there is nothing to improve"
        return {
            "schema": GRPO_SCHEMA,
            "groups": self.groups,
            "usable_groups": usable,
            "degenerate_fraction": round(degenerate_fraction, 4),
            "all_correct_groups": self.all_correct,
            "all_wrong_groups": self.all_wrong,
            "mean_reward": round(self.reward_sum / self.groups, 4),
            "learning_signal": bool(
                degenerate_fraction <= config.max_degenerate_fraction
            ),
            "diagnosis": diagnosis,
        }


def reward_from_verdict(verdict: dict[str, Any], *, format_credit: float = 0.0) -> float:
    """Reward is correctness. Format credit is optional and small.

    Partial credit for well-formed-but-wrong answers is a Goodhart hazard:
    it is far easier to learn the format than the reasoning, so a large
    format term produces a model that is beautifully formatted and wrong.
    """
    if not isinstance(verdict, dict):
        raise ValueError("verdict must be a grader result")
    if not 0.0 <= format_credit <= 0.2:
        raise ValueError(
            "format_credit must stay within [0, 0.2]: a larger share makes "
            "formatting more learnable than correctness"
        )
    if verdict.get("correct"):
        return 1.0
    parsed = verdict.get("parsed")
    answered = parsed is not None and parsed != [] and parsed != ""
    return format_credit if answered else 0.0


__all__ = [
    "GRPO_SCHEMA",
    "MIN_REWARD_STD",
    "GRPOConfig",
    "GRPOTelemetry",
    "group_advantages",
    "grpo_loss",
    "reward_from_verdict",
    "sequence_logprob",
    "step_scores_from_ce",
    "trajectory_shaped_rewards",
]
