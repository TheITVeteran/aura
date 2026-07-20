"""Process critic: does this intermediate state deserve credit? (CP221)

The largest remaining blocker. Final-answer loss can say WHETHER a
trajectory ended correctly; it cannot say WHICH internal step helped. That
is poor credit assignment over latent recurrence, and it is why every
mechanism built so far produces motion nobody can grade -- the branches,
the depth ladder, the protected memory all move state without any signal
that the movement was progress.

ByteDance's latent-trajectory reward work reports substantial gains over
outcome-only RL by distributing credit across the whole internal
trajectory, which is the same conclusion reached here from the opposite
direction (a period-2 oscillation and a fixed-point collapse both went
unnoticed because only endpoints were scored).

The critic predicts, from an intermediate latent state, the probability
that this trajectory ends in a VERIFIED-correct answer. Three disciplines
keep it honest:

* **Trained on real outcomes only.** Labels come from trajectories whose
  answers were actually checked, never from the critic's own opinion.
* **Calibration is measured, not assumed** (Brier score + reliability
  bins). A confident critic that is wrong is worse than no critic.
* **Drift is detected explicitly.** Google's TEMPO finds test-time
  adaptation plateaus when a self-generated reward drifts, and that
  periodic recalibration against ground truth restores gains. So this
  critic reports when its calibration has decayed rather than continuing
  to be trusted silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

PROCESS_CRITIC_SCHEMA = "aura.process_critic.v1"

# Below this many graded observations the critic reports itself unusable
# rather than emitting confident noise.
MIN_OBSERVATIONS = 24
# Brier score above this means predictions are worse than useful; a
# constant 0.5 predictor scores 0.25, so anything near it carries no signal.
MAX_TRUSTWORTHY_BRIER = 0.22


@dataclass
class CriticObservation:
    """One graded (state, outcome) pair from a completed trajectory."""

    features: tuple[float, ...]
    step: int
    verified_correct: bool

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("observation needs at least one feature")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if type(self.verified_correct) is not bool:
            raise ValueError(
                "verified_correct must be a real graded outcome, not a score"
            )


def state_features(state: Any, *, step: int, max_depth: int) -> tuple[float, ...]:
    """Compact, cheap descriptors of a latent state.

    Deliberately small and interpretable rather than a learned encoder:
    a large critic head trained on a few hundred trajectories would overfit
    long before it generalized, and an uninterpretable critic cannot be
    audited when it drifts.
    """
    import mlx.core as mx

    flat = mx.reshape(state, (-1,))
    norm = float(mx.linalg.norm(flat))
    scale = max(norm, 1e-9)
    per_slot = mx.sqrt(mx.mean(mx.square(state), axis=-1))
    spread = float(mx.std(per_slot))
    mean_abs = float(mx.mean(mx.abs(flat)))
    progress = (step / max_depth) if max_depth > 0 else 0.0
    return (
        1.0,                       # bias
        min(norm / 100.0, 10.0),   # magnitude, bounded
        spread / scale * 100.0,    # slot differentiation
        mean_abs,                  # activation density
        progress,                  # how far through the budget
    )


@dataclass
class ProcessCritic:
    """Logistic value head over latent-state features.

    Predicts P(this trajectory ends verified-correct). Fit by gradient
    descent on graded observations; never on its own outputs.
    """

    weights: list[float] = field(default_factory=list)
    observations: int = 0
    _fitted: bool = False

    def predict(self, features: Sequence[float]) -> float:
        import math

        if not self._fitted or not self.weights:
            return 0.5  # honest ignorance, not optimism
        if len(features) != len(self.weights):
            raise ValueError("feature width does not match the fitted critic")
        z = sum(w * f for w, f in zip(self.weights, features, strict=True))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(
        self,
        observations: Sequence[CriticObservation],
        *,
        epochs: int = 200,
        learning_rate: float = 0.1,
        l2: float = 1e-3,
    ) -> dict[str, Any]:
        """Fit on graded outcomes and REPORT whether the result is usable."""
        if len(observations) < MIN_OBSERVATIONS:
            self._fitted = False
            return {
                "schema": PROCESS_CRITIC_SCHEMA,
                "fitted": False,
                "reason": "insufficient_graded_observations",
                "observations": len(observations),
                "required": MIN_OBSERVATIONS,
            }
        width = len(observations[0].features)
        if any(len(row.features) != width for row in observations):
            raise ValueError("observations disagree on feature width")
        outcomes = {row.verified_correct for row in observations}
        if len(outcomes) < 2:
            self._fitted = False
            return {
                "schema": PROCESS_CRITIC_SCHEMA,
                "fitted": False,
                "reason": "single_outcome_class",
                "observations": len(observations),
            }

        import math

        weights = [0.0] * width
        for _ in range(max(1, int(epochs))):
            gradients = [0.0] * width
            for row in observations:
                z = sum(
                    w * f for w, f in zip(weights, row.features, strict=True)
                )
                z = max(-30.0, min(30.0, z))
                prediction = 1.0 / (1.0 + math.exp(-z))
                error = prediction - (1.0 if row.verified_correct else 0.0)
                for index, feature in enumerate(row.features):
                    gradients[index] += error * feature
            count = len(observations)
            weights = [
                w - learning_rate * (g / count + l2 * w)
                for w, g in zip(weights, gradients, strict=True)
            ]
        self.weights = weights
        self.observations = len(observations)
        self._fitted = True
        report = self.calibration(observations)
        report.update(
            {
                "schema": PROCESS_CRITIC_SCHEMA,
                "fitted": True,
                "observations": len(observations),
                "feature_width": width,
            }
        )
        return report

    def calibration(
        self, observations: Sequence[CriticObservation], *, bins: int = 5
    ) -> dict[str, Any]:
        """Brier score, reliability bins, and a usability verdict.

        A critic is only worth consulting if it beats the constant
        predictor it would otherwise be replaced by.
        """
        if not observations:
            return {"brier": 1.0, "trustworthy": False, "reason": "no_data"}
        predictions = [self.predict(row.features) for row in observations]
        truths = [1.0 if row.verified_correct else 0.0 for row in observations]
        brier = sum(
            (p - t) ** 2 for p, t in zip(predictions, truths, strict=True)
        ) / len(observations)
        base_rate = sum(truths) / len(truths)
        baseline_brier = sum(
            (base_rate - t) ** 2 for t in truths
        ) / len(truths)
        reliability: list[dict[str, Any]] = []
        for index in range(bins):
            low, high = index / bins, (index + 1) / bins
            members = [
                (p, t)
                for p, t in zip(predictions, truths, strict=True)
                if (low <= p < high) or (index == bins - 1 and p == 1.0)
            ]
            if members:
                reliability.append(
                    {
                        "bin": [round(low, 2), round(high, 2)],
                        "n": len(members),
                        "mean_prediction": round(
                            sum(p for p, _ in members) / len(members), 4
                        ),
                        "observed_rate": round(
                            sum(t for _, t in members) / len(members), 4
                        ),
                    }
                )
        trustworthy = (
            self._fitted
            and brier <= MAX_TRUSTWORTHY_BRIER
            and brier < baseline_brier
        )
        return {
            "brier": round(brier, 6),
            "baseline_brier": round(baseline_brier, 6),
            "beats_constant_predictor": bool(brier < baseline_brier),
            "trustworthy": bool(trustworthy),
            "reliability": reliability,
        }

    def drift(
        self, recent: Sequence[CriticObservation], *, tolerance: float = 0.05
    ) -> dict[str, Any]:
        """Has calibration decayed since fitting? (the TEMPO failure mode)

        Test-time adaptation plateaus when a self-generated signal drifts
        and nothing recalibrates it. Reporting drift is what makes periodic
        recalibration possible instead of silent decay.
        """
        report = self.calibration(recent)
        drifted = (not report["trustworthy"]) or report["brier"] > (
            MAX_TRUSTWORTHY_BRIER + tolerance
        )
        return {
            "schema": PROCESS_CRITIC_SCHEMA,
            "drifted": bool(drifted),
            "recalibration_required": bool(drifted),
            "recent_brier": report["brier"],
            "recent_observations": len(recent),
        }

    def step_credit(
        self, trajectory_features: Sequence[Sequence[float]]
    ) -> list[float]:
        """Per-step credit: how much each step raised P(success).

        This is the signal outcome-only loss cannot provide. A step that
        moves the state toward a verified answer earns positive credit; a
        step that merely changes the state earns none, which is precisely
        the distinction between thinking and spinning.
        """
        if len(trajectory_features) < 2:
            return []
        values = [self.predict(features) for features in trajectory_features]
        # Pairwise: values[1:] is intentionally one shorter, so this zip
        # must NOT be strict.
        return [
            round(after - before, 6)
            for before, after in zip(values, values[1:])
        ]


__all__ = [
    "MAX_TRUSTWORTHY_BRIER",
    "MIN_OBSERVATIONS",
    "PROCESS_CRITIC_SCHEMA",
    "CriticObservation",
    "ProcessCritic",
    "state_features",
]
