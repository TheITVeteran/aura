"""Gradient descent over thoughts: the hidden state as an optimizable object.

All weights stay frozen; Z is the variable. The differentiable proxy is
deliberately chosen so it CANNOT leak answers or simply sharpen confidence
(the failure mode the spec warns about — "merely pushing the model toward
high confidence would often strengthen confident mistakes"):

    S(Z) = λ_r·R(Z) − λ_d·D(Z, Z₀)

R — problem reconstruction: log-probability mass the workspace readout
assigns to the prompt's own token distribution. A state that can no longer
reconstruct what problem it is solving has lost the thread; pushing R up
keeps the latent computation grounded in the actual question. R contains no
information about the ANSWER.

D — manifold distance: RMS drift + cosine drift from the post-prelude seed
Z₀. Penalizing D keeps optimized states inside the activation distribution
the frozen layers were trained on.

Verifier signal (non-differentiable) enters through greedy hill-climbing:
propose → decode probe → verify → accept/reject. And the Experiment-5
control arm is built in: ``control_mode`` applies matched-magnitude RANDOM
perturbations — computed from the true gradient's step size so magnitudes
match exactly — letting the harness measure whether gradient DIRECTION
(not mere perturbation) is what helps.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm.latent_cortex.types import ComputeBudget, LatentOptConfig
from core.brain.llm.latent_cortex.workspace import per_position_rms

logger = logging.getLogger("Aura.LatentCortex.LatentOpt")
_LINE_SEARCH_EVALS = 12


def prompt_token_distribution(prompt_tokens, vocab_size: int):
    """Empirical unigram distribution of the prompt (1, V) — the R target."""
    import mlx.core as mx

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not prompt_tokens:
        raise ValueError("prompt token distribution requires at least one token")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in prompt_tokens):
        raise ValueError("prompt token outside model vocabulary")
    counts = mx.zeros((vocab_size,))
    ones = mx.ones((len(prompt_tokens),))
    counts = counts.at[mx.array(prompt_tokens)].add(ones)
    return counts / mx.maximum(mx.sum(counts), 1.0)


def build_proxy_loss(
    model,
    z0,
    prompt_tokens: list[int],
    config: LatentOptConfig,
) -> Callable:
    """Loss(z) = −S(z), differentiable w.r.t. z with frozen weights.

    The readout path is norm → lm_head on the mean slot state: cheap, fully
    differentiable, and independent of the KV cache (no cache mutation inside
    the gradient graph).
    """
    import mlx.core as mx

    inner = model.model
    vocab = (
        model.lm_head.weight.shape[0]
        if hasattr(model, "lm_head")
        else inner.embed_tokens.weight.shape[0]
    )
    target = prompt_token_distribution(prompt_tokens, int(vocab))
    z0_rms = per_position_rms(z0)
    z0_flat = mx.reshape(z0, (-1,))
    z0_norm = mx.maximum(mx.linalg.norm(z0_flat), 1e-6)

    def readout_logits(z):
        pooled = mx.mean(z, axis=1, keepdims=True)  # (1,1,D)
        h = inner.norm(pooled)
        if hasattr(model, "lm_head"):
            return model.lm_head(h)
        return inner.embed_tokens.as_linear(h)

    def loss(z):
        # R: cross-entropy of readout against the prompt unigram target.
        logits = readout_logits(z)[0, 0]
        logp = logits - mx.logsumexp(logits)
        reconstruction = mx.sum(target * logp)  # ≤ 0, higher is better
        # D: manifold drift (norm band + direction).
        rms_drift = mx.mean(
            mx.square(per_position_rms(z) - z0_rms)
            / mx.square(mx.maximum(z0_rms, 1e-6))
        )
        z_flat = mx.reshape(z, (-1,))
        cos = mx.sum(z_flat * z0_flat) / (
            mx.maximum(mx.linalg.norm(z_flat), 1e-6) * z0_norm
        )
        manifold = rms_drift + (1.0 - cos)
        return -(config.lambda_reconstruct * reconstruction) + config.lambda_manifold * manifold

    return loss


@dataclass
class OptTrace:
    mode: str = "off"
    loss_trail: list[float] = field(default_factory=list)
    attempts: int = 0
    steps_taken: int = 0
    accepted: int = 0
    rejected: int = 0
    line_search_backtracks: int = 0
    budget_exhausted: bool = False

    def to_receipt(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "loss_trail": [round(v, 6) for v in self.loss_trail],
            "attempts": self.attempts,
            "steps_taken": self.steps_taken,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "line_search_backtracks": self.line_search_backtracks,
            "budget_exhausted": self.budget_exhausted,
        }


class LatentOptimizer:
    """Budgeted optimizer over a workspace state, gradient or control arm."""

    def __init__(
        self,
        loss_fn: Callable,
        config: LatentOptConfig,
        *,
        seed: int = 0,
        budget: ComputeBudget | None = None,
        layer_apps_per_loss: int = 0,
        reserve_layer_apps: int = 0,
    ) -> None:
        if isinstance(layer_apps_per_loss, bool) or not isinstance(
            layer_apps_per_loss, int
        ):
            raise TypeError("layer_apps_per_loss must be an integer")
        if isinstance(reserve_layer_apps, bool) or not isinstance(
            reserve_layer_apps, int
        ):
            raise TypeError("reserve_layer_apps must be an integer")
        if layer_apps_per_loss < 0 or reserve_layer_apps < 0:
            raise ValueError("optimizer compute costs cannot be negative")
        if budget is not None and layer_apps_per_loss <= 0:
            raise ValueError(
                "budgeted latent optimization requires a positive loss-evaluation cost"
            )
        self._loss_fn = loss_fn
        self.config = config
        self._seed = seed
        self._budget = budget
        self._layer_apps_per_loss = layer_apps_per_loss
        self._reserve_layer_apps = reserve_layer_apps
        self.trace = OptTrace(mode="control" if config.control_mode else "gradient")

    def _can_reserve(self, additional_layer_apps: int = 0) -> bool:
        if isinstance(additional_layer_apps, bool) or not isinstance(
            additional_layer_apps, int
        ):
            raise TypeError("additional_layer_apps must be an integer")
        if additional_layer_apps < 0:
            raise ValueError("additional_layer_apps cannot be negative")
        if self._budget is None:
            return True
        admitted = (
            not self._budget.exhausted
            and self._reserve_layer_apps + additional_layer_apps
            <= self._budget.remaining_layer_apps
        )
        if not admitted:
            self.trace.budget_exhausted = True
        return admitted

    def _charge_loss_evals(
        self, count: int, *, additional_reserve_layer_apps: int = 0
    ) -> bool:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("loss-evaluation count must be a non-negative integer")
        if self._budget is None:
            return True
        charge = count * self._layer_apps_per_loss
        if not self._can_reserve(charge + additional_reserve_layer_apps):
            return False
        self._budget.charge_layer_apps(charge)
        return True

    def _clipped_step(self, grad):
        """Gradient step with global-norm clipping; returns (step, magnitude)."""
        import mlx.core as mx

        gnorm = mx.maximum(mx.linalg.norm(mx.reshape(grad, (-1,))), 1e-12)
        scale = mx.minimum(1.0, self.config.max_grad_norm / gnorm)
        step = -self.config.lr * grad * scale
        return step, float(mx.linalg.norm(mx.reshape(step, (-1,))))

    def _propose(
        self,
        z,
        step_index: int,
        *,
        additional_reserve_layer_apps: int = 0,
    ):
        """One optimization move. Control mode consumes the SAME gradient
        computation to derive its magnitude, then discards the direction —
        that is what makes it a matched control rather than a strawman."""
        import mlx.core as mx

        # Forward + backward is conservatively charged as three proxy-loss
        # evaluations. The charge happens before execution and protects the
        # answer-decode reserve, so optimization can never strand completion.
        if not self._charge_loss_evals(
            3, additional_reserve_layer_apps=additional_reserve_layer_apps
        ):
            return z, False, None
        self.trace.attempts += 1
        value, grad = mx.value_and_grad(self._loss_fn)(z)
        value_float = float(value)
        if not math.isfinite(value_float) or not bool(mx.all(mx.isfinite(grad))):
            raise RuntimeError("latent optimizer produced a non-finite value or gradient")
        if not self.trace.loss_trail:
            self.trace.loss_trail.append(value_float)
        step, magnitude = self._clipped_step(grad)
        if not math.isfinite(magnitude):
            raise RuntimeError("latent optimizer produced a non-finite step magnitude")
        if self.config.control_mode:
            key = mx.random.key(90210 + 7 * self._seed + step_index)
            rand = mx.random.normal(z.shape, key=key)
            rand_norm = mx.maximum(mx.linalg.norm(mx.reshape(rand, (-1,))), 1e-12)
            step = rand * (magnitude / rand_norm)
        z_next = z + step
        mx.eval(z_next)
        return z_next, True, value_float

    def step(self, z, step_index: int):
        """Return one proposal without claiming it was accepted.

        ``run`` and ``run_with_verifier`` own acceptance bookkeeping. Keeping
        proposal generation separate prevents verifier-rejected states from
        inflating the accepted-step count in evidence receipts.
        """
        candidate, _, _ = self._propose(z, step_index)
        return candidate

    def run(self, z):
        """Bounded proxy descent under one acceptance policy for both arms."""
        import mlx.core as mx

        for step_index in range(max(0, int(self.config.steps))):
            line_search_cost = _LINE_SEARCH_EVALS * self._layer_apps_per_loss
            candidate, admitted, current_loss = self._propose(
                z,
                step_index,
                additional_reserve_layer_apps=line_search_cost,
            )
            if not admitted:
                break
            if current_loss is None:
                raise RuntimeError("latent optimizer admitted a proposal without a loss")
            if not self._charge_loss_evals(_LINE_SEARCH_EVALS):
                raise RuntimeError("latent optimizer lost an admitted line-search reservation")
            raw_step = candidate - z
            candidates: list[tuple[int, Any, float]] = []
            for backtrack in range(_LINE_SEARCH_EVALS):
                backtracked = z + raw_step * (0.5**backtrack)
                candidate_loss = float(self._loss_fn(backtracked))
                if math.isfinite(candidate_loss) and candidate_loss < current_loss:
                    candidates.append((backtrack, backtracked, candidate_loss))
            if not candidates:
                self.trace.rejected += 1
                break
            backtrack, accepted_state, accepted_loss = candidates[0]
            mx.eval(accepted_state)
            z = accepted_state
            self.trace.steps_taken += 1
            self.trace.accepted += 1
            self.trace.line_search_backtracks += backtrack
            self.trace.loss_trail.append(accepted_loss)
        mx.eval(z)
        return z

    def run_with_verifier(
        self,
        z,
        score_fn: Callable[[Any], float],
        *,
        max_proposals: int | None = None,
        verifier_layer_apps: int = 0,
    ):
        """Greedy hill-climb: proxy-guided proposals, verifier-gated accepts.

        ``score_fn`` is the honesty boundary — it must decode a probe from
        the CANDIDATE state and return a verified score. Rejected proposals
        are fully reverted; the verifier, not the proxy, has the last word.
        """
        if isinstance(verifier_layer_apps, bool) or not isinstance(
            verifier_layer_apps, int
        ):
            raise TypeError("verifier_layer_apps must be an integer")
        if verifier_layer_apps < 0:
            raise ValueError("verifier_layer_apps cannot be negative")
        proposals = max_proposals if max_proposals is not None else self.config.steps
        if not self._can_reserve(verifier_layer_apps):
            return z, float("-inf")
        best_score = float(score_fn(z))
        if not math.isfinite(best_score):
            raise RuntimeError("latent verifier returned a non-finite baseline score")
        for i in range(max(0, int(proposals))):
            candidate, admitted, _ = self._propose(
                z, i, additional_reserve_layer_apps=verifier_layer_apps
            )
            if not admitted:
                break
            candidate_score = float(score_fn(candidate))
            if not math.isfinite(candidate_score):
                self.trace.rejected += 1
                continue
            if candidate_score > best_score:
                z, best_score = candidate, candidate_score
                self.trace.accepted += 1
                self.trace.steps_taken += 1
            else:
                self.trace.rejected += 1
        return z, best_score


__all__ = [
    "LatentOptimizer",
    "OptTrace",
    "build_proxy_loss",
    "prompt_token_distribution",
]
