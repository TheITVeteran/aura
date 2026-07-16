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
from dataclasses import dataclass, field
from typing import Any, Callable

from core.brain.llm.latent_cortex.types import LatentOptConfig
from core.brain.llm.latent_cortex.workspace import per_position_rms

logger = logging.getLogger("Aura.LatentCortex.LatentOpt")


def prompt_token_distribution(prompt_tokens, vocab_size: int):
    """Empirical unigram distribution of the prompt (1, V) — the R target."""
    import mlx.core as mx

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
        rms_drift = mx.mean(mx.square(per_position_rms(z) - z0_rms) / mx.square(z0_rms))
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
    steps_taken: int = 0
    accepted: int = 0
    rejected: int = 0

    def to_receipt(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "loss_trail": [round(v, 6) for v in self.loss_trail],
            "steps_taken": self.steps_taken,
            "accepted": self.accepted,
            "rejected": self.rejected,
        }


class LatentOptimizer:
    """Budgeted optimizer over a workspace state, gradient or control arm."""

    def __init__(self, loss_fn: Callable, config: LatentOptConfig, *, seed: int = 0) -> None:
        self._loss_fn = loss_fn
        self.config = config
        self._seed = seed
        self.trace = OptTrace(mode="control" if config.control_mode else "gradient")

    def _clipped_step(self, grad):
        """Gradient step with global-norm clipping; returns (step, magnitude)."""
        import mlx.core as mx

        gnorm = mx.maximum(mx.linalg.norm(mx.reshape(grad, (-1,))), 1e-12)
        scale = mx.minimum(1.0, self.config.max_grad_norm / gnorm)
        step = -self.config.lr * grad * scale
        return step, float(mx.linalg.norm(mx.reshape(step, (-1,))))

    def step(self, z, step_index: int):
        """One optimization move. Control mode consumes the SAME gradient
        computation to derive its magnitude, then discards the direction —
        that is what makes it a matched control rather than a strawman."""
        import mlx.core as mx

        value, grad = mx.value_and_grad(self._loss_fn)(z)
        self.trace.loss_trail.append(float(value))
        step, magnitude = self._clipped_step(grad)
        if self.config.control_mode:
            key = mx.random.key(90210 + 7 * self._seed + step_index)
            rand = mx.random.normal(z.shape, key=key)
            rand_norm = mx.maximum(mx.linalg.norm(mx.reshape(rand, (-1,))), 1e-12)
            step = rand * (magnitude / rand_norm)
        z_next = z + step
        mx.eval(z_next)
        self.trace.steps_taken += 1
        return z_next

    def run(self, z):
        """Pure proxy descent for ``config.steps`` moves (no verifier)."""
        for i in range(max(0, int(self.config.steps))):
            z = self.step(z, i)
        # Record the final loss for the receipt's trend line.
        import mlx.core as mx

        self.trace.loss_trail.append(float(self._loss_fn(z)))
        mx.eval(z)
        return z

    def run_with_verifier(
        self,
        z,
        score_fn: Callable[[Any], float],
        *,
        max_proposals: int | None = None,
    ):
        """Greedy hill-climb: proxy-guided proposals, verifier-gated accepts.

        ``score_fn`` is the honesty boundary — it must decode a probe from
        the CANDIDATE state and return a verified score. Rejected proposals
        are fully reverted; the verifier, not the proxy, has the last word.
        """
        proposals = max_proposals if max_proposals is not None else self.config.steps
        best_score = float(score_fn(z))
        for i in range(max(0, int(proposals))):
            candidate = self.step(z, i)
            candidate_score = float(score_fn(candidate))
            if candidate_score > best_score:
                z, best_score = candidate, candidate_score
                self.trace.accepted += 1
            else:
                self.trace.rejected += 1
        return z, best_score


__all__ = [
    "LatentOptimizer",
    "OptTrace",
    "build_proxy_loss",
    "prompt_token_distribution",
]
