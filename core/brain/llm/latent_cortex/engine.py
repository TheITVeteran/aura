"""The integrated machine: encode → workspace → branches → schedule →
recurrence → optimize → fast-weights → halt → persist → decode.

One call (`LatentCortexEngine.reason`) executes the complete Anima Rationis
pipeline on a frozen mlx_lm model, under a compute budget, inside checkpoint
invariants, with a fail-honest fallback: if any latent phase trips a guard,
the engine ships the vanilla path WITH A RECEIPT SAYING SO — never a silent
downgrade, never a corrupted cache.

The engine is deliberately synchronous and model-local: it runs inside the
MLX worker process (action "latent_reason") or in-process for tests and the
experiments harness. Async orchestration, budgets-from-the-Will, and health
reporting live in core/brain/latent_cortex_service.py.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable

from core.brain.llm.latent_cortex.branches import BranchEnsemble, BranchState
from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights
from core.brain.llm.latent_cortex.governance import CheckpointInvariant
from core.brain.llm.latent_cortex.latent_opt import LatentOptimizer, build_proxy_loss
from core.brain.llm.latent_cortex.recurrence import WindowRunner
from core.brain.llm.latent_cortex.schedules import LayerSchedule, ScheduleLibrary
from core.brain.llm.latent_cortex.types import (
    ComputeBudget,
    CortexConfig,
    EpisodeReceipt,
    LatentReasoningResult,
)
from core.brain.llm.latent_cortex.workspace import per_position_rms
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LatentCortex.Engine")

# Guard classes the engine treats as "latent phase failed, fall back honest".
_LATENT_PHASE_ERRORS = (RuntimeError, ValueError, TypeError, AttributeError, KeyError)


def _logits_digest(logits) -> str:
    """Stable digest of a logits vector — the causal audit fingerprint."""
    import hashlib

    import mlx.core as mx

    arr = logits.astype(mx.float32)
    mx.eval(arr)
    return hashlib.sha256(memoryview(arr)).hexdigest()[:16]


class LatentCortexEngine:
    """Runs complete latent-reasoning episodes on one frozen model."""

    def __init__(
        self,
        model,
        tokenizer=None,
        config: CortexConfig | None = None,
        *,
        model_path: str | None = None,
        schedule_library: ScheduleLibrary | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CortexConfig()
        problems = self.config.validate()
        if problems:
            raise ValueError(f"invalid CortexConfig: {problems}")
        self.library = schedule_library
        self.invariant = CheckpointInvariant(model, model_path)
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None)
        if not layers:
            raise ValueError("model has no .model.layers — not an mlx_lm decoder")
        self.n_layers = len(layers)
        self.prelude_end = max(1, int(self.n_layers * self.config.prelude_frac))
        self.coda_start = min(
            self.n_layers - 1,
            self.n_layers - max(1, int(self.n_layers * self.config.coda_frac)),
        )
        if self.coda_start - self.prelude_end < 1:
            raise ValueError(
                f"recurrent region empty: prelude_end={self.prelude_end} "
                f"coda_start={self.coda_start} for {self.n_layers} layers"
            )

    # ── Tokenization ────────────────────────────────────────────────────
    def _encode(self, prompt: str | None, messages: list | None, token_ids: list[int] | None):
        if token_ids is not None:
            return list(token_ids)
        if self.tokenizer is None:
            raise ValueError("no tokenizer: pass token_ids for substrate-level use")
        if messages:
            return list(
                self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                )
            )
        return list(self.tokenizer.encode(prompt or ""))

    def _eos_ids(self) -> set[int]:
        if self.tokenizer is None:
            return set()
        ids = getattr(self.tokenizer, "eos_token_ids", None)
        if ids:
            return set(int(i) for i in ids)
        eid = getattr(self.tokenizer, "eos_token_id", None)
        return {int(eid)} if eid is not None else set()

    # ── Model plumbing ──────────────────────────────────────────────────
    def _fresh_cache(self):
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.n_layers)]

    def _logits(self, h):
        inner = self.model.model
        h = inner.norm(h)
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head(h)
        return inner.embed_tokens.as_linear(h)

    def _prefill(self, tokens: list[int], cache, budget: ComputeBudget):
        """Standard full-stack prefill. Returns (embeddings, last-position logits)."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        inner = self.model.model
        arr = mx.array([tokens])
        h = inner.embed_tokens(arr)
        embeddings = h
        mask = create_attention_mask(h, cache)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[i])
        logits = self._logits(h[:, -1:, :])[0, -1]
        mx.eval(logits)
        budget.charge(tokens=len(tokens), layers=self.n_layers)
        return embeddings, logits

    def _sample(self, logits, temperature: float) -> int:
        import mlx.core as mx

        if temperature and temperature > 0:
            return int(mx.random.categorical(logits / temperature))
        return int(mx.argmax(logits))

    def _decode(
        self,
        cache,
        budget: ComputeBudget,
        initial_logits,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[int]:
        """Minimal sampler: first token from ``initial_logits`` (the logits of
        the last persisted position — prompt tail or final thought slot), then
        autoregressive continuation over the populated cache."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        inner = self.model.model
        eos = self._eos_ids()
        limit = max_tokens if max_tokens is not None else self.config.decode_max_tokens
        temp = temperature if temperature is not None else self.config.decode_temperature

        out: list[int] = []
        token = self._sample(initial_logits, temp)
        for _ in range(max(1, int(limit))):
            if token in eos:
                break
            out.append(token)
            if budget.exhausted:
                break
            h = inner.embed_tokens(mx.array([[token]]))
            mask = create_attention_mask(h, cache)
            for i, layer in enumerate(inner.layers):
                h = layer(h, mask, cache[i])
            logits = self._logits(h)[0, -1]
            budget.charge(tokens=1, layers=self.n_layers)
            token = self._sample(logits, temp)
        return out

    # ── Probe decoding for branch selection / verifier loops ────────────
    def _decode_probe(
        self,
        branch: BranchState,
        cache,
        runner: WindowRunner,
        budget: ComputeBudget,
        *,
        max_tokens: int = 48,
    ) -> list[int]:
        """Decode a short probe from a branch WITHOUT disturbing the caches.

        Full-cache snapshot → persist this branch's slots → decode → restore.
        This is what lets a verifier score every branch before exactly one
        winner's state is committed.
        """
        from core.brain.llm.recurrent_depth import (
            _restore_recurrent_caches,
            _snapshot_recurrent_caches,
        )

        snaps = _snapshot_recurrent_caches(cache, 0, self.n_layers)
        try:
            slot_logits = self._persist_branch(branch, cache, runner)
            return self._decode(
                cache, budget, slot_logits, max_tokens=max_tokens, temperature=0.0
            )
        finally:
            _restore_recurrent_caches(cache, 0, self.n_layers, snaps)

    def _persist_branch(self, branch: BranchState, cache, runner: WindowRunner):
        """Commit one branch's slots into every layer's KV (the causal step).

        Returns the last slot position's logits — the next-token distribution
        conditioned on [prompt; refined thoughts], which seeds decoding.
        """
        import mlx.core as mx

        runner.run(branch.workspace.seed_z, cache, 0, self.prelude_end, persist=True)
        z_fin = runner.run(branch.z, cache, self.prelude_end, self.coda_start, persist=True)
        z_out = runner.run(z_fin, cache, self.coda_start, self.n_layers, persist=True)
        logits = self._logits(z_out[:, -1:, :])[0, -1]
        mx.eval(logits)
        return logits

    # ── Schedule resolution ─────────────────────────────────────────────
    def _resolve_schedule(self, domain: str) -> LayerSchedule:
        if self.config.schedule is not None:
            schedule = LayerSchedule.from_dict(self.config.schedule)
            violations = schedule.validate(
                prelude_end=self.prelude_end, coda_start=self.coda_start
            )
            if violations:
                raise ValueError(f"configured schedule invalid: {violations}")
            return schedule
        if self.library is not None:
            return self.library.best_for_domain(
                domain,
                prelude_end=self.prelude_end,
                coda_start=self.coda_start,
                default_repeats=self.config.recurrence.max_steps,
            )
        return LayerSchedule.single_window(
            self.prelude_end, self.coda_start, self.config.recurrence.max_steps
        )

    # ── The integrated episode ──────────────────────────────────────────
    def reason(
        self,
        prompt: str | None = None,
        *,
        messages: list | None = None,
        token_ids: list[int] | None = None,
        budget: ComputeBudget | None = None,
        verifier: Callable[[str], float] | None = None,
        domain: str = "general",
        decode_max_tokens: int | None = None,
    ) -> LatentReasoningResult:
        receipt = EpisodeReceipt(episode_id=uuid.uuid4().hex[:12])
        receipt.n_layers = self.n_layers
        receipt.prelude_end = self.prelude_end
        receipt.coda_start = self.coda_start
        budget = budget or ComputeBudget()
        tokens = self._encode(prompt, messages, token_ids)

        self.invariant.pre_episode()
        receipt.checkpoint_fingerprint = self.invariant.file_receipt.get("fingerprint", "")

        try:
            out_tokens, receipt = self._latent_episode(
                tokens, budget, verifier, domain, receipt, decode_max_tokens
            )
        except _LATENT_PHASE_ERRORS as exc:
            record_degradation(
                "latent_cortex",
                exc,
                action="served vanilla decode with honest fallback receipt",
            )
            receipt.flag(f"fallback_vanilla:{type(exc).__name__}")
            receipt.halting_reason = receipt.halting_reason or "latent_phase_error"
            try:
                cache = self._fresh_cache()
                _, tail_logits = self._prefill(tokens, cache, budget)
                out_tokens = self._decode(
                    cache, budget, tail_logits, max_tokens=decode_max_tokens
                )
            except _LATENT_PHASE_ERRORS as inner_exc:
                record_degradation(
                    "latent_cortex",
                    inner_exc,
                    action="reported failed episode after vanilla fallback also failed",
                    severity="degraded",
                )
                receipt.budget = budget.to_receipt()
                return LatentReasoningResult(
                    ok=False,
                    text="",
                    receipt=receipt,
                    reason=f"latent_and_fallback_failed:{inner_exc}",
                )

        receipt.params_unchanged = self.invariant.post_episode()
        receipt.budget = budget.to_receipt()
        if receipt.params_unchanged is False:
            receipt.flag("checkpoint_invariant_violated")
            return LatentReasoningResult(
                ok=False,
                text="",
                receipt=receipt,
                reason="checkpoint_invariant_violated",
            )

        text = (
            self.tokenizer.decode(out_tokens)
            if self.tokenizer is not None and out_tokens
            else ""
        )
        return LatentReasoningResult(
            ok=True, text=text, receipt=receipt, tokens=out_tokens
        )

    # The latent phases, separated so the fallback wrapper stays readable.
    def _latent_episode(
        self,
        tokens: list[int],
        budget: ComputeBudget,
        verifier,
        domain: str,
        receipt: EpisodeReceipt,
        decode_max_tokens: int | None,
    ) -> tuple[list[int], EpisodeReceipt]:
        import mlx.core as mx

        cache = self._fresh_cache()
        runner = WindowRunner(self.model.model, budget)
        embeddings, _tail_logits = self._prefill(tokens, cache, budget)

        schedule = self._resolve_schedule(domain)
        receipt.schedule_hash = schedule.schedule_hash
        receipt.n_slots = self.config.workspace.n_slots
        receipt.n_branches = self.config.branches.n_branches

        ensemble = BranchEnsemble.seed(
            embeddings,
            self.config.workspace,
            self.config.branches,
            self.config.recurrence,
            runner,
            cache,
            self.prelude_end,
        )

        # ── Recurrent computation under the schedule program ────────────
        for op in schedule.ops:
            for _ in range(op.repeats):
                if ensemble.all_halted() or budget.exhausted:
                    break
                ensemble.step_all(
                    runner,
                    cache,
                    op.start,
                    op.end,
                    budget=budget,
                    alpha_override=op.alpha,
                )
        for branch in ensemble.branches:
            if not branch.halted:
                final, reverted = branch.halting.final_state(branch.z)
                branch.z = final
                branch.workspace.update(final)
                branch.halted = True
                branch.halt_reason = (
                    "budget_exhausted" if budget.exhausted else "schedule_complete"
                ) + ("_reverted" if reverted else "")

        receipt.exchanges = ensemble.exchanges

        # ── Branch selection ─────────────────────────────────────────────
        if verifier is not None and self.tokenizer is not None:
            def branch_score(branch: BranchState) -> float:
                probe = self._decode_probe(branch, cache, runner, budget)
                return float(verifier(self.tokenizer.decode(probe)))

            winner = ensemble.select(score_fn=branch_score)
        else:
            winner = ensemble.select()
        receipt.branch_scores = [float(b.score) for b in ensemble.branches]
        receipt.selected_branch = winner.index
        receipt.steps_taken = winner.steps
        receipt.residual_trail = list(winner.halting.residual_trail)
        receipt.best_step = winner.halting.best_step
        receipt.halting_reason = winner.halt_reason
        receipt.reverted_to_best = winner.halt_reason.endswith("_reverted")

        # ── Latent optimization on the winner ────────────────────────────
        if self.config.latent_opt.enabled:
            loss_fn = build_proxy_loss(self.model, winner.anchor, tokens, self.config.latent_opt)
            optimizer = LatentOptimizer(
                loss_fn, self.config.latent_opt, seed=self.config.workspace.seed
            )
            if verifier is not None and self.tokenizer is not None:
                def z_score(z) -> float:
                    saved = winner.z
                    winner.z = z
                    winner.workspace.update(z)
                    try:
                        probe = self._decode_probe(winner, cache, runner, budget)
                        return float(verifier(self.tokenizer.decode(probe)))
                    finally:
                        winner.z = saved
                        winner.workspace.update(saved)

                z_opt, _ = optimizer.run_with_verifier(winner.z, z_score)
            else:
                z_opt = optimizer.run(winner.z)
            winner.z = z_opt
            winner.workspace.update(z_opt)
            receipt.latent_opt_applied = True
            receipt.latent_opt_mode = optimizer.trace.mode
            receipt.latent_opt_loss_trail = list(optimizer.trace.loss_trail)

        # ── Episode fast weights (attach → optimize → decode under ΔW) ──
        fast_weights: EpisodicFastWeights | None = None
        fw_baseline = None
        if self.config.fast_weights.enabled:
            fast_weights = EpisodicFastWeights(self.config.fast_weights)
            fw_baseline = self._fw_probe()
            seed_stat = float(mx.mean(per_position_rms(winner.z)))
            wrapped = fast_weights.attach(
                self.model.model,
                (self.prelude_end, self.coda_start),
                seed_stat=seed_stat,
                episode_id=receipt.episode_id,
            )
            receipt.fast_weights_applied = True
            receipt.fast_weights_layers = wrapped
            loss_fn = build_proxy_loss(
                self.model, winner.anchor, tokens, self.config.latent_opt
            )

            def fw_loss():
                z_pass = self._nocache_window_pass(winner.z)
                return loss_fn(z_pass)

            fast_weights.optimize(fw_loss)

        try:
            # ── Commit the winner + decode the answer ────────────────────
            slot_logits = self._persist_branch(winner, cache, runner)
            receipt.first_logits_digest = _logits_digest(slot_logits)
            out_tokens = self._decode(
                cache, budget, slot_logits, max_tokens=decode_max_tokens
            )
        finally:
            if fast_weights is not None:
                fast_weights.snapshot_for_export()
                fast_weights.detach()
                proven = fast_weights.prove_erase(self._fw_probe, fw_baseline)
                receipt.fast_weights_erased = proven
                if not proven:
                    receipt.flag("fast_weight_erase_unproven")

        return out_tokens, receipt

    # ── Fast-weight helpers ─────────────────────────────────────────────
    def _nocache_window_pass(self, z):
        """Window pass with no cache (grad-safe: zero side effects)."""
        inner = self.model.model
        h = z
        for i in range(self.prelude_end, self.coda_start):
            h = inner.layers[i](h, None, None)
        return h

    def _fw_probe(self):
        """Deterministic full-stack probe for erase proofs (cache-free)."""
        import mlx.core as mx

        inner = self.model.model
        vocab = inner.embed_tokens.weight.shape[0]
        probe_tokens = mx.array([[i % int(vocab) for i in range(1, 9)]])
        h = inner.embed_tokens(probe_tokens)
        for layer in inner.layers:
            h = layer(h, None, None)
        out = self._logits(h)
        mx.eval(out)
        return out


__all__ = ["LatentCortexEngine"]
