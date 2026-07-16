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
import uuid
from collections.abc import Callable

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
_LATENT_PHASE_ERRORS = (
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    MemoryError,
    OSError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


class _FastWeightCleanupError(RuntimeError):
    """The resident model could not be proven clean after an episode."""


def _logits_digest(logits) -> str:
    """Stable digest of a logits vector — the causal audit fingerprint."""
    import hashlib

    import mlx.core as mx

    arr = logits.astype(mx.float32)
    mx.eval(arr)
    return hashlib.sha256(memoryview(arr)).hexdigest()


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
        if not tokens:
            raise ValueError("latent episode requires at least one input token")
        if not budget.can_afford(len(tokens), self.n_layers):
            raise RuntimeError("compute budget cannot afford prompt prefill")
        budget.charge(tokens=len(tokens), layers=self.n_layers)
        arr = mx.array([tokens])
        h = inner.embed_tokens(arr)
        embeddings = h
        mask = create_attention_mask(h, cache)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[i])
        logits = self._logits(h[:, -1:, :])[0, -1]
        mx.eval(logits)
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
    ) -> tuple[list[int], str]:
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
        if budget.exhausted:
            return out, "budget_exhausted"
        token = self._sample(initial_logits, temp)
        termination = "token_limit"
        for index in range(max(1, int(limit))):
            if token in eos:
                termination = "eos"
                break
            out.append(token)
            if index + 1 >= limit:
                termination = "token_limit"
                break
            if budget.exhausted:
                termination = "budget_exhausted"
                break
            if not budget.can_afford(1, self.n_layers):
                termination = "budget_unaffordable"
                break
            budget.charge(tokens=1, layers=self.n_layers)
            h = inner.embed_tokens(mx.array([[token]]))
            mask = create_attention_mask(h, cache)
            for i, layer in enumerate(inner.layers):
                h = layer(h, mask, cache[i])
            logits = self._logits(h)[0, -1]
            token = self._sample(logits, temp)
        return out, termination

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
            )[0]
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
        ablate_slot: int | None = None,
        ablate_mode: str = "zero",
    ) -> LatentReasoningResult:
        receipt = EpisodeReceipt(episode_id=uuid.uuid4().hex[:12])
        receipt.n_layers = self.n_layers
        receipt.prelude_end = self.prelude_end
        receipt.coda_start = self.coda_start
        budget = budget or ComputeBudget()
        if decode_max_tokens is not None:
            if type(decode_max_tokens) is not int:
                raise TypeError("decode_max_tokens override must be an integer")
            if not 1 <= decode_max_tokens <= 8192:
                raise ValueError("decode_max_tokens override outside [1, 8192]")
        tokens = self._encode(prompt, messages, token_ids)

        self.invariant.pre_episode()
        receipt.checkpoint_fingerprint = self.invariant.file_receipt.get("fingerprint", "")
        receipt.checkpoint_fingerprint_method = self.invariant.file_receipt.get(
            "method", ""
        )
        receipt.checkpoint_file_count = int(
            self.invariant.file_receipt.get("files", 0) or 0
        )

        failure_reason = ""
        out_tokens: list[int] = []
        try:
            try:
                out_tokens, receipt = self._latent_episode(
                    tokens,
                    budget,
                    verifier,
                    domain,
                    receipt,
                    decode_max_tokens,
                    ablate_slot=ablate_slot,
                    ablate_mode=ablate_mode,
                )
            except _FastWeightCleanupError as exc:
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="refused fallback decode and requested resident-worker recycle",
                    severity="critical",
                )
                failure_reason = "fast_weight_cleanup_unproven"
            except _LATENT_PHASE_ERRORS as exc:
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="served vanilla decode with honest fallback receipt",
                )
                receipt.flag(f"fallback_vanilla:{type(exc).__name__}")
                receipt.halting_reason = receipt.halting_reason or "latent_phase_error"
                if receipt.fast_weights_applied and receipt.fast_weights_erased is not True:
                    receipt.flag("fallback_refused_unproven_model_state")
                    failure_reason = "fast_weight_cleanup_unproven"
                else:
                    try:
                        cache = self._fresh_cache()
                        _, tail_logits = self._prefill(tokens, cache, budget)
                        out_tokens, decode_termination = self._decode(
                            cache, budget, tail_logits, max_tokens=decode_max_tokens
                        )
                        receipt.decode_requested_tokens = (
                            decode_max_tokens
                            if decode_max_tokens is not None
                            else self.config.decode_max_tokens
                        )
                        receipt.decode_generated_tokens = len(out_tokens)
                        receipt.decode_termination = decode_termination
                        if decode_termination.startswith("budget_"):
                            receipt.flag(f"decode_{decode_termination}")
                    except _LATENT_PHASE_ERRORS as inner_exc:
                        record_degradation(
                            "latent_cortex",
                            inner_exc,
                            action=(
                                "reported failed episode after vanilla fallback also failed"
                            ),
                            severity="degraded",
                        )
                        failure_reason = f"latent_and_fallback_failed:{inner_exc}"
        finally:
            try:
                receipt.params_unchanged = self.invariant.post_episode()
            except _LATENT_PHASE_ERRORS as exc:
                receipt.params_unchanged = False
                receipt.flag(f"checkpoint_post_probe_failed:{type(exc).__name__}")
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="refused output because the post-episode invariant probe failed",
                    severity="critical",
                )
        receipt.budget = budget.to_receipt()
        if (
            not failure_reason
            and receipt.decode_termination not in {"eos", "token_limit"}
        ):
            failure_reason = f"decode_incomplete:{receipt.decode_termination}"
        if receipt.params_unchanged is False:
            receipt.flag("checkpoint_invariant_violated")
            return LatentReasoningResult(
                ok=False,
                text="",
                receipt=receipt,
                reason="checkpoint_invariant_violated",
            )
        if failure_reason:
            return LatentReasoningResult(
                ok=False,
                text="",
                receipt=receipt,
                reason=failure_reason,
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
        *,
        ablate_slot: int | None = None,
        ablate_mode: str = "zero",
    ) -> tuple[list[int], EpisodeReceipt]:
        import mlx.core as mx

        cache = self._fresh_cache()
        runner = WindowRunner(self.model.model, budget)
        decode_limit = (
            decode_max_tokens
            if decode_max_tokens is not None
            else self.config.decode_max_tokens
        )
        prefill_cost = len(tokens) * self.n_layers
        decode_cost = max(0, int(decode_limit) - 1) * self.n_layers
        persist_cost = self.config.workspace.n_slots * self.n_layers
        fast_weight_probe_cost = (
            8 * self.n_layers if self.config.fast_weights.enabled else 0
        )
        completion_reserve = (
            persist_cost + decode_cost + fast_weight_probe_cost
        )
        fallback_reserve = prefill_cost + decode_cost
        safety_reserve = completion_reserve + fallback_reserve
        branch_seed_cost = (
            self.config.branches.n_branches
            * self.config.workspace.n_slots
            * self.prelude_end
        )
        fast_weight_baseline_cost = fast_weight_probe_cost
        minimum_admission = (
            prefill_cost
            + branch_seed_cost
            + safety_reserve
            + fast_weight_baseline_cost
        )
        if minimum_admission > budget.remaining_layer_apps or budget.exhausted:
            raise RuntimeError(
                "compute budget cannot admit latent minimum while preserving fallback: "
                f"required={minimum_admission} remaining={budget.remaining_layer_apps}"
            )

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
        recurrence_budget_limited = False
        for op in schedule.ops:
            for _ in range(op.repeats):
                if ensemble.all_halted() or budget.exhausted:
                    break
                admitted = ensemble.step_all(
                    runner,
                    cache,
                    op.start,
                    op.end,
                    budget=budget,
                    alpha_override=op.alpha,
                    reserve_layer_apps=safety_reserve,
                )
                if not admitted:
                    recurrence_budget_limited = True
                    break
            if recurrence_budget_limited:
                break
        for branch in ensemble.branches:
            if not branch.halted:
                final, reverted = branch.halting.final_state(branch.z)
                branch.z = final
                branch.workspace.update(final)
                branch.halted = True
                branch.halt_reason = (
                    "budget_reserved"
                    if recurrence_budget_limited
                    else "budget_exhausted"
                    if budget.exhausted
                    else "schedule_complete"
                ) + ("_reverted" if reverted else "")

        receipt.exchanges = ensemble.exchanges

        # ── Branch selection ─────────────────────────────────────────────
        branch_probe_cost = (
            len(ensemble.branches)
            * (self.config.workspace.n_slots + 47)
            * self.n_layers
        )
        if (
            verifier is not None
            and self.tokenizer is not None
            and branch_probe_cost + safety_reserve <= budget.remaining_layer_apps
        ):
            def branch_score(branch: BranchState) -> float:
                probe = self._decode_probe(branch, cache, runner, budget)
                return float(verifier(self.tokenizer.decode(probe)))

            winner = ensemble.select(score_fn=branch_score)
        else:
            if verifier is not None and self.tokenizer is not None:
                receipt.flag("branch_verifier_skipped_budget")
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
                loss_fn,
                self.config.latent_opt,
                seed=self.config.workspace.seed,
                budget=budget,
                # The proxy itself is norm + LM-head work rather than a full
                # transformer pass. Charging one full-stack slot pass per loss
                # evaluation is intentionally conservative and keeps one
                # common economy unit across recurrence, optimization, and
                # decoding until the FLOP ledger lands.
                layer_apps_per_loss=(
                    self.config.workspace.n_slots * self.n_layers
                ),
                reserve_layer_apps=safety_reserve,
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

                z_opt, _ = optimizer.run_with_verifier(
                    winner.z,
                    z_score,
                    verifier_layer_apps=(
                        (self.config.workspace.n_slots + 47) * self.n_layers
                    ),
                )
            else:
                z_opt = optimizer.run(winner.z)
            winner.z = z_opt
            winner.workspace.update(z_opt)
            receipt.latent_opt_applied = optimizer.trace.accepted > 0
            receipt.latent_opt_mode = optimizer.trace.mode
            receipt.latent_opt_loss_trail = list(optimizer.trace.loss_trail)
            receipt.latent_opt_attempts = optimizer.trace.attempts
            receipt.latent_opt_steps = optimizer.trace.accepted
            receipt.latent_opt_rejected = optimizer.trace.rejected
            receipt.latent_opt_budget_exhausted = optimizer.trace.budget_exhausted
            if optimizer.trace.budget_exhausted:
                receipt.flag("latent_opt_budget_exhausted")
            if not receipt.latent_opt_applied:
                receipt.flag("latent_opt_no_accepted_step")

        # ── Episode fast weights (attach → optimize → decode under ΔW) ──
        fast_weights: EpisodicFastWeights | None = None
        fw_baseline = None
        if self.config.fast_weights.enabled:
            fast_weights = EpisodicFastWeights(self.config.fast_weights)
            if fast_weight_baseline_cost + safety_reserve > budget.remaining_layer_apps:
                raise RuntimeError("compute budget cannot admit fast-weight baseline probe")
            fw_baseline = self._fw_probe(budget)
            seed_stat = float(mx.mean(per_position_rms(winner.z)))
            wrapped = fast_weights.attach(
                self.model.model,
                (self.prelude_end, self.coda_start),
                seed_stat=seed_stat,
                episode_id=receipt.episode_id,
            )
            receipt.fast_weights_applied = True
            receipt.fast_weights_layers = wrapped

        try:
            if fast_weights is not None:
                loss_fn = build_proxy_loss(
                    self.model, winner.anchor, tokens, self.config.latent_opt
                )

                def fw_loss():
                    z_pass = self._nocache_window_pass(winner.z)
                    return loss_fn(z_pass)

                fast_weights.optimize(
                    fw_loss,
                    budget=budget,
                    layer_apps_per_forward=(
                        self.config.workspace.n_slots
                        * (self.coda_start - self.prelude_end)
                    ),
                    reserve_layer_apps=safety_reserve,
                )

            # Experiment-3 instrumentation: destroy one refined thought slot
            # just before persistence, so its causal contribution and
            # restoration are measurable.
            if ablate_slot is not None:
                winner.workspace.ablate(int(ablate_slot), mode=ablate_mode)
                winner.z = winner.workspace.z
                receipt.flag(f"slot_ablated:{int(ablate_slot)}:{ablate_mode}")

            # ── Commit the winner + decode the answer ────────────────────
            slot_logits = self._persist_branch(winner, cache, runner)
            receipt.first_logits_digest = _logits_digest(slot_logits)
            out_tokens, decode_termination = self._decode(
                cache, budget, slot_logits, max_tokens=decode_max_tokens
            )
            receipt.decode_requested_tokens = decode_limit
            receipt.decode_generated_tokens = len(out_tokens)
            receipt.decode_termination = decode_termination
            if decode_termination.startswith("budget_"):
                receipt.flag(f"decode_{decode_termination}")
        finally:
            if fast_weights is not None:
                self._finalize_fast_weights(fast_weights, fw_baseline, receipt, budget)

        if fast_weights is not None and receipt.fast_weights_erased is not True:
            raise _FastWeightCleanupError("fast-weight cleanup proof did not pass")

        return out_tokens, receipt

    def _finalize_fast_weights(
        self,
        fast_weights: EpisodicFastWeights,
        baseline,
        receipt: EpisodeReceipt,
        budget: ComputeBudget,
    ) -> None:
        """Best-effort cleanup that never masks the episode's first failure."""
        try:
            try:
                fast_weights.snapshot_for_export()
            except _LATENT_PHASE_ERRORS as exc:
                receipt.flag(f"fast_weight_snapshot_failed:{type(exc).__name__}")
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="discarded fast-weight consolidation snapshot before cleanup",
                    severity="warning",
                )
        finally:
            try:
                fast_weights.detach()
                receipt.fast_weights_erased = fast_weights.prove_erase(
                    lambda: self._fw_probe(budget), baseline
                )
            except _LATENT_PHASE_ERRORS as exc:
                receipt.fast_weights_erased = False
                receipt.flag(f"fast_weight_cleanup_failed:{type(exc).__name__}")
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="refused episode output and requested resident-worker recycle",
                    severity="critical",
                )
        if receipt.fast_weights_erased is not True:
            receipt.flag("fast_weight_erase_unproven")
        lifecycle = fast_weights.lifecycle
        receipt.fast_weight_optimization_attempts = lifecycle.optimization_attempts
        receipt.fast_weight_optimized_steps = lifecycle.optimized_steps
        receipt.fast_weight_rejected_steps = lifecycle.rejected_steps
        receipt.fast_weight_budget_exhausted = lifecycle.budget_exhausted
        if lifecycle.budget_exhausted:
            receipt.flag("fast_weight_budget_exhausted")
        if lifecycle.optimized_steps <= 0:
            receipt.flag("fast_weight_no_accepted_step")

    # ── Fast-weight helpers ─────────────────────────────────────────────
    def _nocache_window_pass(self, z):
        """Window pass with no cache (grad-safe: zero side effects)."""
        inner = self.model.model
        h = z
        for i in range(self.prelude_end, self.coda_start):
            h = inner.layers[i](h, None, None)
        return h

    def _fw_probe(self, budget: ComputeBudget):
        """Deterministic full-stack probe for erase proofs (cache-free)."""
        import mlx.core as mx

        inner = self.model.model
        vocab = inner.embed_tokens.weight.shape[0]
        probe_tokens = mx.array([[i % int(vocab) for i in range(1, 9)]])
        if not budget.can_afford(8, self.n_layers):
            raise RuntimeError("compute budget cannot afford fast-weight erase probe")
        budget.charge(tokens=8, layers=self.n_layers)
        h = inner.embed_tokens(probe_tokens)
        for layer in inner.layers:
            h = layer(h, None, None)
        out = self._logits(h)
        mx.eval(out)
        return out


__all__ = ["LatentCortexEngine"]
