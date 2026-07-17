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

import hashlib
import json
import logging
from pathlib import Path
import time
import uuid
from collections.abc import Callable
from typing import Any

from core.brain.llm.latent_cortex.branches import BranchEnsemble, BranchState
from core.brain.llm.latent_cortex.capability_canaries import (
    CapabilityCanaries,
    compare_canaries,
)
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

_ASSISTANT_ANSWER_BRIDGE = "\nFinal answer:\n"
# v2 demands complete coverage per token spent: compound requests fail the
# product-quality gate when the decode budget is burned on preamble instead
# of the asked-for facets. The cue is generic — it names no specific task.
_ASSISTANT_ANSWER_BRIDGE_V2 = (
    "\nFinal answer (address every part of the request, concisely):\n"
)
_BRIDGE_TEXT_BY_POLICY = {
    "assistant_answer_v1": _ASSISTANT_ANSWER_BRIDGE,
    "assistant_answer_v2": _ASSISTANT_ANSWER_BRIDGE_V2,
}
# Decode discipline: at most this many consecutive pure-newline tokens are
# admitted before newline-family logits are masked for the next sample. Two
# newlines = one blank line — enough for any legitimate paragraph/list break.
_MAX_NEWLINE_RUN = 2
_NEWLINE_RESAMPLE_ATTEMPTS = 4
# Sentence grace: when the token limit lands mid-sentence, sampling may
# continue up to this many extra tokens until sentence-final punctuation —
# still entirely model-sampled tokens, charged to the budget, receipted as
# termination "token_limit_sentence_grace". A truncated tail otherwise
# fails the product gate as a terminal fragment (CP110 live evidence).
_SENTENCE_GRACE_TOKENS = 24
_SENTENCE_TERMINALS = (".", "!", "?", ".\n", "!\n", "?\n")

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


class _LatentEpisodeCancelled(Exception):
    """Cooperative cancellation observed at a checkpoint-safe boundary."""


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
        # Decode discipline state: token→is-pure-newline verdicts (per
        # tokenizer, so per engine) and the last final-decode suppression
        # count for the episode receipt.
        self._newline_token_cache: dict[int, bool] = {}
        self._last_decode_newline_suppressions = 0
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

    @staticmethod
    def _cancel_requested(cancel_check: Callable[[], bool] | None) -> bool:
        if cancel_check is None:
            return False
        try:
            return bool(cancel_check())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def _emit_progress(
        progress: Callable[[dict], None] | None,
        payload: dict,
    ) -> None:
        if progress is None:
            return
        try:
            progress(dict(payload))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.debug("Latent progress callback failed; episode continues.")

    def _stage_checkpoint(
        self,
        *,
        receipt: EpisodeReceipt,
        budget: ComputeBudget,
        stage: str,
        stage_started: float,
        episode_started: float,
        progress: Callable[[dict], None] | None,
        cancel_check: Callable[[], bool] | None,
        **detail,
    ) -> float:
        now = time.monotonic()
        duration_s = max(0.0, now - stage_started)
        receipt.last_stage = str(stage)
        receipt.stage_timings_s[str(stage)] = round(
            float(receipt.stage_timings_s.get(str(stage), 0.0)) + duration_s,
            6,
        )
        self._emit_progress(
            progress,
            {
                "stage": str(stage),
                "stage_duration_s": round(duration_s, 6),
                "elapsed_s": round(max(0.0, now - episode_started), 6),
                "spent_layer_apps": int(budget.spent_layer_apps),
                **detail,
            },
        )
        if self._cancel_requested(cancel_check):
            receipt.flag("soft_cancelled")
            receipt.halting_reason = receipt.halting_reason or f"cancelled_after_{stage}"
            raise _LatentEpisodeCancelled(str(stage))
        return now

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

    def _decode_bridge_tokens(self) -> list[int]:
        policy = self.config.decode_bridge_policy
        if policy == "none":
            return []
        bridge_text = _BRIDGE_TEXT_BY_POLICY.get(policy)
        if bridge_text is None:
            raise ValueError(f"unsupported decode bridge policy: {policy}")
        if self.tokenizer is None:
            raise ValueError("assistant answer decode bridge requires a tokenizer")
        try:
            encoded = self.tokenizer.encode(
                bridge_text,
                add_special_tokens=False,
            )
        except TypeError:
            encoded = self.tokenizer.encode(bridge_text)
        tokens = list(encoded)
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("assistant answer decode bridge produced invalid tokens")
        return tokens

    def _token_ends_sentence(self, token: int) -> bool:
        """True when the token's rendered text ends at a sentence boundary."""
        if self.tokenizer is None:
            return False
        try:
            piece = self.tokenizer.decode([token])
        except (TypeError, ValueError, KeyError, AttributeError):
            return False
        stripped = str(piece).rstrip()
        return stripped.endswith(_SENTENCE_TERMINALS) if stripped else False

    def _is_pure_newline_token(self, token: int) -> bool:
        """True when the token renders to newline-only whitespace."""
        if self.tokenizer is None:
            return False
        cached = self._newline_token_cache.get(token)
        if cached is not None:
            return cached
        try:
            piece = self.tokenizer.decode([token])
        except (TypeError, ValueError, KeyError, AttributeError):
            piece = ""
        verdict = bool(piece) and piece.strip() == "" and "\n" in piece
        self._newline_token_cache[token] = verdict
        return verdict

    def _apply_decode_bridge(self, cache, budget: ComputeBudget, tokens: list[int]):
        """Append a lexical answer cue after thought slots in the same KV owner."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        if not tokens:
            raise ValueError("decode bridge tokens cannot be empty")
        if not budget.can_afford(len(tokens), self.n_layers):
            raise RuntimeError("compute budget cannot admit decode bridge")
        budget.charge(tokens=len(tokens), layers=self.n_layers)
        inner = self.model.model
        h = inner.embed_tokens(mx.array([tokens]))
        mask = create_attention_mask(h, cache)
        for index, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[index])
        logits = self._logits(h)[0, -1]
        mx.eval(logits)
        return logits

    # ── Typed cognitive ingress into the workspace ──────────────────────
    _MAX_COGNITIVE_CONTEXT_ITEMS = 6
    _MAX_COGNITIVE_CONTEXT_CHARS = 400
    _MAX_COGNITIVE_CONTEXT_TOKENS = 64

    def _validate_cognitive_context(
        self, cognitive_context: list | None
    ) -> list[dict]:
        if cognitive_context is None:
            return []
        if not isinstance(cognitive_context, list):
            raise ValueError("cognitive_context must be a list")
        items: list[dict] = []
        for entry in cognitive_context[: self._MAX_COGNITIVE_CONTEXT_ITEMS]:
            if not isinstance(entry, dict):
                raise ValueError("cognitive_context entries must be mappings")
            source = entry.get("source")
            text = entry.get("text")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("cognitive_context entry requires a source string")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("cognitive_context entry requires a text string")
            items.append(
                {
                    "source": source.strip()[:40],
                    "text": text.strip()[: self._MAX_COGNITIVE_CONTEXT_CHARS],
                }
            )
        return items

    def _embed_cognitive_context(
        self, items: list[dict]
    ) -> list[tuple[str, "object"]]:
        """Pooled embed_tokens vectors for each organ item — no layer passes.

        Embedding lookup is table indexing, so the ingress costs no layer
        applications; the seeded slots then ride every subsequent window pass
        exactly like ordinary thought slots (charged as usual).
        """
        import mlx.core as mx

        if not items or self.tokenizer is None:
            return []
        inner = self.model.model
        seeds: list[tuple[str, object]] = []
        for item in items:
            try:
                encoded = self.tokenizer.encode(
                    item["text"], add_special_tokens=False
                )
            except TypeError:
                encoded = self.tokenizer.encode(item["text"])
            tokens = list(encoded)[: self._MAX_COGNITIVE_CONTEXT_TOKENS]
            if not tokens:
                continue
            h = inner.embed_tokens(mx.array([tokens]))
            pooled = mx.mean(h, axis=1, keepdims=True)  # (1,1,D)
            mx.eval(pooled)
            seeds.append((item["source"], pooled))
        return seeds

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

    def _sample(self, logits, temperature: float, top_p: float = 1.0) -> int:
        import mlx.core as mx

        if temperature and temperature > 0:
            scaled = logits / temperature
            if top_p < 1.0:
                probabilities = mx.softmax(scaled)
                sorted_indices = mx.argsort(-probabilities)
                sorted_probabilities = probabilities[sorted_indices]
                cumulative = mx.cumsum(sorted_probabilities)
                keep = (cumulative - sorted_probabilities) < top_p
                filtered_logits = mx.where(
                    keep,
                    mx.log(sorted_probabilities),
                    mx.full(sorted_probabilities.shape, -1e9),
                )
                selected = int(mx.random.categorical(filtered_logits))
                return int(sorted_indices[selected])
            return int(mx.random.categorical(scaled))
        return int(mx.argmax(logits))

    def _decode(
        self,
        cache,
        budget: ComputeBudget,
        initial_logits,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress: Callable[[dict], None] | None = None,
        wall_reserve_s: float = 0.0,
    ) -> tuple[list[int], str]:
        """Minimal sampler: first token from ``initial_logits`` (the logits of
        the last persisted position — prompt tail or final thought slot), then
        autoregressive continuation over the populated cache.

        ``wall_reserve_s`` stops decoding while that much wall clock still
        remains — the engine reserves cleanup time when fast weights are
        attached, so a long answer degrades to token truncation instead of
        endangering the erase proof."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        inner = self.model.model
        eos = self._eos_ids()
        limit = max_tokens if max_tokens is not None else self.config.decode_max_tokens
        temp = temperature if temperature is not None else self.config.decode_temperature
        nucleus = top_p if top_p is not None else self.config.decode_top_p

        out: list[int] = []
        newline_run = 0
        suppressions = 0
        self._last_decode_newline_suppressions = 0
        if budget.exhausted:
            return out, "budget_exhausted"

        penalty = float(self.config.decode_repetition_penalty)
        window = max(1, int(self.config.decode_repetition_window))

        def penalize_repeats(logits):
            """CTRL-style sliding-window repetition penalty.

            Degeneration loops (one line sampled forever) survive any single
            temperature; dividing the logits of recently-emitted tokens is
            the standard, receipted guard. penalty=1.0 disables it."""
            if penalty <= 1.0 or not out:
                return logits
            recent = sorted(set(out[-window:]))
            ids = mx.array(recent)
            gathered = logits[ids]
            adjusted = mx.where(gathered > 0, gathered / penalty, gathered * penalty)
            return logits.at[ids].add(adjusted - gathered)

        def sample_disciplined(logits):
            """Sample under the newline-run discipline.

            A run of more than _MAX_NEWLINE_RUN pure-newline tokens is decode
            babble: it wastes answer budget and independently fails the
            product-quality gate (excessive_blank_lines). Masking newline
            logits for the next sample is a sampling CONSTRAINT — the emitted
            text is still entirely the model's own tokens, never edited."""
            nonlocal suppressions
            logits = penalize_repeats(logits)
            token = self._sample(logits, temp, nucleus)
            if self.tokenizer is None or newline_run < _MAX_NEWLINE_RUN:
                return token
            masked = logits
            for _ in range(_NEWLINE_RESAMPLE_ATTEMPTS):
                if not self._is_pure_newline_token(token):
                    return token
                suppressions += 1
                masked = mx.where(
                    mx.arange(masked.shape[-1]) == token,
                    mx.full(masked.shape, -1e9),
                    masked,
                )
                token = self._sample(masked, temp, nucleus)
            return token

        token = sample_disciplined(initial_logits)
        termination = "token_limit"
        for index in range(max(1, int(limit)) + _SENTENCE_GRACE_TOKENS):
            if self._cancel_requested(cancel_check):
                raise _LatentEpisodeCancelled("decode")
            if token in eos:
                termination = "eos"
                break
            out.append(token)
            newline_run = newline_run + 1 if self._is_pure_newline_token(token) else 0
            if index + 1 >= int(limit):
                sentence_done = (
                    self.tokenizer is None or self._token_ends_sentence(token)
                )
                if sentence_done:
                    termination = (
                        "token_limit"
                        if index + 1 == int(limit)
                        else "token_limit_sentence_grace"
                    )
                    break
                if index + 1 >= int(limit) + _SENTENCE_GRACE_TOKENS:
                    # Grace exhausted without punctuation: still a fragment,
                    # and the receipt says so honestly.
                    termination = "token_limit"
                    break
            if budget.exhausted:
                termination = "budget_exhausted"
                break
            if wall_reserve_s > 0.0 and budget.remaining_wall_s < wall_reserve_s:
                termination = "wall_reserve"
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
            token = sample_disciplined(logits)
            if (index + 1) % 16 == 0:
                self._emit_progress(
                    progress,
                    {
                        "stage": "decode",
                        "decode_generated_tokens": len(out),
                        "decode_requested_tokens": int(limit),
                        "spent_layer_apps": int(budget.spent_layer_apps),
                    },
                )
        self._last_decode_newline_suppressions = suppressions
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
        bridge_tokens: list[int] | None = None,
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
            if bridge_tokens:
                slot_logits = self._apply_decode_bridge(
                    cache,
                    budget,
                    bridge_tokens,
                )
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
        cognitive_context: list | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress: Callable[[dict], None] | None = None,
    ) -> LatentReasoningResult:
        receipt = EpisodeReceipt(episode_id=uuid.uuid4().hex[:12])
        episode_started = time.monotonic()
        receipt.n_layers = self.n_layers
        receipt.prelude_end = self.prelude_end
        receipt.coda_start = self.coda_start
        budget = budget or ComputeBudget()
        if decode_max_tokens is not None:
            if type(decode_max_tokens) is not int:
                raise TypeError("decode_max_tokens override must be an integer")
            if not 1 <= decode_max_tokens <= 8192:
                raise ValueError("decode_max_tokens override outside [1, 8192]")
        context_items = self._validate_cognitive_context(cognitive_context)
        tokens = self._encode(prompt, messages, token_ids)
        encoded_tokens = json.dumps(tokens, separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
        receipt.input_tokens_sha256 = hashlib.sha256(encoded_tokens).hexdigest()
        receipt.input_token_count = len(tokens)
        receipt.decode_temperature = float(self.config.decode_temperature)
        receipt.decode_top_p = float(self.config.decode_top_p)
        receipt.decode_bridge_policy = self.config.decode_bridge_policy

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
                    cognitive_context_items=context_items,
                    cancel_check=cancel_check,
                    progress=progress,
                    episode_started=episode_started,
                )
            except _FastWeightCleanupError as exc:
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="refused fallback decode and requested resident-worker recycle",
                    severity="critical",
                )
                failure_reason = "fast_weight_cleanup_unproven"
            except _LatentEpisodeCancelled:
                receipt.flag("soft_cancelled")
                receipt.halting_reason = receipt.halting_reason or "soft_cancelled"
                failure_reason = "soft_cancelled"
            except _LATENT_PHASE_ERRORS as exc:
                record_degradation(
                    "latent_cortex",
                    exc,
                    action=(
                        "served vanilla decode with honest fallback receipt"
                        if self.config.allow_vanilla_fallback
                        else "failed the full-stack episode without replacing it with vanilla decode"
                    ),
                )
                receipt.halting_reason = receipt.halting_reason or "latent_phase_error"
                if not self.config.allow_vanilla_fallback:
                    receipt.flag("vanilla_fallback_disabled")
                    failure_reason = f"latent_phase_failed:{type(exc).__name__}:{exc}"
                elif receipt.fast_weights_applied and receipt.fast_weights_erased is not True:
                    receipt.flag("fallback_refused_unproven_model_state")
                    failure_reason = "fast_weight_cleanup_unproven"
                else:
                    receipt.flag(f"fallback_vanilla:{type(exc).__name__}")
                    try:
                        cache = self._fresh_cache()
                        _, tail_logits = self._prefill(tokens, cache, budget)
                        out_tokens, decode_termination = self._decode(
                            cache,
                            budget,
                            tail_logits,
                            max_tokens=decode_max_tokens,
                            cancel_check=cancel_check,
                            progress=progress,
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
                    except _LatentEpisodeCancelled:
                        receipt.flag("soft_cancelled")
                        receipt.halting_reason = (
                            receipt.halting_reason or "soft_cancelled"
                        )
                        failure_reason = "soft_cancelled"
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
        receipt.last_stage = "complete" if not failure_reason else receipt.last_stage
        receipt.stage_timings_s["total"] = round(
            max(0.0, time.monotonic() - episode_started),
            6,
        )
        self._emit_progress(
            progress,
            {
                "stage": "complete" if not failure_reason else "failed",
                "last_stage": receipt.last_stage,
                "elapsed_s": receipt.stage_timings_s["total"],
                "reason": failure_reason,
                "spent_layer_apps": int(budget.spent_layer_apps),
            },
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
        cognitive_context_items: list[dict] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress: Callable[[dict], None] | None = None,
        episode_started: float | None = None,
    ) -> tuple[list[int], EpisodeReceipt]:
        import mlx.core as mx

        episode_started = (
            float(episode_started)
            if episode_started is not None
            else time.monotonic()
        )
        stage_started = time.monotonic()
        if self._cancel_requested(cancel_check):
            raise _LatentEpisodeCancelled("admission")
        cache = self._fresh_cache()
        runner = WindowRunner(self.model.model, budget)
        decode_limit = (
            decode_max_tokens
            if decode_max_tokens is not None
            else self.config.decode_max_tokens
        )
        bridge_tokens = self._decode_bridge_tokens()
        prefill_cost = len(tokens) * self.n_layers
        decode_cost = max(0, int(decode_limit) - 1) * self.n_layers
        persist_cost = self.config.workspace.n_slots * self.n_layers
        bridge_cost = len(bridge_tokens) * self.n_layers
        fast_weight_probe_cost = (
            8 * self.n_layers if self.config.fast_weights.enabled else 0
        )
        canaries: CapabilityCanaries | None = None
        canary_pass_cost = 0
        canary_reserve = 0
        if self.config.fast_weights.enabled and self.config.fast_weights.canary_enabled:
            canaries = CapabilityCanaries(
                self.tokenizer,
                vocab_size=int(self.model.model.embed_tokens.weight.shape[0]),
                max_tokens_per_canary=self.config.fast_weights.canary_max_tokens,
            )
            canary_pass_cost = canaries.tokens_per_measurement * self.n_layers
            # One adapted measurement plus one re-measurement per allowed
            # rescale; the baseline pass is charged with the erase baseline.
            canary_reserve = canary_pass_cost * (
                1 + max(0, self.config.fast_weights.canary_rescale_attempts)
            )
        completion_reserve = (
            persist_cost + bridge_cost + decode_cost + fast_weight_probe_cost
        )
        fallback_reserve = prefill_cost + decode_cost
        safety_reserve = completion_reserve + fallback_reserve + canary_reserve
        branch_seed_cost = (
            self.config.branches.n_branches
            * self.config.workspace.n_slots
            * self.prelude_end
        )
        fast_weight_baseline_cost = fast_weight_probe_cost + canary_pass_cost
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
        stage_started = self._stage_checkpoint(
            receipt=receipt,
            budget=budget,
            stage="prefill",
            stage_started=stage_started,
            episode_started=episode_started,
            progress=progress,
            cancel_check=cancel_check,
            input_tokens=len(tokens),
        )

        schedule = self._resolve_schedule(domain)
        receipt.domain = str(domain or "general")
        receipt.schedule_hash = schedule.schedule_hash
        receipt.n_slots = self.config.workspace.n_slots
        receipt.n_branches = self.config.branches.n_branches

        context_seeds = self._embed_cognitive_context(
            list(cognitive_context_items or [])
        )
        ensemble = BranchEnsemble.seed(
            embeddings,
            self.config.workspace,
            self.config.branches,
            self.config.recurrence,
            runner,
            cache,
            self.prelude_end,
            context_seeds=context_seeds,
        )
        if ensemble.branches and ensemble.branches[0].workspace.context_slots:
            seeded = ensemble.branches[0].workspace.context_slots
            by_source = {
                item["source"]: item for item in (cognitive_context_items or [])
            }
            receipt.cognitive_slots = [
                {
                    "slot": row["slot"],
                    "source": row["source"],
                    "text_chars": len(
                        by_source.get(row["source"], {}).get("text", "")
                    ),
                    "text_sha256": hashlib.sha256(
                        by_source.get(row["source"], {})
                        .get("text", "")
                        .encode("utf-8")
                    ).hexdigest(),
                }
                for row in seeded
            ]
        stage_started = self._stage_checkpoint(
            receipt=receipt,
            budget=budget,
            stage="branch_seed",
            stage_started=stage_started,
            episode_started=episode_started,
            progress=progress,
            cancel_check=cancel_check,
            branches=len(ensemble.branches),
            slots=self.config.workspace.n_slots,
            cognitive_slots=len(receipt.cognitive_slots),
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
                stage_started = self._stage_checkpoint(
                    receipt=receipt,
                    budget=budget,
                    stage="recurrence",
                    stage_started=stage_started,
                    episode_started=episode_started,
                    progress=progress,
                    cancel_check=cancel_check,
                    max_branch_steps=max(
                        (branch.steps for branch in ensemble.branches),
                        default=0,
                    ),
                    exchanges=ensemble.exchanges,
                )
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
            * (self.config.workspace.n_slots + len(bridge_tokens) + 47)
            * self.n_layers
        )
        if (
            verifier is not None
            and self.tokenizer is not None
            and branch_probe_cost + safety_reserve <= budget.remaining_layer_apps
        ):
            def branch_score(branch: BranchState) -> float:
                probe = self._decode_probe(
                    branch,
                    cache,
                    runner,
                    budget,
                    bridge_tokens=bridge_tokens,
                )
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
        stage_started = self._stage_checkpoint(
            receipt=receipt,
            budget=budget,
            stage="branch_select",
            stage_started=stage_started,
            episode_started=episode_started,
            progress=progress,
            cancel_check=cancel_check,
            selected_branch=winner.index,
            steps_taken=winner.steps,
            exchanges=ensemble.exchanges,
        )

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
                        probe = self._decode_probe(
                            winner,
                            cache,
                            runner,
                            budget,
                            bridge_tokens=bridge_tokens,
                        )
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
            # "Applied" means the optimizer genuinely RAN (attempts > 0).
            # Under verifier guidance, rejecting every proposal is the
            # verifier doing its job — receipted via latent_opt_steps=0 and
            # the latent_opt_no_accepted_step flag, never faked as success
            # and never treated as "optimization didn't happen".
            receipt.latent_opt_applied = optimizer.trace.attempts > 0
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
            stage_started = self._stage_checkpoint(
                receipt=receipt,
                budget=budget,
                stage="latent_optimization",
                stage_started=stage_started,
                episode_started=episode_started,
                progress=progress,
                cancel_check=cancel_check,
                attempts=optimizer.trace.attempts,
                accepted=optimizer.trace.accepted,
            )

        # ── Episode fast weights (attach → optimize → decode under ΔW) ──
        fast_weights: EpisodicFastWeights | None = None
        fw_baseline = None
        canary_baseline: dict[str, float] | None = None
        if self.config.fast_weights.enabled:
            fast_weights = EpisodicFastWeights(self.config.fast_weights)
            if fast_weight_baseline_cost + safety_reserve > budget.remaining_layer_apps:
                raise RuntimeError("compute budget cannot admit fast-weight baseline probe")
            fw_baseline = self._fw_probe(budget)
            if canaries is not None:
                canary_baseline = canaries.measure(
                    lambda probe_tokens: self._canary_logits(probe_tokens, budget)
                )
            stage_started = self._stage_checkpoint(
                receipt=receipt,
                budget=budget,
                stage="fast_weight_baseline",
                stage_started=stage_started,
                episode_started=episode_started,
                progress=progress,
                cancel_check=cancel_check,
            )
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
                stage_started = self._stage_checkpoint(
                    receipt=receipt,
                    budget=budget,
                    stage="fast_weight_attach",
                    stage_started=stage_started,
                    episode_started=episode_started,
                    progress=progress,
                    cancel_check=cancel_check,
                    wrapped_layers=receipt.fast_weights_layers,
                )
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
                stage_started = self._stage_checkpoint(
                    receipt=receipt,
                    budget=budget,
                    stage="fast_weight_optimization",
                    stage_started=stage_started,
                    episode_started=episode_started,
                    progress=progress,
                    cancel_check=cancel_check,
                    attempts=fast_weights.lifecycle.optimization_attempts,
                    accepted=fast_weights.lifecycle.optimized_steps,
                )
                if canaries is not None and canary_baseline is not None:
                    canary_decision = self._enforce_fast_weight_canaries(
                        canaries,
                        canary_baseline,
                        fast_weights,
                        receipt,
                        budget,
                    )
                    stage_started = self._stage_checkpoint(
                        receipt=receipt,
                        budget=budget,
                        stage="fast_weight_canaries",
                        stage_started=stage_started,
                        episode_started=episode_started,
                        progress=progress,
                        cancel_check=cancel_check,
                        decision=canary_decision,
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
            stage_started = self._stage_checkpoint(
                receipt=receipt,
                budget=budget,
                stage="persist",
                stage_started=stage_started,
                episode_started=episode_started,
                progress=progress,
                cancel_check=cancel_check,
            )
            decode_logits = slot_logits
            if bridge_tokens:
                decode_logits = self._apply_decode_bridge(
                    cache,
                    budget,
                    bridge_tokens,
                )
                serialized_bridge = json.dumps(
                    bridge_tokens,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("ascii")
                receipt.decode_bridge_applied = True
                receipt.decode_bridge_token_count = len(bridge_tokens)
                receipt.decode_bridge_tokens_sha256 = hashlib.sha256(
                    serialized_bridge
                ).hexdigest()
                receipt.decode_bridge_logits_digest = _logits_digest(decode_logits)
                stage_started = self._stage_checkpoint(
                    receipt=receipt,
                    budget=budget,
                    stage="decode_bridge",
                    stage_started=stage_started,
                    episode_started=episode_started,
                    progress=progress,
                    cancel_check=cancel_check,
                    bridge_policy=self.config.decode_bridge_policy,
                    bridge_tokens=len(bridge_tokens),
                )
            out_tokens, decode_termination = self._decode(
                cache,
                budget,
                decode_logits,
                max_tokens=decode_max_tokens,
                cancel_check=cancel_check,
                progress=progress,
                # Cleanup time is sacrosanct: with temporary synapses attached
                # the decode surrenders its tail rather than let the wall
                # clock expire before the erase proof.
                wall_reserve_s=(
                    6.0 if self.config.fast_weights.enabled else 0.0
                ),
            )
            receipt.decode_requested_tokens = decode_limit
            receipt.decode_generated_tokens = len(out_tokens)
            receipt.decode_termination = decode_termination
            receipt.decode_newline_suppressions = int(
                self._last_decode_newline_suppressions
            )
            receipt.decode_repetition_penalty_applied = float(
                self.config.decode_repetition_penalty
            )
            if decode_termination.startswith("budget_") or decode_termination == "wall_reserve":
                receipt.flag(f"decode_{decode_termination}")
            stage_started = self._stage_checkpoint(
                receipt=receipt,
                budget=budget,
                stage="decode",
                stage_started=stage_started,
                episode_started=episode_started,
                progress=progress,
                cancel_check=cancel_check,
                generated_tokens=len(out_tokens),
                termination=decode_termination,
            )
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
                cleanup_overdraft = not budget.can_afford(8, self.n_layers)
                receipt.fast_weights_erased = fast_weights.prove_erase(
                    lambda: self._fw_probe(budget, cleanup=True), baseline
                )
                if cleanup_overdraft:
                    receipt.flag("fast_weight_cleanup_overdraft")
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
        receipt.fast_weight_optimizer = lifecycle.optimizer
        receipt.fast_weight_loss_trail = list(lifecycle.loss_trail)
        receipt.fast_weight_gradient_norm_trail = list(
            lifecycle.gradient_global_norm_trail
        )
        receipt.fast_weight_accepted_step_sizes = list(
            lifecycle.accepted_step_sizes
        )
        receipt.fast_weight_line_search_backtracks = (
            lifecycle.line_search_backtracks
        )
        if lifecycle.budget_exhausted:
            receipt.flag("fast_weight_budget_exhausted")
        if lifecycle.optimized_steps <= 0:
            receipt.flag("fast_weight_no_accepted_step")

        # Consolidation handoff: a mechanically clean episode (accepted
        # descent, proven erase) exports its temporary synapses + evidence
        # to the governed queue. Export is EVIDENCE COLLECTION only — the
        # consolidation consumer and the compounding loop's regression gates
        # decide what (if anything) becomes durable learning.
        if (
            self.config.fast_weights.export_candidates
            and receipt.fast_weights_erased is True
            and not lifecycle.canary_erased
            and lifecycle.optimized_steps > 0
            and len(lifecycle.loss_trail) >= 2
            and lifecycle.loss_trail[-1] < lifecycle.loss_trail[0]
        ):
            try:
                from core.config import DATA_DIR

                queue_dir = Path(DATA_DIR) / "latent_cortex" / "consolidation_queue"
                exported = fast_weights.export_candidate(
                    queue_dir,
                    episode_id=receipt.episode_id,
                    evidence={
                        "schema": "aura.latent_consolidation_candidate.v1",
                        "domain": receipt.domain,
                        "schedule_hash": receipt.schedule_hash,
                        "loss_trail": list(lifecycle.loss_trail),
                        "accepted_step_sizes": list(lifecycle.accepted_step_sizes),
                        "steps_taken": receipt.steps_taken,
                        "checkpoint_fingerprint": receipt.checkpoint_fingerprint,
                        "honest_flags": list(receipt.honest_flags),
                    },
                )
                if exported is not None:
                    receipt.flag("fast_weight_candidate_exported")
            except _LATENT_PHASE_ERRORS as exc:
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="dropped consolidation candidate after export failed",
                    severity="warning",
                )

    # ── Fast-weight helpers ─────────────────────────────────────────────
    def _enforce_fast_weight_canaries(
        self,
        canaries: CapabilityCanaries,
        baseline: dict[str, float],
        fast_weights: EpisodicFastWeights,
        receipt: EpisodeReceipt,
        budget: ComputeBudget,
    ) -> str:
        """Measure protected behaviors under active ΔW; rescale then erase.

        Runs only when at least one optimization step was accepted — before
        that, V is still zero and the adapted function is bit-identical to
        the baseline, so a measurement would spend budget to learn nothing.
        """
        cfg = self.config.fast_weights
        if fast_weights.lifecycle.optimized_steps <= 0:
            receipt.fast_weight_canaries = {
                "evaluated": False,
                "decision": "identity_no_check",
                "rescales": 0,
            }
            return "identity_no_check"
        max_rescales = max(0, int(cfg.canary_rescale_attempts))
        decision = "accepted"
        rescales = 0
        comparison: dict[str, Any] = {}
        while True:
            adapted = canaries.measure(
                lambda probe_tokens: self._canary_logits(probe_tokens, budget)
            )
            comparison = compare_canaries(
                baseline,
                adapted,
                max_logprob_drop=cfg.canary_max_logprob_drop,
            )
            if not comparison["regressed"]:
                break
            if rescales >= max_rescales:
                fast_weights.canary_erase()
                decision = "erased"
                receipt.flag("fast_weight_canary_erased")
                logger.info(
                    "Fast-weight canaries erased ΔW after %d rescales: %s",
                    rescales,
                    ",".join(comparison["regressed"]),
                )
                break
            fast_weights.rescale(0.5)
            rescales += 1
            decision = "rescaled"
            receipt.flag("fast_weight_canary_rescaled")
        receipt.fast_weight_canaries = {
            "evaluated": True,
            "decision": decision,
            "rescales": rescales,
            **comparison,
        }
        return decision

    def _canary_logits(self, probe_tokens: list[int], budget: ComputeBudget):
        """Standard causal full-stack forward over one canary sequence."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        if not budget.can_afford(len(probe_tokens), self.n_layers):
            raise RuntimeError("compute budget cannot afford capability canary pass")
        budget.charge(tokens=len(probe_tokens), layers=self.n_layers)
        inner = self.model.model
        cache = self._fresh_cache()
        h = inner.embed_tokens(mx.array([probe_tokens]))
        mask = create_attention_mask(h, cache)
        for index, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[index])
        logits = self._logits(h)
        mx.eval(logits)
        return logits

    def _nocache_window_pass(self, z):
        """Window pass with no cache (grad-safe: zero side effects)."""
        inner = self.model.model
        h = z
        for i in range(self.prelude_end, self.coda_start):
            h = inner.layers[i](h, None, None)
        return h

    def _fw_probe(self, budget: ComputeBudget, *, cleanup: bool = False):
        """Deterministic full-stack probe for erase proofs (cache-free).

        ``cleanup=True`` marks the post-detach integrity proof: it is a
        SAFETY OBLIGATION, not discretionary work, so it may overdraw an
        exhausted budget (honestly charged and flagged upstream) rather than
        refuse — CP104's live turn proved that refusing cleanup for budget
        reasons converts a slow answer into a critical worker recycle."""
        import mlx.core as mx

        inner = self.model.model
        vocab = inner.embed_tokens.weight.shape[0]
        probe_tokens = mx.array([[i % int(vocab) for i in range(1, 9)]])
        if cleanup:
            budget.charge_cleanup_overdraft(tokens=8, layers=self.n_layers)
        else:
            if not budget.can_afford(8, self.n_layers):
                raise RuntimeError(
                    "compute budget cannot afford fast-weight erase probe"
                )
            budget.charge(tokens=8, layers=self.n_layers)
        h = inner.embed_tokens(probe_tokens)
        for layer in inner.layers:
            h = layer(h, None, None)
        out = self._logits(h)
        mx.eval(out)
        return out


__all__ = ["LatentCortexEngine"]
