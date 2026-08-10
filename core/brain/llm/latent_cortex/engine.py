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
import inspect
import json
import logging
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.action_state_capture import (
    UnknownActionStateApplicationError,
)
from core.brain.llm.latent_cortex.branches import BranchEnsemble, BranchState
from core.brain.llm.latent_cortex.capability_canaries import (
    CapabilityCanaries,
    canary_verdict,
    compare_canaries,
)
from core.brain.llm.latent_cortex.episodic_output_memory import (
    OUTPUT_MEMORY_GAIN_GRID,
    EpisodicOutputMemory,
    build_output_memory_experiment_receipt,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.escape import EscapeConfig
from core.brain.llm.latent_cortex.fast_weight_learning import (
    empty_learning_state,
    finalize_fast_weight_learning_receipt,
    token_sequence_sha256,
    unavailable_admission,
    validate_fast_weight_admission,
)
from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights
from core.brain.llm.latent_cortex.governance import CheckpointInvariant
from core.brain.llm.latent_cortex.latent_opt import LatentOptimizer, build_proxy_loss
from core.brain.llm.latent_cortex.plasticity_sites import PLASTICITY_SITE_REGISTRY
from core.brain.llm.latent_cortex.probe_cache import DecodeProbeCache
from core.brain.llm.latent_cortex.recurrence import WindowRunner
from core.brain.llm.latent_cortex.resource_accounting import (
    build_information_receipt,
    policy_sha256,
    triangular_attention_pairs,
)
from core.brain.llm.latent_cortex.schedules import LayerSchedule, ScheduleLibrary
from core.brain.llm.latent_cortex.semantic_plasticity import (
    build_contrastive_semantic_seeds,
    build_layerwise_trajectory_directions,
)
from core.brain.llm.latent_cortex.teaching_events import (
    build_exact_objective_teaching_event,
)
from core.brain.llm.latent_cortex.telemetry import LatentTelemetry
from core.brain.llm.latent_cortex.test_time_training import (
    MATCHED_LINE_SEARCH_EVALUATIONS,
    build_critic_recalibration_receipt,
    build_matched_compute_receipt,
    build_test_time_training_receipt,
    deterministic_sham_target,
)
from core.brain.llm.latent_cortex.types import (
    ComputeBudget,
    CortexConfig,
    EpisodeReceipt,
    LatentReasoningResult,
)
from core.brain.llm.latent_cortex.value_of_computation import (
    ACTION_TRANSITION_SCHEMA,
    CognitiveStateSignal,
    ValueOfComputationPolicy,
    build_evidence_snapshot,
    transition_reward,
    validate_evidence_snapshot,
)
from core.brain.llm.latent_cortex.verified_best import tensor_sha256
from core.brain.llm.latent_cortex.verifier_gain_search import (
    VERIFIER_GAIN_GRID,
    build_verifier_gain_search_receipt,
)
from core.brain.llm.latent_cortex.workspace import per_position_rms, role_anchor
from core.runtime.errors import record_degradation

# Cognitive-slot sources whose content is RETRIEVED knowledge (already
# epistemically admitted) — eligible for compilation into the fast-weight
# adaptation subspace.
_RETRIEVAL_SLOT_SOURCES = frozenset({"memory", "one_shot_memory", "reference", "world_model"})

logger = logging.getLogger("Aura.LatentCortex.Engine")

_ASSISTANT_ANSWER_BRIDGE = "\nFinal answer:\n"
# v2 demands complete coverage per token spent: compound requests fail the
# product-quality gate when the decode budget is burned on preamble instead
# of the asked-for facets. The cue is generic — it names no specific task.
_ASSISTANT_ANSWER_BRIDGE_V2 = "\nFinal answer (address every part of the request, concisely):\n"
_ASSISTANT_ANSWER_BRIDGE_V3 = (
    "\nFinal answer (do not quote or repeat the request; answer each part "
    "directly and finish the complete response):\n"
)
_ASSISTANT_ANSWER_BRIDGE_V4 = (
    "\nProduce the complete candidate answer now. Be concise, omit preamble, "
    "and end with the exact final-answer contract requested by the user.\n"
)
_BRIDGE_TEXT_BY_POLICY = {
    "assistant_answer_v1": _ASSISTANT_ANSWER_BRIDGE,
    "assistant_answer_v2": _ASSISTANT_ANSWER_BRIDGE_V2,
    "assistant_answer_v3": _ASSISTANT_ANSWER_BRIDGE_V3,
    "assistant_answer_v4": _ASSISTANT_ANSWER_BRIDGE_V4,
}
# Decode discipline: at most this many consecutive pure-newline tokens are
# admitted before newline-family logits are masked for the next sample. Two
# newlines = one blank line — enough for any legitimate paragraph/list break.
_MAX_NEWLINE_RUN = 2
_MAX_TEACHER_TRAJECTORY_TOKENS = 768
_NEWLINE_RESAMPLE_ATTEMPTS = 4
# Sentence grace: when the token limit lands mid-sentence, sampling may
# continue up to this many extra tokens until sentence-final punctuation —
# still entirely model-sampled tokens, charged to the budget, receipted as
# termination "token_limit_sentence_grace". A truncated tail otherwise
# fails the product gate as a terminal fragment (CP110 live evidence).
_SENTENCE_GRACE_TOKENS = 48
_SENTENCE_TERMINALS = (".", "!", "?", ".\n", "!\n", "?\n")


def _normalize_decoded_text(value: Any) -> tuple[str, bool]:
    """Render unsafe decoder controls visibly before text leaves the model boundary."""

    raw = str(value)
    normalized = "".join(
        "\ufffd"
        if (ord(character) < 32 and character not in "\n\r\t") or ord(character) == 127
        else character
        for character in raw
    )
    return normalized, normalized != raw


_ACTION_CONTROL_TEXT: dict[OperationKind, str] = {
    OperationKind.BLIND_RESOLVE: "Derive a candidate directly from the original problem without peer answers.",
    OperationKind.BRANCH: "Advance the private branch strategy without importing another branch state.",
    OperationKind.DECOMPOSE: "Separate the problem into explicit dependencies and subproblems.",
    OperationKind.SEARCH_MEMORY: "Inspect relevant remembered context as evidence, not authority.",
    OperationKind.RETRIEVE_EVIDENCE: "Identify and use the most discriminating available evidence.",
    OperationKind.SIMULATE: "Run a concrete counterfactual simulation and inspect consequences.",
    OperationKind.FALSIFY: "Try to disprove the leading answer with the strongest counterexample.",
    OperationKind.CHECK_ASSUMPTION: "Test the weakest load-bearing assumption before continuing.",
    OperationKind.REGENERATE_FROM_PREFIX: "Preserve valid premises and derive a new continuation.",
    OperationKind.FORMALIZE: "Translate the key relation into explicit formal constraints.",
}

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


class _LatentEpisodeCancelledError(Exception):
    """Cooperative cancellation observed at a checkpoint-safe boundary."""


class _ActionContinuationCapturedError(Exception):
    """Capture-only execution stopped before selecting the first action."""


def _logits_digest(logits) -> str:
    """Stable digest of a logits vector — the causal audit fingerprint."""
    import hashlib

    import mlx.core as mx

    arr = logits.astype(mx.float32)
    mx.eval(arr)
    return hashlib.sha256(memoryview(arr)).hexdigest()


def _contract_admitted_branch_score(
    branch_index: int,
    scores: Mapping[int, float],
    valid_contract_branches: set[int] | None,
) -> float:
    """Exclude a known-invalid terminal candidate before branch selection."""

    if branch_index not in scores:
        raise KeyError("branch score is missing")
    if valid_contract_branches is not None and branch_index not in valid_contract_branches:
        return -math.inf
    score = float(scores[branch_index])
    if not math.isfinite(score):
        raise ValueError("branch score must be finite")
    return score


def _repeats_a_refuted_answer(ratchet: Any, candidate: str) -> bool:
    """Has this episode already refuted this exact answer?

    Compared on the normalised surface, because "391" and "391." are the
    same answer and re-verifying the second is the duplicate work the policy
    exists to remove.
    """
    from core.brain.llm.latent_cortex.commitment_ratchet import (
        ConstraintKind,
        _normalize,
    )

    target = _normalize(candidate)
    if not target:
        return False
    return any(
        tooth.kind is ConstraintKind.EXCLUDES and _normalize(tooth.subject) == target
        for tooth in getattr(ratchet, "teeth", ())
    )


_MAX_LOCAL_REPAIR_GENERATIONS = 3


def _postconditions_lost(baseline: dict[str, Any] | None, adapted: dict[str, Any]) -> list[str]:
    """Postconditions the base function satisfied and the adapted one does not.

    The whole point of a baseline. A canary failing under both is a property
    of the model, not of ΔW, and erasing an update over it would make the
    ladder fire on every episode of a model that was never going to pass.
    Symmetrically, if the base reading is missing, nothing can be attributed
    to ΔW — an unattributable failure is not evidence against it.
    """
    if not adapted.get("evaluated"):
        return []
    if not baseline or not baseline.get("evaluated"):
        return []
    satisfied_on_base = {
        str(item.get("name")) for item in baseline.get("items") or () if item.get("satisfied")
    }
    return sorted(name for name in (adapted.get("failed") or ()) if str(name) in satisfied_on_base)


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
        # Decode discipline state: token→is-pure-newline verdicts (per
        # tokenizer, so per engine) and the last final-decode suppression
        # count for the episode receipt.
        self._newline_token_cache: dict[int, bool] = {}
        self._last_decode_newline_suppressions = 0
        self._last_decode_contract_required = False
        self._last_decode_contract_satisfied = False
        self._last_decode_contract_grace_tokens = 0
        self._last_decode_contract_grace_used_tokens = 0
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
        self.plasticity_site = PLASTICITY_SITE_REGISTRY.resolve(
            self.config.fast_weights.target,
            self.config.fast_weights.layer_placement,
        )
        adapted_layers = self.plasticity_site.layer_indices(
            self.prelude_end,
            self.coda_start,
            max(1, self.config.fast_weights.max_wrapped_layers),
        )
        self.invariant = CheckpointInvariant(
            model,
            model_path,
            tokenizer=tokenizer,
            adapted_layer_indices=adapted_layers,
            adapted_target=self.config.fast_weights.target,
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
            raise _LatentEpisodeCancelledError(str(stage))
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

    def _encode_terminal_instruction(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("terminal language instruction requires a tokenizer")
        rendered = f"\nReasoning disposition:\n{text}\n"
        try:
            encoded = self.tokenizer.encode(rendered, add_special_tokens=False)
        except TypeError:
            encoded = self.tokenizer.encode(rendered)
        tokens = list(encoded)
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("terminal language instruction produced invalid tokens")
        return tokens

    def _terminal_instruction_reserve(self) -> int:
        if self.tokenizer is None:
            return 0
        from core.brain.llm.latent_cortex.terminal_disposition import (
            terminal_instruction_texts,
        )

        return max(
            len(self._encode_terminal_instruction(text)) for text in terminal_instruction_texts()
        )

    @staticmethod
    def _cache_context_tokens(cache: Any, layer_index: int = 0) -> int:
        if not cache or not 0 <= layer_index < len(cache):
            return 0
        item = cache[layer_index]
        offset = getattr(item, "offset", 0) if item is not None else 0
        return max(0, int(offset)) if type(offset) is int else 0

    def _information_receipt(
        self,
        *,
        encoded_tokens: bytes,
        token_count: int,
        context_items: list[dict[str, Any]],
        policy_evidence: dict[str, Any],
        verifier: Callable[[str], float] | None,
        nonparametric_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sources: list[dict[str, Any]] = [
            {
                "source_id": "rendered_model_input",
                "kind": "model_input_tokens",
                "content_sha256": hashlib.sha256(encoded_tokens).hexdigest(),
                "byte_count": len(encoded_tokens),
                "token_count": token_count,
            }
        ]
        for index, item in enumerate(context_items):
            text = str(item["text"])
            payload = text.encode("utf-8")
            context_tokens = 0
            if self.tokenizer is not None:
                try:
                    context_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
                except TypeError:
                    context_tokens = len(self.tokenizer.encode(text))
            sources.append(
                {
                    "source_id": f"cognitive_context:{index}:{item['source']}",
                    "kind": "typed_cognitive_context",
                    "content_sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_count": len(payload),
                    "token_count": context_tokens,
                }
            )
        policy_payload = json.dumps(policy_evidence, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        sources.append(
            {
                "source_id": "value_controller_evidence",
                "kind": "controller_evidence",
                "content_sha256": hashlib.sha256(policy_payload).hexdigest(),
                "byte_count": len(policy_payload),
                "token_count": 0,
            }
        )

        nonparametric_identity = dict(nonparametric_identity or {})
        if nonparametric_identity:
            sources.append(
                {
                    "source_id": "one_shot_nonparametric_memory",
                    "kind": "local_nonparametric_memory_store",
                    "content_sha256": nonparametric_identity["content_sha256"],
                    "byte_count": int(nonparametric_identity["source_bytes"]),
                    "token_count": 0,
                }
            )

        verifier_type = type(verifier) if verifier is not None else None
        verifier_identity = (
            verifier if verifier is not None and inspect.isroutine(verifier) else verifier_type
        )
        verifier_source_sha256 = ""
        if verifier_identity is not None:
            try:
                source_path = inspect.getsourcefile(verifier_identity)
            except (OSError, TypeError):
                source_path = None
            if source_path:
                try:
                    verifier_source_sha256 = hashlib.sha256(
                        Path(source_path).read_bytes()
                    ).hexdigest()
                except OSError:
                    verifier_source_sha256 = "unreadable"
        tokenizer_type = type(self.tokenizer) if self.tokenizer is not None else None
        policies = {
            "tokenizer": policy_sha256(
                {
                    "module": tokenizer_type.__module__ if tokenizer_type else "none",
                    "qualname": tokenizer_type.__qualname__ if tokenizer_type else "none",
                    "chat_template_sha256": hashlib.sha256(
                        str(getattr(self.tokenizer, "chat_template", "")).encode("utf-8")
                    ).hexdigest(),
                }
            ),
            "verifier": policy_sha256(
                {
                    "module": (
                        getattr(verifier_identity, "__module__", "none")
                        if verifier_identity is not None
                        else "none"
                    ),
                    "qualname": (
                        getattr(verifier_identity, "__qualname__", "none")
                        if verifier_identity is not None
                        else "none"
                    ),
                    "source_sha256": verifier_source_sha256 or "none",
                }
            ),
            "tools": policy_sha256({"policy": "no_external_tools_inside_rlc_v1"}),
            "nonparametric_memory": policy_sha256(
                {
                    "policy": "context_only_prompt_tail_recall_v1",
                    "active_source_receipt_sha256": nonparametric_identity.get(
                        "receipt_sha256", "none"
                    ),
                }
            ),
        }
        return build_information_receipt(sources=sources, policies=policies)

    @staticmethod
    def _meter_verifier(
        verifier: Callable[[str], Any] | None,
        budget: ComputeBudget,
    ) -> Callable[[str], Any] | None:
        if verifier is None:
            return None

        def charge(callback: Callable[[str], Any], text: str) -> Any:
            rendered = str(text)
            result = callback(rendered)
            budget.charge_verifier(
                "task_verifier",
                input_bytes=len(rendered.encode("utf-8")),
                output_bytes=len(repr(result).encode("ascii", errors="replace")),
                host_scalar_ops=max(1, len(rendered)),
            )
            return result

        def metered(text: str) -> Any:
            return charge(verifier, text)

        bounded = getattr(verifier, "observe_with_bounds", None)
        if callable(bounded):
            metered.observe_with_bounds = lambda text: charge(bounded, text)
        research_oracle = getattr(verifier, "research_oracle_assessment", None)
        if callable(research_oracle):
            # This method exists only on the benchmark's hidden-answer scorer.
            # Preserve it through metering so the diagnostic oracle arm can
            # distinguish generation failure from selection failure. The
            # returned receipt remains explicitly research-only.
            metered.research_oracle_assessment = lambda text: charge(
                research_oracle,
                text,
            )
        latent_state_score = getattr(verifier, "latent_state_score", None)
        if callable(latent_state_score):
            # Raw latent probes precede task-shape repair. This candidate-local
            # semantic score can guide transient state search, but never branch
            # admission, answer replacement, or serving.
            metered.latent_state_score = lambda text: charge(
                latent_state_score,
                text,
            )
        return metered

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

    def _decode_public_text(
        self,
        tokens,
        *,
        receipt: EpisodeReceipt | None = None,
    ) -> str:
        """Decode model tokens through the auditable public-text boundary."""

        if self.tokenizer is None:
            return ""
        rendered, normalized = _normalize_decoded_text(self.tokenizer.decode(list(tokens)))
        if normalized and receipt is not None:
            receipt.flag("decoded_text_control_characters_normalized")
        return rendered

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

        from core.brain.llm.latent_cortex.recurrence_adapter import coda_adapter_scope

        if not tokens:
            raise ValueError("decode bridge tokens cannot be empty")
        if not budget.can_afford(len(tokens), self.n_layers):
            raise RuntimeError("compute budget cannot admit decode bridge")
        context_tokens = self._cache_context_tokens(cache)
        budget.charge(
            tokens=len(tokens),
            layers=self.n_layers,
            operation="decode_bridge",
            attention_pairs=(
                triangular_attention_pairs(len(tokens), context_tokens=context_tokens)
                * self.n_layers
            ),
            output_head_tokens=1,
        )
        inner = self.model.model
        h = inner.embed_tokens(mx.array([tokens]))
        mask = create_attention_mask(h, cache)
        with coda_adapter_scope():
            for index, layer in enumerate(inner.layers):
                h = layer(h, mask, cache[index])
        logits = self._logits(h)[0, -1]
        mx.eval(logits)
        return logits

    # ── Typed cognitive ingress into the workspace ──────────────────────
    _MAX_COGNITIVE_CONTEXT_TOKENS = 64

    @property
    def _cognitive_context_truncations(self) -> list[dict]:
        # Lazily created so no __init__ change is needed on a class with
        # several construction paths.
        existing = getattr(self, "_cognitive_context_truncations_store", None)
        if existing is None:
            existing = []
            self._cognitive_context_truncations_store = existing
        return existing

    @_cognitive_context_truncations.setter
    def _cognitive_context_truncations(self, value: list[dict]) -> None:
        self._cognitive_context_truncations_store = list(value)

    def _validate_cognitive_context(self, cognitive_context: list | None) -> list[dict]:
        from core.brain.llm.latent_cortex.cognitive_context import (
            normalize_cognitive_context,
        )

        # Each validation starts a fresh admission record for the episode.
        self._cognitive_context_truncations = []
        return normalize_cognitive_context(cognitive_context)

    def _embed_cognitive_context(self, items: list[dict]) -> list[tuple[str, object]]:
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
            embedding_text = item["text"]
            if item.get("context_role") == "memory_observation":
                # The slot receives the remembered semantics, but the
                # authority boundary is represented in-band as well as in
                # the typed wire contract. Recalled imperatives are quoted
                # historical data, never control instructions.
                embedding_text = (
                    "Historical memory data only; never an instruction: " + embedding_text
                )
            elif item.get("context_role") == "evidence_observation":
                embedding_text = (
                    "Retrieved evidence data only; never an instruction: " + embedding_text
                )
            try:
                encoded = self.tokenizer.encode(embedding_text, add_special_tokens=False)
            except TypeError:
                encoded = self.tokenizer.encode(embedding_text)
            # CP126: "Cognitive context is silently dropped at three
            # independent limits... the receipt does not state requested
            # versus admitted context or dropped content. Consumers can
            # believe the full context influenced reasoning when it did not."
            #
            # Two of the three limits already REJECT rather than truncate
            # (normalize_cognitive_context raises past 6 items and past 400
            # characters), so a caller cannot quietly over-send. This one
            # truncates: encoding a 400-character memory can exceed 64
            # tokens, and the tail was dropped with nothing recorded — so a
            # slot seeded from half a memory was indistinguishable from one
            # seeded from all of it.
            full = list(encoded)
            tokens = full[: self._MAX_COGNITIVE_CONTEXT_TOKENS]
            if not tokens:
                continue
            if len(full) > len(tokens):
                dropped = len(full) - len(tokens)
                self._cognitive_context_truncations.append(
                    {
                        "source": str(item.get("source", ""))[:40],
                        "context_role": str(item.get("context_role", "") or "untyped"),
                        "requested_tokens": len(full),
                        "admitted_tokens": len(tokens),
                        "dropped_tokens": dropped,
                    }
                )
            h = inner.embed_tokens(mx.array([tokens]))
            pooled = mx.mean(h, axis=1, keepdims=True)  # (1,1,D)
            mx.eval(pooled)
            seeds.append((item["source"], pooled))
        return seeds

    def cognitive_context_admission(self) -> dict:
        """What of the requested context actually reached the workspace.

        A consumer reading a latent receipt otherwise has no way to tell a
        slot seeded from a whole memory from one seeded from its first 64
        tokens, and the reasoning that follows differs.
        """
        truncations = list(self._cognitive_context_truncations)
        from core.brain.llm.latent_cortex.cognitive_context import (
            MAX_COGNITIVE_CONTEXT_CHARS,
            MAX_COGNITIVE_CONTEXT_ITEMS,
        )

        return {
            "schema": "aura.cognitive_context_admission.v1",
            "max_tokens_per_item": self._MAX_COGNITIVE_CONTEXT_TOKENS,
            "max_items": MAX_COGNITIVE_CONTEXT_ITEMS,
            "max_chars": MAX_COGNITIVE_CONTEXT_CHARS,
            "truncated_items": len(truncations),
            "dropped_tokens_total": sum(t["dropped_tokens"] for t in truncations),
            "complete": not truncations,
            "truncations": truncations,
        }

    def reset_cognitive_context_admission(self) -> None:
        self._cognitive_context_truncations = []

    def _embed_action_controls(self) -> dict[OperationKind, object]:
        """Embed each neural cognitive operator once per episode."""

        import mlx.core as mx

        rows = self._embed_cognitive_context(
            [
                {"source": action.value, "text": instruction}
                for action, instruction in _ACTION_CONTROL_TEXT.items()
            ]
        )
        controls = {OperationKind(source): vector for source, vector in rows}
        dim = int(self.model.model.embed_tokens.weight.shape[-1])
        embedding_rms = mx.mean(per_position_rms(self.model.model.embed_tokens.weight))
        for action in _ACTION_CONTROL_TEXT:
            if action in controls:
                continue
            direction = role_anchor(f"cognitive-action:{action.value}", dim)
            controls[action] = direction.reshape(1, 1, dim) * embedding_rms
            mx.eval(controls[action])
        return controls

    @staticmethod
    def _mean_latest_residual(ensemble: BranchEnsemble) -> float:
        values = [
            branch.halting.residual_trail[-1]
            for branch in ensemble.branches
            if branch.halting.residual_trail
        ]
        if not values:
            return 1.0
        return max(0.0, min(1.0, sum(values) / len(values)))

    @staticmethod
    def _ensemble_snapshot_boundaries(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        """Describe a private ensemble snapshot using tensor-free commitments."""

        branches = snapshot.get("branches") if isinstance(snapshot, dict) else None
        if not isinstance(branches, dict) or not branches:
            raise ValueError("ensemble snapshot has no branch inventory")
        rows = []
        for index in sorted(branches):
            branch = branches[index]
            if not isinstance(branch, dict):
                raise ValueError("ensemble snapshot branch is invalid")
            state_sha256 = tensor_sha256(branch["z"])
            kv_sha256 = str(branch["kv_boundary_sha256"])
            if len(kv_sha256) != 64:
                raise ValueError("ensemble snapshot KV identity is invalid")
            rows.append(
                {
                    "index": int(index),
                    "state_sha256": state_sha256,
                    "kv_boundary_sha256": kv_sha256,
                    "steps": int(branch["steps"]),
                    "operator": str(branch["operator"]),
                    "halted": bool(branch["halted"]),
                }
            )
        return rows

    @staticmethod
    def _canonical_sha256(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _ensemble_snapshot_identity(cls, snapshot: dict[str, Any]) -> tuple[str, str]:
        """Commit a complete private ensemble snapshot without disclosing tensors."""

        from core.brain.llm.latent_cortex.latent_tree_search import (
            ensemble_identity_from_boundaries,
        )

        return ensemble_identity_from_boundaries(cls._ensemble_snapshot_boundaries(snapshot))

    @classmethod
    def _action_intervention_state_components(
        cls,
        *,
        ensemble: BranchEnsemble,
        budget: ComputeBudget,
        episode_context_items: list[dict[str, Any]],
        action_policy_evidence: dict[str, Any],
        state_signal: CognitiveStateSignal,
        active_action_executors: tuple[OperationKind, ...],
        action_intervention: dict[str, Any],
    ) -> dict[str, str]:
        """Commit every action-relevant resident state surface.

        The worker directly measures six components. Durable host state and
        the externally established RNG root remain runner-owned commitments
        from the signed pre-execution capture; the public receipt labels that
        ownership instead of pretending the worker observed those surfaces.
        """

        def commitment_value(value: Any) -> Any:
            if value is None or isinstance(value, (bool, int, str)):
                return value
            if isinstance(value, float):
                if not math.isfinite(value):
                    return str(value)
                return value
            if isinstance(value, dict):
                return {
                    str(key): commitment_value(item)
                    for key, item in sorted(value.items(), key=lambda row: str(row[0]))
                }
            if isinstance(value, (list, tuple)):
                return [commitment_value(item) for item in value]
            if isinstance(value, (set, frozenset)):
                normalized = [commitment_value(item) for item in value]
                return sorted(
                    normalized,
                    key=lambda item: json.dumps(
                        item,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                )
            enum_value = getattr(value, "value", None)
            if isinstance(enum_value, (bool, int, float, str)):
                return commitment_value(enum_value)
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                return {
                    "tensor_sha256": tensor_sha256(value),
                    "shape": [int(item) for item in value.shape],
                    "dtype": str(value.dtype),
                }
            raise ValueError(
                f"action intervention state contains an unsupported value {type(value).__name__}"
            )

        snapshot = ensemble.snapshot_ensemble_runtime()
        latent_rows = [
            {
                "index": branch.index,
                "latent": commitment_value(branch.z),
                "anchor": commitment_value(branch.anchor),
                "workspace": commitment_value(branch.workspace.snapshot()),
            }
            for branch in ensemble.branches
        ]
        branch_rows = []
        for branch in ensemble.branches:
            branch_snapshot = dict(snapshot["branches"][branch.index])
            branch_snapshot.pop("z", None)
            branch_rows.append(
                {
                    "index": branch.index,
                    "runtime_snapshot": commitment_value(branch_snapshot),
                    "savepoint": commitment_value(branch.savepoint),
                    "savepoint_steps": branch.savepoint_steps,
                    "savepoint_kv_boundary_sha256": (branch.savepoint_kv_boundary_sha256),
                    "seed_sha256": branch.seed_sha256,
                    "rng_stream_sha256": branch.rng_stream_sha256,
                    "candidate_sha256": branch.candidate_sha256,
                    "candidate_step": branch.candidate_step,
                    "update_gate": (
                        {
                            "mode": branch.update_gate.mode,
                            "head_sha256": branch.update_gate.head_sha256,
                        }
                        if branch.update_gate is not None
                        else None
                    ),
                    "verified_best_state": commitment_value(branch.verified_best_state),
                    "verified_best_step": branch.verified_best_step,
                    "verified_best_state_sha256": (branch.verified_best_state_sha256),
                    "verified_best_observation": commitment_value(branch.verified_best_observation),
                    "verified_best_trace": commitment_value(branch.verified_best_trace),
                    "verified_finalization": commitment_value(branch.verified_finalization),
                    "uncertainty_runtime": (
                        {
                            "mode": str(getattr(branch.uncertainty_runtime, "mode", "")),
                            "head_sha256": str(
                                getattr(
                                    branch.uncertainty_runtime,
                                    "head_sha256",
                                    "",
                                )
                            ),
                        }
                        if branch.uncertainty_runtime is not None
                        else None
                    ),
                    "mistake_locator_runtime": (
                        {
                            "mode": str(
                                getattr(
                                    branch.mistake_locator_runtime,
                                    "mode",
                                    "",
                                )
                            ),
                            "head_sha256": str(
                                getattr(
                                    branch.mistake_locator_runtime,
                                    "head_sha256",
                                    "",
                                )
                            ),
                        }
                        if branch.mistake_locator_runtime is not None
                        else None
                    ),
                    "recurrent_grounding_trace": commitment_value(branch.recurrent_grounding_trace),
                    "loop_stability_trace": commitment_value(branch.loop_stability_trace),
                    "update_acceptance_trace": commitment_value(branch.update_acceptance_trace),
                    "uncertainty_trace": commitment_value(branch.uncertainty_trace),
                    "mistake_locator_trace": commitment_value(branch.mistake_locator_trace),
                    "reflector_trace": commitment_value(branch.reflector_trace),
                }
            )
        ensemble_state = {
            name: commitment_value(value) for name, value in snapshot.items() if name != "branches"
        }
        ensemble_state.update(
            {
                "context_sha256": str(ensemble._context_sha256),
                "configured_role_lesion": bool(ensemble._configured_role_lesion),
                "seed_alias_free": bool(ensemble._seed_alias_free),
                "seed_states_unique": bool(ensemble._seed_states_unique),
                "rng_streams_unique": bool(ensemble._rng_streams_unique),
                "support_weights": commitment_value(ensemble._support_weights),
                "branches": branch_rows,
                "budget": {
                    "max_layer_apps": budget.max_layer_apps,
                    "spent_layer_apps": budget.spent_layer_apps,
                    "remaining_layer_apps": budget.remaining_layer_apps,
                    "resource_accounting": budget.resource_ledger.to_receipt(),
                    "information_accounting": budget.information_receipt,
                },
            }
        )
        memory_items = [
            item
            for item in episode_context_items
            if item.get("context_role") == "memory_observation"
            or str(item.get("source") or "") in {"memory", "one_shot_memory"}
        ]
        evidence_items = [
            item
            for item in episode_context_items
            if item not in memory_items
            and (
                item.get("context_role") == "evidence_observation"
                or str(item.get("source") or "") in {"reference", "world_model"}
                or str(item.get("source") or "").startswith(("evidence", "tool_observation"))
            )
        ]
        expected = action_intervention["authority_payload"]["starting_state_components"]
        components = {
            "latent_slots_sha256": cls._canonical_sha256(latent_rows),
            "branch_state_sha256": cls._canonical_sha256(ensemble_state),
            "kv_cache_sha256": cls._ensemble_snapshot_identity(snapshot)[1],
            "evidence_state_sha256": cls._canonical_sha256(
                {
                    "action_policy_evidence": action_policy_evidence,
                    "context_items": evidence_items,
                }
            ),
            "memory_state_sha256": cls._canonical_sha256(memory_items),
            "public_action_state_sha256": cls._canonical_sha256(
                {
                    "state_signal": state_signal.to_dict(),
                    "active_action_executors": [action.value for action in active_action_executors],
                }
            ),
            "durable_state_sha256": expected["durable_state_sha256"],
            "rng_state_sha256": expected["rng_state_sha256"],
        }
        return {name: str(components[name]) for name in sorted(components)}

    @staticmethod
    def _policy_uncertainty(bucket: str) -> float:
        if "|u:high" in bucket:
            return 0.8
        if "|u:low" in bucket:
            return 0.2
        return 0.5

    @staticmethod
    def _action_executors(
        *,
        has_controls: bool,
        has_verifier: bool,
        can_execute: bool = False,
    ) -> tuple[OperationKind, ...]:
        executors = {
            OperationKind.BLIND_RESOLVE,
            OperationKind.BRANCH,
            OperationKind.COMPARE,
            OperationKind.BACKTRACK,
            OperationKind.COMPRESS_STATE,
            OperationKind.ANSWER,
            OperationKind.ABSTAIN,
        }
        if has_controls:
            executors.update(
                {
                    OperationKind.DECOMPOSE,
                    OperationKind.SEARCH_MEMORY,
                    OperationKind.RETRIEVE_EVIDENCE,
                    OperationKind.SIMULATE,
                    OperationKind.REGENERATE_FROM_PREFIX,
                    OperationKind.FORMALIZE,
                }
            )
            if has_verifier:
                executors.update({OperationKind.FALSIFY, OperationKind.CHECK_ASSUMPTION})
        if can_execute:
            executors.add(OperationKind.EXECUTE)
        return tuple(action for action in OperationKind if action in executors)

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

    def _fresh_verifier_generation(
        self,
        prompt: str,
        budget: ComputeBudget,
        *,
        max_tokens: int,
        reserve_layer_apps: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        sample_seed: int | None = None,
        final_answer_contract: bool = True,
    ) -> dict[str, Any]:
        """Generate one fresh-context witness without importing solver KV state.

        Most verifier witnesses use the public ``FINAL_ANSWER`` contract. A
        local repair instead continues from ``REPLACEMENT_SUFFIX:`` and is
        parsed by its own strict contract, so forcing ``FINAL_ANSWER`` there
        makes successful termination impossible.
        """

        if sample_seed is None:
            from core.brain.llm.latent_cortex.generative_verifier import (
                FRESH_CONTEXT_SCHEMA as CONTEXT_SCHEMA,
            )
        else:
            from core.brain.llm.latent_cortex.prefix_stability import (
                PREFIX_STABILITY_CONTEXT_SCHEMA as CONTEXT_SCHEMA,
            )

        tokens = self._encode(prompt, None, None)
        # Internal verifier prompts always require the strict FINAL_ANSWER
        # contract, independently of the user-facing decode profile.
        contract_extension = min(64, max_tokens) if final_answer_contract else 0
        required = (len(tokens) + max_tokens + contract_extension) * self.n_layers
        if required + reserve_layer_apps > budget.remaining_layer_apps:
            raise RuntimeError("fresh_verifier_budget_unavailable")
        cache = self._fresh_cache()
        initial_offsets = [self._cache_context_tokens(cache, index) for index in range(len(cache))]
        if not initial_offsets or any(initial_offsets):
            raise RuntimeError("fresh_verifier_cache_not_empty")

        sentinel = object()
        saved_prefill = getattr(self, "_last_prefill_hidden", sentinel)
        state_names = (
            "_last_decode_newline_suppressions",
            "_last_decode_contract_required",
            "_last_decode_contract_satisfied",
            "_last_decode_contract_grace_tokens",
            "_last_decode_contract_grace_used_tokens",
        )
        saved_state = {name: getattr(self, name) for name in state_names}
        try:
            _embeddings, logits = self._prefill(tokens, cache, budget)
            generated, termination = self._decode(
                cache,
                budget,
                logits,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                sample_seed=sample_seed,
                final_answer_contract=final_answer_contract,
                sentence_grace_tokens=0,
                contract_grace_tokens=contract_extension,
            )
            text = self.tokenizer.decode(generated)
            final_offsets = [
                self._cache_context_tokens(cache, index) for index in range(len(cache))
            ]
        finally:
            if saved_prefill is sentinel:
                if hasattr(self, "_last_prefill_hidden"):
                    delattr(self, "_last_prefill_hidden")
            else:
                self._last_prefill_hidden = saved_prefill
            for name, value in saved_state.items():
                setattr(self, name, value)
        return {
            "text": text,
            "context": {
                "schema": CONTEXT_SCHEMA,
                "prompt_token_count": len(tokens),
                "generated_token_count": len(generated),
                "termination": termination,
                "initial_cache_offsets": initial_offsets,
                "final_cache_offsets": final_offsets,
                "all_initial_offsets_zero": True,
                "solver_context_imported": False,
                "parameter_relation": "shared_resident_checkpoint",
                **(
                    {
                        "sample_seed": sample_seed,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                    }
                    if sample_seed is not None
                    else {}
                ),
            },
        }

    def _logits(
        self,
        h,
        *,
        budget: ComputeBudget | None = None,
        operation: str = "output_head",
    ):
        inner = self.model.model
        if budget is not None:
            budget.resource_ledger.charge(
                operation,
                output_head_tokens=int(h.shape[1]),
            )
        h = inner.norm(h)
        self._last_output_hidden = h[:, -1:, :]
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(h)
        else:
            logits = inner.embed_tokens.as_linear(h)
        output_memory = getattr(self, "_active_output_memory", None)
        semantic_adapter = getattr(self, "_active_semantic_output_adapter", None)
        if output_memory is not None and semantic_adapter is not None:
            raise RuntimeError("multiple output-boundary plasticity mechanisms are active")
        if output_memory is not None:
            logits = output_memory.apply(h, logits)
        if semantic_adapter is not None:
            logits = semantic_adapter.apply(h, logits)
        return logits

    def _prefill(self, tokens: list[int], cache, budget: ComputeBudget):
        """Standard full-stack prefill. Returns (embeddings, last-position logits)."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        inner = self.model.model
        if not tokens:
            raise ValueError("latent episode requires at least one input token")
        if not budget.can_afford(len(tokens), self.n_layers):
            raise RuntimeError("compute budget cannot afford prompt prefill")
        context_tokens = self._cache_context_tokens(cache)
        budget.charge(
            tokens=len(tokens),
            layers=self.n_layers,
            operation="prompt_prefill",
            attention_pairs=(
                triangular_attention_pairs(len(tokens), context_tokens=context_tokens)
                * self.n_layers
            ),
            output_head_tokens=1,
        )
        arr = mx.array([tokens])
        h = inner.embed_tokens(arr)
        embeddings = h
        mask = create_attention_mask(h, cache)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[i])
        logits = self._logits(h[:, -1:, :])[0, -1]
        self._last_prefill_hidden = h[:, -1:, :]
        mx.eval(logits, self._last_prefill_hidden)
        return embeddings, logits

    def _sample(
        self,
        logits,
        temperature: float,
        top_p: float = 1.0,
        *,
        budget: ComputeBudget | None = None,
        random_key: Any | None = None,
    ) -> int:
        import mlx.core as mx

        if budget is not None:
            vocab = int(logits.shape[-1])
            multiplier = 8 if temperature > 0.0 and top_p < 1.0 else 3
            budget.charge_tensor_work(
                "decode_sampling",
                element_reads=vocab,
                element_writes=vocab if temperature > 0.0 else 1,
                host_scalar_ops=vocab * multiplier,
            )
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
                selected = int(
                    mx.random.categorical(filtered_logits)
                    if random_key is None
                    else mx.random.categorical(filtered_logits, key=random_key)
                )
                return int(sorted_indices[selected])
            return int(
                mx.random.categorical(scaled)
                if random_key is None
                else mx.random.categorical(scaled, key=random_key)
            )
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
        sentence_grace_tokens: int | None = None,
        contract_grace_tokens: int | None = None,
        token_logprobs_out: list[float] | None = None,
        force_exact_tokens: bool = False,
        external_step_logits: Callable[[int], Any] | None = None,
        external_step_lanes: int = 1,
        sample_seed: int | None = None,
        final_answer_contract: bool | None = None,
        coda_adapter_active: bool = False,
    ) -> tuple[list[int], str]:
        """Minimal sampler: first token from ``initial_logits`` (the logits of
        the last persisted position — prompt tail or final thought slot), then
        autoregressive continuation over the populated cache.

        ``wall_reserve_s`` stops decoding while that much wall clock still
        remains — the engine reserves cleanup time when fast weights are
        attached, so a long answer degrades to token truncation instead of
        endangering the erase proof. ``sentence_grace_tokens=0`` is the hard
        cap used by internal verifier previews; final answers retain the
        product-facing sentence-completion grace by default. Contract tasks
        use a separate bounded window and suppress EOS until completion or
        exhaustion."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        inner = self.model.model
        eos = self._eos_ids()
        limit = max_tokens if max_tokens is not None else self.config.decode_max_tokens
        temp = temperature if temperature is not None else self.config.decode_temperature
        nucleus = top_p if top_p is not None else self.config.decode_top_p
        grace_tokens = (
            _SENTENCE_GRACE_TOKENS if sentence_grace_tokens is None else sentence_grace_tokens
        )
        if type(grace_tokens) is not int or grace_tokens < 0:
            raise ValueError("sentence_grace_tokens must be a non-negative integer")
        contract_grace = (
            self.config.decode_contract_grace_tokens
            if contract_grace_tokens is None
            else contract_grace_tokens
        )
        if type(contract_grace) is not int or not 0 <= contract_grace <= 4096:
            raise ValueError("contract_grace_tokens must be an integer in [0, 4096]")
        if type(force_exact_tokens) is not bool:
            raise TypeError("force_exact_tokens must be boolean")
        if sample_seed is not None and (
            type(sample_seed) is not int or not 0 <= sample_seed <= 0xFFFFFFFF
        ):
            raise ValueError("sample_seed must be null or an integer inside [0, 2^32-1]")
        if final_answer_contract is not None and type(final_answer_contract) is not bool:
            raise TypeError("final_answer_contract must be boolean or null")
        if type(coda_adapter_active) is not bool:
            raise TypeError("coda_adapter_active must be boolean")
        if (
            type(external_step_lanes) is not int
            or external_step_lanes < 1
            or (external_step_logits is None and external_step_lanes != 1)
        ):
            raise ValueError("external decode lane count is invalid")

        out: list[int] = []
        stochastic_draw = 0
        newline_run = 0
        suppressions = 0
        self._last_decode_newline_suppressions = 0
        contract_required = (
            self.config.decode_contract == "final_answer_v1"
            if final_answer_contract is None
            else final_answer_contract
        )
        if contract_required and self.tokenizer is None:
            raise ValueError("final_answer_v1 decode contract requires a tokenizer")
        contract_satisfied = False
        self._last_decode_contract_required = contract_required
        self._last_decode_contract_satisfied = False
        self._last_decode_contract_grace_tokens = contract_grace if contract_required else 0
        self._last_decode_contract_grace_used_tokens = 0
        if budget.exhausted:
            return out, "budget_exhausted"

        penalty = float(self.config.decode_repetition_penalty)
        window = max(1, int(self.config.decode_repetition_window))
        min_tokens = max(0, int(self.config.decode_min_tokens))

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

        def sample_logprob(logits, token: int) -> float:
            if temp <= 0.0:
                return 0.0
            scaled = logits / temp
            if nucleus < 1.0:
                probabilities = mx.softmax(scaled)
                sorted_indices = mx.argsort(-probabilities)
                sorted_probabilities = probabilities[sorted_indices]
                cumulative = mx.cumsum(sorted_probabilities)
                keep = (cumulative - sorted_probabilities) < nucleus
                kept_probabilities = mx.where(
                    keep, sorted_probabilities, mx.zeros_like(sorted_probabilities)
                )
                normalizer = mx.maximum(mx.sum(kept_probabilities), 1e-12)
                selected = mx.sum(
                    mx.where(
                        sorted_indices == int(token),
                        kept_probabilities,
                        mx.zeros_like(kept_probabilities),
                    )
                )
                return float(mx.log(mx.maximum(selected / normalizer, 1e-30)))
            return float(scaled[int(token)] - mx.logsumexp(scaled))

        def sample_disciplined(logits):
            """Sample under the newline-run discipline.

            A run of more than _MAX_NEWLINE_RUN pure-newline tokens is decode
            babble: it wastes answer budget and independently fails the
            product-quality gate (excessive_blank_lines). Masking newline
            logits for the next sample is a sampling CONSTRAINT — the emitted
            text is still entirely the model's own tokens, never edited."""
            nonlocal stochastic_draw, suppressions
            logits = penalize_repeats(logits)
            # EOS floor: below decode_min_tokens, end-of-sequence logits are
            # masked so sampling variance cannot abandon the answer a few
            # tokens in (min-new-tokens, the standard serving constraint).
            if eos and (
                force_exact_tokens
                or len(out) < min_tokens
                or (contract_required and not contract_satisfied)
            ):
                eos_ids = mx.array(sorted(eos))
                gathered = logits[eos_ids]
                logits = logits.at[eos_ids].add(mx.full(gathered.shape, -1e9) - gathered)
            random_key = None
            if sample_seed is not None and temp > 0.0:
                random_key = mx.random.key(
                    (sample_seed + stochastic_draw * 0x9E3779B1) & 0x7FFFFFFF
                )
                stochastic_draw += 1
            token = self._sample(
                logits,
                temp,
                nucleus,
                budget=budget,
                random_key=random_key,
            )
            if self.tokenizer is None or newline_run < _MAX_NEWLINE_RUN:
                return token, sample_logprob(logits, token)
            masked = logits
            for _ in range(_NEWLINE_RESAMPLE_ATTEMPTS):
                if not self._is_pure_newline_token(token):
                    return token, sample_logprob(masked, token)
                suppressions += 1
                masked = mx.where(
                    mx.arange(masked.shape[-1]) == token,
                    mx.full(masked.shape, -1e9),
                    masked,
                )
                random_key = None
                if sample_seed is not None and temp > 0.0:
                    random_key = mx.random.key(
                        (sample_seed + stochastic_draw * 0x9E3779B1) & 0x7FFFFFFF
                    )
                    stochastic_draw += 1
                token = self._sample(
                    masked,
                    temp,
                    nucleus,
                    budget=budget,
                    random_key=random_key,
                )
            return token, sample_logprob(masked, token)

        # Contract-aware termination (CP180): once a single FINAL_ANSWER
        # JSON object completes, more tokens can only break terminality.
        # The full-text check runs only when the newest piece could have
        # closed an object ("}") or on a periodic beat after the marker
        # might exist — text work, never model work.
        def contract_disposition_now():
            if force_exact_tokens or not contract_required:
                return None
            from core.brain.llm.latent_cortex.answer_contract import (
                contract_decode_disposition,
            )

            try:
                text = self.tokenizer.decode(out)
            except (TypeError, ValueError, KeyError, AttributeError):
                return None
            return contract_decode_disposition(text)

        token, token_logprob = sample_disciplined(initial_logits)
        termination = "token_limit"
        decode_started = time.monotonic()
        extension = contract_grace if contract_required else grace_tokens
        for index in range(max(1, int(limit) + extension)):
            if self._cancel_requested(cancel_check):
                raise _LatentEpisodeCancelledError("decode")
            if token in eos:
                termination = "eos"
                break
            out.append(token)
            if token_logprobs_out is not None:
                token_logprobs_out.append(token_logprob)
            contract_disposition = contract_disposition_now()
            contract_satisfied = bool(
                contract_disposition is not None and contract_disposition.value == "complete"
            )
            if contract_satisfied:
                termination = "contract_complete"
                break
            if contract_disposition is not None and contract_disposition.value == "invalid":
                termination = "contract_irrecoverable"
                break
            newline_run = newline_run + 1 if self._is_pure_newline_token(token) else 0
            sentence_done = self.tokenizer is None or self._token_ends_sentence(token)
            if index + 1 >= int(limit):
                if contract_required:
                    if index + 1 >= int(limit) + contract_grace:
                        termination = "token_limit_contract_incomplete"
                        break
                elif sentence_done:
                    termination = (
                        "token_limit" if index + 1 == int(limit) else "token_limit_sentence_grace"
                    )
                    break
                elif index + 1 >= int(limit) + grace_tokens:
                    # Grace exhausted without punctuation: still a fragment,
                    # and the receipt says so honestly.
                    termination = "token_limit"
                    break
            if budget.exhausted:
                termination = "budget_exhausted"
                break
            if wall_reserve_s > 0.0:
                # Time-aware sentence wind-down: when the MEASURED decode
                # rate says another grace window would eat into the cleanup
                # reserve, finish at the next sentence boundary instead of
                # cutting mid-clause (CP115: the fixed rate estimate ran hot
                # on a cold boot and the reserve guillotined token 330).
                rate_s = max(0.02, (time.monotonic() - decode_started) / max(1, len(out)))
                winding_down = budget.remaining_wall_s < wall_reserve_s + extension * rate_s
                if winding_down and sentence_done:
                    termination = "wall_reserve_sentence_grace"
                    break
                if budget.remaining_wall_s < wall_reserve_s:
                    termination = "wall_reserve"
                    break
            if not budget.can_afford(external_step_lanes, self.n_layers):
                termination = "budget_unaffordable"
                break
            if external_step_logits is None:
                context_tokens = self._cache_context_tokens(cache)
                budget.charge(
                    tokens=1,
                    layers=self.n_layers,
                    operation="autoregressive_decode",
                    attention_pairs=max(1, context_tokens + 1) * self.n_layers,
                    output_head_tokens=1,
                )
                h = inner.embed_tokens(mx.array([[token]]))
                mask = create_attention_mask(h, cache)
                if coda_adapter_active:
                    from core.brain.llm.latent_cortex.recurrence_adapter import (
                        coda_adapter_scope,
                    )

                    scope = coda_adapter_scope()
                else:
                    scope = nullcontext()
                with scope:
                    for i, layer in enumerate(inner.layers):
                        h = layer(h, mask, cache[i])
                logits = self._logits(h)[0, -1]
            else:
                logits = external_step_logits(token)
            token, token_logprob = sample_disciplined(logits)
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
        self._last_decode_contract_satisfied = contract_satisfied
        self._last_decode_contract_grace_used_tokens = max(
            0,
            len(out) - int(limit),
        )
        return out, termination

    # ── Probe decoding for branch selection / verifier loops ────────────
    def _decode_probe(
        self,
        branch: BranchState,
        cache,
        runner: WindowRunner,
        budget: ComputeBudget,
        *,
        max_tokens: int | None = None,
        bridge_tokens: list[int] | None = None,
        use_cache: bool = True,
        force_exact_tokens: bool = False,
    ) -> list[int]:
        """Decode a short probe from a branch WITHOUT disturbing the caches.

        Full-cache snapshot → persist this branch's slots → decode → restore.
        This is what lets a verifier score every branch before exactly one
        winner's state is committed. Probes are memoized per episode on the
        exact latent state: an unchanged (seed, z, bridge) triple under an
        unchanged model function decodes once and costs nothing after that.
        """
        from core.brain.llm.recurrent_depth import (
            _restore_recurrent_caches,
            _snapshot_recurrent_caches,
        )

        probe_tokens = self.config.verifier_probe_max_tokens if max_tokens is None else max_tokens
        probe_contract = (
            "final_answer_v1"
            if "final_answer_v1"
            in {self.config.decode_contract, self.config.verifier_probe_contract}
            else "none"
        )
        probe_cache = getattr(self, "_episode_probe_cache", None)
        cache_key = None
        if use_cache and probe_cache is not None:
            cache_key = probe_cache.key(
                branch.workspace.seed_z,
                branch.z,
                list(bridge_tokens or []),
                probe_tokens,
                probe_contract,
            )
            memoized = probe_cache.get(cache_key)
            if memoized is not None:
                return memoized
        spent_before = budget.spent_layer_apps
        snaps = _snapshot_recurrent_caches(cache, 0, self.n_layers)
        kv_tree = getattr(self, "_episode_kv_state_tree", None)
        kv_transaction = None
        if kv_tree is not None:
            kv_transaction = kv_tree.begin_speculation(
                cache,
                start=0,
                end=self.n_layers,
                purpose="verifier_probe",
                branch_index=branch.index,
                parent_sha256=(branch.kv_boundary_sha256 or kv_tree.root_sha256),
            )
        execution_failed = True
        try:
            slot_logits = self._persist_branch(branch, cache, runner, budget)
            if bridge_tokens:
                slot_logits = self._apply_decode_bridge(
                    cache,
                    budget,
                    bridge_tokens,
                )
            decoded = self._decode(
                cache,
                budget,
                slot_logits,
                max_tokens=probe_tokens,
                temperature=0.0,
                sentence_grace_tokens=0,
                contract_grace_tokens=0,
                force_exact_tokens=force_exact_tokens,
                final_answer_contract=probe_contract == "final_answer_v1",
                coda_adapter_active=True,
            )[0]
            execution_failed = False
        finally:
            if kv_transaction is not None:
                kv_transaction.observe_mutation(
                    cache,
                    execution_failed=execution_failed,
                )
                kv_transaction.restore_parent(cache)
            else:
                _restore_recurrent_caches(cache, 0, self.n_layers, snaps)
            if kv_transaction is not None:
                kv_transaction.reject_after_restore(cache)
        if use_cache and probe_cache is not None and cache_key is not None:
            probe_cache.put(
                cache_key,
                decoded,
                budget.spent_layer_apps - spent_before,
            )
        return decoded

    def _counterfactual_probe_evaluator(
        self,
        *,
        branch,
        cache,
        runner,
        budget: ComputeBudget,
        bridge_tokens: list[int],
        verifier,
    ):
        """Build one fixed-compute evaluator that always restores its branch."""

        from core.brain.llm.latent_cortex.counterfactual_probe import (
            CounterfactualProbeResult,
        )
        from core.brain.llm.latent_cortex.verified_best import (
            VerifierObservation,
            validate_observation,
        )

        if verifier is None or self.tokenizer is None:
            return None
        baseline_state = branch.z

        def evaluate(
            _label: str,
            candidate_state,
            _replicate: int,
        ) -> CounterfactualProbeResult:
            import mlx.core as mx

            spent_before = budget.spent_layer_apps
            try:
                projected = mx.array(candidate_state)
                projected = branch.workspace.restore_context_evidence(projected)
                branch.z = projected
                branch.workspace.update(projected)
                probe = self._decode_probe(
                    branch,
                    cache,
                    runner,
                    budget,
                    bridge_tokens=bridge_tokens,
                    use_cache=False,
                    force_exact_tokens=True,
                )
                rendered = self.tokenizer.decode(probe)
                rendered_bytes = rendered.encode("utf-8")
                verifier_input_bytes = max(
                    1024,
                    int(self.config.verifier_probe_max_tokens) * 64,
                )
                if len(rendered_bytes) > verifier_input_bytes:
                    raise ValueError("counterfactual verifier input exceeds its fixed byte budget")
                rendered += " " * (verifier_input_bytes - len(rendered_bytes))
                bounded = getattr(verifier, "observe_with_bounds", None)
                raw_observation = bounded(rendered) if callable(bounded) else verifier(rendered)
                if isinstance(raw_observation, dict) and "observation_sha256" in raw_observation:
                    observation = validate_observation(raw_observation)
                else:
                    observation = VerifierObservation.from_value(raw_observation).to_dict()
                encoded_probe = json.dumps(
                    probe,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("ascii")
                return CounterfactualProbeResult(
                    probe_tokens_sha256=hashlib.sha256(encoded_probe).hexdigest(),
                    probe_token_count=len(probe),
                    observation=observation,
                    layer_apps=budget.spent_layer_apps - spent_before,
                )
            finally:
                branch.z = baseline_state
                branch.workspace.update(baseline_state)

        return evaluate

    @staticmethod
    def _heterogeneous_mix_logits(
        old_logits,
        new_logits,
        *,
        policy: str,
        fusion_weight: float,
        budget: ComputeBudget,
    ):
        """Return one policy distribution and measured old/new JS divergence."""

        import mlx.core as mx

        if policy not in {
            "select_old",
            "select_new",
            "probability_fusion",
        }:
            raise ValueError("heterogeneous decode policy is invalid")
        if not math.isfinite(float(fusion_weight)) or not 0.0 <= float(fusion_weight) <= 1.0:
            raise ValueError("heterogeneous fusion weight is invalid")
        old = old_logits.astype(mx.float32)
        new = new_logits.astype(mx.float32)
        if old.shape != new.shape or len(old.shape) != 1:
            raise ValueError("heterogeneous lane logits shape differs")
        old_log_probability = old - mx.logsumexp(old)
        new_log_probability = new - mx.logsumexp(new)
        old_probability = mx.exp(old_log_probability)
        new_probability = mx.exp(new_log_probability)
        midpoint = 0.5 * (old_probability + new_probability)
        log_midpoint = mx.log(mx.maximum(midpoint, 1e-30))
        js_nats = 0.5 * (
            mx.sum(old_probability * (old_log_probability - log_midpoint))
            + mx.sum(new_probability * (new_log_probability - log_midpoint))
        )
        js_bits = float(js_nats / math.log(2.0))
        if not math.isfinite(js_bits) or not -1e-7 <= js_bits <= 1.0 + 1e-7:
            raise ValueError("heterogeneous JS divergence is invalid")
        js_bits = min(1.0, max(0.0, js_bits))
        if policy == "select_old":
            policy_logits = old
        elif policy == "select_new":
            policy_logits = new
        else:
            gamma = float(fusion_weight)
            mixed_probability = (1.0 - gamma) * old_probability + gamma * new_probability
            policy_logits = mx.log(mx.maximum(mixed_probability, 1e-30))
        vocab = int(old.shape[-1])
        budget.charge_tensor_work(
            "heterogeneous_distribution_integration",
            element_reads=6 * vocab,
            element_writes=5 * vocab,
            scalar_ops=24 * vocab,
            host_scalar_ops=16,
        )
        mx.eval(policy_logits)
        return policy_logits, js_bits

    def _heterogeneous_dual_lane_decode(
        self,
        *,
        branch,
        cache,
        runner,
        budget: ComputeBudget,
        incumbent_state,
        corrected_state,
        policy: str,
        fusion_weight: float,
        bridge_tokens: list[int],
        max_tokens: int,
        temperature: float,
        force_exact_tokens: bool,
        cancel_check: Callable[[], bool] | None = None,
        progress: Callable[[dict], None] | None = None,
        sentence_grace_tokens: int | None = 0,
        contract_grace_tokens: int | None = 0,
        wall_reserve_s: float = 0.0,
        token_logprobs_out: list[float] | None = None,
        phase_checkpoint: Callable[[str], None] | None = None,
        retain_lanes: bool = False,
    ) -> tuple[list[int], str, dict[str, Any]]:
        """Decode two cache-isolated lanes under one selected distribution."""

        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        from core.brain.llm.latent_cortex.verified_best import tensor_sha256
        from core.brain.llm.recurrent_depth import (
            _restore_recurrent_caches,
            _snapshot_recurrent_caches,
        )

        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("heterogeneous decode token count is invalid")
        snapshots = _snapshot_recurrent_caches(
            cache,
            0,
            self.n_layers,
        )
        old_cache = self._fresh_cache()
        new_cache = self._fresh_cache()
        _restore_recurrent_caches(
            old_cache,
            0,
            self.n_layers,
            snapshots,
        )
        _restore_recurrent_caches(
            new_cache,
            0,
            self.n_layers,
            snapshots,
        )
        kv_tree = getattr(self, "_episode_kv_state_tree", None)
        old_transaction = None
        new_transaction = None
        if kv_tree is not None:
            parent_sha256 = branch.kv_boundary_sha256 or kv_tree.root_sha256
            old_transaction = kv_tree.begin_speculation(
                old_cache,
                start=0,
                end=self.n_layers,
                purpose="heterogeneous_incumbent_lane",
                branch_index=branch.index,
                parent_sha256=parent_sha256,
                isolated=True,
            )
            new_transaction = kv_tree.begin_speculation(
                new_cache,
                start=0,
                end=self.n_layers,
                purpose="heterogeneous_corrected_lane",
                branch_index=branch.index,
                parent_sha256=parent_sha256,
                isolated=True,
            )
        saved_state = branch.z
        old_trace: list[str] = []
        new_trace: list[str] = []
        policy_trace: list[str] = []
        divergences: list[float] = []
        old_lane_apps = 0
        new_lane_apps = 0
        lane_execution_failed = True

        def persist(state, lane_cache):
            projected = branch.workspace.restore_context_evidence(mx.array(state))
            branch.z = projected
            branch.workspace.update(projected)
            return self._persist_branch(
                branch,
                lane_cache,
                runner,
                budget,
            )

        try:
            old_logits = persist(incumbent_state, old_cache)
            old_lane_apps += self.config.workspace.n_slots * self.n_layers
            new_logits = persist(corrected_state, new_cache)
            new_lane_apps += self.config.workspace.n_slots * self.n_layers
            if phase_checkpoint is not None:
                phase_checkpoint("persist")
            if bridge_tokens:
                old_logits = self._apply_decode_bridge(
                    old_cache,
                    budget,
                    bridge_tokens,
                )
                old_lane_apps += len(bridge_tokens) * self.n_layers
                new_logits = self._apply_decode_bridge(
                    new_cache,
                    budget,
                    bridge_tokens,
                )
                new_lane_apps += len(bridge_tokens) * self.n_layers
                if phase_checkpoint is not None:
                    phase_checkpoint("decode_bridge")
            old_initial_sha256 = _logits_digest(old_logits)
            new_initial_sha256 = _logits_digest(new_logits)

            def integrate(left, right):
                policy_logits, js_bits = self._heterogeneous_mix_logits(
                    left,
                    right,
                    policy=policy,
                    fusion_weight=fusion_weight,
                    budget=budget,
                )
                old_trace.append(_logits_digest(left))
                new_trace.append(_logits_digest(right))
                policy_trace.append(_logits_digest(policy_logits))
                divergences.append(js_bits)
                return policy_logits

            initial_logits = integrate(old_logits, new_logits)
            inner = self.model.model

            def advance(token: int):
                nonlocal old_lane_apps, new_lane_apps
                lane_logits = []
                for name, lane_cache in (
                    ("old", old_cache),
                    ("new", new_cache),
                ):
                    context_tokens = self._cache_context_tokens(lane_cache)
                    budget.charge(
                        tokens=1,
                        layers=self.n_layers,
                        operation=(
                            "heterogeneous_incumbent_decode"
                            if name == "old"
                            else "heterogeneous_corrected_decode"
                        ),
                        attention_pairs=max(1, context_tokens + 1) * self.n_layers,
                        output_head_tokens=1,
                    )
                    hidden = inner.embed_tokens(mx.array([[token]]))
                    mask = create_attention_mask(hidden, lane_cache)
                    from core.brain.llm.latent_cortex.recurrence_adapter import (
                        coda_adapter_scope,
                    )

                    with coda_adapter_scope():
                        for index, layer in enumerate(inner.layers):
                            hidden = layer(
                                hidden,
                                mask,
                                lane_cache[index],
                            )
                    logits = self._logits(hidden)[0, -1]
                    mx.eval(logits)
                    lane_logits.append(logits)
                    if name == "old":
                        old_lane_apps += self.n_layers
                    else:
                        new_lane_apps += self.n_layers
                return integrate(lane_logits[0], lane_logits[1])

            decoded, termination = self._decode(
                old_cache,
                budget,
                initial_logits,
                max_tokens=max_tokens,
                temperature=temperature,
                cancel_check=cancel_check,
                progress=progress,
                sentence_grace_tokens=sentence_grace_tokens,
                contract_grace_tokens=contract_grace_tokens,
                force_exact_tokens=force_exact_tokens,
                wall_reserve_s=wall_reserve_s,
                token_logprobs_out=token_logprobs_out,
                external_step_logits=advance,
                external_step_lanes=2,
            )
            lane_execution_failed = False
        finally:
            branch.z = saved_state
            branch.workspace.update(saved_state)
            for transaction, lane_cache in (
                (old_transaction, old_cache),
                (new_transaction, new_cache),
            ):
                if transaction is not None and not transaction.closed:
                    transaction.observe_mutation(
                        lane_cache,
                        execution_failed=lane_execution_failed,
                    )
            if old_transaction is not None and not old_transaction.closed:
                if retain_lanes and not lane_execution_failed:
                    old_transaction.commit(
                        label="final_incumbent_lane",
                        authority="heterogeneous_probability_fusion",
                        latent_sha256=tensor_sha256(incumbent_state),
                        final=True,
                    )
                else:
                    old_transaction.discard_isolated(parent_cache=cache)
            if new_transaction is not None and not new_transaction.closed:
                if retain_lanes and not lane_execution_failed:
                    new_transaction.commit(
                        label="final_corrected_lane",
                        authority="heterogeneous_probability_fusion",
                        latent_sha256=tensor_sha256(corrected_state),
                        final=True,
                    )
                else:
                    new_transaction.discard_isolated(parent_cache=cache)
        if not divergences or old_lane_apps <= 0 or old_lane_apps != new_lane_apps:
            raise RuntimeError("heterogeneous decode lane accounting differs")

        def trace_sha256(values: list[str]) -> str:
            return hashlib.sha256(":".join(values).encode("ascii")).hexdigest()

        audit = {
            "incumbent_state_sha256": tensor_sha256(incumbent_state),
            "corrected_state_sha256": tensor_sha256(corrected_state),
            "fusion_weight": round(float(fusion_weight), 10),
            "old_lane_layer_apps": old_lane_apps,
            "new_lane_layer_apps": new_lane_apps,
            "old_initial_logits_sha256": old_initial_sha256,
            "new_initial_logits_sha256": new_initial_sha256,
            "policy_initial_logits_sha256": policy_trace[0],
            "old_logits_trace_sha256": trace_sha256(old_trace),
            "new_logits_trace_sha256": trace_sha256(new_trace),
            "policy_logits_trace_sha256": trace_sha256(policy_trace),
            "mean_js_divergence_bits": round(
                sum(divergences) / len(divergences),
                10,
            ),
            "max_js_divergence_bits": round(max(divergences), 10),
            "divergence_samples": len(divergences),
        }
        return decoded, termination, audit

    def _heterogeneous_policy_evaluator(
        self,
        *,
        branch,
        cache,
        runner,
        budget: ComputeBudget,
        bridge_tokens: list[int],
        verifier,
    ):
        """Build one exact dual-lane policy evaluator."""

        from core.brain.llm.latent_cortex.counterfactual_probe import (
            CounterfactualProbeResult,
        )
        from core.brain.llm.latent_cortex.heterogeneous_integrator import (
            IntegrationPolicyResult,
        )
        from core.brain.llm.latent_cortex.verified_best import (
            VerifierObservation,
            validate_observation,
        )

        if verifier is None or self.tokenizer is None:
            return None

        def evaluate(
            policy: str,
            incumbent_state,
            corrected_state,
            fusion_weight: float,
            _replicate: int,
        ) -> IntegrationPolicyResult:
            spent_before = budget.spent_layer_apps
            decoded, termination, audit = self._heterogeneous_dual_lane_decode(
                branch=branch,
                cache=cache,
                runner=runner,
                budget=budget,
                incumbent_state=incumbent_state,
                corrected_state=corrected_state,
                policy=policy,
                fusion_weight=fusion_weight,
                bridge_tokens=bridge_tokens,
                max_tokens=self.config.verifier_probe_max_tokens,
                temperature=0.0,
                force_exact_tokens=True,
            )
            if (
                len(decoded) != self.config.verifier_probe_max_tokens
                or termination != "token_limit"
            ):
                raise RuntimeError(
                    "heterogeneous verifier probe did not complete its exact "
                    "equal-compute token contract"
                )
            measured_layer_apps = budget.spent_layer_apps - spent_before
            lane_layer_apps = audit["old_lane_layer_apps"] + audit["new_lane_layer_apps"]
            if measured_layer_apps != lane_layer_apps:
                raise RuntimeError(
                    "heterogeneous verifier lane accounting differs from the episode compute budget"
                )
            rendered = self.tokenizer.decode(decoded)
            bounded = getattr(verifier, "observe_with_bounds", None)
            raw_observation = bounded(rendered) if callable(bounded) else verifier(rendered)
            if isinstance(raw_observation, dict) and "observation_sha256" in raw_observation:
                observation = validate_observation(raw_observation)
            else:
                observation = VerifierObservation.from_value(raw_observation).to_dict()
            encoded = json.dumps(
                decoded,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
            policy_audit = {
                key: value for key, value in audit.items() if key != "policy_initial_logits_sha256"
            }
            return IntegrationPolicyResult(
                probe=CounterfactualProbeResult(
                    probe_tokens_sha256=hashlib.sha256(encoded).hexdigest(),
                    probe_token_count=len(decoded),
                    observation=observation,
                    layer_apps=measured_layer_apps,
                ),
                **policy_audit,
            )

        return evaluate

    def _verifier_probe_layer_apps(
        self,
        bridge_tokens: list[int] | None = None,
        *,
        count: int = 1,
    ) -> int:
        """Exact conservative token-layer cost for verifier preview decodes."""
        if type(count) is not int or count < 0:
            raise ValueError("verifier probe count must be a non-negative integer")
        per_probe_tokens = (
            self.config.workspace.n_slots
            + len(bridge_tokens or [])
            + max(0, self.config.verifier_probe_max_tokens - 1)
        )
        return count * per_probe_tokens * self.n_layers

    def _persist_branch(
        self,
        branch: BranchState,
        cache,
        runner: WindowRunner,
        budget: ComputeBudget,
    ):
        """Commit one branch's slots into every layer's KV (the causal step).

        Returns the last slot position's logits — the next-token distribution
        conditioned on [prompt; refined thoughts], which seeds decoding.
        """
        import mlx.core as mx

        from core.brain.llm.latent_cortex.recurrence_adapter import coda_adapter_scope
        from core.learning.role_conditioned_lora import recurrent_branch_index

        with recurrent_branch_index(branch.index):
            runner.run(
                branch.workspace.seed_z,
                cache,
                0,
                self.prelude_end,
                persist=True,
            )
            z_fin = runner.run(
                branch.z,
                cache,
                self.prelude_end,
                self.coda_start,
                persist=True,
            )
            with coda_adapter_scope():
                z_out = runner.run(
                    z_fin,
                    cache,
                    self.coda_start,
                    self.n_layers,
                    persist=True,
                )
        logits = self._logits(
            z_out[:, -1:, :],
            budget=budget,
            operation="persisted_workspace_output_head",
        )[0, -1]
        mx.eval(logits)
        return logits

    # ── Schedule resolution ─────────────────────────────────────────────
    def _resolve_schedule(self, domain: str) -> LayerSchedule:
        if self.config.schedule is not None:
            schedule = LayerSchedule.from_dict(self.config.schedule)
            violations = schedule.validate(prelude_end=self.prelude_end, coda_start=self.coda_start)
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
        action_policy_evidence: dict[str, Any] | None = None,
        action_intervention: dict[str, Any] | None = None,
        action_intervention_consumption: dict[str, Any] | None = None,
        external_execution_offer: dict[str, Any] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress: Callable[[dict], None] | None = None,
        capture_decode_logprobs: bool = False,
        decode_sentence_grace_tokens: int | None = None,
        sample_seed: int | None = None,
        incumbent_artifact: Any | None = None,
        episode_id: str | None = None,
        action_continuation_capture: Callable[[Any], None] | None = None,
        action_continuation_restore: Any | None = None,
        action_continuation_runner_state: Mapping[str, Any] | None = None,
        action_continuation_capture_only: bool = False,
        action_continuation_restore_verified: Callable[[str], None] | None = None,
        nonparametric_memory_enabled: bool = True,
    ) -> LatentReasoningResult:
        if type(capture_decode_logprobs) is not bool:
            raise TypeError("capture_decode_logprobs must be boolean")
        if decode_sentence_grace_tokens is not None and (
            type(decode_sentence_grace_tokens) is not int
            or not 0 <= decode_sentence_grace_tokens <= 4096
        ):
            raise ValueError("decode_sentence_grace_tokens must be null or inside [0, 4096]")
        if sample_seed is not None and (
            type(sample_seed) is not int or not 0 <= sample_seed <= 0xFFFFFFFF
        ):
            raise ValueError("sample_seed must be null or an integer inside [0, 2^32-1]")
        if episode_id is not None and (
            not isinstance(episode_id, str)
            or not episode_id
            or len(episode_id) > 192
            or not episode_id[0].isalnum()
            or any(not (character.isalnum() or character in "._:/;=+-") for character in episode_id)
        ):
            raise ValueError("episode_id is not a valid campaign identifier")
        if type(action_continuation_capture_only) is not bool:
            raise TypeError("action_continuation_capture_only must be boolean")
        if type(nonparametric_memory_enabled) is not bool:
            raise TypeError("nonparametric_memory_enabled must be boolean")
        continuation_requested = (
            action_continuation_capture is not None
            or action_continuation_restore is not None
            or action_continuation_runner_state is not None
            or action_continuation_capture_only
        )
        if continuation_requested:
            from core.brain.llm.latent_cortex.action_continuation import (
                ActionOpportunityContinuation,
            )

            if action_continuation_capture is not None and not callable(
                action_continuation_capture
            ):
                raise TypeError("action_continuation_capture must be callable")
            if action_continuation_restore_verified is not None and not callable(
                action_continuation_restore_verified
            ):
                raise TypeError("action_continuation_restore_verified must be callable")
            if (
                action_continuation_restore_verified is not None
                and action_continuation_restore is None
            ):
                raise ValueError("action_continuation_restore_verified requires a restore")
            if action_continuation_restore is not None and not isinstance(
                action_continuation_restore,
                ActionOpportunityContinuation,
            ):
                raise TypeError("action_continuation_restore has the wrong type")
            if not isinstance(action_continuation_runner_state, Mapping) or set(
                action_continuation_runner_state
            ) != {"durable_state", "rng_state"}:
                raise ValueError("action continuation requires exact runner state")
            if action_continuation_capture_only and action_continuation_capture is None:
                raise ValueError("capture-only continuation requires a capture callback")
        receipt = EpisodeReceipt(
            episode_id=(episode_id if episode_id is not None else uuid.uuid4().hex[:12])
        )
        episode_started = time.monotonic()
        receipt.n_layers = self.n_layers
        receipt.prelude_end = self.prelude_end
        receipt.coda_start = self.coda_start
        budget = budget or ComputeBudget()
        budget.bind_model(self.model)
        if decode_max_tokens is not None:
            if type(decode_max_tokens) is not int:
                raise TypeError("decode_max_tokens override must be an integer")
            if not 1 <= decode_max_tokens <= 8192:
                raise ValueError("decode_max_tokens override outside [1, 8192]")
        context_items = self._validate_cognitive_context(cognitive_context)
        policy_evidence = validate_evidence_snapshot(
            action_policy_evidence
            if action_policy_evidence is not None
            else build_evidence_snapshot(
                bucket=f"{str(domain or 'general')[:24]}|none|short|s:mid|u:mid",
                cells={},
            )
        )
        receipt.value_of_computation = {
            "schema": policy_evidence["schema"],
            "bucket": policy_evidence["bucket"],
            "snapshot_sha256": policy_evidence["snapshot_sha256"],
            "active": True,
        }
        normalized_execution_offer = None
        if external_execution_offer is not None:
            from core.brain.llm.latent_cortex.external_execution import (
                validate_external_execution_offer,
            )

            normalized_execution_offer = validate_external_execution_offer(external_execution_offer)
        normalized_action_intervention = None
        action_intervention_execution_claim = None
        if action_intervention is not None:
            from core.brain.llm.latent_cortex.action_intervention import (
                action_intervention_engine_request_sha256,
                claim_action_intervention_execution,
                validate_action_intervention,
                validate_action_intervention_objective,
            )

            normalized_action_intervention = validate_action_intervention(
                action_intervention,
                require_current_policy=True,
            )
            validate_action_intervention_objective(
                normalized_action_intervention,
                prompt=prompt,
                messages=messages,
                token_ids=token_ids,
            )
            if (
                decode_max_tokens is not None
                or capture_decode_logprobs
                or decode_sentence_grace_tokens is not None
            ):
                raise ValueError("action intervention does not permit direct decode overrides")
            engine_request_sha256 = action_intervention_engine_request_sha256(
                prompt=prompt,
                domain=str(domain),
                config=self.config,
                budget=budget,
                cognitive_context=context_items,
                action_policy_evidence=policy_evidence,
                external_execution_offer=normalized_execution_offer,
                verifier_present=verifier is not None,
                ablate_slot=ablate_slot,
                ablate_mode=ablate_mode,
            )
            if (
                engine_request_sha256
                != normalized_action_intervention["authority_payload"]["engine_request_sha256"]
            ):
                raise ValueError("action intervention engine request differs")
            if not isinstance(action_intervention_consumption, dict):
                raise ValueError("action intervention lacks a worker consumption event")
            action_intervention_execution_claim = claim_action_intervention_execution(
                normalized_action_intervention,
                action_intervention_consumption,
            )
        elif action_intervention_consumption is not None:
            raise ValueError("action intervention consumption lacks an intervention")
        tokens = self._encode(prompt, messages, token_ids)
        verification_objective = str(prompt or "")
        if not verification_objective and messages:
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        verification_objective = content
                        break
        encoded_tokens = json.dumps(tokens, separators=(",", ":"), allow_nan=False).encode("ascii")
        receipt.input_tokens_sha256 = hashlib.sha256(encoded_tokens).hexdigest()
        receipt.input_token_count = len(tokens)
        budget.bind_information(
            self._information_receipt(
                encoded_tokens=encoded_tokens,
                token_count=len(tokens),
                context_items=context_items,
                policy_evidence=policy_evidence,
                verifier=verifier,
            )
        )
        metered_verifier = self._meter_verifier(verifier, budget)
        receipt.decode_temperature = float(self.config.decode_temperature)
        receipt.decode_top_p = float(self.config.decode_top_p)
        receipt.decode_bridge_policy = self.config.decode_bridge_policy
        receipt.decode_incumbent_policy = self.config.decode_incumbent_policy
        receipt.verifier_probe_max_tokens = self.config.verifier_probe_max_tokens
        receipt.verifier_probe_contract = self.config.verifier_probe_contract
        receipt.decode_contract_required = self.config.decode_contract == "final_answer_v1"
        receipt.decode_contract_grace_tokens = (
            self.config.decode_contract_grace_tokens if receipt.decode_contract_required else 0
        )

        self.invariant.pre_episode()
        receipt.checkpoint_fingerprint = self.invariant.file_receipt.get("fingerprint", "")
        receipt.checkpoint_fingerprint_method = self.invariant.file_receipt.get("method", "")
        receipt.checkpoint_file_count = int(self.invariant.file_receipt.get("files", 0) or 0)
        validated_incumbent = None
        if incumbent_artifact is not None:
            if self.config.decode_incumbent_policy != "vanilla_incumbent":
                raise ValueError(
                    "an incumbent artifact requires decode_incumbent_policy=vanilla_incumbent"
                )
            if self.tokenizer is None:
                raise ValueError("an incumbent artifact requires the serving tokenizer")
            from core.brain.llm.latent_cortex.incumbent_artifact import (
                validate_incumbent_artifact,
            )

            validated_incumbent = validate_incumbent_artifact(
                incumbent_artifact,
                input_tokens=tokens,
                checkpoint_fingerprint=receipt.checkpoint_fingerprint,
                checkpoint_fingerprint_method=receipt.checkpoint_fingerprint_method,
                max_tokens=(
                    decode_max_tokens
                    if decode_max_tokens is not None
                    else self.config.decode_max_tokens
                ),
                n_layers=self.n_layers,
                decode=lambda values: self._decode_public_text(list(values), receipt=receipt),
            )
            receipt.incumbent_artifact = dict(validated_incumbent.receipt)
            budget.charge_layer_apps(
                int(validated_incumbent.receipt["compute"]["transformer_layer_apps"]),
                operation="bound_incumbent_generation",
            )

        failure_reason = ""
        continuation_captured_only = False
        out_tokens: list[int] = []
        decode_token_logprobs: list[float] = []
        answer_replacement_private: dict[str, Any] = {}
        transient_cleanup_registry: list[Any] = []
        try:
            try:
                (
                    out_tokens,
                    receipt,
                    answer_replacement_private,
                ) = self._latent_episode(
                    tokens,
                    budget,
                    metered_verifier,
                    domain,
                    receipt,
                    decode_max_tokens,
                    ablate_slot=ablate_slot,
                    ablate_mode=ablate_mode,
                    cognitive_context_items=context_items,
                    action_policy_evidence=policy_evidence,
                    action_intervention=normalized_action_intervention,
                    action_intervention_consumption=action_intervention_consumption,
                    action_intervention_execution_claim=(action_intervention_execution_claim),
                    external_execution_offer=normalized_execution_offer,
                    information_encoded_tokens=encoded_tokens,
                    information_verifier=verifier,
                    verification_objective=verification_objective,
                    cancel_check=cancel_check,
                    progress=progress,
                    episode_started=episode_started,
                    token_logprobs_out=(decode_token_logprobs if capture_decode_logprobs else None),
                    decode_sentence_grace_tokens=decode_sentence_grace_tokens,
                    sample_seed=sample_seed,
                    incumbent_artifact=validated_incumbent,
                    transient_cleanup_registry=transient_cleanup_registry,
                    action_continuation_capture=action_continuation_capture,
                    action_continuation_restore=action_continuation_restore,
                    action_continuation_runner_state=(
                        dict(action_continuation_runner_state)
                        if action_continuation_runner_state is not None
                        else None
                    ),
                    action_continuation_capture_only=action_continuation_capture_only,
                    action_continuation_restore_verified=(action_continuation_restore_verified),
                    nonparametric_memory_enabled=nonparametric_memory_enabled,
                )
                if receipt.answer_replacement.get("decision") == "abstain":
                    # Abstention may end the episode only when the latent lane
                    # owns the output. Under vanilla_incumbent the answer on
                    # the table IS ordinary decode, and failing the episode
                    # discards it -- ok=False, zero tokens, empty text.
                    #
                    # This is the SECOND abstain path; guarding only the one
                    # that sets decode_termination left this one live. Measured
                    # on the 32B: four of fourteen cells returned empty text
                    # against receipts reporting 278-560 generated tokens and
                    # clean eos/token_limit terminations, and three of those
                    # four were tasks ordinary decode got RIGHT. The reason was
                    # known_refutation_has_no_dominant_repair -- the verifier
                    # judged the baseline refuted when it was correct, and a
                    # false refutation then threw the correct answer away.
                    #
                    # Serving the incumbent is weakly better in every case:
                    # equal when the refutation is right (vanilla was wrong
                    # anyway) and strictly better when it is wrong.
                    if self.config.decode_incumbent_policy == "latent":
                        failure_reason = "answer_replacement_abstained"
                    else:
                        receipt.flag("answer_replacement_abstention_declined_under_incumbent")
            except _FastWeightCleanupError as exc:
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="refused fallback decode and requested resident-worker recycle",
                    severity="critical",
                )
                failure_reason = "fast_weight_cleanup_unproven"
            except _LatentEpisodeCancelledError:
                receipt.flag("soft_cancelled")
                receipt.halting_reason = receipt.halting_reason or "soft_cancelled"
                failure_reason = "soft_cancelled"
            except _ActionContinuationCapturedError:
                continuation_captured_only = True
                receipt.last_stage = "action_state_captured"
                receipt.halting_reason = "action_state_captured_before_first_action"
            except UnknownActionStateApplicationError:
                raise
            except _LATENT_PHASE_ERRORS as exc:
                fallback_permitted = (
                    self.config.allow_vanilla_fallback and normalized_action_intervention is None
                )
                record_degradation(
                    "latent_cortex",
                    exc,
                    action=(
                        "served vanilla decode with honest fallback receipt"
                        if fallback_permitted
                        else "failed the full-stack episode without replacing it with vanilla decode"
                    ),
                )
                receipt.halting_reason = receipt.halting_reason or "latent_phase_error"
                if not fallback_permitted:
                    receipt.flag(
                        "campaign_vanilla_fallback_forbidden"
                        if normalized_action_intervention is not None
                        else "vanilla_fallback_disabled"
                    )
                    failure_reason = f"latent_phase_failed:{type(exc).__name__}:{exc}"
                elif receipt.fast_weights_applied and receipt.fast_weights_erased is not True:
                    receipt.flag("fallback_refused_unproven_model_state")
                    failure_reason = "fast_weight_cleanup_unproven"
                else:
                    receipt.flag(f"fallback_vanilla:{type(exc).__name__}")
                    try:
                        decode_token_logprobs.clear()
                        cache = self._fresh_cache()
                        _, tail_logits = self._prefill(tokens, cache, budget)
                        out_tokens, decode_termination = self._decode(
                            cache,
                            budget,
                            tail_logits,
                            max_tokens=decode_max_tokens,
                            cancel_check=cancel_check,
                            progress=progress,
                            token_logprobs_out=(
                                decode_token_logprobs if capture_decode_logprobs else None
                            ),
                            sentence_grace_tokens=decode_sentence_grace_tokens,
                            sample_seed=sample_seed,
                        )
                        receipt.decode_requested_tokens = (
                            decode_max_tokens
                            if decode_max_tokens is not None
                            else self.config.decode_max_tokens
                        )
                        receipt.decode_generated_tokens = len(out_tokens)
                        receipt.decode_termination = decode_termination
                        receipt.decode_contract_satisfied = bool(
                            self._last_decode_contract_satisfied
                        )
                        receipt.decode_contract_grace_used_tokens = int(
                            self._last_decode_contract_grace_used_tokens
                        )
                        if decode_termination.startswith("budget_"):
                            receipt.flag(f"decode_{decode_termination}")
                    except _LatentEpisodeCancelledError:
                        receipt.flag("soft_cancelled")
                        receipt.halting_reason = receipt.halting_reason or "soft_cancelled"
                        failure_reason = "soft_cancelled"
                    except _LATENT_PHASE_ERRORS as inner_exc:
                        record_degradation(
                            "latent_cortex",
                            inner_exc,
                            action=("reported failed episode after vanilla fallback also failed"),
                            severity="degraded",
                        )
                        failure_reason = f"latent_and_fallback_failed:{inner_exc}"
        finally:
            for transient_ledger in transient_cleanup_registry:
                try:
                    transient_ledger.abort_all()
                except _LATENT_PHASE_ERRORS as exc:
                    receipt.flag("transient_constraint_cleanup_unproven")
                    failure_reason = failure_reason or "transient_constraint_cleanup_unproven"
                    record_degradation(
                        "latent_cortex",
                        exc,
                        action="refused the episode because transient authority cleanup failed",
                        severity="critical",
                    )
            try:
                receipt.params_unchanged = self.invariant.post_episode(receipt)
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
        from core.brain.llm.latent_cortex.causal_receipt import (
            build_causal_receipt,
        )

        receipt.causal_receipt = build_causal_receipt(receipt.to_dict())
        if continuation_captured_only:
            receipt.last_stage = "action_state_captured"
            receipt.halting_reason = "action_state_captured_before_first_action"
            return LatentReasoningResult(
                ok=True,
                text="",
                receipt=receipt,
                reason="action_state_captured",
                decode_token_logprobs=decode_token_logprobs,
                answer_replacement_private=answer_replacement_private,
            )
        if not failure_reason and receipt.decode_termination not in {
            "eos",
            # The public answer contract completed: one FINAL_ANSWER JSON
            # object closed and parsed — the strongest completion signal a
            # contract task has (CP180).
            "contract_complete",
            # A bounded negative output remains valid scientific evidence.
            # The live service rejects it as product-incomplete.
            "token_limit_contract_incomplete",
            "token_limit",
            # The limit landed mid-sentence and sampling continued a few
            # model-chosen tokens to the natural boundary — a complete
            # answer, receipted under its own termination kind.
            "token_limit_sentence_grace",
            # Time pressure ended decoding at a sentence boundary (the
            # wall-clock analogue of the token-limit grace). A time-bounded
            # stop has the same epistemic status as a token-bounded one:
            # the product-quality gate — terminal completeness, facet and
            # subject coverage — judges whether the text stands as an
            # answer, not the budget dimension that ended sampling.
            "wall_reserve_sentence_grace",
            "wall_reserve",
            # A separately generated, exactly round-tripped repair cleared
            # the confidence-bound authority gate and replaced the ordinary
            # neural decode.
            "confidence_bound_replacement",
        }:
            failure_reason = f"decode_incomplete:{receipt.decode_termination}"
        if receipt.params_unchanged is False:
            receipt.flag("checkpoint_invariant_violated")
            failure_reason = failure_reason or "checkpoint_invariant_violated"
        if failure_reason:
            # A bounded decode can be an invalid product answer while still
            # being valid raw-policy evidence. Preserve only that explicitly
            # classified neural trace so research callers can grade it and
            # learn from its token log-probabilities. Integrity, cancellation,
            # latent-phase, and cleanup failures remain empty and unusable.
            retain_policy_trace = failure_reason.startswith("decode_incomplete:")
            failure_tokens = out_tokens if retain_policy_trace else []
            failure_text = (
                self._decode_public_text(failure_tokens, receipt=receipt) if failure_tokens else ""
            )
            return LatentReasoningResult(
                ok=False,
                text=failure_text,
                receipt=receipt,
                tokens=failure_tokens,
                reason=failure_reason,
                decode_token_logprobs=decode_token_logprobs,
                answer_replacement_private=answer_replacement_private,
            )

        text = self._decode_public_text(out_tokens, receipt=receipt) if out_tokens else ""
        return LatentReasoningResult(
            ok=True,
            text=text,
            receipt=receipt,
            tokens=out_tokens,
            decode_token_logprobs=decode_token_logprobs,
            answer_replacement_private=answer_replacement_private,
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
        action_policy_evidence: dict[str, Any],
        action_intervention: dict[str, Any] | None = None,
        action_intervention_consumption: dict[str, Any] | None = None,
        action_intervention_execution_claim: dict[str, Any] | None = None,
        external_execution_offer: dict[str, Any] | None = None,
        information_encoded_tokens: bytes,
        information_verifier: Callable[[str], float] | None,
        verification_objective: str = "",
        cancel_check: Callable[[], bool] | None = None,
        progress: Callable[[dict], None] | None = None,
        episode_started: float | None = None,
        token_logprobs_out: list[float] | None = None,
        decode_sentence_grace_tokens: int | None = None,
        sample_seed: int | None = None,
        incumbent_artifact: Any | None = None,
        transient_cleanup_registry: list[Any] | None = None,
        action_continuation_capture: Callable[[Any], None] | None = None,
        action_continuation_restore: Any | None = None,
        action_continuation_runner_state: dict[str, Any] | None = None,
        action_continuation_capture_only: bool = False,
        action_continuation_restore_verified: Callable[[str], None] | None = None,
        nonparametric_memory_enabled: bool = True,
    ) -> tuple[list[int], EpisodeReceipt, dict[str, Any]]:
        import mlx.core as mx

        episode_started = (
            float(episode_started) if episode_started is not None else time.monotonic()
        )
        answer_replacement_private: dict[str, Any] = {}
        stage_started = time.monotonic()
        if self._cancel_requested(cancel_check):
            raise _LatentEpisodeCancelledError("admission")
        cache = self._fresh_cache()
        runner = WindowRunner(self.model.model, budget)
        decode_limit = (
            decode_max_tokens if decode_max_tokens is not None else self.config.decode_max_tokens
        )
        bridge_tokens = self._decode_bridge_tokens()
        terminal_instruction_reserve = self._terminal_instruction_reserve()
        terminal_instruction_tokens: list[int] = []
        terminal_decision = None
        prefill_cost = len(tokens) * self.n_layers
        contract_grace = (
            self.config.decode_contract_grace_tokens
            if self.config.decode_contract == "final_answer_v1"
            else 0
        )
        decode_cost = (
            max(
                0,
                int(decode_limit) + int(contract_grace) - 1,
            )
            * self.n_layers
        )
        persist_cost = self.config.workspace.n_slots * self.n_layers
        bridge_cost = (len(bridge_tokens) + terminal_instruction_reserve) * self.n_layers
        fast_weight_probe_cost = 8 * self.n_layers if self.config.fast_weights.enabled else 0
        fast_weight_verifier_probe_cost = (
            self._verifier_probe_layer_apps(bridge_tokens)
            if self.config.fast_weights.enabled and self.tokenizer is not None
            else 0
        )
        fast_weight_window_forward_cost = (
            self.config.workspace.n_slots * (self.coda_start - self.prelude_end)
            if self.config.fast_weights.enabled
            else 0
        )
        fast_weight_associative_capture_cost = (
            fast_weight_window_forward_cost
            + (
                3
                * min(
                    _MAX_TEACHER_TRAJECTORY_TOKENS,
                    len(tokens) + 256 + 32,
                )
                * self.n_layers
            )
            if self.config.fast_weights.enabled
            and self.config.fast_weights.associative_bootstrap_enabled
            else 0
        )
        output_memory_diagnostic_cost = (
            (
                2
                * (
                    self.config.workspace.n_slots
                    + len(bridge_tokens)
                    + 255
                )
                * self.n_layers
            )
            + (2 * len(OUTPUT_MEMORY_GAIN_GRID) * fast_weight_verifier_probe_cost)
            if self.config.fast_weights.enabled
            and self.config.fast_weights.output_memory_diagnostic_enabled
            else 0
        )
        fast_weight_matched_trial_cost = (
            2
            * self.config.fast_weights.opt_steps
            * (3 + MATCHED_LINE_SEARCH_EVALUATIONS)
            * fast_weight_window_forward_cost
            + fast_weight_verifier_probe_cost
            + fast_weight_associative_capture_cost
            + output_memory_diagnostic_cost
            + (
                2 * len(VERIFIER_GAIN_GRID) * fast_weight_verifier_probe_cost
                if self.config.fast_weights.associative_bootstrap_enabled
                else 0
            )
            if self.config.fast_weights.enabled
            else 0
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
            # A postcondition battery that is never affordable is theatre:
            # it would report "not run" every episode while the receipt kept
            # saying canaries passed. Reserve it so it actually runs.
            if self.config.fast_weights.canary_generated_enabled:
                canary_reserve += canaries.tokens_per_generated_measurement * self.n_layers
        completion_reserve = (
            persist_cost
            + bridge_cost
            + decode_cost
            + fast_weight_probe_cost
            + fast_weight_verifier_probe_cost
        )
        fallback_reserve = prefill_cost + decode_cost
        safety_reserve = completion_reserve + fallback_reserve + canary_reserve
        branch_seed_cost = (
            self.config.branches.n_branches * self.config.workspace.n_slots * self.prelude_end
        )
        # Baseline, measured identity immediately after attach, and the
        # same-query pre-adaptation verifier probe are all mandatory before
        # temporary learning may start. The completion reserve separately
        # protects the matched post-adaptation probe and cleanup proof.
        fast_weight_attach_identity_cost = 2 * fast_weight_probe_cost + canary_pass_cost
        fast_weight_baseline_cost = (
            fast_weight_attach_identity_cost
            + fast_weight_verifier_probe_cost
            + fast_weight_matched_trial_cost
        )
        minimum_admission = (
            prefill_cost + branch_seed_cost + safety_reserve + fast_weight_baseline_cost
        )
        if minimum_admission > budget.remaining_layer_apps or budget.exhausted:
            raise RuntimeError(
                "compute budget cannot admit latent minimum while preserving fallback: "
                f"required={minimum_admission} remaining={budget.remaining_layer_apps}"
            )

        embeddings, prompt_tail_logits = self._prefill(tokens, cache, budget)
        receipt.decode_incumbent_prompt_logits_sha256 = _logits_digest(prompt_tail_logits)
        from core.brain.llm.latent_cortex.kv_state_tree import KVStateTree

        kv_state_tree = KVStateTree(
            cache,
            n_layers=self.n_layers,
            episode_id=receipt.episode_id,
            input_tokens_sha256=receipt.input_tokens_sha256,
        )
        self._episode_kv_state_tree = kv_state_tree
        runner.attach_kv_state_tree(kv_state_tree)
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

        episode_context_items = list(cognitive_context_items or [])
        from core.brain.llm.latent_cortex.nonparametric_context import (
            retrieve_observation,
        )
        from core.brain.llm.latent_cortex.nonparametric_context import (
            validate_receipt as validate_nonparametric_receipt,
        )

        one_shot_observation, one_shot_receipt = retrieve_observation(
            self._last_prefill_hidden,
            self.tokenizer,
            enabled=nonparametric_memory_enabled,
        )
        receipt.nonparametric_memory = validate_nonparametric_receipt(one_shot_receipt)
        one_shot_accounting = receipt.nonparametric_memory["resource_accounting"]
        budget.charge_tensor_work(
            "nonparametric_memory_retrieval",
            element_reads=one_shot_accounting["tensor_element_reads"],
            element_writes=one_shot_accounting["tensor_element_writes"],
            scalar_ops=one_shot_accounting["tensor_scalar_ops"],
            host_scalar_ops=one_shot_accounting["host_scalar_ops"],
        )
        if one_shot_observation is not None:
            from core.brain.llm.latent_cortex.cognitive_context import (
                normalize_cognitive_context,
            )

            episode_context_items.extend(normalize_cognitive_context([one_shot_observation]))
        budget.bind_information(
            self._information_receipt(
                encoded_tokens=information_encoded_tokens,
                token_count=len(tokens),
                context_items=episode_context_items,
                policy_evidence=action_policy_evidence,
                verifier=information_verifier,
                nonparametric_identity=receipt.nonparametric_memory.get("source_identity"),
            )
        )
        context_seeds = self._embed_cognitive_context(episode_context_items)
        telemetry = LatentTelemetry(enabled=bool(self.config.telemetry_enabled))
        # Probe memoization lives exactly one episode: identical latent
        # states decode once; the cache empties the moment ΔW changes the
        # model function.
        self._episode_probe_cache = DecodeProbeCache() if self.config.probe_cache_enabled else None
        escape_cfg = EscapeConfig(**dict(self.config.escape or {}))
        ensemble = BranchEnsemble.seed(
            embeddings,
            self.config.workspace,
            self.config.branches,
            self.config.recurrence,
            runner,
            cache,
            self.prelude_end,
            context_seeds=context_seeds,
            escape_cfg=escape_cfg,
        )
        ensemble.bind_kv_state_tree(kv_state_tree, cache)
        from core.brain.llm.latent_cortex.correlated_support import (
            initial_exchange_weights,
            validate_correlation_evidence,
        )

        branch_roles = [branch.role for branch in ensemble.branches]
        correlation_evidence = validate_correlation_evidence(
            self.config.branch_correlation_evidence,
            roles=branch_roles,
        )
        ensemble.set_support_weights(
            initial_exchange_weights(
                roles=branch_roles,
                correlation_evidence=correlation_evidence,
            )
        )
        ensemble.telemetry = telemetry if telemetry.enabled else None
        stop_gate = self._resolve_halting_head()
        if stop_gate.mode == "learned":
            for branch in ensemble.branches:
                branch.halting.stop_gate = stop_gate
        update_gate = self._resolve_update_gate()
        uncertainty_runtime = self._resolve_uncertainty_head()
        mistake_locator_runtime = self._resolve_mistake_locator()
        contradiction_runtime = self._resolve_contradiction_head()
        for branch in ensemble.branches:
            branch.update_gate = update_gate
            branch.uncertainty_runtime = uncertainty_runtime
            branch.mistake_locator_runtime = mistake_locator_runtime
        from core.brain.llm.latent_cortex.transient_constraints import (
            TransientConstraintConfig,
            TransientConstraintLedger,
        )

        transient_constraint_config = TransientConstraintConfig.from_value(
            self.config.transient_negative_constraints
        )
        transient_constraints = TransientConstraintLedger(
            episode_id=receipt.episode_id,
            objective_sha256=receipt.input_tokens_sha256,
            n_branches=len(ensemble.branches),
            protected_positions={
                branch.index: branch.workspace.context_slot_indices for branch in ensemble.branches
            },
            config=transient_constraint_config,
        )
        if transient_cleanup_registry is not None:
            transient_cleanup_registry.append(transient_constraints)
        information = budget.information_receipt or {}
        information_policies = information.get("policies")
        transient_verifier_policy_sha256 = (
            str(information_policies.get("verifier", ""))
            if isinstance(information_policies, dict)
            else ""
        )
        if ensemble.branches and ensemble.branches[0].workspace.context_slots:
            seeded = ensemble.branches[0].workspace.context_slots
            from core.brain.llm.latent_cortex.cognitive_context import (
                knowledge_metadata,
            )

            receipt.cognitive_slots = [
                {
                    "slot": row["slot"],
                    "context_index": row["context_index"],
                    "source": row["source"],
                    "role": "immutable_evidence",
                    "causal_order": "before_hypothesis",
                    "text_chars": len(episode_context_items[row["context_index"]].get("text", "")),
                    "text_sha256": hashlib.sha256(
                        episode_context_items[row["context_index"]].get("text", "").encode("utf-8")
                    ).hexdigest(),
                    **knowledge_metadata(episode_context_items[row["context_index"]]),
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
        bytecode_events: list[dict[str, Any]] = []
        last_probe_scores: dict[int, float] = {}
        pending_verifier = verifier
        # Verifier-dependent recurrence is withheld until the decoy preflight
        # proves discrimination and repeat stability. The later mixed batch
        # revalidates authority before branch selection and adaptation.
        verifier = None
        if pending_verifier is not None and self.tokenizer is not None:
            try:
                from core.brain.llm.latent_cortex.blind_review import (
                    run_decoy_preflight,
                )

                receipt.verifier_preflight = run_decoy_preflight(
                    pending_verifier,
                    episode_id=receipt.episode_id,
                    objective_sha256=receipt.input_tokens_sha256,
                )
                if receipt.verifier_preflight["verifier_admitted"]:
                    verifier = pending_verifier
                else:
                    receipt.flag("verifier_preflight_decoy_calibration_failed")
            except Exception as exc:
                receipt.flag(f"verifier_preflight_failed:{type(exc).__name__}")
                pending_verifier = None
                verifier = None
        from core.brain.llm.latent_cortex.virtual_quanta import (
            ARM_NAMES,
            VirtualQuantaConfig,
            run_virtual_quanta,
        )

        virtual_quanta_config = VirtualQuantaConfig.from_value(self.config.virtual_quanta)
        virtual_target = ensemble.branches[
            int(receipt.input_tokens_sha256[:8], 16) % len(ensemble.branches)
        ]
        virtual_preflight_sha256 = str(receipt.verifier_preflight.get("receipt_sha256", ""))
        virtual_probe_apps = self._verifier_probe_layer_apps(
            bridge_tokens,
            count=len(ARM_NAMES) * virtual_quanta_config.replicates,
        )
        virtual_evaluator = None
        virtual_unavailable_reason = ""
        if virtual_quanta_config.mode == "disabled":
            virtual_unavailable_reason = "configured_disabled"
        elif verifier is None:
            virtual_unavailable_reason = "admitted_verifier_unavailable"
        elif not callable(getattr(verifier, "observe_with_bounds", None)):
            virtual_unavailable_reason = "bounded_verifier_observation_unavailable"
        elif len(transient_verifier_policy_sha256) != 64 or len(virtual_preflight_sha256) != 64:
            virtual_unavailable_reason = "verifier_authority_commitment_unavailable"
        elif virtual_probe_apps + safety_reserve > budget.remaining_layer_apps:
            virtual_unavailable_reason = "matched_counterfactual_probe_budget_unavailable"
        else:
            virtual_evaluator = self._counterfactual_probe_evaluator(
                branch=virtual_target,
                cache=cache,
                runner=runner,
                budget=budget,
                bridge_tokens=bridge_tokens,
                verifier=verifier,
            )

        virtual_baseline = virtual_target.z

        def apply_virtual_state(state):
            import mlx.core as mx
            import numpy as np

            projected = virtual_target.workspace.restore_context_evidence(mx.array(state))
            mx.eval(projected)
            virtual_target.z = projected
            virtual_target.workspace.update(projected)
            return np.asarray(projected)

        receipt.virtual_quanta = run_virtual_quanta(
            baseline_state=virtual_baseline,
            anchor_state=virtual_target.anchor,
            branch_index=virtual_target.index,
            protected_positions=virtual_target.workspace.context_slot_indices,
            source_positions=virtual_target.workspace.context_slot_indices,
            episode_id=receipt.episode_id,
            objective_sha256=receipt.input_tokens_sha256,
            subject_sha256=hashlib.sha256(str(domain or "general").encode("utf-8")).hexdigest(),
            source_kv_boundary_sha256=virtual_target.kv_boundary_sha256,
            verifier_policy_sha256=transient_verifier_policy_sha256,
            verifier_preflight_sha256=virtual_preflight_sha256,
            created_step=0,
            config=virtual_quanta_config,
            evaluate=virtual_evaluator,
            apply_state=apply_virtual_state,
            restore_state=apply_virtual_state,
            budget=budget,
            unavailable_reason=virtual_unavailable_reason,
        )
        if str(receipt.virtual_quanta["reason"]).startswith("counterfactual_failed:"):
            receipt.flag("virtual_quanta_" + receipt.virtual_quanta["reason"])
        from core.brain.llm.latent_cortex.latent_tree_search import (
            DISABLED as LATENT_TREE_DISABLED,
        )
        from core.brain.llm.latent_cortex.latent_tree_search import (
            LatentTreeSearchConfig,
            build_empty_latent_tree_receipt,
            run_latent_tree_search,
        )
        from core.brain.llm.latent_cortex.latent_tree_search import (
            append_transaction as append_latent_tree_transaction,
        )

        latent_tree_config = LatentTreeSearchConfig.from_value(self.config.latent_tree_search)
        latent_tree_status = (
            "disabled" if latent_tree_config.mode == LATENT_TREE_DISABLED else "not_invoked"
        )
        receipt.latent_tree_search = build_empty_latent_tree_receipt(
            episode_id=receipt.episode_id,
            objective_sha256=receipt.input_tokens_sha256,
            config=latent_tree_config,
            status=latent_tree_status,
            reason=(
                "configured_disabled"
                if latent_tree_status == "disabled"
                else "branch_action_not_selected"
            ),
        )
        value_policy = ValueOfComputationPolicy(action_policy_evidence)
        action_controls = self._embed_action_controls()
        has_memory = any(
            item.get("context_role") == "memory_observation" for item in episode_context_items
        )
        has_evidence = any(
            item.get("context_role") == "evidence_observation"
            or str(item.get("source") or "") in {"reference", "world_model"}
            or str(item.get("source") or "").startswith(("evidence", "tool_observation"))
            for item in episode_context_items
        )
        action_executors = self._action_executors(
            has_controls=bool(action_controls),
            has_verifier=verifier is not None and self.tokenizer is not None,
            can_execute=external_execution_offer is not None,
        )
        active_action_executors = action_executors
        intervention_pending = action_intervention is not None
        intervention_runtime: dict[str, Any] = {}
        continuation_pending = (
            action_continuation_capture is not None or action_continuation_restore is not None
        )
        active_action_continuation = None
        intervention_action: OperationKind | None = None
        intervention_arm = ""
        if action_intervention is not None:
            intervention_authority = action_intervention["authority_payload"]
            intervention_action = OperationKind(intervention_authority["action"])
            intervention_arm = str(intervention_authority["arm"])
            if intervention_action not in action_executors:
                raise ValueError("authenticated action intervention has no resident executor")
            if intervention_action is OperationKind.EXECUTE and external_execution_offer is None:
                raise ValueError("execute intervention lacks a governed external execution offer")
        selected_actions: list[OperationKind] = []
        cognitive_operator_trace: list[dict[str, Any]] = []
        context_focus_trace: list[dict[str, Any]] = []
        action_index = 0
        omitted_action_count = 0
        previous_residual = 1.0
        branch_verifier_scores: dict[int, float] = {}
        branch_verifier_deltas: dict[int, float] = {}
        ensemble.savepoint_all(
            authority="episode_initialization",
        )
        for op_index, op in enumerate(schedule.ops):
            op_kind = getattr(op, "kind", "window")
            if op_kind == "exchange":
                bytecode_events.append(
                    {
                        "op": op_index,
                        "kind": op_kind,
                        "done": ensemble.exchange_now(
                            sync_kind="schedule_bytecode",
                            sync_id=f"schedule:{schedule.schedule_hash}:op:{op_index}",
                            budget=budget,
                        ),
                    }
                )
                continue
            if op_kind == "savepoint":
                bytecode_events.append(
                    {
                        "op": op_index,
                        "kind": op_kind,
                        "branches": ensemble.savepoint_all(
                            authority="schedule_program",
                        ),
                    }
                )
                continue
            if op_kind == "verify_probe":
                event: dict[str, Any] = {
                    "op": op_index,
                    "kind": op_kind,
                    "ran": False,
                }
                probe_cost = self._verifier_probe_layer_apps(bridge_tokens)
                if verifier is None or self.tokenizer is None:
                    event["skip"] = "no_verifier"
                elif probe_cost + safety_reserve > budget.remaining_layer_apps:
                    event["skip"] = "budget"
                    receipt.flag("bytecode_probe_skipped_budget")
                else:
                    candidates = ensemble.active() or list(ensemble.branches)
                    if op.revert_on_drop and last_probe_scores:
                        comparable = [
                            branch for branch in candidates if branch.index in last_probe_scores
                        ]
                        if comparable:
                            candidates = comparable
                    target = min(
                        candidates,
                        key=lambda b: (
                            b.halting.residual_trail[-1]
                            if b.halting.residual_trail
                            else float("inf")
                        ),
                    )
                    probe = self._decode_probe(
                        target,
                        cache,
                        runner,
                        budget,
                        bridge_tokens=bridge_tokens,
                    )
                    probe_score = float(verifier(self.tokenizer.decode(probe)))
                    previous_score = last_probe_scores.get(target.index)
                    event.update(
                        {
                            "ran": True,
                            "branch": target.index,
                            "score": round(probe_score, 6),
                            "previous_score": (
                                round(previous_score, 6) if previous_score is not None else None
                            ),
                        }
                    )
                    if (
                        op.revert_on_drop
                        and previous_score is not None
                        and probe_score < previous_score - 1e-9
                    ):
                        event["reverted_branches"] = int(
                            ensemble.revert_branch_to_savepoint(target)
                        )
                        receipt.flag("bytecode_probe_reverted")
                    elif math.isfinite(probe_score):
                        last_probe_scores[target.index] = max(
                            last_probe_scores.get(target.index, -math.inf),
                            probe_score,
                        )
                bytecode_events.append(event)
                continue
            for repeat_index in range(op.repeats):
                if ensemble.all_halted() or budget.exhausted:
                    break
                if action_index >= self.config.recurrence.max_steps:
                    ensemble.halt_all(
                        "value_controller_action_budget",
                        budget=budget,
                    )
                    receipt.flag("value_controller_action_budget")
                    break
                before_residual = self._mean_latest_residual(ensemble)
                before_disagreement = ensemble.disagreement(budget=budget)
                signal_candidates = ensemble.active() or list(ensemble.branches)
                future_window_turn = repeat_index + 1 < op.repeats or any(
                    getattr(future_op, "kind", "window") == "window" and future_op.repeats > 0
                    for future_op in schedule.ops[op_index + 1 :]
                )
                prospective_target: BranchState | None = None
                pending_constraint_action = None
                if not intervention_pending:
                    for candidate in sorted(signal_candidates, key=lambda branch: branch.index):
                        pending_name = transient_constraints.pending_action(
                            branch_index=candidate.index,
                            action_step=action_index,
                            kv_boundary_sha256=candidate.kv_boundary_sha256,
                            state_sha256=tensor_sha256(candidate.z),
                        )
                        if pending_name is not None:
                            prospective_target = candidate
                            pending_constraint_action = OperationKind(pending_name)
                            break
                if prospective_target is None:
                    prospective_target = min(
                        signal_candidates,
                        key=lambda branch: (
                            branch.halting.residual_trail[-1]
                            if branch.halting.residual_trail
                            else float("inf")
                        ),
                    )
                previous_verifier_score = branch_verifier_scores.get(prospective_target.index)
                previous_verifier_delta = branch_verifier_deltas.get(prospective_target.index)
                remaining_fraction = budget.remaining_layer_apps / max(1, budget.max_layer_apps)
                state_signal = CognitiveStateSignal(
                    step_index=min(action_index, self.config.recurrence.max_steps),
                    max_steps=self.config.recurrence.max_steps,
                    neural_steps=max(
                        (branch.steps for branch in ensemble.branches),
                        default=0,
                    ),
                    min_neural_steps=max(
                        self.config.recurrence.min_steps,
                        self.config.branches.isolation_steps,
                    ),
                    active_branches=len(ensemble.active()),
                    total_branches=len(ensemble.branches),
                    residual=before_residual,
                    residual_delta=max(
                        -1.0,
                        min(1.0, previous_residual - before_residual),
                    ),
                    verifier_score=previous_verifier_score,
                    verifier_delta=previous_verifier_delta,
                    disagreement=before_disagreement,
                    uncertainty=self._policy_uncertainty(value_policy.bucket),
                    budget_remaining_fraction=max(
                        0.0,
                        min(1.0, remaining_fraction),
                    ),
                    has_memory=has_memory,
                    has_evidence=has_evidence,
                    has_verifier=verifier is not None and self.tokenizer is not None,
                    has_savepoint=any(branch.savepoint is not None for branch in ensemble.branches),
                    can_execute=(
                        external_execution_offer is not None
                        and OperationKind.EXECUTE not in selected_actions
                    ),
                    answer_verified=(
                        previous_verifier_score is not None
                        and previous_verifier_score >= 1.0 - 1e-9
                    ),
                    irreducible_uncertainty=(
                        action_index + 1 >= self.config.recurrence.max_steps
                        and previous_verifier_score is not None
                        and previous_verifier_score <= 1e-9
                        and not has_memory
                        and not has_evidence
                    ),
                    pending_constraint_action=pending_constraint_action,
                    omitted_action_count=omitted_action_count,
                    previously_selected=tuple(selected_actions),
                )
                if continuation_pending:
                    from core.brain.llm.latent_cortex.action_continuation import (
                        capture_action_opportunity_continuation,
                        restore_action_opportunity_continuation,
                    )

                    if action_index != 0 or selected_actions or omitted_action_count:
                        raise RuntimeError(
                            "action continuation missed the first action opportunity"
                        )
                    if action_continuation_runner_state is None:
                        raise RuntimeError("action continuation runner state is unavailable")

                    def capture_current_action_frame(
                        *,
                        current_signal: CognitiveStateSignal,
                        current_executors: tuple[OperationKind, ...],
                        current_action_index: int,
                        current_op_index: int,
                        current_branch_index: int,
                        current_layer_end: int,
                    ):
                        return capture_action_opportunity_continuation(
                            ensemble=ensemble,
                            cache=cache,
                            budget=budget,
                            episode_context_items=episode_context_items,
                            action_policy_evidence=action_policy_evidence,
                            state_signal=current_signal,
                            active_action_executors=current_executors,
                            durable_state=action_continuation_runner_state["durable_state"],
                            rng_state=action_continuation_runner_state["rng_state"],
                            episode_step=current_action_index,
                            schedule_step=current_op_index,
                            branch_id=f"branch-{current_branch_index}",
                            layer_index=max(0, current_layer_end - 1),
                            kv_position=max(
                                self._cache_context_tokens(cache, layer_index)
                                for layer_index in range(len(cache))
                            ),
                        )

                    capture_frame_kwargs = {
                        "current_signal": state_signal,
                        "current_executors": active_action_executors,
                        "current_action_index": action_index,
                        "current_op_index": op_index,
                        "current_branch_index": prospective_target.index,
                        "current_layer_end": int(op.end),
                    }
                    rollback_continuation = capture_current_action_frame(**capture_frame_kwargs)
                    active_action_continuation = rollback_continuation
                    if action_continuation_restore is not None:
                        try:
                            restore_action_opportunity_continuation(
                                action_continuation_restore,
                                ensemble=ensemble,
                                cache=cache,
                                budget=budget,
                            )
                            active_action_continuation = capture_current_action_frame(
                                **capture_frame_kwargs
                            )
                            if (
                                active_action_continuation.state_components
                                != action_continuation_restore.state_components
                            ):
                                differing = sorted(
                                    name
                                    for name, value in (
                                        active_action_continuation.state_components.items()
                                    )
                                    if value
                                    != action_continuation_restore.state_components.get(name)
                                )
                                raise RuntimeError(
                                    "restored action continuation differs before action:"
                                    + ",".join(differing)
                                )
                        except Exception as restore_exc:  # noqa: BLE001 - transactional rollback
                            try:
                                restore_action_opportunity_continuation(
                                    rollback_continuation,
                                    ensemble=ensemble,
                                    cache=cache,
                                    budget=budget,
                                )
                                if (
                                    capture_current_action_frame(
                                        **capture_frame_kwargs
                                    ).state_components
                                    != rollback_continuation.state_components
                                ):
                                    raise RuntimeError(
                                        "action continuation rollback verification failed"
                                    )
                            except Exception as rollback_exc:  # noqa: BLE001 - fatal ambiguity
                                raise UnknownActionStateApplicationError(
                                    {
                                        "operation_id": "engine-first-action-restore",
                                        "arm": intervention_arm,
                                        "worker_pid": None,
                                        "request_sha256": "",
                                        "snapshot_sha256": "",
                                    }
                                ) from rollback_exc
                            raise restore_exc
                        if action_continuation_restore_verified is not None:
                            from core.brain.llm.latent_cortex.campaign_journal import (
                                canonical_json_bytes,
                            )

                            action_continuation_restore_verified(
                                hashlib.sha256(
                                    canonical_json_bytes(
                                        active_action_continuation.state_components
                                    )
                                ).hexdigest()
                            )
                    continuation_pending = False
                    if action_continuation_capture is not None:
                        action_continuation_capture(active_action_continuation)
                    if action_continuation_capture_only:
                        raise _ActionContinuationCapturedError
                if intervention_pending:
                    from core.brain.llm.latent_cortex.action_intervention import (
                        CONTROL_ARM,
                    )

                    pre_components = (
                        active_action_continuation.state_components
                        if active_action_continuation is not None
                        else self._action_intervention_state_components(
                            ensemble=ensemble,
                            budget=budget,
                            episode_context_items=episode_context_items,
                            action_policy_evidence=action_policy_evidence,
                            state_signal=state_signal,
                            active_action_executors=active_action_executors,
                            action_intervention=action_intervention,
                        )
                    )
                    pre_state_sha256 = self._canonical_sha256(pre_components)
                    pre_kv_sha256 = pre_components["kv_cache_sha256"]
                    intervention_authority = action_intervention["authority_payload"]
                    if (
                        pre_components != intervention_authority["starting_state_components"]
                        or pre_state_sha256 != intervention_authority["expected_pre_state_sha256"]
                        or pre_kv_sha256 != intervention_authority["expected_pre_kv_sha256"]
                    ):
                        raise ValueError("action intervention resident starting state differs")
                    intervention_runtime = {
                        "pre_state_components": pre_components,
                        "pre_state_sha256": pre_state_sha256,
                        "pre_kv_sha256": pre_kv_sha256,
                        "decision_sha256": "",
                    }
                    intervention_pending = False
                    if intervention_arm == CONTROL_ARM:
                        active_action_executors = tuple(
                            action
                            for action in active_action_executors
                            if action != intervention_action
                        )
                        omitted_action_count += 1
                        action_index += 1
                        post_signal = replace(
                            state_signal,
                            step_index=min(state_signal.max_steps, action_index),
                            omitted_action_count=omitted_action_count,
                            previously_selected=tuple(selected_actions),
                        )
                        if active_action_continuation is not None:
                            post_components = capture_action_opportunity_continuation(
                                ensemble=ensemble,
                                cache=cache,
                                budget=budget,
                                episode_context_items=episode_context_items,
                                action_policy_evidence=action_policy_evidence,
                                state_signal=post_signal,
                                active_action_executors=active_action_executors,
                                durable_state=action_continuation_runner_state["durable_state"],
                                rng_state=action_continuation_runner_state["rng_state"],
                                episode_step=action_index,
                                schedule_step=op_index,
                                branch_id=f"branch-{prospective_target.index}",
                                layer_index=max(0, int(op.end) - 1),
                                kv_position=max(
                                    self._cache_context_tokens(cache, layer_index)
                                    for layer_index in range(len(cache))
                                ),
                            ).state_components
                        else:
                            post_components = self._action_intervention_state_components(
                                ensemble=ensemble,
                                budget=budget,
                                episode_context_items=episode_context_items,
                                action_policy_evidence=action_policy_evidence,
                                state_signal=post_signal,
                                active_action_executors=active_action_executors,
                                action_intervention=action_intervention,
                            )
                        intervention_runtime.update(
                            {
                                "post_state_components": post_components,
                                "post_state_sha256": self._canonical_sha256(post_components),
                                "post_kv_sha256": post_components["kv_cache_sha256"],
                            }
                        )
                        previous_residual = before_residual
                        continue
                    decision = value_policy.choose_forced(
                        state_signal,
                        executors=active_action_executors,
                        action=intervention_action,
                    )
                    intervention_runtime["decision_sha256"] = decision["decision_sha256"]
                else:
                    decision = value_policy.choose(
                        state_signal,
                        executors=active_action_executors,
                    )
                from core.brain.llm.latent_cortex.stop_gate import StopContext

                stop_context = StopContext(
                    action_step=state_signal.step_index,
                    max_steps=self.config.recurrence.max_steps,
                    policy_uncertainty=state_signal.uncertainty,
                    verifier_score=state_signal.verifier_score,
                    verifier_delta=state_signal.verifier_delta,
                    expected_gain_lcb=float(decision["evidence"]["gain_used"]),
                    expected_cost_ucb=float(decision["evidence"]["cost_used"]),
                    quality_measured=update_gate.mode == "learned",
                    evoc_measured=bool(decision["evidence"]["measured"]),
                    budget_remaining_fraction=state_signal.budget_remaining_fraction,
                )
                action = OperationKind(decision["action"])
                spent_before = int(budget.spent_layer_apps)
                action_started_monotonic = time.monotonic()
                outcome = "completed"
                affected_branches = 0
                probe_score: float | None = None
                accepted_verifier_score = previous_verifier_score
                target_runtime_snapshot = ensemble.snapshot_branch_runtime(prospective_target)
                attempt_parent_kv_boundary_sha256 = prospective_target.kv_boundary_sha256
                constraint_application: dict[str, Any] | None = None
                constraint_attempt: dict[str, Any] | None = None
                constraint_pre_state = None
                verification = {
                    "target_branch": None,
                    "observation": {},
                    "decision": "not_run",
                    "restored": False,
                    "attempt_parent_state_sha256": "",
                    "constraint_input_state_sha256": "",
                    "candidate_state_sha256": "",
                    "restore_target_state_sha256": "",
                    "kv_boundary_before_sha256": "",
                    "kv_boundary_after_sha256": "",
                    "branch_step_before": None,
                    "branch_step_after": None,
                }
                tree_search_handled = False
                if (
                    action is OperationKind.BRANCH
                    and latent_tree_config.mode != LATENT_TREE_DISABLED
                ):
                    bounded_observer = getattr(verifier, "observe_with_bounds", None)
                    probe_cost = self._verifier_probe_layer_apps(bridge_tokens)
                    if verifier is None or self.tokenizer is None:
                        receipt.latent_tree_search = build_empty_latent_tree_receipt(
                            episode_id=receipt.episode_id,
                            objective_sha256=receipt.input_tokens_sha256,
                            config=latent_tree_config,
                            status="unavailable",
                            reason="admitted_verifier_unavailable",
                        )
                    elif not callable(bounded_observer):
                        receipt.latent_tree_search = build_empty_latent_tree_receipt(
                            episode_id=receipt.episode_id,
                            objective_sha256=receipt.input_tokens_sha256,
                            config=latent_tree_config,
                            status="unavailable",
                            reason="bounded_verifier_observation_unavailable",
                        )
                    elif probe_cost + safety_reserve > budget.remaining_layer_apps:
                        receipt.latent_tree_search = build_empty_latent_tree_receipt(
                            episode_id=receipt.episode_id,
                            objective_sha256=receipt.input_tokens_sha256,
                            config=latent_tree_config,
                            status="unavailable",
                            reason="root_probe_budget_unavailable",
                        )
                    else:
                        from core.brain.llm.latent_cortex.loop_core import canonical_sha256

                        tree_root_snapshot = ensemble.snapshot_ensemble_runtime()
                        tree_root_boundaries = self._ensemble_snapshot_boundaries(
                            tree_root_snapshot
                        )
                        tree_root_state_sha256, tree_root_kv_sha256 = (
                            self._ensemble_snapshot_identity(tree_root_snapshot)
                        )
                        tree_target_index = prospective_target.index
                        from core.brain.llm.latent_cortex.resource_accounting import (
                            RESOURCE_COUNTERS,
                        )

                        root_resources_before = budget.resource_ledger.totals()
                        root_spent_before = int(budget.spent_layer_apps)
                        tree_probe_cache = self._episode_probe_cache
                        root_cache_key = ""
                        root_cache_hits_before = 0
                        root_cache_saved_before = 0
                        if tree_probe_cache is not None:
                            root_cache_key = tree_probe_cache.key(
                                prospective_target.workspace.seed_z,
                                prospective_target.z,
                                bridge_tokens,
                                self.config.verifier_probe_max_tokens,
                            )
                            root_cache_hits_before = tree_probe_cache.hits
                            root_cache_saved_before = tree_probe_cache.layer_apps_saved
                        root_probe = self._decode_probe(
                            prospective_target,
                            cache,
                            runner,
                            budget,
                            bridge_tokens=bridge_tokens,
                        )
                        root_observation = bounded_observer(self.tokenizer.decode(root_probe))
                        root_resources_after = budget.resource_ledger.totals()
                        root_probe_tokens = [int(token) for token in root_probe]
                        root_evaluation = {
                            "spent_layer_apps": (int(budget.spent_layer_apps) - root_spent_before),
                            "resource_delta": {
                                name: int(root_resources_after[name])
                                - int(root_resources_before[name])
                                for name in RESOURCE_COUNTERS
                            },
                            "probe_tokens_sha256": canonical_sha256(root_probe_tokens),
                            "probe_token_count": len(root_probe_tokens),
                            "target_branch": tree_target_index,
                            "probe_cache_hit": bool(
                                tree_probe_cache is not None
                                and tree_probe_cache.hits > root_cache_hits_before
                            ),
                            "probe_cache_key_sha256": root_cache_key,
                            "probe_cache_layer_apps_saved": (
                                0
                                if tree_probe_cache is None
                                else tree_probe_cache.layer_apps_saved - root_cache_saved_before
                            ),
                        }
                        authority_observation = (
                            prospective_target.verified_best_observation or root_observation
                        )
                        tree_actions = tuple(
                            candidate.value
                            for candidate in (
                                OperationKind.DECOMPOSE,
                                OperationKind.FALSIFY,
                                OperationKind.CHECK_ASSUMPTION,
                                OperationKind.SIMULATE,
                                OperationKind.FORMALIZE,
                                OperationKind.BLIND_RESOLVE,
                            )
                            if candidate in action_controls
                        )

                        def restore_tree_snapshot(snapshot):
                            ensemble.restore_ensemble_runtime(snapshot)
                            restored = ensemble.snapshot_ensemble_runtime()
                            return self._ensemble_snapshot_identity(restored)

                        def recurrent_tree_call_inventory(
                            *,
                            _op_start: int = op.start,
                            _op_end: int = op.end,
                        ) -> list[int]:
                            return [
                                int(row["ordinal"])
                                for row in runner.kv_bound_receipt()["calls"]
                                if row["persist"] is False
                                and row["start"] == _op_start
                                and row["end"] == _op_end
                            ]

                        def expand_tree_state(
                            action_name: str,
                            parent_index: int,
                            child_index: int,
                            *,
                            _action_step: int = action_index,
                            _bounded_observer=bounded_observer,
                            _op_start: int = op.start,
                            _op_end: int = op.end,
                            _op_alpha: float = op.alpha,
                            _probe_cost: int = probe_cost,
                            _stop_context=stop_context,
                            _target_index: int = tree_target_index,
                        ) -> dict[str, Any]:
                            tree_action = OperationKind(action_name)
                            control = action_controls.get(tree_action)
                            if control is None:
                                raise RuntimeError("tree_action_control_unavailable")
                            recurrent_ordinals_before = set(recurrent_tree_call_inventory())
                            operator_rows = ensemble.apply_cognitive_operators(
                                control,
                                action=f"latent_tree:{action_name}",
                                action_step=_action_step,
                                budget=budget,
                            )
                            admitted = ensemble.step_all(
                                runner,
                                cache,
                                _op_start,
                                _op_end,
                                budget=budget,
                                alpha_override=_op_alpha,
                                reserve_layer_apps=safety_reserve,
                                stop_context=_stop_context,
                                transaction_purpose=f"latent_tree:{action_name}",
                            )
                            if not admitted:
                                raise RuntimeError("tree_transition_budget_refused")
                            if _probe_cost + safety_reserve > budget.remaining_layer_apps:
                                raise RuntimeError("tree_probe_budget_refused")
                            target = next(
                                branch
                                for branch in ensemble.branches
                                if branch.index == _target_index
                            )
                            probe_cache = self._episode_probe_cache
                            probe_cache_key = ""
                            probe_cache_hits_before = 0
                            probe_cache_saved_before = 0
                            if probe_cache is not None:
                                probe_cache_key = probe_cache.key(
                                    target.workspace.seed_z,
                                    target.z,
                                    bridge_tokens,
                                    self.config.verifier_probe_max_tokens,
                                )
                                probe_cache_hits_before = probe_cache.hits
                                probe_cache_saved_before = probe_cache.layer_apps_saved
                            probe = self._decode_probe(
                                target,
                                cache,
                                runner,
                                budget,
                                bridge_tokens=bridge_tokens,
                            )
                            observation = _bounded_observer(self.tokenizer.decode(probe))
                            probe_tokens = [int(token) for token in probe]
                            child_snapshot = ensemble.snapshot_ensemble_runtime()
                            child_boundaries = self._ensemble_snapshot_boundaries(child_snapshot)
                            state_sha256, kv_sha256 = self._ensemble_snapshot_identity(
                                child_snapshot
                            )
                            recurrent_kv_call_ordinals = sorted(
                                set(recurrent_tree_call_inventory()) - recurrent_ordinals_before
                            )
                            transition_sha256 = canonical_sha256(
                                {
                                    "episode_id": receipt.episode_id,
                                    "objective_sha256": receipt.input_tokens_sha256,
                                    "action_step": _action_step,
                                    "parent_index": parent_index,
                                    "child_index": child_index,
                                    "action": action_name,
                                    "state_sha256": state_sha256,
                                    "kv_boundary_sha256": kv_sha256,
                                    "target_branch": _target_index,
                                    "operator_receipts": [
                                        canonical_sha256(row) for row in operator_rows
                                    ],
                                }
                            )
                            return {
                                "snapshot": child_snapshot,
                                "state_sha256": state_sha256,
                                "kv_boundary_sha256": kv_sha256,
                                "observation": observation,
                                "transition_sha256": transition_sha256,
                                "target_branch": _target_index,
                                "probe_tokens_sha256": canonical_sha256(probe_tokens),
                                "probe_token_count": len(probe_tokens),
                                "branch_boundaries": child_boundaries,
                                "probe_cache_hit": bool(
                                    probe_cache is not None
                                    and probe_cache.hits > probe_cache_hits_before
                                ),
                                "probe_cache_key_sha256": probe_cache_key,
                                "probe_cache_layer_apps_saved": (
                                    0
                                    if probe_cache is None
                                    else probe_cache.layer_apps_saved - probe_cache_saved_before
                                ),
                                "recurrent_kv_call_ordinals": recurrent_kv_call_ordinals,
                            }

                        transaction = run_latent_tree_search(
                            episode_id=receipt.episode_id,
                            objective_sha256=receipt.input_tokens_sha256,
                            action_step=action_index,
                            root_snapshot=tree_root_snapshot,
                            root_state_sha256=tree_root_state_sha256,
                            root_kv_boundary_sha256=tree_root_kv_sha256,
                            root_branch_boundaries=tree_root_boundaries,
                            root_observation=root_observation,
                            root_evaluation=root_evaluation,
                            authority_observation=authority_observation,
                            actions=tree_actions,
                            config=latent_tree_config,
                            budget=budget,
                            restore_snapshot=restore_tree_snapshot,
                            expand=expand_tree_state,
                            recurrent_call_inventory=recurrent_tree_call_inventory,
                            cancel_check=cancel_check,
                        )
                        append_latent_tree_transaction(
                            receipt.latent_tree_search,
                            transaction,
                        )
                        tree_search_handled = True
                        affected_branches = len(ensemble.branches)
                        outcome = f"latent_tree_{transaction['status']}"
                        if transaction["status"] == "committed":
                            winner_node = int(transaction["winner_node"])
                            winner = transaction["nodes"][winner_node]
                            winner_observation = winner["observation"]
                            winner_target_boundary = next(
                                boundary
                                for boundary in winner["branch_boundaries"]
                                if boundary["index"] == tree_target_index
                            )
                            root_target_boundary = next(
                                boundary
                                for boundary in tree_root_boundaries
                                if boundary["index"] == tree_target_index
                            )
                            from core.brain.llm.latent_cortex.verified_best import (
                                validate_observation,
                            )

                            committed_observation = validate_observation(winner_observation)
                            raw_committed_observation = {
                                key: committed_observation[key]
                                for key in (
                                    "schema",
                                    "score",
                                    "lower_bound",
                                    "upper_bound",
                                    "sample_count",
                                    "basis",
                                    "independent",
                                    "evidence_sha256",
                                )
                            }
                            target = next(
                                branch
                                for branch in ensemble.branches
                                if branch.index == tree_target_index
                            )
                            observation, best_decision, best_restored = (
                                ensemble.observe_verified_best(
                                    target,
                                    raw_committed_observation,
                                    action_step=action_index,
                                    restore_target_state_sha256=tensor_sha256(
                                        target_runtime_snapshot["z"]
                                    ),
                                    budget=budget,
                                )
                            )
                            if best_decision != "promote" or best_restored:
                                raise RuntimeError(
                                    "latent tree authority diverged after committed search"
                                )
                            probe_score = observation.score
                            accepted_verifier_score = observation.score
                            verification = {
                                "target_branch": tree_target_index,
                                "observation": observation.to_dict(),
                                "decision": best_decision,
                                "restored": False,
                                "attempt_parent_state_sha256": tree_root_state_sha256,
                                "constraint_input_state_sha256": "",
                                "candidate_state_sha256": winner_target_boundary["state_sha256"],
                                "restore_target_state_sha256": "",
                                "kv_boundary_before_sha256": root_target_boundary[
                                    "kv_boundary_sha256"
                                ],
                                "kv_boundary_after_sha256": winner_target_boundary[
                                    "kv_boundary_sha256"
                                ],
                                "branch_step_before": target_runtime_snapshot["steps"],
                                "branch_step_after": target.steps,
                            }

                if action is OperationKind.REGENERATE_FROM_PREFIX:
                    affected_branches += ensemble.revert_all_to_savepoint()
                if (
                    action
                    in {
                        OperationKind.SEARCH_MEMORY,
                        OperationKind.RETRIEVE_EVIDENCE,
                    }
                    and not tree_search_handled
                ):
                    focus_receipts = ensemble.apply_context_focus(
                        action=action,
                        action_step=action_index,
                        budget=budget,
                    )
                    context_focus_trace.extend(focus_receipts)
                    affected_branches = max(
                        affected_branches,
                        len(focus_receipts),
                    )
                    outcome = (
                        "memory_context_focused"
                        if action is OperationKind.SEARCH_MEMORY
                        else "evidence_context_focused"
                    )
                if action in {
                    OperationKind.FALSIFY,
                    OperationKind.CHECK_ASSUMPTION,
                }:
                    constraint_pre_state = prospective_target.z
                    constrained_state, constraint_application = transient_constraints.apply_next(
                        prospective_target.z,
                        branch_index=prospective_target.index,
                        action=action.value,
                        action_step=action_index,
                        branch_step=prospective_target.steps,
                        kv_boundary_sha256=(prospective_target.kv_boundary_sha256),
                        budget=budget,
                    )
                    if constraint_application is not None:
                        prospective_target.z = mx.array(constrained_state)
                        prospective_target.z = (
                            prospective_target.workspace.restore_context_evidence(
                                prospective_target.z
                            )
                        )
                        prospective_target.workspace.update(prospective_target.z)
                        receipt.flag("transient_negative_constraint_applied")
                if action in _ACTION_CONTROL_TEXT and not tree_search_handled:
                    operator_receipts = ensemble.apply_cognitive_operators(
                        action_controls[action],
                        action=action.value,
                        action_step=action_index,
                        budget=budget,
                    )
                    cognitive_operator_trace.extend(operator_receipts)
                    affected_branches = max(
                        affected_branches,
                        len(operator_receipts),
                    )
                if (
                    action
                    in {
                        OperationKind.DECOMPOSE,
                        OperationKind.BLIND_RESOLVE,
                        OperationKind.BRANCH,
                        OperationKind.SEARCH_MEMORY,
                        OperationKind.RETRIEVE_EVIDENCE,
                        OperationKind.SIMULATE,
                        OperationKind.FALSIFY,
                        OperationKind.CHECK_ASSUMPTION,
                        OperationKind.REGENERATE_FROM_PREFIX,
                        OperationKind.FORMALIZE,
                    }
                    and not tree_search_handled
                ):
                    try:
                        admitted = ensemble.step_all(
                            runner,
                            cache,
                            op.start,
                            op.end,
                            budget=budget,
                            alpha_override=op.alpha,
                            reserve_layer_apps=safety_reserve,
                            stop_context=stop_context,
                            transaction_purpose=action.value,
                        )
                    except Exception:
                        if constraint_application is not None:
                            if (
                                prospective_target.steps
                                == constraint_application["branch_step_before"]
                            ):
                                prospective_target.z = constraint_pre_state
                                prospective_target.workspace.update(constraint_pre_state)
                                transient_constraints.rollback_application(
                                    reservation_id=constraint_application["reservation_id"],
                                    restored_state=constraint_pre_state,
                                    branch_step_after=(prospective_target.steps),
                                    kv_boundary_after_sha256=(
                                        prospective_target.kv_boundary_sha256
                                    ),
                                    reason="recurrence_failed",
                                )
                            else:
                                transient_constraints.abort_all()
                            constraint_application = None
                        raise
                    if not admitted:
                        if constraint_application is not None:
                            prospective_target.z = constraint_pre_state
                            prospective_target.workspace.update(constraint_pre_state)
                            transient_constraints.rollback_application(
                                reservation_id=constraint_application["reservation_id"],
                                restored_state=constraint_pre_state,
                                branch_step_after=prospective_target.steps,
                                kv_boundary_after_sha256=(prospective_target.kv_boundary_sha256),
                                reason="budget_refused",
                            )
                            constraint_application = None
                        recurrence_budget_limited = True
                        outcome = "budget_refused"
                    elif constraint_application is not None:
                        constraint_application = transient_constraints.commit_application(
                            reservation_id=constraint_application["reservation_id"],
                            branch_step_after=prospective_target.steps,
                            kv_boundary_after_sha256=(prospective_target.kv_boundary_sha256),
                            recurrence_state=prospective_target.z,
                        )
                elif action is OperationKind.COMPARE:
                    affected_branches = int(
                        ensemble.exchange_now(
                            sync_kind="controller_compare",
                            sync_id=f"controller-action:{action_index}",
                            budget=budget,
                        )
                    )
                    outcome = "branches_compared" if affected_branches else "comparison_unavailable"
                elif action is OperationKind.EXECUTE:
                    affected_branches = ensemble.halt_all(
                        "value_controller_external_execute",
                        budget=budget,
                    )
                    outcome = "external_execute_requested"
                elif action is OperationKind.BACKTRACK:
                    affected_branches = ensemble.revert_all_to_savepoint()
                    outcome = "state_restored" if affected_branches else "savepoint_unavailable"
                elif action is OperationKind.COMPRESS_STATE:
                    affected_branches = ensemble.compress_state(budget=budget)
                    outcome = "state_compressed"
                elif action is OperationKind.ANSWER:
                    affected_branches = ensemble.halt_all(
                        "value_controller_answer",
                        budget=budget,
                    )
                    outcome = "answer_selected"
                elif action is OperationKind.ABSTAIN:
                    affected_branches = ensemble.halt_all(
                        "value_controller_abstain",
                        budget=budget,
                    )
                    outcome = "abstention_selected"

                if (
                    not recurrence_budget_limited
                    and action
                    in {
                        OperationKind.FALSIFY,
                        OperationKind.CHECK_ASSUMPTION,
                        OperationKind.COMPARE,
                    }
                    and verifier is not None
                    and self.tokenizer is not None
                ):
                    probe_cost = self._verifier_probe_layer_apps(bridge_tokens)
                    if probe_cost + safety_reserve <= budget.remaining_layer_apps:
                        target = prospective_target
                        probe = self._decode_probe(
                            target,
                            cache,
                            runner,
                            budget,
                            bridge_tokens=bridge_tokens,
                        )
                        rendered_probe = self.tokenizer.decode(probe)
                        bounded_observer = getattr(
                            verifier,
                            "observe_with_bounds",
                            None,
                        )
                        raw_observation = (
                            bounded_observer(rendered_probe)
                            if callable(bounded_observer)
                            else verifier(rendered_probe)
                        )
                        candidate_state_before_verification = target.z
                        incumbent_observation_before_verification = (
                            dict(target.verified_best_observation)
                            if target.verified_best_observation
                            else None
                        )
                        try:
                            (
                                observation,
                                best_decision,
                                best_restored,
                            ) = ensemble.observe_verified_best(
                                target,
                                raw_observation,
                                action_step=action_index,
                                restore_target_state_sha256=tensor_sha256(
                                    target_runtime_snapshot["z"]
                                ),
                                budget=budget,
                            )
                        except (TypeError, ValueError):
                            probe_score = None
                            outcome = "verifier_observation_invalid"
                        else:
                            probe_score = observation.score
                            verification = {
                                "target_branch": target.index,
                                "observation": observation.to_dict(),
                                "decision": best_decision,
                                "restored": best_restored,
                                "attempt_parent_state_sha256": tensor_sha256(
                                    target_runtime_snapshot["z"]
                                ),
                                "constraint_input_state_sha256": tensor_sha256(
                                    constraint_pre_state
                                ),
                                "candidate_state_sha256": tensor_sha256(
                                    candidate_state_before_verification
                                ),
                                "restore_target_state_sha256": (
                                    tensor_sha256(target_runtime_snapshot["z"])
                                    if best_decision == "reject_verified_failure"
                                    else ""
                                ),
                                "kv_boundary_before_sha256": (
                                    target_runtime_snapshot["kv_boundary_sha256"]
                                ),
                                "kv_boundary_after_sha256": target.kv_boundary_sha256,
                                "branch_step_before": target_runtime_snapshot["steps"],
                                "branch_step_after": target.steps,
                            }
                        if probe_score is None:
                            pass
                        elif best_decision == "reject_verified_failure":
                            ensemble.restore_branch_runtime(
                                target,
                                target_runtime_snapshot,
                                preserve_execution_traces=True,
                            )
                            ensemble.commit_verified_failure_restore(
                                target,
                                action_step=action_index,
                            )
                            verification["restored"] = True
                            accepted_verifier_score = previous_verifier_score
                            outcome = "verified_failure_reverted_exact_parent"
                        elif best_decision == "preserve_verified":
                            accepted_verifier_score = float(
                                target.verified_best_observation["score"]
                            )
                            outcome = "verified_best_preserved"
                        elif (
                            previous_verifier_score is not None
                            and probe_score < previous_verifier_score - 1e-9
                        ):
                            reverted = int(ensemble.revert_branch_to_savepoint(target))
                            outcome = f"verifier_regression_reverted_{reverted}"
                        else:
                            accepted_verifier_score = probe_score
                            if (
                                previous_verifier_score is None
                                or probe_score > previous_verifier_score
                            ):
                                ensemble.savepoint_branch(
                                    target,
                                    verified=True,
                                    authority="bounded_verifier_progress",
                                )
                                outcome = "verified_progress_saved"
                        if probe_score is not None:
                            followup = transient_constraints.observe_followup(
                                branch_index=target.index,
                                action_step=action_index,
                                observation=observation.to_dict(),
                            )
                            if followup is not None:
                                if constraint_application is not None and followup.get(
                                    "reservation_id"
                                ) == constraint_application.get("reservation_id"):
                                    constraint_application = followup
                                receipt.flag(
                                    "transient_negative_constraint_"
                                    + (
                                        "reduced_failure"
                                        if followup["failure_reduced"]
                                        else "repeated_failure"
                                        if followup["failure_repeated"]
                                        else "followup_inconclusive"
                                    )
                                )
                            incumbent_lower = (
                                float(incumbent_observation_before_verification["lower_bound"])
                                if incumbent_observation_before_verification
                                and incumbent_observation_before_verification.get("authoritative")
                                is True
                                else None
                            )
                            verified_failure = (
                                action
                                in {
                                    OperationKind.FALSIFY,
                                    OperationKind.CHECK_ASSUMPTION,
                                }
                                and observation.authoritative
                                and (
                                    (
                                        observation.basis
                                        in {
                                            "deterministic_exact",
                                            "calibrated_interval",
                                        }
                                        and observation.upper_bound <= 1e-9
                                    )
                                    or (
                                        incumbent_lower is not None
                                        and observation.upper_bound + 1e-9 < incumbent_lower
                                    )
                                )
                            )
                            if verified_failure:
                                preflight_sha256 = str(
                                    receipt.verifier_preflight.get(
                                        "receipt_sha256",
                                        "",
                                    )
                                )
                                constraint_probe_apps = self._verifier_probe_layer_apps(
                                    bridge_tokens,
                                    count=(3 * transient_constraint_config.replicates),
                                )
                                constraint_recovery_apps = len(
                                    ensemble.active() or ensemble.branches
                                ) * self.config.workspace.n_slots * max(
                                    0, op.end - op.start
                                ) + self._verifier_probe_layer_apps(bridge_tokens)
                                constraint_recovery_wall_reserve_s = max(
                                    0.25,
                                    time.monotonic() - action_started_monotonic,
                                )
                                constraint_evaluator = None
                                constraint_unavailable_reason = ""
                                if (
                                    len(transient_verifier_policy_sha256) != 64
                                    or len(preflight_sha256) != 64
                                ):
                                    constraint_unavailable_reason = (
                                        "verifier_authority_commitment_unavailable"
                                    )
                                elif action_index + 1 >= self.config.recurrence.max_steps:
                                    constraint_unavailable_reason = (
                                        "constraint_recovery_action_budget_unavailable"
                                    )
                                elif not future_window_turn:
                                    constraint_unavailable_reason = (
                                        "constraint_recovery_schedule_turn_unavailable"
                                    )
                                elif decision["mode"] == "campaign_forced":
                                    constraint_unavailable_reason = (
                                        "constraint_recovery_intervention_arm_isolation"
                                    )
                                elif (
                                    constraint_probe_apps
                                    + constraint_recovery_apps
                                    + safety_reserve
                                    > budget.remaining_layer_apps
                                ):
                                    constraint_unavailable_reason = (
                                        "matched_counterfactual_probe_budget_unavailable"
                                    )
                                elif budget.remaining_wall_s <= constraint_recovery_wall_reserve_s:
                                    constraint_unavailable_reason = (
                                        "constraint_recovery_wall_budget_unavailable"
                                    )
                                else:
                                    raw_constraint_evaluator = self._counterfactual_probe_evaluator(
                                        branch=target,
                                        cache=cache,
                                        runner=runner,
                                        budget=budget,
                                        bridge_tokens=bridge_tokens,
                                        verifier=verifier,
                                    )
                                    if raw_constraint_evaluator is not None:

                                        def constraint_evaluator(
                                            label,
                                            state,
                                            replicate,
                                            _evaluator=raw_constraint_evaluator,
                                            _wall_reserve_s=constraint_recovery_wall_reserve_s,
                                        ):
                                            result = _evaluator(
                                                label,
                                                state,
                                                replicate,
                                            )
                                            if budget.remaining_wall_s <= _wall_reserve_s:
                                                raise RuntimeError(
                                                    "constraint recovery wall reserve consumed"
                                                )
                                            return result

                                if (
                                    len(transient_verifier_policy_sha256) == 64
                                    and len(preflight_sha256) == 64
                                ):
                                    constraint_attempt = (
                                        transient_constraints.consider_verified_failure(
                                            parent_state=target.z,
                                            failed_state=(candidate_state_before_verification),
                                            branch_index=target.index,
                                            source_action=action.value,
                                            action_step=action_index,
                                            source_kv_boundary_sha256=(
                                                attempt_parent_kv_boundary_sha256
                                            ),
                                            observation=observation.to_dict(),
                                            incumbent_observation=(
                                                incumbent_observation_before_verification
                                            ),
                                            verifier_policy_sha256=(
                                                transient_verifier_policy_sha256
                                            ),
                                            verifier_preflight_sha256=(preflight_sha256),
                                            evaluate=constraint_evaluator,
                                            evaluation_unavailable_reason=(
                                                constraint_unavailable_reason
                                            ),
                                            budget=budget,
                                        )
                                    )
                                    receipt.flag(
                                        "transient_negative_constraint_"
                                        + constraint_attempt["status"]
                                    )
                                else:
                                    receipt.flag(
                                        "transient_negative_constraint_"
                                        "verifier_authority_unavailable"
                                    )
                    else:
                        outcome = "verifier_probe_budget_refused"

                if decision["mode"] == "campaign_forced":
                    active_action_executors = tuple(
                        candidate
                        for candidate in active_action_executors
                        if candidate != intervention_action
                    )
                after_residual = self._mean_latest_residual(ensemble)
                after_disagreement = ensemble.disagreement(budget=budget)
                if decision["mode"] == "campaign_forced":
                    post_signal = replace(
                        state_signal,
                        step_index=min(
                            state_signal.max_steps,
                            state_signal.step_index + 1,
                        ),
                        neural_steps=max(
                            (branch.steps for branch in ensemble.branches),
                            default=0,
                        ),
                        active_branches=len(ensemble.active()),
                        residual=after_residual,
                        residual_delta=max(
                            -1.0,
                            min(1.0, before_residual - after_residual),
                        ),
                        verifier_score=probe_score,
                        verifier_delta=(
                            probe_score - previous_verifier_score
                            if probe_score is not None and previous_verifier_score is not None
                            else None
                        ),
                        disagreement=after_disagreement,
                        budget_remaining_fraction=max(
                            0.0,
                            min(
                                1.0,
                                budget.remaining_layer_apps / max(1, budget.max_layer_apps),
                            ),
                        ),
                        previously_selected=tuple([*selected_actions, action]),
                    )
                    if active_action_continuation is not None:
                        post_components = capture_action_opportunity_continuation(
                            ensemble=ensemble,
                            cache=cache,
                            budget=budget,
                            episode_context_items=episode_context_items,
                            action_policy_evidence=action_policy_evidence,
                            state_signal=post_signal,
                            active_action_executors=active_action_executors,
                            durable_state=action_continuation_runner_state["durable_state"],
                            rng_state=action_continuation_runner_state["rng_state"],
                            episode_step=state_signal.step_index + 1,
                            schedule_step=op_index,
                            branch_id=f"branch-{prospective_target.index}",
                            layer_index=max(0, int(op.end) - 1),
                            kv_position=max(
                                self._cache_context_tokens(cache, layer_index)
                                for layer_index in range(len(cache))
                            ),
                        ).state_components
                    else:
                        post_components = self._action_intervention_state_components(
                            ensemble=ensemble,
                            budget=budget,
                            episode_context_items=episode_context_items,
                            action_policy_evidence=action_policy_evidence,
                            state_signal=post_signal,
                            active_action_executors=active_action_executors,
                            action_intervention=action_intervention,
                        )
                    intervention_runtime.update(
                        {
                            "post_state_components": post_components,
                            "post_state_sha256": self._canonical_sha256(post_components),
                            "post_kv_sha256": post_components["kv_cache_sha256"],
                        }
                    )
                # The receipt carries eight-decimal public state. Reward
                # metrics must be derived from that same state or a rounding
                # boundary can make an honestly emitted transition fail its
                # independent reconstruction.
                public_before_residual = round(before_residual, 8)
                public_before_disagreement = round(before_disagreement, 8)
                public_after_residual = round(after_residual, 8)
                public_after_disagreement = round(after_disagreement, 8)
                checked = previous_verifier_score is not None and probe_score is not None
                verified_delta = probe_score - previous_verifier_score if checked else 0.0
                before_uncertainty = max(
                    public_before_residual,
                    public_before_disagreement,
                    1.0 - previous_verifier_score if previous_verifier_score is not None else 1.0,
                )
                after_uncertainty = max(
                    public_after_residual,
                    public_after_disagreement,
                    1.0 - probe_score if probe_score is not None else before_uncertainty,
                )
                cost_fraction = max(
                    0.0,
                    min(
                        1.0,
                        (int(budget.spent_layer_apps) - spent_before)
                        / max(1, budget.max_layer_apps),
                    ),
                )
                metrics = transition_reward(
                    verified_delta=max(-1.0, min(1.0, verified_delta)),
                    information_gain=max(
                        -1.0,
                        min(1.0, before_uncertainty - after_uncertainty),
                    ),
                    diversity_gain=max(
                        -1.0,
                        min(
                            1.0,
                            public_after_disagreement - public_before_disagreement,
                        ),
                    ),
                    unsupported_confidence=(
                        max(0.0, min(1.0, -verified_delta)) if checked else 0.0
                    ),
                    cost=cost_fraction,
                )
                transition = {
                    "schema": ACTION_TRANSITION_SCHEMA,
                    "bucket": value_policy.bucket,
                    "snapshot_sha256": action_policy_evidence["snapshot_sha256"],
                    "decision_sha256": decision["decision_sha256"],
                    "step_index": action_index,
                    "action": action.value,
                    "mode": decision["mode"],
                    "outcome": outcome,
                    "checked": checked,
                    "metrics": metrics,
                }
                receipt.cognitive_action_trace.append(
                    {
                        "decision": decision,
                        "transition": transition,
                        "state_signal": state_signal.to_dict(),
                        "state_before": {
                            "residual": public_before_residual,
                            "disagreement": public_before_disagreement,
                            "verifier_score": previous_verifier_score,
                            "budget_remaining_fraction": round(
                                state_signal.budget_remaining_fraction,
                                8,
                            ),
                        },
                        "state_after": {
                            "residual": public_after_residual,
                            "disagreement": public_after_disagreement,
                            "verifier_score": accepted_verifier_score,
                            "observed_verifier_score": probe_score,
                        },
                        "affected_branches": affected_branches,
                        "verification": verification,
                        "transient_constraint": (
                            dict(constraint_application)
                            if constraint_application is not None
                            else {}
                        ),
                        "transient_constraint_attempt": (
                            dict(constraint_attempt) if constraint_attempt is not None else {}
                        ),
                    }
                )
                selected_actions.append(action)
                action_index += 1
                previous_residual = before_residual
                if (
                    verification["target_branch"] is not None
                    and accepted_verifier_score is not None
                ):
                    branch_index = int(verification["target_branch"])
                    branch_verifier_scores[branch_index] = accepted_verifier_score
                    if checked:
                        branch_verifier_deltas[branch_index] = max(
                            -1.0,
                            min(1.0, verified_delta),
                        )
                    else:
                        branch_verifier_deltas.pop(branch_index, None)
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
                    cognitive_action=action.value,
                    cognitive_actions=action_index,
                )
                if recurrence_budget_limited:
                    break
            if recurrence_budget_limited:
                break
        if bytecode_events:
            receipt.bytecode_events = bytecode_events
        receipt.cognitive_operator_trace = cognitive_operator_trace
        receipt.context_focus_trace = context_focus_trace
        if continuation_pending:
            raise RuntimeError(
                "action continuation was not captured by the first recurrent action opportunity"
            )
        if action_intervention is not None:
            if intervention_pending or not intervention_runtime:
                raise ValueError("action intervention was not consumed by the recurrent schedule")
            from core.brain.llm.latent_cortex.action_intervention import (
                build_action_intervention_receipt_authority,
            )

            receipt.value_of_computation["calibration_intervention"] = (
                build_action_intervention_receipt_authority(
                    authority_payload=action_intervention["authority_payload"],
                    intervention_sha256=action_intervention["intervention_sha256"],
                    consumption_event=action_intervention_consumption,
                    execution_claim=action_intervention_execution_claim,
                    pre_state_components=intervention_runtime["pre_state_components"],
                    post_state_components=intervention_runtime["post_state_components"],
                    pre_state_sha256=intervention_runtime["pre_state_sha256"],
                    pre_kv_sha256=intervention_runtime["pre_kv_sha256"],
                    post_state_sha256=intervention_runtime["post_state_sha256"],
                    post_kv_sha256=intervention_runtime["post_kv_sha256"],
                    decision_sha256=intervention_runtime["decision_sha256"],
                    cognitive_action_trace=receipt.cognitive_action_trace,
                )
            )
        receipt.value_of_computation.update(
            {
                "executors": [action.value for action in action_executors],
                "actions_selected": len(receipt.cognitive_action_trace),
                "checked_transitions": sum(
                    int(row["transition"]["checked"]) for row in receipt.cognitive_action_trace
                ),
                "selected_actions": [action.value for action in selected_actions],
            }
        )
        if external_execution_offer is not None:
            from core.brain.llm.latent_cortex.external_execution import (
                build_external_execution_handoff,
            )

            receipt.external_execution_handoff = build_external_execution_handoff(
                external_execution_offer,
                receipt.cognitive_action_trace,
            )
        for branch in ensemble.branches:
            if not branch.halted:
                final, reverted, _source = ensemble.final_state(
                    branch,
                    budget=budget,
                )
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
            if branch.escape is not None:
                branch.escape.finalize()

        receipt.exchanges = ensemble.exchanges

        # ── Branch selection ─────────────────────────────────────────────
        uncertainty_scores = {
            branch.index: float(branch.uncertainty_trace[-1]["estimate"]["correctness_probability"])
            for branch in ensemble.branches
            if (branch.uncertainty_trace and branch.uncertainty_trace[-1]["estimate"]["supported"])
        }
        uncertainty_selection_eligible = len(uncertainty_scores) == len(ensemble.branches)
        from core.brain.llm.latent_cortex.mistake_locator import (
            process_branch_assessment,
        )

        process_assessments = {
            branch.index: process_branch_assessment(
                [dict(row) for row in branch.mistake_locator_trace],
                runtime=mistake_locator_runtime,
                domain=domain,
            )
            for branch in ensemble.branches
        }
        process_scores = {
            index: float(assessment["process_score"])
            for index, assessment in process_assessments.items()
            if assessment["selection_authority_admitted"] is True
        }
        process_selection_eligible = len(process_scores) == len(ensemble.branches)

        def select_without_task_verifier():
            if process_selection_eligible:
                return ensemble.select(score_fn=lambda branch: process_scores[branch.index])
            if uncertainty_selection_eligible:
                return ensemble.select(score_fn=lambda branch: uncertainty_scores[branch.index])
            return ensemble.select()

        winner = select_without_task_verifier()
        selection_basis = (
            "process_verifier"
            if process_selection_eligible
            else "neural_uncertainty"
            if uncertainty_selection_eligible
            else "convergence"
        )
        branch_probe_cost = self._verifier_probe_layer_apps(
            bridge_tokens,
            count=len(ensemble.branches),
        )
        branch_verifier_score: float | None = None
        branch_probe_texts: dict[int, str] = {}
        latent_optimization_probe_texts: dict[int, str] = {}
        research_oracle_candidates: dict[int, str] = {}
        verification_response_contract = str(
            getattr(information_verifier, "response_contract", "") or ""
        )
        blind_scores: dict[int, float] = {}
        if (
            pending_verifier is not None
            and self.tokenizer is not None
            and branch_probe_cost + safety_reserve <= budget.remaining_layer_apps
        ):
            for branch in ensemble.branches:
                probe = self._decode_probe(
                    branch,
                    cache,
                    runner,
                    budget,
                    bridge_tokens=bridge_tokens,
                )
                text = self._decode_public_text(probe, receipt=receipt)
                branch_probe_texts[branch.index] = text
            latent_optimization_probe_texts = dict(branch_probe_texts)
            valid_contract_branches: set[int] | None = None
            if "final_answer_v1" in {
                self.config.decode_contract,
                self.config.verifier_probe_contract,
            }:
                from core.brain.llm.latent_cortex.answer_contract import (
                    contract_answer_state,
                )

                original_branch_probe_texts = dict(branch_probe_texts)
                from core.brain.llm.latent_cortex.contract_repair import (
                    build_contract_repair_receipt,
                    parse_contract_repair_generation,
                    prepare_contract_repair_requests,
                )

                contract_repair_limit = (
                    self.config.local_repair_max_attempts if self.config.local_repair_enabled else 0
                )
                contract_repair_requests = prepare_contract_repair_requests(
                    branch_candidates=original_branch_probe_texts,
                    objective=verification_objective,
                    response_contract=verification_response_contract,
                    max_requests=contract_repair_limit,
                )
                contract_repairs: dict[str, dict[str, Any]] = {}
                contract_repair_failures: dict[str, str] = {}
                for request in contract_repair_requests:
                    request_id = str(request["request_id"])
                    try:
                        generated = self._fresh_verifier_generation(
                            str(request["prompt"]),
                            budget,
                            max_tokens=self.config.local_repair_max_tokens,
                            reserve_layer_apps=safety_reserve,
                        )
                        repaired_candidate = parse_contract_repair_generation(
                            generated["text"],
                            response_contract=verification_response_contract,
                        )
                        generated_context = generated["context"]
                        context = {
                            "prompt_sha256": request["prompt_sha256"],
                            "generated_token_count": generated_context["generated_token_count"],
                            "termination": generated_context["termination"],
                            "initial_cache_offsets": generated_context["initial_cache_offsets"],
                            "final_cache_offsets": generated_context["final_cache_offsets"],
                            "all_initial_offsets_zero": generated_context[
                                "all_initial_offsets_zero"
                            ],
                            "solver_context_imported": generated_context["solver_context_imported"],
                            "parameter_relation": generated_context["parameter_relation"],
                        }
                        contract_repairs[request_id] = {
                            "candidate": generated["text"],
                            "generation_context": context,
                        }
                        branch_probe_texts[int(request["branch"])] = repaired_candidate
                    except RuntimeError as exc:
                        contract_repair_failures[request_id] = (
                            "budget_unavailable"
                            if "budget" in str(exc).lower()
                            else "generation_failed"
                        )
                    except (ImportError, OSError, OverflowError, TypeError, ValueError) as exc:
                        receipt.flag(
                            "contract_repair_generation_rejected:"
                            f"{type(exc).__name__}:{str(exc)[:80]}"
                        )
                        contract_repair_failures[request_id] = "generation_contract_invalid"
                receipt.contract_repair = build_contract_repair_receipt(
                    branch_candidates=original_branch_probe_texts,
                    objective=verification_objective,
                    response_contract=verification_response_contract,
                    generated_repairs=contract_repairs,
                    execution_failures=contract_repair_failures,
                    max_requests=contract_repair_limit,
                    max_tokens=self.config.local_repair_max_tokens,
                )
                contract_states = {
                    index: contract_answer_state(text)
                    for index, text in sorted(branch_probe_texts.items())
                }
                if verification_response_contract:
                    from core.brain.llm.latent_cortex.task_verifiers import (
                        check_response_contract,
                    )

                    for index, state in contract_states.items():
                        response_state = check_response_contract(
                            branch_probe_texts[index],
                            verification_response_contract,
                        )
                        state["response_contract_valid"] = bool(response_state["valid"])
                        if state["valid"] and not response_state["valid"]:
                            state["valid"] = False
                            state["reason"] = "response_contract_invalid"
                receipt.branch_contract = [
                    {
                        "branch": index,
                        "marker_count": state["marker_count"],
                        "complete": state["complete"],
                        "valid": state["valid"],
                        "reason": str(state["reason"])[:120],
                        "response_contract_valid": state.get("response_contract_valid"),
                    }
                    for index, state in contract_states.items()
                ]
                valid_contract_branches = {
                    index for index, state in contract_states.items() if state["valid"]
                }
                if callable(
                    getattr(
                        pending_verifier,
                        "research_oracle_assessment",
                        None,
                    )
                ):
                    # The deployable blind review needs a complete comparable
                    # inventory and still fails closed below. The diagnostic
                    # oracle has exact hidden task truth, so retaining the
                    # contract-valid subset lets it answer whether ANY valid
                    # correct candidate was generated without granting that
                    # subset serving authority.
                    research_oracle_candidates = {
                        index: text
                        for index, text in branch_probe_texts.items()
                        if index in valid_contract_branches
                    }
                if not valid_contract_branches and self.config.allow_vanilla_fallback:
                    receipt.flag("branch_selection_no_contract_valid_candidate")
                    raise RuntimeError("no_contract_valid_latent_branch")
            contract_inventory_incomplete = valid_contract_branches is not None and len(
                valid_contract_branches
            ) != len(ensemble.branches)
            if contract_inventory_incomplete:
                receipt.flag("branch_selection_contract_inventory_incomplete")
                branch_probe_texts = {}
                verifier = None
                winner = select_without_task_verifier()
            else:
                if callable(
                    getattr(
                        pending_verifier,
                        "research_oracle_assessment",
                        None,
                    )
                ):
                    research_oracle_candidates = dict(branch_probe_texts)
                from core.brain.llm.latent_cortex.blind_review import (
                    run_decoy_balanced_review,
                )

                try:
                    (
                        blind_scores,
                        receipt.blind_review,
                        receipt.decoy_verification,
                    ) = run_decoy_balanced_review(
                        branch_probe_texts,
                        pending_verifier,
                        episode_id=receipt.episode_id,
                        objective_sha256=receipt.input_tokens_sha256,
                        isolation_receipt=ensemble.isolation_receipt(
                            runner.cache_discipline_receipt()
                        ),
                    )
                except Exception as exc:
                    receipt.flag(f"branch_decoy_review_failed:{type(exc).__name__}")
                    branch_probe_texts = {}
                    verifier = None
                    winner = select_without_task_verifier()
                else:
                    if receipt.decoy_verification["selection_admitted"]:
                        verifier = pending_verifier
                        if valid_contract_branches is not None:
                            if not valid_contract_branches:
                                receipt.flag("branch_selection_no_contract_valid_candidate")
                            else:
                                invalid_count = len(ensemble.branches) - len(
                                    valid_contract_branches
                                )
                                if invalid_count:
                                    receipt.flag(
                                        f"branch_selection_contract_rejected:{invalid_count}"
                                    )
                        winner = ensemble.select(
                            score_fn=lambda branch: _contract_admitted_branch_score(
                                branch.index,
                                blind_scores,
                                valid_contract_branches,
                            )
                        )
                        selection_basis = "task_verifier"
                        if math.isfinite(float(winner.score)):
                            branch_verifier_score = float(winner.score)
                    else:
                        receipt.flag("branch_verifier_decoy_calibration_failed")
                        verifier = None
                        winner = select_without_task_verifier()
        else:
            if pending_verifier is not None and self.tokenizer is not None:
                receipt.flag("branch_verifier_skipped_budget")
            verifier = None
            winner = select_without_task_verifier()

        # Localize branch disagreements while the exact probe inventory and
        # action lineage are both available. Repair generation is deferred
        # until the established verifier mesh has consumed its own budget.
        prepared_repairs: list[dict[str, Any]] = []
        generated_repairs: dict[str, dict[str, Any]] = {}
        if receipt.cognitive_operator_trace:
            from core.brain.llm.latent_cortex.structural_diversity import (
                build_structural_diversity_receipt,
            )

            receipt.structural_diversity = build_structural_diversity_receipt(
                n_branches=len(ensemble.branches),
                cognitive_slots=receipt.cognitive_slots,
                operator_trace=receipt.cognitive_operator_trace,
                action_trace=receipt.cognitive_action_trace,
                branch_isolation=ensemble.isolation_receipt(runner.cache_discipline_receipt()),
            )
            if receipt.structural_diversity.get("certified") is not True:
                receipt.flag("structural_diversity_unproven")
            from core.brain.llm.latent_cortex.disagreement_graph import (
                build_disagreement_graph_receipt,
                decompose_branch_candidates,
            )

            candidate_decompositions: dict[str, dict[str, Any]] = {}
            if len(branch_probe_texts) == len(ensemble.branches):
                try:
                    candidate_decompositions = decompose_branch_candidates(
                        branch_probe_texts,
                        objective=verification_objective,
                    )
                except (TypeError, ValueError) as exc:
                    # Probe text is untrusted model output. A malformed branch
                    # is negative evidence for structural diagnosis, not an
                    # infrastructure failure for the completed latent episode.
                    receipt.flag(f"branch_candidate_decomposition_invalid:{type(exc).__name__}")
            receipt.disagreement_graph = build_disagreement_graph_receipt(
                n_branches=len(ensemble.branches),
                operator_trace=receipt.cognitive_operator_trace,
                action_trace=receipt.cognitive_action_trace,
                structural_diversity=receipt.structural_diversity,
                candidate_decompositions=candidate_decompositions,
                blind_review=receipt.blind_review,
            )
            from core.brain.llm.latent_cortex.diagnostic_action_selector import (
                build_candidate_routes,
                build_diagnostic_action_selector_receipt,
            )

            candidate_routes = (
                build_candidate_routes(
                    branch_probe_texts,
                    objective=verification_objective,
                    candidate_decompositions=candidate_decompositions,
                )
                if candidate_decompositions
                else {}
            )
            receipt.diagnostic_action_selection = build_diagnostic_action_selector_receipt(
                disagreement_graph=receipt.disagreement_graph,
                candidate_routes=candidate_routes,
                action_policy_evidence=action_policy_evidence,
                value_policy=receipt.value_of_computation,
                action_trace=receipt.cognitive_action_trace,
            )
            from core.brain.llm.latent_cortex.local_repair import (
                build_local_repair_receipt,
                parse_local_repair_generation,
                prepare_local_repair_requests,
            )

            repair_limit = (
                self.config.local_repair_max_attempts if self.config.local_repair_enabled else 0
            )
            # ── the episode's commitments ────────────────────────────────
            #
            # Branch sampling is i.i.d., and i.i.d. sampling from a peaked
            # model re-derives its mode: measured here twice already, as
            # cos(pass1, pass2) = 0.9994 and as "collapse is cheapest". The
            # consequence is that best-of-N behaves like best-of-2 and the
            # repair generation that follows redraws from the SAME
            # distribution that produced the refuted answers.
            #
            # Committing each refuted branch turns the redraw into a draw
            # from the residual distribution: P(correct) goes from p* to
            # p*/(1 - m), which dominates for every N. Only branches the
            # task verifier actually refuted are committed — an unscored or
            # undecided branch has not been shown wrong, and excluding it
            # would remove an answer for not having been checked.
            episode_ratchet = self._build_episode_ratchet(
                objective=verification_objective,
                branch_texts=branch_probe_texts,
                blind_scores=blind_scores,
            )
            receipt.commitment_ratchet = episode_ratchet.receipt()
            # Coverage on the operator view, not only in a receipt nobody
            # opens: eight passes producing two distinct answers is the
            # condition that explains every flat RLC result, and it has been
            # invisible.
            try:
                from core.brain.llm.latent_cortex import commitment_telemetry

                commitment_telemetry.sample(
                    receipt.commitment_ratchet, passes=len(ensemble.branches)
                )
            except _LATENT_PHASE_ERRORS as exc:
                record_degradation(
                    "latent_cortex_engine",
                    exc,
                    severity="debug",
                    action="episode ran without commitment telemetry",
                )
            prepared_repairs = prepare_local_repair_requests(
                disagreement_graph=receipt.disagreement_graph,
                diagnostic_selection=receipt.diagnostic_action_selection,
                branch_candidates=branch_probe_texts,
                objective=verification_objective,
                max_requests=repair_limit,
                # Requirements only. Exclusions are enforced by
                # rejection below, not by naming them in the prompt.
                conditioning=episode_ratchet.conditioning_block(),
            )

        # SPARK-044: counterfactual generation is allowed to resolve only a
        # calibrated top task-verifier tie. Every tied candidate receives the
        # same exact arithmetic interventions in a fresh zero-offset context;
        # deterministic recomputation, not generated prose, assigns the
        # robustness score. Stronger task-verifier evidence is never displaced.
        if (
            self.config.counterfactual_verifier_enabled
            and verifier is not None
            and verification_objective.strip()
            and len(branch_probe_texts) == len(ensemble.branches)
            and len(blind_scores) == len(ensemble.branches)
        ):
            try:
                from core.brain.llm.latent_cortex.counterfactual_verifier import (
                    run_counterfactual_verifier,
                )

                receipt.counterfactual_verifier = run_counterfactual_verifier(
                    branch_probe_texts,
                    objective=verification_objective,
                    task_scores={branch: round(score, 6) for branch, score in blind_scores.items()},
                    selected_branch=winner.index,
                    max_atoms=self.config.counterfactual_verifier_max_atoms,
                    max_interventions=(self.config.counterfactual_verifier_max_interventions),
                    generate=lambda prompt: self._fresh_verifier_generation(
                        prompt,
                        budget,
                        max_tokens=self.config.counterfactual_verifier_max_tokens,
                        reserve_layer_apps=safety_reserve,
                    ),
                )
                if receipt.counterfactual_verifier["selection_authority_admitted"]:
                    selected = int(receipt.counterfactual_verifier["selected_branch"])
                    winner = next(
                        branch for branch in ensemble.branches if branch.index == selected
                    )
                    selection_basis = f"{selection_basis}_counterfactual_tiebreak"
            except (
                ImportError,
                OSError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                receipt.flag(f"counterfactual_verifier_abstained:{type(exc).__name__}")
                receipt.counterfactual_verifier = {
                    "requested": True,
                    "available": False,
                    "reason": f"{type(exc).__name__}:{exc}"[:240],
                    "selection_effect": "none",
                }
        elif self.config.counterfactual_verifier_enabled:
            receipt.counterfactual_verifier = {
                "requested": True,
                "available": False,
                "reason": (
                    "admitted_task_verifier_unavailable"
                    if verifier is None
                    else "verification_objective_unavailable"
                    if not verification_objective.strip()
                    else "complete_branch_probe_inventory_unavailable"
                ),
                "selection_effect": "none",
            }

        # SPARK-042: challenge the provisional winner in an entirely fresh KV
        # context. This is intentionally a refutation veto, not another
        # holistic model score: same-checkpoint prose never certifies itself.
        # Only a deterministic witness relation reconstructed by the verifier
        # envelope can remove the selected branch.
        generative_only_branch_refuted = False
        if (
            self.config.generative_verifier_enabled
            and verifier is not None
            and verification_objective.strip()
            and winner.index in branch_probe_texts
        ):
            try:
                from core.brain.llm.latent_cortex.generative_verifier import (
                    bind_selection_effect,
                    run_generative_verifier,
                )

                receipt.generative_verifier = run_generative_verifier(
                    branch_probe_texts[winner.index],
                    objective=verification_objective,
                    max_atoms=self.config.generative_verifier_max_atoms,
                    generate=lambda prompt: self._fresh_verifier_generation(
                        prompt,
                        budget,
                        max_tokens=self.config.generative_verifier_max_tokens,
                        reserve_layer_apps=safety_reserve,
                    ),
                )
                if receipt.generative_verifier["causal_refutation"]:
                    alternatives = [
                        branch for branch in ensemble.branches if branch.index != winner.index
                    ]
                    if alternatives:
                        vetoed = winner.index
                        winner = max(alternatives, key=lambda branch: branch.score)
                        selection_basis = f"{selection_basis}_generative_refutation_veto"
                        receipt.generative_verifier = bind_selection_effect(
                            receipt.generative_verifier,
                            vetoed_branch=vetoed,
                            replacement_branch=winner.index,
                        )
                    else:
                        receipt.generative_verifier = bind_selection_effect(
                            receipt.generative_verifier,
                            vetoed_branch=winner.index,
                            replacement_branch=None,
                        )
                        generative_only_branch_refuted = True
            except (
                ImportError,
                OSError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                receipt.flag(f"generative_verifier_abstained:{type(exc).__name__}")
                receipt.generative_verifier = {
                    "requested": True,
                    "available": False,
                    "reason": f"{type(exc).__name__}:{exc}"[:240],
                    "selection_effect": "none",
                }
        elif self.config.generative_verifier_enabled:
            receipt.generative_verifier = {
                "requested": True,
                "available": False,
                "reason": (
                    "admitted_task_verifier_unavailable"
                    if verifier is None
                    else "verification_objective_unavailable"
                    if not verification_objective.strip()
                    else "branch_probe_unavailable"
                ),
                "selection_effect": "none",
            }
        if generative_only_branch_refuted:
            raise RuntimeError("generative_verifier_refuted_only_branch")

        # SPARK-045: measure whether the selected conclusion recurs when the
        # resident checkpoint is restarted from the same deterministically
        # verified prefix. Every continuation receives a fresh zero-offset KV
        # cache and a local deterministic RNG key. This evidence is diagnostic
        # only: it cannot choose a branch or certify correctness.
        if (
            self.config.prefix_stability_enabled
            and verification_objective.strip()
            and winner.index in branch_probe_texts
        ):
            try:
                from core.brain.llm.latent_cortex.prefix_stability import (
                    run_prefix_stability_verifier,
                )

                receipt.prefix_stability = run_prefix_stability_verifier(
                    branch_probe_texts[winner.index],
                    objective=verification_objective,
                    samples=self.config.prefix_stability_samples,
                    temperature=self.config.prefix_stability_temperature,
                    top_p=self.config.prefix_stability_top_p,
                    seed_root=self.config.prefix_stability_seed,
                    calibrator_config=self.config.prefix_stability_calibrator,
                    generate=lambda prompt, seed, temperature, top_p: (
                        self._fresh_verifier_generation(
                            prompt,
                            budget,
                            max_tokens=self.config.prefix_stability_max_tokens,
                            reserve_layer_apps=safety_reserve,
                            temperature=temperature,
                            top_p=top_p,
                            sample_seed=seed,
                        )
                    ),
                )
            except (
                ImportError,
                OSError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                receipt.flag(f"prefix_stability_abstained:{type(exc).__name__}")
                receipt.prefix_stability = {
                    "requested": True,
                    "available": False,
                    "reason": f"{type(exc).__name__}:{exc}"[:240],
                    "selection_effect": "none",
                    "correctness_effect": "none",
                }
        elif self.config.prefix_stability_enabled:
            receipt.prefix_stability = {
                "requested": True,
                "available": False,
                "reason": (
                    "verification_objective_unavailable"
                    if not verification_objective.strip()
                    else "selected_branch_probe_unavailable"
                ),
                "selection_effect": "none",
                "correctness_effect": "none",
            }
        # SPARK-049 executes only after the prior verifier mesh. The attempt
        # may spend compute above the protected completion/fallback reserve,
        # adds a separately verified candidate, and cannot mutate a branch or
        # replace the accepted answer.
        if receipt.cognitive_operator_trace:
            repair_failures: dict[str, str] = {}
            for repair_request in prepared_repairs:
                request_id = str(repair_request["request_id"])
                rejected_redraws = 0
                generation_attempts = 0
                try:
                    # `local_repair_max_attempts` bounds repair FRONTIERS, not
                    # redraws inside one frontier. Give rejection sampling
                    # small, explicit generation headroom while preserving the
                    # one-verifier-call budget. A duplicate is never routed
                    # through decomposition or the deterministic verifier.
                    for generation_attempt in range(_MAX_LOCAL_REPAIR_GENERATIONS):
                        generation_attempts += 1
                        generated = self._fresh_verifier_generation(
                            str(repair_request["prompt"]),
                            budget,
                            max_tokens=self.config.local_repair_max_tokens,
                            reserve_layer_apps=safety_reserve,
                            final_answer_contract=False,
                        )
                        generated_context = generated["context"]
                        termination = str(generated_context.get("termination") or "")
                        # A repair that reached its token ceiling with a parseable
                        # suffix is still a repair. The suffix parser below is the
                        # real contract; an invalid payload remains inadmissible.
                        if termination not in {
                            "eos",
                            "token_limit",
                            "token_limit_sentence_grace",
                        }:
                            raise ValueError(f"repair generation terminated {termination!r}")
                        repaired_candidate = parse_local_repair_generation(
                            generated["text"],
                            prefix=str(repair_request["prefix"]),
                            tail=str(repair_request.get("tail") or ""),
                        )
                        # Rejection, not prompt-listing. Naming a refuted answer
                        # lost to i.i.d. in the A/B; an exact redraw is discarded
                        # locally and does not spend a verifier call.
                        if _repeats_a_refuted_answer(episode_ratchet, repaired_candidate):
                            rejected_redraws += 1
                            receipt.flag(
                                "local_repair_rejected_redraw:"
                                f"request={request_id}:generation={generation_attempt + 1}"
                            )
                            continue

                        from core.brain.llm.latent_cortex.atomic_decomposition import (
                            build_atomic_decomposition,
                        )

                        build_atomic_decomposition(
                            repaired_candidate,
                            objective=verification_objective,
                        )
                        generated_repairs[request_id] = {
                            "candidate": repaired_candidate,
                            "generation_context": {
                                "prompt_sha256": repair_request["prompt_sha256"],
                                "generated_token_count": generated_context["generated_token_count"],
                                "termination": generated_context["termination"],
                                "initial_cache_offsets": generated_context["initial_cache_offsets"],
                                "final_cache_offsets": generated_context["final_cache_offsets"],
                                "all_initial_offsets_zero": generated_context[
                                    "all_initial_offsets_zero"
                                ],
                                "solver_context_imported": generated_context[
                                    "solver_context_imported"
                                ],
                                "parameter_relation": generated_context["parameter_relation"],
                            },
                        }
                        break
                    else:
                        repair_failures[request_id] = "repeated_refuted_answer"
                    receipt.flag(
                        "local_repair_rejection_sampling:"
                        f"request={request_id}:generations={generation_attempts}:"
                        f"rejected_redraws={rejected_redraws}"
                    )
                except RuntimeError as exc:
                    repair_failures[request_id] = (
                        "budget_unavailable"
                        if "budget" in str(exc).lower()
                        else "generation_failed"
                    )
                except (ImportError, OSError, OverflowError, TypeError, ValueError) as exc:
                    # Record WHAT failed. "generation_contract_invalid" alone
                    # cost several diagnostic rounds because it covers the
                    # termination check, the contract parse, and the
                    # decomposition rebuild, which have entirely different
                    # fixes.
                    receipt.flag(
                        f"local_repair_generation_rejected:{type(exc).__name__}:{str(exc)[:80]}"
                    )
                    repair_failures[request_id] = "generation_contract_invalid"
            receipt.local_repair = build_local_repair_receipt(
                disagreement_graph=receipt.disagreement_graph,
                diagnostic_selection=receipt.diagnostic_action_selection,
                branch_candidates=branch_probe_texts,
                objective=verification_objective,
                generated_repairs=generated_repairs,
                execution_failures=repair_failures,
                max_requests=repair_limit,
            )
        receipt.branch_scores = [float(b.score) for b in ensemble.branches]
        receipt.selected_branch = winner.index
        receipt.branch_selection_admitted = True
        receipt.steps_taken = winner.steps
        receipt.residual_trail = list(winner.halting.residual_trail)
        receipt.best_step = (
            winner.verified_best_step
            if (not self.config.recurrence.fixed_depth and winner.verified_best_step >= 0)
            else winner.halting.best_step
        )
        receipt.halting_reason = winner.halt_reason
        receipt.reverted_to_best = winner.halt_reason.endswith("_reverted")
        telemetry.record_selection(receipt.branch_scores, winner.index)
        from core.brain.llm.latent_cortex.recurrent_grounding import (
            build_recurrent_grounding_receipt,
        )

        receipt.recurrent_grounding = build_recurrent_grounding_receipt(
            input_tokens_sha256=receipt.input_tokens_sha256,
            input_token_count=receipt.input_token_count,
            cognitive_slots=receipt.cognitive_slots,
            branches=list(ensemble.branches),
            n_slots=int(self.config.workspace.n_slots),
            comm_slot=int(self.config.branches.comm_slot),
            selected_branch=winner.index,
        )
        from core.brain.llm.latent_cortex.loop_core import build_loop_core_contract
        from core.brain.llm.latent_cortex.loop_stability import (
            build_loop_stability_receipt,
        )

        loop_core = build_loop_core_contract(
            prelude_end=self.prelude_end,
            coda_start=self.coda_start,
            max_steps=self.config.recurrence.max_steps,
            min_steps=self.config.recurrence.min_steps,
            alpha=self.config.recurrence.alpha,
            alpha_schedule=self.config.recurrence.alpha_schedule,
            rms_clip_ratio=self.config.recurrence.rms_clip_ratio,
            convergence_eps=self.config.recurrence.convergence_eps,
            divergence_ratio=self.config.recurrence.divergence_ratio,
            fixed_depth=self.config.recurrence.fixed_depth,
        )
        discarded_tree_kv_ordinals = sorted(
            {
                int(ordinal)
                for transaction in receipt.latent_tree_search.get("transactions", [])
                for ordinal in transaction["discarded_recurrent_kv_call_ordinals"]
            }
        )
        receipt.loop_stability = build_loop_stability_receipt(
            branches=list(ensemble.branches),
            selected_branch=winner.index,
            loop_core=loop_core,
            kv_bound=runner.kv_bound_receipt(),
            recurrent_grounding=receipt.recurrent_grounding,
            excluded_speculative_kv_call_ordinals=discarded_tree_kv_ordinals,
        )
        from core.brain.llm.latent_cortex.update_gate import (
            LEARNED as LEARNED_UPDATE_GATE,
        )
        from core.brain.llm.latent_cortex.update_gate import (
            build_update_gate_receipt,
        )

        receipt.update_acceptance = build_update_gate_receipt(
            branches=list(ensemble.branches),
            selected_branch=winner.index,
            gate=update_gate,
            recurrent_grounding=receipt.recurrent_grounding,
            loop_stability=receipt.loop_stability,
        )
        if (
            update_gate.mode == LEARNED_UPDATE_GATE
            and not receipt.update_acceptance["head_was_causal"]
        ):
            receipt.flag("learned_update_gate_not_causal")
        from core.brain.llm.latent_cortex.bidirectional_reflector import (
            build_bidirectional_reflector_receipt,
        )

        receipt.bidirectional_reflector = build_bidirectional_reflector_receipt(
            branches=list(ensemble.branches),
            update_acceptance=receipt.update_acceptance,
            selected_branch=winner.index,
            budget=budget,
        )
        from core.brain.llm.latent_cortex.contradiction_tensor import (
            build_contradiction_tensor_receipt,
        )

        receipt.contradiction_tensor = build_contradiction_tensor_receipt(
            reflector=receipt.bidirectional_reflector,
            runtime=contradiction_runtime,
            selected_branch=winner.index,
            budget=budget,
        )
        from core.brain.llm.latent_cortex.neural_uncertainty import (
            build_neural_uncertainty_receipt,
        )

        receipt.neural_uncertainty = build_neural_uncertainty_receipt(
            branches=list(ensemble.branches),
            runtime=uncertainty_runtime,
            update_acceptance=receipt.update_acceptance,
            selected_branch=winner.index,
            selection_basis=selection_basis,
        )
        from core.brain.llm.latent_cortex.contradiction_perturber import (
            ContradictionPerturberConfig,
            run_contradiction_perturbation,
        )

        perturber_config = ContradictionPerturberConfig.from_value(
            self.config.contradiction_perturber
        )
        information = budget.information_receipt or {}
        policies = information.get("policies")
        verifier_policy_sha256 = (
            str(policies.get("verifier", "")) if isinstance(policies, dict) else ""
        )
        decoy_review_sha256 = (
            str(receipt.decoy_verification.get("receipt_sha256", ""))
            if receipt.decoy_verification.get("selection_admitted") is True
            else ""
        )
        heterogeneous_incumbent_state = winner.z
        arm_count = 3 * perturber_config.replicates
        arm_layer_apps = self._verifier_probe_layer_apps(
            bridge_tokens,
            count=arm_count,
        )
        evaluation_unavailable_reason = ""
        arm_evaluator = None
        if verifier is None or self.tokenizer is None:
            evaluation_unavailable_reason = "independent_admitted_verifier_unavailable"
        elif arm_layer_apps + safety_reserve > budget.remaining_layer_apps:
            evaluation_unavailable_reason = "counterfactual_probe_budget_unavailable"
        else:
            arm_evaluator = self._counterfactual_probe_evaluator(
                branch=winner,
                cache=cache,
                runner=runner,
                budget=budget,
                bridge_tokens=bridge_tokens,
                verifier=verifier,
            )

        resulting_state, receipt.contradiction_perturbation = run_contradiction_perturbation(
            baseline=winner.z,
            anchor=winner.anchor,
            protected_positions=winner.workspace.context_slot_indices,
            contradiction_tensor=receipt.contradiction_tensor,
            selected_branch=winner.index,
            config=perturber_config,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            evaluate=arm_evaluator,
            evaluation_unavailable_reason=evaluation_unavailable_reason,
            budget=budget,
        )
        if receipt.contradiction_perturbation["state_mutation_applied"]:
            import mlx.core as mx

            winner.z = winner.workspace.restore_context_evidence(mx.array(resulting_state))
            winner.workspace.update(winner.z)
            receipt.flag("contradiction_perturbation_retained")
        elif receipt.contradiction_perturbation["status"] == "restored":
            receipt.flag("contradiction_perturbation_restored")
            if str(receipt.contradiction_perturbation["reason"]).startswith("evaluation_failed:"):
                receipt.flag("contradiction_perturbation_evaluation_failed")

        from core.brain.llm.latent_cortex.local_exploration import (
            LocalExplorationConfig,
            run_local_exploration,
        )

        exploration_config = LocalExplorationConfig.from_value(self.config.local_exploration)
        exploration_probe_count = 3 * exploration_config.candidates * exploration_config.replicates
        exploration_layer_apps = self._verifier_probe_layer_apps(
            bridge_tokens,
            count=exploration_probe_count,
        )
        exploration_unavailable_reason = ""
        exploration_evaluator = None
        if verifier is None or self.tokenizer is None:
            exploration_unavailable_reason = "independent_admitted_verifier_unavailable"
        elif exploration_layer_apps + safety_reserve > budget.remaining_layer_apps:
            exploration_unavailable_reason = "counterfactual_probe_budget_unavailable"
        else:
            exploration_evaluator = self._counterfactual_probe_evaluator(
                branch=winner,
                cache=cache,
                runner=runner,
                budget=budget,
                bridge_tokens=bridge_tokens,
                verifier=verifier,
            )
        explored_state, receipt.local_exploration = run_local_exploration(
            baseline=winner.z,
            protected_positions=winner.workspace.context_slot_indices,
            contradiction_tensor=receipt.contradiction_tensor,
            contradiction_perturbation=receipt.contradiction_perturbation,
            neural_uncertainty=receipt.neural_uncertainty,
            selected_branch=winner.index,
            config=exploration_config,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            evaluate=exploration_evaluator,
            evaluation_unavailable_reason=exploration_unavailable_reason,
            budget=budget,
        )
        if receipt.local_exploration["state_mutation_applied"]:
            import mlx.core as mx

            winner.z = winner.workspace.restore_context_evidence(mx.array(explored_state))
            winner.workspace.update(winner.z)
            receipt.flag("local_exploration_retained")
        elif receipt.local_exploration["status"] == "restored":
            receipt.flag("local_exploration_restored")
            if str(receipt.local_exploration["reason"]).startswith("evaluation_failed:"):
                receipt.flag("local_exploration_evaluation_failed")
        from core.brain.llm.latent_cortex.heterogeneous_integrator import (
            POLICIES,
            HeterogeneousIntegrationConfig,
            run_heterogeneous_integration,
        )

        heterogeneous_config = HeterogeneousIntegrationConfig.from_value(
            self.config.heterogeneous_integration
        )
        heterogeneous_probe_count = 2 * len(POLICIES) * heterogeneous_config.replicates
        heterogeneous_probe_apps = self._verifier_probe_layer_apps(
            bridge_tokens,
            count=heterogeneous_probe_count,
        )
        fused_completion_extra = persist_cost + bridge_cost + decode_cost
        heterogeneous_unavailable_reason = ""
        heterogeneous_evaluator = None
        if verifier is None or self.tokenizer is None:
            heterogeneous_unavailable_reason = "independent_admitted_verifier_unavailable"
        elif (
            heterogeneous_probe_apps + fused_completion_extra + safety_reserve
            > budget.remaining_layer_apps
        ):
            heterogeneous_unavailable_reason = "counterfactual_probe_budget_unavailable"
        else:
            heterogeneous_evaluator = self._heterogeneous_policy_evaluator(
                branch=winner,
                cache=cache,
                runner=runner,
                budget=budget,
                bridge_tokens=bridge_tokens,
                verifier=verifier,
            )
        (
            integrated_state,
            heterogeneous_policy,
            receipt.heterogeneous_integration,
        ) = run_heterogeneous_integration(
            incumbent_state=heterogeneous_incumbent_state,
            corrected_state=winner.z,
            contradiction_perturbation=receipt.contradiction_perturbation,
            local_exploration=receipt.local_exploration,
            config=heterogeneous_config,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            evaluate=heterogeneous_evaluator,
            evaluation_unavailable_reason=(heterogeneous_unavailable_reason),
            budget=budget,
        )
        import mlx.core as mx

        winner.z = winner.workspace.restore_context_evidence(mx.array(integrated_state))
        winner.workspace.update(winner.z)
        heterogeneous_finalized = bool(
            receipt.heterogeneous_integration["policies"]
        ) and receipt.heterogeneous_integration["status"] in {
            "selected",
            "abstained",
        }
        heterogeneous_fusion_context = (
            (
                heterogeneous_incumbent_state,
                integrated_state,
                float(receipt.heterogeneous_integration["fusion_weight"]),
            )
            if heterogeneous_policy == "probability_fusion"
            else None
        )
        if heterogeneous_finalized:
            receipt.flag(f"heterogeneous_integration_finalized:{heterogeneous_policy}")
        from core.brain.llm.latent_cortex.mistake_locator import (
            build_mistake_locator_receipt,
        )

        receipt.mistake_locator = build_mistake_locator_receipt(
            branches=list(ensemble.branches),
            runtime=mistake_locator_runtime,
            update_acceptance=receipt.update_acceptance,
            selected_branch=winner.index,
            domain=domain,
            process_selection_used=selection_basis == "process_verifier",
        )
        from core.brain.llm.latent_cortex.verifier_fusion import (
            build_verifier_fusion_receipt,
        )

        receipt.verifier_fusion = build_verifier_fusion_receipt(
            blind_review=receipt.blind_review,
            decoy_verification=receipt.decoy_verification,
            generative_verifier=receipt.generative_verifier,
            counterfactual_verifier=receipt.counterfactual_verifier,
            prefix_stability=receipt.prefix_stability,
            neural_uncertainty=receipt.neural_uncertainty,
            mistake_locator=receipt.mistake_locator,
            selected_branch=winner.index,
            evidence=self.config.verifier_fusion_evidence,
        )
        escape_receipts: dict[str, Any] = {}
        for branch in ensemble.branches:
            if branch.escape is not None and branch.escape.attempts:
                escape_receipts[str(branch.index)] = branch.escape.to_receipt()
                outcomes = {a.outcome for a in branch.escape.attempts}
                if "retained" in outcomes:
                    receipt.flag("attractor_escape_retained")
                if outcomes & {"failed", "unresolved"}:
                    receipt.flag("attractor_escape_failed")
        receipt.escape = escape_receipts
        from core.brain.llm.latent_cortex.stop_gate import (
            build_stop_gate_receipt,
        )

        receipt.halting = build_stop_gate_receipt(
            branches=list(ensemble.branches),
            gate=stop_gate,
            update_acceptance=receipt.update_acceptance,
            loop_stability=receipt.loop_stability,
            cognitive_action_trace=receipt.cognitive_action_trace,
        )
        if stop_gate.mode == "learned" and not receipt.halting["head_was_causal"]:
            receipt.flag("learned_halting_not_causal")
        from core.brain.llm.latent_cortex.terminal_disposition import (
            classify_terminal_disposition,
        )

        terminal_decision = classify_terminal_disposition(
            halting_reason=receipt.halting_reason,
            halting=receipt.halting,
            loop_stability=receipt.loop_stability,
            cognitive_action_trace=receipt.cognitive_action_trace,
            budget=budget.to_receipt(),
        )
        # The disposition is always classified and receipted. Injecting it as
        # language ahead of the decode is a separate, configurable act: the
        # text is an instruction, so an arm that receives it is not comparable
        # to a control arm that does not.
        terminal_instruction_suppressed = self.config.terminal_instruction_policy == "suppressed"
        if self.tokenizer is not None and not terminal_instruction_suppressed:
            terminal_instruction_tokens = self._encode_terminal_instruction(
                terminal_decision.instruction
            )
        from core.brain.llm.latent_cortex.verified_best import (
            build_verified_best_receipt,
        )

        receipt.verified_best_state = build_verified_best_receipt(
            branches=list(ensemble.branches),
            cognitive_action_trace=receipt.cognitive_action_trace,
            loop_stability=receipt.loop_stability,
        )
        receipt.transient_negative_constraints = transient_constraints.finalize(
            final_action_step=action_index,
        )
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
        # Best VERIFIED score of the winner's final latent state — becomes
        # the pre-adaptation reference for fast-weight verifier arbitration.
        latent_opt_verifier_score = float("-inf")
        if self.config.latent_opt.enabled and not heterogeneous_finalized:
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
                layer_apps_per_loss=(self.config.workspace.n_slots * self.n_layers),
                scalar_ops_per_loss=(
                    (21 * self.config.workspace.n_slots + 8)
                    * budget.resource_ledger.profile.hidden_size
                    + 8 * budget.resource_ledger.profile.vocab_size
                    + 2
                    * budget.resource_ledger.profile.hidden_size
                    * budget.resource_ledger.profile.vocab_size
                ),
                reserve_layer_apps=safety_reserve,
                protected_slots=winner.workspace.context_slot_indices,
            )
            latent_search_verifier = verifier
            if (
                latent_search_verifier is None
                and pending_verifier is not None
                and self.tokenizer is not None
                and receipt.verifier_preflight.get("verifier_admitted") is True
                and callable(getattr(pending_verifier, "latent_state_score", None))
            ):
                # Branch selection can lose authority when the strict candidate
                # inventory is incomplete. That does not erase the separately
                # admitted candidate-local semantic scorer: it can still reject
                # latent drift on the already selected branch, but it receives
                # no branch-selection or answer-replacement authority.
                latent_search_verifier = pending_verifier
                receipt.flag("latent_opt_candidate_local_score_without_branch_selection")
            if latent_search_verifier is not None and self.tokenizer is not None:
                latent_state_score = getattr(
                    latent_search_verifier,
                    "latent_state_score",
                    None,
                )
                latent_state_score_enabled = callable(latent_state_score)

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
                        decoded = self.tokenizer.decode(probe)
                        return float(
                            latent_state_score(decoded)
                            if latent_state_score_enabled
                            else latent_search_verifier(decoded)
                        )
                    finally:
                        winner.z = saved
                        winner.workspace.update(saved)

                z_opt, latent_opt_verifier_score = optimizer.run_with_verifier(
                    winner.z,
                    z_score,
                    verifier_layer_apps=self._verifier_probe_layer_apps(bridge_tokens),
                    initial_score=(
                        float(latent_state_score(latent_optimization_probe_texts[winner.index]))
                        if latent_state_score_enabled
                        and winner.index in latent_optimization_probe_texts
                        else branch_verifier_score
                    ),
                    accept_non_regression=(self.config.verifier_accept_non_regression),
                    commit_requires_score_improvement=latent_state_score_enabled,
                )
                if latent_state_score_enabled:
                    optimizer.trace.verifier_score_source = (
                        "semantic_candidate_score_for_latent_search_only_v1"
                    )
            else:
                z_opt = optimizer.run(winner.z)
            winner.z = winner.workspace.restore_context_evidence(z_opt)
            winner.workspace.update(winner.z)
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
            receipt.latent_opt_verifier = optimizer.trace.verifier_receipt()
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
        fast_weight_learning_state: dict[str, Any] | None = None
        fast_weight_teaching_event: dict[str, Any] = {}
        fast_weight_target_tokens: list[int] = []
        fw_verifier_pre_tokens: list[int] = []
        fw_verifier_pre_text = ""
        fw_verifier_pre: float | None = None
        fast_weight_decode_active = False
        canary_baseline: dict[str, float] | None = None
        canary_generated_baseline: dict[str, Any] | None = None
        fw_initial_snapshot: tuple[dict[str, Any], ...] = ()
        fw_treatment_snapshot: tuple[dict[str, Any], ...] = ()
        fw_treatment_trace: dict[str, Any] = {}
        fw_sham_trace: dict[str, Any] = {}
        fw_sham_initial_snapshot: tuple[dict[str, Any], ...] = ()
        fw_sham_target_tokens: list[int] = []
        fw_sham_semantic_seed_vectors = None
        fw_sham_trajectory_directions = None
        fw_incumbent_input_features = None
        fw_treatment_output_corrections = None
        fw_sham_output_corrections = None
        fw_sham_probe_tokens: list[int] = []
        fw_sham_score: float | None = None
        if self.config.fast_weights.enabled and not heterogeneous_finalized:
            winner_state_sha256 = tensor_sha256(winner.z)
            objective_sha256 = hashlib.sha256(verification_objective.encode("utf-8")).hexdigest()
            admission = unavailable_admission(
                source_sha256=hashlib.sha256(b"").hexdigest(),
                objective_sha256=objective_sha256,
                reason="verifier_unavailable",
            )
            evidence_provider = getattr(
                information_verifier,
                "fast_weight_learning_evidence",
                None,
            )
            fast_weight_candidate_verifier = verifier
            if (
                fast_weight_candidate_verifier is None
                and pending_verifier is not None
                and receipt.verifier_preflight.get("verifier_admitted") is True
            ):
                # An incomplete branch inventory removes authority to choose
                # among branches; it does not invalidate a preflight-admitted
                # verifier's candidate-local evidence on the already selected
                # state. Latent optimization already preserves this narrower
                # authority. Fast weights must use the same distinction.
                fast_weight_candidate_verifier = pending_verifier
                receipt.flag(
                    "fast_weight_candidate_local_evidence_without_branch_selection"
                )
            if fast_weight_candidate_verifier is not None and self.tokenizer is not None:
                probe_cost = self._verifier_probe_layer_apps(bridge_tokens)
                if probe_cost + safety_reserve > budget.remaining_layer_apps:
                    raise RuntimeError("compute budget cannot admit fast-weight evidence probe")
                evaluation_index = len(getattr(information_verifier, "evaluations", ()))
                fw_verifier_pre_tokens = self._decode_probe(
                    winner,
                    cache,
                    runner,
                    budget,
                    bridge_tokens=bridge_tokens,
                    use_cache=False,
                    force_exact_tokens=True,
                )
                fw_verifier_pre_text = self.tokenizer.decode(fw_verifier_pre_tokens)
                fw_verifier_pre = float(
                    fast_weight_candidate_verifier(fw_verifier_pre_text)
                )
                source_sha256 = hashlib.sha256(fw_verifier_pre_text.encode("utf-8")).hexdigest()
                if callable(evidence_provider):
                    try:
                        admission, fast_weight_target_tokens = evidence_provider(
                            fw_verifier_pre_text,
                            evaluation_index=evaluation_index,
                            tokenizer=self.tokenizer,
                            structural_diversity=receipt.structural_diversity,
                        )
                        admission = validate_fast_weight_admission(
                            admission,
                            expected_source_sha256=source_sha256,
                            expected_objective_sha256=objective_sha256,
                        )
                        if admission["target_token_count"] != len(
                            fast_weight_target_tokens
                        ) or admission["target_tokens_sha256"] != token_sequence_sha256(
                            fast_weight_target_tokens
                        ):
                            raise ValueError(
                                "fast-weight private target differs from its admission commitment"
                            )
                    except _LATENT_PHASE_ERRORS as exc:
                        receipt.flag(f"fast_weight_evidence_rejected:{type(exc).__name__}")
                        admission = unavailable_admission(
                            source_sha256=source_sha256,
                            objective_sha256=objective_sha256,
                            reason="candidate_evaluation_unavailable",
                        )
                        fast_weight_target_tokens = []
                else:
                    admission = unavailable_admission(
                        source_sha256=source_sha256,
                        objective_sha256=objective_sha256,
                        reason="verifier_provider_untrusted",
                    )
            if (
                admission["admitted"] is not True
                and self.config.verified_objective_teacher_enabled
                and callable(evidence_provider)
                and self.tokenizer is not None
                and fw_verifier_pre_text
            ):
                try:
                    (
                        fast_weight_teaching_event,
                        admission,
                        fast_weight_target_tokens,
                    ) = build_exact_objective_teaching_event(
                        objective=verification_objective,
                        incumbent_candidate=fw_verifier_pre_text,
                        source_state_sha256=winner_state_sha256,
                        tokenizer=self.tokenizer,
                        structural_diversity=receipt.structural_diversity,
                    )
                    receipt.flag("fast_weight_exact_objective_teacher_admitted")
                except _LATENT_PHASE_ERRORS as exc:
                    receipt.flag(
                        "fast_weight_exact_objective_teacher_unavailable:"
                        f"{type(exc).__name__}"
                    )
            fast_weight_learning_state = empty_learning_state(
                episode_id=receipt.episode_id,
                input_tokens_sha256=receipt.input_tokens_sha256,
                selected_branch=winner.index,
                winner_state_sha256=winner_state_sha256,
                admission=admission,
                teaching_event=fast_weight_teaching_event,
            )
            stage_started = self._stage_checkpoint(
                receipt=receipt,
                budget=budget,
                stage="fast_weight_baseline",
                stage_started=stage_started,
                episode_started=episode_started,
                progress=progress,
                cancel_check=cancel_check,
                admitted=bool(admission["admitted"]),
                admission_reason=str(admission["reason"]),
            )
            if admission["admitted"]:
                vocab_size = int(self.model.model.embed_tokens.weight.shape[0])
                if any(not 0 <= token < vocab_size for token in fast_weight_target_tokens):
                    raise RuntimeError(
                        "fast-weight evidence target contains an out-of-vocabulary token"
                    )
                if fast_weight_attach_identity_cost + safety_reserve > budget.remaining_layer_apps:
                    raise RuntimeError("compute budget cannot admit fast-weight identity probes")
                fast_weights = EpisodicFastWeights(self.config.fast_weights)
                if self._episode_probe_cache is not None:
                    fast_weights.on_function_change = self._episode_probe_cache.invalidate
                fw_baseline = self._fw_probe(budget)
                if canaries is not None:
                    canary_baseline = canaries.measure(
                        lambda probe_tokens: self._canary_logits(
                            probe_tokens,
                            budget,
                        )
                    )
                    # The postconditions need a base reading for the same
                    # reason the likelihood battery does: a canary the BASE
                    # function already fails says nothing about ΔW. A
                    # random-weight substrate model fails "answer in exactly
                    # one word" before any adaptation exists; erasing ΔW over
                    # that would be blaming the update for the model.
                    canary_generated_baseline = self._run_generated_canaries(canaries, budget)
                seed_stat = float(mx.mean(per_position_rms(winner.z)))
                retrieval_seed_vectors = None
                seed_source = "retrieval"
                retrieval_indices = [
                    int(row["slot"])
                    for row in receipt.cognitive_slots
                    if row.get("source") in _RETRIEVAL_SLOT_SOURCES
                    or str(row.get("source") or "").startswith("memory.")
                ]
                if retrieval_indices:
                    retrieval_seed_vectors = winner.z[0, retrieval_indices, :]
                if fast_weight_teaching_event:
                    incumbent_tokens = self.tokenizer.encode(
                        fw_verifier_pre_text,
                        add_special_tokens=False,
                    )
                    retrieval_seed_vectors = build_contrastive_semantic_seeds(
                        self.model,
                        target_tokens=fast_weight_target_tokens,
                        contrast_tokens=incumbent_tokens,
                        rank=self.config.fast_weights.rank,
                    )
                    seed_source = "verified_semantic_contrast"
                    vocab_size = int(self.model.model.embed_tokens.weight.shape[0])
                    fw_sham_target_tokens = deterministic_sham_target(
                        fast_weight_target_tokens,
                        vocab_size=vocab_size,
                        episode_id=receipt.episode_id,
                    )
                    fw_sham_semantic_seed_vectors = build_contrastive_semantic_seeds(
                        self.model,
                        target_tokens=fw_sham_target_tokens,
                        contrast_tokens=incumbent_tokens,
                        rank=self.config.fast_weights.rank,
                    )
                wrapped = fast_weights.attach(
                    self.model.model,
                    (self.prelude_end, self.coda_start),
                    seed_stat=seed_stat,
                    episode_id=receipt.episode_id,
                    seed_vectors=retrieval_seed_vectors,
                    seed_source=seed_source,
                )
                receipt.fast_weights_applied = True
                receipt.fast_weights_layers = wrapped
                receipt.flag(f"fast_weight_site:{self.plasticity_site.site_id}")
                try:
                    fast_weight_learning_state["lease"] = fast_weights.lease_receipt()
                    attach_probe = self._fw_probe(budget)
                    pre_probe_sha256 = tensor_sha256(fw_baseline)
                    post_probe_sha256 = tensor_sha256(attach_probe)
                    state_after_attach_sha256 = tensor_sha256(winner.z)
                    fast_weight_learning_state["attach_identity"] = {
                        "measured": True,
                        "pre_probe_sha256": pre_probe_sha256,
                        "post_probe_sha256": post_probe_sha256,
                        "exact": pre_probe_sha256 == post_probe_sha256,
                        "winner_state_before_sha256": winner_state_sha256,
                        "winner_state_after_sha256": (state_after_attach_sha256),
                    }
                    if (
                        pre_probe_sha256 != post_probe_sha256
                        or state_after_attach_sha256 != winner_state_sha256
                    ):
                        raise RuntimeError("fast-weight attachment failed measured identity")
                    if (
                        fast_weight_teaching_event
                        and self.config.fast_weights.output_memory_diagnostic_enabled
                        and fast_weight_candidate_verifier is not None
                        and fw_verifier_pre is not None
                    ):
                        treatment_keys = self._capture_forced_output_keys(
                            winner,
                            cache,
                            runner,
                            budget,
                            bridge_tokens=bridge_tokens,
                            target_tokens=fast_weight_target_tokens,
                            operation="output_memory_treatment_capture",
                        )
                        sham_keys = self._capture_forced_output_keys(
                            winner,
                            cache,
                            runner,
                            budget,
                            bridge_tokens=bridge_tokens,
                            target_tokens=fw_sham_target_tokens,
                            operation="output_memory_sham_capture",
                        )
                        fast_weight_learning_state["controls"][
                            "output_associative_memory"
                        ] = self._evaluate_output_memory_controls(
                            winner,
                            cache,
                            runner,
                            budget,
                            bridge_tokens=bridge_tokens,
                            verifier=fast_weight_candidate_verifier,
                            baseline_score=float(fw_verifier_pre),
                            treatment_keys=treatment_keys,
                            treatment_tokens=fast_weight_target_tokens,
                            sham_keys=sham_keys,
                            sham_tokens=fw_sham_target_tokens,
                        )
                        receipt.flag("output_associative_memory_diagnostic_completed")
                    if fast_weight_teaching_event:
                        (
                            _target_input_features,
                            target_features,
                            target_context_tokens,
                        ) = (
                            self._capture_teacher_forced_trajectory(
                                fast_weights,
                                budget,
                                objective=verification_objective,
                                answer_tokens=fast_weight_target_tokens,
                                operation="fast_weight_target_trajectory",
                            )
                        )
                        (
                            fw_incumbent_input_features,
                            incumbent_features,
                            incumbent_context_tokens,
                        ) = (
                            self._capture_teacher_forced_trajectory(
                                fast_weights,
                                budget,
                                objective=verification_objective,
                                answer_tokens=incumbent_tokens,
                                operation="fast_weight_incumbent_trajectory",
                            )
                        )
                        (
                            _sham_input_features,
                            sham_features,
                            sham_context_tokens,
                        ) = (
                            self._capture_teacher_forced_trajectory(
                                fast_weights,
                                budget,
                                objective=verification_objective,
                                answer_tokens=fw_sham_target_tokens,
                                operation="fast_weight_sham_trajectory",
                            )
                        )
                        treatment_directions = build_layerwise_trajectory_directions(
                            target_features,
                            incumbent_features,
                            rank=self.config.fast_weights.rank,
                        )
                        fw_sham_trajectory_directions = (
                            build_layerwise_trajectory_directions(
                                sham_features,
                                incumbent_features,
                                rank=self.config.fast_weights.rank,
                            )
                        )
                        fw_treatment_output_corrections = {
                            layer: target_features[layer] - incumbent_features[layer]
                            for layer in target_features
                        }
                        fw_sham_output_corrections = {
                            layer: sham_features[layer] - incumbent_features[layer]
                            for layer in sham_features
                        }
                        fast_weights.reseed_output_subspace_by_layer(
                            treatment_directions,
                            seed_source="verified_semantic_contrast",
                        )
                        fast_weights.capture_input_summaries(
                            lambda: self._decode_probe(
                                winner,
                                cache,
                                runner,
                                budget,
                                bridge_tokens=bridge_tokens,
                                use_cache=False,
                                force_exact_tokens=True,
                            )
                        )
                        layers = sorted(treatment_directions)
                        fast_weight_learning_state["controls"][
                            "trajectory_transplant"
                        ] = {
                            "schema": "aura.fast_weight_trajectory_transplant.v1",
                            "site_id": self.plasticity_site.site_id,
                            "layers": layers,
                            "rank": self.config.fast_weights.rank,
                            "target_context_sha256": token_sequence_sha256(
                                target_context_tokens
                            ),
                            "incumbent_context_sha256": token_sequence_sha256(
                                incumbent_context_tokens
                            ),
                            "sham_context_sha256": token_sequence_sha256(
                                sham_context_tokens
                            ),
                            "query_activation_sha256s": {
                                str(layer): digest
                                for layer, digest in (
                                    fast_weights.input_feature_commitments().items()
                                )
                            },
                            "target_direction_sha256s": {
                                str(layer): tensor_sha256(treatment_directions[layer])
                                for layer in layers
                            },
                            "sham_direction_sha256s": {
                                str(layer): tensor_sha256(
                                    fw_sham_trajectory_directions[layer]
                                )
                                for layer in layers
                            },
                        }
                        receipt.flag(
                            f"fast_weight_teacher_trajectory_transplant:{len(layers)}"
                        )
                except BaseException:  # noqa: BLE001 - mutation cleanup must survive cancellation
                    self._finalize_fast_weights(
                        fast_weights,
                        fw_baseline,
                        receipt,
                        budget,
                        learning_state=fast_weight_learning_state,
                    )
                    raise
                fw_initial_snapshot = fast_weights.snapshot_delta()
                if fast_weights.lifecycle.retrieval_seeded_columns > 0:
                    receipt.flag(
                        "fast_weight_retrieval_compiled:"
                        f"{fast_weights.lifecycle.retrieval_seeded_columns}"
                    )
                if fast_weights.lifecycle.semantic_seeded_columns > 0:
                    receipt.flag(
                        "fast_weight_verified_semantic_subspace:"
                        f"{fast_weights.lifecycle.semantic_seeded_columns}"
                    )
            else:
                receipt.flag(f"fast_weight_not_admitted:{admission['reason']}")

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
                # Attachment is a literal pass-through until the byte-level
                # identity probe above succeeds. Only then may the temporary
                # adaptation function enter the execution graph.
                fast_weights.activate_adaptation_path()
                if (
                    fast_weight_teaching_event
                    and self.config.fast_weights.associative_bootstrap_enabled
                ):
                    if (
                        fw_incumbent_input_features is not None
                        and fw_treatment_output_corrections is not None
                    ):
                        write_receipt = fast_weights.install_supervised_trajectory_map(
                            fw_incumbent_input_features,
                            fw_treatment_output_corrections,
                            gain=self.config.fast_weights.associative_bootstrap_gain,
                            regularization=(
                                self.config.fast_weights.associative_bootstrap_regularization
                            ),
                        )
                        fast_weight_learning_state["controls"][
                            "supervised_trajectory_map"
                        ] = write_receipt
                        receipt.flag(
                            "fast_weight_supervised_trajectory_write:"
                            f"{len(write_receipt['layers'])}"
                        )
                    else:
                        write_receipt = fast_weights.install_minimum_norm_keys(
                            gain=self.config.fast_weights.associative_bootstrap_gain,
                            regularization=(
                                self.config.fast_weights.associative_bootstrap_regularization
                            ),
                        )
                        receipt.flag(
                            "fast_weight_minimum_norm_write:"
                            f"{len(write_receipt['layers'])}"
                        )
                        receipt.flag("fast_weight_answer_decode_keys_compiled")
                # The temporary synapses optimize only toward the exact
                # evidence atoms admitted above. Prompt reconstruction remains
                # the latent-state optimizer's objective; using it here would
                # let unverified query text become a learning target.
                loss_fn = build_proxy_loss(
                    self.model,
                    winner.anchor,
                    fast_weight_target_tokens,
                    self.config.latent_opt,
                )

                def fw_loss():
                    z_pass = self._nocache_window_pass(
                        winner.z,
                        branch_index=winner.index,
                    )
                    return loss_fn(z_pass)

                fast_weights.optimize(
                    fw_loss,
                    budget=budget,
                    layer_apps_per_forward=(
                        self.config.workspace.n_slots * (self.coda_start - self.prelude_end)
                    ),
                    tokens_per_forward=self.config.workspace.n_slots,
                    layers_per_forward=(self.coda_start - self.prelude_end),
                    reserve_layer_apps=safety_reserve,
                    fixed_line_search_evaluations=(MATCHED_LINE_SEARCH_EVALUATIONS),
                    operation_prefix="fast_weight_treatment",
                )
                fw_treatment_snapshot = fast_weights.snapshot_delta()
                fw_treatment_trace = fast_weights.optimization_trace()
                fast_weights.restore_delta(
                    fw_initial_snapshot,
                    reason="fast_weights_matched_control_reset",
                )
                if fw_sham_trajectory_directions is not None:
                    fast_weights.reseed_output_subspace_by_layer(
                        fw_sham_trajectory_directions,
                        seed_source="verified_semantic_contrast",
                    )
                    receipt.flag("fast_weight_matched_trajectory_subspace")
                    fw_sham_initial_snapshot = fast_weights.snapshot_delta()
                    if (
                        fw_incumbent_input_features is not None
                        and fw_sham_output_corrections is not None
                    ):
                        fast_weights.install_supervised_trajectory_map(
                            fw_incumbent_input_features,
                            fw_sham_output_corrections,
                            gain=self.config.fast_weights.associative_bootstrap_gain,
                            regularization=(
                                self.config.fast_weights.associative_bootstrap_regularization
                            ),
                        )
                    else:
                        fast_weights.install_minimum_norm_keys(
                            gain=self.config.fast_weights.associative_bootstrap_gain,
                            regularization=(
                                self.config.fast_weights.associative_bootstrap_regularization
                            ),
                        )
                elif fw_sham_semantic_seed_vectors is not None:
                    fast_weights.reseed_output_subspace(
                        fw_sham_semantic_seed_vectors,
                        seed_source="verified_semantic_contrast",
                    )
                    receipt.flag("fast_weight_matched_semantic_subspace")
                    fw_sham_initial_snapshot = fast_weights.snapshot_delta()
                    fast_weights.install_minimum_norm_keys(
                        gain=self.config.fast_weights.associative_bootstrap_gain,
                        regularization=(
                            self.config.fast_weights.associative_bootstrap_regularization
                        ),
                    )
                fast_weights.reset_optimization_trace()
                vocab_size = int(self.model.model.embed_tokens.weight.shape[0])
                if not fw_sham_target_tokens:
                    fw_sham_target_tokens = deterministic_sham_target(
                        fast_weight_target_tokens,
                        vocab_size=vocab_size,
                        episode_id=receipt.episode_id,
                    )
                sham_loss_fn = build_proxy_loss(
                    self.model,
                    winner.anchor,
                    fw_sham_target_tokens,
                    self.config.latent_opt,
                )

                def fw_sham_loss():
                    z_pass = self._nocache_window_pass(
                        winner.z,
                        branch_index=winner.index,
                    )
                    return sham_loss_fn(z_pass)

                fast_weights.optimize(
                    fw_sham_loss,
                    budget=budget,
                    layer_apps_per_forward=(
                        self.config.workspace.n_slots * (self.coda_start - self.prelude_end)
                    ),
                    tokens_per_forward=self.config.workspace.n_slots,
                    layers_per_forward=(self.coda_start - self.prelude_end),
                    reserve_layer_apps=safety_reserve,
                    fixed_line_search_evaluations=(MATCHED_LINE_SEARCH_EVALUATIONS),
                    operation_prefix="fast_weight_sham",
                )
                fw_sham_trace = fast_weights.optimization_trace()
                if (
                    fast_weight_candidate_verifier is not None
                    and self.tokenizer is not None
                    and fast_weight_teaching_event
                    and fw_sham_initial_snapshot
                ):
                    (
                        fw_treatment_snapshot,
                        fw_sham_probe_tokens,
                        fw_sham_score,
                        gain_search_receipt,
                    ) = self._search_fast_weight_gains(
                        verifier=fast_weight_candidate_verifier,
                        fast_weights=fast_weights,
                        winner=winner,
                        cache=cache,
                        runner=runner,
                        budget=budget,
                        bridge_tokens=bridge_tokens,
                        safety_reserve=safety_reserve,
                        baseline_score=float(fw_verifier_pre),
                        treatment_initial=fw_initial_snapshot,
                        treatment_candidate=fw_treatment_snapshot,
                        sham_initial=fw_sham_initial_snapshot,
                        sham_candidate=fast_weights.snapshot_delta(),
                    )
                    fast_weight_learning_state["controls"][
                        "verifier_gain_search"
                    ] = gain_search_receipt
                elif (
                    fast_weight_candidate_verifier is not None
                    and self.tokenizer is not None
                ):
                    fw_sham_probe_tokens = self._decode_probe(
                        winner,
                        cache,
                        runner,
                        budget,
                        bridge_tokens=bridge_tokens,
                        use_cache=False,
                        force_exact_tokens=True,
                    )
                    fw_sham_score = float(
                        fast_weight_candidate_verifier(
                            self.tokenizer.decode(fw_sham_probe_tokens)
                        )
                    )
                fast_weights.restore_delta(
                    fw_treatment_snapshot,
                    reason="fast_weights_matched_treatment_restore",
                )
                fast_weights.restore_optimization_trace(fw_treatment_trace)
                lifecycle = fast_weights.lifecycle
                fast_weight_learning_state["optimization"] = {
                    "optimizer": lifecycle.optimizer,
                    "attempts": lifecycle.optimization_attempts,
                    "accepted_steps": lifecycle.optimized_steps,
                    "rejected_steps": lifecycle.rejected_steps,
                    "budget_exhausted": lifecycle.budget_exhausted,
                    "loss_trail": list(lifecycle.loss_trail),
                    "gradient_norm_trail": list(lifecycle.gradient_global_norm_trail),
                    "accepted_step_sizes": list(lifecycle.accepted_step_sizes),
                    "line_search_backtracks": (lifecycle.line_search_backtracks),
                }
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
                        generated_baseline=canary_generated_baseline,
                    )
                    telemetry.record_fast_weights(receipt.fast_weight_canaries)
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
                    fast_weight_learning_state["controls"] = {
                        "decision": canary_decision,
                        "capability_canaries": dict(receipt.fast_weight_canaries),
                        "trajectory_transplant": fast_weight_learning_state["controls"][
                            "trajectory_transplant"
                        ],
                        "supervised_trajectory_map": fast_weight_learning_state[
                            "controls"
                        ]["supervised_trajectory_map"],
                        "output_associative_memory": fast_weight_learning_state["controls"][
                            "output_associative_memory"
                        ],
                        "verifier_gain_search": fast_weight_learning_state["controls"][
                            "verifier_gain_search"
                        ],
                        "test_time_training": fast_weight_learning_state["controls"][
                            "test_time_training"
                        ],
                    }
                else:
                    fast_weight_learning_state["controls"] = {
                        "decision": "accepted",
                        "capability_canaries": {},
                        "trajectory_transplant": fast_weight_learning_state["controls"][
                            "trajectory_transplant"
                        ],
                        "supervised_trajectory_map": fast_weight_learning_state[
                            "controls"
                        ]["supervised_trajectory_map"],
                        "output_associative_memory": fast_weight_learning_state["controls"][
                            "output_associative_memory"
                        ],
                        "verifier_gain_search": fast_weight_learning_state["controls"][
                            "verifier_gain_search"
                        ],
                        "test_time_training": fast_weight_learning_state["controls"][
                            "test_time_training"
                        ],
                    }
                if (
                    fast_weight_candidate_verifier is not None
                    and self.tokenizer is not None
                ):
                    verifier_decision = self._enforce_fast_weight_verifier(
                        fast_weight_candidate_verifier,
                        fast_weights,
                        winner,
                        cache,
                        runner,
                        budget,
                        bridge_tokens,
                        receipt,
                        pre_score=fw_verifier_pre,
                        pre_tokens=fw_verifier_pre_tokens,
                        pre_text=fw_verifier_pre_text,
                        learning_state=fast_weight_learning_state,
                        safety_reserve=safety_reserve,
                        treatment_trace=fw_treatment_trace,
                        treatment_target_tokens=fast_weight_target_tokens,
                        sham_trace=fw_sham_trace,
                        sham_target_tokens=fw_sham_target_tokens,
                        sham_probe_tokens=fw_sham_probe_tokens,
                        sham_score=fw_sham_score,
                    )
                    stage_started = self._stage_checkpoint(
                        receipt=receipt,
                        budget=budget,
                        stage="fast_weight_verifier",
                        stage_started=stage_started,
                        episode_started=episode_started,
                        progress=progress,
                        cancel_check=cancel_check,
                        decision=verifier_decision,
                    )
                else:
                    fast_weights.canary_erase()
                    fast_weight_learning_state["disposition"] = "rejected_verifier_unavailable"

                fast_weight_decode_active = bool(
                    fast_weight_learning_state["disposition"] == "accepted_causal_improvement"
                    and fast_weights.handles
                    and fast_weights.lifecycle.lease_acquired
                    and not fast_weights.lifecycle.lease_released
                )

            # Experiment-3 instrumentation: destroy one refined thought slot
            # just before persistence, so its causal contribution and
            # restoration are measurable.
            if ablate_slot is not None:
                winner.workspace.ablate(int(ablate_slot), mode=ablate_mode)
                winner.z = winner.workspace.z
                receipt.flag(f"slot_ablated:{int(ablate_slot)}:{ablate_mode}")

            # The matched causal probe is valid only for the exact winner
            # state it measured. Any later experimental mutation, unexpected
            # lease release, or wrapper loss removes authority to decode under
            # the adaptation; continue on the clean base function instead.
            if fast_weight_decode_active and fast_weights is not None:
                lineage_intact = (
                    tensor_sha256(winner.z) == fast_weight_learning_state["winner_state_sha256"]
                    and bool(fast_weights.handles)
                    and fast_weights.lifecycle.lease_acquired
                    and not fast_weights.lifecycle.lease_released
                )
                if not lineage_intact:
                    if fast_weights.handles:
                        fast_weights.canary_erase()
                    fast_weight_learning_state["disposition"] = "rejected_state_lineage_changed"
                    fast_weight_decode_active = False
                    receipt.flag("fast_weight_state_lineage_changed")

            # Branch probes were captured before latent optimization and
            # fast-weight adaptation. Refresh the selected candidate from the
            # exact state that is about to decode, or remove the stale entry;
            # downstream replacement and research arbitration must never
            # grade the pre-adaptation snapshot as the adapted computation.
            adaptation_changed_candidate_state = bool(
                receipt.latent_opt_steps > 0 or fast_weight_decode_active or ablate_slot is not None
            )
            deployable_candidate_available = winner.index in branch_probe_texts
            research_candidate_available = winner.index in research_oracle_candidates
            research_oracle_enabled = callable(
                getattr(
                    pending_verifier,
                    "research_oracle_assessment",
                    None,
                )
            )
            if adaptation_changed_candidate_state and (
                deployable_candidate_available
                or research_candidate_available
                or research_oracle_enabled
            ):
                from core.brain.llm.latent_cortex.post_adaptation_candidate import (
                    advance_post_adaptation_candidate,
                    build_post_adaptation_candidate_receipt,
                )

                post_probe_cost = self._verifier_probe_layer_apps(bridge_tokens)
                if post_probe_cost + safety_reserve > budget.remaining_layer_apps:
                    branch_probe_texts.pop(winner.index, None)
                    research_oracle_candidates.pop(winner.index, None)
                    receipt.flag("post_adaptation_candidate_unmeasured_budget")
                else:
                    prior_candidate = (
                        branch_probe_texts[winner.index]
                        if deployable_candidate_available
                        else research_oracle_candidates.get(winner.index)
                    )
                    post_probe_tokens = self._decode_probe(
                        winner,
                        cache,
                        runner,
                        budget,
                        bridge_tokens=bridge_tokens,
                        use_cache=False,
                        force_exact_tokens=True,
                    )
                    post_probe_text = self._decode_public_text(
                        post_probe_tokens,
                        receipt=receipt,
                    )
                    transition, admitted_candidate = advance_post_adaptation_candidate(
                        selected_branch=winner.index,
                        prior_candidate=prior_candidate,
                        observed_candidate=post_probe_text,
                        stage="post_final_adaptation",
                        strict_answer_contract=(
                            "final_answer_v1"
                            in {
                                self.config.decode_contract,
                                self.config.verifier_probe_contract,
                            }
                        ),
                        response_contract=verification_response_contract,
                        adaptation_evidence={
                            "latent_opt_attempts": receipt.latent_opt_attempts,
                            "latent_opt_accepted_steps": receipt.latent_opt_steps,
                            "fast_weight_disposition": (
                                str(fast_weight_learning_state.get("disposition", "not_applied"))
                                if fast_weight_learning_state is not None
                                else "not_applied"
                            ),
                            "fast_weight_decode_active": fast_weight_decode_active,
                            "slot_ablation_applied": ablate_slot is not None,
                            "prior_disagreement_sha256": str(
                                receipt.disagreement_graph.get("receipt_sha256", "")
                            ),
                            "prior_diagnostic_sha256": str(
                                receipt.diagnostic_action_selection.get("receipt_sha256", "")
                            ),
                            "prior_local_repair_sha256": str(
                                receipt.local_repair.get("receipt_sha256", "")
                            ),
                            "prior_blind_review_sha256": str(
                                receipt.blind_review.get("receipt_sha256", "")
                            ),
                        },
                    )
                    receipt.post_adaptation_candidate = build_post_adaptation_candidate_receipt(
                        [transition]
                    )
                    if admitted_candidate is None:
                        if deployable_candidate_available:
                            branch_probe_texts.pop(winner.index, None)
                        research_oracle_candidates.pop(winner.index, None)
                        receipt.flag("post_adaptation_candidate_contract_rejected")
                    else:
                        if deployable_candidate_available:
                            branch_probe_texts[winner.index] = admitted_candidate
                        if research_oracle_enabled:
                            research_oracle_candidates[winner.index] = admitted_candidate

                    # Confidence-bound replacement validates candidate text
                    # against these exact receipts. Rebuild them from the
                    # refreshed inventory; pre-adaptation decompositions and
                    # generated repairs cannot transfer to changed text.
                    from core.brain.llm.latent_cortex.diagnostic_action_selector import (
                        build_candidate_routes,
                        build_diagnostic_action_selector_receipt,
                    )
                    from core.brain.llm.latent_cortex.disagreement_graph import (
                        build_disagreement_graph_receipt,
                        decompose_branch_candidates,
                    )
                    from core.brain.llm.latent_cortex.local_repair import (
                        build_local_repair_receipt,
                    )

                    post_candidate_decompositions: dict[str, dict[str, Any]] = {}
                    if len(branch_probe_texts) == len(ensemble.branches):
                        try:
                            from core.brain.llm.latent_cortex.blind_review import (
                                run_decoy_balanced_review,
                            )

                            (
                                blind_scores,
                                receipt.blind_review,
                                receipt.decoy_verification,
                            ) = run_decoy_balanced_review(
                                branch_probe_texts,
                                pending_verifier,
                                episode_id=receipt.episode_id,
                                objective_sha256=receipt.input_tokens_sha256,
                                isolation_receipt=ensemble.isolation_receipt(
                                    runner.cache_discipline_receipt()
                                ),
                            )
                            post_candidate_decompositions = decompose_branch_candidates(
                                branch_probe_texts,
                                objective=verification_objective,
                            )
                        except (TypeError, ValueError) as exc:
                            receipt.flag(
                                f"post_adaptation_candidate_evidence_invalid:{type(exc).__name__}"
                            )
                    receipt.disagreement_graph = build_disagreement_graph_receipt(
                        n_branches=len(ensemble.branches),
                        operator_trace=receipt.cognitive_operator_trace,
                        action_trace=receipt.cognitive_action_trace,
                        structural_diversity=receipt.structural_diversity,
                        candidate_decompositions=post_candidate_decompositions,
                        blind_review=receipt.blind_review,
                    )
                    post_candidate_routes = (
                        build_candidate_routes(
                            branch_probe_texts,
                            objective=verification_objective,
                            candidate_decompositions=(post_candidate_decompositions),
                        )
                        if post_candidate_decompositions
                        else {}
                    )
                    receipt.diagnostic_action_selection = build_diagnostic_action_selector_receipt(
                        disagreement_graph=receipt.disagreement_graph,
                        candidate_routes=post_candidate_routes,
                        action_policy_evidence=action_policy_evidence,
                        value_policy=receipt.value_of_computation,
                        action_trace=receipt.cognitive_action_trace,
                    )
                    repair_limit = (
                        self.config.local_repair_max_attempts
                        if self.config.local_repair_enabled
                        else 0
                    )
                    receipt.local_repair = build_local_repair_receipt(
                        disagreement_graph=receipt.disagreement_graph,
                        diagnostic_selection=(receipt.diagnostic_action_selection),
                        branch_candidates=branch_probe_texts,
                        objective=verification_objective,
                        generated_repairs={},
                        execution_failures={},
                        max_requests=repair_limit,
                    )

            # ── Commit the winner + decode the answer ────────────────────
            # ``bridge_tokens`` is what the model actually reads before the
            # answer. It has two independent sources, and conflating them let a
            # ``decode_bridge_policy="none"`` episode publish
            # ``decode_bridge_applied=True`` with a 43-token count — the policy
            # bridge was empty and the terminal-disposition language supplied
            # every one of those tokens. Count the policy bridge on its own so
            # the bridge receipt answers for the policy alone.
            policy_bridge_token_count = len(bridge_tokens)
            if terminal_instruction_tokens:
                bridge_tokens.extend(terminal_instruction_tokens)
            final_fusion_audit = None
            final_decode_transaction = None
            latent_decode_authorized = self.config.decode_incumbent_policy == "latent"
            public_bridge_tokens = list(bridge_tokens) if latent_decode_authorized else []
            effective_terminal_instruction_tokens = (
                list(terminal_instruction_tokens) if latent_decode_authorized else []
            )
            effective_terminal_instruction_policy = (
                self.config.terminal_instruction_policy
                if latent_decode_authorized
                else "suppressed"
            )
            heterogeneous_decode_applied = bool(
                heterogeneous_fusion_context is not None and latent_decode_authorized
            )
            if heterogeneous_decode_applied:

                def checkpoint_fusion_phase(stage: str) -> None:
                    nonlocal stage_started
                    detail = (
                        {
                            "bridge_policy": self.config.decode_bridge_policy,
                            "bridge_tokens": len(bridge_tokens),
                        }
                        if stage == "decode_bridge"
                        else {}
                    )
                    stage_started = self._stage_checkpoint(
                        receipt=receipt,
                        budget=budget,
                        stage=stage,
                        stage_started=stage_started,
                        episode_started=episode_started,
                        progress=progress,
                        cancel_check=cancel_check,
                        **detail,
                    )

                (
                    fusion_incumbent_state,
                    fusion_corrected_state,
                    fusion_weight,
                ) = heterogeneous_fusion_context
                (
                    out_tokens,
                    decode_termination,
                    final_fusion_audit,
                ) = self._heterogeneous_dual_lane_decode(
                    branch=winner,
                    cache=cache,
                    runner=runner,
                    budget=budget,
                    incumbent_state=fusion_incumbent_state,
                    corrected_state=fusion_corrected_state,
                    policy="probability_fusion",
                    fusion_weight=fusion_weight,
                    bridge_tokens=bridge_tokens,
                    max_tokens=decode_limit,
                    temperature=float(self.config.decode_temperature),
                    force_exact_tokens=False,
                    cancel_check=cancel_check,
                    progress=progress,
                    sentence_grace_tokens=decode_sentence_grace_tokens,
                    contract_grace_tokens=None,
                    wall_reserve_s=0.0,
                    token_logprobs_out=token_logprobs_out,
                    phase_checkpoint=checkpoint_fusion_phase,
                    retain_lanes=True,
                )
                receipt.first_logits_digest = final_fusion_audit["policy_initial_logits_sha256"]
                receipt.flag("heterogeneous_fusion_decode_applied")
            elif latent_decode_authorized:
                persist_transaction = kv_state_tree.begin_speculation(
                    cache,
                    start=0,
                    end=self.n_layers,
                    purpose="final_winner_persist",
                    branch_index=winner.index,
                    parent_sha256=(winner.kv_boundary_sha256 or kv_state_tree.root_sha256),
                )
                slot_logits = self._persist_branch(
                    winner,
                    cache,
                    runner,
                    budget,
                )
                persist_transaction.observe_mutation(cache)
                winner.kv_boundary_sha256 = persist_transaction.commit(
                    label="winner_workspace_persisted",
                    authority="selected_branch_commit",
                    latent_sha256=tensor_sha256(winner.z),
                    final=False,
                )
                final_decode_transaction = kv_state_tree.begin_speculation(
                    cache,
                    start=0,
                    end=self.n_layers,
                    purpose="final_output_decode",
                    branch_index=winner.index,
                    parent_sha256=winner.kv_boundary_sha256,
                )
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
            else:
                # Recurrence, branch selection, verification, and learning may
                # still run and remain fully receipted, but they do not own the
                # public answer until an independent gain gate promotes them.
                # Restore the immutable prompt root before opening the output
                # transaction. This is the checkpoint's ordinary decode lane,
                # not a latent reconstruction of it.
                if fast_weight_decode_active and fast_weights is not None:
                    fast_weights.canary_erase()
                    fast_weight_decode_active = False
                    if fast_weight_learning_state is not None:
                        fast_weight_learning_state["disposition"] = (
                            "accepted_probe_not_output_under_incumbent_policy"
                        )
                kv_state_tree.restore_boundary(cache, kv_state_tree.root_sha256)
                if incumbent_artifact is not None:
                    out_tokens = list(incumbent_artifact.tokens)
                    decode_termination = str(incumbent_artifact.receipt["output"]["termination"])
                    final_decode_transaction = kv_state_tree.begin_speculation(
                        cache,
                        start=0,
                        end=self.n_layers,
                        purpose="bind_canonical_incumbent_artifact",
                        branch_index=winner.index,
                        parent_sha256=kv_state_tree.root_sha256,
                    )
                    # The artifact was generated on the paired ordinary lane;
                    # this transaction binds its selection without pretending
                    # that the RLC regenerated or mutated those bytes.
                    final_decode_transaction.observe_mutation(cache)
                    winner.kv_boundary_sha256 = final_decode_transaction.commit(
                        label="bound_vanilla_incumbent_output",
                        authority="canonical_ordinary_decode_artifact",
                        latent_sha256="",
                        final=True,
                    )
                else:
                    final_decode_transaction = kv_state_tree.begin_speculation(
                        cache,
                        start=0,
                        end=self.n_layers,
                        purpose="final_vanilla_incumbent_decode",
                        branch_index=winner.index,
                        parent_sha256=kv_state_tree.root_sha256,
                    )
                    decode_logits = prompt_tail_logits
                receipt.first_logits_digest = receipt.decode_incumbent_prompt_logits_sha256
                stage_started = self._stage_checkpoint(
                    receipt=receipt,
                    budget=budget,
                    stage="incumbent_restore",
                    stage_started=stage_started,
                    episode_started=episode_started,
                    progress=progress,
                    cancel_check=cancel_check,
                    decode_authority="vanilla_incumbent",
                )
            receipt.decode_prefix_token_count = len(public_bridge_tokens)
            receipt.decode_prefix_composition = {
                "policy_bridge_tokens": (
                    policy_bridge_token_count if latent_decode_authorized else 0
                ),
                "candidate_probe_bridge_tokens": policy_bridge_token_count,
                "terminal_instruction_tokens": len(effective_terminal_instruction_tokens),
                "terminal_instruction_policy": effective_terminal_instruction_policy,
                "configured_terminal_instruction_policy": (self.config.terminal_instruction_policy),
            }
            if public_bridge_tokens:
                serialized_bridge = json.dumps(
                    public_bridge_tokens,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("ascii")
                receipt.decode_bridge_applied = policy_bridge_token_count > 0
                receipt.decode_bridge_token_count = policy_bridge_token_count
                receipt.decode_bridge_tokens_sha256 = hashlib.sha256(serialized_bridge).hexdigest()
                if heterogeneous_decode_applied:
                    receipt.decode_bridge_logits_digest = final_fusion_audit[
                        "policy_initial_logits_sha256"
                    ]
                elif latent_decode_authorized:
                    # Only the latent lane may be steered by a decode prefix.
                    #
                    # Under vanilla_incumbent the branch above deliberately
                    # restores the immutable prompt root and sets
                    # decode_logits = prompt_tail_logits so the public answer
                    # IS ordinary decode. Applying the bridge here overwrote
                    # that: the 27-token terminal disposition was appended to
                    # the cache, so the "incumbent" conditioned on prompt +
                    # instruction and was not ordinary decode at all. Measured
                    # consequence: 14 of 14 full-stack answers differed from
                    # vanilla's, and the arm scored 3/14 against vanilla's
                    # 5/14 -- below a floor that was supposed to be structural.
                    #
                    # A disposition-conditioned answer is not forbidden; it is
                    # simply not free. It must win as a candidate under the
                    # same lower-bound-dominance rule as any other, instead of
                    # being imposed on the incumbent before anything is
                    # measured.
                    decode_logits = self._apply_decode_bridge(
                        cache,
                        budget,
                        public_bridge_tokens,
                    )
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
            elif latent_decode_authorized:
                decode_logits = slot_logits
            if not heterogeneous_decode_applied and incumbent_artifact is None:
                out_tokens, decode_termination = self._decode(
                    cache,
                    budget,
                    decode_logits,
                    max_tokens=decode_max_tokens,
                    cancel_check=cancel_check,
                    progress=progress,
                    # Cleanup time is sacrosanct: with temporary synapses
                    # attached the decode surrenders its tail rather than let
                    # the wall clock expire before the erase proof.
                    wall_reserve_s=(6.0 if fast_weights is not None else 0.0),
                    token_logprobs_out=token_logprobs_out,
                    sentence_grace_tokens=decode_sentence_grace_tokens,
                    sample_seed=sample_seed,
                    coda_adapter_active=latent_decode_authorized,
                )
                final_decode_transaction.observe_mutation(cache)
                winner.kv_boundary_sha256 = final_decode_transaction.commit(
                    label="final_output_lane",
                    authority=(
                        "confidence_bound_output_candidate"
                        if latent_decode_authorized
                        else "vanilla_incumbent_output"
                    ),
                    latent_sha256=(tensor_sha256(winner.z) if latent_decode_authorized else ""),
                    final=True,
                )
            receipt.decode_requested_tokens = decode_limit
            receipt.decode_generated_tokens = len(out_tokens)
            receipt.decode_termination = decode_termination
            receipt.decode_contract_satisfied = bool(self._last_decode_contract_satisfied)
            receipt.decode_contract_grace_used_tokens = int(
                self._last_decode_contract_grace_used_tokens
            )
            receipt.decode_newline_suppressions = int(self._last_decode_newline_suppressions)
            receipt.decode_repetition_penalty_applied = float(self.config.decode_repetition_penalty)
            if heterogeneous_finalized and final_fusion_audit is not None:
                from core.brain.llm.latent_cortex.heterogeneous_integrator import (
                    build_heterogeneous_decode_receipt,
                )

                receipt.heterogeneous_decode = build_heterogeneous_decode_receipt(
                    integration=receipt.heterogeneous_integration,
                    output_tokens=out_tokens,
                    termination=decode_termination,
                    first_logits_sha256=(receipt.first_logits_digest),
                    fusion_audit=final_fusion_audit,
                )
            if receipt.cognitive_operator_trace:
                from core.brain.llm.latent_cortex.answer_replacement import (
                    MAX_REPLACEMENT_OUTPUT_TOKENS,
                    build_answer_replacement_receipt,
                )

                baseline_text = (
                    self._decode_public_text(out_tokens, receipt=receipt) if out_tokens else ""
                )

                def encode_replacement(value: str) -> list[int]:
                    if self.tokenizer is None:
                        raise ValueError("replacement tokenizer is unavailable")
                    try:
                        encoded = self.tokenizer.encode(
                            value,
                            add_special_tokens=False,
                        )
                    except TypeError:
                        encoded = self.tokenizer.encode(value)
                    return list(encoded)

                replacement_output_limit = min(
                    MAX_REPLACEMENT_OUTPUT_TOKENS,
                    int(decode_limit)
                    + (
                        int(self.config.decode_contract_grace_tokens)
                        if receipt.decode_contract_required
                        else int(
                            _SENTENCE_GRACE_TOKENS
                            if decode_sentence_grace_tokens is None
                            else decode_sentence_grace_tokens
                        )
                    ),
                )
                (
                    replacement_receipt,
                    accepted_tokens,
                    answer_replacement_private,
                ) = build_answer_replacement_receipt(
                    disagreement_graph=receipt.disagreement_graph,
                    diagnostic_selection=receipt.diagnostic_action_selection,
                    local_repair=receipt.local_repair,
                    selected_branch=winner.index,
                    branch_candidates=branch_probe_texts,
                    generated_repairs=generated_repairs,
                    objective=verification_objective,
                    baseline_text=baseline_text,
                    baseline_tokens=out_tokens,
                    encode=encode_replacement,
                    decode=lambda values: self._decode_public_text(
                        values,
                        receipt=receipt,
                    ),
                    # Do NOT couple this to latent_decode_authorized.
                    #
                    # That coupling made a win structurally impossible. Under
                    # "latent" the recurrent path owns the answer outright, so
                    # there is no floor and the episode can score far below
                    # ordinary decode. Under "vanilla_incumbent" the floor
                    # holds -- and replacement was force-disabled, so the
                    # episode was exactly ordinary decode at several times the
                    # cost, incapable of improving on it. Neither policy could
                    # both keep the floor and gain, which is why this path had
                    # never beaten vanilla.
                    #
                    # The lower-bound-dominance rule exists precisely to
                    # promote safely FROM a safe baseline: it replaces only
                    # when a candidate's lower confidence bound clears the
                    # incumbent's upper bound plus a margin. Gating it on
                    # "latent already owns the output" let it fire only once it
                    # was no longer needed. Everything it needs is already
                    # built here under either policy -- the incumbent answer as
                    # baseline, the branch probes and repairs as candidates.
                    #
                    # This is also what the incumbent branch's own comment
                    # promised: recurrence and verification "do not own the
                    # public answer until an independent gain gate promotes
                    # them". This is that gate.
                    enabled=self.config.answer_replacement_enabled,
                    objective_program_enabled=self.config.objective_program_enabled,
                    margin=self.config.answer_replacement_margin,
                    max_output_tokens=replacement_output_limit,
                )
                receipt.answer_replacement = replacement_receipt
                decision = replacement_receipt["decision"]
                if decision == "replace":
                    out_tokens = accepted_tokens
                    if token_logprobs_out is not None:
                        token_logprobs_out.clear()
                    decode_termination = "confidence_bound_replacement"
                    receipt.decode_contract_grace_used_tokens = max(
                        0,
                        len(out_tokens) - int(decode_limit),
                    )
                    receipt.flag("confidence_bound_answer_replaced")
                    if receipt.decode_contract_required:
                        from core.brain.llm.latent_cortex.answer_contract import (
                            is_contract_complete,
                        )

                        receipt.decode_contract_satisfied = is_contract_complete(
                            self._decode_public_text(out_tokens, receipt=receipt)
                        )
                elif decision == "abstain":
                    if latent_decode_authorized:
                        out_tokens = []
                        if token_logprobs_out is not None:
                            token_logprobs_out.clear()
                        decode_termination = "confidence_bound_abstention"
                        receipt.flag("confidence_bound_answer_abstained")
                    else:
                        # Under vanilla_incumbent the answer on the table IS
                        # ordinary decode, so emptying it cannot be an
                        # improvement. Abstention fires when the verifier
                        # believes the baseline refuted and nothing better
                        # exists -- which rests entirely on the verifier being
                        # right. A false refutation then discards an answer
                        # ordinary decode got correct, and a true one costs
                        # nothing to serve because vanilla was wrong anyway.
                        # Empty is worse than the incumbent when the incumbent
                        # is right and merely equal when it is not, so
                        # retaining is weakly better in every case.
                        #
                        # Measured: this emptied 1 of 3 probe answers outright
                        # (vanilla 77 chars, full stack 0) while the arm scored
                        # 3/14 against vanilla's 5/14 -- below a floor that was
                        # supposed to be structural.
                        receipt.flag("confidence_bound_abstention_declined_under_incumbent")
                receipt.decode_generated_tokens = len(out_tokens)
                receipt.decode_termination = decode_termination
                research_oracle = getattr(
                    pending_verifier,
                    "research_oracle_assessment",
                    None,
                )
                if callable(research_oracle):
                    # This is the hidden-answer diagnostic arm, not a serving
                    # policy. It answers whether recurrence GENERATED a
                    # correct candidate that the deployable evidence stack did
                    # not promote. The receipt says so explicitly and the live
                    # worker never constructs a verifier with this method.
                    oracle_branch = (
                        winner.index
                        if winner.index in research_oracle_candidates
                        else min(research_oracle_candidates, default=-1)
                    )
                    # Hidden-ground-truth selection is the diagnostic
                    # intervention being measured. Prefer any exact-correct
                    # valid candidate; otherwise assess the latent winner (or
                    # first valid survivor) and retain the current output.
                    for branch_index, candidate_text in sorted(research_oracle_candidates.items()):
                        try:
                            assessment = research_oracle(candidate_text)
                        except (TypeError, ValueError):
                            continue
                        if isinstance(assessment, Mapping) and assessment.get("correct") is True:
                            oracle_branch = branch_index
                            break
                    recurrent_text = research_oracle_candidates.get(oracle_branch)
                    if not isinstance(recurrent_text, str) or not recurrent_text:
                        receipt.flag("research_oracle_selected_candidate_unavailable")
                    else:
                        try:
                            recurrent_tokens = encode_replacement(recurrent_text)
                            if (
                                not recurrent_tokens
                                or self._decode_public_text(
                                    recurrent_tokens,
                                    receipt=receipt,
                                )
                                != recurrent_text
                            ):
                                raise ValueError("research oracle recurrent output binding failed")
                            from core.brain.llm.latent_cortex.research_oracle_arbitration import (
                                build_research_oracle_arbitration,
                            )

                            current_text = self._decode_public_text(
                                out_tokens,
                                receipt=receipt,
                            )
                            (
                                receipt.research_oracle_arbitration,
                                oracle_tokens,
                            ) = build_research_oracle_arbitration(
                                current_text=current_text,
                                current_tokens=out_tokens,
                                recurrent_text=recurrent_text,
                                recurrent_tokens=recurrent_tokens,
                                selected_branch=oracle_branch,
                                assess=research_oracle,
                            )
                        except (TypeError, ValueError) as exc:
                            receipt.flag(
                                "research_oracle_arbitration_rejected:"
                                f"{type(exc).__name__}:{str(exc)[:80]}"
                            )
                        else:
                            if receipt.research_oracle_arbitration["decision"] == "replace":
                                out_tokens = oracle_tokens
                                if token_logprobs_out is not None:
                                    token_logprobs_out.clear()
                                decode_termination = "research_oracle_replacement"
                                receipt.flag("research_oracle_answer_replaced")
                            receipt.decode_generated_tokens = len(out_tokens)
                            receipt.decode_termination = decode_termination
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
                self._finalize_fast_weights(
                    fast_weights,
                    fw_baseline,
                    receipt,
                    budget,
                    learning_state=fast_weight_learning_state,
                )

        if fast_weight_learning_state is not None:
            final_text = self._decode_public_text(out_tokens, receipt=receipt)
            fast_weight_learning_state["final_answer"] = {
                "decoded_under_adaptation": fast_weight_decode_active,
                "tokens_sha256": token_sequence_sha256(out_tokens),
                "text_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
                "token_count": len(out_tokens),
            }
            receipt.fast_weight_learning = finalize_fast_weight_learning_receipt(
                fast_weight_learning_state
            )

        if fast_weights is not None and receipt.fast_weights_erased is not True:
            raise _FastWeightCleanupError("fast-weight cleanup proof did not pass")

        receipt.recurrence_adapter = runner.adapter_receipt()
        from core.brain.llm.latent_cortex.kv_state_tree import (
            validate_kv_state_tree_receipt,
        )

        receipt.kv_state_tree = validate_kv_state_tree_receipt(
            kv_state_tree.receipt(),
            episode_id=receipt.episode_id,
            input_tokens_sha256=receipt.input_tokens_sha256,
            n_layers=self.n_layers,
            expected_n_branches=self.config.branches.n_branches,
            require_final=True,
        )
        receipt.branch_isolation = ensemble.isolation_receipt(runner.cache_discipline_receipt())
        if (
            self.config.branches.n_branches > 1
            and receipt.branch_isolation.get("certified") is not True
        ):
            receipt.flag("branch_isolation_unproven")
        from core.brain.llm.latent_cortex.branch_exchange import (
            build_branch_exchange_trace,
        )

        # An empty, validated trace proves no declared synchronization point
        # fired. Leaving the field empty cannot distinguish that valid outcome
        # from disabled or skipped exchange instrumentation.
        receipt.branch_exchange = build_branch_exchange_trace(
            exchanges=ensemble.exchange_receipts,
            n_branches=len(ensemble.branches),
            n_slots=int(self.config.workspace.n_slots),
            comm_slot=int(self.config.branches.comm_slot),
            exchange_gamma=float(self.config.branches.exchange_gamma),
            branch_isolation=receipt.branch_isolation,
            cognitive_slots=receipt.cognitive_slots,
            exchange_interval=int(self.config.branches.exchange_interval),
            schedule_hash=receipt.schedule_hash,
            bytecode_events=receipt.bytecode_events,
            cognitive_action_trace=receipt.cognitive_action_trace,
        )
        if receipt.cognitive_operator_trace:
            from core.brain.llm.latent_cortex.correlated_support import (
                build_correlated_support_receipt,
            )

            receipt.correlated_support = build_correlated_support_receipt(
                structural_diversity=receipt.structural_diversity,
                correlation_evidence=correlation_evidence,
            )
        receipt.latent_telemetry = telemetry.to_receipt()
        if self._episode_probe_cache is not None:
            receipt.probe_cache = self._episode_probe_cache.to_receipt()
        if terminal_decision is None:
            raise RuntimeError("terminal disposition was not classified")
        from core.brain.llm.latent_cortex.terminal_disposition import (
            finalize_terminal_disposition_receipt,
        )

        output_text = self._decode_public_text(out_tokens, receipt=receipt)
        receipt.terminal_disposition = finalize_terminal_disposition_receipt(
            terminal_decision,
            instruction_tokens=effective_terminal_instruction_tokens,
            instruction_policy=effective_terminal_instruction_policy,
            full_bridge_tokens=public_bridge_tokens,
            output_tokens=out_tokens,
            output_text=output_text,
            output_source=(
                "research_oracle_candidate"
                if receipt.decode_termination == "research_oracle_replacement"
                else "resident_model_repair"
                if receipt.decode_termination == "confidence_bound_replacement"
                else "resident_model_decode"
                if self.tokenizer is not None
                else "substrate_model_decode"
            ),
        )
        return out_tokens, receipt, answer_replacement_private

    def _finalize_fast_weights(
        self,
        fast_weights: EpisodicFastWeights,
        baseline,
        receipt: EpisodeReceipt,
        budget: ComputeBudget,
        *,
        learning_state: dict[str, Any] | None = None,
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
        receipt.fast_weight_gradient_norm_trail = list(lifecycle.gradient_global_norm_trail)
        receipt.fast_weight_accepted_step_sizes = list(lifecycle.accepted_step_sizes)
        receipt.fast_weight_line_search_backtracks = lifecycle.line_search_backtracks
        if learning_state is not None:
            learning_state["lease"] = fast_weights.lease_receipt()
            learning_state["optimization"] = {
                "optimizer": lifecycle.optimizer,
                "attempts": lifecycle.optimization_attempts,
                "accepted_steps": lifecycle.optimized_steps,
                "rejected_steps": lifecycle.rejected_steps,
                "budget_exhausted": lifecycle.budget_exhausted,
                "loss_trail": list(lifecycle.loss_trail),
                "gradient_norm_trail": list(lifecycle.gradient_global_norm_trail),
                "accepted_step_sizes": list(lifecycle.accepted_step_sizes),
                "line_search_backtracks": lifecycle.line_search_backtracks,
            }
            learning_state["cleanup"] = {
                "required": True,
                "detached": lifecycle.erased,
                "erase_proven": lifecycle.erase_proven,
                "lease_released": lifecycle.lease_released,
                "conflicts": (lifecycle.detach_conflicts + lifecycle.lease_conflicts),
                "pre_probe_sha256": (lifecycle.erase_probe_before_sha256),
                "post_probe_sha256": (lifecycle.erase_probe_after_sha256),
                "erased_layer_ids": [
                    f"layers.{index}.{lifecycle.target}" for index in lifecycle.layers
                ],
            }
        try:
            from core.brain.llm.latent_cortex.runtime_integrity import (
                build_fast_weight_cleanup_proof,
            )

            receipt.fast_weight_cleanup = build_fast_weight_cleanup_proof(
                episode_id=receipt.episode_id,
                input_tokens_sha256=receipt.input_tokens_sha256,
                detached=lifecycle.erased,
                erase_proven=lifecycle.erase_proven,
                lease_released=lifecycle.lease_released,
                conflicts=(lifecycle.detach_conflicts + lifecycle.lease_conflicts),
                pre_probe_sha256=lifecycle.erase_probe_before_sha256,
                post_probe_sha256=lifecycle.erase_probe_after_sha256,
                layer_ids=[f"layers.{index}.{lifecycle.target}" for index in lifecycle.layers],
            )
        except (TypeError, ValueError) as exc:
            receipt.fast_weight_cleanup = {}
            receipt.flag(f"fast_weight_cleanup_proof_failed:{type(exc).__name__}")
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
            # No checkpoint fingerprint ⇒ no provenance ⇒ never evidence.
            # This keeps anonymous-model episodes (tests, ad-hoc engines
            # built without model_path) out of the LIVE queue — three
            # hidden-size-64 candidates leaked in on Jul 16 and crashed the
            # first real 32B consolidation train at attach.
            and receipt.checkpoint_fingerprint
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

    # ── Learned halting attachment (CP234 seam made loadable) ──────────
    def _resolve_halting_head(self):
        """Load the pinned calibrated public-signal stop policy."""

        from core.brain.llm.latent_cortex.stop_gate import StopGateRuntime

        halting = self.config.halting or {}
        if str(halting.get("mode", "residual")) != "learned":
            return StopGateRuntime.from_config(halting)
        head_path = Path(str(halting.get("head_path", ""))).expanduser()
        try:
            stat = head_path.stat()
        except OSError as exc:
            raise ValueError(
                f"learned halting requested but head is unreadable: {head_path}"
            ) from exc
        cache_key = (
            str(head_path),
            str(halting.get("head_sha256", "")),
            stat.st_mtime_ns,
            stat.st_size,
        )
        cached = getattr(self, "_halting_head_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        try:
            gate = StopGateRuntime.from_config(halting)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"learned halting head failed to load: {head_path}") from exc
        self._halting_head_cache = (cache_key, gate)
        return gate

    def _resolve_update_gate(self):
        """Load the pinned calibrated per-transition admission head."""

        from core.brain.llm.latent_cortex.update_gate import UpdateGateRuntime

        config = self.config.update_gate
        if not config or str(config.get("mode", "passthrough")) == "passthrough":
            return UpdateGateRuntime.from_config(config)
        path = Path(str(config.get("head_path", ""))).expanduser()
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError(
                f"learned update gate requested but head is unreadable: {path}"
            ) from exc
        cache_key = (
            str(path),
            str(config.get("head_sha256", "")),
            stat.st_mtime_ns,
            stat.st_size,
        )
        cached = getattr(self, "_update_gate_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        gate = UpdateGateRuntime.from_config(config)
        self._update_gate_cache = (cache_key, gate)
        return gate

    def _resolve_uncertainty_head(self):
        """Load the pinned calibrated hidden-state correctness head."""

        from core.brain.llm.latent_cortex.neural_uncertainty import (
            NeuralUncertaintyRuntime,
        )

        config = self.config.uncertainty_head
        if not config or str(config.get("mode", "unavailable")) == "unavailable":
            return NeuralUncertaintyRuntime.from_config(config)
        path = Path(str(config.get("head_path", ""))).expanduser()
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError(
                f"learned uncertainty requested but head is unreadable: {path}"
            ) from exc
        cache_key = (
            str(path),
            str(config.get("head_sha256", "")),
            stat.st_mtime_ns,
            stat.st_size,
        )
        cached = getattr(self, "_uncertainty_head_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        runtime = NeuralUncertaintyRuntime.from_config(config)
        self._uncertainty_head_cache = (cache_key, runtime)
        return runtime

    def _resolve_mistake_locator(self):
        """Load the pinned, OOD-admitted transition mistake locator."""

        from core.brain.llm.latent_cortex.mistake_locator import (
            MistakeLocatorRuntime,
        )

        config = self.config.mistake_locator
        if not config or str(config.get("mode", "unavailable")) == "unavailable":
            return MistakeLocatorRuntime.from_config(config)
        path = Path(str(config.get("head_path", ""))).expanduser()
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError(
                f"learned mistake locator requested but head is unreadable: {path}"
            ) from exc
        cache_key = (
            str(path),
            str(config.get("head_sha256", "")),
            stat.st_mtime_ns,
            stat.st_size,
        )
        cached = getattr(self, "_mistake_locator_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        runtime = MistakeLocatorRuntime.from_config(config)
        self._mistake_locator_cache = (cache_key, runtime)
        return runtime

    def _resolve_contradiction_head(self):
        """Load the pinned, full-trace contradiction tensor head."""

        from core.brain.llm.latent_cortex.contradiction_tensor import (
            ContradictionTensorRuntime,
        )

        config = self.config.contradiction_head
        if not config or str(config.get("mode", "unavailable")) == "unavailable":
            return ContradictionTensorRuntime.from_config(config)
        path = Path(str(config.get("head_path", ""))).expanduser()
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError(
                f"learned contradiction tensor requested but head is unreadable: {path}"
            ) from exc
        cache_key = (
            str(path),
            str(config.get("head_sha256", "")),
            stat.st_mtime_ns,
            stat.st_size,
        )
        cached = getattr(self, "_contradiction_head_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        runtime = ContradictionTensorRuntime.from_config(config)
        self._contradiction_head_cache = (cache_key, runtime)
        return runtime

    # ── Fast-weight helpers ─────────────────────────────────────────────
    #: A branch scoring at or below this from the blind task verifier is
    #: REFUTED, not merely weak. Above it the verifier is expressing a
    #: preference, and a preference is not grounds for an irreversible
    #: exclusion — the exclusion has to be the verifier saying no, or the
    #: search removes answers on the strength of an opinion.
    REFUTATION_SCORE_CEILING = 0.0

    def _build_episode_ratchet(
        self,
        *,
        objective: str,
        branch_texts: Mapping[int, str],
        blind_scores: Mapping[int, float],
    ) -> Any:
        """Commit what this episode has established, strongest evidence first.

        Three sources, in order: refutations (a verifier said no), the
        prompt's own stated requirements (free and exact, and routinely
        dropped several passes deep), then unanimous agreement across
        independently sampled branches. Agreement is committed LAST and only
        on unanimity, because a majority-vote constraint is a consensus
        mechanism and consensus is what collapsed into a local basin here
        before.

        The candidate pool is the branch texts, so every narrowing on the
        receipt is MEASURED — the fraction of live candidates a constraint
        actually eliminated — rather than asserted.
        """
        from core.brain.llm.latent_cortex.commitment_extraction import (
            propose_constraints,
        )
        from core.brain.llm.latent_cortex.commitment_ratchet import CommitmentRatchet

        texts = [str(text) for text in branch_texts.values() if str(text).strip()]
        refuted = [
            str(branch_texts.get(index) or "")
            for index, score in blind_scores.items()
            if float(score) <= self.REFUTATION_SCORE_CEILING
            and str(branch_texts.get(index) or "").strip()
        ]
        ratchet = CommitmentRatchet(texts)
        for constraint in propose_constraints(
            objective=objective,
            candidates=texts,
            refuted=refuted,
        ):
            ratchet.commit(constraint)
        ratchet.seal()
        return ratchet

    def _run_generated_canaries(
        self, canaries: CapabilityCanaries, budget: ComputeBudget
    ) -> dict[str, Any]:
        """Decode the postcondition battery when the budget can afford it.

        Refuses rather than overruns: an episode that cannot pay for
        generation gets a receipt saying the behavioral check did NOT run,
        which is the honest reading of a likelihood-only pass.
        """
        if not self.config.fast_weights.canary_generated_enabled:
            return {
                "evaluated": False,
                "reason": "generated battery disabled by configuration",
                "items": [],
                "failed": [],
            }
        cost = canaries.tokens_per_generated_measurement
        if cost <= 0:
            return {
                "evaluated": False,
                "reason": "no generated battery available for this model",
                "items": [],
                "failed": [],
            }
        if not budget.can_afford(cost, self.n_layers):
            return {
                "evaluated": False,
                "reason": (
                    "compute budget could not afford the generated battery "
                    f"({cost} tokens); only likelihood was measured"
                ),
                "items": [],
                "failed": [],
            }
        try:
            return canaries.measure_generated(
                lambda probe_tokens: self._canary_logits(probe_tokens, budget)
            )
        except _LATENT_PHASE_ERRORS as exc:
            return {
                "evaluated": False,
                "reason": f"generated battery failed: {type(exc).__name__}: {exc}",
                "items": [],
                "failed": [],
            }

    def _enforce_fast_weight_canaries(
        self,
        canaries: CapabilityCanaries,
        baseline: dict[str, float],
        fast_weights: EpisodicFastWeights,
        receipt: EpisodeReceipt,
        budget: ComputeBudget,
        generated_baseline: dict[str, Any] | None = None,
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
        magnitude_history: list[dict[str, Any]] = []
        behavioral_evaluated = False
        # Bounded by construction: one measurement per rescale attempt plus
        # the initial one; the final regressed pass erases ΔW and exits.
        for attempt in range(max_rescales + 1):
            try:
                magnitude = fast_weights.effective_delta_metrics()
            except _LATENT_PHASE_ERRORS as exc:
                magnitude = {
                    "schema": "aura.fast_weight_delta_magnitude.v1",
                    "finite": False,
                    "layer_count": len(fast_weights.handles),
                    "max_effective_delta_rms": None,
                    "layers": [],
                    "error_class": type(exc).__name__,
                }
            max_delta_rms = magnitude.get("max_effective_delta_rms")
            structural_regression = bool(
                magnitude.get("finite") is not True
                or isinstance(max_delta_rms, bool)
                or not isinstance(max_delta_rms, (int, float))
                or float(max_delta_rms) > float(cfg.canary_max_effective_delta_rms)
            )
            magnitude_history.append(
                {
                    "attempt": attempt + 1,
                    **magnitude,
                    "threshold_effective_delta_rms": round(
                        float(cfg.canary_max_effective_delta_rms), 12
                    ),
                    "structural_regression": structural_regression,
                }
            )
            if structural_regression:
                comparison = {
                    "items": [],
                    "regressed": [
                        "effective_delta_rms"
                        if magnitude.get("finite") is True
                        else "effective_delta_measurement"
                    ],
                    "max_drop": 0.0,
                    "threshold_logprob_drop": float(cfg.canary_max_logprob_drop),
                }
            else:
                adapted = canaries.measure(
                    lambda probe_tokens: self._canary_logits(probe_tokens, budget)
                )
                behavioral_evaluated = True
                # CP126 68633adf/dfd4858b/a27de3ad: teacher-forced likelihood
                # of a memorized continuation is a fingerprint, not a
                # postcondition. The generated battery decodes under the
                # adapted function and CHECKS what came out — one word when
                # one was demanded, a tool call that parses and names a real
                # tool, an identity that survives a prompt asserting a
                # different one. It costs an order of magnitude more, so it
                # runs when the budget can afford it and is reported as
                # not-run when it cannot.
                generated = self._run_generated_canaries(canaries, budget)
                comparison = compare_canaries(
                    baseline,
                    adapted,
                    max_logprob_drop=cfg.canary_max_logprob_drop,
                    generated=generated,
                    generated_behaviors=(
                        canaries.behaviors_with_generated_evidence
                        if generated.get("evaluated")
                        else frozenset()
                    ),
                )
                # A failed postcondition is a regression. Without this the
                # ladder would rescale on a likelihood drop but ignore a ΔW
                # that emits unparseable tool calls.
                newly_failed = _postconditions_lost(generated_baseline, generated)
                generated["regressions"] = newly_failed
                generated["already_failing_on_base"] = sorted(
                    set(generated.get("failed") or ()) - set(newly_failed)
                )
                if newly_failed:
                    comparison["regressed"] = list(comparison["regressed"]) + [
                        f"postcondition:{name}" for name in newly_failed
                    ]
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
            "behavioral_evaluated": behavioral_evaluated,
            # FINGERPRINT_ONLY is not a pass. It says the postcondition
            # battery did not run, so nothing here is evidence that the
            # adapted function still behaves — only that it still assigns
            # similar probability to some remembered strings.
            "verdict": canary_verdict(comparison) if comparison else {},
            "decision": decision,
            "rescales": rescales,
            "threshold_effective_delta_rms": round(float(cfg.canary_max_effective_delta_rms), 12),
            "delta_magnitude_history": magnitude_history,
            **comparison,
        }
        return decision

    def _enforce_fast_weight_verifier(
        self,
        verifier: Callable[[str], float],
        fast_weights: EpisodicFastWeights,
        winner: BranchState,
        cache,
        runner: WindowRunner,
        budget: ComputeBudget,
        bridge_tokens: list[int] | None,
        receipt: EpisodeReceipt,
        *,
        pre_score: float | None,
        pre_tokens: list[int],
        pre_text: str,
        learning_state: dict[str, Any],
        safety_reserve: int,
        treatment_trace: Mapping[str, Any],
        treatment_target_tokens: list[int],
        sham_trace: Mapping[str, Any],
        sham_target_tokens: list[int],
        sham_probe_tokens: list[int],
        sham_score: float | None,
    ) -> str:
        """Give the task verifier the last word over the adapted function.

        Decode one probe under active ΔW and compare its verified score
        against the verified score of the SAME latent state before
        adaptation. Only a changed token sequence with a strict verified
        improvement survives to final decode. Equality, missing evidence,
        budget pressure, or regression erases ΔW before the answer path.
        """
        lifecycle = fast_weights.lifecycle
        if lifecycle.canary_erased or not fast_weights.handles:
            receipt.fast_weight_verifier = {
                "evaluated": False,
                "decision": "already_erased",
            }
            learning_state["disposition"] = "rejected_capability_regression"
            return "already_erased"
        if lifecycle.optimized_steps <= 0:
            fast_weights.canary_erase()
            receipt.fast_weight_verifier = {
                "evaluated": False,
                "decision": "erased_no_accepted_step",
            }
            learning_state["disposition"] = "rejected_no_accepted_step"
            return "erased_no_accepted_step"
        if pre_score is None or not math.isfinite(pre_score):
            fast_weights.canary_erase()
            receipt.fast_weight_verifier = {
                "evaluated": False,
                "decision": "erased_no_reference",
            }
            receipt.flag("fast_weight_verifier_no_reference")
            learning_state["disposition"] = "rejected_verifier_unavailable"
            return "erased_no_reference"
        if (
            sham_score is None
            or not math.isfinite(sham_score)
            or not sham_probe_tokens
            or not treatment_trace
            or not sham_trace
        ):
            fast_weights.canary_erase()
            receipt.fast_weight_verifier = {
                "evaluated": False,
                "decision": "erased_matched_control_unavailable",
            }
            receipt.flag("fast_weight_matched_control_unavailable")
            learning_state["disposition"] = "rejected_verifier_unavailable"
            return "erased_matched_control_unavailable"
        probe_cost = self._verifier_probe_layer_apps(bridge_tokens)
        if probe_cost + safety_reserve > budget.remaining_layer_apps:
            fast_weights.canary_erase()
            receipt.fast_weight_verifier = {
                "evaluated": False,
                "decision": "erased_budget_unavailable",
                "pre_score": round(float(pre_score), 6),
            }
            receipt.flag("fast_weight_verifier_skipped_budget")
            learning_state["disposition"] = "rejected_verifier_unavailable"
            return "erased_budget_unavailable"
        winner_state_before_sha256 = tensor_sha256(winner.z)
        probe = self._decode_probe(
            winner,
            cache,
            runner,
            budget,
            bridge_tokens=bridge_tokens,
            use_cache=False,
            force_exact_tokens=True,
        )
        post_text = self.tokenizer.decode(probe)
        post_score = float(verifier(post_text))
        winner_state_after_sha256 = tensor_sha256(winner.z)
        pre_tokens_sha256 = token_sequence_sha256(pre_tokens)
        post_tokens_sha256 = token_sequence_sha256(probe)
        token_sequence_changed = pre_tokens_sha256 != post_tokens_sha256
        strict_improvement = bool(
            math.isfinite(post_score)
            and post_score > float(pre_score) + 1e-6
            and token_sequence_changed
            and winner_state_before_sha256
            == winner_state_after_sha256
            == learning_state["winner_state_sha256"]
        )
        learning_state["causal_probe"] = {
            "evaluated": True,
            "pre_tokens_sha256": pre_tokens_sha256,
            "post_tokens_sha256": post_tokens_sha256,
            "pre_text_sha256": hashlib.sha256(pre_text.encode("utf-8")).hexdigest(),
            "post_text_sha256": hashlib.sha256(post_text.encode("utf-8")).hexdigest(),
            "pre_score": float(pre_score),
            "post_score": (float(post_score) if math.isfinite(post_score) else None),
            "token_sequence_changed": token_sequence_changed,
            "strict_improvement": strict_improvement,
            "winner_state_before_sha256": winner_state_before_sha256,
            "winner_state_after_sha256": winner_state_after_sha256,
        }
        if not math.isfinite(post_score):
            fast_weights.canary_erase()
            receipt.fast_weight_verifier = {
                "evaluated": True,
                "decision": "erased_nonfinite_score",
                "pre_score": round(float(pre_score), 6),
            }
            receipt.flag("fast_weight_verifier_erased")
            learning_state["disposition"] = "rejected_non_improvement"
            return "erased_nonfinite_score"
        forward_layer_apps = self.config.workspace.n_slots * (self.coda_start - self.prelude_end)
        probe_layer_apps = self._verifier_probe_layer_apps(bridge_tokens)

        def arm_receipt(
            *,
            arm: str,
            trace: Mapping[str, Any],
            target_tokens: list[int],
            probe_tokens: list[int],
            score: float,
        ) -> dict[str, Any]:
            gradients = int(trace["gradient_evaluations"])
            line_searches = int(trace["line_search_evaluations"])
            return {
                "arm": arm,
                "target_tokens_sha256": token_sequence_sha256(target_tokens),
                "optimizer": str(trace["optimizer"]),
                "attempts": int(trace["attempts"]),
                "forward_evaluations": gradients + line_searches,
                "backward_evaluations": gradients,
                "line_search_evaluations": line_searches,
                "layer_apps": ((3 * gradients + line_searches) * forward_layer_apps),
                "probe_layer_apps": probe_layer_apps,
                "probe_tokens_sha256": token_sequence_sha256(probe_tokens),
                "probe_token_count": len(probe_tokens),
                "score": float(score),
            }

        critic_before = learning_state["admission"]["critic_recalibration"]
        critic_after = build_critic_recalibration_receipt(
            str(critic_before["verifier_family"])
        )
        matched_compute = build_matched_compute_receipt(
            treatment=arm_receipt(
                arm="treatment",
                trace=treatment_trace,
                target_tokens=treatment_target_tokens,
                probe_tokens=probe,
                score=post_score,
            ),
            sham=arm_receipt(
                arm="sham",
                trace=sham_trace,
                target_tokens=sham_target_tokens,
                probe_tokens=sham_probe_tokens,
                score=float(sham_score),
            ),
            baseline_tokens_sha256=pre_tokens_sha256,
            baseline_score=float(pre_score),
            critic_before=critic_before,
            critic_after=critic_after,
        )
        learning_state["controls"]["test_time_training"] = build_test_time_training_receipt(
            critic_recalibration=critic_before,
            pseudo_label_admission=learning_state["admission"]["pseudo_label_admission"],
            matched_compute=matched_compute,
        )
        if not strict_improvement:
            fast_weights.canary_erase()
            decision = (
                "erased_no_causal_effect"
                if not token_sequence_changed
                else "erased_non_improvement"
            )
            receipt.flag("fast_weight_verifier_erased")
            learning_state["disposition"] = (
                "rejected_no_causal_effect"
                if not token_sequence_changed
                else "rejected_non_improvement"
            )
            logger.info(
                "Fast-weight verifier erased ΔW: no strict causal gain "
                "(score %.4f → %.4f, tokens_changed=%s)",
                float(pre_score),
                post_score,
                token_sequence_changed,
            )
        elif matched_compute["accepted"] is not True:
            fast_weights.canary_erase()
            decision = "erased_matched_control"
            receipt.flag("fast_weight_matched_control_rejected")
            learning_state["disposition"] = "rejected_matched_control"
            logger.info(
                "Fast-weight verifier erased treatment after matched control: %s",
                matched_compute["reason"],
            )
        else:
            decision = "accepted_causal_improvement"
            learning_state["disposition"] = "accepted_causal_improvement"
        receipt.fast_weight_verifier = {
            "evaluated": True,
            "decision": decision,
            "pre_score": round(float(pre_score), 6),
            "post_score": round(post_score, 6),
            "pre_tokens_sha256": pre_tokens_sha256,
            "post_tokens_sha256": post_tokens_sha256,
            "token_sequence_changed": token_sequence_changed,
            "winner_state_unchanged": (winner_state_before_sha256 == winner_state_after_sha256),
            "matched_compute_sha256": matched_compute["receipt_sha256"],
            "incremental_gain_over_sham": matched_compute["incremental_gain_over_sham"],
        }
        return decision

    def _search_fast_weight_gains(
        self,
        *,
        verifier: Callable[[str], float],
        fast_weights: EpisodicFastWeights,
        winner: BranchState,
        cache,
        runner: WindowRunner,
        budget: ComputeBudget,
        bridge_tokens: list[int] | None,
        safety_reserve: int,
        baseline_score: float,
        treatment_initial: Sequence[Mapping[str, Any]],
        treatment_candidate: Sequence[Mapping[str, Any]],
        sham_initial: Sequence[Mapping[str, Any]],
        sham_candidate: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[dict[str, Any], ...], list[int], float, dict[str, Any]]:
        """Select signed episodic strength with the actual public verifier."""

        probe_cost = self._verifier_probe_layer_apps(bridge_tokens)
        required = 2 * len(VERIFIER_GAIN_GRID) * probe_cost
        if required + safety_reserve > budget.remaining_layer_apps:
            raise RuntimeError("compute budget cannot admit matched verifier gain search")

        snapshots = {
            "treatment": (treatment_initial, treatment_candidate),
            "sham": (sham_initial, sham_candidate),
        }
        rows: dict[str, list[dict[str, Any]]] = {"treatment": [], "sham": []}
        private: dict[str, list[tuple[float, tuple[dict[str, Any], ...], list[int], float]]] = {
            "treatment": [],
            "sham": [],
        }
        for arm in ("treatment", "sham"):
            initial, candidate = snapshots[arm]
            for index, gain in enumerate(VERIFIER_GAIN_GRID):
                fast_weights.interpolate_delta(
                    initial,
                    candidate,
                    gain=gain,
                    reason=f"fast_weights_{arm}_gain_{index}",
                )
                probe = self._decode_probe(
                    winner,
                    cache,
                    runner,
                    budget,
                    bridge_tokens=bridge_tokens,
                    use_cache=False,
                    force_exact_tokens=True,
                )
                score = float(verifier(self.tokenizer.decode(probe)))
                snapshot = fast_weights.snapshot_delta()
                row = {
                    "arm": arm,
                    "index": index,
                    "gain": float(gain),
                    "probe_tokens_sha256": token_sequence_sha256(probe),
                    "probe_token_count": len(probe),
                    "score": score,
                    "layer_apps": probe_cost,
                }
                rows[arm].append(row)
                private[arm].append((float(gain), snapshot, probe, score))

        gain_receipt = build_verifier_gain_search_receipt(
            treatment_rows=rows["treatment"],
            sham_rows=rows["sham"],
            baseline_score=baseline_score,
        )
        treatment_gain = gain_receipt["selected_treatment_gain"]
        sham_gain = gain_receipt["selected_sham_gain"]
        selected_treatment = next(row for row in private["treatment"] if row[0] == treatment_gain)
        selected_sham = next(row for row in private["sham"] if row[0] == sham_gain)
        fast_weights.restore_delta(
            selected_treatment[1],
            reason="fast_weights_verifier_selected_treatment_gain",
        )
        return (
            selected_treatment[1],
            selected_sham[2],
            selected_sham[3],
            gain_receipt,
        )

    def _canary_logits(self, probe_tokens: list[int], budget: ComputeBudget):
        """Standard causal full-stack forward over one canary sequence."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        if not budget.can_afford(len(probe_tokens), self.n_layers):
            raise RuntimeError("compute budget cannot afford capability canary pass")
        budget.charge(
            tokens=len(probe_tokens),
            layers=self.n_layers,
            operation="capability_canary",
            attention_pairs=(triangular_attention_pairs(len(probe_tokens)) * self.n_layers),
            output_head_tokens=len(probe_tokens),
        )
        inner = self.model.model
        cache = self._fresh_cache()
        h = inner.embed_tokens(mx.array([probe_tokens]))
        mask = create_attention_mask(h, cache)
        for index, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[index])
        logits = self._logits(h)
        mx.eval(logits)
        return logits

    def _nocache_window_pass(self, z, *, branch_index: int):
        """Window pass with no cache (grad-safe: zero side effects)."""
        from core.brain.llm.latent_cortex.recurrence_adapter import (
            recurrence_adapter_scope,
        )
        from core.learning.role_conditioned_lora import recurrent_branch_index

        inner = self.model.model
        h = z
        with recurrent_branch_index(branch_index), recurrence_adapter_scope():
            for i in range(self.prelude_end, self.coda_start):
                h = inner.layers[i](h, None, None)
        return h

    def _capture_teacher_forced_trajectory(
        self,
        fast_weights: EpisodicFastWeights,
        budget: ComputeBudget,
        *,
        objective: str,
        answer_tokens: Sequence[int],
        operation: str,
    ) -> tuple[dict[int, Any], dict[int, Any], list[int]]:
        """Capture aligned private layer inputs/outputs for one answer."""

        import mlx.core as mx

        if self.tokenizer is None:
            raise RuntimeError("teacher trajectory capture requires a tokenizer")
        answer = [int(token) for token in answer_tokens]
        if not answer:
            raise ValueError("teacher trajectory answer is empty")
        answer = answer[:256]
        prefix = self.tokenizer.encode(
            f"{objective}\nCandidate answer:\n",
            add_special_tokens=False,
        )
        prefix = [int(token) for token in prefix]
        prefix_budget = _MAX_TEACHER_TRAJECTORY_TOKENS - len(answer)
        if prefix_budget <= 0:
            raise RuntimeError("teacher trajectory has no objective context budget")
        if len(prefix) > prefix_budget:
            head = prefix_budget // 2
            prefix = prefix[:head] + prefix[-(prefix_budget - head) :]
        context_tokens = prefix + answer
        layer_apps = len(context_tokens) * self.n_layers
        if not budget.can_afford(len(context_tokens), self.n_layers):
            raise RuntimeError("compute budget cannot admit teacher trajectory capture")
        budget.charge(
            tokens=len(context_tokens),
            layers=self.n_layers,
            operation=operation,
            attention_pairs=(
                triangular_attention_pairs(len(context_tokens)) * self.n_layers
            ),
        )

        def forward():
            inner = self.model.model
            h = inner.embed_tokens(mx.array([context_tokens]))
            for layer in inner.layers:
                h = layer(h, None, None)
            mx.eval(h)
            return h

        inputs, outputs = fast_weights.capture_io_features(
            forward,
            token_start=len(prefix),
        )
        if layer_apps <= 0:
            raise RuntimeError("teacher trajectory accounting is invalid")
        return inputs, outputs, context_tokens

    def _capture_forced_output_keys(
        self,
        branch: BranchState,
        cache,
        runner: WindowRunner,
        budget: ComputeBudget,
        *,
        bridge_tokens: Sequence[int],
        target_tokens: Sequence[int],
        operation: str,
    ):
        """Capture the exact normalized output states preceding target tokens."""

        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        from core.brain.llm.recurrent_depth import (
            _restore_recurrent_caches,
            _snapshot_recurrent_caches,
        )

        targets = [int(token) for token in target_tokens]
        if not targets:
            raise ValueError("output-memory target token sequence is empty")
        snaps = _snapshot_recurrent_caches(cache, 0, self.n_layers)
        prior_memory = getattr(self, "_active_output_memory", None)
        prior_hidden = getattr(self, "_last_output_hidden", None)
        try:
            self._active_output_memory = None
            self._persist_branch(branch, cache, runner, budget)
            if bridge_tokens:
                self._apply_decode_bridge(cache, budget, list(bridge_tokens))
            keys = [mx.stop_gradient(self._last_output_hidden[0, -1].astype(mx.float32))]
            inner = self.model.model
            for token in targets[:-1]:
                if not budget.can_afford(1, self.n_layers):
                    raise RuntimeError("compute budget cannot admit output-memory capture")
                context_tokens = self._cache_context_tokens(cache)
                budget.charge(
                    tokens=1,
                    layers=self.n_layers,
                    operation=operation,
                    attention_pairs=max(1, context_tokens + 1) * self.n_layers,
                    output_head_tokens=1,
                )
                h = inner.embed_tokens(mx.array([[token]]))
                mask = create_attention_mask(h, cache)
                from core.brain.llm.latent_cortex.recurrence_adapter import (
                    coda_adapter_scope,
                )

                with coda_adapter_scope():
                    for index, layer in enumerate(inner.layers):
                        h = layer(h, mask, cache[index])
                self._logits(h)
                keys.append(
                    mx.stop_gradient(self._last_output_hidden[0, -1].astype(mx.float32))
                )
            stacked = mx.stack(keys)
            mx.eval(stacked)
            return stacked
        finally:
            self._active_output_memory = prior_memory
            if prior_hidden is None:
                if hasattr(self, "_last_output_hidden"):
                    delattr(self, "_last_output_hidden")
            else:
                self._last_output_hidden = prior_hidden
            _restore_recurrent_caches(cache, 0, self.n_layers, snaps)

    def _evaluate_output_memory_controls(
        self,
        branch: BranchState,
        cache,
        runner: WindowRunner,
        budget: ComputeBudget,
        *,
        bridge_tokens: Sequence[int],
        verifier: Callable[[str], float],
        baseline_score: float,
        treatment_keys,
        treatment_tokens: Sequence[int],
        sham_keys,
        sham_tokens: Sequence[int],
    ) -> dict[str, Any]:
        """Run verified and sham output memories under identical teacher-free probes."""

        treatment = EpisodicOutputMemory(treatment_keys, treatment_tokens)
        sham = EpisodicOutputMemory(sham_keys, sham_tokens)
        treatment_identity = treatment.receipt()
        sham_identity = sham.receipt()

        def run_arm(label: str, memory: EpisodicOutputMemory) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for gain in OUTPUT_MEMORY_GAIN_GRID:
                memory.reset(gain=gain)
                self._active_output_memory = memory
                try:
                    probe = self._decode_probe(
                        branch,
                        cache,
                        runner,
                        budget,
                        max_tokens=len(memory.target_tokens),
                        bridge_tokens=list(bridge_tokens),
                        use_cache=False,
                        force_exact_tokens=True,
                    )
                finally:
                    self._active_output_memory = None
                text = self.tokenizer.decode(probe)
                rows.append(
                    {
                        "arm": label,
                        "gain": gain,
                        "score": float(verifier(text)),
                        "probe_tokens_sha256": token_sequence_sha256(probe),
                        "probe_token_count": len(probe),
                        "matches": memory.matches,
                        "misses": memory.misses,
                        "minimum_similarity": memory.minimum_similarity,
                    }
                )
            return rows

        try:
            treatment_rows = run_arm("treatment", treatment)
            sham_rows = run_arm("sham", sham)
        finally:
            self._active_output_memory = None
            treatment.erase()
            sham.erase()
        return build_output_memory_experiment_receipt(
            baseline_score=baseline_score,
            treatment_identity=treatment_identity,
            sham_identity=sham_identity,
            treatment_rows=treatment_rows,
            sham_rows=sham_rows,
            erase_proven=(
                treatment.erased
                and sham.erased
                and getattr(self, "_active_output_memory", None) is None
            ),
        )

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
            budget.charge_cleanup_overdraft(
                tokens=8,
                layers=self.n_layers,
                operation="fast_weight_erase_cleanup_probe",
                attention_pairs=8 * 8 * self.n_layers,
                output_head_tokens=8,
            )
        else:
            if not budget.can_afford(8, self.n_layers):
                raise RuntimeError("compute budget cannot afford fast-weight erase probe")
            budget.charge(
                tokens=8,
                layers=self.n_layers,
                operation="fast_weight_integrity_probe",
                attention_pairs=8 * 8 * self.n_layers,
                output_head_tokens=8,
            )
        h = inner.embed_tokens(probe_tokens)
        for layer in inner.layers:
            h = layer(h, None, None)
        out = self._logits(h)
        mx.eval(out)
        return out


__all__ = ["LatentCortexEngine"]
