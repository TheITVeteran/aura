"""Deterministic measured-integrity fixtures shared by RLC contract tests."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from core.brain.llm.latent_cortex.runtime_integrity import (
    ADAPTED_LAYER_SCHEMA,
    PARAMETER_CANARY_SCHEMA,
    STACK_MEASUREMENT_SCHEMA,
    bind_worker_runtime_integrity,
    build_engine_runtime_integrity,
    build_fast_weight_cleanup_proof,
    canonical_sha256,
)


def complete_serving_stack() -> dict[str, Any]:
    adapters: list[dict[str, Any]] = []
    return {
        "worker_adapters": adapters,
        "worker_adapter_stack_sha256": canonical_sha256(adapters),
        "worker_tokenizer": {"tokenizer.json": "a" * 64},
        "worker_runtime_tokenizer": {
            "type": "tests.DeterministicTokenizer",
            "vocab_size": 128,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "unk_token_id": 3,
            "special_tokens_sha256": "d" * 64,
            "chat_template_sha256": "b" * 64,
        },
        "worker_quantization": {
            "bits": 4,
            "group_size": 64,
            "dtype": "float16",
            "model_type": "qwen2",
            "config_sha256": "c" * 64,
        },
        "worker_stack_identity_gaps": [],
    }


def complete_worker_identity(
    *,
    boot_id: str = "1" * 32,
    pid: int = 4242,
    model_path: str = "/models/test-32b",
) -> dict[str, Any]:
    return {
        "schema": "aura.latent_cortex.worker_identity.v1",
        "worker_boot_id": boot_id,
        "worker_pid": pid,
        "worker_model_path": model_path,
        "worker_model_parameter_count": 32_000_000_000,
        "worker_model_stored_parameter_element_count": 5_000_000_000,
        "worker_model_parameter_count_basis": "architecture_config_logical",
        "worker_source_sha256": "2" * 64,
        "worker_affective_steering_active": True,
        "worker_affective_steering_alpha": 0.30,
        **complete_serving_stack(),
    }


def _parameter_measurement(digest: str = "d" * 64) -> dict[str, Any]:
    return {
        "schema": PARAMETER_CANARY_SCHEMA,
        "method": "fixed_stride_tensor_canary_sha256_v1",
        "stride": 7,
        "elements_per_tensor": 64,
        "parameter_leaf_count": 128,
        "sampled_tensor_count": 19,
        "sampled_element_count": 1216,
        "sha256": digest,
    }


def _adapted_measurement(digest: str = "e" * 64) -> dict[str, Any]:
    return {
        "schema": ADAPTED_LAYER_SCHEMA,
        "method": "exact_target_parameter_bytes_sha256_v1",
        "target": "o_proj",
        "layer_ids": ["layers.1.o_proj", "layers.2.o_proj"],
        "tensor_count": 4,
        "element_count": 4096,
        "sha256": digest,
    }


def _stack_measurement(
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stack = dict(identity or complete_serving_stack())
    return {
        "schema": STACK_MEASUREMENT_SCHEMA,
        "identity": stack,
        "identity_sha256": canonical_sha256(stack),
    }


def engine_runtime_integrity(
    *,
    episode_id: str,
    input_tokens_sha256: str,
    fast_weights_applied: bool = False,
    fast_weight_learning: Mapping[str, Any] | None = None,
    fast_weight_cleanup: Mapping[str, Any] | None = None,
    probe_cache: Mapping[str, Any] | None = None,
    checkpoint_required: bool = True,
    checkpoint_fingerprint: str = "f" * 64,
    checkpoint_method: str = "sha256",
    checkpoint_file_count: int = 8,
) -> dict[str, Any]:
    parameters = _parameter_measurement()
    adapted = _adapted_measurement()
    if fast_weights_applied and fast_weight_cleanup is None:
        cleanup = (
            fast_weight_learning.get("cleanup")
            if isinstance(fast_weight_learning, Mapping)
            else None
        )
        cleanup = cleanup if isinstance(cleanup, Mapping) else {}
        fast_weight_cleanup = build_fast_weight_cleanup_proof(
            episode_id=episode_id,
            input_tokens_sha256=input_tokens_sha256,
            detached=cleanup.get("detached", True) is True,
            erase_proven=cleanup.get("erase_proven", True) is True,
            lease_released=cleanup.get("lease_released", True) is True,
            conflicts=int(cleanup.get("conflicts", 0)),
            pre_probe_sha256=str(
                cleanup.get("pre_probe_sha256") or "7" * 64
            ),
            post_probe_sha256=str(
                cleanup.get("post_probe_sha256") or "7" * 64
            ),
            layer_ids=list(
                cleanup.get("erased_layer_ids")
                or ["layers.1.o_proj", "layers.2.o_proj"]
            ),
        )
    return build_engine_runtime_integrity(
        episode_id=episode_id,
        input_tokens_sha256=input_tokens_sha256,
        checkpoint={
            "required": checkpoint_required,
            "fingerprint": (
                checkpoint_fingerprint if checkpoint_required else ""
            ),
            "method": checkpoint_method if checkpoint_required else "",
            "files": checkpoint_file_count if checkpoint_required else 0,
        },
        parameters_before=parameters,
        parameters_after=parameters,
        adapted_layers_before=adapted,
        adapted_layers_after=adapted,
        serving_stack_before=_stack_measurement(),
        serving_stack_after=_stack_measurement(),
        fast_weights_applied=fast_weights_applied,
        fast_weight_learning=fast_weight_learning,
        fast_weight_cleanup=fast_weight_cleanup,
        probe_cache=probe_cache,
    )


@lru_cache(maxsize=1)
def _accepted_admission_template() -> dict[str, Any]:
    from core.brain.llm.latent_cortex.fast_weight_learning import (
        build_fast_weight_admission,
    )
    from core.brain.llm.latent_cortex.task_verifiers import (
        EpisodeTaskVerifier,
    )

    class _ByteTokenizer:
        @staticmethod
        def encode(
            text: str,
            add_special_tokens: bool = False,
        ) -> list[int]:
            del add_special_tokens
            return list(text.encode())

    candidate = "2 + 2 = 4."
    verifier = EpisodeTaskVerifier("Check the calculation.")
    return build_fast_weight_admission(
        verifier.evaluate(candidate),
        candidate=candidate,
        objective=verifier.objective,
        evaluation_index=0,
        tokenizer=_ByteTokenizer(),
    )[0]


def accepted_fast_weight_learning(
    *,
    episode_id: str,
    input_tokens_sha256: str,
) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.fast_weight_learning import (
        empty_learning_state,
        finalize_fast_weight_learning_receipt,
        token_sequence_sha256,
    )

    admission = copy.deepcopy(_accepted_admission_template())
    winner_sha256 = hashlib.sha256(b"winner").hexdigest()
    state = empty_learning_state(
        episode_id=episode_id,
        input_tokens_sha256=input_tokens_sha256,
        selected_branch=0,
        winner_state_sha256=winner_sha256,
        admission=admission,
    )
    probe_sha256 = hashlib.sha256(b"identity-probe").hexdigest()
    state["lease"] = {
        "schema": "aura.rlc.fast_weight_model_lease.v1",
        "owner_sha256": hashlib.sha256(b"owner").hexdigest(),
        "model_sha256": hashlib.sha256(b"model").hexdigest(),
        "acquired": True,
        "released": True,
        "conflicts": 0,
    }
    state["attach_identity"] = {
        "measured": True,
        "pre_probe_sha256": probe_sha256,
        "post_probe_sha256": probe_sha256,
        "exact": True,
        "winner_state_before_sha256": winner_sha256,
        "winner_state_after_sha256": winner_sha256,
    }
    state["optimization"] = {
        "optimizer": "rms_normalized_sgd_backtracking_v1",
        "attempts": 1,
        "accepted_steps": 1,
        "rejected_steps": 0,
        "budget_exhausted": False,
        "loss_trail": [1.0, 0.5],
        "gradient_norm_trail": [0.25],
        "accepted_step_sizes": [0.01],
        "line_search_backtracks": 0,
    }
    state["controls"] = {
        "decision": "accepted",
        "capability_canaries": {"decision": "accepted"},
    }
    state["causal_probe"] = {
        "evaluated": True,
        "pre_tokens_sha256": token_sequence_sha256([1]),
        "post_tokens_sha256": token_sequence_sha256([2]),
        "pre_text_sha256": admission["source_sha256"],
        "post_text_sha256": hashlib.sha256(b"improved").hexdigest(),
        "pre_score": 0.5,
        "post_score": 0.75,
        "token_sequence_changed": True,
        "strict_improvement": True,
        "winner_state_before_sha256": winner_sha256,
        "winner_state_after_sha256": winner_sha256,
    }
    state["final_answer"] = {
        "decoded_under_adaptation": True,
        "tokens_sha256": token_sequence_sha256([9, 10]),
        "text_sha256": hashlib.sha256(b"answer").hexdigest(),
        "token_count": 2,
    }
    state["cleanup"] = {
        "required": True,
        "detached": True,
        "erase_proven": True,
        "lease_released": True,
        "conflicts": 0,
        "pre_probe_sha256": probe_sha256,
        "post_probe_sha256": probe_sha256,
        "erased_layer_ids": ["layers.1.o_proj", "layers.2.o_proj"],
    }
    state["disposition"] = "accepted_causal_improvement"
    return finalize_fast_weight_learning_receipt(state)


def bound_runtime_integrity(
    *,
    episode_id: str,
    input_tokens_sha256: str,
    worker_identity: Mapping[str, Any] | None = None,
    fast_weights_applied: bool = False,
    fast_weight_learning: Mapping[str, Any] | None = None,
    fast_weight_cleanup: Mapping[str, Any] | None = None,
    probe_cache: Mapping[str, Any] | None = None,
    checkpoint_fingerprint: str = "f" * 64,
    checkpoint_method: str = "sha256",
    checkpoint_file_count: int = 8,
) -> dict[str, Any]:
    worker = dict(worker_identity or complete_worker_identity())
    return bind_worker_runtime_integrity(
        engine_runtime_integrity(
            episode_id=episode_id,
            input_tokens_sha256=input_tokens_sha256,
            fast_weights_applied=fast_weights_applied,
            fast_weight_learning=fast_weight_learning,
            fast_weight_cleanup=fast_weight_cleanup,
            probe_cache=probe_cache,
            checkpoint_fingerprint=checkpoint_fingerprint,
            checkpoint_method=checkpoint_method,
            checkpoint_file_count=checkpoint_file_count,
        ),
        worker_identity=worker,
    )


def attach_bound_runtime_integrity(
    receipt: dict[str, Any],
    *,
    worker_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    worker = dict(worker_identity or receipt.get("worker_identity") or complete_worker_identity())
    episode_id = str(receipt.get("episode_id") or "episode-runtime-integrity")
    input_tokens_sha256 = str(receipt.get("input_tokens_sha256") or "7" * 64)
    receipt["episode_id"] = episode_id
    receipt["input_tokens_sha256"] = input_tokens_sha256
    receipt.setdefault("checkpoint_fingerprint", "f" * 64)
    receipt.setdefault("checkpoint_fingerprint_method", "sha256")
    receipt.setdefault("checkpoint_file_count", 8)
    receipt["worker_identity"] = worker
    receipt["runtime_integrity"] = bound_runtime_integrity(
        episode_id=episode_id,
        input_tokens_sha256=input_tokens_sha256,
        worker_identity=worker,
        fast_weights_applied=receipt.get("fast_weights_applied") is True,
        fast_weight_learning=receipt.get("fast_weight_learning"),
        fast_weight_cleanup=receipt.get("fast_weight_cleanup"),
        probe_cache=receipt.get("probe_cache"),
        checkpoint_fingerprint=str(
            receipt.get("checkpoint_fingerprint") or ""
        ),
        checkpoint_method=str(
            receipt.get("checkpoint_fingerprint_method") or ""
        ),
        checkpoint_file_count=int(
            receipt.get("checkpoint_file_count") or 0
        ),
    )
    return receipt


__all__ = [
    "attach_bound_runtime_integrity",
    "accepted_fast_weight_learning",
    "bound_runtime_integrity",
    "complete_serving_stack",
    "complete_worker_identity",
    "engine_runtime_integrity",
]
