#!/usr/bin/env python3
"""Produce one answer-blind recurrent candidate for full-engine composition."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Any, Final

from core.brain.llm.latent_cortex.answer_contract import (
    ContractDecodeDisposition,
    contract_decode_disposition,
)
from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
)
from tools.rlc_complete_system_closed_book import build_integrated_candidate

PRODUCER_SCHEMA: Final = "aura.rlc.integrated_recurrent_producer.v1"
RESOURCE_ESTIMATOR: Final = "unified_general_recurrent_structural_v1"
SOURCE_NAME: Final = "unified_recurrent_controller"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_general_recurrent_resource_receipt(
    *,
    profile: ModelComputeProfile,
    prompt_tokens: int,
    generated_tokens: int,
    prelude_end: int,
    coda_start: int,
    recurrence_depth: int,
    correction_rank: int,
    depth_basis_size: int,
    renormalize: bool,
) -> dict[str, Any]:
    """Account the fixed-depth incremental recurrence operation graph.

    The estimator counts the actual incremental query lengths used by the
    cache topology. Controller tensor work is kept separate from transformer
    FLOPs, so recurrent control never disappears into a nominal token count.
    """

    if not isinstance(profile, ModelComputeProfile):
        raise TypeError("recurrent resource profile is invalid")
    prompt = _positive_integer(prompt_tokens, name="prompt_tokens")
    generated = _positive_integer(generated_tokens, name="generated_tokens")
    depth = _positive_integer(recurrence_depth, name="recurrence_depth")
    rank = _positive_integer(correction_rank, name="correction_rank")
    basis = _positive_integer(depth_basis_size, name="depth_basis_size")
    if (
        type(prelude_end) is not int
        or type(coda_start) is not int
        or not 0 <= prelude_end < coda_start <= profile.num_hidden_layers
        or rank > profile.hidden_size
        or type(renormalize) is not bool
    ):
        raise ValueError("recurrent resource topology is invalid")

    window = coda_start - prelude_end
    effective_layers = (
        prelude_end
        + depth * window
        + (profile.num_hidden_layers - coda_start)
    )
    query_tokens = prompt + generated - 1
    # The first call evaluates the complete prompt. Each later cached call has
    # one query attending to the prompt plus all prior generated tokens.
    attention_pairs_per_layer = prompt * prompt + sum(
        prompt + offset for offset in range(1, generated)
    )
    ledger = ResourceLedger(profile)
    ledger.charge(
        "incremental_recurrent_transformer",
        transformer_layer_apps=query_tokens * effective_layers,
        attention_query_key_pairs=attention_pairs_per_layer * effective_layers,
        output_head_tokens=query_tokens,
    )

    hidden = profile.hidden_size
    repeated_steps = depth - 1
    # Structural tensor-op estimate for the explicit correction, transport,
    # normalization and halt operators. It is deterministic from the executed
    # token shapes and deliberately excludes transformer work counted above.
    correction_per_token = (
        4 * hidden * rank
        + rank
        + hidden
        + 2 * basis * rank
        + rank
    )
    transport_per_token = 18 * hidden + 2 * basis + 12
    halt_per_token = 5 * hidden + 12
    anchor_normalization_per_token = 2 * hidden if renormalize else 0
    reentry_normalization_per_token = 3 * hidden if renormalize else 0
    controller_ops = query_tokens * (
        repeated_steps * (correction_per_token + transport_per_token)
        + depth * halt_per_token
        + anchor_normalization_per_token
        + repeated_steps * reentry_normalization_per_token
    )
    ledger.charge(
        f"{RESOURCE_ESTIMATOR}:controller",
        tensor_element_reads=controller_ops,
        tensor_element_writes=controller_ops,
        tensor_scalar_ops=controller_ops,
        host_scalar_ops=generated * (depth * 8 + repeated_steps * basis),
    )
    return ledger.to_receipt()


def _completion_check(tokenizer: Any) -> Callable[[Sequence[int]], bool]:
    def complete(token_ids: Sequence[int]) -> bool:
        text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
        return contract_decode_disposition(text) in {
            ContractDecodeDisposition.COMPLETE,
            ContractDecodeDisposition.INVALID,
        }

    return complete


def produce_integrated_recurrent_candidate(
    *,
    model: Any,
    tokenizer: Any,
    task: Any,
    loaded: Any,
    public_tokens: Sequence[int],
    max_tokens: int,
    recurrence_depth: int | None = None,
    activity: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode and seal one qualified recurrent candidate without scoring it."""

    tokens = tuple(int(value) for value in public_tokens)
    if not tokens or any(value < 0 for value in tokens):
        raise ValueError("integrated recurrent prompt tokens are invalid")
    depth = (
        int(loaded.receipt["recurrence_depth"])
        if recurrence_depth is None
        else _positive_integer(recurrence_depth, name="recurrence_depth")
    )
    plan = loaded.spec.plan_at(depth)
    started = time.perf_counter()
    generated, stopped, latency_ms = loaded.decode_general_recurrent_tokens(
        model,
        tokens,
        max_tokens=_positive_integer(max_tokens, name="max_tokens"),
        recurrence_depth=depth,
        completion_check=_completion_check(tokenizer),
        activity=activity,
    )
    elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000.0)))
    if not generated or stopped is not True:
        raise RuntimeError("integrated recurrent candidate did not terminate")
    text = tokenizer.decode(list(generated), skip_special_tokens=True).strip()
    if contract_decode_disposition(text) is not ContractDecodeDisposition.COMPLETE:
        raise RuntimeError("integrated recurrent candidate contract is invalid")
    if abs(elapsed_ms - int(latency_ms)) > max(250, elapsed_ms // 10):
        raise RuntimeError("integrated recurrent latency receipt differs")

    profile = ModelComputeProfile.from_model(model)
    resource = build_general_recurrent_resource_receipt(
        profile=profile,
        prompt_tokens=len(tokens),
        generated_tokens=len(generated),
        prelude_end=int(plan.prelude_end),
        coda_start=int(plan.coda_start),
        recurrence_depth=depth,
        correction_rank=int(loaded.controller.config.correction_rank),
        depth_basis_size=int(loaded.controller.config.depth_basis_size),
        renormalize=bool(plan.renormalize),
    )
    package_receipt = dict(loaded.receipt)
    producer_body = {
        "schema": PRODUCER_SCHEMA,
        "source": SOURCE_NAME,
        "task_id": str(task.task_id),
        "prompt_sha256": hashlib.sha256(str(task.public.prompt).encode()).hexdigest(),
        "public_token_sha256": _canonical_sha256(list(tokens)),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "generated_token_sha256": _canonical_sha256(list(generated)),
        "generated_tokens": len(generated),
        "recurrence_depth": depth,
        "package_id": str(package_receipt.get("package_id") or ""),
        "manifest_sha256": str(package_receipt.get("manifest_sha256") or ""),
        "controller_sha256": str(package_receipt.get("controller_sha256") or ""),
        "resource_accounting_sha256": resource["receipt_sha256"],
        "resource_estimator": RESOURCE_ESTIMATOR,
        "same_public_information": True,
        "answer_key_used": False,
        "score_observed": False,
        "serving_authority": False,
    }
    for name in ("package_id", "manifest_sha256", "controller_sha256"):
        value = producer_body[name]
        if not value or (name != "package_id" and not _sha256(value)):
            raise RuntimeError("integrated recurrent package identity is invalid")
    producer = {
        **producer_body,
        "receipt_sha256": _canonical_sha256(producer_body),
    }
    candidate = build_integrated_candidate(
        source=SOURCE_NAME,
        task_id=str(task.task_id),
        text=text,
        resource_accounting=resource,
        source_receipt=producer,
        source_receipt_sha256=producer["receipt_sha256"],
    )
    return candidate, producer


__all__ = [
    "PRODUCER_SCHEMA",
    "RESOURCE_ESTIMATOR",
    "SOURCE_NAME",
    "build_general_recurrent_resource_receipt",
    "produce_integrated_recurrent_candidate",
]
