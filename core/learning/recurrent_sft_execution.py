"""Shared recurrent-SFT model topology and tokenizer projection contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Never

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec


class RecurrentSFTExecutionError(ValueError):
    """Training and evaluation no longer agree on the recurrent execution path."""


def _fail(code: str) -> Never:
    raise RecurrentSFTExecutionError(
        str(code or "recurrent_sft_execution_invalid")
    )


def _tokens(value: Any, *, role: str) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(type(token) is not int or token < 0 for token in value)
    ):
        _fail(f"recurrent_sft_{role}_tokens_invalid")
    return list(value)


def project_chat_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_seq_length: int,
) -> list[dict[str, Any]]:
    """Project validated chat rows into the exact live-path token boundary."""

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        _fail("recurrent_sft_tokenizer_template_missing")
    if type(max_seq_length) is not int or max_seq_length < 1:
        _fail("recurrent_sft_max_sequence_length_invalid")
    try:
        from mlx_lm.tuner.datasets import ChatDataset
    except ImportError as exc:
        raise RecurrentSFTExecutionError(
            "recurrent_sft_chat_dataset_unavailable"
        ) from exc
    source_rows = [dict(row) for row in rows]
    dataset = ChatDataset(
        source_rows,
        tokenizer,
        mask_prompt=True,
    )
    projected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(source_rows):
        messages = row.get("messages")
        tools = row.get("tools")
        metadata = row.get("_meta")
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(messages[-1], Mapping)
            or messages[-1].get("role") != "assistant"
            or not isinstance(metadata, Mapping)
            or not isinstance(metadata.get("example_id"), str)
            or metadata["example_id"] in seen_ids
        ):
            _fail("recurrent_sft_chat_row_invalid")
        seen_ids.add(metadata["example_id"])
        processed = dataset.process(row)
        if (
            not isinstance(processed, tuple)
            or len(processed) != 2
            or type(processed[1]) is not int
        ):
            _fail("recurrent_sft_chat_projection_invalid")
        full = _tokens(
            processed[0],
            role=f"full_{index}",
        )
        prefix = _tokens(
            apply_template(
                messages[:-1],
                tools=tools,
                add_generation_prompt=True,
                return_dict=False,
            ),
            role=f"prefix_{index}",
        )
        if (
            processed[1] != len(prefix)
            or len(prefix) >= len(full)
            or full[: len(prefix)] != prefix
            or len(full) > max_seq_length
        ):
            _fail("recurrent_sft_chat_token_boundary_invalid")
        projected.append(
            {
                "example_id": metadata["example_id"],
                "family": metadata.get("family"),
                "target_kind": metadata.get("target_kind"),
                "prompt_tokens": prefix,
                "answer_tokens": full[len(prefix) :],
                "full_token_count": len(full),
            }
        )
    return projected


def wrap_recurrent_window(
    model: Any,
    *,
    spec: RLCExecutionSpec,
    lora_rank: int,
    lora_dropout: float,
    lora_scale: float,
    lora_targets: Sequence[str],
) -> list[str]:
    """Freeze a model and attach slot-scoped LoRA to its recurrent window."""

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedLoRALinear,
    )

    if (
        type(lora_rank) is not int
        or lora_rank < 1
        or isinstance(lora_dropout, bool)
        or not isinstance(lora_dropout, (int, float))
        or isinstance(lora_scale, bool)
        or not isinstance(lora_scale, (int, float))
        or not lora_targets
    ):
        _fail("recurrent_sft_adapter_config_invalid")
    model.freeze()
    layers = model.model.layers
    prelude_end = max(1, int(len(layers) * spec.prelude_frac))
    coda_start = min(
        len(layers) - 1,
        len(layers) - int(len(layers) * spec.coda_frac),
    )
    wrapped: list[str] = []
    for layer_index in range(prelude_end, coda_start):
        layer = layers[layer_index]
        for target in lora_targets:
            parent_name = (
                "self_attn"
                if hasattr(layer.self_attn, target)
                else "mlp"
                if hasattr(layer.mlp, target)
                else ""
            )
            if not parent_name:
                continue
            parent = getattr(layer, parent_name)
            base = getattr(parent, target)
            site = f"model.layers.{layer_index}.{parent_name}.{target}"
            setattr(
                parent,
                target,
                ScopedLoRALinear.from_base(
                    base,
                    r=lora_rank,
                    dropout=float(lora_dropout),
                    scale=float(lora_scale),
                    # Identity travels with the projection so an activation
                    # receipt can name the sites that fired. Without it, a
                    # projection that was wrapped and never applied anything
                    # is invisible inside the aggregate call count -- the
                    # CP227 failure one level down.
                    block_index=layer_index,
                    site=site,
                ),
            )
            wrapped.append(site)
    if not wrapped:
        _fail("recurrent_sft_no_projections_wrapped")
    return wrapped


def adapter_tensor_dict(model: Any) -> dict[str, Any]:
    """Return a validated flat adapter tree from the wrapped model."""

    from mlx.utils import tree_flatten

    tensors = dict(tree_flatten(model.trainable_parameters()))
    assert_adapter_tensor_topology(tensors, tensors)
    return tensors


def adapter_tensor_fingerprint(tensors: Mapping[str, Any]) -> str:
    """Hash exact adapter names, dtypes, shapes, and tensor bytes."""

    import numpy as np

    assert_adapter_tensor_topology(tensors, tensors)
    digest = hashlib.sha256()
    for name in sorted(tensors):
        array = np.asarray(tensors[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(array.shape),
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def assert_adapter_tensor_topology(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> None:
    """Require exactly the slot-scoped LoRA tensor names and no base weights."""

    if (
        not expected
        or set(expected) != set(observed)
        or any(
            not (key.endswith(".lora_a") or key.endswith(".lora_b"))
            for key in observed
        )
    ):
        _fail("recurrent_sft_adapter_tensor_topology_invalid")


__all__ = [
    "RecurrentSFTExecutionError",
    "adapter_tensor_dict",
    "adapter_tensor_fingerprint",
    "assert_adapter_tensor_topology",
    "project_chat_rows",
    "wrap_recurrent_window",
]
