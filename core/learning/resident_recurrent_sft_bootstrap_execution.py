"""Exact dataset projection and sampling for resident recurrent SFT."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    SAMPLER_NAME,
    sha256_json,
)
from core.learning.resident_recurrent_sft_bootstrap_state import ZERO_SHA256

PROJECTED_EXAMPLE_SCHEMA: Final = "aura.resident_recurrent_sft_projected_example.v1"
SAMPLING_RECEIPT_SCHEMA: Final = "aura.resident_recurrent_sft_sampling_receipt.v1"
ANSWER_INSTRUCTION: Final = (
    "Solve the task using the recurrent workspace. Return exactly one final line "
    "in the form FINAL_ANSWER: {JSON value}. Do not write anything after it."
)
MAX_ROWS: Final = 100_000


class ResidentSFTBootstrapExecutionError(ValueError):
    """Dataset projection or deterministic scheduling violated its contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentSFTBootstrapExecutionError(code)


def _tokens(value: Any, *, role: str) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(type(token) is not int or token < 0 for token in value)
    ):
        _fail(f"resident_sft_execution_{role}_tokens_invalid")
    return list(value)


def _encode(tokenizer: Any, text: str, *, role: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        _fail("resident_sft_execution_tokenizer_encode_missing")
    try:
        encoded = encode(text, add_special_tokens=False)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ResidentSFTBootstrapExecutionError(
            f"resident_sft_execution_{role}_encode_failed"
        ) from exc
    return _tokens(encoded, role=role)


def _decode(tokenizer: Any, tokens: Sequence[int], *, role: str) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        _fail("resident_sft_execution_tokenizer_decode_missing")
    try:
        decoded = decode(list(tokens), skip_special_tokens=False)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ResidentSFTBootstrapExecutionError(
            f"resident_sft_execution_{role}_decode_failed"
        ) from exc
    if not isinstance(decoded, str):
        _fail(f"resident_sft_execution_{role}_decode_invalid")
    return decoded


def project_example(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    """Project one authority-normalized row onto the live chat token boundary."""

    if not isinstance(row, Mapping) or set(row) != {
        "task_id",
        "family",
        "depth",
        "prompt",
        "answer",
        "ordinal",
    }:
        _fail("resident_sft_execution_row_schema_invalid")
    if type(max_seq_length) is not int or not 32 <= max_seq_length <= 32_768:
        _fail("resident_sft_execution_max_seq_length_invalid")
    task_id = row.get("task_id")
    family = row.get("family")
    depth = row.get("depth")
    prompt = row.get("prompt")
    answer = row.get("answer")
    ordinal = row.get("ordinal")
    if (
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(family, str)
        or not family
        or type(depth) is not int
        or not 1 <= depth <= 64
        or not isinstance(prompt, str)
        or not prompt
        or not isinstance(answer, str)
        or not answer.startswith("FINAL_ANSWER:")
        or type(ordinal) is not int
        or ordinal < 0
    ):
        _fail("resident_sft_execution_row_invalid")
    render = getattr(tokenizer, "apply_chat_template", None)
    if not callable(render):
        _fail("resident_sft_execution_chat_template_missing")
    content = f"{ANSWER_INSTRUCTION}\n\n{prompt}"
    try:
        rendered = render(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ResidentSFTBootstrapExecutionError(
            "resident_sft_execution_chat_template_failed"
        ) from exc
    if not isinstance(rendered, str) or not rendered or answer in rendered:
        _fail("resident_sft_execution_prompt_render_invalid")

    prompt_tokens = _encode(tokenizer, rendered, role="prompt")
    answer_text_tokens = _encode(tokenizer, answer, role="answer")
    if _decode(tokenizer, answer_text_tokens, role="answer") != answer:
        _fail("resident_sft_execution_answer_round_trip_mismatch")
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0:
        _fail("resident_sft_execution_eos_token_invalid")
    answer_tokens = list(answer_text_tokens)
    if answer_tokens[-1] != eos:
        answer_tokens.append(eos)
    if len(prompt_tokens) + len(answer_tokens) > max_seq_length:
        _fail("resident_sft_execution_sequence_budget_exceeded")

    identity_material = {
        "schema": PROJECTED_EXAMPLE_SCHEMA,
        "task_id": task_id,
        "family": family,
        "depth": depth,
        "ordinal": ordinal,
        "prompt_text_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "answer_text_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "prompt_tokens_sha256": sha256_json(prompt_tokens),
        "answer_tokens_sha256": sha256_json(answer_tokens),
        "prompt_token_count": len(prompt_tokens),
        "answer_token_count": len(answer_tokens),
        "eos_token_id": eos,
        "bridge_tokens": [],
        "max_seq_length": max_seq_length,
    }
    return {
        **identity_material,
        "example_id": sha256_json(identity_material),
        "prompt_tokens": prompt_tokens,
        "answer_tokens": answer_tokens,
    }


def project_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_seq_length: int,
) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or not 1 <= len(rows) <= MAX_ROWS
    ):
        _fail("resident_sft_execution_rows_invalid")
    projected = [
        project_example(row, tokenizer=tokenizer, max_seq_length=max_seq_length)
        for row in rows
    ]
    identities = [row["example_id"] for row in projected]
    if len(identities) != len(set(identities)):
        _fail("resident_sft_execution_projected_identity_duplicate")
    return projected


def execution_spec_for_projected_row(
    row: Mapping[str, Any],
    *,
    base_spec: RLCExecutionSpec,
) -> RLCExecutionSpec:
    """Bind a projected example's authorized depth to the executed graph.

    Dataset depth is part of the example identity and sampling strata.  It must
    therefore select the actual recurrent transition count, rather than remain
    descriptive metadata while every row executes ``base_spec`` unchanged.
    Only ``recurrent_steps`` may vary; all other campaign controls remain bound
    to the authority-signed base specification.
    """

    if not isinstance(row, Mapping):
        _fail("resident_sft_execution_projected_row_invalid")
    depth = row.get("depth")
    if type(depth) is not int or not 1 <= depth <= 64:
        _fail("resident_sft_execution_projected_depth_invalid")
    try:
        executed = base_spec.with_depth(depth)
    except (TypeError, ValueError) as exc:
        raise ResidentSFTBootstrapExecutionError(
            "resident_sft_execution_projected_depth_spec_invalid"
        ) from exc
    base_controls = base_spec.to_dict()
    executed_controls = executed.to_dict()
    base_controls.pop("recurrent_steps")
    executed_controls.pop("recurrent_steps")
    if base_controls != executed_controls or executed.recurrent_steps != depth:
        _fail("resident_sft_execution_projected_depth_binding_drift")
    return executed


def _schedule_digest(seed: int, epoch: int, *parts: Any) -> bytes:
    return hashlib.sha256(
        repr((seed, epoch, *parts)).encode("utf-8")
    ).digest()


def _sampling_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or not 1 <= len(rows) <= MAX_ROWS
    ):
        _fail("resident_sft_execution_sampling_rows_invalid")
    material: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("resident_sft_execution_sampling_row_invalid")
        example_id = row.get("example_id")
        family = row.get("family")
        depth = row.get("depth")
        if (
            not isinstance(example_id, str)
            or len(example_id) != 64
            or any(character not in "0123456789abcdef" for character in example_id)
            or example_id in seen
            or not isinstance(family, str)
            or not family
            or type(depth) is not int
            or not 1 <= depth <= 64
        ):
            _fail("resident_sft_execution_sampling_row_identity_invalid")
        seen.add(example_id)
        material.append(dict(row))
    return material


def family_depth_balanced_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    """Interleave family/depth strata while visiting each row exactly once."""

    material = _sampling_rows(rows)
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        _fail("resident_sft_execution_sampling_seed_invalid")
    if type(epoch) is not int or epoch < 0:
        _fail("resident_sft_execution_sampling_epoch_invalid")
    buckets: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(material):
        buckets[row["family"]][row["depth"]].append(index)
    for family, depth_buckets in buckets.items():
        for depth, indices in depth_buckets.items():
            indices.sort(
                key=lambda index: (
                    _schedule_digest(
                        seed,
                        epoch,
                        "row",
                        family,
                        depth,
                        material[index]["example_id"],
                    ),
                    index,
                )
            )
    families = sorted(
        buckets,
        key=lambda family: (
            _schedule_digest(seed, epoch, "family", family),
            family,
        ),
    )
    depths = {
        family: sorted(
            buckets[family],
            key=lambda depth: (
                _schedule_digest(seed, epoch, "depth", family, depth),
                depth,
            ),
        )
        for family in families
    }
    positions = {
        (family, depth): 0 for family in families for depth in depths[family]
    }
    depth_cursor = {family: 0 for family in families}
    order: list[int] = []
    round_index = 0
    while len(order) < len(material):
        active_families = [
            family
            for family in families
            if any(
                positions[(family, depth)] < len(buckets[family][depth])
                for depth in depths[family]
            )
        ]
        if not active_families:
            break
        offset = round_index % len(active_families)
        for family in active_families[offset:] + active_families[:offset]:
            family_depths = depths[family]
            for attempt in range(len(family_depths)):
                depth_position = (depth_cursor[family] + attempt) % len(family_depths)
                depth = family_depths[depth_position]
                position_key = (family, depth)
                position = positions[position_key]
                if position >= len(buckets[family][depth]):
                    continue
                order.append(buckets[family][depth][position])
                positions[position_key] = position + 1
                depth_cursor[family] = (depth_position + 1) % len(family_depths)
                break
        round_index += 1
    if len(order) != len(material) or sorted(order) != list(range(len(material))):
        _fail("resident_sft_execution_sampling_not_without_replacement")
    return order


def validate_family_depth_balanced_order(
    rows: Sequence[Mapping[str, Any]],
    order: Sequence[int],
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    observed = list(order)
    expected = family_depth_balanced_order(rows, seed=seed, epoch=epoch)
    if observed != expected:
        _fail("resident_sft_execution_sampling_order_drift")
    return observed


def sampling_receipt(
    rows: Sequence[Mapping[str, Any]],
    order: Sequence[int],
    *,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    material = _sampling_rows(rows)
    observed = validate_family_depth_balanced_order(
        material,
        order,
        seed=seed,
        epoch=epoch,
    )
    family_counts = Counter(material[index]["family"] for index in observed)
    stratum_counts = Counter(
        f"{material[index]['family']}:{material[index]['depth']}" for index in observed
    )
    body = {
        "schema": SAMPLING_RECEIPT_SCHEMA,
        "sampler": SAMPLER_NAME,
        "seed": seed,
        "epoch": epoch,
        "row_count": len(material),
        "order_sha256": sha256_json(observed),
        "all_rows_once": len(observed) == len(set(observed)) == len(material),
        "family_counts": dict(sorted(family_counts.items())),
        "family_depth_counts": dict(sorted(stratum_counts.items())),
    }
    return {**body, "receipt_sha256": sha256_json(body)}


def advance_sample_history(
    previous_sha256: str,
    *,
    example_id: str,
    step: int,
    epoch: int,
    cursor: int,
) -> str:
    if (
        not isinstance(previous_sha256, str)
        or len(previous_sha256) != 64
        or any(character not in "0123456789abcdef" for character in previous_sha256)
        or not isinstance(example_id, str)
        or len(example_id) != 64
        or any(character not in "0123456789abcdef" for character in example_id)
        or type(step) is not int
        or step < 1
        or type(epoch) is not int
        or epoch < 0
        or type(cursor) is not int
        or cursor < 1
    ):
        _fail("resident_sft_execution_sample_history_input_invalid")
    return sha256_json(
        {
            "previous_sha256": previous_sha256,
            "example_id": example_id,
            "step": step,
            "epoch": epoch,
            "cursor": cursor,
        }
    )


def initial_sample_history() -> str:
    return ZERO_SHA256


def adapter_topology_sha256(tensors: Mapping[str, Any]) -> str:
    """Bind trainable names, shapes, and dtypes without binding their values."""

    if not isinstance(tensors, Mapping) or not tensors:
        _fail("resident_sft_execution_adapter_topology_empty")
    rows: list[dict[str, Any]] = []
    for name in sorted(tensors):
        tensor = tensors[name]
        shape = getattr(tensor, "shape", None)
        dtype = getattr(tensor, "dtype", None)
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(shape, (list, tuple))
            or not shape
            or any(type(dimension) is not int or dimension < 1 for dimension in shape)
            or dtype is None
        ):
            _fail("resident_sft_execution_adapter_topology_invalid")
        rows.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": str(dtype),
            }
        )
    return sha256_json(rows)


__all__ = [
    "ANSWER_INSTRUCTION",
    "PROJECTED_EXAMPLE_SCHEMA",
    "ResidentSFTBootstrapExecutionError",
    "SAMPLING_RECEIPT_SCHEMA",
    "advance_sample_history",
    "adapter_topology_sha256",
    "execution_spec_for_projected_row",
    "family_depth_balanced_order",
    "initial_sample_history",
    "project_example",
    "project_rows",
    "sampling_receipt",
    "validate_family_depth_balanced_order",
]
