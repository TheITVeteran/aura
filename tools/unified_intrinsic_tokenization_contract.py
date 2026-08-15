"""Create-once tokenizer contract for unified intrinsic training data."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from core.learning.recurrence_curriculum import (
    RecurrenceTrainingTask,
    StructuredTransitionProgram,
    StructuredTransitionTrace,
)
from core.learning.recurrent_action_schema import (
    ACTION_CARDINALITY,
    ACTION_SLOT_NAMES,
    action_value_semantic_label,
)
from core.learning.recurrent_answer_emission import (
    tokenizer_answer_emission_contract,
)
from core.learning.recurrent_literal_grounding import (
    LITERAL_MAX_VALUE,
    LiteralObservationContract,
    tokenizer_digit_token_ids,
)
from core.learning.recurrent_opcode_grounding import tokenizer_opcode_contract
from core.learning.recurrent_state_schema import STATE_CARDINALITY, STATE_SLOT_NAMES
from core.runtime.atomic_writer import atomic_write_bytes_if_absent
from tools.unified_intrinsic_resident_identity import canonical_bytes, canonical_sha256

TOKENIZED_DATASET_SCHEMA: Final = "aura.unified_intrinsic.tokenized_dataset.v1"
TOKENIZED_DATASET_FILENAME: Final = "tokenized_dataset.json"
SOURCE_DATASET_SCHEMA: Final = "aura.unified_intrinsic_dataset.v1"
SOURCE_DATASET_FILENAME: Final = "dataset.json"
MAX_FROZEN_DATASET_BYTES: Final = 512 * 1024 * 1024


class UnifiedTokenizationContractError(RuntimeError):
    """Frozen text or tokenizer behavior differs from its campaign contract."""


def _canonical_document(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def _frozen_path(path: Path, *, strict: bool) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise UnifiedTokenizationContractError("frozen_dataset_path_is_symlink")
    return expanded.resolve(strict=strict)


def _read_frozen_bytes(path: Path, *, error: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) & 0o222
                or not 0 < before.st_size <= MAX_FROZEN_DATASET_BYTES
            ):
                raise UnifiedTokenizationContractError(error)
            chunks: list[bytes] = []
            remaining = MAX_FROZEN_DATASET_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedTokenizationContractError(error) from exc
    payload = b"".join(chunks)
    if (
        len(payload) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise UnifiedTokenizationContractError(error)
    return payload


def freeze_source_dataset(
    path: Path,
    train_tasks: Sequence[RecurrenceTrainingTask],
    holdout_tasks: Sequence[RecurrenceTrainingTask],
) -> dict[str, Any]:
    """Freeze exact private programs, not merely their generation parameters."""

    path = _frozen_path(path, strict=False)
    if path.name != SOURCE_DATASET_FILENAME:
        raise UnifiedTokenizationContractError("source_dataset_path_invalid")
    train_rows = [{"task_id": task.task_id, **asdict(task)} for task in train_tasks]
    holdout_rows = [
        {"task_id": task.task_id, **asdict(task)} for task in holdout_tasks
    ]
    train_ids = {row["task_id"] for row in train_rows}
    holdout_ids = {row["task_id"] for row in holdout_rows}
    train_prompts = {row["prompt"] for row in train_rows}
    holdout_prompts = {row["prompt"] for row in holdout_rows}
    if (
        not train_rows
        or not holdout_rows
        or train_ids & holdout_ids
        or train_prompts & holdout_prompts
    ):
        raise UnifiedTokenizationContractError("source_dataset_partitions_invalid")
    document = {
        "schema": SOURCE_DATASET_SCHEMA,
        "train": train_rows,
        "holdout": holdout_rows,
    }
    payload = _canonical_document(document)
    atomic_write_bytes_if_absent(path, payload, mode=0o400)
    observed = _read_frozen_bytes(path, error="source_dataset_unreadable")
    if observed != payload:
        raise UnifiedTokenizationContractError("unified recurrence dataset differs")
    body = {
        "schema": SOURCE_DATASET_SCHEMA,
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
        "train_task_ids_sha256": canonical_sha256(sorted(train_ids)),
        "holdout_task_ids_sha256": canonical_sha256(sorted(holdout_ids)),
        "partition_overlap": 0,
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def load_source_dataset(
    path: Path,
) -> tuple[list[RecurrenceTrainingTask], list[RecurrenceTrainingTask]]:
    path = _frozen_path(path, strict=True)
    if path.name != SOURCE_DATASET_FILENAME:
        raise UnifiedTokenizationContractError("source_dataset_path_invalid")
    try:
        payload = _read_frozen_bytes(path, error="source_dataset_unreadable")
        decoded = json.loads(payload.decode("ascii"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise UnifiedTokenizationContractError("source_dataset_unreadable") from exc
    if (
        not isinstance(decoded, dict)
        or payload != _canonical_document(decoded)
        or set(decoded) != {"schema", "train", "holdout"}
        or decoded.get("schema") != SOURCE_DATASET_SCHEMA
    ):
        raise UnifiedTokenizationContractError("source_dataset_contract_differs")

    def restore_trace(value: Any) -> StructuredTransitionTrace | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {
            "family",
            "depth",
            "field_names",
            "states",
        }:
            raise UnifiedTokenizationContractError("source_dataset_trace_differs")
        return StructuredTransitionTrace(
            family=value["family"],
            depth=value["depth"],
            field_names=tuple(value["field_names"]),
            states=tuple(tuple(state) for state in value["states"]),
        )

    def restore_task(value: Any) -> RecurrenceTrainingTask:
        required = {
            "task_id",
            "prompt",
            "answer",
            "depth",
            "family",
            "seed",
            "solution",
            "transition_trace",
            "transition_program",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise UnifiedTokenizationContractError("source_dataset_task_differs")
        trace = restore_trace(value["transition_trace"])
        raw_program = value["transition_program"]
        program: StructuredTransitionProgram | None = None
        if raw_program is not None:
            if not isinstance(raw_program, dict) or set(raw_program) != {
                "state_trace",
                "action_field_names",
                "actions",
            }:
                raise UnifiedTokenizationContractError(
                    "source_dataset_program_differs"
                )
            program_trace = restore_trace(raw_program["state_trace"])
            if program_trace is None or program_trace != trace:
                raise UnifiedTokenizationContractError(
                    "source_dataset_program_trace_differs"
                )
            if trace is None:
                raise UnifiedTokenizationContractError(
                    "source_dataset_program_trace_differs"
                )
            program = StructuredTransitionProgram(
                # The equality check above verifies the redundant frozen copy.
                # Bind the restored program to the task's canonical trace so
                # every downstream state/action audit has one authority.
                state_trace=trace,
                action_field_names=tuple(raw_program["action_field_names"]),
                actions=tuple(tuple(action) for action in raw_program["actions"]),
            )
        task = RecurrenceTrainingTask(
            prompt=value["prompt"],
            answer=value["answer"],
            depth=value["depth"],
            family=value["family"],
            seed=value["seed"],
            solution=value["solution"],
            transition_trace=trace,
            transition_program=program,
        )
        if task.task_id != value["task_id"]:
            raise UnifiedTokenizationContractError("source_dataset_task_id_differs")
        return task

    train = decoded["train"]
    holdout = decoded["holdout"]
    if not isinstance(train, list) or not train or not isinstance(holdout, list) or not holdout:
        raise UnifiedTokenizationContractError("source_dataset_partitions_empty")
    return [restore_task(row) for row in train], [restore_task(row) for row in holdout]


def _token_ids(tokenizer: Any, text: str, *, special: bool) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=special)
    except TypeError:
        if special:
            encoded = tokenizer.encode(text)
        else:
            raise UnifiedTokenizationContractError(
                "tokenizer_cannot_disable_special_tokens"
            ) from None
    if (
        not isinstance(encoded, (list, tuple))
        or not encoded
        or any(type(token_id) is not int or token_id < 0 for token_id in encoded)
    ):
        raise UnifiedTokenizationContractError("tokenizer_returned_invalid_ids")
    return [int(token_id) for token_id in encoded]


def _target_text(answer: str, bridge: str) -> str:
    target = answer
    marker = "FINAL_ANSWER:"
    if bridge.strip().endswith(marker) and marker in target:
        target = target.split(marker, 1)[1].lstrip()
    if not target:
        raise UnifiedTokenizationContractError("tokenized_target_is_empty")
    return target


def _task_row(
    tokenizer: Any,
    task: Any,
    *,
    partition: str,
    bridge: str,
) -> dict[str, Any]:
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise UnifiedTokenizationContractError(
            f"chat_template_failed:{task.task_id}"
        ) from exc
    if not isinstance(rendered, str) or not rendered:
        raise UnifiedTokenizationContractError(
            f"chat_template_returned_invalid_text:{task.task_id}"
        )
    rendered += bridge
    target = _target_text(task.answer, bridge)
    prompt_ids = _token_ids(tokenizer, rendered, special=True)
    target_ids = _token_ids(tokenizer, target, special=False)
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0:
        raise UnifiedTokenizationContractError("tokenizer_eos_token_invalid")
    target_ids.append(int(eos))
    body = {
        "partition": partition,
        "task_id": task.task_id,
        "family": task.family,
        "depth": task.depth,
        "seed": task.seed,
        "rendered_prompt": rendered,
        "rendered_prompt_sha256": hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest(),
        "prompt_token_ids": prompt_ids,
        "target_text": target,
        "target_text_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        "target_token_ids": target_ids,
    }
    return {**body, "row_sha256": canonical_sha256(body)}


def _grounding_rows(tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot_name in STATE_SLOT_NAMES:
        for value in range(STATE_CARDINALITY):
            label = f"Internal state {slot_name}={value}"
            rows.append(
                {
                    "kind": "state",
                    "slot": slot_name,
                    "value": value,
                    "label": label,
                    "token_ids": _token_ids(tokenizer, label, special=False),
                }
            )
    for slot_name in ACTION_SLOT_NAMES:
        for value in range(ACTION_CARDINALITY):
            label = action_value_semantic_label(slot_name, value)
            rows.append(
                {
                    "kind": "action",
                    "slot": slot_name,
                    "value": value,
                    "label": label,
                    "token_ids": _token_ids(tokenizer, label, special=False),
                }
            )
    for value in range(LITERAL_MAX_VALUE + 1):
        label = str(value)
        rows.append(
            {
                "kind": "literal",
                "slot": None,
                "value": value,
                "label": label,
                "token_ids": _token_ids(tokenizer, label, special=False),
            }
        )
    return rows


def build_tokenized_dataset_document(
    tokenizer: Any,
    train_tasks: Sequence[Any],
    holdout_tasks: Sequence[Any],
    *,
    bridge: str,
    dataset_identity: Mapping[str, Any],
    tokenizer_identity_sha256: str,
) -> dict[str, Any]:
    """Bind every model-facing byte and token id before training starts."""

    if (
        not train_tasks
        or not holdout_tasks
        or not isinstance(bridge, str)
        or not bridge
        or len(tokenizer_identity_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tokenizer_identity_sha256)
    ):
        raise UnifiedTokenizationContractError("tokenization_inputs_invalid")
    dataset_sha256 = dataset_identity.get("identity_sha256")
    if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64:
        raise UnifiedTokenizationContractError("dataset_identity_invalid")
    train_rows = [
        _task_row(tokenizer, task, partition="train", bridge=bridge)
        for task in train_tasks
    ]
    holdout_rows = [
        _task_row(tokenizer, task, partition="holdout", bridge=bridge)
        for task in holdout_tasks
    ]
    if {row["task_id"] for row in train_rows} & {
        row["task_id"] for row in holdout_rows
    }:
        raise UnifiedTokenizationContractError("tokenized_partitions_overlap")
    digit_ids = tokenizer_digit_token_ids(tokenizer)
    literal_contract = LiteralObservationContract(digit_ids)
    opcode_contract = tokenizer_opcode_contract(tokenizer)
    answer_contract = tokenizer_answer_emission_contract(tokenizer, opcode_contract)
    grounding = _grounding_rows(tokenizer)
    body = {
        "schema": TOKENIZED_DATASET_SCHEMA,
        "dataset_identity_sha256": dataset_sha256,
        "tokenizer_identity_sha256": tokenizer_identity_sha256,
        "bridge": bridge,
        "bridge_sha256": hashlib.sha256(bridge.encode("utf-8")).hexdigest(),
        "train": train_rows,
        "holdout": holdout_rows,
        "grounding": grounding,
        "literal_observation_contract": {
            **literal_contract.to_dict(),
            "contract_sha256": literal_contract.contract_sha256,
        },
        "opcode_observation_contract": {
            **opcode_contract.to_dict(),
            "contract_sha256": opcode_contract.contract_sha256,
        },
        "answer_emission_contract": {
            **answer_contract.to_dict(),
            "contract_sha256": answer_contract.contract_sha256,
        },
    }
    return {**body, "document_sha256": canonical_sha256(body)}


def _identity(path: Path, payload: bytes, document: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": TOKENIZED_DATASET_SCHEMA,
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "document_sha256": document["document_sha256"],
        "dataset_identity_sha256": document["dataset_identity_sha256"],
        "tokenizer_identity_sha256": document["tokenizer_identity_sha256"],
        "train_count": len(document["train"]),
        "holdout_count": len(document["holdout"]),
        "grounding_count": len(document["grounding"]),
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def freeze_tokenized_dataset(
    path: Path,
    tokenizer: Any,
    train_tasks: Sequence[Any],
    holdout_tasks: Sequence[Any],
    *,
    bridge: str,
    dataset_identity: Mapping[str, Any],
    tokenizer_identity_sha256: str,
) -> dict[str, Any]:
    path = _frozen_path(path, strict=False)
    if path.name != TOKENIZED_DATASET_FILENAME:
        raise UnifiedTokenizationContractError("tokenized_dataset_path_invalid")
    document = build_tokenized_dataset_document(
        tokenizer,
        train_tasks,
        holdout_tasks,
        bridge=bridge,
        dataset_identity=dataset_identity,
        tokenizer_identity_sha256=tokenizer_identity_sha256,
    )
    payload = _canonical_document(document)
    atomic_write_bytes_if_absent(path, payload, mode=0o400)
    observed = _read_frozen_bytes(path, error="tokenized_dataset_unreadable")
    if observed != payload:
        raise UnifiedTokenizationContractError("tokenized_dataset_differs")
    return _identity(path, payload, document)


def verify_tokenized_dataset(
    path: Path,
    tokenizer: Any,
    train_tasks: Sequence[Any],
    holdout_tasks: Sequence[Any],
    *,
    bridge: str,
    dataset_identity: Mapping[str, Any],
    tokenizer_identity_sha256: str,
) -> dict[str, Any]:
    path = _frozen_path(path, strict=True)
    if path.name != TOKENIZED_DATASET_FILENAME:
        raise UnifiedTokenizationContractError("tokenized_dataset_path_invalid")
    try:
        observed = _read_frozen_bytes(path, error="tokenized_dataset_unreadable")
        decoded = json.loads(observed.decode("ascii"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise UnifiedTokenizationContractError("tokenized_dataset_unreadable") from exc
    expected = build_tokenized_dataset_document(
        tokenizer,
        train_tasks,
        holdout_tasks,
        bridge=bridge,
        dataset_identity=dataset_identity,
        tokenizer_identity_sha256=tokenizer_identity_sha256,
    )
    if (
        not isinstance(decoded, dict)
        or observed != _canonical_document(decoded)
        or decoded != expected
    ):
        raise UnifiedTokenizationContractError("tokenized_dataset_differs")
    return _identity(path, observed, decoded)


__all__ = [
    "SOURCE_DATASET_FILENAME",
    "SOURCE_DATASET_SCHEMA",
    "TOKENIZED_DATASET_FILENAME",
    "TOKENIZED_DATASET_SCHEMA",
    "UnifiedTokenizationContractError",
    "build_tokenized_dataset_document",
    "freeze_source_dataset",
    "freeze_tokenized_dataset",
    "load_source_dataset",
    "verify_tokenized_dataset",
]
