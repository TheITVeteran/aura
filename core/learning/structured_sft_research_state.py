"""Crash-consistent state for bounded synthetic recurrent-SFT research."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Never

from core.learning.structured_sft_research_authority import (
    SAMPLER,
    deterministic_order,
)
from core.runtime.atomic_writer import (
    atomic_write_bytes,
    atomic_write_text,
    durable_unlink,
    ensure_private_directory,
    interprocess_file_lock,
)

CHECKPOINT_SCHEMA: Final = "aura.rlc.synthetic_recurrent_sft_checkpoint.v1"
POINTER_SCHEMA: Final = "aura.rlc.synthetic_recurrent_sft_checkpoint_pointer.v1"
JOURNAL_EVENT_SCHEMA: Final = "aura.rlc.synthetic_recurrent_sft_journal_event.v1"
JOURNAL_POINTER_SCHEMA: Final = "aura.rlc.synthetic_recurrent_sft_journal_pointer.v1"
ZERO_SHA256: Final = "0" * 64
_MAX_TRAIL_ENTRIES = 100_000
_MAX_ORDER_SIZE = 1_000_000


class StructuredSFTResearchStateError(RuntimeError):
    """Durable research state is incomplete, stale, or inconsistent."""


def _fail(code: str) -> Never:
    raise StructuredSFTResearchStateError(str(code or "structured_sft_state_invalid"))


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        _fail("structured_sft_state_noncanonical_value")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_nonnegative(value: Any, *, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        _fail(f"structured_sft_state_{role}_invalid")
    return float(value)


def _trail(value: Any, *, role: str) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or len(value) > _MAX_TRAIL_ENTRIES
        or any(not isinstance(entry, dict) for entry in value)
    ):
        _fail(f"structured_sft_state_{role}_invalid")
    return list(value)


def validate_checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "authority_sha256",
        "dataset_sha256",
        "tokenization_identity_sha256",
        "model_identity_sha256",
        "source_closure_sha256",
        "execution_spec_sha256",
        "trainer_config_sha256",
        "step",
        "optimizer_updates",
        "epoch",
        "cursor",
        "order",
        "sampler",
        "seed",
        "train_example_count",
        "validation_example_count",
        "elapsed_training_s",
        "invocation_count",
        "loss_trail",
        "validation_trail",
        "pending_losses",
        "baseline_validation",
        "last_step_committed",
        "terminal",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        _fail("structured_sft_state_schema_invalid")
    for role in (
        "authority_sha256",
        "dataset_sha256",
        "tokenization_identity_sha256",
        "model_identity_sha256",
        "source_closure_sha256",
        "execution_spec_sha256",
        "trainer_config_sha256",
    ):
        if not _is_sha256(state.get(role)):
            _fail(f"structured_sft_state_{role}_invalid")
    step = state.get("step")
    optimizer_updates = state.get("optimizer_updates")
    epoch = state.get("epoch")
    cursor = state.get("cursor")
    seed = state.get("seed")
    train_count = state.get("train_example_count")
    valid_count = state.get("validation_example_count")
    invocation = state.get("invocation_count")
    if (
        type(step) is not int
        or step < 0
        or type(optimizer_updates) is not int
        or optimizer_updates != step
        or type(epoch) is not int
        or epoch < 0
        or type(cursor) is not int
        or cursor < 0
        or type(seed) is not int
        or seed < 0
        or type(train_count) is not int
        or not 1 <= train_count <= _MAX_ORDER_SIZE
        or type(valid_count) is not int
        or not 1 <= valid_count <= _MAX_ORDER_SIZE
        or type(invocation) is not int
        or invocation < 1
    ):
        _fail("structured_sft_state_cursor_invalid")
    order = state.get("order")
    if (
        not isinstance(order, list)
        or len(order) != train_count
        or cursor > len(order)
        or order != deterministic_order(train_count, seed=seed, epoch=epoch)
    ):
        _fail("structured_sft_state_order_invalid")
    if state.get("sampler") != SAMPLER:
        _fail("structured_sft_state_sampler_invalid")
    _finite_nonnegative(state.get("elapsed_training_s"), role="elapsed")
    _trail(state.get("loss_trail"), role="loss_trail")
    _trail(state.get("validation_trail"), role="validation_trail")
    pending = state.get("pending_losses")
    if (
        not isinstance(pending, list)
        or len(pending) > 100_000
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in pending
        )
    ):
        _fail("structured_sft_state_pending_losses_invalid")
    baseline = state.get("baseline_validation")
    if baseline is not None and not isinstance(baseline, dict):
        _fail("structured_sft_state_baseline_invalid")
    if type(state.get("last_step_committed")) is not bool:
        _fail("structured_sft_state_commit_flag_invalid")
    if step > 0 and state.get("last_step_committed") is not True:
        _fail("structured_sft_state_partial_update_forbidden")
    if type(state.get("terminal")) is not bool:
        _fail("structured_sft_state_terminal_invalid")
    return json.loads(canonical_json_bytes(state))


@dataclass(frozen=True, slots=True)
class LoadedStructuredSFTCheckpoint:
    checkpoint_dir: Path
    state: dict[str, Any]
    adapter_tensors: dict[str, Any]
    optimizer_state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class InspectedStructuredSFTCheckpoint:
    checkpoint_dir: Path
    complete_sha256: str
    state: dict[str, Any]
    adapter_binding: dict[str, Any]
    optimizer_binding: dict[str, Any]


def _write_safetensors(path: Path, tensors: Mapping[str, Any]) -> bytes:
    import mlx.core as mx

    if not tensors:
        _fail("structured_sft_state_tensor_mapping_empty")
    scratch = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.safetensors"
    try:
        mx.save_safetensors(str(scratch), dict(tensors))
        payload = scratch.read_bytes()
    finally:
        durable_unlink(scratch, missing_ok=True)
    atomic_write_bytes(path, payload, mode=0o600)
    return payload


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StructuredSFTResearchStateError(
            f"structured_sft_state_{role}_unreadable"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"structured_sft_state_{role}_invalid")
    return value


def _contained_generation(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.startswith("checkpoints/"):
        _fail("structured_sft_state_checkpoint_path_invalid")
    checkpoint_root = (root / "checkpoints").resolve(strict=True)
    generation = (root / value).resolve(strict=True)
    if generation.parent != checkpoint_root or not generation.is_dir():
        _fail("structured_sft_state_checkpoint_path_escape")
    return generation


def save_checkpoint(
    out_dir: Path,
    *,
    adapter_tensors: Mapping[str, Any],
    optimizer_tensors: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Path:
    """Publish complete immutable tensors before advancing ``latest``."""

    validated = validate_checkpoint_state(state)
    root = ensure_private_directory(out_dir.expanduser())
    checkpoints = ensure_private_directory(root / "checkpoints")
    with interprocess_file_lock(root / ".checkpoint.lock"):
        checkpoint_id = f"step-{validated['step']:08d}-{uuid.uuid4().hex}"
        generation = ensure_private_directory(checkpoints / checkpoint_id)
        try:
            adapter_payload = _write_safetensors(
                generation / "quarantine_adapter.safetensors",
                adapter_tensors,
            )
            optimizer_payload = _write_safetensors(
                generation / "optimizer.safetensors",
                optimizer_tensors,
            )
            complete = {
                **validated,
                "schema": CHECKPOINT_SCHEMA,
                "checkpoint_id": checkpoint_id,
                "created_unix": time.time(),
                "adapter": {
                    "path": "quarantine_adapter.safetensors",
                    "sha256": sha256_bytes(adapter_payload),
                    "size_bytes": len(adapter_payload),
                },
                "optimizer": {
                    "path": "optimizer.safetensors",
                    "sha256": sha256_bytes(optimizer_payload),
                    "size_bytes": len(optimizer_payload),
                },
            }
            complete_payload = canonical_json_bytes(complete)
            atomic_write_bytes(
                generation / "complete.json",
                complete_payload,
                mode=0o600,
            )
            pointer = {
                "schema": POINTER_SCHEMA,
                "checkpoint": f"checkpoints/{checkpoint_id}",
                "complete_sha256": sha256_bytes(complete_payload),
            }
            atomic_write_text(
                root / "latest.json",
                canonical_json_bytes(pointer).decode("ascii"),
                encoding="ascii",
                mode=0o600,
            )
        except BaseException:
            # An unreachable generation cannot be mistaken for the latest
            # complete checkpoint and remains available for forensic cleanup.
            raise
        return generation


def _load_bound_tensors(
    generation: Path,
    state: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    import mlx.core as mx

    binding = state.get(role)
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256", "size_bytes"}
        or not isinstance(binding.get("path"), str)
        or Path(binding["path"]).name != binding["path"]
        or not _is_sha256(binding.get("sha256"))
        or type(binding.get("size_bytes")) is not int
        or binding["size_bytes"] <= 0
    ):
        _fail(f"structured_sft_state_{role}_binding_invalid")
    path = (generation / binding["path"]).resolve(strict=True)
    if path.parent != generation:
        _fail(f"structured_sft_state_{role}_path_escape")
    payload = path.read_bytes()
    if (
        len(payload) != binding["size_bytes"]
        or sha256_bytes(payload) != binding["sha256"]
    ):
        _fail(f"structured_sft_state_{role}_commitment_mismatch")
    tensors = mx.load(str(path))
    if not isinstance(tensors, dict) or not tensors:
        _fail(f"structured_sft_state_{role}_tensor_container_invalid")
    return dict(tensors)


def _validate_bound_file(
    generation: Path,
    state: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    binding = state.get(role)
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256", "size_bytes"}
        or not isinstance(binding.get("path"), str)
        or Path(binding["path"]).name != binding["path"]
        or not _is_sha256(binding.get("sha256"))
        or type(binding.get("size_bytes")) is not int
        or binding["size_bytes"] <= 0
    ):
        _fail(f"structured_sft_state_{role}_binding_invalid")
    path = (generation / binding["path"]).resolve(strict=True)
    if path.parent != generation:
        _fail(f"structured_sft_state_{role}_path_escape")
    payload = path.read_bytes()
    if (
        len(payload) != binding["size_bytes"]
        or sha256_bytes(payload) != binding["sha256"]
    ):
        _fail(f"structured_sft_state_{role}_commitment_mismatch")
    return dict(binding)


def inspect_checkpoint(
    out_dir: Path,
    *,
    expected_bindings: Mapping[str, str],
) -> InspectedStructuredSFTCheckpoint:
    """Verify a complete checkpoint without deserializing model tensors."""

    expected_roles = {
        "authority_sha256",
        "dataset_sha256",
        "tokenization_identity_sha256",
        "model_identity_sha256",
        "source_closure_sha256",
        "execution_spec_sha256",
        "trainer_config_sha256",
    }
    if set(expected_bindings) != expected_roles or any(
        not _is_sha256(value) for value in expected_bindings.values()
    ):
        _fail("structured_sft_state_expected_bindings_invalid")
    root = out_dir.expanduser().resolve(strict=True)
    with interprocess_file_lock(root / ".checkpoint.lock"):
        pointer = _read_json(root / "latest.json", role="pointer")
        if (
            set(pointer) != {"schema", "checkpoint", "complete_sha256"}
            or pointer.get("schema") != POINTER_SCHEMA
            or not _is_sha256(pointer.get("complete_sha256"))
        ):
            _fail("structured_sft_state_pointer_invalid")
        generation = _contained_generation(root, pointer["checkpoint"])
        complete_payload = (generation / "complete.json").read_bytes()
        if sha256_bytes(complete_payload) != pointer["complete_sha256"]:
            _fail("structured_sft_state_complete_commitment_mismatch")
        complete = _read_json(generation / "complete.json", role="complete")
        metadata = {
            key: value
            for key, value in complete.items()
            if key
            not in {
                "schema",
                "checkpoint_id",
                "created_unix",
                "adapter",
                "optimizer",
            }
        }
        if (
            complete.get("schema") != CHECKPOINT_SCHEMA
            or complete.get("checkpoint_id") != generation.name
            or not isinstance(complete.get("created_unix"), (int, float))
        ):
            _fail("structured_sft_state_complete_invalid")
        state = validate_checkpoint_state(metadata)
        if any(state.get(role) != value for role, value in expected_bindings.items()):
            _fail("structured_sft_state_protocol_binding_mismatch")
        adapter_binding = _validate_bound_file(
            generation,
            complete,
            role="adapter",
        )
        optimizer_binding = _validate_bound_file(
            generation,
            complete,
            role="optimizer",
        )
        return InspectedStructuredSFTCheckpoint(
            checkpoint_dir=generation,
            complete_sha256=pointer["complete_sha256"],
            state=state,
            adapter_binding=adapter_binding,
            optimizer_binding=optimizer_binding,
        )


def load_checkpoint(
    out_dir: Path,
    *,
    expected_bindings: Mapping[str, str],
) -> LoadedStructuredSFTCheckpoint:
    """Load only a fully committed generation with exact protocol bindings."""

    inspected = inspect_checkpoint(
        out_dir,
        expected_bindings=expected_bindings,
    )
    with interprocess_file_lock(
        out_dir.expanduser().resolve(strict=True) / ".checkpoint.lock"
    ):
        adapter = _load_bound_tensors(
            inspected.checkpoint_dir,
            {"adapter": inspected.adapter_binding},
            role="adapter",
        )
        optimizer_tensors = _load_bound_tensors(
            inspected.checkpoint_dir,
            {"optimizer": inspected.optimizer_binding},
            role="optimizer",
        )
        from mlx.utils import tree_unflatten

        optimizer_state = tree_unflatten(optimizer_tensors)
        if not isinstance(optimizer_state, dict):
            _fail("structured_sft_state_optimizer_tree_invalid")
        return LoadedStructuredSFTCheckpoint(
            checkpoint_dir=inspected.checkpoint_dir,
            state=inspected.state,
            adapter_tensors=adapter,
            optimizer_state=optimizer_state,
        )


def _validate_event(raw: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "sequence",
        "event_type",
        "payload",
        "recorded_at_unix",
        "previous_event_sha256",
        "event_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        _fail("structured_sft_journal_event_schema_invalid")
    body = dict(raw)
    observed = body.pop("event_sha256", None)
    if (
        raw.get("schema") != JOURNAL_EVENT_SCHEMA
        or type(raw.get("sequence")) is not int
        or raw["sequence"] < 1
        or not isinstance(raw.get("event_type"), str)
        or not raw["event_type"]
        or not isinstance(raw.get("payload"), Mapping)
        or not isinstance(raw.get("recorded_at_unix"), (int, float))
        or not _is_sha256(raw.get("previous_event_sha256"))
        or observed != sha256_bytes(canonical_json_bytes(body))
    ):
        _fail("structured_sft_journal_event_invalid")
    return json.loads(canonical_json_bytes(raw))


def append_journal_event(
    run_dir: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one immutable hash-linked event and atomically advance its head."""

    root = ensure_private_directory(run_dir.expanduser())
    journal = ensure_private_directory(root / "journal")
    with interprocess_file_lock(root / ".journal.lock"):
        pointer_path = root / "journal_latest.json"
        if pointer_path.exists():
            pointer = _read_json(pointer_path, role="journal_pointer")
            if (
                set(pointer) != {
                    "schema",
                    "sequence",
                    "event",
                    "event_sha256",
                }
                or pointer.get("schema") != JOURNAL_POINTER_SCHEMA
                or type(pointer.get("sequence")) is not int
                or not _is_sha256(pointer.get("event_sha256"))
            ):
                _fail("structured_sft_journal_pointer_invalid")
            prior_path = (root / pointer["event"]).resolve(strict=True)
            if prior_path.parent != journal.resolve(strict=True):
                _fail("structured_sft_journal_pointer_escape")
            prior = _validate_event(_read_json(prior_path, role="journal_event"))
            if (
                prior["sequence"] != pointer["sequence"]
                or prior["event_sha256"] != pointer["event_sha256"]
            ):
                _fail("structured_sft_journal_pointer_mismatch")
            sequence = prior["sequence"] + 1
            previous = prior["event_sha256"]
        else:
            sequence = 1
            previous = ZERO_SHA256
        body = {
            "schema": JOURNAL_EVENT_SCHEMA,
            "sequence": sequence,
            "event_type": str(event_type or ""),
            "payload": dict(payload),
            "recorded_at_unix": time.time(),
            "previous_event_sha256": previous,
        }
        event = _validate_event(
            {
                **body,
                "event_sha256": sha256_bytes(canonical_json_bytes(body)),
            }
        )
        filename = f"{sequence:08d}-{event['event_sha256']}.json"
        event_path = journal / filename
        atomic_write_bytes(
            event_path,
            canonical_json_bytes(event),
            mode=0o600,
        )
        pointer = {
            "schema": JOURNAL_POINTER_SCHEMA,
            "sequence": sequence,
            "event": f"journal/{filename}",
            "event_sha256": event["event_sha256"],
        }
        atomic_write_text(
            pointer_path,
            canonical_json_bytes(pointer).decode("ascii"),
            encoding="ascii",
            mode=0o600,
        )
        return event


def validate_journal(run_dir: Path) -> list[dict[str, Any]]:
    """Reconstruct the complete immutable journal from sequence one."""

    root = run_dir.expanduser().resolve(strict=True)
    pointer = _read_json(root / "journal_latest.json", role="journal_pointer")
    if pointer.get("schema") != JOURNAL_POINTER_SCHEMA:
        _fail("structured_sft_journal_pointer_invalid")
    count = pointer.get("sequence")
    if type(count) is not int or count < 1:
        _fail("structured_sft_journal_pointer_invalid")
    files = sorted((root / "journal").glob("*.json"))
    if len(files) != count:
        _fail("structured_sft_journal_sequence_gap")
    events: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    for sequence, path in enumerate(files, start=1):
        event = _validate_event(_read_json(path, role="journal_event"))
        if (
            event["sequence"] != sequence
            or event["previous_event_sha256"] != previous
            or path.name != f"{sequence:08d}-{event['event_sha256']}.json"
        ):
            _fail("structured_sft_journal_chain_invalid")
        events.append(event)
        previous = event["event_sha256"]
    if (
        pointer.get("event") != f"journal/{files[-1].name}"
        or pointer.get("event_sha256") != previous
    ):
        _fail("structured_sft_journal_head_mismatch")
    return events


__all__ = [
    "CHECKPOINT_SCHEMA",
    "InspectedStructuredSFTCheckpoint",
    "JOURNAL_EVENT_SCHEMA",
    "JOURNAL_POINTER_SCHEMA",
    "LoadedStructuredSFTCheckpoint",
    "POINTER_SCHEMA",
    "StructuredSFTResearchStateError",
    "append_journal_event",
    "canonical_json_bytes",
    "inspect_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
    "sha256_bytes",
    "validate_checkpoint_state",
    "validate_journal",
]
