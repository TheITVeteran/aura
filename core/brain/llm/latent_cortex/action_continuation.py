"""Portable resident state at the first recurrent action opportunity.

The calibration campaign must run treatment and control from one *actual*
resident state, not two executions which merely report equal summary hashes.
This module owns the private, deterministic codec and the exact in-worker
capture/restore boundary.  It deliberately contains no trust, storage, IPC, or
campaign policy: those layers wrap this object without learning its contents.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from core.brain.llm.recurrent_depth import (
    _restore_recurrent_caches,
    _snapshot_recurrent_caches,
)

PORTABLE_STATE_SCHEMA: Final = "aura.rlc.portable_resident_state.v1"
ACTION_OPPORTUNITY_CONTINUATION_SCHEMA: Final = (
    "aura.rlc.action_opportunity_continuation.v1"
)
STATE_VALUE_NAMES: Final = (
    "branch_state",
    "durable_state",
    "evidence_state",
    "kv_cache",
    "latent_slots",
    "memory_state",
    "public_action_state",
    "rng_state",
)

_MAGIC = b"AURARLCSTATE\x01"
_MAX_DEPTH = 96
_MAX_ITEMS = 1_000_000
_MAX_TENSOR_BYTES = 512 * 1024 * 1024
_MAX_SCALAR_BYTES = 64 * 1024 * 1024


class ActionContinuationError(ValueError):
    """A portable continuation was malformed, incomplete, or inapplicable."""


def _u64(value: int) -> bytes:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ActionContinuationError("portable_state_length_invalid")
    return struct.pack(">Q", value)


def _read_u64(view: memoryview, offset: int) -> tuple[int, int]:
    end = offset + 8
    if end > len(view):
        raise ActionContinuationError("portable_state_truncated")
    return struct.unpack(">Q", view[offset:end])[0], end


def _bytes_parts(value: bytes, *, chunk_bytes: int) -> Iterator[bytes]:
    for offset in range(0, len(value), chunk_bytes):
        yield value[offset : offset + chunk_bytes]


def _mapping_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, str):
        return (0, value)
    if type(value) is int:
        return (1, value)
    raise ActionContinuationError("portable_state_mapping_key_invalid")


def _tensor_source(value: Any) -> str:
    module = type(value).__module__
    return "mlx" if module == "mlx.core" or module.startswith("mlx.") else "numpy"


def _iter_value(value: Any, *, depth: int, chunk_bytes: int) -> Iterator[bytes]:
    if depth > _MAX_DEPTH:
        raise ActionContinuationError("portable_state_depth_exceeded")
    if value is None:
        yield b"N"
        return
    if type(value) is bool:
        yield b"T" if value else b"F"
        return
    if type(value) is int:
        encoded = str(value).encode("ascii")
        yield b"I" + _u64(len(encoded)) + encoded
        return
    if type(value) is float:
        if math.isnan(value):
            yield b"X"
            return
        if math.isinf(value):
            yield b"P" if value > 0 else b"M"
            return
        yield b"R" + struct.pack(">d", value)
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) > _MAX_SCALAR_BYTES:
            raise ActionContinuationError("portable_state_scalar_too_large")
        yield b"S" + _u64(len(encoded))
        yield from _bytes_parts(encoded, chunk_bytes=chunk_bytes)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        view = memoryview(value)
        if len(view) > _MAX_SCALAR_BYTES:
            raise ActionContinuationError("portable_state_scalar_too_large")
        yield b"B" + _u64(len(view))
        for offset in range(0, len(view), chunk_bytes):
            yield view[offset : offset + chunk_bytes].tobytes()
        return
    if isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype") and not isinstance(value, type)
    ):
        source_name = _tensor_source(value)
        logical_dtype_name = str(value.dtype).removeprefix("mlx.core.")
        if source_name == "mlx" and logical_dtype_name == "bfloat16":
            import mlx.core as mx

            array = np.asarray(value.view(mx.uint16))
        else:
            array = np.asarray(value)
        if array.dtype.hasobject:
            raise ActionContinuationError("portable_state_tensor_object_dtype")
        contiguous = np.ascontiguousarray(array)
        byte_count = int(contiguous.nbytes)
        if byte_count > _MAX_TENSOR_BYTES:
            raise ActionContinuationError("portable_state_tensor_too_large")
        source = source_name.encode("ascii")
        dtype_name = (
            logical_dtype_name if source_name == "mlx" else str(contiguous.dtype.name)
        ).encode("ascii")
        dtype_str = str(contiguous.dtype.str).encode("ascii")
        yield b"A" + _u64(len(source)) + source
        yield _u64(len(dtype_name)) + dtype_name
        yield _u64(len(dtype_str)) + dtype_str
        yield _u64(contiguous.ndim)
        for dimension in contiguous.shape:
            yield _u64(int(dimension))
        yield _u64(byte_count)
        raw = memoryview(contiguous).cast("B")
        for offset in range(0, len(raw), chunk_bytes):
            yield raw[offset : offset + chunk_bytes].tobytes()
        return
    if isinstance(value, Mapping):
        if len(value) > _MAX_ITEMS:
            raise ActionContinuationError("portable_state_item_bound_exceeded")
        items = sorted(value.items(), key=lambda item: _mapping_key(item[0]))
        yield b"D" + _u64(len(items))
        for key, item in items:
            yield from _iter_value(key, depth=depth + 1, chunk_bytes=chunk_bytes)
            yield from _iter_value(item, depth=depth + 1, chunk_bytes=chunk_bytes)
        return
    if isinstance(value, tuple):
        tag = b"U"
    elif isinstance(value, list):
        tag = b"L"
    elif isinstance(value, (set, frozenset)):
        encoded_items = [PortableStateComponent.from_value(item).to_bytes() for item in value]
        encoded_items.sort()
        yield b"E" + _u64(len(encoded_items))
        for encoded in encoded_items:
            yield b"Q" + _u64(len(encoded))
            yield from _bytes_parts(encoded, chunk_bytes=chunk_bytes)
        return
    else:
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, (str, int)):
            yield from _iter_value(enum_value, depth=depth + 1, chunk_bytes=chunk_bytes)
            return
        raise ActionContinuationError(
            "portable_state_type_unsupported:" + type(value).__qualname__
        )
    if len(value) > _MAX_ITEMS:
        raise ActionContinuationError("portable_state_item_bound_exceeded")
    yield tag + _u64(len(value))
    for item in value:
        yield from _iter_value(item, depth=depth + 1, chunk_bytes=chunk_bytes)


class _Decoder:
    def __init__(self, raw: bytes) -> None:
        self.view = memoryview(raw)
        self.offset = 0
        self.items = 0

    def _take(self, count: int) -> bytes:
        end = self.offset + count
        if count < 0 or end > len(self.view):
            raise ActionContinuationError("portable_state_truncated")
        value = self.view[self.offset:end].tobytes()
        self.offset = end
        return value

    def _length(self) -> int:
        value, self.offset = _read_u64(self.view, self.offset)
        return value

    def value(self, *, depth: int = 0) -> Any:
        if depth > _MAX_DEPTH:
            raise ActionContinuationError("portable_state_depth_exceeded")
        self.items += 1
        if self.items > _MAX_ITEMS:
            raise ActionContinuationError("portable_state_item_bound_exceeded")
        tag = self._take(1)
        if tag == b"N":
            return None
        if tag == b"T":
            return True
        if tag == b"F":
            return False
        if tag == b"I":
            raw = self._take(self._length())
            try:
                value = int(raw.decode("ascii"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise ActionContinuationError("portable_state_integer_invalid") from exc
            if str(value).encode("ascii") != raw:
                raise ActionContinuationError("portable_state_integer_noncanonical")
            return value
        if tag == b"R":
            value = struct.unpack(">d", self._take(8))[0]
            return value
        if tag == b"P":
            return math.inf
        if tag == b"M":
            return -math.inf
        if tag == b"X":
            return math.nan
        if tag in {b"S", b"B"}:
            length = self._length()
            if length > _MAX_SCALAR_BYTES:
                raise ActionContinuationError("portable_state_scalar_too_large")
            raw = self._take(length)
            if tag == b"B":
                return raw
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ActionContinuationError("portable_state_string_invalid") from exc
        if tag == b"A":
            source = self._take(self._length()).decode("ascii")
            dtype_name = self._take(self._length()).decode("ascii")
            dtype_str = self._take(self._length()).decode("ascii")
            ndim = self._length()
            if source not in {"mlx", "numpy"} or ndim > 16:
                raise ActionContinuationError("portable_state_tensor_header_invalid")
            shape = tuple(self._length() for _ in range(ndim))
            byte_count = self._length()
            if byte_count > _MAX_TENSOR_BYTES:
                raise ActionContinuationError("portable_state_tensor_too_large")
            if source == "mlx" and dtype_name == "bfloat16":
                storage_dtype = np.dtype("uint16")
            else:
                try:
                    storage_dtype = np.dtype(dtype_name)
                except TypeError:
                    try:
                        storage_dtype = np.dtype(dtype_str)
                    except TypeError as exc:
                        raise ActionContinuationError(
                            "portable_state_tensor_dtype_invalid"
                        ) from exc
            expected = int(storage_dtype.itemsize) * math.prod(shape)
            if expected != byte_count or storage_dtype.hasobject:
                raise ActionContinuationError("portable_state_tensor_size_invalid")
            array = (
                np.frombuffer(self._take(byte_count), dtype=storage_dtype)
                .reshape(shape)
                .copy()
            )
            if source == "mlx":
                try:
                    import mlx.core as mx

                    restored = mx.array(array)
                    if dtype_name == "bfloat16":
                        restored = restored.view(mx.bfloat16)
                    mx.eval(restored)
                    return restored
                except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                    raise ActionContinuationError("portable_state_mlx_restore_failed") from exc
            return array
        if tag in {b"L", b"U"}:
            count = self._length()
            if count > _MAX_ITEMS:
                raise ActionContinuationError("portable_state_item_bound_exceeded")
            values = [self.value(depth=depth + 1) for _ in range(count)]
            return tuple(values) if tag == b"U" else values
        if tag == b"E":
            count = self._length()
            if count > _MAX_ITEMS:
                raise ActionContinuationError("portable_state_item_bound_exceeded")
            values = []
            previous = b""
            for _ in range(count):
                if self._take(1) != b"Q":
                    raise ActionContinuationError("portable_state_set_item_invalid")
                encoded = self._take(self._length())
                if previous and encoded <= previous:
                    raise ActionContinuationError("portable_state_set_noncanonical")
                previous = encoded
                nested = PortableStateComponent.from_bytes(encoded).decode()
                values.append(nested)
            try:
                return set(values)
            except TypeError as exc:
                raise ActionContinuationError("portable_state_set_item_unhashable") from exc
        if tag == b"D":
            count = self._length()
            if count > _MAX_ITEMS:
                raise ActionContinuationError("portable_state_item_bound_exceeded")
            result: dict[Any, Any] = {}
            previous: tuple[int, Any] | None = None
            for _ in range(count):
                key = self.value(depth=depth + 1)
                ordering = _mapping_key(key)
                if previous is not None and ordering <= previous:
                    raise ActionContinuationError("portable_state_mapping_noncanonical")
                previous = ordering
                result[key] = self.value(depth=depth + 1)
            return result
        raise ActionContinuationError("portable_state_tag_invalid")


@dataclass(frozen=True, slots=True)
class PortableStateComponent:
    """One deterministic component, encoded lazily when captured."""

    _value: Any = field(default=None, repr=False)
    _raw: bytes | None = field(default=None, repr=False)

    @classmethod
    def from_value(cls, value: Any) -> PortableStateComponent:
        return cls(_value=value)

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview) -> PortableStateComponent:
        value = bytes(raw)
        if not value.startswith(_MAGIC):
            raise ActionContinuationError("portable_state_magic_invalid")
        return cls(_raw=value)

    def iter_encoded_chunks(self, *, chunk_bytes: int = 1024 * 1024) -> Iterator[bytes]:
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 8 * 1024 * 1024:
            raise ActionContinuationError("portable_state_chunk_bound_invalid")
        if self._raw is not None:
            yield from _bytes_parts(self._raw, chunk_bytes=chunk_bytes)
            return
        yield _MAGIC
        yield from _iter_value(self._value, depth=0, chunk_bytes=chunk_bytes)

    def to_bytes(self) -> bytes:
        return b"".join(self.iter_encoded_chunks())

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.iter_encoded_chunks():
            digest.update(chunk)
        return digest.hexdigest()

    def decode(self) -> Any:
        raw = self._raw if self._raw is not None else self.to_bytes()
        decoder = _Decoder(raw[len(_MAGIC) :])
        value = decoder.value()
        if decoder.offset != len(decoder.view):
            raise ActionContinuationError("portable_state_trailing_bytes")
        return value


def _component(value: Any) -> PortableStateComponent:
    return value if isinstance(value, PortableStateComponent) else PortableStateComponent.from_value(value)


@dataclass(frozen=True, slots=True)
class ActionOpportunityContinuation:
    """Complete private state immediately before action opportunity one."""

    private_state: dict[str, PortableStateComponent] = field(repr=False)
    episode_step: int
    schedule_step: int
    branch_id: str
    layer_index: int
    kv_position: int
    schema: str = ACTION_OPPORTUNITY_CONTINUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ACTION_OPPORTUNITY_CONTINUATION_SCHEMA:
            raise ActionContinuationError("action_continuation_schema_invalid")
        if set(self.private_state) != set(STATE_VALUE_NAMES):
            raise ActionContinuationError("action_continuation_components_invalid")
        if any(not isinstance(value, PortableStateComponent) for value in self.private_state.values()):
            raise ActionContinuationError("action_continuation_component_type_invalid")
        for value in (self.episode_step, self.schedule_step, self.layer_index, self.kv_position):
            if type(value) is not int or value < 0:
                raise ActionContinuationError("action_continuation_position_invalid")
        if not isinstance(self.branch_id, str) or not self.branch_id:
            raise ActionContinuationError("action_continuation_branch_invalid")

    @property
    def state_components(self) -> dict[str, str]:
        return {
            f"{name}_sha256": self.private_state[name].sha256()
            for name in sorted(self.private_state)
        }


def _context_partition(items: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    memory: list[dict] = []
    evidence: list[dict] = []
    for raw in items:
        item = dict(raw)
        source = str(item.get("source") or "")
        if item.get("context_role") == "memory_observation" or source in {
            "memory",
            "one_shot_memory",
        }:
            memory.append(item)
        elif item.get("context_role") == "evidence_observation" or source in {
            "reference",
            "world_model",
        } or source.startswith(("evidence", "tool_observation")):
            evidence.append(item)
    return memory, evidence


def capture_action_opportunity_continuation(
    *,
    ensemble: Any,
    cache: Sequence[Any],
    budget: Any,
    episode_context_items: Sequence[Mapping[str, Any]],
    action_policy_evidence: Mapping[str, Any],
    state_signal: Any,
    active_action_executors: Sequence[Any],
    durable_state: Any,
    rng_state: Any,
    episode_step: int,
    schedule_step: int,
    branch_id: str,
    layer_index: int,
    kv_position: int,
) -> ActionOpportunityContinuation:
    """Capture every state surface before the engine can select an action."""

    runtime = ensemble.snapshot_ensemble_runtime()
    latent_rows: dict[int, dict[str, Any]] = {}
    for branch in ensemble.branches:
        branch_runtime = runtime["branches"][branch.index]
        latent_rows[branch.index] = {
            "z": branch_runtime.pop("z"),
            "last_loop_delta": branch_runtime.pop("last_loop_delta"),
            "anchor": branch.anchor,
            "workspace": branch.workspace.snapshot(),
            "savepoint": branch.savepoint,
            "verified_best_state": branch.verified_best_state,
        }
        # KV lineage node ids are per-episode salted capabilities. The exact
        # cache bytes live in ``kv_cache``; carrying a node capability across
        # workers would both leak private lineage and make equal states encode
        # differently. Restore binds the decoded bytes to the fresh worker's
        # current root node before applying branch runtime.
        branch_runtime["kv_boundary_sha256"] = "resident_first_opportunity_kv"
    branch_state = {
        "schema": PORTABLE_STATE_SCHEMA,
        "runtime": runtime,
        "context_sha256": ensemble._context_sha256,
        "configured_role_lesion": ensemble._configured_role_lesion,
        "seed_alias_free": ensemble._seed_alias_free,
        "seed_states_unique": ensemble._seed_states_unique,
        "rng_streams_unique": ensemble._rng_streams_unique,
        "support_weights": dict(ensemble._support_weights),
        "branches": {
            branch.index: {
                "savepoint_steps": branch.savepoint_steps,
                "savepoint_kv_boundary_sha256": "resident_first_opportunity_kv",
                "seed_sha256": branch.seed_sha256,
                "rng_stream_sha256": branch.rng_stream_sha256,
                "candidate_sha256": branch.candidate_sha256,
                "candidate_step": branch.candidate_step,
                "evidence_anchor_sha256": branch.evidence_anchor_sha256,
                "initial_hypothesis_sha256": branch.initial_hypothesis_sha256,
                "recurrent_grounding_trace": list(branch.recurrent_grounding_trace),
                "loop_stability_trace": list(branch.loop_stability_trace),
                "update_acceptance_trace": list(branch.update_acceptance_trace),
                "uncertainty_trace": list(branch.uncertainty_trace),
                "mistake_locator_trace": list(branch.mistake_locator_trace),
                "reflector_trace": list(branch.reflector_trace),
                "verified_best_step": branch.verified_best_step,
                "verified_best_state_sha256": branch.verified_best_state_sha256,
                "verified_best_observation": dict(branch.verified_best_observation),
                "verified_best_trace": list(branch.verified_best_trace),
                "verified_finalization": dict(branch.verified_finalization),
            }
            for branch in ensemble.branches
        },
        "budget": {
            "max_layer_apps": budget.max_layer_apps,
            "spent_layer_apps": budget.spent_layer_apps,
            "remaining_layer_apps": budget.remaining_layer_apps,
            "resource_accounting": budget.resource_ledger.to_receipt(),
            "information_accounting": budget.information_receipt,
        },
    }
    memory_items, evidence_items = _context_partition(episode_context_items)
    public_action_state = {
        "state_signal": state_signal.to_dict(),
        "active_action_executors": [
            str(getattr(action, "value", action)) for action in active_action_executors
        ],
    }
    private_state = {
        "branch_state": _component(branch_state),
        "durable_state": _component(durable_state),
        "evidence_state": _component(
            {
                "action_policy_evidence": dict(action_policy_evidence),
                "context_items": evidence_items,
            }
        ),
        "kv_cache": _component(_snapshot_recurrent_caches(cache, 0, len(cache))),
        "latent_slots": _component(latent_rows),
        "memory_state": _component(memory_items),
        "public_action_state": _component(public_action_state),
        "rng_state": _component(rng_state),
    }
    return ActionOpportunityContinuation(
        private_state=private_state,
        episode_step=episode_step,
        schedule_step=schedule_step,
        branch_id=branch_id,
        layer_index=layer_index,
        kv_position=kv_position,
    )


def restore_action_opportunity_continuation(
    continuation: ActionOpportunityContinuation,
    *,
    ensemble: Any,
    cache: Sequence[Any],
    budget: Any,
) -> None:
    """Install one continuation into an equivalent fresh first-opportunity frame."""

    if not isinstance(continuation, ActionOpportunityContinuation):
        raise ActionContinuationError("action_continuation_required")
    decoded = {name: value.decode() for name, value in continuation.private_state.items()}
    branch_state = decoded["branch_state"]
    latent_rows = decoded["latent_slots"]
    if not isinstance(branch_state, dict) or branch_state.get("schema") != PORTABLE_STATE_SCHEMA:
        raise ActionContinuationError("action_continuation_branch_state_invalid")
    if set(latent_rows) != {branch.index for branch in ensemble.branches}:
        raise ActionContinuationError("action_continuation_branch_inventory_mismatch")
    captured_budget = branch_state["budget"]
    if (
        captured_budget["max_layer_apps"] != budget.max_layer_apps
        or captured_budget["spent_layer_apps"] != budget.spent_layer_apps
        or captured_budget["remaining_layer_apps"] != budget.remaining_layer_apps
        or captured_budget["resource_accounting"] != budget.resource_ledger.to_receipt()
        or captured_budget["information_accounting"] != budget.information_receipt
    ):
        raise ActionContinuationError("action_continuation_budget_frame_mismatch")

    current_boundaries = {
        branch.index: (branch.kv_boundary_sha256, branch.savepoint_kv_boundary_sha256)
        for branch in ensemble.branches
    }
    _restore_recurrent_caches(cache, 0, len(cache), decoded["kv_cache"])
    runtime = branch_state["runtime"]
    for branch in ensemble.branches:
        index = branch.index
        latent = latent_rows[index]
        runtime_branch = runtime["branches"][index]
        runtime_branch["z"] = latent["z"]
        runtime_branch["last_loop_delta"] = latent["last_loop_delta"]
        runtime_branch["kv_boundary_sha256"] = current_boundaries[index][0]
        branch.savepoint_kv_boundary_sha256 = current_boundaries[index][1]
    ensemble.restore_ensemble_runtime(runtime)
    ensemble._context_sha256 = str(branch_state["context_sha256"])
    ensemble._configured_role_lesion = bool(branch_state["configured_role_lesion"])
    ensemble._seed_alias_free = bool(branch_state["seed_alias_free"])
    ensemble._seed_states_unique = bool(branch_state["seed_states_unique"])
    ensemble._rng_streams_unique = bool(branch_state["rng_streams_unique"])
    ensemble.set_support_weights(branch_state["support_weights"])
    for branch in ensemble.branches:
        index = branch.index
        latent = latent_rows[index]
        metadata = branch_state["branches"][index]
        branch.z = latent["z"]
        branch.anchor = latent["anchor"]
        branch.workspace.restore(latent["workspace"])
        branch.savepoint = latent["savepoint"]
        branch.verified_best_state = latent["verified_best_state"]
        for name in (
            "savepoint_steps",
            "seed_sha256",
            "rng_stream_sha256",
            "candidate_sha256",
            "candidate_step",
            "evidence_anchor_sha256",
            "initial_hypothesis_sha256",
            "verified_best_step",
            "verified_best_state_sha256",
        ):
            setattr(branch, name, metadata[name])
        for name in (
            "recurrent_grounding_trace",
            "loop_stability_trace",
            "update_acceptance_trace",
            "uncertainty_trace",
            "mistake_locator_trace",
            "reflector_trace",
            "verified_best_trace",
        ):
            setattr(branch, name, list(metadata[name]))
        for name in (
            "verified_best_observation",
            "verified_finalization",
        ):
            setattr(branch, name, dict(metadata[name]))


__all__ = [
    "ACTION_OPPORTUNITY_CONTINUATION_SCHEMA",
    "ActionContinuationError",
    "ActionOpportunityContinuation",
    "PortableStateComponent",
    "STATE_VALUE_NAMES",
    "capture_action_opportunity_continuation",
    "restore_action_opportunity_continuation",
]
