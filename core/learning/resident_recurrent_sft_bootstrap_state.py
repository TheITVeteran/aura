"""Crash-consistent resident recurrent-SFT bootstrap checkpoints.

Each optimizer update is published as a new immutable generation. Adapter and
optimizer tensors plus the complete state manifest are durable before the
single ``latest.json`` pointer advances. Resume therefore observes either the
previous complete update or the next complete update, never a partial one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Never, cast

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    SAMPLER_NAME,
    sha256_json,
    validate_authority,
)
from core.runtime.atomic_writer import (
    atomic_write_bytes,
    atomic_write_text,
    durable_unlink,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError

CHECKPOINT_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_checkpoint.v1"
POINTER_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_pointer.v1"
ZERO_SHA256: Final = "0" * 64
MAX_EXAMPLES: Final = 100_000
MAX_TRAIL_ENTRIES: Final = 100_000
MAX_ARTIFACT_BYTES: Final = 1 << 50
MAX_METADATA_BYTES: Final = 16 * 1024 * 1024

BINDING_ROLES: Final = frozenset(
    {
        "authority_sha256",
        "campaign_scope_sha256",
        "artifact_root_identity_sha256",
        "dataset_sha256",
        "model_identity_sha256",
        "behavior_identity_sha256",
        "personality_identity_sha256",
        "tokenizer_identity_sha256",
        "source_closure_sha256",
        "execution_spec_sha256",
        "trainer_config_sha256",
        "runtime_identity_sha256",
        "trust_policy_sha256",
    }
)


class ResidentSFTBootstrapStateError(RuntimeError):
    """A resident bootstrap checkpoint is partial, stale, or inconsistent."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentSFTBootstrapStateError(code)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalized_json(value: Any, *, role: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ResidentSFTBootstrapStateError(f"resident_sft_state_{role}_invalid") from exc


def _nonnegative_float(value: Any, *, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        _fail(f"resident_sft_state_{role}_invalid")
    return float(value)


def _trail(value: Any, *, role: str) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_TRAIL_ENTRIES
        or any(not isinstance(entry, Mapping) for entry in value)
    ):
        _fail(f"resident_sft_state_{role}_invalid")
    normalized = _normalized_json(value, role=role)
    if not isinstance(normalized, list):
        _fail(f"resident_sft_state_{role}_invalid")
    return normalized


def authority_state_bindings(authority: Mapping[str, Any]) -> dict[str, str]:
    """Derive the exact checkpoint protocol bindings from validated authority."""

    validated = validate_authority(authority)
    model = validated["model"]
    bindings = {
        "authority_sha256": validated["authority_sha256"],
        "campaign_scope_sha256": sha256_json(validated["campaign_scope"]),
        "artifact_root_identity_sha256": sha256_json(validated["artifact_root_identity"]),
        "dataset_sha256": validated["dataset"]["dataset_sha256"],
        "model_identity_sha256": model["base_checkpoint"]["fingerprint"],
        "behavior_identity_sha256": model["behavior_bundle"]["bundle_sha256"],
        "personality_identity_sha256": model["personality_bundle"]["identity_sha256"],
        "tokenizer_identity_sha256": validated["tokenizer"]["identity_sha256"],
        "source_closure_sha256": sha256_json(validated["sources"]),
        "execution_spec_sha256": validated["execution_spec"]["semantic_sha256"],
        "trainer_config_sha256": sha256_json(validated["trainer"]),
        "runtime_identity_sha256": validated["runtime"]["identity_sha256"],
        "trust_policy_sha256": validated["trust_policy"]["semantic_sha256"],
    }
    return validate_expected_bindings(bindings)


def validate_expected_bindings(bindings: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(bindings, Mapping) or set(bindings) != BINDING_ROLES:
        _fail("resident_sft_state_expected_bindings_invalid")
    normalized = dict(bindings)
    if any(not _is_sha256(value) for value in normalized.values()):
        _fail("resident_sft_state_expected_bindings_invalid")
    return normalized


def order_sha256(*, order: Sequence[int], seed: int, epoch: int) -> str:
    digest: str = sha256_json(
        {
            "epoch": epoch,
            "order": list(order),
            "sampler": SAMPLER_NAME,
            "seed": seed,
        }
    )
    return digest


def validate_checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    required = set(BINDING_ROLES) | {
        "checkpoint_sequence",
        "step",
        "optimizer_updates",
        "epoch",
        "cursor",
        "order",
        "order_sha256",
        "sampler",
        "seed",
        "train_example_count",
        "validation_example_count",
        "elapsed_training_s",
        "invocation_count",
        "sample_history_sha256",
        "initial_adapter_sha256",
        "adapter_topology_sha256",
        "loss_trail",
        "validation_trail",
        "pending_losses",
        "baseline_validation",
        "last_step_committed",
        "terminal",
        "halt_reason",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        _fail("resident_sft_state_schema_invalid")
    bindings = validate_expected_bindings({role: state[role] for role in BINDING_ROLES})
    for role in (
        "order_sha256",
        "sample_history_sha256",
        "initial_adapter_sha256",
        "adapter_topology_sha256",
    ):
        if not _is_sha256(state.get(role)):
            _fail(f"resident_sft_state_{role}_invalid")

    integers = {
        "checkpoint_sequence": (1, 1_000_000),
        "step": (0, 1_000_000),
        "optimizer_updates": (0, 1_000_000),
        "epoch": (0, 1_000_000),
        "cursor": (0, MAX_EXAMPLES),
        "seed": (0, 2**63 - 1),
        "train_example_count": (1, MAX_EXAMPLES),
        "validation_example_count": (1, MAX_EXAMPLES),
        "invocation_count": (1, 1_000_000),
    }
    for role, (minimum, maximum) in integers.items():
        value = state.get(role)
        if type(value) is not int or not minimum <= value <= maximum:
            _fail(f"resident_sft_state_{role}_invalid")
    if state["optimizer_updates"] != state["step"]:
        _fail("resident_sft_state_optimizer_update_mismatch")
    if state.get("sampler") != SAMPLER_NAME:
        _fail("resident_sft_state_sampler_invalid")

    count = state["train_example_count"]
    order = state.get("order")
    if (
        not isinstance(order, list)
        or len(order) != count
        or any(type(index) is not int for index in order)
        or sorted(order) != list(range(count))
        or state["cursor"] > count
    ):
        _fail("resident_sft_state_order_not_without_replacement")
    if state["step"] != state["epoch"] * count + state["cursor"]:
        _fail("resident_sft_state_sample_position_invalid")
    if state["order_sha256"] != order_sha256(
        order=order,
        seed=state["seed"],
        epoch=state["epoch"],
    ):
        _fail("resident_sft_state_order_digest_mismatch")

    _nonnegative_float(state.get("elapsed_training_s"), role="elapsed_training_s")
    loss_trail = _trail(state.get("loss_trail"), role="loss_trail")
    validation_trail = _trail(
        state.get("validation_trail"),
        role="validation_trail",
    )
    pending = state.get("pending_losses")
    if (
        not isinstance(pending, list)
        or len(pending) > MAX_TRAIL_ENTRIES
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in pending
        )
    ):
        _fail("resident_sft_state_pending_losses_invalid")
    baseline = state.get("baseline_validation")
    if not isinstance(baseline, Mapping) or not baseline:
        _fail("resident_sft_state_baseline_validation_invalid")
    normalized_baseline = _normalized_json(baseline, role="baseline_validation")
    if not isinstance(normalized_baseline, dict):
        _fail("resident_sft_state_baseline_validation_invalid")
    if state.get("last_step_committed") is not True:
        _fail("resident_sft_state_partial_update_forbidden")
    if type(state.get("terminal")) is not bool:
        _fail("resident_sft_state_terminal_invalid")
    halt_reason = state.get("halt_reason")
    if halt_reason is not None and (
        not isinstance(halt_reason, str) or not halt_reason or len(halt_reason) > 160
    ):
        _fail("resident_sft_state_halt_reason_invalid")
    if state["terminal"] != (halt_reason is not None):
        _fail("resident_sft_state_terminal_reason_mismatch")

    normalized = {
        **bindings,
        **{key: state[key] for key in required - BINDING_ROLES},
        "elapsed_training_s": float(state["elapsed_training_s"]),
        "loss_trail": loss_trail,
        "validation_trail": validation_trail,
        "pending_losses": [float(value) for value in pending],
        "baseline_validation": normalized_baseline,
    }
    material = _normalized_json(normalized, role="checkpoint")
    if not isinstance(material, dict):
        _fail("resident_sft_state_checkpoint_invalid")
    return material


def validate_checkpoint_descendant(
    ancestor: Mapping[str, Any],
    descendant: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove that ``descendant`` extends one validated checkpoint history."""

    earlier = validate_checkpoint_state(ancestor)
    later = validate_checkpoint_state(descendant)
    if any(earlier[role] != later[role] for role in BINDING_ROLES):
        _fail("resident_sft_state_descendant_binding_mismatch")
    if (
        later["checkpoint_sequence"] < earlier["checkpoint_sequence"]
        or later["step"] < earlier["step"]
        or later["invocation_count"] < earlier["invocation_count"]
        or later["elapsed_training_s"] < earlier["elapsed_training_s"]
        or (earlier["terminal"] and later != earlier)
    ):
        _fail("resident_sft_state_descendant_nonmonotonic")
    immutable_roles = (
        "sampler",
        "seed",
        "train_example_count",
        "validation_example_count",
        "initial_adapter_sha256",
        "adapter_topology_sha256",
        "baseline_validation",
    )
    if any(earlier[role] != later[role] for role in immutable_roles):
        _fail("resident_sft_state_descendant_identity_mismatch")
    earlier_losses = earlier["loss_trail"]
    later_losses = later["loss_trail"]
    earlier_validation = earlier["validation_trail"]
    later_validation = later["validation_trail"]
    if (
        later_losses[: len(earlier_losses)] != earlier_losses
        or later_validation[: len(earlier_validation)] != earlier_validation
    ):
        _fail("resident_sft_state_descendant_history_rewritten")
    additional_losses = later_losses[len(earlier_losses) :]
    if len(additional_losses) != later["step"] - earlier["step"]:
        _fail("resident_sft_state_descendant_history_incomplete")
    history_sha256 = earlier["sample_history_sha256"]
    expected_step = earlier["step"]
    for record in additional_losses:
        expected_step += 1
        if (
            record.get("step") != expected_step
            or type(record.get("epoch")) is not int
            or type(record.get("cursor")) is not int
            or not _is_sha256(record.get("example_id"))
        ):
            _fail("resident_sft_state_descendant_history_invalid")
        history_sha256 = sha256_json(
            {
                "previous_sha256": history_sha256,
                "example_id": record["example_id"],
                "step": record["step"],
                "epoch": record["epoch"],
                "cursor": record["cursor"],
            }
        )
    if (
        history_sha256 != later["sample_history_sha256"]
        or (
            additional_losses
            and (
                additional_losses[-1]["epoch"] != later["epoch"]
                or additional_losses[-1]["cursor"] != later["cursor"]
            )
        )
    ):
        _fail("resident_sft_state_descendant_history_mismatch")
    return earlier, later


@dataclass(frozen=True, slots=True)
class InspectedResidentSFTCheckpoint:
    checkpoint_dir: Path
    complete_sha256: str
    state: dict[str, Any]
    adapter_binding: dict[str, Any]
    optimizer_binding: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedResidentSFTCheckpoint:
    checkpoint_dir: Path
    complete_sha256: str
    state: dict[str, Any]
    adapter_tensors: dict[str, Any]
    optimizer_tensors: dict[str, Any]


def _root(
    path: Path,
    *,
    create: bool,
    custody: DirectoryCustody | None = None,
) -> Path:
    if custody is not None:
        custody.verify()
        if path.expanduser().absolute() != custody.path:
            _fail("resident_sft_state_root_custody_mismatch")
        return custody.path
    requested = path.expanduser()
    if requested.is_symlink():
        _fail("resident_sft_state_root_symlink_forbidden")
    if create:
        ensure_private_directory(requested)
    try:
        resolved = requested.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ResidentSFTBootstrapStateError("resident_sft_state_root_unavailable") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        _fail("resident_sft_state_root_invalid")
    return resolved


def _read_bytes(
    path: Path,
    *,
    role: str,
    max_bytes: int = MAX_ARTIFACT_BYTES,
    custody: DirectoryCustody | None = None,
) -> bytes:
    try:
        if custody is not None:
            try:
                relative = path.expanduser().absolute().relative_to(custody.path).as_posix()
            except ValueError:
                _fail(f"resident_sft_state_{role}_outside_custody")
            payload = custody.read_bytes(relative, max_bytes=max_bytes)
            if not payload:
                _fail(f"resident_sft_state_{role}_size_invalid")
            return payload
        if path.is_symlink() or not path.is_file():
            _fail(f"resident_sft_state_{role}_file_invalid")
        size = path.stat().st_size
        if not 1 <= size <= max_bytes:
            _fail(f"resident_sft_state_{role}_size_invalid")
        payload = path.read_bytes()
    except ResidentSFTBootstrapStateError:
        raise
    except (OSError, FileNotFoundError, SecurePathCustodyError) as exc:
        raise ResidentSFTBootstrapStateError(f"resident_sft_state_{role}_unreadable") from exc
    if len(payload) != size:
        _fail(f"resident_sft_state_{role}_size_invalid")
    return payload


def _read_json(
    path: Path,
    *,
    role: str,
    custody: DirectoryCustody | None = None,
) -> dict[str, Any]:
    payload = _read_bytes(
        path,
        role=role,
        max_bytes=MAX_METADATA_BYTES,
        custody=custody,
    )
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResidentSFTBootstrapStateError(f"resident_sft_state_{role}_json_invalid") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        _fail(f"resident_sft_state_{role}_noncanonical")
    return value


def _write_safetensors(
    path: Path,
    tensors: Mapping[str, Any],
    *,
    role: str,
    custody: DirectoryCustody | None = None,
) -> bytes:
    if (
        not isinstance(tensors, Mapping)
        or not tensors
        or any(not isinstance(name, str) or not name for name in tensors)
    ):
        _fail(f"resident_sft_state_{role}_tensor_mapping_invalid")
    import mlx.core as mx

    scratch_parent: tempfile.TemporaryDirectory[str] | None = None
    if custody is not None:
        scratch_parent = tempfile.TemporaryDirectory(prefix="aura-resident-sft-")
        scratch = Path(scratch_parent.name) / f"{role}.safetensors"
    else:
        scratch = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.safetensors"
    try:
        mx.save_safetensors(str(scratch), dict(tensors))
        payload = _read_bytes(scratch, role=f"{role}_scratch")
    finally:
        if scratch_parent is None:
            durable_unlink(scratch, missing_ok=True)
        else:
            scratch_parent.cleanup()
    if custody is None:
        atomic_write_bytes(path, payload, mode=0o600)
    else:
        try:
            relative = path.expanduser().absolute().relative_to(custody.path).as_posix()
            custody.atomic_write_bytes(relative, payload, mode=0o600)
        except (ValueError, SecurePathCustodyError) as exc:
            raise ResidentSFTBootstrapStateError(
                f"resident_sft_state_{role}_custodied_write_failed"
            ) from exc
    return payload


def _artifact_binding(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _contained_generation(
    root: Path,
    value: Any,
    *,
    custody: DirectoryCustody | None = None,
) -> Path:
    if (
        not isinstance(value, str)
        or not value.startswith("checkpoints/")
        or Path(value).parts != ("checkpoints", Path(value).name)
    ):
        _fail("resident_sft_state_checkpoint_path_invalid")
    checkpoint_root_path = root / "checkpoints"
    generation_path = root / value
    if custody is not None:
        try:
            descriptor = custody.open_directory(value)
        except SecurePathCustodyError as exc:
            raise ResidentSFTBootstrapStateError(
                "resident_sft_state_checkpoint_path_escape"
            ) from exc
        os.close(descriptor)
        return generation_path
    if checkpoint_root_path.is_symlink() or generation_path.is_symlink():
        _fail("resident_sft_state_checkpoint_path_symlink_forbidden")
    checkpoint_root = checkpoint_root_path.resolve(strict=True)
    generation = generation_path.resolve(strict=True)
    if generation.parent != checkpoint_root or generation.is_symlink() or not generation.is_dir():
        _fail("resident_sft_state_checkpoint_path_escape")
    return generation


def _validate_binding(
    generation: Path,
    complete: Mapping[str, Any],
    *,
    role: str,
    custody: DirectoryCustody | None = None,
) -> tuple[dict[str, Any], bytes]:
    binding = complete.get(role)
    expected_name = f"{role}.safetensors"
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256", "size_bytes"}
        or binding.get("path") != expected_name
        or not _is_sha256(binding.get("sha256"))
        or type(binding.get("size_bytes")) is not int
        or not 1 <= binding["size_bytes"] <= MAX_ARTIFACT_BYTES
    ):
        _fail(f"resident_sft_state_{role}_binding_invalid")
    payload = _read_bytes(
        generation / expected_name,
        role=role,
        custody=custody,
    )
    if len(payload) != binding["size_bytes"] or sha256_bytes(payload) != binding["sha256"]:
        _fail(f"resident_sft_state_{role}_commitment_mismatch")
    return dict(binding), payload


@contextmanager
def _checkpoint_lock(
    root: Path,
    custody: DirectoryCustody | None,
) -> Iterator[None]:
    if custody is not None:
        try:
            with custody.file_lock(".checkpoint.lock"):
                yield
            return
        except SecurePathCustodyError as exc:
            raise ResidentSFTBootstrapStateError(
                "resident_sft_state_checkpoint_lock_failed"
            ) from exc
    with interprocess_file_lock(root / ".checkpoint.lock"):
        yield


def _inspect_generation(
    generation: Path,
    *,
    expected: Mapping[str, str],
    custody: DirectoryCustody | None,
    expected_complete_sha256: str | None = None,
    expected_sequence: int | None = None,
) -> InspectedResidentSFTCheckpoint:
    complete_payload = _read_bytes(
        generation / "complete.json",
        role="complete",
        max_bytes=MAX_METADATA_BYTES,
        custody=custody,
    )
    complete_sha256 = sha256_bytes(complete_payload)
    if expected_complete_sha256 is not None and complete_sha256 != expected_complete_sha256:
        _fail("resident_sft_state_complete_commitment_mismatch")
    complete = _read_json(
        generation / "complete.json",
        role="complete",
        custody=custody,
    )
    if set(complete) != {
        "schema",
        "checkpoint_id",
        "created_at_unix",
        "state",
        "adapter",
        "optimizer",
    }:
        _fail("resident_sft_state_complete_schema_invalid")
    created = complete.get("created_at_unix")
    if (
        complete.get("schema") != CHECKPOINT_SCHEMA
        or complete.get("checkpoint_id") != generation.name
        or isinstance(created, bool)
        or not isinstance(created, (int, float))
        or not math.isfinite(float(created))
        or float(created) <= 0.0
    ):
        _fail("resident_sft_state_complete_invalid")
    raw_state = complete.get("state")
    if not isinstance(raw_state, Mapping):
        _fail("resident_sft_state_complete_state_invalid")
    state = validate_checkpoint_state(raw_state)
    if expected_sequence is not None and state["checkpoint_sequence"] != expected_sequence:
        _fail("resident_sft_state_sequence_mismatch")
    if any(state[role] != value for role, value in expected.items()):
        _fail("resident_sft_state_protocol_binding_mismatch")
    adapter_binding, _adapter = _validate_binding(
        generation,
        complete,
        role="adapter",
        custody=custody,
    )
    optimizer_binding, _optimizer = _validate_binding(
        generation,
        complete,
        role="optimizer",
        custody=custody,
    )
    return InspectedResidentSFTCheckpoint(
        checkpoint_dir=generation,
        complete_sha256=complete_sha256,
        state=state,
        adapter_binding=adapter_binding,
        optimizer_binding=optimizer_binding,
    )


def inspect_checkpoint(
    out_dir: Path,
    *,
    expected_bindings: Mapping[str, Any],
    custody: DirectoryCustody | None = None,
    _lock: bool = True,
) -> InspectedResidentSFTCheckpoint:
    expected = validate_expected_bindings(expected_bindings)
    root = _root(out_dir, create=False, custody=custody)
    lock = _checkpoint_lock(root, custody) if _lock else nullcontext()
    with lock:
        pointer = _read_json(root / "latest.json", role="pointer", custody=custody)
        if (
            set(pointer) != {"schema", "checkpoint", "checkpoint_sequence", "complete_sha256"}
            or pointer.get("schema") != POINTER_SCHEMA
            or type(pointer.get("checkpoint_sequence")) is not int
            or pointer["checkpoint_sequence"] < 1
            or not _is_sha256(pointer.get("complete_sha256"))
        ):
            _fail("resident_sft_state_pointer_invalid")
        generation = _contained_generation(root, pointer["checkpoint"], custody=custody)
        return _inspect_generation(
            generation,
            expected=expected,
            custody=custody,
            expected_complete_sha256=pointer["complete_sha256"],
            expected_sequence=pointer["checkpoint_sequence"],
        )


def inspect_checkpoint_generation(
    out_dir: Path,
    *,
    checkpoint: str,
    expected_bindings: Mapping[str, Any],
    custody: DirectoryCustody | None = None,
    _lock: bool = True,
) -> InspectedResidentSFTCheckpoint:
    """Verify one immutable generation without consulting ``latest.json``."""

    expected = validate_expected_bindings(expected_bindings)
    root = _root(out_dir, create=False, custody=custody)
    lock = _checkpoint_lock(root, custody) if _lock else nullcontext()
    with lock:
        generation = _contained_generation(root, checkpoint, custody=custody)
        return _inspect_generation(
            generation,
            expected=expected,
            custody=custody,
        )


def _validate_transition(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> None:
    if previous is None:
        if current["checkpoint_sequence"] != 1 or current["step"] not in {0, 1}:
            _fail("resident_sft_state_initial_transition_invalid")
        return
    if current["checkpoint_sequence"] != previous["checkpoint_sequence"] + 1:
        _fail("resident_sft_state_checkpoint_sequence_invalid")
    delta = current["step"] - previous["step"]
    if delta == 1:
        if previous["terminal"]:
            _fail("resident_sft_state_terminal_resume_forbidden")
        return
    if delta == 0 and current["terminal"] and not previous["terminal"]:
        return
    _fail("resident_sft_state_nonmonotonic_transition")


def save_checkpoint(
    out_dir: Path,
    *,
    adapter_tensors: Mapping[str, Any],
    optimizer_tensors: Mapping[str, Any],
    state: Mapping[str, Any],
    custody: DirectoryCustody | None = None,
) -> Path:
    """Durably publish one complete optimizer update and advance ``latest``."""

    validated = validate_checkpoint_state(state)
    root = _root(out_dir, create=True, custody=custody)
    checkpoint_path = root / "checkpoints"
    if custody is not None:
        try:
            checkpoints = custody.ensure_directory("checkpoints")
        except SecurePathCustodyError as exc:
            raise ResidentSFTBootstrapStateError(
                "resident_sft_state_checkpoint_root_symlink_forbidden"
            ) from exc
    else:
        if checkpoint_path.is_symlink():
            _fail("resident_sft_state_checkpoint_root_symlink_forbidden")
        checkpoints = ensure_private_directory(checkpoint_path).resolve(strict=True)
    with _checkpoint_lock(root, custody):
        previous: dict[str, Any] | None = None
        previous_inspected: InspectedResidentSFTCheckpoint | None = None
        latest_exists = (
            custody.file_exists("latest.json")
            if custody is not None
            else (root / "latest.json").exists()
        )
        if latest_exists:
            previous_inspected = inspect_checkpoint(
                root,
                expected_bindings={role: validated[role] for role in BINDING_ROLES},
                custody=custody,
                _lock=False,
            )
            previous = previous_inspected.state
        _validate_transition(previous, validated)
        checkpoint_id = (
            f"sequence-{validated['checkpoint_sequence']:08d}-"
            f"step-{validated['step']:08d}-{uuid.uuid4().hex}"
        )
        generation = (
            custody.ensure_directory(f"checkpoints/{checkpoint_id}")
            if custody is not None
            else ensure_private_directory(checkpoints / checkpoint_id)
        )
        adapter_payload = _write_safetensors(
            generation / "adapter.safetensors",
            adapter_tensors,
            role="adapter",
            custody=custody,
        )
        optimizer_payload = _write_safetensors(
            generation / "optimizer.safetensors",
            optimizer_tensors,
            role="optimizer",
            custody=custody,
        )
        if previous is not None and validated["step"] == previous["step"]:
            if (
                previous_inspected is None
                or sha256_bytes(adapter_payload) != previous_inspected.adapter_binding["sha256"]
                or sha256_bytes(optimizer_payload) != previous_inspected.optimizer_binding["sha256"]
            ):
                _fail("resident_sft_state_terminal_tensor_drift")
        complete = {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "created_at_unix": time.time(),
            "state": validated,
            "adapter": _artifact_binding("adapter.safetensors", adapter_payload),
            "optimizer": _artifact_binding("optimizer.safetensors", optimizer_payload),
        }
        complete_payload = canonical_json_bytes(complete)
        if custody is None:
            atomic_write_bytes(generation / "complete.json", complete_payload, mode=0o600)
        else:
            custody.atomic_write_bytes(
                f"checkpoints/{checkpoint_id}/complete.json",
                complete_payload,
                mode=0o600,
            )
        pointer = {
            "schema": POINTER_SCHEMA,
            "checkpoint": f"checkpoints/{checkpoint_id}",
            "checkpoint_sequence": validated["checkpoint_sequence"],
            "complete_sha256": sha256_bytes(complete_payload),
        }
        pointer_payload = canonical_json_bytes(pointer)
        if custody is None:
            atomic_write_text(
                root / "latest.json",
                pointer_payload.decode("ascii"),
                encoding="ascii",
                mode=0o600,
            )
        else:
            custody.atomic_write_bytes("latest.json", pointer_payload, mode=0o600)
        return cast(Path, generation)


def load_checkpoint(
    out_dir: Path,
    *,
    expected_bindings: Mapping[str, Any],
    custody: DirectoryCustody | None = None,
) -> LoadedResidentSFTCheckpoint:
    inspected = inspect_checkpoint(
        out_dir,
        expected_bindings=expected_bindings,
        custody=custody,
    )
    import mlx.core as mx

    if custody is None:
        adapter = mx.load(str(inspected.checkpoint_dir / "adapter.safetensors"))
        optimizer = mx.load(str(inspected.checkpoint_dir / "optimizer.safetensors"))
    else:
        checkpoint_relative = inspected.checkpoint_dir.relative_to(custody.path)
        adapter_payload = custody.read_bytes(
            checkpoint_relative / "adapter.safetensors",
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        optimizer_payload = custody.read_bytes(
            checkpoint_relative / "optimizer.safetensors",
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        with tempfile.TemporaryDirectory(prefix="aura-resident-sft-load-") as temporary:
            adapter_path = Path(temporary) / "adapter.safetensors"
            optimizer_path = Path(temporary) / "optimizer.safetensors"
            atomic_write_bytes(adapter_path, adapter_payload, durable=False)
            atomic_write_bytes(optimizer_path, optimizer_payload, durable=False)
            adapter = mx.load(str(adapter_path))
            optimizer = mx.load(str(optimizer_path))
    if not isinstance(adapter, dict) or not adapter:
        _fail("resident_sft_state_adapter_tensor_container_invalid")
    if not isinstance(optimizer, dict) or not optimizer:
        _fail("resident_sft_state_optimizer_tensor_container_invalid")
    return LoadedResidentSFTCheckpoint(
        checkpoint_dir=inspected.checkpoint_dir,
        complete_sha256=inspected.complete_sha256,
        state=inspected.state,
        adapter_tensors=dict(adapter),
        optimizer_tensors=dict(optimizer),
    )


__all__ = [
    "BINDING_ROLES",
    "CHECKPOINT_SCHEMA",
    "InspectedResidentSFTCheckpoint",
    "LoadedResidentSFTCheckpoint",
    "POINTER_SCHEMA",
    "ResidentSFTBootstrapStateError",
    "ZERO_SHA256",
    "authority_state_bindings",
    "inspect_checkpoint",
    "inspect_checkpoint_generation",
    "load_checkpoint",
    "order_sha256",
    "save_checkpoint",
    "sha256_bytes",
    "validate_checkpoint_descendant",
    "validate_checkpoint_state",
    "validate_expected_bindings",
]
