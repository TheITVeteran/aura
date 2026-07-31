"""Immutable pre-measurement state lineage for verified recurrent updates.

The initial CP420Q adapter/optimizer snapshots and each successful transaction's
post-update snapshots form the only tensor states in this chain.  Every
intervention objective first publishes a sealed intent that references one of
those states and proves the live tensors are byte-equivalent at array level.
Rejected groups do not publish intents or advance the successful-update
ordinal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never, cast

from core.learning.recurrent_grpo import (
    VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V2,
    VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3,
    RecurrentGRPOConfig,
    recurrent_policy_tensor_map_sha256,
    validate_verified_trajectory_group_source_binding,
)
from core.learning.verified_transition_policy_probe import (
    inspect_initial_adapter_snapshot,
    inspect_initial_optimizer_snapshot,
    validate_initial_policy_state_custody,
)
from core.learning.verified_transition_update import (
    VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2,
)
from core.runtime.atomic_writer import (
    atomic_write_bytes_if_absent,
    durable_replace,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.file_read_gateway import (
    open_stable_readonly_binary,
    read_stable_bytes,
)

PRE_MEASUREMENT_INTENT_SCHEMA = "aura.verified_transition.pre_measurement_intent.v1"
PRE_MEASUREMENT_GENERATION_SCHEMA = "aura.verified_transition.pre_measurement_generation.v1"
PRE_MEASUREMENT_RECONCILIATION_SCHEMA = "aura.verified_transition.pre_measurement_reconciliation.v1"
PRE_MEASUREMENT_ORIGIN_SCHEMA = "aura.verified_transition.pre_measurement_origin.v1"
PRE_MEASUREMENT_STATE_SOURCE_SCHEMA = "aura.verified_transition.pre_measurement_state_source.v1"
RECURRENT_GRPO_CONFIG_CONTRACT_SCHEMA = "aura.recurrent_grpo_config.exact_float.v1"
BRIDGE_TOKEN_BINDING_SCHEMA = "aura.verified_transition.bridge_token_binding.v1"

_INTENT_DIRECTORY = "00000000-intent"
_ABANDONED_DIRECTORY = "00000001-abandoned"
_INTENT_FILES = frozenset({"pre-measurement.json", "generation.json"})
_ABANDONED_FILES = frozenset({"reconciliation.json", "generation.json"})
_ENTRY_RE = re.compile(r"^seq-(?P<sequence>[0-9]{8})-(?P<admission>[0-9a-f]{64})$")
_TEMP_RE = re.compile(r"^\.tmp-(?:00000000-intent|00000001-abandoned)-[0-9a-f]{32}$")
_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024


class VerifiedTransitionMeasurementChainError(RuntimeError):
    """The pre-measurement chain is incomplete, unsafe, or inconsistent."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTransitionMeasurementChainError(code)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise VerifiedTransitionMeasurementChainError(
            "pre_measurement_document_not_canonicalizable"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed["receipt_sha256"] = _digest(sealed)
    return sealed


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"pre_measurement_{role}_invalid")
    return value


def _integer(value: Any, *, role: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > (1 << 63) - 1:
        _fail(f"pre_measurement_{role}_invalid")
    return value


def _clone(value: Any, *, role: str) -> Any:
    try:
        return json.loads(_canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        _fail(f"pre_measurement_{role}_not_canonical_json")


def _validate_seal(value: Mapping[str, Any], *, role: str) -> None:
    receipt = _sha256(value.get("receipt_sha256"), role=f"{role}_receipt")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if receipt != _digest(unsigned):
        _fail(f"pre_measurement_{role}_digest_mismatch")


def _private_metadata(
    path: Path,
    *,
    directory: bool,
    role: str,
) -> os.stat_result:
    if path.is_symlink():
        _fail(f"pre_measurement_{role}_symlink_rejected")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise VerifiedTransitionMeasurementChainError(f"pre_measurement_{role}_unreadable") from exc
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or (not directory and metadata.st_nlink != 1)
    ):
        _fail(f"pre_measurement_{role}_not_private_owned_{'directory' if directory else 'file'}")
    return metadata


def _ensure_root(path: str | Path) -> Path:
    lexical = Path(path).expanduser().absolute()
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            _fail("pre_measurement_root_symlink_component_rejected")
    root = ensure_private_directory(lexical)
    _private_metadata(root, directory=True, role="root")
    return root.resolve(strict=True)


def _ensure_child(parent: Path, name: str, *, role: str) -> Path:
    child = parent / name
    if os.path.lexists(child):
        if child.is_symlink():
            _fail(f"pre_measurement_{role}_symlink_rejected")
    else:
        child.mkdir(mode=0o700)
        _fsync_directory(parent)
    os.chmod(child, 0o700)
    _private_metadata(child, directory=True, role=role)
    return child


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _read_document(path: Path, *, role: str) -> dict[str, Any]:
    metadata = _private_metadata(path, directory=False, role=role)
    if stat.S_IMODE(metadata.st_mode) & 0o222:
        _fail(f"pre_measurement_{role}_is_writable")
    try:
        payload = read_stable_bytes(path, max_bytes=_MAX_DOCUMENT_BYTES)
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedTransitionMeasurementChainError(
            f"pre_measurement_{role}_json_invalid"
        ) from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        _fail(f"pre_measurement_{role}_json_noncanonical")
    return value


def _tensor_maps_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if set(left) != set(right) or not left:
        return False
    try:
        import mlx.core as mx

        for key in sorted(left):
            if tuple(left[key].shape) != tuple(right[key].shape) or str(left[key].dtype) != str(
                right[key].dtype
            ):
                return False
        comparisons = [mx.array_equal(left[key], right[key]) for key in sorted(left)]
        mx.eval(*comparisons)
        return all(bool(value) for value in comparisons)
    except Exception:
        return False


def _load_bound_safetensors(
    path: Path,
    binding: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Load exactly the private inode whose bytes satisfy the custody binding."""

    _private_metadata(path, directory=False, role=role)
    size = _integer(binding.get("size_bytes"), role=f"{role}_size", minimum=1)
    expected_digest = _sha256(
        binding.get("sha256"),
        role=f"{role}_artifact",
    )
    expected_count = _integer(
        binding.get("tensor_count"),
        role=f"{role}_tensor_count",
        minimum=1,
    )
    expected_keys_digest = _sha256(
        binding.get("tensor_keys_sha256"),
        role=f"{role}_tensor_keys",
    )
    try:
        import mlx.core as mx

        with open_stable_readonly_binary(path, max_bytes=size) as (
            handle,
            identity,
        ):
            payload = handle.read(size + 1)
            if len(payload) != size or len(payload) != identity.size:
                _fail(f"pre_measurement_{role}_read_length_mismatch")
            if hashlib.sha256(payload).hexdigest() != expected_digest:
                _fail(f"pre_measurement_{role}_digest_mismatch")
            handle.seek(0)
            tensors = mx.load(handle, format="safetensors")
            if not isinstance(tensors, Mapping) or not tensors:
                _fail(f"pre_measurement_{role}_tensor_map_invalid")
            mx.eval(*tensors.values())
    except VerifiedTransitionMeasurementChainError:
        raise
    except Exception as exc:
        raise VerifiedTransitionMeasurementChainError(
            f"pre_measurement_{role}_load_failed"
        ) from exc
    keys = sorted(tensors)
    if len(keys) != expected_count or _digest(keys) != expected_keys_digest:
        _fail(f"pre_measurement_{role}_tensor_inventory_mismatch")
    return dict(tensors)


def recurrent_grpo_config_contract(
    config: RecurrentGRPOConfig,
) -> dict[str, Any]:
    """Serialize every optimizer-objective threshold without float rounding."""

    if not isinstance(config, RecurrentGRPOConfig):
        _fail("pre_measurement_recurrent_grpo_config_invalid")
    return {
        "schema": RECURRENT_GRPO_CONFIG_CONTRACT_SCHEMA,
        "clip_epsilon_hex": float(config.clip_epsilon).hex(),
        "kl_coefficient_hex": float(config.kl_coefficient).hex(),
        "advantage_clip_hex": float(config.advantage_clip).hex(),
        "max_initial_clip_fraction_hex": (float(config.max_initial_clip_fraction).hex()),
        "max_initial_old_policy_approx_kl_hex": (
            float(config.max_initial_old_policy_approx_kl).hex()
        ),
    }


def validate_recurrent_grpo_config_contract(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "clip_epsilon_hex",
        "kl_coefficient_hex",
        "advantage_clip_hex",
        "max_initial_clip_fraction_hex",
        "max_initial_old_policy_approx_kl_hex",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail("pre_measurement_recurrent_grpo_config_schema_invalid")
    document = cast(dict[str, Any], _clone(value, role="recurrent_grpo_config"))
    if document.get("schema") != RECURRENT_GRPO_CONFIG_CONTRACT_SCHEMA:
        _fail("pre_measurement_recurrent_grpo_config_schema_invalid")
    try:
        config = RecurrentGRPOConfig(
            clip_epsilon=float.fromhex(document["clip_epsilon_hex"]),
            kl_coefficient=float.fromhex(document["kl_coefficient_hex"]),
            advantage_clip=float.fromhex(document["advantage_clip_hex"]),
            max_initial_clip_fraction=float.fromhex(document["max_initial_clip_fraction_hex"]),
            max_initial_old_policy_approx_kl=float.fromhex(
                document["max_initial_old_policy_approx_kl_hex"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise VerifiedTransitionMeasurementChainError(
            "pre_measurement_recurrent_grpo_config_invalid"
        ) from exc
    if recurrent_grpo_config_contract(config) != document:
        _fail("pre_measurement_recurrent_grpo_config_reconstruction_mismatch")
    return document


def recurrent_grpo_config_from_contract(value: Any) -> RecurrentGRPOConfig:
    """Reconstruct the exact objective configuration from sealed hex values."""

    document = validate_recurrent_grpo_config_contract(value)
    return RecurrentGRPOConfig(
        clip_epsilon=float.fromhex(document["clip_epsilon_hex"]),
        kl_coefficient=float.fromhex(document["kl_coefficient_hex"]),
        advantage_clip=float.fromhex(document["advantage_clip_hex"]),
        max_initial_clip_fraction=float.fromhex(document["max_initial_clip_fraction_hex"]),
        max_initial_old_policy_approx_kl=float.fromhex(
            document["max_initial_old_policy_approx_kl_hex"]
        ),
    )


def bridge_token_binding(tokens: Sequence[int]) -> dict[str, Any]:
    if isinstance(tokens, (str, bytes, bytearray)) or any(
        type(token) is not int or token < 0 for token in tokens
    ):
        _fail("pre_measurement_bridge_tokens_invalid")
    normalized = list(tokens)
    if normalized:
        _fail("pre_measurement_intervention_bridge_tokens_nonempty")
    return {
        "schema": BRIDGE_TOKEN_BINDING_SCHEMA,
        "tokens": normalized,
        "tokens_sha256": _digest(normalized),
    }


def _validate_bridge_token_binding(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "tokens", "tokens_sha256"}
        or value.get("schema") != BRIDGE_TOKEN_BINDING_SCHEMA
        or value.get("tokens") != []
        or value.get("tokens_sha256") != _digest([])
    ):
        _fail("pre_measurement_bridge_token_binding_invalid")
    return cast(dict[str, Any], _clone(value, role="bridge_token_binding"))


def _normalized_artifact_binding(
    binding: Mapping[str, Any],
    *,
    path: Path,
    role: str,
) -> dict[str, Any]:
    required = {
        "path",
        "sha256",
        "size_bytes",
        "tensor_count",
        "tensor_keys_sha256",
    }
    if not isinstance(binding, Mapping):
        _fail(f"pre_measurement_{role}_binding_invalid")
    normalized = {
        "path": str(path.resolve(strict=True)),
        "sha256": binding.get("sha256"),
        "size_bytes": binding.get("size_bytes"),
        "tensor_count": binding.get("tensor_count"),
        "tensor_keys_sha256": binding.get("tensor_keys_sha256"),
    }
    if (
        set(normalized) != required
        or not Path(normalized["path"]).is_absolute()
        or type(normalized["size_bytes"]) is not int
        or normalized["size_bytes"] <= 0
        or type(normalized["tensor_count"]) is not int
        or normalized["tensor_count"] <= 0
    ):
        _fail(f"pre_measurement_{role}_binding_invalid")
    _sha256(normalized["sha256"], role=f"{role}_artifact")
    _sha256(normalized["tensor_keys_sha256"], role=f"{role}_tensor_keys")
    return normalized


def _validate_artifact_binding(value: Any, *, role: str) -> dict[str, Any]:
    required = {
        "path",
        "sha256",
        "size_bytes",
        "tensor_count",
        "tensor_keys_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail(f"pre_measurement_{role}_binding_invalid")
    document = cast(dict[str, Any], _clone(value, role=f"{role}_binding"))
    raw_path = document.get("path")
    if (
        not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or Path(raw_path).resolve(strict=False) != Path(raw_path)
        or type(document.get("size_bytes")) is not int
        or document["size_bytes"] <= 0
        or type(document.get("tensor_count")) is not int
        or document["tensor_count"] <= 0
    ):
        _fail(f"pre_measurement_{role}_binding_invalid")
    _sha256(document.get("sha256"), role=f"{role}_artifact")
    _sha256(
        document.get("tensor_keys_sha256"),
        role=f"{role}_tensor_keys",
    )
    return document


def validate_pre_measurement_state_source(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "kind",
        "successful_update_ordinal",
        "source_sequence",
        "source_admission_sha256",
        "source_receipt_sha256",
        "policy_sha256",
        "adapter_artifact",
        "optimizer_artifact",
        "state_source_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail("pre_measurement_state_source_schema_invalid")
    document = cast(dict[str, Any], _clone(value, role="state_source"))
    unsigned = dict(document)
    observed = unsigned.pop("state_source_sha256")
    kind = document.get("kind")
    ordinal = _integer(
        document.get("successful_update_ordinal"),
        role="successful_update_ordinal",
        minimum=1,
    )
    if (
        document.get("schema") != PRE_MEASUREMENT_STATE_SOURCE_SCHEMA
        or observed != _digest(unsigned)
        or kind not in {"initial_policy_state", "prior_transaction_post_state"}
    ):
        _fail("pre_measurement_state_source_invalid")
    _sha256(observed, role="state_source")
    _sha256(document.get("source_receipt_sha256"), role="state_source_receipt")
    _sha256(document.get("policy_sha256"), role="state_source_policy")
    _validate_artifact_binding(
        document.get("adapter_artifact"),
        role="state_source_adapter",
    )
    _validate_artifact_binding(
        document.get("optimizer_artifact"),
        role="state_source_optimizer",
    )
    source_sequence = document.get("source_sequence")
    source_admission = document.get("source_admission_sha256")
    if kind == "initial_policy_state":
        if ordinal != 1 or source_sequence is not None or source_admission is not None:
            _fail("pre_measurement_initial_state_source_invalid")
    else:
        _integer(
            source_sequence,
            role="state_source_sequence",
        )
        _sha256(source_admission, role="state_source_admission")
        if ordinal < 2:
            _fail("pre_measurement_prior_state_source_ordinal_invalid")
    return document


def validate_pre_measurement_intent(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "sequence",
        "trainer_step",
        "group_admission_sha256",
        "reservation_sha256",
        "policy_before_sha256",
        "provider_contract_sha256",
        "training_protocol_sha256",
        "campaign_manifest_sha256",
        "campaign_schedule_root_sha256",
        "group_manifest_sha256",
        "execution_spec_sha256",
        "trainer_step_static_sha256",
        "trajectory_source_binding",
        "recurrent_grpo_config",
        "bridge_token_binding",
        "objective_input_sha256",
        "state_source",
        "recorded_at_unix_ns",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail("pre_measurement_intent_schema_invalid")
    document = cast(dict[str, Any], _clone(value, role="intent"))
    if document.get("schema") != PRE_MEASUREMENT_INTENT_SCHEMA:
        _fail("pre_measurement_intent_schema_invalid")
    _validate_seal(document, role="intent")
    sequence = _integer(document.get("sequence"), role="sequence")
    if document.get("trainer_step") != sequence + 1:
        _fail("pre_measurement_trainer_step_sequence_mismatch")
    for role in (
        "group_admission_sha256",
        "reservation_sha256",
        "policy_before_sha256",
        "provider_contract_sha256",
        "training_protocol_sha256",
        "campaign_manifest_sha256",
        "campaign_schedule_root_sha256",
        "group_manifest_sha256",
        "execution_spec_sha256",
        "trainer_step_static_sha256",
        "objective_input_sha256",
    ):
        _sha256(document.get(role), role=role)
    _integer(
        document.get("recorded_at_unix_ns"),
        role="recorded_at",
        minimum=1,
    )
    try:
        source_binding = validate_verified_trajectory_group_source_binding(
            document.get("trajectory_source_binding")
        )
    except (TypeError, ValueError) as exc:
        raise VerifiedTransitionMeasurementChainError(
            "pre_measurement_trajectory_source_binding_invalid"
        ) from exc
    if (
        source_binding["schema"]
        not in {
            VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V2,
            VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V3,
        }
        or source_binding["group_admission_sha256"] != document["group_admission_sha256"]
        or source_binding["policy_sha256"] != document["policy_before_sha256"]
        or source_binding["execution_spec_sha256"] != document["execution_spec_sha256"]
        or source_binding["config"].get("intervention_config") is None
    ):
        _fail("pre_measurement_trajectory_source_binding_mismatch")
    config = validate_recurrent_grpo_config_contract(document.get("recurrent_grpo_config"))
    if float.fromhex(config["advantage_clip_hex"]) != float(source_binding["advantage_clip"]):
        _fail("pre_measurement_advantage_clip_mismatch")
    bridge = _validate_bridge_token_binding(document.get("bridge_token_binding"))
    state_source = validate_pre_measurement_state_source(document.get("state_source"))
    if state_source["policy_sha256"] != document["policy_before_sha256"]:
        _fail("pre_measurement_state_policy_mismatch")
    objective_inputs = {
        "group_admission_sha256": document["group_admission_sha256"],
        "trajectory_source_binding": source_binding,
        "recurrent_grpo_config": config,
        "bridge_token_binding": bridge,
    }
    if document["objective_input_sha256"] != _digest(objective_inputs):
        _fail("pre_measurement_objective_input_mismatch")
    return document


def _state_source(
    *,
    kind: str,
    successful_update_ordinal: int,
    source_sequence: int | None,
    source_admission_sha256: str | None,
    source_receipt_sha256: str,
    policy_sha256: str,
    adapter_artifact: Mapping[str, Any],
    optimizer_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": PRE_MEASUREMENT_STATE_SOURCE_SCHEMA,
        "kind": kind,
        "successful_update_ordinal": successful_update_ordinal,
        "source_sequence": source_sequence,
        "source_admission_sha256": source_admission_sha256,
        "source_receipt_sha256": source_receipt_sha256,
        "policy_sha256": policy_sha256,
        "adapter_artifact": dict(adapter_artifact),
        "optimizer_artifact": dict(optimizer_artifact),
    }
    return validate_pre_measurement_state_source({**body, "state_source_sha256": _digest(body)})


class VerifiedTransitionMeasurementChainStore:
    """Durable pre-objective intent chain over initial and post-update states."""

    def __init__(
        self,
        transaction_root: str | Path,
        *,
        transaction_store: Any,
        initial_policy_state_custody: Mapping[str, Any],
        provider_contract_sha256: str,
        training_protocol_sha256: str,
    ) -> None:
        self.transaction_root = _ensure_root(transaction_root)
        self.root = _ensure_child(
            self.transaction_root,
            "pre-measurements",
            role="collection",
        )
        self.entries = _ensure_child(
            self.root,
            "entries",
            role="entries",
        )
        self.transaction_store = transaction_store
        self.provider_contract_sha256 = _sha256(
            provider_contract_sha256,
            role="provider_contract",
        )
        self.training_protocol_sha256 = _sha256(
            training_protocol_sha256,
            role="training_protocol",
        )
        try:
            custody = validate_initial_policy_state_custody(initial_policy_state_custody)
            observed_adapter = inspect_initial_adapter_snapshot(
                custody["initial_adapter_path"],
                execution_spec_sha256=custody["execution_spec_sha256"],
            )
            observed_optimizer = inspect_initial_optimizer_snapshot(
                custody["initial_optimizer_path"]
            )
        except Exception as exc:
            if isinstance(exc, VerifiedTransitionMeasurementChainError):
                raise
            raise VerifiedTransitionMeasurementChainError(
                "pre_measurement_initial_custody_unavailable"
            ) from exc
        if (
            observed_adapter != custody["initial_adapter_artifact"]
            or observed_optimizer != custody["initial_optimizer_artifact"]
        ):
            _fail("pre_measurement_initial_custody_artifact_mismatch")
        self.initial_custody = custody
        origin = _seal(
            {
                "schema": PRE_MEASUREMENT_ORIGIN_SCHEMA,
                "provider_contract_sha256": self.provider_contract_sha256,
                "training_protocol_sha256": self.training_protocol_sha256,
                "initial_policy_state_custody": custody,
            }
        )
        origin_path = self.root / "origin.json"
        with interprocess_file_lock(self.root / ".origin.lock"):
            if atomic_write_bytes_if_absent(
                origin_path,
                _canonical_json_bytes(origin),
                durable=True,
                mode=0o400,
            ):
                os.chmod(origin_path, 0o400)
            existing = _read_document(origin_path, role="origin")
            if existing != origin:
                _fail("pre_measurement_origin_identity_conflict")

    @classmethod
    def open(
        cls,
        transaction_root: str | Path,
        **kwargs: Any,
    ) -> VerifiedTransitionMeasurementChainStore:
        return cls(transaction_root, **kwargs)

    def _entry_dir(self, sequence: int, admission_sha256: str) -> Path:
        sequence = _integer(sequence, role="sequence")
        admission = _sha256(admission_sha256, role="admission")
        return self.entries / f"seq-{sequence:08d}-{admission}"

    def _cleanup_temporaries(self, entry: Path) -> None:
        for path in tuple(entry.iterdir()):
            if not path.name.startswith(".tmp-"):
                continue
            if path.is_symlink() or _TEMP_RE.fullmatch(path.name) is None:
                _fail("pre_measurement_temporary_generation_invalid")
            _private_metadata(
                path,
                directory=True,
                role="temporary_generation",
            )
            os.chmod(path, 0o700)
            shutil.rmtree(path)
        _fsync_directory(entry)

    def _publish_generation(
        self,
        *,
        entry: Path,
        name: str,
        document_name: str,
        document: Mapping[str, Any],
        generation_kind: str,
    ) -> Path:
        target = entry / name
        if os.path.lexists(target):
            if target.is_symlink():
                _fail("pre_measurement_generation_symlink_rejected")
            return target
        temporary = entry / f".tmp-{name}-{uuid.uuid4().hex}"
        temporary.mkdir(mode=0o700)
        try:
            payload = _canonical_json_bytes(document)
            _write_file(temporary / document_name, payload)
            generation = _seal(
                {
                    "schema": PRE_MEASUREMENT_GENERATION_SCHEMA,
                    "generation": 0 if name == _INTENT_DIRECTORY else 1,
                    "kind": generation_kind,
                    "sequence": document["sequence"],
                    "group_admission_sha256": document["group_admission_sha256"],
                    "document": {
                        "path": document_name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                        "receipt_sha256": document["receipt_sha256"],
                    },
                }
            )
            _write_file(
                temporary / "generation.json",
                _canonical_json_bytes(generation),
            )
            os.chmod(temporary, 0o500)
            _fsync_directory(temporary)
            durable_replace(temporary, target)
            return target
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _load_generation(
        self,
        entry: Path,
        *,
        name: str,
        document_name: str,
        document_validator: Any,
        expected_kind: str,
    ) -> dict[str, Any] | None:
        generation_dir = entry / name
        if not os.path.lexists(generation_dir):
            return None
        metadata = _private_metadata(
            generation_dir,
            directory=True,
            role="generation",
        )
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            _fail("pre_measurement_generation_is_writable")
        expected_files = _INTENT_FILES if name == _INTENT_DIRECTORY else _ABANDONED_FILES
        if {path.name for path in generation_dir.iterdir()} != expected_files:
            _fail("pre_measurement_generation_file_set_invalid")
        generation = _read_document(
            generation_dir / "generation.json",
            role="generation_record",
        )
        required = {
            "schema",
            "generation",
            "kind",
            "sequence",
            "group_admission_sha256",
            "document",
            "receipt_sha256",
        }
        if (
            set(generation) != required
            or generation.get("schema") != PRE_MEASUREMENT_GENERATION_SCHEMA
            or generation.get("generation") != (0 if name == _INTENT_DIRECTORY else 1)
            or generation.get("kind") != expected_kind
        ):
            _fail("pre_measurement_generation_schema_invalid")
        _validate_seal(generation, role="generation")
        binding = generation.get("document")
        if not isinstance(binding, Mapping) or set(binding) != {
            "path",
            "sha256",
            "size_bytes",
            "receipt_sha256",
        }:
            _fail("pre_measurement_generation_binding_invalid")
        if binding.get("path") != document_name:
            _fail("pre_measurement_generation_binding_path_invalid")
        path = generation_dir / document_name
        document = document_validator(_read_document(path, role="generation_document"))
        payload = _canonical_json_bytes(document)
        if (
            generation["sequence"] != document["sequence"]
            or generation["group_admission_sha256"] != document["group_admission_sha256"]
            or binding.get("sha256") != hashlib.sha256(payload).hexdigest()
            or binding.get("size_bytes") != len(payload)
            or binding.get("receipt_sha256") != document["receipt_sha256"]
        ):
            _fail("pre_measurement_generation_binding_mismatch")
        return document

    def load(
        self,
        *,
        sequence: int,
        admission_sha256: str,
    ) -> dict[str, Any] | None:
        entry = self._entry_dir(sequence, admission_sha256)
        with interprocess_file_lock(self.root / f".seq-{sequence:08d}-{admission_sha256}.lock"):
            if not os.path.lexists(entry):
                return None
            if entry.is_symlink():
                _fail("pre_measurement_entry_symlink_rejected")
            _private_metadata(entry, directory=True, role="entry")
            self._cleanup_temporaries(entry)
            if any(
                path.name not in {_INTENT_DIRECTORY, _ABANDONED_DIRECTORY}
                for path in entry.iterdir()
            ):
                _fail("pre_measurement_entry_generation_name_invalid")
            return self._load_generation(
                entry,
                name=_INTENT_DIRECTORY,
                document_name="pre-measurement.json",
                document_validator=validate_pre_measurement_intent,
                expected_kind="pre_measurement_intent",
            )

    def inventory(self) -> tuple[dict[str, Any], ...]:
        with interprocess_file_lock(self.root / ".inventory.lock"):
            observed: list[tuple[int, str]] = []
            for path in self.entries.iterdir():
                if path.is_symlink():
                    _fail("pre_measurement_inventory_symlink_rejected")
                match = _ENTRY_RE.fullmatch(path.name)
                if match is None:
                    _fail("pre_measurement_inventory_name_invalid")
                _private_metadata(
                    path,
                    directory=True,
                    role="inventory_entry",
                )
                observed.append((int(match.group("sequence")), match.group("admission")))
            observed.sort()
            sequences = [sequence for sequence, _admission in observed]
            if len(sequences) != len(set(sequences)):
                _fail("pre_measurement_inventory_duplicate_sequence")
            intents = []
            for sequence, admission in observed:
                intent = self.load(
                    sequence=sequence,
                    admission_sha256=admission,
                )
                if intent is None:
                    _fail("pre_measurement_inventory_intent_missing")
                intents.append(intent)
            return tuple(intents)

    def _abandoned(
        self,
        *,
        sequence: int,
        admission_sha256: str,
    ) -> bool:
        entry = self._entry_dir(sequence, admission_sha256)
        return (
            self._load_generation(
                entry,
                name=_ABANDONED_DIRECTORY,
                document_name="reconciliation.json",
                document_validator=validate_pre_measurement_reconciliation,
                expected_kind="pre_measurement_abandoned",
            )
            is not None
        )

    def assert_no_orphans(self) -> None:
        transactions = {
            (
                int(transaction.pending_step["sequence"]),
                str(transaction.pending_step["group_admission_sha256"]),
            ): transaction
            for transaction in self.transaction_store.inventory(load_tensors=False)
        }
        for intent in self.inventory():
            key = (
                int(intent["sequence"]),
                str(intent["group_admission_sha256"]),
            )
            if key not in transactions and not self._abandoned(
                sequence=key[0],
                admission_sha256=key[1],
            ):
                _fail("pre_measurement_orphan_requires_reconciliation")
            transaction = transactions.get(key)
            if transaction is not None:
                pending = transaction.pending_step
                if pending.get("pre_measurement_sha256") != intent["receipt_sha256"]:
                    _fail("pre_measurement_transaction_binding_mismatch")

    def reconcile_interrupted_admissions(
        self,
        *,
        update_journal: Any,
        live_adapter_tensors: Mapping[str, Any],
        live_optimizer_tensors: Mapping[str, Any],
        observed_policy_sha256: str,
        reconciled_at_unix_ns: int,
    ) -> tuple[dict[str, Any], ...]:
        """Burn pre-stage admissions only after exact checkpoint restoration."""

        policy = _sha256(observed_policy_sha256, role="recovery_policy")
        reconciled_at = _integer(
            reconciled_at_unix_ns,
            role="recovery_time",
            minimum=1,
        )
        transactions = {
            str(transaction.pending_step["group_admission_sha256"]): transaction
            for transaction in self.transaction_store.inventory(load_tensors=False)
        }
        intents = {str(intent["group_admission_sha256"]): intent for intent in self.inventory()}
        journal_inventory = update_journal.inventory()
        journal_entries = {str(entry["admission_sha256"]): entry for entry in journal_inventory}
        if len(journal_entries) != len(journal_inventory):
            _fail("pre_measurement_journal_duplicate_admission")

        for admission, intent in intents.items():
            entry = journal_entries.get(admission)
            if entry is None:
                _fail("pre_measurement_reservation_missing")
            reservation = entry["reservation"]
            if (
                reservation.get("receipt_sha256") != intent["reservation_sha256"]
                or reservation.get("policy_before_sha256") != intent["policy_before_sha256"]
            ):
                _fail("pre_measurement_reservation_binding_mismatch")
            objective = entry["objective"]
            if objective is not None and (
                objective.get("pre_measurement_sha256") != intent["receipt_sha256"]
            ):
                _fail("pre_measurement_objective_binding_mismatch")

        recovered: list[dict[str, Any]] = []
        for entry in journal_inventory:
            reservation = entry["reservation"]
            if (
                reservation.get("schema") != VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2
                or reservation.get("pre_measurement_required") is not True
            ):
                continue
            admission = str(entry["admission_sha256"])
            sequence = int(reservation["campaign_sequence"])
            intent = intents.get(admission)
            transaction = transactions.get(admission)
            if transaction is not None:
                pending = transaction.pending_step
                if (
                    entry["reconciliation"] is not None
                    or self._abandoned(
                        sequence=sequence,
                        admission_sha256=admission,
                    )
                    or pending.get("sequence") != sequence
                    or pending.get("pre_measurement_sha256")
                    != (intent["receipt_sha256"] if intent is not None else None)
                ):
                    _fail("pre_measurement_staged_recovery_conflict")
                continue
            if entry["commit"] is not None:
                _fail("pre_measurement_commit_without_staged_state")
            if intent is not None and (
                intent["sequence"] != sequence
                or intent["trainer_step"] != reservation["trainer_step"]
                or intent["execution_spec_sha256"] != reservation["execution_spec_sha256"]
                or intent["group_manifest_sha256"] != reservation["group_manifest_sha256"]
            ):
                _fail("pre_measurement_reservation_scope_mismatch")
            if entry["objective"] is not None and intent is None:
                _fail("pre_measurement_objective_without_intent")

            before = str(reservation["policy_before_sha256"])
            _source, expected_adapter, expected_optimizer = self._resolve_source(
                sequence=sequence,
                policy_before_sha256=before,
            )
            if not _tensor_maps_equal(
                expected_adapter,
                live_adapter_tensors,
            ) or not _tensor_maps_equal(
                expected_optimizer,
                live_optimizer_tensors,
            ):
                _fail("pre_measurement_checkpoint_state_not_restored")
            reconciliation = entry["reconciliation"]
            if reconciliation is None:
                changed = policy != before
                reconciliation = update_journal.reconcile(
                    admission_sha256=admission,
                    reservation_sha256=str(reservation["receipt_sha256"]),
                    policy_before_sha256=before,
                    observed_policy_sha256=policy,
                    classification=(
                        "policy_changed_without_commit" if changed else "reserved_no_policy_change"
                    ),
                    reconciled_at_unix_ns=reconciled_at,
                )
            if (
                reconciliation.get("policy_before_sha256") != before
                or reconciliation.get("observed_policy_sha256") != policy
                or reconciliation.get("classification") != "reserved_no_policy_change"
                or reconciliation.get("requires_checkpoint_recovery") is not False
            ):
                _fail("pre_measurement_checkpoint_state_not_restored")

            measurement_reconciliation = None
            if intent is not None:
                if not self._abandoned(
                    sequence=sequence,
                    admission_sha256=admission,
                ):
                    measurement_reconciliation = self.reconcile_orphan(
                        sequence=sequence,
                        admission_sha256=admission,
                        live_adapter_tensors=live_adapter_tensors,
                        live_optimizer_tensors=live_optimizer_tensors,
                        reconciled_at_unix_ns=reconciled_at,
                    )
                else:
                    entry_dir = self._entry_dir(sequence, admission)
                    measurement_reconciliation = self._load_generation(
                        entry_dir,
                        name=_ABANDONED_DIRECTORY,
                        document_name="reconciliation.json",
                        document_validator=(validate_pre_measurement_reconciliation),
                        expected_kind="pre_measurement_abandoned",
                    )
            recovered.append(
                {
                    "sequence": sequence,
                    "admission_sha256": admission,
                    "update_reconciliation": reconciliation,
                    "measurement_reconciliation": (measurement_reconciliation),
                    "requires_fresh_campaign": True,
                }
            )

        self.assert_no_orphans()
        return tuple(recovered)

    def _load_initial_tensors(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        adapter = _load_bound_safetensors(
            Path(self.initial_custody["initial_adapter_path"]),
            self.initial_custody["initial_adapter_artifact"],
            role="initial_adapter",
        )
        optimizer = _load_bound_safetensors(
            Path(self.initial_custody["initial_optimizer_path"]),
            self.initial_custody["initial_optimizer_artifact"],
            role="initial_optimizer",
        )
        if (
            recurrent_policy_tensor_map_sha256(
                adapter,
                self.initial_custody["execution_spec_sha256"],
            )
            != self.initial_custody["initial_policy_sha256"]
        ):
            _fail("pre_measurement_initial_adapter_policy_mismatch")
        return adapter, optimizer

    def _resolve_source(
        self,
        *,
        sequence: int,
        policy_before_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        transactions = self.transaction_store.inventory(load_tensors=False)
        if not transactions:
            custody = self.initial_custody
            adapter, optimizer = self._load_initial_tensors()
            adapter_artifact = _normalized_artifact_binding(
                {
                    key: custody["initial_adapter_artifact"][key]
                    for key in (
                        "path",
                        "sha256",
                        "size_bytes",
                        "tensor_count",
                        "tensor_keys_sha256",
                    )
                },
                path=Path(custody["initial_adapter_path"]),
                role="initial_adapter",
            )
            optimizer_artifact = _normalized_artifact_binding(
                {
                    key: custody["initial_optimizer_artifact"][key]
                    for key in (
                        "path",
                        "sha256",
                        "size_bytes",
                        "tensor_count",
                        "tensor_keys_sha256",
                    )
                },
                path=Path(custody["initial_optimizer_path"]),
                role="initial_optimizer",
            )
            source = _state_source(
                kind="initial_policy_state",
                successful_update_ordinal=1,
                source_sequence=None,
                source_admission_sha256=None,
                source_receipt_sha256=custody["custody_sha256"],
                policy_sha256=custody["initial_policy_sha256"],
                adapter_artifact=adapter_artifact,
                optimizer_artifact=optimizer_artifact,
            )
        else:
            latest = transactions[-1]
            source_sequence = int(latest.pending_step["sequence"])
            source_admission = str(latest.pending_step["group_admission_sha256"])
            if source_sequence >= sequence:
                _fail("pre_measurement_future_or_cyclic_state_source")
            if tuple(event["kind"] for event in latest.events) != (
                "update_commit",
                "campaign_terminal",
                "trainer_checkpoint",
            ):
                _fail("pre_measurement_prior_transaction_incomplete")
            if not isinstance(
                latest.pending_step.get("pre_measurement_sha256"),
                str,
            ):
                _fail("pre_measurement_prior_transaction_chain_missing")
            loaded = self.transaction_store.load(
                sequence=source_sequence,
                admission_sha256=source_admission,
                load_tensors=True,
            )
            if loaded is None or loaded.adapter_tensors is None or loaded.optimizer_tensors is None:
                _fail("pre_measurement_prior_transaction_tensors_missing")
            stage_dir = loaded.transaction_dir / "generations" / "00000000-staged"
            source = _state_source(
                kind="prior_transaction_post_state",
                successful_update_ordinal=len(transactions) + 1,
                source_sequence=source_sequence,
                source_admission_sha256=source_admission,
                source_receipt_sha256=loaded.stage["receipt_sha256"],
                policy_sha256=loaded.pending_step["policy_after_sha256"],
                adapter_artifact=_normalized_artifact_binding(
                    loaded.stage["adapter"],
                    path=stage_dir / "adapter.safetensors",
                    role="prior_adapter",
                ),
                optimizer_artifact=_normalized_artifact_binding(
                    loaded.stage["optimizer"],
                    path=stage_dir / "optimizer.safetensors",
                    role="prior_optimizer",
                ),
            )
            adapter = loaded.adapter_tensors
            optimizer = loaded.optimizer_tensors
        if source["policy_sha256"] != policy_before_sha256:
            _fail("pre_measurement_source_policy_mismatch")
        return source, adapter, optimizer

    def begin(
        self,
        *,
        sequence: int,
        trainer_step: int,
        group_admission_sha256: str,
        reservation_sha256: str,
        policy_before_sha256: str,
        campaign_manifest_sha256: str,
        campaign_schedule_root_sha256: str,
        group_manifest_sha256: str,
        execution_spec_sha256: str,
        trainer_step_static: Mapping[str, Any],
        trajectory_source_binding: Mapping[str, Any],
        recurrent_grpo_config: RecurrentGRPOConfig,
        bridge_tokens: Sequence[int],
        live_adapter_tensors: Mapping[str, Any],
        live_optimizer_tensors: Mapping[str, Any],
        recorded_at_unix_ns: int,
    ) -> dict[str, Any]:
        sequence = _integer(sequence, role="sequence")
        if trainer_step != sequence + 1:
            _fail("pre_measurement_trainer_step_sequence_mismatch")
        admission = _sha256(
            group_admission_sha256,
            role="group_admission",
        )
        existing_intent = self.load(
            sequence=sequence,
            admission_sha256=admission,
        )
        if existing_intent is not None and self._abandoned(
            sequence=sequence,
            admission_sha256=admission,
        ):
            _fail("pre_measurement_admission_permanently_burned")
        self.assert_no_orphans()
        source, expected_adapter, expected_optimizer = self._resolve_source(
            sequence=sequence,
            policy_before_sha256=_sha256(
                policy_before_sha256,
                role="policy_before",
            ),
        )
        if not _tensor_maps_equal(
            expected_adapter,
            live_adapter_tensors,
        ):
            _fail("pre_measurement_live_adapter_state_mismatch")
        if not _tensor_maps_equal(
            expected_optimizer,
            live_optimizer_tensors,
        ):
            _fail("pre_measurement_live_optimizer_state_mismatch")
        try:
            trajectory = validate_verified_trajectory_group_source_binding(
                trajectory_source_binding
            )
        except (TypeError, ValueError) as exc:
            raise VerifiedTransitionMeasurementChainError(
                "pre_measurement_trajectory_source_binding_invalid"
            ) from exc
        config = recurrent_grpo_config_contract(recurrent_grpo_config)
        bridge = bridge_token_binding(bridge_tokens)
        objective_inputs = {
            "group_admission_sha256": admission,
            "trajectory_source_binding": trajectory,
            "recurrent_grpo_config": config,
            "bridge_token_binding": bridge,
        }
        intent = _seal(
            {
                "schema": PRE_MEASUREMENT_INTENT_SCHEMA,
                "sequence": sequence,
                "trainer_step": trainer_step,
                "group_admission_sha256": admission,
                "reservation_sha256": _sha256(
                    reservation_sha256,
                    role="reservation",
                ),
                "policy_before_sha256": policy_before_sha256,
                "provider_contract_sha256": self.provider_contract_sha256,
                "training_protocol_sha256": self.training_protocol_sha256,
                "campaign_manifest_sha256": _sha256(
                    campaign_manifest_sha256,
                    role="campaign_manifest",
                ),
                "campaign_schedule_root_sha256": _sha256(
                    campaign_schedule_root_sha256,
                    role="campaign_schedule_root",
                ),
                "group_manifest_sha256": _sha256(
                    group_manifest_sha256,
                    role="group_manifest",
                ),
                "execution_spec_sha256": _sha256(
                    execution_spec_sha256,
                    role="execution_spec",
                ),
                "trainer_step_static_sha256": _digest(
                    _clone(
                        trainer_step_static,
                        role="trainer_step_static",
                    )
                ),
                "trajectory_source_binding": trajectory,
                "recurrent_grpo_config": config,
                "bridge_token_binding": bridge,
                "objective_input_sha256": _digest(objective_inputs),
                "state_source": source,
                "recorded_at_unix_ns": _integer(
                    recorded_at_unix_ns,
                    role="recorded_at",
                    minimum=1,
                ),
            }
        )
        intent = validate_pre_measurement_intent(intent)
        entry = self._entry_dir(sequence, admission)
        with interprocess_file_lock(self.root / f".seq-{sequence:08d}-{admission}.lock"):
            if not os.path.lexists(entry):
                entry.mkdir(mode=0o700)
                _fsync_directory(self.entries)
            _private_metadata(entry, directory=True, role="entry")
            self._cleanup_temporaries(entry)
            target = self._publish_generation(
                entry=entry,
                name=_INTENT_DIRECTORY,
                document_name="pre-measurement.json",
                document=intent,
                generation_kind="pre_measurement_intent",
            )
            reopened = self._load_generation(
                entry,
                name=_INTENT_DIRECTORY,
                document_name="pre-measurement.json",
                document_validator=validate_pre_measurement_intent,
                expected_kind="pre_measurement_intent",
            )
            if target != entry / _INTENT_DIRECTORY or reopened != intent:
                _fail("pre_measurement_intent_identity_conflict")
            return intent

    def reconcile_orphan(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        live_adapter_tensors: Mapping[str, Any],
        live_optimizer_tensors: Mapping[str, Any],
        reconciled_at_unix_ns: int,
    ) -> dict[str, Any]:
        intent = self.load(
            sequence=sequence,
            admission_sha256=admission_sha256,
        )
        if intent is None:
            _fail("pre_measurement_orphan_intent_missing")
        if (
            self.transaction_store.load(
                sequence=sequence,
                admission_sha256=admission_sha256,
                load_tensors=False,
            )
            is not None
        ):
            _fail("pre_measurement_orphan_has_transaction")
        source, expected_adapter, expected_optimizer = self._resolve_source(
            sequence=sequence,
            policy_before_sha256=intent["policy_before_sha256"],
        )
        if source != intent["state_source"]:
            _fail("pre_measurement_orphan_state_source_changed")
        if not _tensor_maps_equal(
            expected_adapter,
            live_adapter_tensors,
        ) or not _tensor_maps_equal(
            expected_optimizer,
            live_optimizer_tensors,
        ):
            _fail("pre_measurement_orphan_live_state_changed")
        reconciliation = _seal(
            {
                "schema": PRE_MEASUREMENT_RECONCILIATION_SCHEMA,
                "sequence": sequence,
                "group_admission_sha256": admission_sha256,
                "pre_measurement_sha256": intent["receipt_sha256"],
                "classification": "abandoned_before_post_state_stage",
                "admission_reusable": False,
                "requires_fresh_admission": True,
                "reconciled_at_unix_ns": _integer(
                    reconciled_at_unix_ns,
                    role="reconciled_at",
                    minimum=1,
                ),
            }
        )
        entry = self._entry_dir(sequence, admission_sha256)
        with interprocess_file_lock(self.root / f".seq-{sequence:08d}-{admission_sha256}.lock"):
            self._publish_generation(
                entry=entry,
                name=_ABANDONED_DIRECTORY,
                document_name="reconciliation.json",
                document=reconciliation,
                generation_kind="pre_measurement_abandoned",
            )
            reopened = self._load_generation(
                entry,
                name=_ABANDONED_DIRECTORY,
                document_name="reconciliation.json",
                document_validator=validate_pre_measurement_reconciliation,
                expected_kind="pre_measurement_abandoned",
            )
            if reopened != reconciliation:
                _fail("pre_measurement_reconciliation_identity_conflict")
        return reconciliation


def validate_pre_measurement_reconciliation(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "sequence",
        "group_admission_sha256",
        "pre_measurement_sha256",
        "classification",
        "admission_reusable",
        "requires_fresh_admission",
        "reconciled_at_unix_ns",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail("pre_measurement_reconciliation_schema_invalid")
    document = cast(dict[str, Any], _clone(value, role="reconciliation"))
    if (
        document.get("schema") != PRE_MEASUREMENT_RECONCILIATION_SCHEMA
        or document.get("classification") != "abandoned_before_post_state_stage"
        or document.get("admission_reusable") is not False
        or document.get("requires_fresh_admission") is not True
    ):
        _fail("pre_measurement_reconciliation_invalid")
    _validate_seal(document, role="reconciliation")
    _integer(document.get("sequence"), role="reconciliation_sequence")
    _integer(
        document.get("reconciled_at_unix_ns"),
        role="reconciliation_time",
        minimum=1,
    )
    _sha256(
        document.get("group_admission_sha256"),
        role="reconciliation_admission",
    )
    _sha256(
        document.get("pre_measurement_sha256"),
        role="reconciliation_intent",
    )
    return document


def load_pre_measurement_for_transaction(
    transaction_root: str | Path,
    *,
    sequence: int,
    admission_sha256: str,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Load one intent without constructing or mutating its chain store."""

    root = Path(transaction_root).resolve(strict=True)
    entry = (
        root
        / "pre-measurements"
        / "entries"
        / f"seq-{sequence:08d}-{admission_sha256}"
        / _INTENT_DIRECTORY
    )
    document = validate_pre_measurement_intent(
        _read_document(
            entry / "pre-measurement.json",
            role="transaction_intent",
        )
    )
    if (
        document["sequence"] != sequence
        or document["group_admission_sha256"] != admission_sha256
        or document["receipt_sha256"] != expected_receipt_sha256
    ):
        _fail("pre_measurement_transaction_reference_mismatch")
    return document


def load_pre_measurement_state_tensors(
    intent: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact private adapter and optimizer in one validated intent."""

    document = validate_pre_measurement_intent(intent)
    source = validate_pre_measurement_state_source(document["state_source"])
    adapter_binding = source["adapter_artifact"]
    optimizer_binding = source["optimizer_artifact"]
    adapter = _load_bound_safetensors(
        Path(adapter_binding["path"]),
        adapter_binding,
        role="external_replay_adapter",
    )
    optimizer = _load_bound_safetensors(
        Path(optimizer_binding["path"]),
        optimizer_binding,
        role="external_replay_optimizer",
    )
    if (
        recurrent_policy_tensor_map_sha256(
            adapter,
            document["execution_spec_sha256"],
        )
        != document["policy_before_sha256"]
    ):
        _fail("pre_measurement_external_replay_policy_mismatch")
    return adapter, optimizer


__all__ = [
    "BRIDGE_TOKEN_BINDING_SCHEMA",
    "PRE_MEASUREMENT_INTENT_SCHEMA",
    "PRE_MEASUREMENT_ORIGIN_SCHEMA",
    "PRE_MEASUREMENT_RECONCILIATION_SCHEMA",
    "PRE_MEASUREMENT_STATE_SOURCE_SCHEMA",
    "RECURRENT_GRPO_CONFIG_CONTRACT_SCHEMA",
    "VerifiedTransitionMeasurementChainError",
    "VerifiedTransitionMeasurementChainStore",
    "bridge_token_binding",
    "load_pre_measurement_for_transaction",
    "load_pre_measurement_state_tensors",
    "recurrent_grpo_config_from_contract",
    "recurrent_grpo_config_contract",
    "validate_pre_measurement_intent",
    "validate_pre_measurement_reconciliation",
    "validate_pre_measurement_state_source",
    "validate_recurrent_grpo_config_contract",
]
