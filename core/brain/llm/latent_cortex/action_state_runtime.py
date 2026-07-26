"""Resident transport contract for private first-action continuations.

Only public trust material and receipts cross the ordinary MLX IPC channel.
The bearer handle and continuation values are recovered inside the resident
worker from the owner-only, externally keyed snapshot store.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Never

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from core.brain.llm.latent_cortex.action_continuation import (
    ActionOpportunityContinuation,
    PortableStateComponent,
)
from core.brain.llm.latent_cortex.action_state_capture import (
    CONTROL_ARM,
    TREATMENT_ARM,
    ActionStateCaptureError,
    PrivateActionSnapshotStore,
    VerifiedActionStateCaptureRequest,
    admit_action_state_capture_request,
    normalized_latent_reason_request_sha256,
    replay_action_state_capture_request,
)
from core.brain.llm.latent_cortex.action_state_key_custody import (
    KeychainSnapshotKeyCustodian,
)
from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.worker_capture_identity import (
    validate_worker_capture_launch_challenge,
    validate_worker_capture_origin_binding,
)

ACTION_STATE_RUNTIME_SCHEMA: Final = "aura.rlc.action_state_runtime.v1"
ACTION_STATE_RESTORE_RECEIPT_SCHEMA: Final = (
    "aura.rlc.action_state_runtime.restore_receipt.v1"
)
RESIDENT_MODEL_IDENTITY_SCHEMA: Final = "aura.rlc.resident_model_identity.v1"
RUNNER_DURABLE_STATE_SCHEMA: Final = "aura.rlc.action_state_runner.durable.v1"
RUNNER_RNG_STATE_SCHEMA: Final = "aura.rlc.action_state_runner.rng_root.v1"
_MODES: Final = frozenset({"capture", "restore"})
_ARMS: Final = frozenset({TREATMENT_ARM, CONTROL_ARM})
_COMMON_FIELDS: Final = {
    "schema",
    "mode",
    "capture_request",
    "trusted_root_public_key_pem_b64",
    "current_policy_document",
    "model_identity",
    "execution_identity",
    "latent_reason_request",
    "resident_worker_origin_binding",
}


class ActionStateRuntimeError(ValueError):
    """Stable fail-closed resident transport error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    if not isinstance(code, str) or not code or code != code.strip():
        raise ActionStateRuntimeError("action_state_runtime_error_code_invalid")
    raise ActionStateRuntimeError(code)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"action_state_runtime_{role}_invalid")
    try:
        return json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        _fail(f"action_state_runtime_{role}_invalid")


def _decode_b64(value: Any, *, role: str) -> bytes:
    if not isinstance(value, str):
        _fail(f"action_state_runtime_{role}_invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, TypeError, ValueError):
        _fail(f"action_state_runtime_{role}_invalid")
    if not raw:
        _fail(f"action_state_runtime_{role}_invalid")
    return raw


def runner_state_for_capture_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact signed runner roots without transporting state.

    The random capture id is the RNG root; the durable root binds the campaign
    location and immutable request identities.  A runner computes these values
    before signing its commitments, and the worker independently reconstructs
    them from the signed payload.
    """

    value = _mapping(payload, role="capture_payload")
    durable_names = (
        "campaign_name",
        "campaign_design_sha256",
        "campaign_protocol_sha256",
        "policy_sha256",
        "policy_revision",
        "pair_id",
        "task_id",
        "task_payload_sha256",
        "model_identity_sha256",
        "model_weights_identity_sha256",
        "execution_identity_sha256",
        "latent_reason_request_sha256",
    )
    rng_names = (
        "capture_id",
        "pair_id",
        "task_id",
        "action",
        "calibration_bucket",
        "bucket_classifier_sha256",
        "bucket_evidence_sha256",
    )
    if any(name not in value for name in (*durable_names, *rng_names)):
        _fail("action_state_runtime_capture_payload_incomplete")
    durable_state = {
        "schema": RUNNER_DURABLE_STATE_SCHEMA,
        **{name: value[name] for name in durable_names},
    }
    rng_body = {name: value[name] for name in rng_names}
    rng_state = {
        "schema": RUNNER_RNG_STATE_SCHEMA,
        **rng_body,
        "seed_root_sha256": _digest(rng_body),
    }
    return {"durable_state": durable_state, "rng_state": rng_state}


def runner_state_commitments(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return portable-codec commitments for request construction."""

    state = runner_state_for_capture_payload(payload)
    return {
        "runner_durable_state_commitment_sha256": PortableStateComponent.from_value(
            state["durable_state"]
        ).sha256(),
        "runner_rng_root_commitment_sha256": PortableStateComponent.from_value(
            state["rng_state"]
        ).sha256(),
    }


def validate_resident_model_identity(value: Any) -> dict[str, Any]:
    """Validate the stable compute identity shared by replacement workers."""

    identity = _mapping(value, role="resident_model_identity")
    expected_fields = {
        "schema",
        "worker_model_path",
        "worker_model_parameter_count",
        "worker_model_stored_parameter_element_count",
        "worker_model_parameter_count_basis",
        "checkpoint",
        "worker_adapters",
        "worker_adapter_stack_sha256",
        "worker_tokenizer",
        "worker_quantization",
        "worker_stack_identity_gaps",
    }
    checkpoint = identity.get("checkpoint")
    model_path = identity.get("worker_model_path")
    adapters = identity.get("worker_adapters")
    tokenizer = identity.get("worker_tokenizer")
    quantization = identity.get("worker_quantization")
    adapters_valid = isinstance(adapters, list) and all(
        isinstance(adapter, dict)
        and set(adapter) == {"name", "type", "rank", "scale"}
        and isinstance(adapter.get("name"), str)
        and bool(adapter["name"])
        and isinstance(adapter.get("type"), str)
        and bool(adapter["type"])
        and type(adapter.get("rank")) is int
        and adapter["rank"] >= 0
        and not isinstance(adapter.get("scale"), bool)
        and isinstance(adapter.get("scale"), (int, float))
        and math.isfinite(float(adapter["scale"]))
        for adapter in adapters or []
    )
    tokenizer_valid = isinstance(tokenizer, dict) and bool(tokenizer) and all(
        isinstance(filename, str)
        and bool(filename)
        and Path(filename).name == filename
        and _is_sha256(digest)
        for filename, digest in tokenizer.items()
    )
    quantization_valid = (
        isinstance(quantization, dict)
        and set(quantization)
        == {"bits", "group_size", "dtype", "model_type", "config_sha256"}
        and type(quantization.get("bits")) is int
        and quantization["bits"] >= 0
        and type(quantization.get("group_size")) is int
        and quantization["group_size"] >= 0
        and isinstance(quantization.get("dtype"), str)
        and bool(quantization["dtype"])
        and isinstance(quantization.get("model_type"), str)
        and bool(quantization["model_type"])
        and _is_sha256(quantization.get("config_sha256"))
    )
    if (
        set(identity) != expected_fields
        or identity.get("schema") != RESIDENT_MODEL_IDENTITY_SCHEMA
        or not isinstance(model_path, str)
        or not model_path
        or not Path(model_path).is_absolute()
        or str(Path(model_path).resolve(strict=False)) != model_path
        or type(identity.get("worker_model_parameter_count")) is not int
        or identity["worker_model_parameter_count"] <= 0
        or type(identity.get("worker_model_stored_parameter_element_count")) is not int
        or identity["worker_model_stored_parameter_element_count"] <= 0
        or identity.get("worker_model_parameter_count_basis")
        not in {"architecture_config_logical", "stored_tensor_elements"}
        or (
            identity["worker_model_parameter_count_basis"]
            == "architecture_config_logical"
            and identity["worker_model_parameter_count"]
            < identity["worker_model_stored_parameter_element_count"]
        )
        or (
            identity["worker_model_parameter_count_basis"]
            == "stored_tensor_elements"
            and identity["worker_model_parameter_count"]
            != identity["worker_model_stored_parameter_element_count"]
        )
        or not isinstance(checkpoint, dict)
        or set(checkpoint) != {"fingerprint", "method", "files"}
        or checkpoint.get("method") != "sha256"
        or not _is_sha256(checkpoint.get("fingerprint"))
        or type(checkpoint.get("files")) is not int
        or checkpoint["files"] <= 0
        or not adapters_valid
        or not _is_sha256(identity.get("worker_adapter_stack_sha256"))
        or _digest(adapters) != identity["worker_adapter_stack_sha256"]
        or not tokenizer_valid
        or not quantization_valid
        or identity.get("worker_stack_identity_gaps") != []
    ):
        _fail("action_state_runtime_resident_model_identity_invalid")
    return identity


def resident_model_weights_identity_sha256(model_identity: Mapping[str, Any]) -> str:
    """Commit the exact checkpoint and every attached computation modifier."""

    identity = validate_resident_model_identity(model_identity)
    return _digest(
        {
            "checkpoint": identity["checkpoint"],
            "worker_adapters": identity["worker_adapters"],
            "worker_adapter_stack_sha256": identity[
                "worker_adapter_stack_sha256"
            ],
            "worker_tokenizer": identity["worker_tokenizer"],
            "worker_quantization": identity["worker_quantization"],
        }
    )


def resident_model_identity_for_worker(
    worker_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute claim-grade model identity from the loaded resident worker."""

    worker = _mapping(worker_identity, role="worker_identity")
    from core.brain.llm.latent_cortex.governance import (
        checkpoint_file_fingerprint,
    )

    checkpoint = checkpoint_file_fingerprint(worker.get("worker_model_path", ""))
    identity = {
        "schema": RESIDENT_MODEL_IDENTITY_SCHEMA,
        "worker_model_path": worker.get("worker_model_path"),
        "worker_model_parameter_count": worker.get("worker_model_parameter_count"),
        "worker_model_stored_parameter_element_count": worker.get(
            "worker_model_stored_parameter_element_count"
        ),
        "worker_model_parameter_count_basis": worker.get(
            "worker_model_parameter_count_basis"
        ),
        "checkpoint": checkpoint,
        "worker_adapters": worker.get("worker_adapters"),
        "worker_adapter_stack_sha256": worker.get(
            "worker_adapter_stack_sha256"
        ),
        "worker_tokenizer": worker.get("worker_tokenizer"),
        "worker_quantization": worker.get("worker_quantization"),
        "worker_stack_identity_gaps": worker.get("worker_stack_identity_gaps"),
    }
    return validate_resident_model_identity(identity)


def _expected_supervisor_key(
    launch_challenge: Mapping[str, Any],
    *,
    now_unix: int,
) -> bytes:
    challenge = validate_worker_capture_launch_challenge(
        launch_challenge,
        now_unix=now_unix,
    )
    raw = _decode_b64(
        challenge.get("supervisor_public_key_b64"),
        role="supervisor_public_key",
    )
    if len(raw) != 32:
        _fail("action_state_runtime_supervisor_public_key_invalid")
    return raw


@dataclass(frozen=True, slots=True)
class AdmittedActionStateRuntime:
    mode: str
    admission: VerifiedActionStateCaptureRequest = field(repr=False)
    trusted_root_public_key_pem: bytes = field(repr=False)
    capture_supervisor_public_key: bytes
    resident_supervisor_public_key: bytes
    model_identity: dict[str, Any]
    execution_identity: dict[str, Any]
    latent_reason_request: dict[str, Any]
    runner_state: dict[str, Any] = field(repr=False)
    resident_worker_origin_binding: dict[str, Any]
    capture_receipt: dict[str, Any] | None = None
    arm: str | None = None


def build_action_state_runtime_frame(
    *,
    mode: str,
    capture_request: Mapping[str, Any],
    trusted_root_public_key_pem: bytes,
    current_policy_document: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    latent_reason_request: Mapping[str, Any],
    resident_worker_origin_binding: Mapping[str, Any] | None = None,
    capture_receipt: Mapping[str, Any] | None = None,
    arm: str | None = None,
) -> dict[str, Any]:
    """Build the public-only frame used by runner, service, and client."""

    if mode not in _MODES or not isinstance(trusted_root_public_key_pem, bytes):
        _fail("action_state_runtime_frame_arguments_invalid")
    body = {
        "schema": ACTION_STATE_RUNTIME_SCHEMA,
        "mode": mode,
        "capture_request": _mapping(capture_request, role="capture_request"),
        "trusted_root_public_key_pem_b64": base64.b64encode(
            trusted_root_public_key_pem
        ).decode("ascii"),
        "current_policy_document": _mapping(
            current_policy_document, role="policy"
        ),
        "model_identity": _mapping(model_identity, role="model_identity"),
        "execution_identity": _mapping(
            execution_identity, role="execution_identity"
        ),
        "latent_reason_request": _mapping(
            latent_reason_request, role="latent_reason_request"
        ),
        "resident_worker_origin_binding": _mapping(
            resident_worker_origin_binding
            or _mapping(capture_request, role="capture_request").get(
                "request_payload", {}
            ).get("worker_origin_binding"),
            role="resident_worker_origin_binding",
        ),
    }
    if mode == "restore":
        if capture_receipt is None or arm not in _ARMS:
            _fail("action_state_runtime_restore_arguments_invalid")
        body["capture_receipt"] = _mapping(
            capture_receipt, role="capture_receipt"
        )
        body["arm"] = arm
    elif capture_receipt is not None or arm is not None:
        _fail("action_state_runtime_capture_arguments_invalid")
    return body


def admit_action_state_runtime(
    value: Any,
    *,
    worker_launch_challenge: Mapping[str, Any],
    now_unix: int,
) -> AdmittedActionStateRuntime:
    """Admit one public capture/restore frame at the resident boundary."""

    wire = _mapping(value, role="frame")
    mode = wire.get("mode")
    if mode not in _MODES:
        _fail("action_state_runtime_mode_invalid")
    expected_fields = set(_COMMON_FIELDS)
    if mode == "restore":
        expected_fields.update({"capture_receipt", "arm"})
    if set(wire) != expected_fields or wire.get("schema") != ACTION_STATE_RUNTIME_SCHEMA:
        _fail("action_state_runtime_fields_invalid")
    root_pem = _decode_b64(
        wire.get("trusted_root_public_key_pem_b64"),
        role="trust_root",
    )
    expected_supervisor = _expected_supervisor_key(
        worker_launch_challenge,
        now_unix=now_unix,
    )
    raw_capture_request = _mapping(
        wire.get("capture_request"), role="capture_request"
    )
    capture_binding = _mapping(
        raw_capture_request.get("request_payload", {}).get(
            "worker_origin_binding"
        ),
        role="capture_worker_origin_binding",
    )
    capture_challenge = _mapping(
        capture_binding.get("launch_challenge"),
        role="capture_worker_launch_challenge",
    )
    capture_supervisor = _decode_b64(
        capture_challenge.get("supervisor_public_key_b64"),
        role="capture_supervisor_public_key",
    )
    if len(capture_supervisor) != 32:
        _fail("action_state_runtime_capture_supervisor_public_key_invalid")
    resident_binding = _mapping(
        wire.get("resident_worker_origin_binding"),
        role="resident_worker_origin_binding",
    )
    current_policy_document = _mapping(
        wire.get("current_policy_document"), role="policy"
    )
    try:
        if mode == "capture":
            admission = admit_action_state_capture_request(
                raw_capture_request,
                trusted_root_public_key_pem=root_pem,
                expected_supervisor_public_key=capture_supervisor,
                current_policy_document=current_policy_document,
                now_unix=now_unix,
            )
        else:
            admission = replay_action_state_capture_request(
                raw_capture_request,
                trusted_root_public_key_pem=root_pem,
                expected_supervisor_public_key=capture_supervisor,
            )
            current_policy = validate_campaign_trust_policy(
                current_policy_document,
                trusted_root_public_key_pem=root_pem,
                expected_campaign_name=admission.payload["campaign_name"],
                expected_policy_sha256=admission.payload["policy_sha256"],
                expected_protocol_sha256=admission.payload[
                    "campaign_protocol_sha256"
                ],
                minimum_policy_revision=admission.payload["policy_revision"],
                now_unix=now_unix,
            )
            if (
                canonical_json_bytes(current_policy.document)
                != canonical_json_bytes(admission.policy.document)
                or now_unix > admission.payload["capture_not_after_unix"]
            ):
                _fail("action_state_runtime_restore_authority_not_current")
    except (ActionStateCaptureError, TypeError, ValueError) as exc:
        raise ActionStateRuntimeError("action_state_runtime_capture_request_rejected") from exc
    try:
        resident_binding = validate_worker_capture_origin_binding(
            resident_binding,
            expected_supervisor_public_key=expected_supervisor,
            now_unix=now_unix,
        )
    except (TypeError, ValueError) as exc:
        raise ActionStateRuntimeError(
            "action_state_runtime_resident_worker_rejected"
        ) from exc
    current_challenge = validate_worker_capture_launch_challenge(
        worker_launch_challenge,
        expected_supervisor_public_key=expected_supervisor,
        now_unix=now_unix,
    )
    if (
        resident_binding.get("launch_challenge") != current_challenge
        or (
            mode == "capture"
            and resident_binding != admission.payload["worker_origin_binding"]
        )
    ):
        _fail("action_state_runtime_resident_worker_mismatch")
    model_identity = validate_resident_model_identity(wire.get("model_identity"))
    execution_identity = _mapping(
        wire.get("execution_identity"), role="execution_identity"
    )
    latent_request = _mapping(
        wire.get("latent_reason_request"), role="latent_reason_request"
    )
    payload = admission.payload
    if (
        _digest(model_identity) != payload["model_identity_sha256"]
        or resident_model_weights_identity_sha256(model_identity)
        != payload["model_weights_identity_sha256"]
        or _digest(execution_identity) != payload["execution_identity_sha256"]
        or normalized_latent_reason_request_sha256(latent_request)
        != payload["latent_reason_request_sha256"]
    ):
        _fail("action_state_runtime_public_binding_mismatch")
    runner_state = runner_state_for_capture_payload(payload)
    commitments = runner_state_commitments(payload)
    if any(payload[name] != digest for name, digest in commitments.items()):
        _fail("action_state_runtime_runner_commitment_mismatch")
    capture_receipt = None
    arm = None
    if mode == "restore":
        capture_receipt = _mapping(
            wire.get("capture_receipt"), role="capture_receipt"
        )
        arm = wire.get("arm")
        if arm not in _ARMS:
            _fail("action_state_runtime_arm_invalid")
        if capture_receipt.get("request_sha256") != admission.request_sha256:
            _fail("action_state_runtime_receipt_request_mismatch")
    return AdmittedActionStateRuntime(
        mode=mode,
        admission=admission,
        trusted_root_public_key_pem=root_pem,
        capture_supervisor_public_key=capture_supervisor,
        resident_supervisor_public_key=expected_supervisor,
        model_identity=model_identity,
        execution_identity=execution_identity,
        latent_reason_request=latent_request,
        runner_state=runner_state,
        resident_worker_origin_binding=resident_binding,
        capture_receipt=capture_receipt,
        arm=arm,
    )


def action_state_store_root() -> Path:
    """Return the fixed owner-private store root.

    Tests may redirect the root explicitly. Production ignores an injected
    root unless the process is running under pytest.
    """

    override = os.environ.get("AURA_RLC_ACTION_STATE_TEST_ROOT", "")
    if override:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            _fail("action_state_runtime_test_root_outside_tests")
        return Path(override).expanduser().absolute()
    return Path.home() / ".aura" / "private" / "rlc-action-state-v1"


def open_action_state_store() -> tuple[
    PrivateActionSnapshotStore,
    KeychainSnapshotKeyCustodian,
]:
    """Open production custody read-only and bind it to the fixed store."""

    custodian = KeychainSnapshotKeyCustodian.from_system()
    with ExitStack() as cleanup:
        cleanup.callback(custodian.close)
        store = PrivateActionSnapshotStore(
            action_state_store_root(),
            key_custodian=custodian,
        )
        cleanup.pop_all()
    return store, custodian


def provision_action_state_store_custody() -> dict[str, Any]:
    """Provision custody in the parent only when the lab lane is requested."""

    custodian = KeychainSnapshotKeyCustodian.provision_system()
    try:
        return dict(custodian.identity)
    finally:
        custodian.close()


def continuation_from_private_state(
    private_state: Mapping[str, Any],
    capture_receipt: Mapping[str, Any],
) -> ActionOpportunityContinuation:
    """Reconstruct one continuation using only worker-private state."""

    state = dict(private_state)
    opportunity = _mapping(
        capture_receipt.get("first_action_opportunity"), role="opportunity"
    )
    continuation = ActionOpportunityContinuation(
        private_state=state,
        episode_step=opportunity.get("episode_step"),
        schedule_step=opportunity.get("schedule_step"),
        branch_id=opportunity.get("branch_id"),
        layer_index=opportunity.get("layer_index"),
        kv_position=opportunity.get("kv_position"),
    )
    if continuation.state_components != capture_receipt.get("state_components"):
        _fail("action_state_runtime_continuation_receipt_mismatch")
    return continuation


def build_action_state_restore_receipt(
    *,
    capture_receipt: Mapping[str, Any],
    custody_restore_receipt: Mapping[str, Any],
    action_intervention: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    worker_private_key: Ed25519PrivateKey,
    custody_lifecycle_receipts: Mapping[str, Any],
    resident_worker_origin_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Sign the public evidence that one private arm restore completed."""

    capture = _mapping(capture_receipt, role="capture_receipt")
    restore = _mapping(custody_restore_receipt, role="custody_restore_receipt")
    intervention = _mapping(action_intervention, role="action_intervention")
    runtime = _mapping(runtime_identity, role="runtime_identity")
    lifecycle = _mapping(
        custody_lifecycle_receipts, role="custody_lifecycle_receipts"
    )
    resident_origin = _mapping(
        resident_worker_origin_binding,
        role="resident_worker_origin_binding",
    )
    if not isinstance(worker_private_key, Ed25519PrivateKey):
        _fail("action_state_runtime_worker_private_key_invalid")
    if (
        restore.get("request_sha256") != capture.get("request_sha256")
        or restore.get("snapshot_sha256")
        != capture.get("private_snapshot_envelope_sha256")
        or restore.get("state_components") != capture.get("state_components")
        or restore.get("state_sha256") != capture.get("state_sha256")
        or restore.get("post_apply_state_sha256") != capture.get("state_sha256")
        or restore.get("arm")
        != intervention.get("authority_payload", {}).get("arm")
    ):
        _fail("action_state_runtime_restore_binding_mismatch")
    body = {
        "schema": ACTION_STATE_RESTORE_RECEIPT_SCHEMA,
        "request_sha256": capture["request_sha256"],
        "capture_receipt_sha256": capture["receipt_sha256"],
        "intervention_sha256": intervention["intervention_sha256"],
        "runtime_identity_sha256": _digest(runtime),
        "pair_id": restore["pair_id"],
        "arm": restore["arm"],
        "custody_restore_receipt": restore,
        "custody_lifecycle_receipts": lifecycle,
        "resident_worker_origin_binding": resident_origin,
    }
    public_raw = worker_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed = canonical_json_bytes(body)
    origin = {
        "algorithm": "Ed25519",
        "key_id": hashlib.sha256(public_raw).hexdigest(),
        "public_key_b64": base64.b64encode(public_raw).decode("ascii"),
        "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        "signature_b64": base64.b64encode(
            worker_private_key.sign(signed)
        ).decode("ascii"),
    }
    complete = {**body, "worker_origin": origin}
    return {**complete, "receipt_sha256": _digest(complete)}


def validate_action_state_restore_receipt(
    value: Any,
    *,
    capture_receipt: Mapping[str, Any],
    action_intervention: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    expected_worker_public_key_b64: str,
    expected_supervisor_public_key: bytes,
) -> dict[str, Any]:
    """Independently replay one public restore receipt."""

    receipt = _mapping(value, role="restore_receipt")
    expected_fields = {
        "schema",
        "request_sha256",
        "capture_receipt_sha256",
        "intervention_sha256",
        "runtime_identity_sha256",
        "pair_id",
        "arm",
        "custody_restore_receipt",
        "custody_lifecycle_receipts",
        "resident_worker_origin_binding",
        "worker_origin",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        _fail("action_state_runtime_restore_receipt_fields")
    complete = {name: receipt[name] for name in expected_fields - {"receipt_sha256"}}
    if (
        receipt.get("schema") != ACTION_STATE_RESTORE_RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != _digest(complete)
    ):
        _fail("action_state_runtime_restore_receipt_invalid")
    capture = _mapping(capture_receipt, role="capture_receipt")
    intervention = _mapping(action_intervention, role="action_intervention")
    runtime = _mapping(runtime_identity, role="runtime_identity")
    custody = _mapping(
        receipt.get("custody_restore_receipt"), role="custody_restore_receipt"
    )
    lifecycle = _mapping(
        receipt.get("custody_lifecycle_receipts"),
        role="custody_lifecycle_receipts",
    )
    try:
        resident_origin = validate_worker_capture_origin_binding(
            receipt.get("resident_worker_origin_binding"),
            expected_supervisor_public_key=expected_supervisor_public_key,
        )
    except (TypeError, ValueError) as exc:
        raise ActionStateRuntimeError(
            "action_state_runtime_restore_resident_origin_invalid"
        ) from exc
    custody_body = {
        name: item for name, item in custody.items() if name != "restore_receipt_sha256"
    }
    if (
        custody.get("restore_receipt_sha256") != _digest(custody_body)
        or custody.get("request_sha256") != capture.get("request_sha256")
        or custody.get("snapshot_sha256")
        != capture.get("private_snapshot_envelope_sha256")
        or custody.get("state_components") != capture.get("state_components")
        or custody.get("state_sha256") != capture.get("state_sha256")
        or custody.get("post_apply_state_sha256") != capture.get("state_sha256")
        or custody.get("all_bytes_verified_before_return") is not True
        or custody.get("state_applied_before_return") is not True
        or receipt.get("request_sha256") != capture.get("request_sha256")
        or receipt.get("capture_receipt_sha256") != capture.get("receipt_sha256")
        or receipt.get("intervention_sha256")
        != intervention.get("intervention_sha256")
        or receipt.get("runtime_identity_sha256") != _digest(runtime)
        or receipt.get("pair_id") != custody.get("pair_id")
        or receipt.get("arm") != custody.get("arm")
        or receipt.get("arm")
        != intervention.get("authority_payload", {}).get("arm")
    ):
        _fail("action_state_runtime_restore_receipt_binding_mismatch")
    if set(lifecycle) == {"complete"}:
        if lifecycle.get("complete") is not False:
            _fail("action_state_runtime_lifecycle_receipt_invalid")
    elif set(lifecycle) == {"complete", "seal_receipt", "erasure_receipt"}:
        seal = _mapping(lifecycle.get("seal_receipt"), role="seal_receipt")
        erasure = _mapping(
            lifecycle.get("erasure_receipt"), role="erasure_receipt"
        )
        seal_body = {
            name: item for name, item in seal.items() if name != "seal_receipt_sha256"
        }
        erasure_body = {
            name: item
            for name, item in erasure.items()
            if name != "erasure_receipt_sha256"
        }
        if (
            lifecycle.get("complete") is not True
            or seal.get("seal_receipt_sha256") != _digest(seal_body)
            or seal.get("request_sha256") != capture.get("request_sha256")
            or seal.get("snapshot_sha256")
            != capture.get("private_snapshot_envelope_sha256")
            or seal.get("both_arms_used_exactly_once") is not True
            or set(seal.get("arm_restore_receipts", {})) != _ARMS
            or erasure.get("erasure_receipt_sha256") != _digest(erasure_body)
            or erasure.get("request_sha256") != capture.get("request_sha256")
            or erasure.get("snapshot_sha256")
            != capture.get("private_snapshot_envelope_sha256")
            or erasure.get("seal_receipt_sha256")
            != seal.get("seal_receipt_sha256")
            or erasure.get("all_snapshot_files_absent") is not True
            or erasure.get("cryptographic_key_destroyed") is not True
            or erasure.get("ciphertext_namespace_deleted") is not True
        ):
            _fail("action_state_runtime_lifecycle_receipt_invalid")
    else:
        _fail("action_state_runtime_lifecycle_receipt_fields")
    origin = _mapping(receipt.get("worker_origin"), role="worker_origin")
    if set(origin) != {
        "algorithm",
        "key_id",
        "public_key_b64",
        "signed_payload_sha256",
        "signature_b64",
    }:
        _fail("action_state_runtime_restore_origin_fields")
    expected_public = _decode_b64(
        expected_worker_public_key_b64,
        role="expected_worker_public_key",
    )
    observed_public = _decode_b64(
        origin.get("public_key_b64"), role="worker_public_key"
    )
    resident_worker_identity = _mapping(
        resident_origin.get("worker_identity"), role="resident_worker_identity"
    )
    signed = canonical_json_bytes(
        {name: receipt[name] for name in expected_fields - {"worker_origin", "receipt_sha256"}}
    )
    if (
        len(expected_public) != 32
        or observed_public != expected_public
        or resident_worker_identity.get("public_key_b64")
        != expected_worker_public_key_b64
        or origin.get("algorithm") != "Ed25519"
        or origin.get("key_id") != hashlib.sha256(observed_public).hexdigest()
        or origin.get("signed_payload_sha256") != hashlib.sha256(signed).hexdigest()
    ):
        _fail("action_state_runtime_restore_origin_invalid")
    try:
        Ed25519PublicKey.from_public_bytes(observed_public).verify(
            _decode_b64(origin.get("signature_b64"), role="worker_signature"),
            signed,
        )
    except (InvalidSignature, ValueError) as exc:
        raise ActionStateRuntimeError(
            "action_state_runtime_restore_signature_invalid"
        ) from exc
    return receipt


def assert_public_runtime_result(value: Mapping[str, Any]) -> None:
    """Reject accidental private-custody material before an IPC response."""

    encoded = canonical_json_bytes(dict(value))
    forbidden = (
        b"asc1_",
        b"wrapped_key_b64",
        b"plaintext_key_sha256",
        b"private_state",
        b"runner_state",
        b"portable_state_v1",
    )
    if any(marker in encoded for marker in forbidden):
        _fail("action_state_runtime_private_material_leaked")


def verify_action_state_pair_evidence(
    *,
    capture_receipt: Mapping[str, Any],
    restore_receipts: Mapping[str, Mapping[str, Any]],
    interventions: Mapping[str, Mapping[str, Any]],
    runtime_identity: Mapping[str, Any],
    expected_worker_public_keys_b64: Mapping[str, str],
    expected_supervisor_public_keys: Mapping[str, bytes],
) -> dict[str, Any]:
    """Independently verify two once-only restores and terminal erasure."""

    if (
        set(restore_receipts) != _ARMS
        or set(interventions) != _ARMS
        or set(expected_worker_public_keys_b64) != _ARMS
        or set(expected_supervisor_public_keys) != _ARMS
    ):
        _fail("action_state_runtime_pair_coverage_invalid")
    verified = {
        arm: validate_action_state_restore_receipt(
            restore_receipts[arm],
            capture_receipt=capture_receipt,
            action_intervention=interventions[arm],
            runtime_identity=runtime_identity,
            expected_worker_public_key_b64=expected_worker_public_keys_b64[arm],
            expected_supervisor_public_key=expected_supervisor_public_keys[arm],
        )
        for arm in sorted(_ARMS)
    }
    lifecycle = [
        arm
        for arm, receipt in verified.items()
        if receipt["custody_lifecycle_receipts"]["complete"] is True
    ]
    if len(lifecycle) != 1:
        _fail("action_state_runtime_terminal_erasure_invalid")
    terminal = verified[lifecycle[0]]["custody_lifecycle_receipts"]
    seal_arms = terminal["seal_receipt"]["arm_restore_receipts"]
    if any(
        seal_arms[arm]
        != verified[arm]["custody_restore_receipt"]["restore_receipt_sha256"]
        for arm in _ARMS
    ):
        _fail("action_state_runtime_pair_restore_lineage_mismatch")
    body = {
        "schema": "aura.rlc.action_state_runtime.pair_verification.v1",
        "request_sha256": capture_receipt.get("request_sha256"),
        "capture_receipt_sha256": capture_receipt.get("receipt_sha256"),
        "restore_receipt_sha256": {
            arm: verified[arm]["receipt_sha256"] for arm in sorted(_ARMS)
        },
        "both_arms_restored_once": True,
        "terminal_erasure_verified": True,
        "terminal_arm": lifecycle[0],
    }
    return {**body, "verification_sha256": _digest(body)}


__all__ = [
    "ACTION_STATE_RUNTIME_SCHEMA",
    "ACTION_STATE_RESTORE_RECEIPT_SCHEMA",
    "RESIDENT_MODEL_IDENTITY_SCHEMA",
    "ActionStateRuntimeError",
    "AdmittedActionStateRuntime",
    "admit_action_state_runtime",
    "assert_public_runtime_result",
    "build_action_state_runtime_frame",
    "build_action_state_restore_receipt",
    "continuation_from_private_state",
    "open_action_state_store",
    "provision_action_state_store_custody",
    "resident_model_identity_for_worker",
    "resident_model_weights_identity_sha256",
    "runner_state_commitments",
    "runner_state_for_capture_payload",
    "validate_action_state_restore_receipt",
    "validate_resident_model_identity",
    "verify_action_state_pair_evidence",
]
