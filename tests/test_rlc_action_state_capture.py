"""Adversarial contracts for the SPARK-051 private state-capture protocol."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing as mp
import os
import signal
import stat
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex import action_state_capture as capture_module
from core.brain.llm.latent_cortex.action_continuation import PortableStateComponent
from core.brain.llm.latent_cortex.action_state_capture import (
    CONTROL_ARM,
    TREATMENT_ARM,
    ActionStateCaptureError,
    PrivateActionSnapshotStore,
    UnknownActionStateApplicationError,
    admit_action_state_capture_request,
    build_action_state_capture_receipt,
    build_action_state_capture_request,
    replay_action_state_capture_request,
    validate_action_state_capture_receipt,
    validate_action_state_capture_receipt_public,
)
from core.brain.llm.latent_cortex.action_state_key_custody import (
    KeychainSnapshotKeyCustodian,
)
from core.brain.llm.latent_cortex.action_state_runtime import (
    ActionStateRuntimeError,
    admit_action_state_runtime,
    assert_public_runtime_result,
    build_action_state_restore_receipt,
    build_action_state_runtime_frame,
    continuation_from_private_state,
    runner_state_for_capture_payload,
    validate_action_state_restore_receipt,
    verify_action_state_pair_evidence,
)
from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    TASK_ISSUER,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.worker_capture_identity import (
    build_worker_capture_identity,
    build_worker_capture_launch_authority,
    build_worker_capture_origin_binding,
)

NOW = 10_000
CAMPAIGN = "spark-051-state-capture"


class _KeychainBackend:
    def __init__(self, key: bytes = b"K" * 32) -> None:
        self.value = base64.b64encode(key).decode("ascii")

    def get_password(self, _service: str, _account: str) -> str:
        return self.value

    def set_password(self, _service: str, _account: str, password: str) -> bool:
        self.value = password
        return True


def _custodian(key: bytes = b"K" * 32) -> KeychainSnapshotKeyCustodian:
    return KeychainSnapshotKeyCustodian(_KeychainBackend(key))


def _sha(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _state_sha(value) -> str:
    if isinstance(value, PortableStateComponent):
        return value.sha256()
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(f"spark-051:{label}".encode()).digest()
    )


def _public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


SUPERVISOR_KEY = _private("worker-supervisor")
SUPERVISOR_PUBLIC_KEY = SUPERVISOR_KEY.public_key()


def _policy_document(
    *,
    root: Ed25519PrivateKey,
    role_keys: dict[str, Ed25519PrivateKey],
    revision: int = 1,
    previous_policy_sha256: str | None = None,
    revoked_key_ids: list[str] | None = None,
) -> dict:
    roles = {}
    for role, key in role_keys.items():
        raw = _public_raw(key)
        roles[role] = {
            "signer_id": f"{role}-signer-r{revision}",
            "organization_id": f"{role}-organization-r{revision}",
            "public_key_b64": base64.b64encode(raw).decode("ascii"),
            "key_id": hashlib.sha256(raw).hexdigest(),
            "implementation_sha256": hashlib.sha256(
                f"{role}:implementation:r{revision}".encode()
            ).hexdigest(),
            "release_sha256": hashlib.sha256(f"{role}:release:r{revision}".encode()).hexdigest(),
            "custody_class": "external_service",
            "custody_evidence_sha256": hashlib.sha256(
                f"{role}:custody:r{revision}".encode()
            ).hexdigest(),
        }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "spark-051-state-capture-policy",
        "policy_revision": revision,
        "campaign_name": CAMPAIGN,
        "protocol_sha256": hashlib.sha256(b"spark-051-state-capture-protocol").hexdigest(),
        "previous_policy_sha256": previous_policy_sha256,
        "revoked_key_ids": list(revoked_key_ids or []),
        "issued_at_unix": NOW - 200,
        "not_before_unix": NOW - 100,
        "expires_at_unix": NOW + 1_000,
        "roles": roles,
    }
    root_raw = _public_raw(root)
    signed = canonical_json_bytes(body)
    return {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }


def _trust_fixture():
    root = _private("root")
    role_keys = {role: _private(role) for role in CAMPAIGN_TRUST_ROLES}
    document = _policy_document(root=root, role_keys=role_keys)
    root_pem = _public_pem(root)
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=root_pem,
        expected_campaign_name=CAMPAIGN,
        now_unix=NOW,
    )
    return root, root_pem, role_keys, policy


def _latent_request(*, prompt: str = "Find the invariant.") -> dict:
    return {
        "prompt": prompt,
        "messages": None,
        "domain": "reasoning",
        "config": {"recurrent_steps": 4, "branch_count": 2},
        "budget": {"estimated_flops": 10**12},
        "runtime_controls": None,
        "cognitive_context": [
            {
                "source": "public-task",
                "text": "Public evidence only.",
                "instruction_authority": False,
            }
        ],
    }


MODEL_IDENTITY = {
    "schema": "test.model_identity.v1",
    "checkpoint_sha256": "1" * 64,
    "logical_parameter_count": 32_763_876_352,
}
EXECUTION_IDENTITY = {
    "schema": "test.execution_identity.v1",
    "runtime_bundle_sha256": "2" * 64,
    "schedule_sha256": "3" * 64,
}
RUNTIME_IDENTITY = {
    "schema": "aura.latent_cortex.runtime_identity.v1",
    "identity_bound": True,
    "launch_mode": "direct",
    "installed_app_required": False,
    "installed_app_verified": False,
    "source_verified": True,
    "source_root": "/sealed/aura",
    "source_commit": "4" * 40,
    "source_branch": "main",
    "workspace_state_sha256": "5" * 64,
    "source_dirty": False,
    "source_change_count": 0,
    "shell_assets_sha256": "6" * 64,
    "bundle_identifier": "",
    "app_executable_sha256": "",
    "launch_manifest_sha256": "",
    "issues": [],
}
CAMPAIGN_DESIGN_SHA256 = "d" * 64
WORKER_BOOT_ID = "4" * 32


def _build_request(
    policy,
    role_keys,
    worker_key,
    *,
    pair_id: str = "pair-0001",
    task_id: str = "task-0001",
    latent_request: dict | None = None,
    private_state: dict | None = None,
    supervisor_key: Ed25519PrivateKey = SUPERVISOR_KEY,
    expected_supervisor_public_key=None,
    challenge_lifetime_s: int = 600,
    worker_origin_binding_override: dict | None = None,
):
    state = private_state or _private_state()
    authority = build_worker_capture_launch_authority(
        issued_at_unix=NOW - 1,
        lifetime_s=challenge_lifetime_s,
        private_key=supervisor_key,
        challenge_nonce=hashlib.sha256(b"spark-051:worker-launch").digest(),
        challenge_id=hashlib.sha256(b"spark-051:worker-launch").hexdigest()[:32],
    )
    worker_identity = build_worker_capture_identity(
        worker_boot_id=WORKER_BOOT_ID,
        worker_pid=4242,
        private_key=worker_key,
        launch_challenge=authority.challenge,
        now_unix=NOW,
    )
    origin_binding = build_worker_capture_origin_binding(
        authority,
        worker_identity.public_identity,
        attested_at_unix=NOW,
        expected_worker_pid=4242,
    )
    if worker_origin_binding_override is not None:
        origin_binding = worker_origin_binding_override
    request = build_action_state_capture_request(
        policy=policy,
        runner_private_key=role_keys[CAMPAIGN_RUNNER],
        signed_at_unix=NOW,
        expected_supervisor_public_key=(
            expected_supervisor_public_key or SUPERVISOR_PUBLIC_KEY
        ),
        capture_id="a" * 32,
        capture_not_after_unix=NOW + 500,
        campaign_design_sha256=CAMPAIGN_DESIGN_SHA256,
        pair_id=pair_id,
        task_id=task_id,
        task_payload_sha256="7" * 64,
        action="falsify",
        model_identity=MODEL_IDENTITY,
        model_weights_identity_sha256="8" * 64,
        execution_identity=EXECUTION_IDENTITY,
        calibration_bucket="reasoning|falsify|medium",
        bucket_classifier_sha256="9" * 64,
        bucket_evidence_sha256="b" * 64,
        latent_reason_request=latent_request or _latent_request(),
        runner_durable_state_commitment_sha256=_state_sha(state["durable_state"]),
        runner_rng_root_commitment_sha256=_state_sha(state["rng_state"]),
        worker_origin_binding=origin_binding,
    )
    return request


def _private_state(tag: str = "A") -> dict:
    return {
        "branch_state": f"PRIVATE-BRANCH-{tag}".encode(),
        "durable_state": {
            "conversation_epoch": 17,
            "private_marker": f"PRIVATE-DURABLE-{tag}",
        },
        "evidence_state": {
            "secret_evidence": f"PRIVATE-EVIDENCE-{tag}",
            "score": 0.75,
        },
        "kv_cache": f"PRIVATE-KV-{tag}".encode() * 4,
        "latent_slots": bytearray(f"PRIVATE-LATENT-{tag}".encode()),
        "memory_state": [f"PRIVATE-MEMORY-{tag}", {"turn": 7}],
        "public_action_state": {
            "candidate": "falsify",
            "private_marker": f"PRIVATE-ACTION-{tag}",
        },
        "rng_state": f"PRIVATE-RNG-{tag}".encode(),
    }


def _case(tmp_path: Path, *, tag: str = "A"):
    root, root_pem, role_keys, policy = _trust_fixture()
    worker_key = _private(f"worker-{tag}")
    latent_request = _latent_request()
    private_state = _private_state(tag)
    request = _build_request(
        policy,
        role_keys,
        worker_key,
        latent_request=latent_request,
        private_state=private_state,
    )
    admission = admit_action_state_capture_request(
        request,
        trusted_root_public_key_pem=root_pem,
        expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
        current_policy_document=policy.document,
        now_unix=NOW,
    )
    custodian = _custodian()
    store = PrivateActionSnapshotStore(
        tmp_path / f"private-store-{tag}",
        key_custodian=custodian,
    )
    publication = store.publish(admission, private_state, created_at_unix=NOW + 1)
    receipt = build_action_state_capture_receipt(
        admission=admission,
        publication=publication,
        worker_private_key=worker_key,
        captured_at_unix=NOW + 1,
        latent_reason_request=latent_request,
        model_identity=MODEL_IDENTITY,
        execution_identity=EXECUTION_IDENTITY,
        runtime_identity=RUNTIME_IDENTITY,
        episode_step=3,
        schedule_step=0,
        branch_id="branch-0",
        layer_index=47,
        kv_position=128,
    )
    return {
        "root": root,
        "root_pem": root_pem,
        "role_keys": role_keys,
        "policy": policy,
        "worker_key": worker_key,
        "supervisor_public_key": SUPERVISOR_PUBLIC_KEY,
        "latent_request": latent_request,
        "request": request,
        "admission": admission,
        "custodian": custodian,
        "store": store,
        "publication": publication,
        "receipt": receipt,
    }


def _unpublished_case(tmp_path: Path, *, tag: str) -> dict:
    _, root_pem, role_keys, policy = _trust_fixture()
    worker_key = _private(f"worker-{tag}")
    private_state = _private_state(tag)
    request = _build_request(
        policy,
        role_keys,
        worker_key,
        private_state=private_state,
    )
    admission = admit_action_state_capture_request(
        request,
        trusted_root_public_key_pem=root_pem,
        expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
        current_policy_document=policy.document,
        now_unix=NOW,
    )
    custodian = _custodian()
    return {
        "admission": admission,
        "custodian": custodian,
        "private_state": private_state,
        "store": PrivateActionSnapshotStore(
            tmp_path / f"private-store-{tag}",
            key_custodian=custodian,
        ),
    }


def _validate(case, receipt=None, *, latent_request=None, publication=None):
    return validate_action_state_capture_receipt(
        receipt or case["receipt"],
        request=case["request"],
        publication=publication or case["publication"],
        trusted_root_public_key_pem=case["root_pem"],
        expected_supervisor_public_key=case["supervisor_public_key"],
        latent_reason_request=latent_request or case["latent_request"],
        model_identity=MODEL_IDENTITY,
        execution_identity=EXECUTION_IDENTITY,
        runtime_identity=RUNTIME_IDENTITY,
        expected_campaign_design_sha256=CAMPAIGN_DESIGN_SHA256,
    )


def _restore(store, handle, admission, *, arm: str, restored_at_unix: int):
    def apply_state(state: dict) -> str:
        components = {f"{name}_sha256": _state_sha(value) for name, value in sorted(state.items())}
        return _sha(components)

    return store.restore_and_apply(
        handle,
        admission,
        arm=arm,
        restored_at_unix=restored_at_unix,
        apply_state=apply_state,
    )


def test_portable_resident_components_stream_through_encrypted_store(tmp_path):
    raw_state = _private_state("PORTABLE")
    private_state = {
        name: PortableStateComponent.from_value(value) for name, value in raw_state.items()
    }
    _, root_pem, role_keys, policy = _trust_fixture()
    worker_key = _private("worker-portable")
    request = _build_request(
        policy,
        role_keys,
        worker_key,
        private_state=private_state,
    )
    admission = admit_action_state_capture_request(
        request,
        trusted_root_public_key_pem=root_pem,
        expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
        current_policy_document=policy.document,
        now_unix=NOW,
    )
    store = PrivateActionSnapshotStore(
        tmp_path / "portable-resident-store",
        key_custodian=_custodian(),
    )
    publication = store.publish(admission, private_state, created_at_unix=NOW + 1)

    applied: list[dict] = []

    def apply_state(state: dict) -> str:
        applied.append(state)
        return _sha(
            {
                f"{name}_sha256": value.sha256()
                for name, value in sorted(state.items())
            }
        )

    restored = store.restore_and_apply(
        publication.handle,
        admission,
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
        apply_state=apply_state,
    )
    assert restored.receipt["post_apply_state_sha256"] == _sha(
        publication.state_components
    )
    assert len(applied) == 1
    for name in private_state:
        assert isinstance(applied[0][name], PortableStateComponent)
        assert applied[0][name].to_bytes() == private_state[name].to_bytes()


def test_real_qwen_continuation_survives_encrypted_snapshot_round_trip(tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=8,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=64,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    engine = LatentCortexEngine(
        model,
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=2, seed=3),
            recurrence=RecurrenceConfig(max_steps=2, min_steps=1),
            branches=BranchConfig(n_branches=1, isolation_steps=1),
            prelude_frac=0.25,
            coda_frac=0.25,
            decode_max_tokens=2,
            allow_vanilla_fallback=False,
        ),
    )
    captured = []
    result = engine.reason(
        token_ids=[1, 2, 3, 4],
        action_continuation_capture=captured.append,
        action_continuation_runner_state={
            "durable_state": {"conversation_epoch": 19},
            "rng_state": {"root_seed": 104_729},
        },
        action_continuation_capture_only=True,
    )
    assert result.ok is True and len(captured) == 1
    continuation = captured[0]

    _, root_pem, role_keys, policy = _trust_fixture()
    worker_key = _private("worker-real-continuation")
    request = _build_request(
        policy,
        role_keys,
        worker_key,
        private_state=continuation.private_state,
    )
    admission = admit_action_state_capture_request(
        request,
        trusted_root_public_key_pem=root_pem,
        expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
        current_policy_document=policy.document,
        now_unix=NOW,
    )
    store = PrivateActionSnapshotStore(
        tmp_path / "real-qwen-continuation-store",
        key_custodian=_custodian(),
    )
    publication = store.publish(
        admission,
        continuation.private_state,
        created_at_unix=NOW + 1,
    )
    restored = store.restore_and_apply(
        publication.handle,
        admission,
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
        apply_state=lambda state: _sha(
            {
                f"{name}_sha256": value.sha256()
                for name, value in sorted(state.items())
            }
        ),
    )
    assert restored.receipt["state_components"] == continuation.state_components
    for name, value in restored.state.items():
        assert value.to_bytes() == continuation.private_state[name].to_bytes()


def _restore_until_sigkill(store_root, handle, admission, started, custodian) -> None:
    store = PrivateActionSnapshotStore(store_root, key_custodian=custodian)

    def begin_application(_state: dict) -> str:
        started.set()
        signal.pause()
        return "f" * 64

    store.restore_and_apply(
        handle,
        admission,
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
        apply_state=begin_application,
    )


def _race_same_arm_restore(
    store_root, handle, admission, ready, start, results, custodian
) -> None:
    store = PrivateActionSnapshotStore(store_root, key_custodian=custodian)
    ready.put(os.getpid())
    start.wait(timeout=10.0)
    try:
        restored = _restore(
            store,
            handle,
            admission,
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 2,
        )
        results.put(("restored", restored.receipt["operation_id"]))
    except ActionStateCaptureError as exc:
        results.put(("rejected", str(exc)))
    finally:
        store.close()


def _publish_until_sigkill(
    store_root, admission, private_state, started, custodian
) -> None:
    store = PrivateActionSnapshotStore(store_root, key_custodian=custodian)

    def pause_before_commit(name: str) -> None:
        if name == "publish_before_bundle_commit":
            started.set()
            signal.pause()

    capture_module._crash_boundary = pause_before_commit
    try:
        store.publish(admission, private_state, created_at_unix=NOW + 1)
    finally:
        store.close()


def _race_same_capture_publish(
    store_root,
    admission,
    private_state,
    ready,
    start,
    results,
    custodian,
) -> None:
    store = PrivateActionSnapshotStore(store_root, key_custodian=custodian)
    ready.put(os.getpid())
    start.wait(timeout=10.0)
    try:
        publication = store.publish(admission, private_state, created_at_unix=NOW + 1)
        results.put(("published", publication.handle, publication.snapshot_sha256))
    except Exception as exc:  # noqa: BLE001 - preserve subprocess failure evidence
        results.put(("failed", type(exc).__name__, str(exc)))
    finally:
        store.close()


def _resign_receipt(receipt: dict, worker_key: Ed25519PrivateKey) -> dict:
    body = {
        name: item
        for name, item in receipt.items()
        if name not in {"worker_origin", "receipt_sha256"}
    }
    raw = _public_raw(worker_key)
    signed = canonical_json_bytes(body)
    origin_body = {
        "schema": "aura.rlc.action_state_capture.worker_origin.v1",
        "algorithm": "Ed25519",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": hashlib.sha256(raw).hexdigest(),
        "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        "signature_b64": base64.b64encode(worker_key.sign(signed)).decode("ascii"),
    }
    origin = {**origin_body, "origin_sha256": _sha(origin_body)}
    complete = {**body, "worker_origin": origin}
    return {**complete, "receipt_sha256": _sha(complete)}


def test_honest_round_trip_lifecycle_and_public_nonleakage(tmp_path: Path):
    case = _case(tmp_path)
    assert _validate(case) == case["receipt"]
    request_text = json.dumps(case["request"], sort_keys=True)
    receipt_text = json.dumps(case["receipt"], sort_keys=True)
    for secret in (
        "Find the invariant.",
        "PRIVATE-BRANCH-A",
        "PRIVATE-EVIDENCE-A",
        "PRIVATE-KV-A",
        "PRIVATE-LATENT-A",
        "PRIVATE-MEMORY-A",
        case["publication"].handle,
    ):
        assert secret not in request_text
        assert secret not in receipt_text
    assert "model_weights" not in str(_private_state()).lower()
    assert case["receipt"]["component_observation_owners"] == {
        "branch_state_sha256": "resident_worker_measured_before_first_action_opportunity",
        "durable_state_sha256": ("runner_supplied_worker_commitment_verified_and_snapshotted"),
        "evidence_state_sha256": "resident_worker_measured_before_first_action_opportunity",
        "kv_cache_sha256": "resident_worker_measured_before_first_action_opportunity",
        "latent_slots_sha256": "resident_worker_measured_before_first_action_opportunity",
        "memory_state_sha256": "resident_worker_measured_before_first_action_opportunity",
        "public_action_state_sha256": ("resident_worker_measured_before_first_action_opportunity"),
        "rng_state_sha256": "runner_supplied_worker_commitment_verified_and_snapshotted",
    }


def test_independent_public_receipt_replay_needs_no_private_handle(tmp_path: Path):
    case = _case(tmp_path)

    assert (
        validate_action_state_capture_receipt_public(
            case["receipt"],
            request=case["request"],
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
            latent_reason_request=case["latent_request"],
            model_identity=MODEL_IDENTITY,
            execution_identity=EXECUTION_IDENTITY,
            runtime_identity=RUNTIME_IDENTITY,
            expected_campaign_design_sha256=CAMPAIGN_DESIGN_SHA256,
        )
        == case["receipt"]
    )

    attacked = deepcopy(case["receipt"])
    attacked["private_snapshot_envelope_sha256"] = "9" * 64
    with pytest.raises(ActionStateCaptureError, match="receipt_hash_mismatch"):
        validate_action_state_capture_receipt_public(
            attacked,
            request=case["request"],
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
            latent_reason_request=case["latent_request"],
            model_identity=MODEL_IDENTITY,
            execution_identity=EXECUTION_IDENTITY,
            runtime_identity=RUNTIME_IDENTITY,
            expected_campaign_design_sha256=CAMPAIGN_DESIGN_SHA256,
        )
    with pytest.raises(ActionStateCaptureError, match="receipt_request_mismatch"):
        validate_action_state_capture_receipt_public(
            case["receipt"],
            request=case["request"],
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
            latent_reason_request=case["latent_request"],
            model_identity=MODEL_IDENTITY,
            execution_identity=EXECUTION_IDENTITY,
            runtime_identity=RUNTIME_IDENTITY,
            expected_campaign_design_sha256="e" * 64,
        )
    with pytest.raises(ActionStateCaptureError, match="runtime_identity_unbound"):
        validate_action_state_capture_receipt_public(
            case["receipt"],
            request=case["request"],
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
            latent_reason_request=case["latent_request"],
            model_identity=MODEL_IDENTITY,
            execution_identity=EXECUTION_IDENTITY,
            runtime_identity={**RUNTIME_IDENTITY, "identity_bound": False},
            expected_campaign_design_sha256=CAMPAIGN_DESIGN_SHA256,
        )

    handle_hash = hashlib.sha256(
        case["publication"].handle.encode("ascii")
    ).hexdigest()
    paths = case["store"]._paths_for_handle_hash(handle_hash)
    snapshot_dir = paths["snapshot"]
    key_path = paths["key"]
    assert key_path.exists()
    assert key_path.read_bytes() != bytes.fromhex(case["publication"].handle[5:])
    ciphertext = b"".join(path.read_bytes() for path in sorted(snapshot_dir.glob("chunks/*/*.bin")))
    for secret in (
        b"PRIVATE-BRANCH-A",
        b"PRIVATE-EVIDENCE-A",
        b"PRIVATE-KV-A",
        b"PRIVATE-LATENT-A",
        b"PRIVATE-MEMORY-A",
    ):
        assert secret not in ciphertext

    treatment = _restore(
        case["store"],
        case["publication"].handle,
        case["admission"],
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
    )
    control = _restore(
        case["store"],
        case["publication"].handle,
        case["admission"],
        arm=CONTROL_ARM,
        restored_at_unix=NOW + 3,
    )
    assert treatment.state == control.state
    assert treatment.state["branch_state"] == b"PRIVATE-BRANCH-A"
    assert treatment.state["evidence_state"]["secret_evidence"] == ("PRIVATE-EVIDENCE-A")
    assert treatment.receipt["all_bytes_verified_before_return"] is True

    seal = case["store"].seal(
        case["publication"].handle,
        case["admission"],
        sealed_at_unix=NOW + 4,
    )
    assert seal["both_arms_used_exactly_once"] is True
    erasure = case["store"].erase(
        case["publication"].handle,
        case["admission"],
        erased_at_unix=NOW + 5,
    )
    assert erasure["all_snapshot_files_absent"] is True
    assert erasure["cryptographic_key_destroyed"] is True
    assert erasure["ciphertext_namespace_deleted"] is True
    assert not key_path.exists()
    assert not snapshot_dir.exists()
    with pytest.raises(ActionStateCaptureError, match="private_snapshot_erased"):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 6,
        )


def _runtime_case(tmp_path: Path) -> dict:
    _root, root_pem, role_keys, policy = _trust_fixture()
    worker_key = _private("runtime-worker")
    draft = _build_request(policy, role_keys, worker_key)
    runner_state = runner_state_for_capture_payload(draft["request_payload"])
    private_state = {
        name: PortableStateComponent.from_value(value)
        for name, value in {
            "branch_state": {"branch": 1},
            "durable_state": runner_state["durable_state"],
            "evidence_state": {"evidence": "bounded"},
            "kv_cache": {"layers": [1, 2]},
            "latent_slots": {"slot": [0.25, 0.75]},
            "memory_state": [{"memory": "present"}],
            "public_action_state": {"candidate": "falsify"},
            "rng_state": runner_state["rng_state"],
        }.items()
    }
    request = _build_request(
        policy,
        role_keys,
        worker_key,
        private_state=private_state,
    )
    origin = request["request_payload"]["worker_origin_binding"]
    frame = build_action_state_runtime_frame(
        mode="capture",
        capture_request=request,
        trusted_root_public_key_pem=root_pem,
        current_policy_document=policy.document,
        model_identity=MODEL_IDENTITY,
        execution_identity=EXECUTION_IDENTITY,
        latent_reason_request=_latent_request(),
    )
    admitted = admit_action_state_runtime(
        frame,
        worker_launch_challenge=origin["launch_challenge"],
        now_unix=NOW,
    )
    store = PrivateActionSnapshotStore(
        tmp_path / "runtime-store",
        key_custodian=_custodian(),
    )
    publication = store.publish(
        admitted.admission,
        private_state,
        created_at_unix=NOW + 1,
    )
    capture_receipt = build_action_state_capture_receipt(
        admission=admitted.admission,
        publication=publication,
        worker_private_key=worker_key,
        captured_at_unix=NOW + 1,
        latent_reason_request=_latent_request(),
        model_identity=MODEL_IDENTITY,
        execution_identity=EXECUTION_IDENTITY,
        runtime_identity=RUNTIME_IDENTITY,
        episode_step=0,
        schedule_step=0,
        branch_id="branch-0",
        layer_index=7,
        kv_position=32,
    )
    return {
        "root_pem": root_pem,
        "policy": policy,
        "worker_key": worker_key,
        "worker_public_key_b64": origin["worker_identity"]["public_key_b64"],
        "request": request,
        "frame": frame,
        "admitted": admitted,
        "store": store,
        "publication": publication,
        "capture_receipt": capture_receipt,
    }


def _runtime_intervention(arm: str) -> dict:
    return {
        "intervention_sha256": hashlib.sha256(arm.encode()).hexdigest(),
        "authority_payload": {"arm": arm},
    }


def test_runtime_frame_reconstructs_runner_roots_without_transporting_them(
    tmp_path: Path,
):
    case = _runtime_case(tmp_path)

    assert case["admitted"].runner_state == runner_state_for_capture_payload(
        case["request"]["request_payload"]
    )
    assert "runner_state" not in json.dumps(case["frame"], sort_keys=True)
    recovered = case["store"].publication_for_request(case["admitted"].admission)
    assert recovered.handle == case["publication"].handle
    assert recovered.state_components == case["publication"].state_components


def test_restore_rotates_parent_and_worker_without_retrusting_capture(
    tmp_path: Path,
):
    _root, root_pem, role_keys, policy = _trust_fixture()
    capture_worker = _private("expired-capture-worker")
    draft = _build_request(
        policy,
        role_keys,
        capture_worker,
        challenge_lifetime_s=10,
    )
    runner_state = runner_state_for_capture_payload(draft["request_payload"])
    state = {
        name: PortableStateComponent.from_value(value)
        for name, value in {
            "branch_state": {"branch": 1},
            "durable_state": runner_state["durable_state"],
            "evidence_state": {},
            "kv_cache": {"layers": []},
            "latent_slots": {"slot": 1},
            "memory_state": [],
            "public_action_state": {"candidate": "falsify"},
            "rng_state": runner_state["rng_state"],
        }.items()
    }
    request = _build_request(
        policy,
        role_keys,
        capture_worker,
        private_state=state,
        challenge_lifetime_s=10,
    )
    capture_admission = admit_action_state_capture_request(
        request,
        trusted_root_public_key_pem=root_pem,
        expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
        current_policy_document=policy.document,
        now_unix=NOW,
    )
    store = PrivateActionSnapshotStore(
        tmp_path / "rotation-store",
        key_custodian=_custodian(),
    )
    publication = store.publish(
        capture_admission,
        state,
        created_at_unix=NOW + 1,
    )
    receipt = build_action_state_capture_receipt(
        admission=capture_admission,
        publication=publication,
        worker_private_key=capture_worker,
        captured_at_unix=NOW + 1,
        latent_reason_request=_latent_request(),
        model_identity=MODEL_IDENTITY,
        execution_identity=EXECUTION_IDENTITY,
        runtime_identity=RUNTIME_IDENTITY,
        episode_step=0,
        schedule_step=0,
        branch_id="branch-0",
        layer_index=7,
        kv_position=32,
    )

    replacement_supervisor = _private("replacement-supervisor")
    replacement_authority = build_worker_capture_launch_authority(
        issued_at_unix=NOW + 19,
        lifetime_s=100,
        private_key=replacement_supervisor,
        challenge_nonce=hashlib.sha256(b"replacement-challenge").digest(),
        challenge_id=hashlib.sha256(b"replacement-challenge").hexdigest()[:32],
    )
    replacement_worker = build_worker_capture_identity(
        worker_boot_id="e" * 32,
        worker_pid=5252,
        private_key=_private("replacement-worker"),
        launch_challenge=replacement_authority.challenge,
        now_unix=NOW + 20,
    )
    replacement_binding = build_worker_capture_origin_binding(
        replacement_authority,
        replacement_worker.public_identity,
        attested_at_unix=NOW + 20,
        expected_worker_pid=5252,
    )
    frame = build_action_state_runtime_frame(
        mode="restore",
        capture_request=request,
        trusted_root_public_key_pem=root_pem,
        current_policy_document=policy.document,
        model_identity=MODEL_IDENTITY,
        execution_identity=EXECUTION_IDENTITY,
        latent_reason_request=_latent_request(),
        resident_worker_origin_binding=replacement_binding,
        capture_receipt=receipt,
        arm=TREATMENT_ARM,
    )

    admitted = admit_action_state_runtime(
        frame,
        worker_launch_challenge=replacement_authority.challenge,
        now_unix=NOW + 20,
    )
    assert admitted.admission.request_sha256 == capture_admission.request_sha256
    assert (
        admitted.resident_worker_origin_binding["worker_identity"]
        == replacement_worker.public_identity
    )
    assert admitted.capture_supervisor_public_key == _public_raw(SUPERVISOR_KEY)
    assert admitted.resident_supervisor_public_key == _public_raw(
        replacement_supervisor
    )
    validate_action_state_capture_receipt(
        receipt,
        request=admitted.admission.request,
        publication=publication,
        trusted_root_public_key_pem=root_pem,
        expected_supervisor_public_key=admitted.capture_supervisor_public_key,
        latent_reason_request=_latent_request(),
        model_identity=MODEL_IDENTITY,
        execution_identity=EXECUTION_IDENTITY,
        runtime_identity=RUNTIME_IDENTITY,
        expected_campaign_design_sha256=admitted.admission.payload[
            "campaign_design_sha256"
        ],
    )
    with pytest.raises(ActionStateCaptureError, match="worker_origin_binding"):
        validate_action_state_capture_receipt(
            receipt,
            request=admitted.admission.request,
            publication=publication,
            trusted_root_public_key_pem=root_pem,
            expected_supervisor_public_key=admitted.resident_supervisor_public_key,
            latent_reason_request=_latent_request(),
            model_identity=MODEL_IDENTITY,
            execution_identity=EXECUTION_IDENTITY,
            runtime_identity=RUNTIME_IDENTITY,
            expected_campaign_design_sha256=admitted.admission.payload[
                "campaign_design_sha256"
            ],
        )


def test_runtime_pair_is_signed_once_only_and_erased_after_second_arm(
    tmp_path: Path,
):
    case = _runtime_case(tmp_path)
    receipts: dict[str, dict] = {}
    interventions = {
        arm: _runtime_intervention(arm) for arm in (TREATMENT_ARM, CONTROL_ARM)
    }

    for index, arm in enumerate((TREATMENT_ARM, CONTROL_ARM), start=2):
        restore = case["store"].restore_and_apply(
            case["publication"].handle,
            case["admitted"].admission,
            arm=arm,
            restored_at_unix=NOW + index,
            apply_state=lambda state: _sha(
                {
                    f"{name}_sha256": value.sha256()
                    for name, value in state.items()
                }
            ),
        )
        lifecycle: dict[str, object] = {"complete": False}
        if arm == CONTROL_ARM:
            seal = case["store"].seal(
                case["publication"].handle,
                case["admitted"].admission,
                sealed_at_unix=NOW + 5,
            )
            erasure = case["store"].erase(
                case["publication"].handle,
                case["admitted"].admission,
                erased_at_unix=NOW + 6,
            )
            lifecycle = {
                "complete": True,
                "seal_receipt": seal,
                "erasure_receipt": erasure,
            }
        receipts[arm] = build_action_state_restore_receipt(
            capture_receipt=case["capture_receipt"],
            custody_restore_receipt=restore.receipt,
            action_intervention=interventions[arm],
            runtime_identity=RUNTIME_IDENTITY,
            worker_private_key=case["worker_key"],
            custody_lifecycle_receipts=lifecycle,
            resident_worker_origin_binding=case["request"]["request_payload"][
                "worker_origin_binding"
            ],
        )
        validate_action_state_restore_receipt(
            receipts[arm],
            capture_receipt=case["capture_receipt"],
            action_intervention=interventions[arm],
            runtime_identity=RUNTIME_IDENTITY,
            expected_worker_public_key_b64=case["worker_public_key_b64"],
            expected_supervisor_public_key=_public_raw(SUPERVISOR_KEY),
        )

    verdict = verify_action_state_pair_evidence(
        capture_receipt=case["capture_receipt"],
        restore_receipts=receipts,
        interventions=interventions,
        runtime_identity=RUNTIME_IDENTITY,
        expected_worker_public_keys_b64={
            arm: case["worker_public_key_b64"]
            for arm in (TREATMENT_ARM, CONTROL_ARM)
        },
        expected_supervisor_public_keys={
            arm: _public_raw(SUPERVISOR_KEY)
            for arm in (TREATMENT_ARM, CONTROL_ARM)
        },
    )
    assert verdict["both_arms_restored_once"] is True
    assert verdict["terminal_erasure_verified"] is True
    assert_public_runtime_result(
        {
            "action_state_capture_receipt": case["capture_receipt"],
            "action_state_restore_receipts": receipts,
            "verification": verdict,
        }
    )
    with pytest.raises(ActionStateCaptureError, match="publication_not_found"):
        case["store"].publication_for_request(case["admitted"].admission)


def test_runtime_rejects_binding_signature_and_private_transport_tampering(
    tmp_path: Path,
):
    case = _runtime_case(tmp_path)
    attacked = deepcopy(case["frame"])
    attacked["model_identity"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ActionStateRuntimeError, match="public_binding_mismatch"):
        admit_action_state_runtime(
            attacked,
            worker_launch_challenge=case["request"]["request_payload"][
                "worker_origin_binding"
            ]["launch_challenge"],
            now_unix=NOW,
        )

    with pytest.raises(ActionStateRuntimeError, match="private_material_leaked"):
        assert_public_runtime_result({"opaque": case["publication"].handle})

    intervention = _runtime_intervention(TREATMENT_ARM)
    restore = case["store"].restore_and_apply(
        case["publication"].handle,
        case["admitted"].admission,
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
        apply_state=lambda state: _sha(
            {
                f"{name}_sha256": value.sha256()
                for name, value in state.items()
            }
        ),
    )
    signed = build_action_state_restore_receipt(
        capture_receipt=case["capture_receipt"],
        custody_restore_receipt=restore.receipt,
        action_intervention=intervention,
        runtime_identity=RUNTIME_IDENTITY,
        worker_private_key=case["worker_key"],
        custody_lifecycle_receipts={"complete": False},
        resident_worker_origin_binding=case["request"]["request_payload"][
            "worker_origin_binding"
        ],
    )
    signed["worker_origin"]["signature_b64"] = base64.b64encode(b"x" * 64).decode()
    signed_body = {name: item for name, item in signed.items() if name != "receipt_sha256"}
    signed["receipt_sha256"] = _sha(signed_body)
    with pytest.raises(ActionStateRuntimeError, match="signature_invalid"):
        validate_action_state_restore_receipt(
            signed,
            capture_receipt=case["capture_receipt"],
            action_intervention=intervention,
            runtime_identity=RUNTIME_IDENTITY,
            expected_worker_public_key_b64=case["worker_public_key_b64"],
            expected_supervisor_public_key=_public_raw(SUPERVISOR_KEY),
        )


def test_continuation_reconstruction_binds_public_opportunity(tmp_path: Path):
    case = _runtime_case(tmp_path)
    restore = case["store"].restore_and_apply(
        case["publication"].handle,
        case["admitted"].admission,
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
        apply_state=lambda state: case["capture_receipt"]["state_sha256"],
    )
    continuation = continuation_from_private_state(
        restore.state,
        case["capture_receipt"],
    )
    assert continuation.state_components == case["capture_receipt"]["state_components"]
    assert continuation.branch_id == "branch-0"


def test_state_mismatch_quarantines_ambiguous_application_instead_of_retrying(
    tmp_path: Path,
):
    case = _case(tmp_path)

    with pytest.raises(UnknownActionStateApplicationError) as quarantined:
        case["store"].restore_and_apply(
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 2,
            apply_state=lambda _state: "f" * 64,
        )
    assert quarantined.value.quarantine_evidence == {
        "state": "UNKNOWN_APPLICATION",
        "operation_id": quarantined.value.quarantine_evidence["operation_id"],
        "arm": TREATMENT_ARM,
        "worker_boot_id": WORKER_BOOT_ID,
        "worker_pid": 4242,
        "request_sha256": case["admission"].request_sha256,
        "snapshot_sha256": case["publication"].snapshot_sha256,
        "process_replacement_required": True,
        "same_process_retry_allowed": False,
    }
    assert len(quarantined.value.quarantine_evidence["operation_id"]) == 32
    with pytest.raises(
        ActionStateCaptureError,
        match="unknown_application_process_replacement_required",
    ):
        case["store"].recover(case["publication"].handle, case["admission"])
    with pytest.raises(
        ActionStateCaptureError,
        match="unknown_application_process_replacement_required",
    ):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 2,
        )


def test_crash_after_state_apply_quarantines_worker_and_arm(
    tmp_path: Path,
    monkeypatch,
):
    case = _case(tmp_path)
    applied = 0

    def crash_after_apply(name: str):
        if name == "restore_after_state_apply":
            raise RuntimeError("simulated death after state installation")

    def apply_state(state: dict) -> str:
        nonlocal applied
        applied += 1
        return _sha({f"{name}_sha256": _state_sha(value) for name, value in sorted(state.items())})

    monkeypatch.setattr(capture_module, "_crash_boundary", crash_after_apply)
    with pytest.raises(UnknownActionStateApplicationError) as quarantined:
        case["store"].restore_and_apply(
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 2,
            apply_state=apply_state,
        )
    assert applied == 1
    assert isinstance(quarantined.value.__cause__, RuntimeError)
    assert "after state installation" in str(quarantined.value.__cause__)

    monkeypatch.setattr(capture_module, "_crash_boundary", lambda _name: None)
    handle_hash = hashlib.sha256(case["publication"].handle.encode("ascii")).hexdigest()
    operation_path = case["store"]._paths_for_handle_hash(handle_hash)["operation"]
    operation = json.loads(operation_path.read_bytes())
    assert operation["stage"] == "application_started"
    assert operation["worker_boot_id"] == WORKER_BOOT_ID
    assert operation["worker_pid"] == 4242
    with pytest.raises(
        ActionStateCaptureError,
        match="unknown_application_process_replacement_required",
    ):
        case["store"].recover(case["publication"].handle, case["admission"])


def test_crash_after_durable_start_marker_quarantines_even_before_callback(
    tmp_path: Path,
    monkeypatch,
):
    case = _case(tmp_path)
    apply_calls = 0

    def crash_after_start(name: str):
        if name == "restore_after_application_started":
            raise RuntimeError("simulated death after durable start")

    def apply_state(_state: dict) -> str:
        nonlocal apply_calls
        apply_calls += 1
        return "f" * 64

    monkeypatch.setattr(capture_module, "_crash_boundary", crash_after_start)
    with pytest.raises(UnknownActionStateApplicationError) as quarantined:
        case["store"].restore_and_apply(
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 2,
            apply_state=apply_state,
        )
    assert apply_calls == 0
    assert isinstance(quarantined.value.__cause__, RuntimeError)
    assert "after durable start" in str(quarantined.value.__cause__)
    monkeypatch.setattr(capture_module, "_crash_boundary", lambda _name: None)
    with pytest.raises(
        ActionStateCaptureError,
        match="unknown_application_process_replacement_required",
    ):
        case["store"].recover(case["publication"].handle, case["admission"])


def test_sigkill_inside_state_application_is_durably_quarantined(tmp_path: Path):
    case = _case(tmp_path)
    context = mp.get_context("spawn")
    started = context.Event()
    child = context.Process(
        target=_restore_until_sigkill,
        args=(
            case["store"].root,
            case["publication"].handle,
            case["admission"],
            started,
            case["custodian"],
        ),
    )
    child.start()
    try:
        assert started.wait(timeout=10.0)
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
        child.join(timeout=10.0)
        assert child.exitcode == -signal.SIGKILL
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5.0)

    with pytest.raises(UnknownActionStateApplicationError) as quarantined:
        case["store"].recover(case["publication"].handle, case["admission"])
    assert quarantined.value.quarantine_evidence["worker_pid"] == 4242
    assert quarantined.value.quarantine_evidence["same_process_retry_allowed"] is False


def test_concurrent_processes_cannot_consume_the_same_arm_twice(tmp_path: Path):
    case = _case(tmp_path, tag="ConcurrentArm")
    context = mp.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    children = [
        context.Process(
            target=_race_same_arm_restore,
            args=(
                case["store"].root,
                case["publication"].handle,
                case["admission"],
                ready,
                start,
                results,
                case["custodian"],
            ),
        )
        for _ in range(2)
    ]
    for child in children:
        child.start()
    try:
        assert {ready.get(timeout=10.0), ready.get(timeout=10.0)} == {
            child.pid for child in children
        }
        start.set()
        outcomes = [results.get(timeout=15.0), results.get(timeout=15.0)]
        for child in children:
            child.join(timeout=10.0)
            assert child.exitcode == 0
    finally:
        start.set()
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(timeout=5.0)

    assert sorted(kind for kind, _detail in outcomes) == ["rejected", "restored"]
    rejection = next(detail for kind, detail in outcomes if kind == "rejected")
    assert rejection == "private_snapshot_arm_already_used"


def test_legacy_uncommitted_restore_is_never_assumed_preapply(
    tmp_path: Path,
    monkeypatch,
):
    case = _case(tmp_path)

    def crash_after_prepare(name: str):
        if name == "restore_after_prepare":
            raise RuntimeError("leave a prepared v2 operation")

    monkeypatch.setattr(capture_module, "_crash_boundary", crash_after_prepare)
    with pytest.raises(RuntimeError, match="prepared v2"):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 2,
        )
    monkeypatch.setattr(capture_module, "_crash_boundary", lambda _name: None)

    handle_hash = hashlib.sha256(
        case["publication"].handle.encode("ascii")
    ).hexdigest()
    operation_path = case["store"]._paths_for_handle_hash(handle_hash)["operation"]
    operation = json.loads(operation_path.read_bytes())
    legacy_body = {
        name: value
        for name, value in operation.items()
        if name
        not in {
            "handle_authentication_sha256",
            "operation_sha256",
            "application_started_at_unix",
            "worker_boot_id",
            "worker_pid",
        }
    }
    legacy_body["schema"] = capture_module.PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA_V1
    legacy = case["store"]._operation_document(
        legacy_body,
        handle_secret=case["store"]._handle_secret(case["publication"].handle),
    )
    case["store"]._atomic_publish(
        operation_path,
        canonical_json_bytes(legacy) + b"\n",
        replace=True,
    )

    with pytest.raises(UnknownActionStateApplicationError) as quarantined:
        case["store"].recover(case["publication"].handle, case["admission"])
    assert quarantined.value.quarantine_evidence["worker_boot_id"] == ""
    assert quarantined.value.quarantine_evidence["worker_pid"] is None
    assert quarantined.value.quarantine_evidence["process_replacement_required"] is True


def test_private_snapshot_requires_exact_durable_and_rng_commitments(
    tmp_path: Path,
):
    _root, root_pem, role_keys, policy = _trust_fixture()
    worker_key = _private("worker-commitment")
    committed_state = _private_state("COMMITTED")
    request = _build_request(
        policy,
        role_keys,
        worker_key,
        private_state=committed_state,
    )
    admission = admit_action_state_capture_request(
        request,
        trusted_root_public_key_pem=root_pem,
        expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
        current_policy_document=policy.document,
        now_unix=NOW,
    )
    drifted_state = deepcopy(committed_state)
    drifted_state["durable_state"]["conversation_epoch"] += 1
    drifted_state["rng_state"] = b"DIFFERENT-PRIVATE-RNG"
    with pytest.raises(
        ActionStateCaptureError,
        match="private_snapshot_runner_state_commitment_mismatch",
    ):
        PrivateActionSnapshotStore(
            tmp_path / "commitment-store",
            key_custodian=_custodian(),
        ).publish(
            admission,
            drifted_state,
            created_at_unix=NOW + 1,
        )


@pytest.mark.parametrize(
    ("component_limit", "snapshot_limit", "expected_error"),
    [
        (8, 1_024, "private_snapshot_component_too_large"),
        (1_024, 20, "private_snapshot_too_large"),
    ],
)
def test_private_snapshot_buffering_limits_are_hard_admission_boundaries(
    tmp_path: Path,
    monkeypatch,
    component_limit: int,
    snapshot_limit: int,
    expected_error: str,
):
    _, root_pem, role_keys, policy = _trust_fixture()
    worker_key = _private(f"worker-{expected_error}")
    private_state = _private_state("LIMIT")
    request = _build_request(
        policy,
        role_keys,
        worker_key,
        private_state=private_state,
    )
    admission = admit_action_state_capture_request(
        request,
        trusted_root_public_key_pem=root_pem,
        expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
        current_policy_document=policy.document,
        now_unix=NOW,
    )
    monkeypatch.setattr(capture_module, "_MAX_COMPONENT_BYTES", component_limit)
    monkeypatch.setattr(capture_module, "_MAX_SNAPSHOT_BYTES", snapshot_limit)
    store = PrivateActionSnapshotStore(
        tmp_path / expected_error,
        key_custodian=_custodian(),
    )

    with pytest.raises(ActionStateCaptureError, match=expected_error):
        store.publish(admission, private_state, created_at_unix=NOW + 1)

    assert not any((store.root / "bundles").iterdir())
    assert not any((store.root / "transactions").iterdir())


def test_streaming_codec_bounds_chunks_for_binary_and_large_json_values(
    tmp_path: Path,
    monkeypatch,
):
    case = _unpublished_case(tmp_path, tag="Streaming")
    state = deepcopy(case["private_state"])
    state["branch_state"] = b"B" * 257
    state["memory_state"] = {"single_large_scalar": "J" * 257}
    request = case["admission"].payload
    state["durable_state"] = case["private_state"]["durable_state"]
    state["rng_state"] = case["private_state"]["rng_state"]
    emitted_sizes: list[int] = []
    original_stream = capture_module._state_value_stream

    def observed_stream(value):
        value_type, parts = original_stream(value)

        def observed_parts():
            for part in parts:
                emitted_sizes.append(len(part))
                yield part

        return value_type, observed_parts()

    monkeypatch.setattr(capture_module, "_CHUNK_BYTES", 64)
    monkeypatch.setattr(capture_module, "_state_value_stream", observed_stream)
    publication = case["store"].publish(
        case["admission"],
        state,
        created_at_unix=NOW + 1,
    )

    assert request["runner_durable_state_commitment_sha256"] == _state_sha(
        state["durable_state"]
    )
    assert emitted_sizes and max(emitted_sizes) <= 64
    handle_hash = hashlib.sha256(publication.handle.encode("ascii")).hexdigest()
    envelope_path = case["store"]._paths_for_handle_hash(handle_hash)["snapshot"] / "envelope.json"
    envelope = json.loads(envelope_path.read_bytes())
    by_name = {item["name"]: item for item in envelope["components"]}
    assert len(by_name["branch_state"]["chunks"]) > 4
    assert len(by_name["memory_state"]["chunks"]) > 4
    assert all(
        chunk["plaintext_byte_count"] <= 64
        for component in envelope["components"]
        for chunk in component["chunks"]
    )
    restored = _restore(
        case["store"],
        publication.handle,
        case["admission"],
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
    )
    assert restored.state == state


def test_bundle_contains_only_wrapped_dek_and_wrong_custodian_cannot_restore(
    tmp_path: Path,
    monkeypatch,
):
    case = _unpublished_case(tmp_path, tag="Custody")
    captured_data_keys: list[bytes] = []
    wrap_data_key = case["custodian"].wrap_data_key

    def capture_key(data_key: bytes, *, context_sha256: str):
        captured_data_keys.append(data_key)
        return wrap_data_key(data_key, context_sha256=context_sha256)

    monkeypatch.setattr(case["custodian"], "wrap_data_key", capture_key)
    publication = case["store"].publish(
        case["admission"],
        case["private_state"],
        created_at_unix=NOW + 1,
    )
    assert len(captured_data_keys) == 1
    raw_dek = captured_data_keys[0]
    handle_hash = hashlib.sha256(publication.handle.encode("ascii")).hexdigest()
    paths = case["store"]._paths_for_handle_hash(handle_hash)
    wrapped = json.loads(paths["key"].read_bytes())
    assert wrapped["schema"] == "aura.rlc.snapshot_key_custody.wrapped_dek.v1"
    assert wrapped["custody_identity"]["custody_class"] == "macos_keychain"
    assert all(raw_dek not in path.read_bytes() for path in paths["bundle"].rglob("*") if path.is_file())

    root = case["store"].root
    case["store"].close()
    wrong_store = PrivateActionSnapshotStore(
        root,
        key_custodian=_custodian(b"W" * 32),
    )
    with pytest.raises(ActionStateCaptureError, match="key_custody_failed"):
        _restore(
            wrong_store,
            publication.handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 2,
        )


def test_sigkill_before_bundle_commit_leaves_no_visible_snapshot_and_retry_recovers(
    tmp_path: Path,
):
    case = _unpublished_case(tmp_path, tag="PublishSigkill")
    store = case["store"]
    context = mp.get_context("spawn")
    started = context.Event()
    child = context.Process(
        target=_publish_until_sigkill,
        args=(
            store.root,
            case["admission"],
            case["private_state"],
            started,
            case["custodian"],
        ),
    )
    child.start()
    try:
        assert started.wait(timeout=15.0)
        assert child.pid is not None
        assert not any((store.root / "bundles").iterdir())
        assert any((store.root / "transactions").iterdir())
        os.kill(child.pid, signal.SIGKILL)
        child.join(timeout=10.0)
        assert child.exitcode == -signal.SIGKILL
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5.0)

    publication = store.publish(
        case["admission"],
        case["private_state"],
        created_at_unix=NOW + 1,
    )
    assert len(tuple((store.root / "bundles").iterdir())) == 1
    assert not any((store.root / "transactions").iterdir())
    restored = _restore(
        store,
        publication.handle,
        case["admission"],
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
    )
    assert restored.state == case["private_state"]


def test_post_commit_crash_retry_returns_the_committed_publication(
    tmp_path: Path,
    monkeypatch,
):
    case = _unpublished_case(tmp_path, tag="PublishPostCommit")
    store = case["store"]

    def crash_after_commit(name: str) -> None:
        if name == "publish_after_bundle_commit":
            raise RuntimeError("simulated post-commit process death")

    monkeypatch.setattr(capture_module, "_crash_boundary", crash_after_commit)
    with pytest.raises(RuntimeError, match="post-commit process death"):
        store.publish(
            case["admission"],
            case["private_state"],
            created_at_unix=NOW + 1,
        )
    committed_bundles = tuple((store.root / "bundles").iterdir())
    assert len(committed_bundles) == 1
    assert not any((store.root / "transactions").iterdir())

    monkeypatch.setattr(capture_module, "_crash_boundary", lambda _name: None)
    recovered = store.publish(
        case["admission"],
        case["private_state"],
        created_at_unix=NOW + 1,
    )
    assert hashlib.sha256(recovered.handle.encode("ascii")).hexdigest() == (
        committed_bundles[0].name
    )
    assert len(tuple((store.root / "bundles").iterdir())) == 1
    restored = _restore(
        store,
        recovered.handle,
        case["admission"],
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
    )
    assert restored.state == case["private_state"]


def test_disk_full_before_bundle_commit_rolls_back_the_entire_staging_tree(
    tmp_path: Path,
    monkeypatch,
):
    case = _unpublished_case(tmp_path, tag="PublishDiskFull")
    store = case["store"]
    atomic_publish = store._atomic_publish

    def fail_publication(path: Path, payload: bytes, *, replace: bool) -> None:
        if path.name == "publication.json":
            cause = OSError(28, "No space left on device")
            raise ActionStateCaptureError("private_snapshot_publish_failed") from cause
        atomic_publish(path, payload, replace=replace)

    monkeypatch.setattr(store, "_atomic_publish", fail_publication)
    with pytest.raises(
        ActionStateCaptureError,
        match="private_snapshot_publication_io_failed",
    ) as failed:
        store.publish(
            case["admission"],
            case["private_state"],
            created_at_unix=NOW + 1,
        )
    assert isinstance(failed.value.__cause__, OSError)
    assert failed.value.__cause__.errno == 28

    assert not any((store.root / "bundles").iterdir())
    assert not any((store.root / "transactions").iterdir())


def test_concurrent_same_request_publish_is_single_copy_and_idempotent(tmp_path: Path):
    case = _unpublished_case(tmp_path, tag="PublishRace")
    store = case["store"]
    context = mp.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    children = [
        context.Process(
            target=_race_same_capture_publish,
            args=(
                store.root,
                case["admission"],
                case["private_state"],
                ready,
                start,
                results,
                case["custodian"],
            ),
        )
        for _ in range(2)
    ]
    for child in children:
        child.start()
    try:
        assert {ready.get(timeout=10.0), ready.get(timeout=10.0)} == {
            child.pid for child in children
        }
        start.set()
        outcomes = [results.get(timeout=20.0), results.get(timeout=20.0)]
        for child in children:
            child.join(timeout=10.0)
            assert child.exitcode == 0
    finally:
        start.set()
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(timeout=5.0)

    assert {outcome[0] for outcome in outcomes} == {"published"}
    assert len({outcome[1] for outcome in outcomes}) == 1
    assert len({outcome[2] for outcome in outcomes}) == 1
    assert len(tuple((store.root / "bundles").iterdir())) == 1
    assert not any((store.root / "transactions").iterdir())


def test_private_snapshot_lifecycle_rejects_impossible_timestamp_order(
    tmp_path: Path,
):
    case = _case(tmp_path)
    with pytest.raises(ActionStateCaptureError, match="restore_time_invalid"):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW,
        )
    _restore(
        case["store"],
        case["publication"].handle,
        case["admission"],
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
    )
    _restore(
        case["store"],
        case["publication"].handle,
        case["admission"],
        arm=CONTROL_ARM,
        restored_at_unix=NOW + 3,
    )
    with pytest.raises(ActionStateCaptureError, match="seal_time_invalid"):
        case["store"].seal(
            case["publication"].handle,
            case["admission"],
            sealed_at_unix=NOW + 2,
        )
    case["store"].seal(
        case["publication"].handle,
        case["admission"],
        sealed_at_unix=NOW + 4,
    )
    with pytest.raises(ActionStateCaptureError, match="erasure_time_invalid"):
        case["store"].erase(
            case["publication"].handle,
            case["admission"],
            erased_at_unix=NOW + 3,
        )


def test_public_request_and_receipt_tamper_and_extra_fields_fail_closed(
    tmp_path: Path,
):
    case = _case(tmp_path)
    attacked = deepcopy(case["request"])
    attacked["request_payload"]["task_id"] = "different-task"
    body = {name: item for name, item in attacked.items() if name != "request_sha256"}
    attacked["request_sha256"] = _sha(body)
    with pytest.raises(ActionStateCaptureError):
        admit_action_state_capture_request(
            attacked,
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
            current_policy_document=case["policy"].document,
            now_unix=NOW,
        )

    attacked = deepcopy(case["request"])
    attacked["unexpected"] = True
    with pytest.raises(ActionStateCaptureError, match="request_fields"):
        replay_action_state_capture_request(
            attacked,
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
        )

    attacked = deepcopy(case["request"])
    attacked["request_payload"]["policy_revision"] = True
    attacked["runner_attestation"] = build_role_attestation(
        case["policy"],
        role=CAMPAIGN_RUNNER,
        payload=attacked["request_payload"],
        signed_at_unix=NOW,
        private_key=case["role_keys"][CAMPAIGN_RUNNER],
    )
    body = {name: item for name, item in attacked.items() if name != "request_sha256"}
    attacked["request_sha256"] = _sha(body)
    with pytest.raises(ActionStateCaptureError):
        admit_action_state_capture_request(
            attacked,
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
            current_policy_document=case["policy"].document,
            now_unix=NOW,
        )

    attacked = deepcopy(case["receipt"])
    attacked["runtime_identity_sha256"] = "0" * 64
    with pytest.raises(ActionStateCaptureError):
        _validate(case, attacked)
    attacked = deepcopy(case["receipt"])
    attacked["unexpected"] = True
    with pytest.raises(ActionStateCaptureError, match="receipt_fields"):
        _validate(case, attacked)


def test_claim_grade_request_rejects_self_rooted_and_legacy_worker_origins():
    _root, _root_pem, role_keys, policy = _trust_fixture()
    worker_key = _private("worker-origin-adversary")
    rogue_supervisor = _private("rogue-worker-supervisor")

    with pytest.raises(ActionStateCaptureError, match="worker_origin_binding_invalid"):
        _build_request(
            policy,
            role_keys,
            worker_key,
            supervisor_key=rogue_supervisor,
            expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
        )

    legacy_identity = build_worker_capture_identity(
        worker_boot_id=WORKER_BOOT_ID,
        worker_pid=4242,
        private_key=worker_key,
    )
    with pytest.raises(ActionStateCaptureError, match="worker_origin_binding_invalid"):
        _build_request(
            policy,
            role_keys,
            worker_key,
            worker_origin_binding_override=legacy_identity.public_identity,
        )


def test_expired_launch_challenge_blocks_current_admission_but_not_history():
    _root, root_pem, role_keys, policy = _trust_fixture()
    request = _build_request(
        policy,
        role_keys,
        _private("worker-expiring-origin"),
        challenge_lifetime_s=120,
    )

    with pytest.raises(ActionStateCaptureError, match="worker_origin_binding_not_current"):
        admit_action_state_capture_request(
            request,
            trusted_root_public_key_pem=root_pem,
            expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
            current_policy_document=policy.document,
            now_unix=NOW + 120,
        )
    assert replay_action_state_capture_request(
        request,
        trusted_root_public_key_pem=root_pem,
        expected_supervisor_public_key=SUPERVISOR_PUBLIC_KEY,
    ).request_sha256 == request["request_sha256"]


def test_runner_role_substitution_is_rejected(tmp_path: Path):
    case = _case(tmp_path)
    attacked = deepcopy(case["request"])
    attacked["runner_attestation"] = build_role_attestation(
        case["policy"],
        role=TASK_ISSUER,
        payload=attacked["request_payload"],
        signed_at_unix=NOW,
        private_key=case["role_keys"][TASK_ISSUER],
    )
    body = {name: item for name, item in attacked.items() if name != "request_sha256"}
    attacked["request_sha256"] = _sha(body)
    with pytest.raises(ActionStateCaptureError):
        admit_action_state_capture_request(
            attacked,
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
            current_policy_document=case["policy"].document,
            now_unix=NOW,
        )


def test_current_policy_rejects_stale_revoked_runner_but_replay_is_historical(
    tmp_path: Path,
):
    case = _case(tmp_path)
    new_role_keys = dict(case["role_keys"])
    new_role_keys[CAMPAIGN_RUNNER] = _private("campaign-runner-r2")
    old_runner_id = hashlib.sha256(_public_raw(case["role_keys"][CAMPAIGN_RUNNER])).hexdigest()
    revision_two_document = _policy_document(
        root=case["root"],
        role_keys=new_role_keys,
        revision=2,
        previous_policy_sha256=case["policy"].policy_sha256,
        revoked_key_ids=[old_runner_id],
    )
    revision_two = validate_campaign_trust_policy(
        revision_two_document,
        trusted_root_public_key_pem=case["root_pem"],
        expected_campaign_name=CAMPAIGN,
        now_unix=NOW,
    )
    assert old_runner_id in revision_two.document["revoked_key_ids"]
    with pytest.raises(ActionStateCaptureError, match="superseded_policy"):
        admit_action_state_capture_request(
            case["request"],
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
            current_policy_document=revision_two.document,
            now_unix=NOW,
        )
    assert (
        replay_action_state_capture_request(
            case["request"],
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
        ).request_sha256
        == case["admission"].request_sha256
    )
    with pytest.raises(ActionStateCaptureError):
        admit_action_state_capture_request(
            case["request"],
            trusted_root_public_key_pem=case["root_pem"],
            expected_supervisor_public_key=case["supervisor_public_key"],
            current_policy_document=case["policy"].document,
            now_unix=case["policy"].document["expires_at_unix"],
        )


def test_exact_latent_request_drift_is_rejected(tmp_path: Path):
    case = _case(tmp_path)
    drifted = _latent_request(prompt="A semantically different task.")
    with pytest.raises(ActionStateCaptureError, match="runtime_request_drift"):
        build_action_state_capture_receipt(
            admission=case["admission"],
            publication=case["publication"],
            worker_private_key=case["worker_key"],
            captured_at_unix=NOW + 1,
            latent_reason_request=drifted,
            model_identity=MODEL_IDENTITY,
            execution_identity=EXECUTION_IDENTITY,
            runtime_identity=RUNTIME_IDENTITY,
            episode_step=3,
            schedule_step=0,
            branch_id="branch-0",
            layer_index=47,
            kv_position=128,
        )
    with pytest.raises(ActionStateCaptureError):
        _validate(case, latent_request=drifted)
    private_answer_request = _latent_request()
    private_answer_request["private_answer"] = "do not expose"
    with pytest.raises(ActionStateCaptureError, match="latent_request_invalid"):
        _build_request(
            case["policy"],
            case["role_keys"],
            case["worker_key"],
            latent_request=private_answer_request,
        )


def test_wrong_first_opportunity_and_action_or_output_leakage_are_rejected(
    tmp_path: Path,
):
    case = _case(tmp_path)
    wrong = deepcopy(case["receipt"])
    wrong["first_action_opportunity"]["ordinal"] = 2
    opportunity_body = {
        name: item
        for name, item in wrong["first_action_opportunity"].items()
        if name != "opportunity_sha256"
    }
    wrong["first_action_opportunity"]["opportunity_sha256"] = _sha(opportunity_body)
    wrong = _resign_receipt(wrong, case["worker_key"])
    with pytest.raises(ActionStateCaptureError, match="opportunity_invalid"):
        _validate(case, wrong)

    leaked = deepcopy(case["receipt"])
    leaked["action_executed"] = True
    leaked["action_trace_count"] = 1
    leaked["decode_started"] = True
    leaked["decoded_token_count"] = 1
    leaked["output_present"] = True
    leaked["output_sha256"] = hashlib.sha256(b"leaked").hexdigest()
    leaked["output_byte_count"] = 6
    leaked = _resign_receipt(leaked, case["worker_key"])
    with pytest.raises(ActionStateCaptureError, match="action_or_output_leakage"):
        _validate(case, leaked)


def test_component_relabel_and_rehash_cannot_escape_snapshot_binding(
    tmp_path: Path,
):
    case = _case(tmp_path)
    attacked = deepcopy(case["receipt"])
    components = attacked["state_components"]
    components["branch_state_sha256"], components["evidence_state_sha256"] = (
        components["evidence_state_sha256"],
        components["branch_state_sha256"],
    )
    attacked["state_sha256"] = _sha(components)
    opportunity = attacked["first_action_opportunity"]
    opportunity["pre_action_state_sha256"] = attacked["state_sha256"]
    opportunity_body = {
        name: item for name, item in opportunity.items() if name != "opportunity_sha256"
    }
    opportunity["opportunity_sha256"] = _sha(opportunity_body)
    attacked = _resign_receipt(attacked, case["worker_key"])
    with pytest.raises(ActionStateCaptureError, match="receipt_state_invalid"):
        _validate(case, attacked)

    other = _case(tmp_path, tag="B")
    with pytest.raises(ActionStateCaptureError, match="publication_binding_mismatch"):
        _validate(case, publication=other["publication"])


def test_pair_local_exactly_once_arm_ledger_and_cross_pair_reuse(
    tmp_path: Path,
):
    case = _case(tmp_path)
    _restore(
        case["store"],
        case["publication"].handle,
        case["admission"],
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
    )
    with pytest.raises(ActionStateCaptureError, match="arm_already_used"):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 3,
        )
    other_request = _build_request(
        case["policy"],
        case["role_keys"],
        case["worker_key"],
        pair_id="pair-OTHER",
        task_id="task-OTHER",
    )
    other_admission = admit_action_state_capture_request(
        other_request,
        trusted_root_public_key_pem=case["root_pem"],
        expected_supervisor_public_key=case["supervisor_public_key"],
        current_policy_document=case["policy"].document,
        now_unix=NOW,
    )
    with pytest.raises(ActionStateCaptureError, match="handle_binding_mismatch"):
        _restore(
            case["store"],
            case["publication"].handle,
            other_admission,
            arm=CONTROL_ARM,
            restored_at_unix=NOW + 3,
        )
    with pytest.raises(ActionStateCaptureError, match="pair_incomplete"):
        case["store"].seal(
            case["publication"].handle,
            case["admission"],
            sealed_at_unix=NOW + 4,
        )


def test_opaque_handle_authentication_prevents_arm_ledger_reset(
    tmp_path: Path,
):
    case = _case(tmp_path)
    _restore(
        case["store"],
        case["publication"].handle,
        case["admission"],
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
    )
    handle_hash = hashlib.sha256(case["publication"].handle.encode("ascii")).hexdigest()
    ledger_path = case["store"]._paths_for_handle_hash(handle_hash)["ledger"]
    attacked = json.loads(ledger_path.read_bytes())
    attacked["uses"][TREATMENT_ARM] = None
    attacked["sequence"] -= 1
    body = {name: item for name, item in attacked.items() if name != "ledger_sha256"}
    # An attacker can recompute public hashes, but not the HMAC derived from
    # the opaque handle token, which is never exposed outside the private bundle.
    attacked["ledger_sha256"] = _sha(body)
    ledger_path.write_bytes(canonical_json_bytes(attacked) + b"\n")
    ledger_path.chmod(0o600)
    with pytest.raises(ActionStateCaptureError, match="ledger_authentication_mismatch"):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 3,
        )


def _first_chunk(case) -> Path:
    handle_hash = hashlib.sha256(case["publication"].handle.encode("ascii")).hexdigest()
    chunk_root = case["store"]._paths_for_handle_hash(handle_hash)["snapshot"] / "chunks"
    return sorted(chunk_root.glob("*/*.bin"))[0]


@pytest.mark.parametrize(
    "attack",
    ["permissions", "symlink", "hardlink", "truncation", "corruption"],
)
def test_private_snapshot_storage_attacks_fail_closed(tmp_path: Path, attack: str):
    case = _case(tmp_path, tag=attack[:1].upper())
    chunk = _first_chunk(case)
    if attack == "permissions":
        chunk.chmod(0o644)
    elif attack == "symlink":
        payload = chunk.read_bytes()
        chunk.unlink()
        outside = tmp_path / "outside-private-state.bin"
        outside.write_bytes(payload)
        outside.chmod(0o600)
        chunk.symlink_to(outside)
    elif attack == "hardlink":
        os.link(chunk, tmp_path / "hardlink-copy.bin")
    elif attack == "truncation":
        payload = chunk.read_bytes()
        chunk.write_bytes(payload[:-1])
        chunk.chmod(0o600)
    else:
        payload = chunk.read_bytes()
        chunk.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        chunk.chmod(0o600)
    with pytest.raises(ActionStateCaptureError):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 2,
        )


def test_atomic_publication_permissions_and_root_symlink_rejection(
    tmp_path: Path,
):
    case = _case(tmp_path)
    for path in case["store"].root.rglob("*"):
        observed = path.lstat()
        if stat.S_ISDIR(observed.st_mode):
            assert stat.S_IMODE(observed.st_mode) == 0o700
        elif stat.S_ISREG(observed.st_mode):
            assert stat.S_IMODE(observed.st_mode) == 0o600
            assert observed.st_nlink == 1
    real = tmp_path / "real-root"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked-root"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ActionStateCaptureError, match="directory_unsafe"):
        PrivateActionSnapshotStore(linked, key_custodian=_custodian())


def test_open_store_rejects_root_path_replacement_without_touching_attacker_tree(
    tmp_path: Path,
):
    case = _case(tmp_path, tag="RootBinding")
    root = case["store"].root
    retained = root.with_name(f"{root.name}-retained")
    root.rename(retained)
    root.mkdir(mode=0o700)
    for name in capture_module._NAMESPACE_NAMES:
        (root / name).mkdir(mode=0o700)
    attacker_lock = root / ".action-state-capture.lock"
    attacker_lock.write_bytes(b"attacker lock")
    attacker_lock.chmod(0o600)
    before = tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*")))

    with pytest.raises(ActionStateCaptureError, match="root_binding_changed"):
        case["store"].recover(case["publication"].handle, case["admission"])

    after = tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*")))
    assert after == before


def test_open_store_rejects_namespace_replacement_before_private_io(tmp_path: Path):
    case = _case(tmp_path, tag="NamespaceBinding")
    handles = case["store"].root / "handles"
    retained = case["store"].root / "handles-retained"
    handles.rename(retained)
    handles.mkdir(mode=0o700)
    marker = handles / "attacker.json"
    marker.write_bytes(b"attacker")
    marker.chmod(0o600)

    with pytest.raises(ActionStateCaptureError, match="namespace_binding_changed"):
        case["store"].recover(case["publication"].handle, case["admission"])

    assert marker.read_bytes() == b"attacker"
    assert sorted(path.name for path in handles.iterdir()) == ["attacker.json"]


def test_open_store_rejects_lock_file_substitution(tmp_path: Path):
    case = _case(tmp_path, tag="LockBinding")
    lock_path = case["store"].root / ".action-state-capture.lock"
    lock_path.rename(case["store"].root / ".action-state-capture.lock-retained")
    lock_path.write_bytes(b"replacement")
    lock_path.chmod(0o600)

    with pytest.raises(ActionStateCaptureError, match="lock_binding_changed"):
        case["store"].recover(case["publication"].handle, case["admission"])

    assert lock_path.read_bytes() == b"replacement"


def test_restore_and_erase_crash_boundaries_recover_deterministically(tmp_path: Path, monkeypatch):
    case = _case(tmp_path)

    def crash_after_prepare(name: str):
        if name == "restore_after_prepare":
            raise RuntimeError("simulated process death")

    monkeypatch.setattr(capture_module, "_crash_boundary", crash_after_prepare)
    with pytest.raises(RuntimeError, match="simulated process death"):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 2,
        )
    monkeypatch.setattr(capture_module, "_crash_boundary", lambda _name: None)
    assert case["store"].recover(case["publication"].handle, case["admission"]) == {
        "recovered": "rolled_back_precommit_restore"
    }
    _restore(
        case["store"],
        case["publication"].handle,
        case["admission"],
        arm=TREATMENT_ARM,
        restored_at_unix=NOW + 2,
    )

    def crash_after_commit(name: str):
        if name == "restore_after_ledger_commit":
            raise RuntimeError("simulated post-commit death")

    monkeypatch.setattr(capture_module, "_crash_boundary", crash_after_commit)
    with pytest.raises(RuntimeError, match="post-commit death"):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=CONTROL_ARM,
            restored_at_unix=NOW + 3,
        )
    monkeypatch.setattr(capture_module, "_crash_boundary", lambda _name: None)
    assert case["store"].recover(case["publication"].handle, case["admission"]) == {
        "recovered": "finalized_committed_restore"
    }
    with pytest.raises(ActionStateCaptureError, match="arm_already_used"):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=CONTROL_ARM,
            restored_at_unix=NOW + 3,
        )

    handle_hash = hashlib.sha256(case["publication"].handle.encode("ascii")).hexdigest()
    abandoned = (
        case["store"]._paths_for_handle_hash(handle_hash)["bundle"] / ".tmp-abandoned"
    )
    abandoned.write_bytes(b"torn temporary publication")
    abandoned.chmod(0o600)
    assert case["store"].recover(case["publication"].handle, case["admission"]) == {
        "recovered": "none"
    }
    assert not abandoned.exists()

    case["store"].seal(
        case["publication"].handle,
        case["admission"],
        sealed_at_unix=NOW + 4,
    )

    def crash_erase(name: str):
        if name == "erase_after_prepare":
            raise RuntimeError("simulated erasure death")

    monkeypatch.setattr(capture_module, "_crash_boundary", crash_erase)
    with pytest.raises(RuntimeError, match="erasure death"):
        case["store"].erase(
            case["publication"].handle,
            case["admission"],
            erased_at_unix=NOW + 5,
        )
    monkeypatch.setattr(capture_module, "_crash_boundary", lambda _name: None)
    assert case["store"].recover(case["publication"].handle, case["admission"]) == {
        "recovered": "completed_erasure"
    }
    with pytest.raises(ActionStateCaptureError, match="private_snapshot_erased"):
        _restore(
            case["store"],
            case["publication"].handle,
            case["admission"],
            arm=TREATMENT_ARM,
            restored_at_unix=NOW + 6,
        )
