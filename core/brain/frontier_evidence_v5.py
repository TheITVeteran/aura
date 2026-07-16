"""Cryptographic and statistical contracts for frontier evidence protocol v5.

This module deliberately contains no model execution.  It verifies evidence
created by mutually distinct actors: the challenge issuer, generation worker,
correctness verifier, and run coordinator.  A candidate-controlled process can
store or relay these envelopes, but it cannot manufacture an admissible run
without every pinned signing key.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import itertools
import json
import math
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

PROTOCOL_VERSION = 5
MAX_CHALLENGE_LIFETIME_S = 3_600.0
MAX_CHALLENGE_CLOCK_SKEW_S = 30.0
PROTOCOL_MANIFEST_SCHEMA = "aura.frontier_protocol.v1"
TASK_SPEC_SCHEMA = "aura.frontier_task_spec.v1"
CHALLENGE_COMMIT_SCHEMA = "aura.frontier_challenge_commit.v1"
CHALLENGE_REVEAL_SCHEMA = "aura.frontier_challenge_reveal.v1"
WORKER_RECEIPT_SCHEMA = "aura.frontier_worker_receipt.v2"
SUPERVISOR_OBSERVATION_SCHEMA = "aura.frontier_supervisor_observation.v1"
CORRECTNESS_RECEIPT_SCHEMA = "aura.frontier_correctness_receipt.v1"
RUN_ENVELOPE_SCHEMA = "aura.frontier_run_envelope.v2"
EFFECTIVE_RUNTIME_MANIFEST_SCHEMA = "aura.effective_runtime_manifest.v1"
SOURCE_IDENTITY_SCHEMA = "aura.frontier_source_identity.v1"
RELEASE_ATTESTATION_SCHEMA = "aura.frontier_release_attestation.v1"
EVIDENCE_INDEX_SCHEMA = "aura.frontier_evidence_index.v1"
EVIDENCE_ENTRY_SCHEMA = "aura.frontier_evidence_index_entry.v1"
TRUST_BASIS_SCHEMA = "aura.frontier_trust_basis.v1"

MATCHED_BUDGET: dict[str, Any] = {
    "time_budget_s": 20.0,
    "hard_timeout_s": 20.0,
    "sample_budget": 3,
    "max_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed_policy": "challenge_item_derived",
    "tools_allowed": False,
    "network_allowed": False,
    "cache_policy": "no_read_no_write",
}

PROTOCOL_MANIFEST_BODY: dict[str, Any] = {
    "schema": PROTOCOL_MANIFEST_SCHEMA,
    "protocol_version": PROTOCOL_VERSION,
    "budget": MATCHED_BUDGET,
    "execution": {
        "fresh_generation_process": True,
        "immutable_source_checkout": True,
        "sealed_evaluation": True,
        "independent_correctness_verifier": True,
        "independent_run_signer": True,
        "coordinator_observed_hard_deadlines": True,
        "external_trust_pins_bound_to_run": True,
        "commit_reveal_challenge": True,
    },
    "required_receipts": [
        WORKER_RECEIPT_SCHEMA,
        CORRECTNESS_RECEIPT_SCHEMA,
        RUN_ENVELOPE_SCHEMA,
    ],
    "required_observations": [SUPERVISOR_OBSERVATION_SCHEMA],
}


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical bytes used for every v5 hash and signature."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def protocol_manifest() -> dict[str, Any]:
    body = copy.deepcopy(PROTOCOL_MANIFEST_BODY)
    return {**body, "manifest_sha256": sha256_json(body)}


PROTOCOL_MANIFEST = protocol_manifest()
PROTOCOL_MANIFEST_SHA256 = PROTOCOL_MANIFEST["manifest_sha256"]


def require_sha256(value: Any, *, field_name: str) -> str:
    digest = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def _require_canonical_text(value: Any, *, field_name: str, max_bytes: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > max_bytes
    ):
        raise ValueError(f"{field_name} is not canonical text")
    return value


def _require_finite(value: Any, *, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field_name} must be finite")
    return result


def _decode_base64(value: Any, *, field_name: str, expected_bytes: int | None = None) -> bytes:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} is missing or noncanonical")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not canonical base64") from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{field_name} is not canonical base64")
    if expected_bytes is not None and len(decoded) != expected_bytes:
        raise ValueError(f"{field_name} has the wrong byte length")
    return decoded


def verify_signed_envelope(
    raw: Any,
    *,
    schema: str,
    trusted_keys: Mapping[str, str] | None,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Verify an exact envelope and return an unmodified deep copy.

    Semantic validators must reject noncanonical field values; they must never
    normalize and reconstruct a signed payload.  This keeps a verified
    signature valid when the envelope is serialized again.
    """

    if not isinstance(raw, dict) or raw.get("schema") != schema:
        raise ValueError(f"{role} envelope schema is invalid")
    if set(raw) != {"schema", "signed_payload", "signer"}:
        raise ValueError(f"{role} envelope fields are invalid")
    payload = raw.get("signed_payload")
    signer = raw.get("signer")
    if not isinstance(payload, dict) or not isinstance(signer, dict):
        raise ValueError(f"{role} envelope is incomplete")
    if set(signer) != {
        "algorithm",
        "signer_id",
        "public_key_b64",
        "signature_b64",
    }:
        raise ValueError(f"{role} signer fields are invalid")
    if signer.get("algorithm") != "Ed25519":
        raise ValueError(f"{role} signature algorithm is unsupported")
    signer_id = _require_canonical_text(signer.get("signer_id"), field_name=f"{role} signer_id")
    trust = dict(trusted_keys or {})
    trusted_key = trust.get(signer_id)
    if not trusted_key:
        raise ValueError(f"{role} signer is not explicitly trusted")
    if signer.get("public_key_b64") != trusted_key:
        raise ValueError(f"{role} signer key does not match the trust root")
    public_key = _decode_base64(
        trusted_key,
        field_name=f"{role} public key",
        expected_bytes=32,
    )
    signature = _decode_base64(
        signer.get("signature_b64"),
        field_name=f"{role} signature",
        expected_bytes=64,
    )
    signed_bytes = canonical_json_bytes(payload)
    if len(signed_bytes) > 16 * 1024 * 1024:
        raise ValueError(f"{role} signed payload is unreasonably large")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, signed_bytes)
    except (InvalidSignature, ValueError) as exc:
        raise ValueError(f"{role} signature verification failed") from exc
    exact = copy.deepcopy(raw)
    return exact, exact["signed_payload"], signer_id


def validate_protocol_manifest(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != canonical_json_bytes(
        PROTOCOL_MANIFEST
    ):
        raise ValueError("frontier protocol manifest is not the pinned v5 protocol")
    return copy.deepcopy(raw)


def build_trust_basis(
    *,
    evaluator_keys: Mapping[str, str] | None,
    worker_keys: Mapping[str, str] | None,
    verifiers: Mapping[str, Mapping[str, str]] | None,
    run_keys: Mapping[str, str] | None,
    release_keys: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Canonicalize explicit external pins and reject cross-role key reuse."""

    role_maps = {
        "evaluator_keys": dict(evaluator_keys or {}),
        "worker_keys": dict(worker_keys or {}),
        "run_keys": dict(run_keys or {}),
        "release_keys": dict(release_keys or {}),
    }
    if any(not values for values in role_maps.values()) or not verifiers:
        raise ValueError("frontier trust basis requires every independent role")
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for role, values in role_maps.items():
        for signer_id, public_key in values.items():
            canonical_id = _require_canonical_text(
                signer_id, field_name=f"{role} signer_id"
            )
            if canonical_id in seen_ids:
                raise ValueError("frontier trust basis reuses a signer identity across roles")
            _decode_base64(
                public_key,
                field_name=f"{role} public key",
                expected_bytes=32,
            )
            if public_key in seen_keys:
                raise ValueError("frontier trust basis reuses a cryptographic key across roles")
            seen_ids.add(canonical_id)
            seen_keys.add(public_key)
    canonical_verifiers: dict[str, dict[str, str]] = {}
    for verifier_id, raw_pin in dict(verifiers or {}).items():
        canonical_id = _require_canonical_text(
            verifier_id, field_name="verifier signer_id"
        )
        pin = dict(raw_pin or {})
        if set(pin) != {
            "public_key_b64",
            "implementation_sha256",
            "release_sha256",
        }:
            raise ValueError("frontier verifier trust pin is incomplete")
        public_key = pin["public_key_b64"]
        _decode_base64(
            public_key,
            field_name="verifier public key",
            expected_bytes=32,
        )
        require_sha256(
            pin["implementation_sha256"], field_name="verifier implementation"
        )
        require_sha256(pin["release_sha256"], field_name="verifier release")
        if canonical_id in seen_ids or public_key in seen_keys:
            raise ValueError("frontier verifier identity or key is not role-independent")
        seen_ids.add(canonical_id)
        seen_keys.add(public_key)
        canonical_verifiers[canonical_id] = pin
    body = {
        "schema": TRUST_BASIS_SCHEMA,
        "basis": "externally_supplied_explicit_pins",
        "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        **role_maps,
        "verifiers": canonical_verifiers,
    }
    return {**body, "manifest_sha256": sha256_json(body)}


def validate_trust_basis(
    raw: Any,
    *,
    evaluator_keys: Mapping[str, str] | None,
    worker_keys: Mapping[str, str] | None,
    verifiers: Mapping[str, Mapping[str, str]] | None,
    run_keys: Mapping[str, str] | None,
    release_keys: Mapping[str, str] | None,
) -> dict[str, Any]:
    expected = build_trust_basis(
        evaluator_keys=evaluator_keys,
        worker_keys=worker_keys,
        verifiers=verifiers,
        run_keys=run_keys,
        release_keys=release_keys,
    )
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != canonical_json_bytes(
        expected
    ):
        raise ValueError("frontier evidence trust basis differs from external pins")
    return copy.deepcopy(raw)


def validate_effective_runtime_manifest(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != EFFECTIVE_RUNTIME_MANIFEST_SCHEMA:
        raise ValueError("effective runtime manifest schema is invalid")
    required = {
        "schema",
        "subject_id",
        "base_model_manifest_sha256",
        "tokenizer_sha256",
        "prompt_template_sha256",
        "execution_identity",
        "inference_libraries",
        "adapters_sha256",
        "steering_sha256",
        "modifiers",
        "cache_policy",
        "generation_parameters",
        "runtime_isolation",
        "manifest_sha256",
    }
    if set(raw) != required:
        raise ValueError("effective runtime manifest fields are invalid")
    manifest = copy.deepcopy(raw)
    _require_canonical_text(manifest.get("subject_id"), field_name="runtime subject_id")
    for field_name in (
        "base_model_manifest_sha256",
        "tokenizer_sha256",
        "prompt_template_sha256",
    ):
        require_sha256(manifest.get(field_name), field_name=field_name)
    execution_identity = manifest.get("execution_identity")
    if not isinstance(execution_identity, dict) or set(execution_identity) != {
        "worker_implementation_sha256",
        "python_executable_sha256",
        "library_lock_sha256",
        "operating_system",
        "os_release",
        "machine",
        "python_implementation",
        "python_version",
        "inference_backend",
    }:
        raise ValueError("effective runtime execution identity is incomplete")
    for field_name in (
        "worker_implementation_sha256",
        "python_executable_sha256",
        "library_lock_sha256",
    ):
        require_sha256(execution_identity.get(field_name), field_name=field_name)
    for field_name in (
        "operating_system",
        "os_release",
        "machine",
        "python_implementation",
        "python_version",
        "inference_backend",
    ):
        _require_canonical_text(
            execution_identity.get(field_name), field_name=f"execution {field_name}"
        )
    libraries = manifest.get("inference_libraries")
    if not isinstance(libraries, dict) or not libraries:
        raise ValueError("effective runtime inference libraries are missing")
    for name, version in libraries.items():
        _require_canonical_text(name, field_name="inference library name")
        _require_canonical_text(version, field_name=f"inference library {name}")
    if execution_identity["library_lock_sha256"] != sha256_json(libraries):
        raise ValueError("effective runtime library lock does not match its versions")
    for field_name in ("adapters_sha256", "steering_sha256"):
        values = manifest.get(field_name)
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f"effective runtime {field_name} is malformed")
        for digest in values:
            require_sha256(digest, field_name=field_name)
    if not isinstance(manifest.get("modifiers"), dict):
        raise ValueError("effective runtime modifiers are malformed")
    if manifest.get("cache_policy") != {
        "prompt_cache": "disabled",
        "result_cache": "disabled",
        "playbook_cache": "disabled",
        "clear_before_run": True,
    }:
        raise ValueError("effective runtime cache policy is not sealed")
    if manifest.get("generation_parameters") != MATCHED_BUDGET:
        raise ValueError("effective runtime generation parameters are not matched")
    if manifest.get("runtime_isolation") != {
        "fresh_process": True,
        "immutable_source": True,
        "network_enabled": False,
        "tools_enabled": False,
        "sealed_evaluation_enforced": True,
    }:
        raise ValueError("effective runtime isolation is not sealed")
    digest = require_sha256(manifest.get("manifest_sha256"), field_name="runtime manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if sha256_json(body) != digest:
        raise ValueError("effective runtime manifest digest mismatch")
    return manifest


def validate_source_identity(
    raw: Any,
    *,
    trusted_release_keys: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != SOURCE_IDENTITY_SCHEMA:
        raise ValueError("frontier source identity schema is invalid")
    required = {
        "schema",
        "repository_id",
        "canonical_remote_sha256",
        "commit_sha",
        "tree_sha",
        "release_ref",
        "release_commit_sha",
        "release_attestation",
        "release_attestation_sha256",
        "head_descends_from_release",
        "clean",
        "immutable_checkout",
        "imports_after_verification",
        "identity_sha256",
    }
    if set(raw) != required:
        raise ValueError("frontier source identity fields are invalid")
    identity = copy.deepcopy(raw)
    _require_canonical_text(identity.get("repository_id"), field_name="repository_id")
    _require_canonical_text(identity.get("release_ref"), field_name="release_ref")
    require_sha256(identity.get("canonical_remote_sha256"), field_name="canonical remote")
    release_envelope, release_payload, _ = verify_signed_envelope(
        identity.get("release_attestation"),
        schema=RELEASE_ATTESTATION_SCHEMA,
        trusted_keys=trusted_release_keys,
        role="release attestation",
    )
    if set(release_payload) != {
        "repository_id",
        "canonical_remote_sha256",
        "release_ref",
        "release_commit_sha",
        "issued_at_unix",
    }:
        raise ValueError("release attestation fields are invalid")
    for field_name in (
        "repository_id",
        "canonical_remote_sha256",
        "release_ref",
        "release_commit_sha",
    ):
        if release_payload.get(field_name) != identity.get(field_name):
            raise ValueError("source identity contradicts its release attestation")
    _require_finite(
        release_payload.get("issued_at_unix"),
        field_name="release attestation issue time",
        minimum=0,
    )
    attestation_digest = require_sha256(
        identity.get("release_attestation_sha256"), field_name="release attestation"
    )
    if sha256_json(release_envelope) != attestation_digest:
        raise ValueError("release attestation digest mismatch")
    for field_name in ("commit_sha", "tree_sha", "release_commit_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(identity.get(field_name) or "")):
            raise ValueError(f"frontier source {field_name} is invalid")
    for field_name in (
        "head_descends_from_release",
        "clean",
        "immutable_checkout",
        "imports_after_verification",
    ):
        if identity.get(field_name) is not True:
            raise ValueError(f"frontier source identity lacks {field_name}")
    digest = require_sha256(identity.get("identity_sha256"), field_name="source identity")
    body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if sha256_json(body) != digest:
        raise ValueError("frontier source identity digest mismatch")
    return identity


def identity_freeze_sha256(
    *,
    source_identity_sha256: str,
    candidate_runtime_sha256: str,
    reference_runtime_sha256: str,
) -> str:
    return sha256_json(
        {
            "source_identity_sha256": require_sha256(
                source_identity_sha256, field_name="source identity"
            ),
            "candidate_runtime_sha256": require_sha256(
                candidate_runtime_sha256, field_name="candidate runtime"
            ),
            "reference_runtime_sha256": require_sha256(
                reference_runtime_sha256, field_name="reference runtime"
            ),
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        }
    )


def validate_challenge_bundle(
    raw: Any,
    *,
    trusted_evaluator_keys: Mapping[str, str] | None,
    expected_identity_freeze_sha256: str | None = None,
    verification_time_unix: float | None = None,
    require_fresh: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"commit", "reveal"}:
        raise ValueError("challenge bundle is incomplete")
    commit, commit_payload, commit_signer = verify_signed_envelope(
        raw["commit"],
        schema=CHALLENGE_COMMIT_SCHEMA,
        trusted_keys=trusted_evaluator_keys,
        role="challenge commit",
    )
    reveal, reveal_payload, reveal_signer = verify_signed_envelope(
        raw["reveal"],
        schema=CHALLENGE_REVEAL_SCHEMA,
        trusted_keys=trusted_evaluator_keys,
        role="challenge reveal",
    )
    if commit_signer != reveal_signer:
        raise ValueError("challenge commit and reveal signer mismatch")
    if set(commit_payload) != {
        "challenge_id",
        "nonce_sha256",
        "identity_freeze_sha256",
        "protocol_manifest_sha256",
        "committed_at_unix",
    }:
        raise ValueError("challenge commitment fields are invalid")
    if set(reveal_payload) != {
        "challenge_id",
        "nonce_b64",
        "commit_envelope_sha256",
        "revealed_at_unix",
        "expires_at_unix",
    }:
        raise ValueError("challenge reveal fields are invalid")
    challenge_id = _require_canonical_text(
        commit_payload.get("challenge_id"), field_name="challenge_id"
    )
    if reveal_payload.get("challenge_id") != challenge_id:
        raise ValueError("challenge reveal identity mismatch")
    if commit_payload.get("protocol_manifest_sha256") != PROTOCOL_MANIFEST_SHA256:
        raise ValueError("challenge is not bound to the v5 protocol")
    freeze = require_sha256(
        commit_payload.get("identity_freeze_sha256"), field_name="identity freeze"
    )
    if expected_identity_freeze_sha256 is not None and freeze != require_sha256(
        expected_identity_freeze_sha256, field_name="expected identity freeze"
    ):
        raise ValueError("challenge was not committed after the measured identities froze")
    nonce = _decode_base64(
        reveal_payload.get("nonce_b64"), field_name="challenge nonce"
    )
    if len(nonce) < 32:
        raise ValueError("challenge nonce lacks 256 bits of entropy")
    if hashlib.sha256(nonce).hexdigest() != commit_payload.get("nonce_sha256"):
        raise ValueError("challenge reveal does not open its commitment")
    if reveal_payload.get("commit_envelope_sha256") != sha256_json(commit):
        raise ValueError("challenge reveal is not bound to the signed commitment")
    committed = _require_finite(
        commit_payload.get("committed_at_unix"), field_name="challenge commit time", minimum=0
    )
    revealed = _require_finite(
        reveal_payload.get("revealed_at_unix"), field_name="challenge reveal time", minimum=0
    )
    expires = _require_finite(
        reveal_payload.get("expires_at_unix"), field_name="challenge expiry", minimum=0
    )
    if not committed < revealed < expires:
        raise ValueError("challenge commit/reveal chronology is invalid")
    if expires - revealed > MAX_CHALLENGE_LIFETIME_S:
        raise ValueError("challenge validity window exceeds the protocol bound")
    if require_fresh:
        now = _require_finite(
            verification_time_unix,
            field_name="challenge verification time",
            minimum=0,
        )
        if now + MAX_CHALLENGE_CLOCK_SKEW_S < revealed:
            raise ValueError("challenge reveal is not yet valid")
        if now - MAX_CHALLENGE_CLOCK_SKEW_S > expires:
            raise ValueError("challenge has expired")
    bundle = {"commit": commit, "reveal": reveal}
    return {
        **bundle,
        "challenge_id": challenge_id,
        "nonce": nonce,
        "identity_freeze_sha256": freeze,
        "revealed_at_unix": revealed,
        "expires_at_unix": expires,
        "bundle_sha256": sha256_json(bundle),
        "evaluator_id": commit_signer,
    }


def validate_task_spec(
    raw: Any,
    *,
    trusted_evaluator_keys: Mapping[str, str] | None,
    trusted_verifiers: Mapping[str, Mapping[str, str]] | None,
    challenge: Mapping[str, Any],
    expected_items: Sequence[Mapping[str, Any]],
    battery_version: str,
    seed: int,
    per_class: int,
) -> dict[str, Any]:
    envelope, payload, evaluator_id = verify_signed_envelope(
        raw,
        schema=TASK_SPEC_SCHEMA,
        trusted_keys=trusted_evaluator_keys,
        role="task specification",
    )
    if set(payload) != {
        "battery_version",
        "seed",
        "per_class",
        "challenge_bundle_sha256",
        "protocol_manifest",
        "protocol_manifest_sha256",
        "verifier_identity",
        "effective_n",
        "items",
        "issued_at_unix",
    }:
        raise ValueError("task specification fields are invalid")
    if (
        payload.get("battery_version") != battery_version
        or payload.get("seed") != seed
        or payload.get("per_class") != per_class
    ):
        raise ValueError("task specification battery instance mismatch")
    validate_protocol_manifest(payload.get("protocol_manifest"))
    if payload.get("protocol_manifest_sha256") != PROTOCOL_MANIFEST_SHA256:
        raise ValueError("task specification protocol digest mismatch")
    if payload.get("challenge_bundle_sha256") != challenge.get("bundle_sha256"):
        raise ValueError("task specification is not bound to the challenge reveal")
    issued = _require_finite(
        payload.get("issued_at_unix"), field_name="task specification issue time", minimum=0
    )
    if issued < float(challenge["revealed_at_unix"]):
        raise ValueError("task specification predates challenge reveal")
    verifier = payload.get("verifier_identity")
    if not isinstance(verifier, dict) or set(verifier) != {
        "verifier_id",
        "public_key_b64",
        "implementation_sha256",
        "release_sha256",
    }:
        raise ValueError("task specification verifier identity is invalid")
    verifier_id = _require_canonical_text(
        verifier.get("verifier_id"), field_name="verifier_id"
    )
    pinned = dict((trusted_verifiers or {}).get(verifier_id) or {})
    if not pinned:
        raise ValueError("task specification verifier is not independently pinned")
    expected_pin_fields = {"public_key_b64", "implementation_sha256", "release_sha256"}
    if set(pinned) != expected_pin_fields or any(
        verifier.get(field_name) != pinned.get(field_name)
        for field_name in expected_pin_fields
    ):
        raise ValueError("task specification verifier identity does not match its pin")
    _decode_base64(
        verifier["public_key_b64"], field_name="verifier public key", expected_bytes=32
    )
    require_sha256(verifier["implementation_sha256"], field_name="verifier implementation")
    require_sha256(verifier["release_sha256"], field_name="verifier release")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != len(expected_items):
        raise ValueError("task specification item coverage is incomplete")
    if payload.get("effective_n") != len(items):
        raise ValueError("task specification effective sample count is false")
    item_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for index, (observed, expected) in enumerate(zip(items, expected_items, strict=True)):
        if not isinstance(observed, dict) or observed != dict(expected):
            raise ValueError(f"task specification item {index} does not match the battery")
        item_id = require_sha256(observed.get("item_id"), field_name="task item_id")
        prompt_hash = require_sha256(
            observed.get("prompt_sha256"), field_name="task prompt_sha256"
        )
        for field_name in (
            "grader_implementation_sha256",
            "expected_answer_commitment_sha256",
            "hidden_case_commitment_sha256",
        ):
            require_sha256(observed.get(field_name), field_name=field_name)
        if item_id in item_ids or prompt_hash in prompt_hashes:
            raise ValueError("task specification contains duplicated effective samples")
        item_ids.add(item_id)
        prompt_hashes.add(prompt_hash)
    return {
        "envelope": envelope,
        "payload": payload,
        "task_spec_sha256": sha256_json(envelope),
        "evaluator_id": evaluator_id,
        "verifier_id": verifier_id,
        "verifier_identity": copy.deepcopy(verifier),
    }


def _validate_resource_usage(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "input_tokens",
        "output_tokens",
        "token_count_method",
        "candidate_count",
        "generation_calls",
        "tool_calls",
        "network_calls",
        "cache_reads",
        "cache_writes",
        "deadline_exceeded",
        "wall_time_s",
        "peak_memory_bytes",
    }:
        raise ValueError("worker resource usage fields are invalid")
    usage = copy.deepcopy(raw)
    for field_name in (
        "input_tokens",
        "output_tokens",
        "candidate_count",
        "generation_calls",
        "tool_calls",
        "network_calls",
        "cache_reads",
        "cache_writes",
        "peak_memory_bytes",
    ):
        value = usage.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"worker resource usage {field_name} is invalid")
    if usage["input_tokens"] < 1 or usage["output_tokens"] > MATCHED_BUDGET["max_tokens"]:
        raise ValueError("worker token budget was not enforced")
    if usage["token_count_method"] != "tokenizer_exact":
        raise ValueError("worker token use was not measured by the effective tokenizer")
    if not 1 <= usage["candidate_count"] <= MATCHED_BUDGET["sample_budget"]:
        raise ValueError("worker candidate budget was not enforced")
    if not 1 <= usage["generation_calls"] <= MATCHED_BUDGET["sample_budget"]:
        raise ValueError("worker generation-call budget was not enforced")
    if any(usage[field] != 0 for field in ("tool_calls", "network_calls", "cache_reads", "cache_writes")):
        raise ValueError("worker used a prohibited external or cached resource")
    if usage.get("deadline_exceeded") is not False:
        raise ValueError("worker exceeded the hard deadline")
    wall = _require_finite(usage.get("wall_time_s"), field_name="worker wall time", minimum=0)
    if wall > MATCHED_BUDGET["hard_timeout_s"]:
        raise ValueError("worker wall time exceeded the hard budget")
    return usage


def expected_request_id(
    *, run_id: str, run_nonce_sha256: str, item_id: str, attempt_index: int
) -> str:
    return sha256_json(
        {
            "run_id": run_id,
            "run_nonce_sha256": run_nonce_sha256,
            "item_id": item_id,
            "attempt_index": attempt_index,
        }
    )


def validate_worker_receipt(
    raw: Any,
    *,
    trusted_worker_keys: Mapping[str, str] | None,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    envelope, payload, signer_id = verify_signed_envelope(
        raw,
        schema=WORKER_RECEIPT_SCHEMA,
        trusted_keys=trusted_worker_keys,
        role="generation worker",
    )
    required = {
        "run_id",
        "run_nonce_b64",
        "run_nonce_sha256",
        "item_id",
        "request_id",
        "attempt_index",
        "prompt_sha256",
        "output_sha256",
        "source_identity_sha256",
        "runtime_manifest_sha256",
        "model_stability_sha256",
        "protocol_manifest_sha256",
        "challenge_bundle_sha256",
        "started_at_unix",
        "completed_at_unix",
        "elapsed_s",
        "decoding_parameters",
        "resource_usage",
        "sealed_evaluation_enforced",
        "fallbacks_used",
    }
    if set(payload) != required:
        raise ValueError("generation worker receipt fields are invalid")
    for field_name in (
        "run_id",
        "run_nonce_sha256",
        "item_id",
        "prompt_sha256",
        "output_sha256",
        "source_identity_sha256",
        "runtime_manifest_sha256",
        "model_stability_sha256",
        "protocol_manifest_sha256",
        "challenge_bundle_sha256",
    ):
        expected = bindings.get(field_name)
        if expected is not None and payload.get(field_name) != expected:
            raise ValueError(f"generation worker receipt {field_name} binding mismatch")
        require_sha256(payload.get(field_name), field_name=field_name)
    run_nonce = _decode_base64(
        payload.get("run_nonce_b64"),
        field_name="run nonce",
    )
    if len(run_nonce) < 32 or hashlib.sha256(run_nonce).hexdigest() != payload.get(
        "run_nonce_sha256"
    ):
        raise ValueError("worker receipt does not reveal its committed 256-bit run nonce")
    attempt = payload.get("attempt_index")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("generation worker attempt index is invalid")
    expected_id = expected_request_id(
        run_id=payload["run_id"],
        run_nonce_sha256=payload["run_nonce_sha256"],
        item_id=payload["item_id"],
        attempt_index=attempt,
    )
    if payload.get("request_id") != expected_id:
        raise ValueError("generation worker request identity is invalid")
    started = _require_finite(payload.get("started_at_unix"), field_name="worker start", minimum=0)
    completed = _require_finite(
        payload.get("completed_at_unix"), field_name="worker completion", minimum=0
    )
    elapsed = _require_finite(payload.get("elapsed_s"), field_name="worker elapsed", minimum=0)
    if completed < started or abs((completed - started) - elapsed) > 0.25:
        raise ValueError("generation worker timing receipt is inconsistent")
    if elapsed > MATCHED_BUDGET["hard_timeout_s"]:
        raise ValueError("generation worker exceeded the hard time budget")
    if payload.get("decoding_parameters") != MATCHED_BUDGET:
        raise ValueError("generation worker decoding parameters are not matched")
    usage = _validate_resource_usage(payload.get("resource_usage"))
    if abs(float(usage["wall_time_s"]) - elapsed) > 0.25:
        raise ValueError("generation worker usage contradicts elapsed time")
    if payload.get("sealed_evaluation_enforced") is not True:
        raise ValueError("generation worker did not enforce sealed evaluation")
    fallbacks = payload.get("fallbacks_used")
    if not isinstance(fallbacks, list) or any(
        not isinstance(value, str) or value != value.strip() for value in fallbacks
    ):
        raise ValueError("generation worker fallback receipt is malformed")
    return {
        "envelope": envelope,
        "payload": payload,
        "signer_id": signer_id,
        "receipt_sha256": sha256_json(envelope),
    }


def validate_supervisor_observation(
    raw: Any,
    *,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate coordinator-observed IPC timing separately from worker claims."""

    required = {
        "schema",
        "run_id",
        "run_nonce_sha256",
        "item_id",
        "request_id",
        "attempt_index",
        "prompt_sha256",
        "output_sha256",
        "observed_wall_time_s",
        "deadline_s",
        "deadline_exceeded",
        "process_pid",
        "process_running_after_response",
        "observed_at_unix",
    }
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != SUPERVISOR_OBSERVATION_SCHEMA
        or set(raw) != required
    ):
        raise ValueError("execution supervisor observation fields are invalid")
    observation = copy.deepcopy(raw)
    for field_name in (
        "run_id",
        "run_nonce_sha256",
        "item_id",
        "request_id",
        "prompt_sha256",
        "output_sha256",
    ):
        expected = bindings.get(field_name)
        if expected is not None and observation.get(field_name) != expected:
            raise ValueError(f"execution supervisor {field_name} binding mismatch")
        require_sha256(observation.get(field_name), field_name=field_name)
    attempt = observation.get("attempt_index")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("execution supervisor attempt index is invalid")
    if bindings.get("attempt_index") is not None and attempt != bindings["attempt_index"]:
        raise ValueError("execution supervisor attempt index binding mismatch")
    elapsed = _require_finite(
        observation.get("observed_wall_time_s"),
        field_name="execution supervisor wall time",
        minimum=0,
    )
    deadline = _require_finite(
        observation.get("deadline_s"),
        field_name="execution supervisor deadline",
        minimum=0,
    )
    if deadline != MATCHED_BUDGET["hard_timeout_s"]:
        raise ValueError("execution supervisor used an unpinned deadline")
    if observation.get("deadline_exceeded") is not False or elapsed > deadline:
        raise ValueError("execution supervisor observed a hard-deadline violation")
    pid = observation.get("process_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("execution supervisor process identity is invalid")
    if observation.get("process_running_after_response") is not True:
        raise ValueError("generation worker did not survive the supervised response")
    _require_finite(
        observation.get("observed_at_unix"),
        field_name="execution supervisor observation time",
        minimum=0,
    )
    return observation


def validate_correctness_receipt(
    raw: Any,
    *,
    trusted_verifiers: Mapping[str, Mapping[str, str]] | None,
    verifier_identity: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    verifier_id = str(verifier_identity.get("verifier_id") or "")
    pinned = dict((trusted_verifiers or {}).get(verifier_id) or {})
    trusted_key = pinned.get("public_key_b64")
    envelope, payload, signer_id = verify_signed_envelope(
        raw,
        schema=CORRECTNESS_RECEIPT_SCHEMA,
        trusted_keys={verifier_id: trusted_key} if trusted_key else {},
        role="correctness verifier",
    )
    if signer_id != verifier_id:
        raise ValueError("correctness receipt signer is not the task verifier")
    required = {
        "run_id",
        "item_id",
        "output_sha256",
        "task_spec_sha256",
        "challenge_bundle_sha256",
        "verifier_implementation_sha256",
        "verifier_release_sha256",
        "expected_answer_commitment_sha256",
        "hidden_case_commitment_sha256",
        "correct",
        "checked",
        "grader_execution_sha256",
        "graded_at_unix",
    }
    if set(payload) != required:
        raise ValueError("correctness receipt fields are invalid")
    for field_name in (
        "run_id",
        "item_id",
        "output_sha256",
        "task_spec_sha256",
        "challenge_bundle_sha256",
        "expected_answer_commitment_sha256",
        "hidden_case_commitment_sha256",
    ):
        expected = bindings.get(field_name)
        if expected is not None and payload.get(field_name) != expected:
            raise ValueError(f"correctness receipt {field_name} binding mismatch")
        require_sha256(payload.get(field_name), field_name=field_name)
    if payload.get("verifier_implementation_sha256") != verifier_identity.get(
        "implementation_sha256"
    ) or payload.get("verifier_release_sha256") != verifier_identity.get("release_sha256"):
        raise ValueError("correctness receipt verifier implementation is not pinned")
    if not isinstance(payload.get("correct"), bool) or payload.get("checked") is not True:
        raise ValueError("correctness receipt is not an independent checked verdict")
    require_sha256(payload.get("grader_execution_sha256"), field_name="grader execution")
    _require_finite(payload.get("graded_at_unix"), field_name="graded time", minimum=0)
    return {
        "envelope": envelope,
        "payload": payload,
        "signer_id": signer_id,
        "receipt_sha256": sha256_json(envelope),
    }


def validate_run_envelope(
    raw: Any,
    *,
    trusted_run_keys: Mapping[str, str] | None,
    bindings: Mapping[str, Any],
    worker_receipts: Sequence[Mapping[str, Any]],
    supervisor_observations: Sequence[Mapping[str, Any]],
    correctness_receipts: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    challenge: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not worker_receipts
        or len(worker_receipts) != len(supervisor_observations)
        or len(worker_receipts) != len(correctness_receipts)
        or len(worker_receipts) != len(outputs)
    ):
        raise ValueError("run envelope evidence coverage is incomplete")
    envelope, payload, signer_id = verify_signed_envelope(
        raw,
        schema=RUN_ENVELOPE_SCHEMA,
        trusted_keys=trusted_run_keys,
        role="run coordinator",
    )
    required = {
        "run_id",
        "run_nonce_b64",
        "run_nonce_sha256",
        "task_spec_sha256",
        "challenge_bundle_sha256",
        "protocol_manifest_sha256",
        "source_identity_sha256",
        "runtime_manifest_sha256",
        "reference_artifact_sha256",
        "trust_basis_sha256",
        "worker_receipt_sha256",
        "supervisor_observation_sha256",
        "correctness_receipt_sha256",
        "outputs_sha256",
        "worker_signer_ids",
        "verifier_id",
        "started_at_unix",
        "completed_at_unix",
        "budget_summary",
    }
    if set(payload) != required:
        raise ValueError("run envelope fields are invalid")
    for field_name in (
        "run_id",
        "run_nonce_sha256",
        "task_spec_sha256",
        "challenge_bundle_sha256",
        "protocol_manifest_sha256",
        "source_identity_sha256",
        "runtime_manifest_sha256",
        "reference_artifact_sha256",
        "trust_basis_sha256",
    ):
        expected = bindings.get(field_name)
        if expected is not None and payload.get(field_name) != expected:
            raise ValueError(f"run envelope {field_name} binding mismatch")
        require_sha256(payload.get(field_name), field_name=field_name)
    run_nonce = _decode_base64(
        payload.get("run_nonce_b64"),
        field_name="run envelope nonce",
    )
    if len(run_nonce) < 32 or hashlib.sha256(run_nonce).hexdigest() != payload.get(
        "run_nonce_sha256"
    ):
        raise ValueError("run envelope does not reveal its committed 256-bit nonce")
    expected_worker_digests = [item["receipt_sha256"] for item in worker_receipts]
    expected_supervisor_digests = [sha256_json(item) for item in supervisor_observations]
    expected_correctness_digests = [item["receipt_sha256"] for item in correctness_receipts]
    if payload.get("worker_receipt_sha256") != expected_worker_digests:
        raise ValueError("run envelope does not bind every worker receipt")
    if payload.get("supervisor_observation_sha256") != expected_supervisor_digests:
        raise ValueError("run envelope does not bind every supervisor observation")
    if payload.get("correctness_receipt_sha256") != expected_correctness_digests:
        raise ValueError("run envelope does not bind every correctness receipt")
    if payload.get("outputs_sha256") != sha256_json(list(outputs)):
        raise ValueError("run envelope does not bind candidate outputs")
    worker_ids = [item["signer_id"] for item in worker_receipts]
    if payload.get("worker_signer_ids") != worker_ids:
        raise ValueError("run envelope worker signer identities are incomplete")
    if payload.get("verifier_id") != bindings.get("verifier_id"):
        raise ValueError("run envelope verifier identity mismatch")
    started = _require_finite(payload.get("started_at_unix"), field_name="run start", minimum=0)
    completed = _require_finite(
        payload.get("completed_at_unix"), field_name="run completion", minimum=0
    )
    if completed < started or started < float(challenge["revealed_at_unix"]):
        raise ValueError("run chronology predates challenge reveal")
    if completed > float(challenge["expires_at_unix"]):
        raise ValueError("run completed after challenge expiry")
    if any(
        float(item["payload"]["started_at_unix"]) < started
        or float(item["payload"]["completed_at_unix"]) > completed
        for item in worker_receipts
    ):
        raise ValueError("worker receipt falls outside the signed run window")
    if any(
        float(correctness["payload"]["graded_at_unix"])
        < float(worker["payload"]["completed_at_unix"])
        or float(correctness["payload"]["graded_at_unix"]) > completed
        for worker, correctness in zip(
            worker_receipts,
            correctness_receipts,
            strict=True,
        )
    ):
        raise ValueError("correctness receipt chronology is outside the signed run window")
    usage = [item["payload"]["resource_usage"] for item in worker_receipts]
    supervised_wall = [float(item["observed_wall_time_s"]) for item in supervisor_observations]
    expected_budget_summary = {
        "item_count": len(usage),
        "total_input_tokens": sum(item["input_tokens"] for item in usage),
        "total_output_tokens": sum(item["output_tokens"] for item in usage),
        "total_candidate_count": sum(item["candidate_count"] for item in usage),
        "total_generation_calls": sum(item["generation_calls"] for item in usage),
        "maximum_item_wall_time_s": max((item["wall_time_s"] for item in usage), default=0),
        "maximum_supervisor_wall_time_s": max(supervised_wall, default=0),
        "all_within_budget": True,
    }
    if payload.get("budget_summary") != expected_budget_summary:
        raise ValueError("run envelope budget summary is not measured evidence")
    return {
        "envelope": envelope,
        "payload": payload,
        "signer_id": signer_id,
        "envelope_sha256": sha256_json(envelope),
    }


def validate_evidence_role_separation(
    *,
    challenge: Mapping[str, Any],
    task_spec: Mapping[str, Any],
    worker_receipts: Sequence[Mapping[str, Any]],
    correctness_receipts: Sequence[Mapping[str, Any]],
    run_envelope: Mapping[str, Any],
) -> None:
    """Require distinct cryptographic principals for every evidence role."""

    evaluator_id = challenge.get("evaluator_id")
    if task_spec.get("evaluator_id") != evaluator_id:
        raise ValueError("task specification signer differs from the challenge issuer")
    if not worker_receipts or not correctness_receipts:
        raise ValueError("independent evidence roles require complete receipts")
    worker_ids = {str(item.get("signer_id") or "") for item in worker_receipts}
    worker_keys = {
        str(item.get("envelope", {}).get("signer", {}).get("public_key_b64") or "")
        for item in worker_receipts
    }
    if len(worker_ids) != 1 or len(worker_keys) != 1 or "" in worker_keys:
        raise ValueError("one stable generation-worker identity must sign the run")
    if any(
        item.get("signer_id") != task_spec.get("verifier_id")
        for item in correctness_receipts
    ):
        raise ValueError("correctness receipts do not share the pinned verifier identity")
    evaluator_key = str(
        challenge.get("commit", {}).get("signer", {}).get("public_key_b64") or ""
    )
    verifier_key = str(
        task_spec.get("verifier_identity", {}).get("public_key_b64") or ""
    )
    run_key = str(
        run_envelope.get("envelope", {}).get("signer", {}).get("public_key_b64") or ""
    )
    role_keys = {evaluator_key, next(iter(worker_keys)), verifier_key, run_key}
    if "" in role_keys or len(role_keys) != 4:
        raise ValueError(
            "evaluator, worker, verifier, and run coordinator keys must be independent"
        )


def make_index_entry(
    *,
    report: Mapping[str, Any],
    evidence_sha256: str,
    previous_entry_sha256: str | None,
) -> dict[str, Any]:
    require_sha256(evidence_sha256, field_name="evidence blob")
    if previous_entry_sha256 is not None:
        require_sha256(previous_entry_sha256, field_name="previous index entry")
    body = {
        "schema": EVIDENCE_ENTRY_SCHEMA,
        "previous_entry_sha256": previous_entry_sha256,
        "evidence_sha256": evidence_sha256,
        "evidence_class": report.get("evidence_class"),
        "at": report.get("generated_at_unix"),
        "battery_version": report.get("battery_version"),
        "challenge_id": report.get("challenge_id"),
        "comparison_stratum_sha256": report.get("comparison_stratum_sha256"),
        "overall_gap": report.get("overall_gap"),
        "overall_candidate_score": report.get("overall_candidate_score"),
        "effective_n": report.get("effective_n"),
    }
    return {**body, "entry_sha256": sha256_json(body)}


def validate_index_chain(
    entries: Any,
    *,
    max_entries: int,
    initial_previous_sha256: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or len(entries) > max_entries:
        raise ValueError("evidence index retention bound is invalid")
    normalized: list[dict[str, Any]] = []
    previous = initial_previous_sha256
    if previous is not None:
        require_sha256(previous, field_name="pruned index anchor")
    for raw in entries:
        if not isinstance(raw, dict) or set(raw) != {
            "schema",
            "previous_entry_sha256",
            "evidence_sha256",
            "evidence_class",
            "at",
            "battery_version",
            "challenge_id",
            "comparison_stratum_sha256",
            "overall_gap",
            "overall_candidate_score",
            "effective_n",
            "entry_sha256",
        }:
            raise ValueError("evidence index entry is malformed")
        entry = copy.deepcopy(raw)
        if entry.get("schema") != EVIDENCE_ENTRY_SCHEMA:
            raise ValueError("evidence index entry schema is invalid")
        if entry.get("previous_entry_sha256") != previous:
            raise ValueError("evidence index hash chain is broken")
        require_sha256(entry.get("evidence_sha256"), field_name="evidence blob")
        if entry.get("comparison_stratum_sha256") is not None:
            require_sha256(
                entry.get("comparison_stratum_sha256"), field_name="comparison stratum"
            )
        digest = require_sha256(entry.get("entry_sha256"), field_name="index entry")
        body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        if sha256_json(body) != digest:
            raise ValueError("evidence index entry digest mismatch")
        previous = digest
        normalized.append(entry)
    return normalized


def analyze_gap_trend(
    entries: Sequence[Mapping[str, Any]],
    *,
    minimum_runs: int = 5,
    minimum_effect: float = 0.02,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return endpoint telemetry separately from a conservative trend claim."""

    measured = [
        entry
        for entry in entries
        if isinstance(entry.get("overall_gap"), (int, float))
        and not isinstance(entry.get("overall_gap"), bool)
        and math.isfinite(float(entry["overall_gap"]))
    ]
    gaps = [float(entry["overall_gap"]) for entry in measured]
    result: dict[str, Any] = {
        "points": len(entries),
        "measured_points": len(gaps),
        "minimum_runs": minimum_runs,
        "minimum_effect": minimum_effect,
        "alpha": alpha,
        "claim_eligible": False,
        "direction": "insufficient",
    }
    if not gaps:
        result["endpoint_delta"] = None
        return result
    result.update(
        {
            "first_gap": gaps[0],
            "latest_gap": gaps[-1],
            "endpoint_delta": round(gaps[-1] - gaps[0], 6),
        }
    )
    strata = {entry.get("comparison_stratum_sha256") for entry in measured}
    challenges = {entry.get("challenge_id") for entry in measured}
    result["matched_comparison_stratum"] = len(strata) == 1 and None not in strata
    result["unique_challenges"] = len(challenges - {None})
    effective_sizes = {entry.get("effective_n") for entry in measured}
    result["matched_effective_n"] = (
        len(effective_sizes) == 1
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in effective_sizes)
    )
    result["consecutive_nonworsening"] = all(
        later <= earlier + 1e-12
        for earlier, later in zip(gaps[:-1], gaps[1:], strict=True)
    )
    if len(gaps) < minimum_runs:
        return result
    x = [float(index) for index in range(len(gaps))]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(gaps)
    sxx = sum((value - x_mean) ** 2 for value in x)
    slope = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, gaps, strict=True)) / sxx
    intercept = y_mean - slope * x_mean
    residuals = [yv - (intercept + slope * xv) for xv, yv in zip(x, gaps, strict=True)]
    residual_variance = sum(value * value for value in residuals) / max(1, len(gaps) - 2)
    slope_se = math.sqrt(residual_variance / sxx) if sxx else math.inf
    z = slope / slope_se if slope_se > 0 and math.isfinite(slope_se) else (
        -math.inf if slope < 0 else math.inf if slope > 0 else 0.0
    )
    p_value = 2.0 * (1.0 - statistics.NormalDist().cdf(abs(z)))
    ci_low = slope - 1.96 * slope_se
    ci_high = slope + 1.96 * slope_se

    def sampled_slope(values: Sequence[float]) -> float | None:
        sample_x_mean = statistics.fmean(range(len(values)))
        sample_y_mean = statistics.fmean(values)
        sample_sxx = sum(
            (index - sample_x_mean) ** 2 for index in range(len(values))
        )
        if sample_sxx <= 0:
            return None
        return sum(
            (index - sample_x_mean) * (value - sample_y_mean)
            for index, value in enumerate(values)
        ) / sample_sxx

    rng = random.Random(int(sha256_json(gaps)[:16], 16))
    bootstrap_slopes: list[float] = []
    indexed = list(zip(x, gaps, strict=True))
    for _ in range(5000):
        sample = [indexed[rng.randrange(len(indexed))] for _ in indexed]
        sample.sort(key=lambda pair: pair[0])
        distinct_x = {pair[0] for pair in sample}
        if len(distinct_x) < 2:
            continue
        sx_mean = statistics.fmean(pair[0] for pair in sample)
        sy_mean = statistics.fmean(pair[1] for pair in sample)
        sample_sxx = sum((pair[0] - sx_mean) ** 2 for pair in sample)
        bootstrap_slopes.append(
            sum((xv - sx_mean) * (yv - sy_mean) for xv, yv in sample)
            / sample_sxx
        )
    bootstrap_slopes.sort()

    def percentile(values: Sequence[float], quantile: float) -> float:
        if not values:
            return math.nan
        position = (len(values) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(values[lower])
        weight = position - lower
        return float(values[lower] * (1.0 - weight) + values[upper] * weight)

    bootstrap_ci = [
        percentile(bootstrap_slopes, 0.025),
        percentile(bootstrap_slopes, 0.975),
    ]
    if len(gaps) <= 8:
        permutations = itertools.permutations(gaps)
        extreme = 0
        permutation_count = 0
        for permuted in permutations:
            permutation_slope = sampled_slope(permuted)
            if permutation_slope is None:
                continue
            permutation_count += 1
            extreme += int(abs(permutation_slope) >= abs(slope) - 1e-15)
    else:
        permutation_count = 20000
        extreme = 0
        permuted = list(gaps)
        for _ in range(permutation_count):
            rng.shuffle(permuted)
            permutation_slope = sampled_slope(permuted)
            extreme += int(
                permutation_slope is not None
                and abs(permutation_slope) >= abs(slope) - 1e-15
            )
    permutation_p = (extreme + 1) / (permutation_count + 1)
    result.update(
        {
            "slope": round(slope, 8),
            "slope_standard_error": round(slope_se, 8),
            "slope_confidence_95": [round(ci_low, 8), round(ci_high, 8)],
            "normal_approx_p": round(p_value, 8),
            "bootstrap_slope_confidence_95": [
                round(bootstrap_ci[0], 8),
                round(bootstrap_ci[1], 8),
            ],
            "permutation_two_sided_p": round(permutation_p, 8),
            "permutation_samples": permutation_count,
            "effect_threshold_met": gaps[-1] - gaps[0] <= -minimum_effect,
            "significance_threshold_met": (
                permutation_p < alpha
                and math.isfinite(bootstrap_ci[1])
                and bootstrap_ci[1] < 0.0
            ),
        }
    )
    eligible = bool(
        result["matched_comparison_stratum"]
        and result["matched_effective_n"]
        and result["unique_challenges"] >= minimum_runs
        and result["consecutive_nonworsening"]
        and result["effect_threshold_met"]
        and result["significance_threshold_met"]
    )
    result["claim_eligible"] = eligible
    result["direction"] = "closing" if eligible else "not_established"
    return result
