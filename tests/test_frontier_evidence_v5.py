"""Adversarial coverage for the frontier evidence protocol v5 trust boundaries."""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from core.brain.frontier_evidence_v5 import (
    CHALLENGE_COMMIT_SCHEMA,
    CHALLENGE_REVEAL_SCHEMA,
    CORRECTNESS_RECEIPT_SCHEMA,
    EFFECTIVE_RUNTIME_MANIFEST_SCHEMA,
    MATCHED_BUDGET,
    PROTOCOL_MANIFEST,
    PROTOCOL_MANIFEST_SHA256,
    RELEASE_ATTESTATION_SCHEMA,
    RUN_ENVELOPE_SCHEMA,
    SOURCE_IDENTITY_SCHEMA,
    SUPERVISOR_OBSERVATION_SCHEMA,
    TASK_SPEC_SCHEMA,
    WORKER_RECEIPT_SCHEMA,
    analyze_gap_trend,
    build_trust_basis,
    expected_request_id,
    identity_freeze_sha256,
    sha256_json,
    validate_challenge_bundle,
    validate_effective_runtime_manifest,
    validate_source_identity,
)
from core.brain.frontier_gap import (
    BATTERY_VERSION,
    CAPABILITY_EVIDENCE_CLASS,
    CONTROL_EVIDENCE_CLASS,
    MODEL_MANIFEST_SCHEMA,
    MODEL_STABILITY_SCHEMA,
    REFERENCE_SCHEMA,
    REJECTED_EVIDENCE_CLASS,
    SCHEMA_VERSION,
    SOURCE_PROVENANCE_SCHEMA,
    SOURCE_STABILITY_SCHEMA,
    GapLedger,
    SolverObservation,
    battery_manifest,
    build_battery,
    canonical_json_bytes,
    run_battery,
    validate_capability_report,
    validate_reference_artifact,
)
from tools.measure_frontier_gap import (
    ConcurrentEvidenceUpdateError,
    EvidenceBlobStore,
    PriorArtifactCorruptionError,
    _artifact_state_sha256,
    _evidence_persistence_lock,
    _hash_regular_file_beneath,
    _make_tree_read_only,
    _read_prior_snapshot,
    _read_prior_strict,
    _restore_tree_writable,
    _sealed_command_env,
    collect_model_manifest,
)

pytestmark = pytest.mark.unit

TEST_COMMIT = "a" * 40
TEST_TREE = "b" * 40
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def _public_key_b64(key: Ed25519PrivateKey) -> str:
    return base64.b64encode(
        key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode("ascii")


def _sign(
    schema: str,
    payload: dict[str, Any],
    *,
    signer_id: str,
    key: Ed25519PrivateKey,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "signed_payload": copy.deepcopy(payload),
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": signer_id,
            "public_key_b64": _public_key_b64(key),
            "signature_b64": base64.b64encode(
                key.sign(canonical_json_bytes(payload))
            ).decode("ascii"),
        },
    }


@dataclass
class Trust:
    evaluator: Ed25519PrivateKey
    worker: Ed25519PrivateKey
    verifier: Ed25519PrivateKey
    run: Ed25519PrivateKey
    release: Ed25519PrivateKey
    verifier_implementation: str = "7" * 64
    verifier_release: str = "8" * 64

    @classmethod
    def create(cls) -> Trust:
        return cls(*(Ed25519PrivateKey.generate() for _ in range(5)))

    @property
    def evaluator_keys(self) -> dict[str, str]:
        return {"eval-lab": _public_key_b64(self.evaluator)}

    @property
    def worker_keys(self) -> dict[str, str]:
        return {"generation-worker": _public_key_b64(self.worker)}

    @property
    def run_keys(self) -> dict[str, str]:
        return {"run-coordinator": _public_key_b64(self.run)}

    @property
    def release_keys(self) -> dict[str, str]:
        return {"release-authority": _public_key_b64(self.release)}

    @property
    def verifiers(self) -> dict[str, dict[str, str]]:
        return {
            "independent-verifier": {
                "public_key_b64": _public_key_b64(self.verifier),
                "implementation_sha256": self.verifier_implementation,
                "release_sha256": self.verifier_release,
            }
        }

    @property
    def trust_basis(self) -> dict[str, Any]:
        return build_trust_basis(
            evaluator_keys=self.evaluator_keys,
            worker_keys=self.worker_keys,
            verifiers=self.verifiers,
            run_keys=self.run_keys,
            release_keys=self.release_keys,
        )


def _model_manifest(marker: str) -> dict[str, Any]:
    files = [
        {"path": "weights.safetensors", "size": 10, "sha256": marker * 64},
        {"path": "config.json", "size": 20, "sha256": "5" * 64},
        {"path": "tokenizer.json", "size": 30, "sha256": "6" * 64},
    ]
    body = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "model_path": f"/models/{marker}",
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
        "roles": {
            "weights": ["weights.safetensors"],
            "configuration": ["config.json"],
            "tokenizer": ["tokenizer.json"],
            "adapters": [],
        },
    }
    return {**body, "manifest_sha256": sha256_json(body)}


def _model_window(marker: str) -> dict[str, Any]:
    before = _model_manifest(marker)
    after = copy.deepcopy(before)
    body = {"before": before, "after": after}
    return {
        "schema": MODEL_STABILITY_SCHEMA,
        **body,
        "stable": True,
        "window_sha256": sha256_json(body),
    }


def _runtime(subject_id: str, base_model_sha256: str, marker: str) -> dict[str, Any]:
    libraries = {"mlx": "1.2.3", "python": "3.12.4"}
    body = {
        "schema": EFFECTIVE_RUNTIME_MANIFEST_SCHEMA,
        "subject_id": subject_id,
        "base_model_manifest_sha256": base_model_sha256,
        "tokenizer_sha256": marker * 64,
        "prompt_template_sha256": ("f" if marker != "f" else "e") * 64,
        "execution_identity": {
            "worker_implementation_sha256": "1" * 64,
            "python_executable_sha256": "2" * 64,
            "library_lock_sha256": sha256_json(libraries),
            "operating_system": "Darwin",
            "os_release": "25.0",
            "machine": "arm64",
            "python_implementation": "CPython",
            "python_version": "3.12.4",
            "inference_backend": "mlx",
        },
        "inference_libraries": libraries,
        "adapters_sha256": [],
        "steering_sha256": [],
        "modifiers": {"contrastive_decoding": False, "recurrent_loops": 1},
        "cache_policy": {
            "prompt_cache": "disabled",
            "result_cache": "disabled",
            "playbook_cache": "disabled",
            "clear_before_run": True,
        },
        "generation_parameters": copy.deepcopy(MATCHED_BUDGET),
        "runtime_isolation": {
            "fresh_process": True,
            "immutable_source": True,
            "network_enabled": False,
            "tools_enabled": False,
            "sealed_evaluation_enforced": True,
        },
    }
    return {**body, "manifest_sha256": sha256_json(body)}


def _source_identity(trust: Trust) -> dict[str, Any]:
    release_payload = {
        "repository_id": "github.com/bryan/aura",
        "canonical_remote_sha256": "9" * 64,
        "release_ref": "refs/tags/aura-v5-test",
        "release_commit_sha": TEST_COMMIT,
        "issued_at_unix": 90.0,
    }
    release = _sign(
        RELEASE_ATTESTATION_SCHEMA,
        release_payload,
        signer_id="release-authority",
        key=trust.release,
    )
    body = {
        "schema": SOURCE_IDENTITY_SCHEMA,
        **{key: release_payload[key] for key in (
            "repository_id",
            "canonical_remote_sha256",
            "release_ref",
            "release_commit_sha",
        )},
        "commit_sha": TEST_COMMIT,
        "tree_sha": TEST_TREE,
        "release_attestation": release,
        "release_attestation_sha256": sha256_json(release),
        "head_descends_from_release": True,
        "clean": True,
        "immutable_checkout": True,
        "imports_after_verification": True,
    }
    return {**body, "identity_sha256": sha256_json(body)}


def _source_provenance(source_identity: dict[str, Any]) -> dict[str, Any]:
    workspace = hashlib.sha256()
    workspace.update(TEST_COMMIT.encode())
    workspace.update(TEST_TREE.encode())
    workspace.update(hashlib.sha256(b"").digest())
    paths = (
        "tools/measure_frontier_gap.py",
        "core/brain/frontier_gap.py",
        "core/brain/frontier_evidence_v5.py",
        "core/brain/reasoning_amplifier_v2.py",
        "core/brain/verifiers/registry.py",
        "core/brain/llm/mlx_client.py",
        "core/brain/llm/model_registry.py",
        "core/runtime/dynamic_execution_gateway.py",
    )
    return {
        "schema": SOURCE_PROVENANCE_SCHEMA,
        "commit_sha": TEST_COMMIT,
        "tree_sha": TEST_TREE,
        "clean": True,
        "workspace_diff_sha256": EMPTY_SHA,
        "index_diff_sha256": EMPTY_SHA,
        "untracked_content_sha256": EMPTY_SHA,
        "workspace_state_sha256": workspace.hexdigest(),
        "execution_component_sha256": {
            path: hashlib.sha256(path.encode()).hexdigest() for path in paths
        },
        "source_identity": copy.deepcopy(source_identity),
        "issues": [],
    }


def _source_window(source_identity: dict[str, Any]) -> dict[str, Any]:
    before = _source_provenance(source_identity)
    after = copy.deepcopy(before)
    body = {"before": before, "after": after}
    return {
        "schema": SOURCE_STABILITY_SCHEMA,
        **body,
        "stable": True,
        "window_sha256": sha256_json(body),
    }


def _answer_for(item: Any) -> str:
    if item.task_class == "math":
        left, right = re.search(r"Compute (\d+) \* (\d+)", item.prompt).groups()
        return str(int(left) * int(right))
    if item.task_class == "reasoning":
        return re.findall(r"([A-Z][a-z]+) is older than", item.prompt)[0]
    if item.task_class == "coding":
        function_name = re.search(r"`([a-z0-9_]+)\(xs\)`", item.prompt).group(1)
        operation = function_name.partition("_")[0]
        return f"def {function_name}(xs):\n    return {operation}(xs)"
    candidates = (
        "Au", "Mars", "6", "Tokyo", "carbon dioxide", "Na", "Pacific Ocean",
        "90", "Nairobi", "oxygen", "barometer", "2", "South America", "ampere",
        "evaporation", "Wellington",
    )
    return next(candidate for candidate in candidates if item.grade(candidate))


def _challenge(trust: Trust, freeze: str, nonce: bytes) -> dict[str, Any]:
    commit_payload = {
        "challenge_id": f"challenge-{hashlib.sha256(nonce).hexdigest()[:16]}",
        "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
        "identity_freeze_sha256": freeze,
        "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        "committed_at_unix": 100.0,
    }
    commit = _sign(
        CHALLENGE_COMMIT_SCHEMA,
        commit_payload,
        signer_id="eval-lab",
        key=trust.evaluator,
    )
    reveal = _sign(
        CHALLENGE_REVEAL_SCHEMA,
        {
            "challenge_id": commit_payload["challenge_id"],
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "commit_envelope_sha256": sha256_json(commit),
            "revealed_at_unix": 110.0,
            "expires_at_unix": 1000.0,
        },
        signer_id="eval-lab",
        key=trust.evaluator,
    )
    return {"commit": commit, "reveal": reveal}


def _task_spec(
    trust: Trust,
    *,
    seed: int,
    per_class: int,
    nonce: bytes,
    challenge: dict[str, Any],
) -> dict[str, Any]:
    manifest = battery_manifest(
        seed=seed,
        per_class=per_class,
        challenge_nonce=nonce,
    )
    return _sign(
        TASK_SPEC_SCHEMA,
        {
            "battery_version": BATTERY_VERSION,
            "seed": seed,
            "per_class": per_class,
            "challenge_bundle_sha256": sha256_json(challenge),
            "protocol_manifest": copy.deepcopy(PROTOCOL_MANIFEST),
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
            "verifier_identity": {
                "verifier_id": "independent-verifier",
                "public_key_b64": _public_key_b64(trust.verifier),
                "implementation_sha256": trust.verifier_implementation,
                "release_sha256": trust.verifier_release,
            },
            "effective_n": len(manifest["items"]),
            "items": copy.deepcopy(manifest["items"]),
            "issued_at_unix": 111.0,
        },
        signer_id="eval-lab",
        key=trust.evaluator,
    )


def _worker_receipt(
    trust: Trust,
    *,
    run_id: str,
    run_nonce: bytes,
    index: int,
    item: Any,
    output_sha256: str,
    source_identity_sha256: str,
    runtime_manifest_sha256: str,
    model_stability_sha256: str,
    challenge_sha256: str,
) -> dict[str, Any]:
    run_nonce_sha256 = hashlib.sha256(run_nonce).hexdigest()
    started = 120.0 + index
    elapsed = 0.2
    payload = {
        "run_id": run_id,
        "run_nonce_b64": base64.b64encode(run_nonce).decode("ascii"),
        "run_nonce_sha256": run_nonce_sha256,
        "item_id": item.item_id,
        "request_id": expected_request_id(
            run_id=run_id,
            run_nonce_sha256=run_nonce_sha256,
            item_id=item.item_id,
            attempt_index=index,
        ),
        "attempt_index": index,
        "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
        "output_sha256": output_sha256,
        "source_identity_sha256": source_identity_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "model_stability_sha256": model_stability_sha256,
        "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        "challenge_bundle_sha256": challenge_sha256,
        "started_at_unix": started,
        "completed_at_unix": started + elapsed,
        "elapsed_s": elapsed,
        "decoding_parameters": copy.deepcopy(MATCHED_BUDGET),
        "resource_usage": {
            "input_tokens": 16,
            "output_tokens": 8,
            "token_count_method": "tokenizer_exact",
            "candidate_count": 1,
            "generation_calls": 1,
            "tool_calls": 0,
            "network_calls": 0,
            "cache_reads": 0,
            "cache_writes": 0,
            "deadline_exceeded": False,
            "wall_time_s": elapsed,
            "peak_memory_bytes": 4096,
        },
        "sealed_evaluation_enforced": True,
        "fallbacks_used": [],
    }
    return _sign(
        WORKER_RECEIPT_SCHEMA,
        payload,
        signer_id="generation-worker",
        key=trust.worker,
    )


def _supervisor_observation(
    *,
    run_id: str,
    run_nonce: bytes,
    index: int,
    item: Any,
    output_sha256: str,
    elapsed: float = 0.21,
) -> dict[str, Any]:
    run_nonce_sha256 = hashlib.sha256(run_nonce).hexdigest()
    return {
        "schema": SUPERVISOR_OBSERVATION_SCHEMA,
        "run_id": run_id,
        "run_nonce_sha256": run_nonce_sha256,
        "item_id": item.item_id,
        "request_id": expected_request_id(
            run_id=run_id,
            run_nonce_sha256=run_nonce_sha256,
            item_id=item.item_id,
            attempt_index=index,
        ),
        "attempt_index": index,
        "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
        "output_sha256": output_sha256,
        "observed_wall_time_s": elapsed,
        "deadline_s": MATCHED_BUDGET["hard_timeout_s"],
        "deadline_exceeded": False,
        "process_pid": 12345,
        "process_running_after_response": True,
        "observed_at_unix": 130.0 + index,
    }


def _correctness_receipt(
    trust: Trust,
    *,
    run_id: str,
    item: Any,
    output_sha256: str,
    task_spec_sha256: str,
    challenge_sha256: str,
    correct: bool,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "item_id": item.item_id,
        "output_sha256": output_sha256,
        "task_spec_sha256": task_spec_sha256,
        "challenge_bundle_sha256": challenge_sha256,
        "verifier_implementation_sha256": trust.verifier_implementation,
        "verifier_release_sha256": trust.verifier_release,
        "expected_answer_commitment_sha256": item.expected_answer_commitment_sha256,
        "hidden_case_commitment_sha256": item.hidden_case_commitment_sha256,
        "correct": correct,
        "checked": True,
        "grader_execution_sha256": sha256_json(
            {"item_id": item.item_id, "output_sha256": output_sha256, "correct": correct}
        ),
        "graded_at_unix": 140.0,
    }
    return _sign(
        CORRECTNESS_RECEIPT_SCHEMA,
        payload,
        signer_id="independent-verifier",
        key=trust.verifier,
    )


def _run_envelope(
    trust: Trust,
    *,
    run_id: str,
    run_nonce: bytes,
    task_spec: dict[str, Any],
    challenge: dict[str, Any],
    source_identity_sha256: str,
    runtime_manifest_sha256: str,
    reference_artifact_sha256: str,
    trust_basis_sha256: str,
    worker_receipts: list[dict[str, Any]],
    supervisor_observations: list[dict[str, Any]],
    correctness_receipts: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    run_nonce_sha256 = hashlib.sha256(run_nonce).hexdigest()
    usage = [receipt["signed_payload"]["resource_usage"] for receipt in worker_receipts]
    supervised_wall = [
        float(observation["observed_wall_time_s"])
        for observation in supervisor_observations
    ]
    payload = {
        "run_id": run_id,
        "run_nonce_b64": base64.b64encode(run_nonce).decode("ascii"),
        "run_nonce_sha256": run_nonce_sha256,
        "task_spec_sha256": sha256_json(task_spec),
        "challenge_bundle_sha256": sha256_json(challenge),
        "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        "source_identity_sha256": source_identity_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "reference_artifact_sha256": reference_artifact_sha256,
        "trust_basis_sha256": trust_basis_sha256,
        "worker_receipt_sha256": [sha256_json(receipt) for receipt in worker_receipts],
        "supervisor_observation_sha256": [
            sha256_json(observation) for observation in supervisor_observations
        ],
        "correctness_receipt_sha256": [
            sha256_json(receipt) for receipt in correctness_receipts
        ],
        "outputs_sha256": sha256_json(outputs),
        "worker_signer_ids": ["generation-worker"] * len(worker_receipts),
        "verifier_id": "independent-verifier",
        "started_at_unix": 115.0,
        "completed_at_unix": 160.0,
        "budget_summary": {
            "item_count": len(usage),
            "total_input_tokens": sum(item["input_tokens"] for item in usage),
            "total_output_tokens": sum(item["output_tokens"] for item in usage),
            "total_candidate_count": sum(item["candidate_count"] for item in usage),
            "total_generation_calls": sum(item["generation_calls"] for item in usage),
            "maximum_item_wall_time_s": max(item["wall_time_s"] for item in usage),
            "maximum_supervisor_wall_time_s": max(supervised_wall),
            "all_within_budget": True,
        },
    }
    return _sign(
        RUN_ENVELOPE_SCHEMA,
        payload,
        signer_id="run-coordinator",
        key=trust.run,
    )


def _reference_artifact(
    trust: Trust,
    *,
    seed: int = 41,
    per_class: int = 2,
    nonce: bytes = b"r" * 32,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_identity = _source_identity(trust)
    candidate_model = _model_window("c")
    reference_model = _model_window("d")
    candidate_runtime = _runtime(
        "aura-candidate", candidate_model["before"]["manifest_sha256"], "a"
    )
    reference_runtime = _runtime(
        "frontier-model", reference_model["before"]["manifest_sha256"], "b"
    )
    freeze = identity_freeze_sha256(
        source_identity_sha256=source_identity["identity_sha256"],
        candidate_runtime_sha256=candidate_runtime["manifest_sha256"],
        reference_runtime_sha256=reference_runtime["manifest_sha256"],
    )
    challenge = _challenge(trust, freeze, nonce)
    task_spec = _task_spec(
        trust,
        seed=seed,
        per_class=per_class,
        nonce=nonce,
        challenge=challenge,
    )
    items = build_battery(seed=seed, per_class=per_class, challenge_nonce=nonce)
    source_digest = "e" * 64
    reference_context = sha256_json(
        {
            "model_id": "frontier-model",
            "source": "independent-evaluation-lab",
            "source_identity_sha256": source_digest,
            "effective_runtime_manifest_sha256": reference_runtime["manifest_sha256"],
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        }
    )
    run_id = "1" * 64
    run_nonce = b"w" * 32
    outputs: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    supervisors: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        answer = _answer_for(item)
        output_sha = hashlib.sha256(answer.encode()).hexdigest()
        outputs.append(
            {"index": index, "item_id": item.item_id, "answer": answer, "output_sha256": output_sha}
        )
        workers.append(
            _worker_receipt(
                trust,
                run_id=run_id,
                run_nonce=run_nonce,
                index=index,
                item=item,
                output_sha256=output_sha,
                source_identity_sha256=source_digest,
                runtime_manifest_sha256=reference_runtime["manifest_sha256"],
                model_stability_sha256=reference_model["window_sha256"],
                challenge_sha256=sha256_json(challenge),
            )
        )
        supervisors.append(
            _supervisor_observation(
                run_id=run_id,
                run_nonce=run_nonce,
                index=index,
                item=item,
                output_sha256=output_sha,
            )
        )
        correctness.append(
            _correctness_receipt(
                trust,
                run_id=run_id,
                item=item,
                output_sha256=output_sha,
                task_spec_sha256=sha256_json(task_spec),
                challenge_sha256=sha256_json(challenge),
                correct=True,
            )
        )
    run = _run_envelope(
        trust,
        run_id=run_id,
        run_nonce=run_nonce,
        task_spec=task_spec,
        challenge=challenge,
        source_identity_sha256=source_digest,
        runtime_manifest_sha256=reference_runtime["manifest_sha256"],
        reference_artifact_sha256=reference_context,
        trust_basis_sha256=trust.trust_basis["manifest_sha256"],
        worker_receipts=workers,
        supervisor_observations=supervisors,
        correctness_receipts=correctness,
        outputs=outputs,
    )
    signed = {
        "model_id": "frontier-model",
        "source": "independent-evaluation-lab",
        "measured_at_unix": 170.0,
        "battery_version": BATTERY_VERSION,
        "seed": seed,
        "per_class": per_class,
        "scores": {task_class: 1.0 for task_class in ("math", "reasoning", "coding", "factual")},
        "budget": copy.deepcopy(MATCHED_BUDGET),
        "protocol_manifest": copy.deepcopy(PROTOCOL_MANIFEST),
        "trust_basis": trust.trust_basis,
        "challenge": challenge,
        "task_spec": task_spec,
        "effective_runtime_manifest": reference_runtime,
        "model_stability": reference_model,
        "source_identity_sha256": source_digest,
        "reference_context_sha256": reference_context,
        "outputs": outputs,
        "item_receipts": workers,
        "supervisor_observations": supervisors,
        "correctness_receipts": correctness,
        "run_envelope": run,
    }
    return (
        _sign(REFERENCE_SCHEMA, signed, signer_id="eval-lab", key=trust.evaluator),
        source_identity,
        candidate_runtime,
        candidate_model,
    )


def _source_tree_resolver(commit_sha: str) -> str:
    if commit_sha != TEST_COMMIT:
        raise ValueError("unknown test commit")
    return TEST_TREE


def _source_component_resolver(commit_sha: str, relative_path: str) -> str:
    if commit_sha != TEST_COMMIT:
        raise ValueError("unknown test commit")
    return hashlib.sha256(relative_path.encode()).hexdigest()


def _capability_report(
    trust: Trust,
    *,
    correct_per_class: int = 2,
    seed: int = 41,
    per_class: int = 2,
) -> dict[str, Any]:
    artifact, source_identity, candidate_runtime, candidate_model = _reference_artifact(
        trust,
        seed=seed,
        per_class=per_class,
    )
    reference = validate_reference_artifact(
        artifact,
        seed=seed,
        per_class=per_class,
        trusted_evaluator_keys=trust.evaluator_keys,
        trusted_worker_keys=trust.worker_keys,
        trusted_verifiers=trust.verifiers,
        trusted_run_keys=trust.run_keys,
        trusted_release_keys=trust.release_keys,
        expected_identity_freeze_sha256=artifact["signed_payload"]["challenge"]["commit"][
            "signed_payload"
        ]["identity_freeze_sha256"],
    )
    items = build_battery(
        seed=seed,
        per_class=per_class,
        challenge_nonce=reference.challenge_nonce,
    )
    seen = dict.fromkeys(("math", "reasoning", "coding", "factual"), 0)
    run_id = "3" * 64
    run_nonce = b"c" * 32
    cursor = 0

    async def solver(prompt: str, task_type: str) -> SolverObservation:
        nonlocal cursor
        item = items[cursor]
        index = cursor
        cursor += 1
        assert (prompt, task_type) == (item.prompt, item.task_type)
        correct = seen[item.task_class] < correct_per_class
        seen[item.task_class] += 1
        answer = _answer_for(item) if correct else "incorrect"
        output_sha = hashlib.sha256(answer.encode()).hexdigest()
        return SolverObservation(
            answer=answer,
            verified=True,
            receipt=_worker_receipt(
                trust,
                run_id=run_id,
                run_nonce=run_nonce,
                index=index,
                item=item,
                output_sha256=output_sha,
                source_identity_sha256=source_identity["identity_sha256"],
                runtime_manifest_sha256=candidate_runtime["manifest_sha256"],
                model_stability_sha256=candidate_model["window_sha256"],
                challenge_sha256=sha256_json(reference.challenge),
            ),
            supervisor_observation=_supervisor_observation(
                run_id=run_id,
                run_nonce=run_nonce,
                index=index,
                item=item,
                output_sha256=output_sha,
            ),
        )

    report = asyncio.run(
        run_battery(
            solver,
            seed=seed,
            per_class=per_class,
            reference=reference,
            grade_to_foundry=False,
        )
    )
    correctness: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for item, evidence in zip(items, report["items"], strict=True):
        receipt = _correctness_receipt(
            trust,
            run_id=run_id,
            item=item,
            output_sha256=evidence["output_sha256"],
            task_spec_sha256=sha256_json(reference.task_spec),
            challenge_sha256=sha256_json(reference.challenge),
            correct=bool(evidence["correct"]),
        )
        evidence["correctness_receipt"] = receipt
        correctness.append(receipt)
        outputs.append(
            {
                "index": evidence["index"],
                "item_id": evidence["item_id"],
                "answer": evidence["answer"],
                "output_sha256": evidence["output_sha256"],
            }
        )
    report["correctness_receipts"] = correctness
    report["run_envelope"] = _run_envelope(
        trust,
        run_id=run_id,
        run_nonce=run_nonce,
        task_spec=reference.task_spec,
        challenge=reference.challenge,
        source_identity_sha256=source_identity["identity_sha256"],
        runtime_manifest_sha256=candidate_runtime["manifest_sha256"],
        reference_artifact_sha256=sha256_json(reference.to_dict()),
        trust_basis_sha256=trust.trust_basis["manifest_sha256"],
        worker_receipts=[evidence["receipt"] for evidence in report["items"]],
        supervisor_observations=[
            evidence["supervisor_observation"] for evidence in report["items"]
        ],
        correctness_receipts=correctness,
        outputs=outputs,
    )
    source_window = _source_window(source_identity)
    report.update(
        {
            "solver_mode": "amplifier_mlx_worker_v5",
            "measurement_subject": f"aura_model:{candidate_runtime['manifest_sha256']}",
            "evidence_class": CAPABILITY_EVIDENCE_CLASS,
            "capability_claim_eligible": True,
            "execution": {
                "attempted": len(items),
                "completed": len(items),
                "failed": 0,
                "invalid": 0,
                "empty": 0,
                "unverified": 0,
                "fallback_items": 0,
                "disqualifying_fallbacks": 0,
                "errors": [],
            },
            "source_provenance": source_window["after"],
            "source_stability": source_window,
            "source_identity": source_identity,
            "candidate_model": candidate_model,
            "effective_runtime_manifest": candidate_runtime,
            "trust_basis": trust.trust_basis,
            "eligibility_reasons": [],
            "reference_validation_error": None,
        }
    )
    return report


def _validate_report(report: dict[str, Any], trust: Trust) -> dict[str, Any]:
    return validate_capability_report(
        report,
        trusted_evaluator_keys=trust.evaluator_keys,
        trusted_worker_keys=trust.worker_keys,
        trusted_verifiers=trust.verifiers,
        trusted_run_keys=trust.run_keys,
        trusted_release_keys=trust.release_keys,
        source_tree_resolver=_source_tree_resolver,
        source_component_resolver=_source_component_resolver,
    )


def test_complete_v5_reference_and_candidate_validate_end_to_end() -> None:
    trust = Trust.create()
    report = _capability_report(trust)
    validated = _validate_report(report, trust)
    assert validated["overall_candidate_score"] == 1.0
    assert validated["overall_gap"] == 0.0


def test_worker_receipt_binds_output_nonce_runtime_source_and_actual_budget() -> None:
    trust = Trust.create()
    report = _capability_report(trust)
    receipt = report["items"][0]["receipt"]
    receipt["signed_payload"]["output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signature verification|output_sha256"):
        _validate_report(report, trust)

    report = _capability_report(trust)
    receipt = report["items"][0]["receipt"]
    receipt["signed_payload"]["resource_usage"]["output_tokens"] = 257
    report["items"][0]["receipt"] = _sign(
        WORKER_RECEIPT_SCHEMA,
        receipt["signed_payload"],
        signer_id="generation-worker",
        key=trust.worker,
    )
    with pytest.raises(ValueError, match="token budget"):
        _validate_report(report, trust)


def test_supervisor_observation_independently_enforces_wall_clock_budget() -> None:
    trust = Trust.create()
    report = _capability_report(trust)
    report["items"][0]["supervisor_observation"]["observed_wall_time_s"] = 21.0
    report["items"][0]["supervisor_observation"]["deadline_exceeded"] = True

    with pytest.raises(ValueError, match="deadline"):
        _validate_report(report, trust)


def test_run_signer_binds_nonce_and_enforces_generation_then_grading_chronology() -> None:
    trust = Trust.create()
    report = _capability_report(trust)
    run = report["run_envelope"]
    run["signed_payload"]["run_nonce_b64"] = base64.b64encode(b"x" * 32).decode(
        "ascii"
    )
    report["run_envelope"] = _sign(
        RUN_ENVELOPE_SCHEMA,
        run["signed_payload"],
        signer_id="run-coordinator",
        key=trust.run,
    )
    with pytest.raises(ValueError, match="256-bit nonce"):
        _validate_report(report, trust)

    report = _capability_report(trust)
    correctness = report["correctness_receipts"][0]
    correctness["signed_payload"]["graded_at_unix"] = 100.0
    replacement = _sign(
        CORRECTNESS_RECEIPT_SCHEMA,
        correctness["signed_payload"],
        signer_id="independent-verifier",
        key=trust.verifier,
    )
    report["correctness_receipts"][0] = replacement
    report["items"][0]["correctness_receipt"] = replacement
    run_nonce = base64.b64decode(
        report["run_envelope"]["signed_payload"]["run_nonce_b64"]
    )
    outputs = [
        {
            "index": item["index"],
            "item_id": item["item_id"],
            "answer": item["answer"],
            "output_sha256": item["output_sha256"],
        }
        for item in report["items"]
    ]
    report["run_envelope"] = _run_envelope(
        trust,
        run_id=report["run_envelope"]["signed_payload"]["run_id"],
        run_nonce=run_nonce,
        task_spec=report["task_spec"],
        challenge=report["challenge"],
        source_identity_sha256=report["source_identity"]["identity_sha256"],
        runtime_manifest_sha256=report["effective_runtime_manifest"][
            "manifest_sha256"
        ],
        reference_artifact_sha256=report["reference_artifact_sha256"],
        trust_basis_sha256=report["trust_basis"]["manifest_sha256"],
        worker_receipts=[item["receipt"] for item in report["items"]],
        supervisor_observations=[
            item["supervisor_observation"] for item in report["items"]
        ],
        correctness_receipts=report["correctness_receipts"],
        outputs=outputs,
    )
    with pytest.raises(ValueError, match="chronology"):
        _validate_report(report, trust)


def test_run_signer_must_be_cryptographically_independent_from_worker() -> None:
    trust = Trust.create()
    report = _capability_report(trust)
    payload = report["run_envelope"]["signed_payload"]
    report["run_envelope"] = _sign(
        RUN_ENVELOPE_SCHEMA,
        payload,
        signer_id="candidate-run-coordinator",
        key=trust.worker,
    )
    trusted_run_keys = {
        **trust.run_keys,
        "candidate-run-coordinator": _public_key_b64(trust.worker),
    }
    with pytest.raises(ValueError, match="cryptographic key|keys must be independent"):
        validate_capability_report(
            report,
            trusted_evaluator_keys=trust.evaluator_keys,
            trusted_worker_keys=trust.worker_keys,
            trusted_verifiers=trust.verifiers,
            trusted_run_keys=trusted_run_keys,
            trusted_release_keys=trust.release_keys,
            source_tree_resolver=_source_tree_resolver,
            source_component_resolver=_source_component_resolver,
        )


def test_external_trust_basis_is_bound_and_rejects_cross_role_key_reuse() -> None:
    trust = Trust.create()
    report = _capability_report(trust)
    report["trust_basis"]["worker_keys"]["generation-worker"] = _public_key_b64(
        Ed25519PrivateKey.generate()
    )
    with pytest.raises(ValueError, match="trust basis differs"):
        _validate_report(report, trust)

    with pytest.raises(ValueError, match="reuses a cryptographic key"):
        build_trust_basis(
            evaluator_keys=trust.evaluator_keys,
            worker_keys=trust.worker_keys,
            verifiers=trust.verifiers,
            run_keys=trust.run_keys,
            release_keys={"release-authority": _public_key_b64(trust.evaluator)},
        )


def test_task_spec_pins_grader_hidden_commitments_and_independent_verifier() -> None:
    trust = Trust.create()
    report = _capability_report(trust)
    task = report["task_spec"]
    task["signed_payload"]["items"][0]["grader_implementation_sha256"] = "0" * 64
    report["task_spec"] = _sign(
        TASK_SPEC_SCHEMA,
        task["signed_payload"],
        signer_id="eval-lab",
        key=trust.evaluator,
    )
    with pytest.raises(ValueError, match="task or challenge differs|task specification"):
        _validate_report(report, trust)

    report = _capability_report(trust)
    wrong_pins = copy.deepcopy(trust.verifiers)
    wrong_pins["independent-verifier"]["implementation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="trust basis|verifier identity"):
        validate_capability_report(
            report,
            trusted_evaluator_keys=trust.evaluator_keys,
            trusted_worker_keys=trust.worker_keys,
            trusted_verifiers=wrong_pins,
            trusted_run_keys=trust.run_keys,
            trusted_release_keys=trust.release_keys,
            source_tree_resolver=_source_tree_resolver,
            source_component_resolver=_source_component_resolver,
        )


def test_effective_runtime_requires_exact_model_modifiers_versions_and_sealing() -> None:
    runtime = _runtime("subject", "c" * 64, "a")
    assert validate_effective_runtime_manifest(runtime) == runtime
    weakened = copy.deepcopy(runtime)
    weakened["cache_policy"]["prompt_cache"] = "enabled"
    body = {key: value for key, value in weakened.items() if key != "manifest_sha256"}
    weakened["manifest_sha256"] = sha256_json(body)
    with pytest.raises(ValueError, match="cache policy"):
        validate_effective_runtime_manifest(weakened)

    incomplete = copy.deepcopy(runtime)
    incomplete["execution_identity"].pop("worker_implementation_sha256")
    body = {key: value for key, value in incomplete.items() if key != "manifest_sha256"}
    incomplete["manifest_sha256"] = sha256_json(body)
    with pytest.raises(ValueError, match="execution identity"):
        validate_effective_runtime_manifest(incomplete)


def test_source_identity_requires_signed_release_ancestry_and_preimport_checkout() -> None:
    trust = Trust.create()
    identity = _source_identity(trust)
    assert validate_source_identity(
        identity, trusted_release_keys=trust.release_keys
    ) == identity
    with pytest.raises(ValueError, match="explicitly trusted"):
        validate_source_identity(identity)
    weakened = copy.deepcopy(identity)
    weakened["imports_after_verification"] = False
    body = {key: value for key, value in weakened.items() if key != "identity_sha256"}
    weakened["identity_sha256"] = sha256_json(body)
    with pytest.raises(ValueError, match="imports_after_verification"):
        validate_source_identity(weakened, trusted_release_keys=trust.release_keys)


def test_execution_attestation_is_separate_from_short_answer_correctness() -> None:
    trust = Trust.create()
    report = _capability_report(trust)
    assert report["items"][0]["task_type"] in {"math", "logic", "factual", "code"}
    report["items"][0]["verified"] = False
    report["items"][0]["execution_attested"] = False
    report["execution"]["unverified"] = 1
    validated = _validate_report(report, trust)
    assert validated["items"][0]["correct"] is True


def test_reference_round_trip_preserves_exact_signed_payload_without_normalization() -> None:
    trust = Trust.create()
    artifact, _source, _runtime_manifest, _model = _reference_artifact(trust)
    artifact["signed_payload"]["scores"] = {
        task_class: 1 for task_class in ("math", "reasoning", "coding", "factual")
    }
    artifact = _sign(
        REFERENCE_SCHEMA,
        artifact["signed_payload"],
        signer_id="eval-lab",
        key=trust.evaluator,
    )
    validated = validate_reference_artifact(
        artifact,
        seed=41,
        per_class=2,
        trusted_evaluator_keys=trust.evaluator_keys,
        trusted_worker_keys=trust.worker_keys,
        trusted_verifiers=trust.verifiers,
        trusted_run_keys=trust.run_keys,
        trusted_release_keys=trust.release_keys,
    )
    assert validated.to_dict() == artifact

    noncanonical = copy.deepcopy(artifact)
    noncanonical["signed_payload"]["source"] = " independent-evaluation-lab "
    noncanonical = _sign(
        REFERENCE_SCHEMA,
        noncanonical["signed_payload"],
        signer_id="eval-lab",
        key=trust.evaluator,
    )
    with pytest.raises(ValueError, match="noncanonical"):
        validate_reference_artifact(
            noncanonical,
            seed=41,
            per_class=2,
            trusted_evaluator_keys=trust.evaluator_keys,
            trusted_worker_keys=trust.worker_keys,
            trusted_verifiers=trust.verifiers,
            trusted_run_keys=trust.run_keys,
            trusted_release_keys=trust.release_keys,
        )


def test_malformed_or_wrong_schema_prior_artifact_fails_closed_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "latest.json"
    original = b'{"schema": "wrong", broken'
    path.write_bytes(original)
    with pytest.raises(PriorArtifactCorruptionError) as captured:
        _read_prior_strict(path)
    assert captured.value.raw_bytes == original
    assert path.read_bytes() == original


def test_evidence_blob_flush_round_trips_exact_disk_bytes(tmp_path: Path) -> None:
    from core.runtime.file_write_gateway import get_file_write_gateway

    store = EvidenceBlobStore(tmp_path / "evidence-v5")
    payload = {"schema_version": 5, "items": [{"answer": "exact"}]}
    digest = sha256_json(payload)
    store.stage(digest, payload)
    asyncio.run(store.flush(get_file_write_gateway()))

    assert store.pending == {}
    assert store.resolve(digest) == payload
    on_disk = (store.root / f"{digest}.json").read_bytes()
    envelope = json.loads(on_disk)
    assert sha256_json(envelope) != digest
    assert sha256_json(envelope["payload"]) == digest


def test_exclusive_blob_publication_never_replaces_the_winner(tmp_path: Path) -> None:
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    path = tmp_path / "blob.json"

    async def publish() -> tuple[bool, bool]:
        first, second = await asyncio.gather(
            gateway.write_bytes_if_absent_async(
                path, b"first", source="frontier_gap.test"
            ),
            gateway.write_bytes_if_absent_async(
                path, b"second", source="frontier_gap.test"
            ),
        )
        return first, second

    outcomes = asyncio.run(publish())
    assert sorted(outcomes) == [False, True]
    assert path.read_bytes() == (b"first" if outcomes[0] else b"second")


def test_frontier_artifact_compare_and_swap_detects_a_changed_head(
    tmp_path: Path,
) -> None:
    path = tmp_path / "latest.json"
    prior, expected_state = _read_prior_snapshot(path)
    assert prior == {}
    path.write_text('{"schema":"changed"}', encoding="utf-8")

    async def commit_check() -> None:
        async with _evidence_persistence_lock(path):
            if _artifact_state_sha256(path) != expected_state:
                raise ConcurrentEvidenceUpdateError("changed")

    with pytest.raises(ConcurrentEvidenceUpdateError, match="changed"):
        asyncio.run(commit_check())


def test_read_only_checkout_never_chmods_an_external_symlink_target(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("outside", encoding="utf-8")
    external.chmod(0o600)
    (checkout / "escape").symlink_to(external)

    with pytest.raises(RuntimeError, match="unsafe symlink"):
        _make_tree_read_only(checkout)
    assert external.stat().st_mode & 0o777 == 0o600


def test_read_only_checkout_rejects_a_broken_symlink_explicitly(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "missing-link").symlink_to("not-present")

    with pytest.raises(RuntimeError, match="broken symlink"):
        _make_tree_read_only(checkout)


def test_read_only_checkout_allows_internal_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    target = checkout / "nested" / "asset.txt"
    target.parent.mkdir(parents=True)
    target.write_text("inside", encoding="utf-8")
    (checkout / "asset-link").symlink_to(Path("nested") / "asset.txt")

    try:
        _make_tree_read_only(checkout)
        assert not (target.stat().st_mode & 0o222)
        assert (checkout / "asset-link").is_symlink()
    finally:
        _restore_tree_writable(checkout)


def test_model_manifest_hashes_exact_regular_files_without_following_links(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    manifest = collect_model_manifest(str(model))
    assert manifest["file_count"] == 3
    assert manifest["total_bytes"] == len(b"weights") + 4
    assert manifest["roles"]["weights"] == ["weights.safetensors"]

    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.bin").write_bytes(b"secret")
    (model / "escape").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        collect_model_manifest(str(model))
    with pytest.raises(OSError):
        _hash_regular_file_beneath(model, "escape/secret.bin")


def test_signed_evidence_commands_receive_only_role_scoped_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("AURA_FRONTIER_WORKER_MODEL", "/models/aura")
    monkeypatch.setenv("AURA_FRONTIER_VERIFIER_KEY", "verifier-secret")
    monkeypatch.setenv("AURA_FRONTIER_COMMON_RUN", "shared")

    worker = _sealed_command_env("worker")
    assert "OPENAI_API_KEY" not in worker
    assert worker["AURA_FRONTIER_WORKER_MODEL"] == "/models/aura"
    assert "AURA_FRONTIER_VERIFIER_KEY" not in worker
    assert worker["AURA_FRONTIER_COMMON_RUN"] == "shared"
    assert worker["PYTHONNOUSERSITE"] == "1"


def test_trend_requires_repeated_matched_unique_significant_monotonic_runs() -> None:
    stratum = "a" * 64
    closing = [0.50, 0.42, 0.34, 0.26, 0.18]
    entries = [
        {
            "overall_gap": gap,
            "comparison_stratum_sha256": stratum,
            "challenge_id": f"challenge-{index}",
            "effective_n": 20,
        }
        for index, gap in enumerate(closing)
    ]
    trend = analyze_gap_trend(entries)
    assert trend["endpoint_delta"] == pytest.approx(-0.32)
    assert trend["direction"] == "closing"
    assert trend["claim_eligible"] is True

    regressing = copy.deepcopy(entries)
    regressing[2]["overall_gap"] = 0.48
    rejected = analyze_gap_trend(regressing)
    assert rejected["endpoint_delta"] < 0
    assert rejected["consecutive_nonworsening"] is False
    assert rejected["direction"] == "not_established"

    mixed = copy.deepcopy(entries)
    mixed[-1]["comparison_stratum_sha256"] = "b" * 64
    assert analyze_gap_trend(mixed)["claim_eligible"] is False


def test_battery_uses_unique_effective_samples_and_commit_reveal_binding() -> None:
    trust = Trust.create()
    nonce = b"n" * 32
    items = build_battery(seed=77, per_class=5, challenge_nonce=nonce)
    assert len(items) == 20
    assert len({item.item_id for item in items}) == 20
    assert len({hashlib.sha256(item.prompt.encode()).hexdigest() for item in items}) == 20
    manifest = battery_manifest(seed=77, per_class=5, challenge_nonce=nonce)
    assert manifest["effective_n"] == 20
    challenge = _challenge(trust, "f" * 64, nonce)
    validated = validate_challenge_bundle(
        challenge,
        trusted_evaluator_keys=trust.evaluator_keys,
        expected_identity_freeze_sha256="f" * 64,
    )
    assert validated["nonce"] == nonce
    with pytest.raises(ValueError, match="expired"):
        validate_challenge_bundle(
            challenge,
            trusted_evaluator_keys=trust.evaluator_keys,
            expected_identity_freeze_sha256="f" * 64,
            verification_time_unix=2_000.0,
            require_fresh=True,
        )
    challenge["reveal"]["signed_payload"]["nonce_b64"] = base64.b64encode(
        b"x" * 32
    ).decode("ascii")
    with pytest.raises(ValueError, match="signature verification|commitment"):
        validate_challenge_bundle(
            challenge,
            trusted_evaluator_keys=trust.evaluator_keys,
            expected_identity_freeze_sha256="f" * 64,
        )


def test_content_addressed_ledgers_are_bounded_hash_chained_and_retain_rejections(
    tmp_path: Path,
) -> None:
    store = EvidenceBlobStore(tmp_path / "evidence-v5")
    ledger = GapLedger(
        evidence_class=REJECTED_EVIDENCE_CLASS,
        capability_claim_eligible=False,
        max_entries=2,
    )
    for index in range(3):
        report = {
            "schema_version": SCHEMA_VERSION,
            "battery_version": BATTERY_VERSION,
            "generated_at_unix": float(index),
            "evidence_class": REJECTED_EVIDENCE_CLASS,
            "capability_claim_eligible": False,
            "challenge_id": f"challenge-{index}",
            "comparison_stratum_sha256": "a" * 64,
            "overall_gap": 0.5,
            "overall_candidate_score": 0.5,
            "effective_n": 1,
            "items": [{"answer": f"retained-output-{index}"}],
        }
        ledger.add(report, evidence_blob_writer=store.stage)
    assert ledger.pruned_count == 1
    assert len(ledger.runs) == 2
    assert store.resolve(ledger.runs[-1]["evidence_sha256"])["items"][0][
        "answer"
    ] == "retained-output-2"
    restored = GapLedger.from_dict(
        ledger.to_dict(),
        evidence_class=REJECTED_EVIDENCE_CLASS,
        evidence_blob_resolver=store.resolve,
    )
    assert restored.to_dict() == ledger.to_dict()

    tampered = ledger.to_dict()
    tampered["runs"][-1]["previous_entry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash chain"):
        GapLedger.from_dict(
            tampered,
            evidence_class=REJECTED_EVIDENCE_CLASS,
            evidence_blob_resolver=store.resolve,
        )


def test_control_ledger_cannot_be_reclassified_as_capability() -> None:
    report = {
        "schema_version": SCHEMA_VERSION,
        "battery_version": BATTERY_VERSION,
        "generated_at_unix": 1.0,
        "evidence_class": CONTROL_EVIDENCE_CLASS,
        "capability_claim_eligible": False,
        "challenge_id": None,
        "comparison_stratum_sha256": None,
        "overall_gap": None,
        "overall_candidate_score": 1.0,
        "effective_n": 1,
    }
    blobs: dict[str, dict[str, Any]] = {}
    ledger = GapLedger(
        evidence_class=CONTROL_EVIDENCE_CLASS,
        capability_claim_eligible=False,
    )
    ledger.add(
        report,
        evidence_blob_writer=lambda digest, payload: blobs.setdefault(digest, payload),
    )
    with pytest.raises(ValueError, match="class"):
        GapLedger.from_dict(
            ledger.to_dict(),
            evidence_class=CAPABILITY_EVIDENCE_CLASS,
            evidence_blob_resolver=lambda digest: blobs[digest],
        )
