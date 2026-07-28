"""Proof contracts for generic verified training task envelopes."""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import inspect
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    TASK_ISSUER,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.learning.answer_channel_curriculum import (
    TASK_GENERATORS as ANSWER_CHANNEL_GENERATORS,
)
from core.learning.recurrence_curriculum import (
    TASK_GENERATORS as RECURRENCE_GENERATORS,
)
from core.learning.verifiable_tasks import KNOWLEDGE_FREE, VerifiableTask
from core.learning.verified_training_task import (
    PASS_NAMES,
    SCORER_REGISTRY,
    PublicVerifiedTrainingTask,
    VerifiedTrainingTaskError,
    assemble_answer_authority,
    build_verified_training_task,
    prepare_answer_authority_payload,
    score_verified_training_outputs,
    scorer_registry_identity,
    seal_training_output,
    validate_answer_authority,
    validate_public_training_task,
    validate_scorer_registry_identity,
)

SIGNED_AT = 1_800_000_200
NONCE = b"proof-grade-answer-nonce-material" * 2


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


def _pin(role: str, key: Ed25519PrivateKey) -> dict[str, str]:
    raw = _public_raw(key)
    return {
        "signer_id": f"{role}-signer",
        "organization_id": f"{role}-organization",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": hashlib.sha256(f"{role}:impl".encode()).hexdigest(),
        "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
        "custody_class": "test_fixture",
        "custody_evidence_sha256": hashlib.sha256(
            f"{role}:custody".encode()
        ).hexdigest(),
    }


@pytest.fixture
def trust() -> tuple[Any, dict[str, Ed25519PrivateKey]]:
    root = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "verified-training-task-test",
        "policy_revision": 1,
        "campaign_name": "verified-training-task-test",
        "protocol_sha256": "7" * 64,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": 1_800_000_000,
        "not_before_unix": 1_800_000_100,
        "expires_at_unix": 1_800_086_400,
        "roles": {
            role: _pin(role, role_keys[role]) for role in CAMPAIGN_TRUST_ROLES
        },
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name="verified-training-task-test",
        now_unix=SIGNED_AT,
    )
    return policy, role_keys


def _verifiable(
    *,
    task_id: str = "math-proof-1",
    expected: Any = 42,
    grader: str = "numeric",
    metadata: dict[str, Any] | None = None,
) -> VerifiableTask:
    return VerifiableTask(
        task_id=task_id,
        prompt="Compute 6 multiplied by 7. Reply with FINAL_ANSWER: <number>",
        domain="math",
        depth=2,
        knowledge=KNOWLEDGE_FREE,
        grader=grader,
        expected=expected,
        metadata={"tolerance": 1e-9} if metadata is None else metadata,
    )


def _outputs(pass_0: str, pass_1: str) -> tuple[dict[str, str], dict[str, str]]:
    responses = {"pass_0": pass_0, "pass_1": pass_1}
    seals = {name: seal_training_output(text) for name, text in responses.items()}
    return responses, seals


def _authority(
    task: Any,
    *,
    policy: Any,
    role_keys: dict[str, Ed25519PrivateKey],
    pass_0: str,
    pass_1: str,
    nonce: bytes = NONCE,
) -> tuple[PublicVerifiedTrainingTask, dict[str, Any], dict[str, str]]:
    public, sealed = build_verified_training_task(task, answer_nonce=nonce)
    responses, seals = _outputs(pass_0, pass_1)
    payload = prepare_answer_authority_payload(
        public,
        sealed,
        sealed_outputs=seals,
    )
    attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=payload,
        signed_at_unix=SIGNED_AT,
        private_key=role_keys[TASK_ISSUER],
    )
    authority = assemble_answer_authority(
        public,
        payload=payload,
        task_issuer_attestation=attestation,
        policy=policy,
    )
    return public, authority, responses


@pytest.mark.parametrize(
    "task",
    [
        _verifiable(),
        RECURRENCE_GENERATORS["causal_intervention"](4, 77),
        ANSWER_CHANNEL_GENERATORS["typed_boolean"](3, 19),
    ],
)
def test_all_three_training_task_types_build_canonical_public_commitments(task):
    public, sealed = build_verified_training_task(task, answer_nonce=NONCE)
    document = validate_public_training_task(public)

    assert canonical_json_bytes(document) == public.canonical_bytes()
    assert document["task_id"] == task.task_id
    assert document["prompt"] == task.prompt
    assert document["domain"] == task.domain
    assert document["depth"] == task.depth
    assert document["answer_commitment_sha256"] == sealed.answer_commitment_sha256
    assert document["scorer_registry_sha256"] == scorer_registry_identity()[
        "registry_sha256"
    ]


def test_public_commitment_is_immutable_and_contains_no_answer_material():
    task = _verifiable(expected=["needle-secret"], grader="exact_set", metadata={})
    public, sealed = build_verified_training_task(task, answer_nonce=NONCE)
    serialized = public.canonical_bytes().decode("utf-8")

    assert "needle-secret" not in serialized
    assert base64.b64encode(NONCE).decode("ascii") not in serialized
    assert "expected" not in serialized
    assert "answer_nonce" not in serialized
    assert "needle-secret" not in repr(sealed)
    with pytest.raises(FrozenInstanceError):
        public.task_commitment_sha256 = "0" * 64  # type: ignore[misc]

    copy_out = public.to_dict()
    copy_out["task_id"] = "attacker"
    assert public.to_dict()["task_id"] == task.task_id


@pytest.mark.parametrize(
    ("task", "pass_0", "pass_1", "expected"),
    [
        (_verifiable(), "FINAL_ANSWER: 41", "FINAL_ANSWER: 42", (False, True)),
        (
            RECURRENCE_GENERATORS["khop"](3, 18),
            'FINAL_ANSWER: {"node":999}',
            RECURRENCE_GENERATORS["khop"](3, 18).answer,
            (False, True),
        ),
        (
            ANSWER_CHANNEL_GENERATORS["json_copy"](2, 23),
            "FINAL_ANSWER: not-json",
            ANSWER_CHANNEL_GENERATORS["json_copy"](2, 23).answer,
            (False, True),
        ),
    ],
)
def test_scorer_replays_both_sealed_outputs_from_revealed_answer(
    trust,
    task,
    pass_0,
    pass_1,
    expected,
):
    policy, role_keys = trust
    public, authority, responses = _authority(
        task,
        policy=policy,
        role_keys=role_keys,
        pass_0=pass_0,
        pass_1=pass_1,
    )

    first = score_verified_training_outputs(
        public,
        authority,
        outputs=responses,
        policy=policy,
    )
    second = score_verified_training_outputs(
        public,
        authority,
        outputs=responses,
        policy=policy,
    )

    assert tuple(first["results"][name]["verdict"]["correct"] for name in PASS_NAMES) == expected
    assert first == second
    assert first["scorer_registry_sha256"] == scorer_registry_identity()[
        "registry_sha256"
    ]


def test_answer_authority_requires_two_exact_output_seals():
    public, sealed = build_verified_training_task(_verifiable(), answer_nonce=NONCE)

    with pytest.raises(VerifiedTrainingTaskError, match="sealed_outputs_schema_invalid"):
        prepare_answer_authority_payload(
            public,
            sealed,
            sealed_outputs={"pass_0": "1" * 64},
        )
    with pytest.raises(VerifiedTrainingTaskError, match="sealed_output_pass_1_invalid"):
        prepare_answer_authority_payload(
            public,
            sealed,
            sealed_outputs={"pass_0": "1" * 64, "pass_1": "not-a-seal"},
        )


def test_task_swap_and_answer_swap_are_rejected():
    first_public, first_answer = build_verified_training_task(
        _verifiable(task_id="first", expected=42),
        answer_nonce=NONCE,
    )
    second_public, second_answer = build_verified_training_task(
        _verifiable(task_id="second", expected=43),
        answer_nonce=b"second-proof-grade-answer-nonce" * 2,
    )
    _, seals = _outputs("FINAL_ANSWER: 42", "FINAL_ANSWER: 43")

    with pytest.raises(VerifiedTrainingTaskError, match="sealed_answer_public_task_mismatch"):
        prepare_answer_authority_payload(
            first_public,
            second_answer,
            sealed_outputs=seals,
        )
    with pytest.raises(VerifiedTrainingTaskError, match="sealed_answer_public_task_mismatch"):
        prepare_answer_authority_payload(
            second_public,
            first_answer,
            sealed_outputs=seals,
        )


def test_signed_answer_authority_cannot_be_swapped_between_tasks(trust):
    policy, role_keys = trust
    first_public, first_authority, responses = _authority(
        _verifiable(task_id="first", expected=42),
        policy=policy,
        role_keys=role_keys,
        pass_0="FINAL_ANSWER: 41",
        pass_1="FINAL_ANSWER: 42",
    )
    second_public, _sealed = build_verified_training_task(
        _verifiable(task_id="second", expected=42),
        answer_nonce=NONCE,
    )

    assert validate_answer_authority(first_public, first_authority, policy=policy)
    with pytest.raises(VerifiedTrainingTaskError, match="public_task_sha256_mismatch"):
        score_verified_training_outputs(
            second_public,
            first_authority,
            outputs=responses,
            policy=policy,
        )


def test_output_substitution_is_rejected_after_answer_reveal(trust):
    policy, role_keys = trust
    public, authority, responses = _authority(
        _verifiable(),
        policy=policy,
        role_keys=role_keys,
        pass_0="FINAL_ANSWER: 41",
        pass_1="FINAL_ANSWER: 42",
    )
    attacked = dict(responses)
    attacked["pass_1"] = "FINAL_ANSWER: 41"

    with pytest.raises(VerifiedTrainingTaskError, match="pass_1_seal_mismatch"):
        score_verified_training_outputs(
            public,
            authority,
            outputs=attacked,
            policy=policy,
        )


def test_scorer_substitution_and_registry_substitution_fail_closed():
    public, sealed = build_verified_training_task(_verifiable(), answer_nonce=NONCE)
    attacked = public.to_dict()
    attacked["scorer_id"] = "verifiable.boolean.v1"
    core_keys = (
        "schema",
        "task_type",
        "task_id",
        "prompt",
        "domain",
        "depth",
        "public_parameters",
        "scorer_id",
        "scorer_registry_sha256",
    )
    core = {key: attacked[key] for key in core_keys}
    attacked["task_core_sha256"] = hashlib.sha256(
        canonical_json_bytes(core)
    ).hexdigest()
    unsigned = dict(attacked)
    unsigned.pop("task_commitment_sha256")
    attacked["task_commitment_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    assert validate_public_training_task(attacked)
    _, seals = _outputs("FINAL_ANSWER: 41", "FINAL_ANSWER: 42")
    with pytest.raises(VerifiedTrainingTaskError, match="sealed_answer_public_task_mismatch"):
        prepare_answer_authority_payload(attacked, sealed, sealed_outputs=seals)

    registry_attack = copy.deepcopy(scorer_registry_identity())
    registry_attack["scorers"][0]["implementation_sha256"] = "0" * 64
    with pytest.raises(VerifiedTrainingTaskError, match="scorer_registry_identity_mismatch"):
        validate_scorer_registry_identity(registry_attack)


def test_ignored_scorer_parameters_and_derived_task_id_drift_are_rejected():
    with pytest.raises(VerifiedTrainingTaskError, match="unused_scorer_parameters"):
        build_verified_training_task(
            _verifiable(expected=True, grader="boolean", metadata={"tolerance": 1}),
            answer_nonce=NONCE,
        )

    task = RECURRENCE_GENERATORS["khop"](3, 18)
    public, _sealed = build_verified_training_task(task, answer_nonce=NONCE)
    attacked = public.to_dict()
    attacked["task_id"] = "recurrence-khop-d3-s999"
    with pytest.raises(VerifiedTrainingTaskError, match="task_id_mismatch"):
        validate_public_training_task(attacked)


def test_json_scorer_rejects_duplicate_and_non_finite_payloads(trust):
    policy, role_keys = trust
    task = VerifiableTask(
        task_id="json-proof-1",
        prompt="Return the exact object.",
        domain="logic",
        depth=1,
        knowledge=KNOWLEDGE_FREE,
        grader="json",
        expected={"x": 2},
        metadata={},
    )
    public, authority, responses = _authority(
        task,
        policy=policy,
        role_keys=role_keys,
        pass_0='FINAL_ANSWER: {"x":1,"x":2}',
        pass_1='FINAL_ANSWER: {"x":NaN}',
    )
    receipt = score_verified_training_outputs(
        public,
        authority,
        outputs=responses,
        policy=policy,
    )

    assert receipt["results"]["pass_0"]["verdict"]["reason"] == "unparseable"
    assert receipt["results"]["pass_1"]["verdict"]["reason"] == "unparseable"


def test_attested_answer_or_scorer_swap_is_rejected_even_when_resigned(trust):
    policy, role_keys = trust
    public, sealed = build_verified_training_task(_verifiable(), answer_nonce=NONCE)
    _, seals = _outputs("FINAL_ANSWER: 41", "FINAL_ANSWER: 42")
    payload = prepare_answer_authority_payload(public, sealed, sealed_outputs=seals)

    for field, replacement, error in (
        ("expected", 43, "answer_commitment_mismatch"),
        ("scorer_id", "verifiable.boolean.v1", "scorer_id_mismatch"),
    ):
        attacked = copy.deepcopy(payload)
        attacked[field] = replacement
        attestation = build_role_attestation(
            policy,
            role=TASK_ISSUER,
            payload=attacked,
            signed_at_unix=SIGNED_AT,
            private_key=role_keys[TASK_ISSUER],
        )
        with pytest.raises(VerifiedTrainingTaskError, match=error):
            assemble_answer_authority(
                public,
                payload=attacked,
                task_issuer_attestation=attestation,
                policy=policy,
            )


def test_wrong_task_issuer_signature_is_rejected(trust):
    policy, role_keys = trust
    public, sealed = build_verified_training_task(_verifiable(), answer_nonce=NONCE)
    _, seals = _outputs("FINAL_ANSWER: 41", "FINAL_ANSWER: 42")
    payload = prepare_answer_authority_payload(public, sealed, sealed_outputs=seals)
    # Ed25519 private keys are intentionally non-serializable and Python 3.14
    # refuses to deepcopy them.  The adversary only replaces one mapping
    # entry, so a shallow container copy preserves the test's semantics.
    wrong = dict(role_keys)
    wrong[TASK_ISSUER] = Ed25519PrivateKey.generate()

    with pytest.raises(ValueError, match="private_key_mismatch"):
        build_role_attestation(
            policy,
            role=TASK_ISSUER,
            payload=payload,
            signed_at_unix=SIGNED_AT,
            private_key=wrong[TASK_ISSUER],
        )


def test_corrupted_task_issuer_attestation_is_rejected_by_local_verifier(trust):
    policy, role_keys = trust
    public, authority, _responses = _authority(
        _verifiable(),
        policy=policy,
        role_keys=role_keys,
        pass_0="FINAL_ANSWER: 41",
        pass_1="FINAL_ANSWER: 42",
    )
    attacked = copy.deepcopy(authority)
    attacked["task_issuer_attestation"]["signature_b64"] = base64.b64encode(
        b"x" * 64
    ).decode("ascii")
    unsigned = dict(attacked)
    unsigned.pop("authority_sha256")
    attacked["authority_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    with pytest.raises(VerifiedTrainingTaskError, match="attestation_signature_invalid"):
        validate_answer_authority(public, attacked, policy=policy)


def test_public_metadata_leakage_and_non_json_values_are_rejected():
    leaking = _verifiable(metadata={"answer": "needle-secret"})
    with pytest.raises(VerifiedTrainingTaskError, match="public_task_secret_field"):
        build_verified_training_task(leaking, answer_nonce=NONCE)

    non_json = _verifiable(metadata={"allowed_values": (1, 2)})
    with pytest.raises(VerifiedTrainingTaskError, match="non_json_type"):
        build_verified_training_task(non_json, answer_nonce=NONCE)


def test_noncanonical_and_duplicate_json_documents_are_rejected():
    public, _sealed = build_verified_training_task(_verifiable(), answer_nonce=NONCE)
    pretty = json.dumps(public.to_dict(), indent=2).encode("utf-8")
    duplicate = public.canonical_bytes().replace(
        b'{"answer_commitment_sha256":',
        b'{"schema":"duplicate","answer_commitment_sha256":',
        1,
    )

    with pytest.raises(VerifiedTrainingTaskError, match="not_canonical"):
        validate_public_training_task(pretty)
    with pytest.raises(VerifiedTrainingTaskError, match="duplicate_key|schema_invalid"):
        validate_public_training_task(duplicate)


@pytest.mark.parametrize("nonce", [b"", b"short", b"x" * 257, "x" * 32])
def test_answer_commitment_requires_bounded_bytes_nonce(nonce):
    with pytest.raises(VerifiedTrainingTaskError, match="answer_nonce_invalid"):
        build_verified_training_task(_verifiable(), answer_nonce=nonce)  # type: ignore[arg-type]


def test_task_type_domain_depth_and_exact_class_are_validated():
    public, _sealed = build_verified_training_task(_verifiable(), answer_nonce=NONCE)
    for field, replacement, error in (
        ("task_type", "mystery", "public_task_type_invalid"),
        ("domain", "not valid", "public_task_domain_invalid"),
        ("depth", True, "public_task_depth_invalid"),
    ):
        attacked = public.to_dict()
        attacked[field] = replacement
        with pytest.raises(VerifiedTrainingTaskError, match=error):
            validate_public_training_task(attacked)

    class VerifiableSubclass(VerifiableTask):
        pass

    spoof = VerifiableSubclass(**_verifiable().__dict__)
    with pytest.raises(VerifiedTrainingTaskError, match="task_type_unsupported"):
        build_verified_training_task(spoof, answer_nonce=NONCE)


def test_production_module_has_no_frontier_registry_or_private_signing_api():
    module_path = Path(inspect.getsourcefile(build_verified_training_task) or "")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("frontier_tasks" in name for name in imported)
    assert not any("FrontierTaskRegistry" in name for name in imported)
    assert "private_key" not in inspect.signature(build_verified_training_task).parameters
    assert "private_key" not in inspect.signature(assemble_answer_authority).parameters


def test_isolated_production_import_does_not_transitively_load_frontier_registry():
    script = """
import sys
import core.learning.verified_training_task
assert 'core.brain.llm.latent_cortex.frontier_tasks' not in sys.modules
assert not any(name.startswith('core.brain.llm.latent_cortex') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_scorer_registry_is_immutable_source_bound_and_complete():
    identity = scorer_registry_identity()
    assert identity["module"] == "core.learning.verified_training_task"
    assert len(identity["module_source_sha256"]) == 64
    assert {row["scorer_id"] for row in identity["scorers"]} == set(
        SCORER_REGISTRY
    )
    assert validate_scorer_registry_identity(identity) == identity
    with pytest.raises(TypeError):
        SCORER_REGISTRY["attacker"] = lambda *_args: {}  # type: ignore[index]
