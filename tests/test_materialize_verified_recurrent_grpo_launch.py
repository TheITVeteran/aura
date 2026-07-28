"""End-to-end contract tests for production launch materialization."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
)
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    REQUIRED_SOURCE_ROLES,
)
from core.learning.recurrence_curriculum import khop_reachability
from core.learning.verified_token_trace import (
    build_tokenizer_bundle_identity,
    tokenizer_file_bindings_from_bytes,
)
from core.learning.verified_training_task import build_verified_training_task
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_policy_probe import (
    build_initial_recurrent_policy_probe,
)
from core.learning.verified_transition_production_factory import (
    COMMAND_SIGNER_RESPONSE_SCHEMA,
)
from core.learning.verified_transition_provider import TASK_COMMITMENT_SCHEMA
from tools import materialize_verified_recurrent_grpo_launch as materializer

NOW = 1_900_000_000


def test_materializer_is_directly_executable_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(materializer.__file__).resolve()), "--help"],
        cwd=Path(materializer.__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--initial-policy-probe" in completed.stdout


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _write(path: Path, payload: bytes, *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


def _public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _policy_material(
    tmp_path: Path,
    *,
    campaign_id: str,
    protocol_sha256: str,
    shared_custody: bool = False,
) -> tuple[Path, Path, Path, Path]:
    root_key = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    signer_material = {}
    for signer_role in (TASK_ISSUER, EVIDENCE_VERIFIER):
        signer_raw = role_keys[signer_role].private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        signer = tmp_path / f"{signer_role}-external-signer.py"
        signer.write_text(
            f"#!{sys.executable}\n"
            "import base64,json,sys\n"
            "from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey\n"
            f"key=Ed25519PrivateKey.from_private_bytes(bytes.fromhex('{signer_raw.hex()}'))\n"
            "envelope=json.loads(sys.stdin.buffer.read())\n"
            "request=envelope['signature_request']\n"
            "signature=key.sign(base64.b64decode(request['signed_payload_b64']))\n"
            f"response={{'schema':'{COMMAND_SIGNER_RESPONSE_SCHEMA}',"
            "'request_sha256':request['request_sha256'],"
            "'signature_b64':base64.b64encode(signature).decode('ascii')}\n"
            "sys.stdout.write(json.dumps(response,sort_keys=True,separators=(',',':'))+'\\n')\n",
            encoding="ascii",
        )
        signer.chmod(0o700)
        release = _write(
            tmp_path / f"{signer_role}-release.json",
            canonical_json_bytes({"release": f"{signer_role}-fixture-v1"}),
        )
        custody = _write(
            tmp_path / f"{signer_role}-custody.json",
            canonical_json_bytes(
                {
                    "custody": (
                        "shared-fixture-external"
                        if shared_custody
                        else f"{signer_role}-fixture-external"
                    )
                }
            ),
        )
        signer_material[signer_role] = {
            "signer": signer,
            "release": release,
            "custody": custody,
            "implementation_sha256": hashlib.sha256(
                signer.read_bytes()
            ).hexdigest(),
            "release_sha256": hashlib.sha256(
                release.read_bytes()
            ).hexdigest(),
            "custody_sha256": hashlib.sha256(
                custody.read_bytes()
            ).hexdigest(),
        }
    roles = {}
    for role, key in role_keys.items():
        public = _public_raw(key)
        signer_pin = signer_material.get(role)
        roles[role] = {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(public).decode("ascii"),
            "key_id": hashlib.sha256(public).hexdigest(),
            "implementation_sha256": (
                signer_pin["implementation_sha256"]
                if signer_pin
                else _sha(f"{role}-implementation")
            ),
            "release_sha256": (
                signer_pin["release_sha256"]
                if signer_pin
                else _sha(f"{role}-release")
            ),
            "custody_class": "external_service",
            "custody_evidence_sha256": (
                signer_pin["custody_sha256"]
                if signer_pin
                else _sha(f"{role}-custody")
            ),
        }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "materializer-fixture-policy",
        "policy_revision": 1,
        "campaign_name": campaign_id,
        "protocol_sha256": protocol_sha256,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": NOW - 20,
        "not_before_unix": NOW - 10,
        "expires_at_unix": NOW + 10_000,
        "roles": roles,
    }
    root_raw = _public_raw(root_key)
    root_pem = root_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signed = canonical_json_bytes(body)
    policy = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root_key.sign(signed)).decode(
                "ascii"
            ),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    policy_path = _write(
        tmp_path / "policy.json",
        canonical_json_bytes(policy) + b"\n",
    )
    root_path = _write(tmp_path / "root.pem", root_pem)
    config_paths = {}
    for signer_role, material in signer_material.items():
        signer_config = {
            "schema": materializer.SIGNER_CONFIG_SCHEMA,
            "identity": f"{signer_role}-fixture-external-signer",
            "executable": str(material["signer"]),
            "executable_sha256": material["implementation_sha256"],
            "release_manifest": str(material["release"]),
            "custody_evidence": str(material["custody"]),
            "arguments": [],
            "timeout_millis": 5_000,
            "inherited_environment_names": [],
        }
        config_paths[signer_role] = _write(
            tmp_path / f"{signer_role}-signer-config.json",
            canonical_json_bytes(signer_config) + b"\n",
        )
    return (
        policy_path,
        root_path,
        config_paths[TASK_ISSUER],
        config_paths[EVIDENCE_VERIFIER],
    )


@pytest.mark.parametrize(
    "crash_role",
    [
        "materialization_intent",
        "task_answer_nonces",
        "task_commitments",
        "provider-config_json",
        "launch_bundle",
        "launch_bundle_digest",
        "materialization_receipt",
    ],
)
def test_materializer_publishes_and_reopens_exact_externally_rooted_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_role: str,
) -> None:
    campaign_id = "resident-32b-recurrent-grpo-cp-test"
    source = _write(tmp_path / "source.py", b"x = 1\n")
    source_binding = {
        "path": "source.py",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size_bytes": source.stat().st_size,
    }
    model = tmp_path / "model"
    model.mkdir(mode=0o700)
    dataset_sha = _sha("dataset")
    spec_sha = _sha("execution")
    contract_body = {
        "campaign_id": campaign_id,
        "model": {
            "path": "model",
            "base_checkpoint": {"fingerprint": _sha("model")},
            "behavior_bundle": {"bundle_sha256": _sha("behavior")},
        },
        "execution_spec": {"semantic_sha256": spec_sha},
        "sources": {
            role: source_binding for role in REQUIRED_SOURCE_ROLES
        },
        "paths": {
            "verified_launch_bundle": "artifacts/launch/launch-bundle.json",
            "training_output": "artifacts/training",
        },
        "training": {
            "dataset": {"sha256": dataset_sha},
            "parameters": {
                "task_source": "recurrence_curriculum",
                "domains": ["khop_reachability"],
                "depths": [1],
                "train_per_cell": 1,
                "holdout_per_cell": 1,
                "seed": 7,
                "max_steps": 1,
                "group_size": 2,
                "max_tokens": 8,
                "cot": True,
            },
            "argv": [
                "tools/train_grpo.py",
                "--model",
                "model",
                "--out-dir",
                "artifacts/training",
            ],
        },
    }
    contract = {
        **contract_body,
        "contract_sha256": hashlib.sha256(
            canonical_json_bytes(contract_body)
        ).hexdigest(),
    }
    contract_path = _write(
        tmp_path / "preregistration.json",
        canonical_json_bytes(contract),
    )
    bundle_identity = build_tokenizer_bundle_identity(
        tokenizer_class="fixture.Tokenizer",
        tokenizer_files=tokenizer_file_bindings_from_bytes(
            {"tokenizer.json": b"fixture"}
        ),
        chat_template=None,
        special_token_map={},
        encode_options={},
        decode_options={},
        implementation_source_sha256=_sha("tokenizer-implementation"),
    )
    probe = build_initial_recurrent_policy_probe(
        campaign_id=campaign_id,
        initial_policy_sha256=_sha("initial-policy"),
        dataset_sha256=dataset_sha,
        execution_spec_sha256=spec_sha,
        base_checkpoint=contract["model"]["base_checkpoint"],
        model_behavior_bundle=contract["model"]["behavior_bundle"],
        tokenizer_bundle=bundle_identity,
        adapter_initialization={
            "seed": 7,
            "rank": 8,
            "layers": 8,
            "targets": ["q_proj"],
        },
        source_bindings={
            role: source_binding for role in REQUIRED_SOURCE_ROLES
        },
        created_at_unix_ns=NOW * 1_000_000_000,
    )
    probe_path = _write(
        tmp_path / "probe.json",
        canonical_json_bytes(probe),
    )
    (
        policy_path,
        root_path,
        task_signer_path,
        verifier_signer_path,
    ) = _policy_material(
        tmp_path,
        campaign_id=campaign_id,
        protocol_sha256=contract["contract_sha256"],
    )
    task = khop_reachability(1, 9)
    nonce = b"n" * 32
    public, _sealed = build_verified_training_task(task, answer_nonce=nonce)
    commitment = {
        "schema": TASK_COMMITMENT_SCHEMA,
        "sequence": 0,
        "task_id": task.task_id,
        "trainer_sample_seed": 11,
        "immutable_task_sha256": hashlib.sha256(
            canonical_json_bytes(public.to_dict())
        ).hexdigest(),
        "prompt_tokens_sha256": _sha("prompt"),
        "recurrent_execution_spec_sha256": spec_sha,
        "sample_seeds": [21, 22],
    }

    monkeypatch.setattr(materializer.prereg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        materializer.prereg,
        "validate_contract",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        materializer,
        "build_resident_tokenizer_trace_adapter",
        lambda *_args, **_kwargs: SimpleNamespace(
            bundle_identity=bundle_identity
        ),
    )
    monkeypatch.setattr(
        materializer,
        "_task_material",
        lambda **_kwargs: (
            [commitment],
            {task.task_id: public.to_dict()},
            {task.task_id: nonce},
        ),
    )
    original_publish = materializer._publish
    injected = {"raised": False}

    def fail_once(path, payload, *, role):
        result = original_publish(path, payload, role=role)
        if role == crash_role and not injected["raised"]:
            injected["raised"] = True
            raise materializer.LaunchMaterializationError(
                "injected_materialization_crash"
            )
        return result

    monkeypatch.setattr(materializer, "_publish", fail_once)
    with pytest.raises(
        materializer.LaunchMaterializationError,
        match="injected_materialization_crash",
    ):
        materializer.materialize_launch(
            preregistration_path=contract_path,
            initial_policy_probe_path=probe_path,
            trust_policy_path=policy_path,
            trust_root_path=root_path,
            task_issuer_signer_config_path=task_signer_path,
            evidence_verifier_signer_config_path=verifier_signer_path,
            now_unix_ns=NOW * 1_000_000_000,
            tokenizer=object(),
        )
    monkeypatch.setattr(materializer, "_publish", original_publish)

    receipt = materializer.materialize_launch(
        preregistration_path=contract_path,
        initial_policy_probe_path=probe_path,
        trust_policy_path=policy_path,
        trust_root_path=root_path,
        task_issuer_signer_config_path=task_signer_path,
        evidence_verifier_signer_config_path=verifier_signer_path,
        now_unix_ns=NOW * 1_000_000_000,
        tokenizer=object(),
    )

    assert receipt["reopened"] is True
    assert injected["raised"] is True
    assert receipt["task_count"] == 1
    assert receipt["claim_boundary"].startswith("launch_custody_only")
    bundle_path = Path(receipt["bundle_path"])
    assert bundle_path.is_file()
    assert (
        hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        == receipt["bundle_sha256"]
    )
    assert (
        bundle_path.with_name("launch-bundle.sha256")
        .read_text(encoding="ascii")
        .strip()
        == receipt["bundle_sha256"]
    )
    for config_path in (task_signer_path, verifier_signer_path):
        signer_config = json.loads(config_path.read_bytes())
        Path(signer_config["executable"]).unlink()
    recovered = materializer.materialize_launch(
        preregistration_path=contract_path,
        initial_policy_probe_path=probe_path,
        trust_policy_path=policy_path,
        trust_root_path=root_path,
        task_issuer_signer_config_path=task_signer_path,
        evidence_verifier_signer_config_path=verifier_signer_path,
        now_unix_ns=(NOW + 20_000) * 1_000_000_000,
        tokenizer=None,
    )
    assert recovered == receipt

    verifier_config = json.loads(verifier_signer_path.read_bytes())
    verifier_config["timeout_millis"] += 1
    verifier_signer_path.write_bytes(
        canonical_json_bytes(verifier_config) + b"\n"
    )
    with pytest.raises(
        materializer.LaunchMaterializationError,
        match="materialization_intent_mismatch",
    ):
        materializer.materialize_launch(
            preregistration_path=contract_path,
            initial_policy_probe_path=probe_path,
            trust_policy_path=policy_path,
            trust_root_path=root_path,
            task_issuer_signer_config_path=task_signer_path,
            evidence_verifier_signer_config_path=verifier_signer_path,
            now_unix_ns=(NOW + 200) * 1_000_000_000,
            tokenizer=object(),
        )


def test_materializer_rejects_noncanonical_external_configuration(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "config.json", b'{ "schema": "drift" }\n')

    with pytest.raises(
        materializer.LaunchMaterializationError,
        match="signer_config_schema_invalid",
    ):
        materializer._load_signer(path)


def test_materializer_rejects_distinct_signers_with_shared_custody(
    tmp_path: Path,
) -> None:
    (
        _policy_path,
        _root_path,
        task_signer_path,
        verifier_signer_path,
    ) = _policy_material(
        tmp_path,
        campaign_id="shared-custody-campaign",
        protocol_sha256=_sha("shared-custody-protocol"),
        shared_custody=True,
    )
    _task_document, task_broker = materializer._load_signer(
        task_signer_path
    )
    _verifier_document, verifier_broker = materializer._load_signer(
        verifier_signer_path
    )

    assert task_broker.identity != verifier_broker.identity
    with pytest.raises(
        materializer.LaunchMaterializationError,
        match="signer_role_separation_required",
    ):
        materializer._validate_signer_role_separation(
            task_broker,
            verifier_broker,
        )


def test_bound_tokenizer_loads_from_validated_resident_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir(mode=0o700)
    _write(
        model / "config.json",
        canonical_json_bytes({"eos_token_id": [7, 9]}) + b"\n",
    )
    observed: dict[str, object] = {}
    sentinel = object()

    def fake_load_tokenizer(path: Path, *, eos_token_ids: object) -> object:
        observed["path"] = path
        observed["eos_token_ids"] = eos_token_ids
        return sentinel

    monkeypatch.setattr(
        "mlx_lm.utils.load_tokenizer",
        fake_load_tokenizer,
    )

    assert materializer._load_bound_tokenizer(model) is sentinel
    assert observed == {
        "path": model,
        "eos_token_ids": [7, 9],
    }


@pytest.mark.parametrize("invalid_eos", [None, [], [7, "9"], -1])
def test_bound_tokenizer_rejects_invalid_resident_eos_contract(
    tmp_path: Path,
    invalid_eos: object,
) -> None:
    model = tmp_path / hashlib.sha256(
        repr(invalid_eos).encode("utf-8")
    ).hexdigest()
    model.mkdir(mode=0o700)
    _write(
        model / "config.json",
        canonical_json_bytes({"eos_token_id": invalid_eos}) + b"\n",
    )

    with pytest.raises(
        materializer.LaunchMaterializationError,
        match="model_tokenizer_eos_invalid",
    ):
        materializer._load_bound_tokenizer(model)
