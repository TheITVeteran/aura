"""SPARK-004 frozen baseline bundle contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.execution_spec import RLC_EXECUTION_SPEC_SCHEMA
from core.brain.llm.latent_cortex.frozen_baseline import (
    FROZEN_BASELINE_FILE,
    FROZEN_BASELINE_SCHEMA,
    FROZEN_BASELINE_SIGNATURE_FILE,
    FrozenBaselineError,
    build_frozen_baseline_certificate,
    planned_bundle_files,
    publish_frozen_baseline_bundle,
    sign_frozen_baseline_certificate,
    validate_frozen_baseline_certificate,
    verify_frozen_baseline,
    verify_frozen_baseline_model,
    verify_frozen_baseline_signature,
    verify_frozen_baseline_sources,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha(canonical_json_bytes(value))


def _runtime_identity() -> dict:
    body = {
        "python": "3.12.0",
        "platform_system": "Darwin",
        "platform_release": "25.0.0",
        "platform_machine": "arm64",
        "dependencies": {"mlx": "0.31.2", "mlx-lm": "0.22.0", "numpy": "2.0.0"},
    }
    return {**body, "identity_sha256": _canonical_sha(body)}


def _behavior_files() -> list[dict]:
    files = []
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        payload = f'{{"file":"{name}"}}'.encode("ascii")
        files.append(
            {"path": name, "sha256": _sha(payload), "size_bytes": len(payload)}
        )
    return files


def _execution_spec() -> dict:
    return {
        "schema": RLC_EXECUTION_SPEC_SCHEMA,
        "recurrent_steps": 4,
        "n_slots": 16,
        "slot_seed": 7,
    }


def _material(payloads: dict[str, bytes]) -> dict:
    behavior_files = _behavior_files()
    spec = _execution_spec()
    manifest_payload = payloads["control_manifests/pilot_contract.json"]
    vanilla_payload = payloads["measurements/vanilla_eval.json"]
    rlc_payload = payloads["measurements/rlc_eval.json"]
    return {
        "schema": FROZEN_BASELINE_SCHEMA,
        "baseline_id": "spark004-test-baseline",
        "purpose": "unit-test baseline",
        "frozen_at_unix": 1_784_900_000,
        "git_commit": "a" * 40,
        "worktree_clean": True,
        "environment": {
            "runtime": _runtime_identity(),
            "observed_physical_memory_bytes": 68719476736,
            "observed_cpu_count": 16,
        },
        "model": {
            "path": "/models/resident-32b",
            "checkpoint": {"fingerprint": "b" * 64, "method": "sha256", "files": 3},
            "behavior_bundle": {
                "bundle_sha256": _canonical_sha(behavior_files),
                "file_count": len(behavior_files),
                "files": behavior_files,
            },
        },
        "adapters": {
            "personality": {
                "present": False,
                "bundle_sha256": "",
                "file_count": 0,
                "files": [],
            },
            "attached_at_baseline": [],
        },
        "decoding": {
            "execution_spec": spec,
            "execution_spec_sha256": _canonical_sha(spec),
            "sampling": {
                "decode_max_tokens": 512,
                "decode_temperature": 0.0,
                "decode_top_p": 1.0,
            },
        },
        "task_generators": {
            "registry_version": "2026.07.18.2",
            "frontier_domains": ["algebra", "logic"],
            "excluded_training_families": ["khop"],
            "recurrence_training_families": ["parity", "chain"],
            "sources": [
                {
                    "repo_path": "core/task_generator.py",
                    "sha256": _sha(b"generator-source"),
                    "size_bytes": len(b"generator-source"),
                }
            ],
        },
        "control_manifests": [
            {
                "name": "pilot_contract",
                "bundle_path": "control_manifests/pilot_contract.json",
                "source_path": "config/latent_cortex/pilot_contract.json",
                "sha256": _sha(manifest_payload),
                "size_bytes": len(manifest_payload),
            }
        ],
        "resource_envelope": {
            "declared": {
                "host_memory_bytes": 68719476736,
                "detached_timeout_s": 93600,
                "exclusive_resident_model_owner": True,
            }
        },
        "randomization": {
            "training_seed": 2026072102,
            "slot_seed": 7,
            "eval_seed": 424242,
            "seed_policy": "fixed_preregistered_seeds",
            "sources": [
                {
                    "repo_path": "core/rng_streams.py",
                    "sha256": _sha(b"rng-source"),
                    "size_bytes": len(b"rng-source"),
                }
            ],
        },
        "measurements": [
            {
                "name": "vanilla_eval",
                "role": "vanilla",
                "bundle_path": "measurements/vanilla_eval.json",
                "source_path": "artifacts/vanilla_eval.json",
                "sha256": _sha(vanilla_payload),
                "size_bytes": len(vanilla_payload),
                "summary": {"accuracy": 0.29},
            },
            {
                "name": "rlc_eval",
                "role": "rlc",
                "bundle_path": "measurements/rlc_eval.json",
                "source_path": "artifacts/rlc_eval.json",
                "sha256": _sha(rlc_payload),
                "size_bytes": len(rlc_payload),
                "summary": {"accuracy": 0.31},
            },
        ],
    }


def _payloads() -> dict[str, bytes]:
    return {
        "control_manifests/pilot_contract.json": b'{"contract":true}\n',
        "measurements/vanilla_eval.json": b'{"accuracy":0.29}\n',
        "measurements/rlc_eval.json": b'{"accuracy":0.31}\n',
    }


@pytest.fixture()
def sealed_bundle(tmp_path: Path) -> tuple[Path, dict, dict, Ed25519PrivateKey]:
    payloads = _payloads()
    certificate = build_frozen_baseline_certificate(_material(payloads))
    private_key = Ed25519PrivateKey.generate()
    signature = sign_frozen_baseline_certificate(
        certificate, private_key=private_key
    )
    root = tmp_path / "bundle"
    publish_frozen_baseline_bundle(
        root,
        certificate=certificate,
        file_payloads=payloads,
        signature=signature,
    )
    return root, certificate, signature, private_key


def _public_pem(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _unlock(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    os.chmod(root, 0o755)


def _relock(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


class TestCertificate:
    def test_build_and_validate_roundtrip(self) -> None:
        certificate = build_frozen_baseline_certificate(_material(_payloads()))
        assert validate_frozen_baseline_certificate(certificate) == certificate

    def test_certificate_digest_tamper_rejected(self) -> None:
        certificate = build_frozen_baseline_certificate(_material(_payloads()))
        tampered = {**certificate, "baseline_id": "spark004-other"}
        with pytest.raises(FrozenBaselineError) as error:
            validate_frozen_baseline_certificate(tampered)
        assert error.value.code == "frozen_baseline_certificate_digest_mismatch"

    def test_dirty_worktree_rejected(self) -> None:
        material = _material(_payloads())
        material["worktree_clean"] = False
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_worktree_dirty"

    def test_measurement_coverage_requires_vanilla_and_rlc(self) -> None:
        material = _material(_payloads())
        material["measurements"] = [
            record
            for record in material["measurements"]
            if record["role"] == "vanilla"
        ]
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_measurement_coverage_invalid"

    def test_paired_measurement_alone_satisfies_coverage(self) -> None:
        payloads = _payloads()
        material = _material(payloads)
        paired = dict(material["measurements"][0])
        paired["role"] = "paired"
        material["measurements"] = [paired]
        certificate = build_frozen_baseline_certificate(material)
        assert certificate["measurements"][0]["role"] == "paired"

    def test_execution_spec_digest_mismatch_rejected(self) -> None:
        material = _material(_payloads())
        material["decoding"]["execution_spec_sha256"] = "c" * 64
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_decoding_spec_digest_mismatch"

    def test_wrong_execution_spec_schema_rejected(self) -> None:
        material = _material(_payloads())
        material["decoding"]["execution_spec"]["schema"] = "aura.other.v1"
        material["decoding"]["execution_spec_sha256"] = _canonical_sha(
            material["decoding"]["execution_spec"]
        )
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_decoding_spec_invalid"

    def test_environment_identity_mismatch_rejected(self) -> None:
        material = _material(_payloads())
        material["environment"]["runtime"]["identity_sha256"] = "d" * 64
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_environment_identity_mismatch"

    def test_behavior_bundle_requires_tokenizer_files(self) -> None:
        material = _material(_payloads())
        files = [
            record
            for record in material["model"]["behavior_bundle"]["files"]
            if record["path"] != "tokenizer.json"
        ]
        material["model"]["behavior_bundle"] = {
            "bundle_sha256": _canonical_sha(files),
            "file_count": len(files),
            "files": files,
        }
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_model_behavior_invalid"

    def test_personality_bundle_digest_recomputed(self) -> None:
        material = _material(_payloads())
        material["adapters"]["personality"] = {
            "present": True,
            "bundle_sha256": "e" * 64,
            "file_count": 1,
            "files": [
                {"path": "adapter.safetensors", "sha256": "f" * 64, "size_bytes": 8}
            ],
        }
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_adapters_personality_invalid"

    def test_measurement_role_vocabulary_enforced(self) -> None:
        material = _material(_payloads())
        material["measurements"][0]["role"] = "treatment"
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_measurement_role_invalid"

    def test_bundle_path_prefix_enforced(self) -> None:
        material = _material(_payloads())
        material["measurements"][0]["bundle_path"] = "elsewhere/vanilla.json"
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_measurement_path_invalid"

    def test_resource_envelope_requires_declared_values(self) -> None:
        material = _material(_payloads())
        material["resource_envelope"] = {"declared": {}}
        with pytest.raises(FrozenBaselineError) as error:
            build_frozen_baseline_certificate(material)
        assert error.value.code == "frozen_baseline_resource_envelope_invalid"


class TestSignature:
    def test_signature_roundtrip(self) -> None:
        certificate = build_frozen_baseline_certificate(_material(_payloads()))
        private_key = Ed25519PrivateKey.generate()
        signature = sign_frozen_baseline_certificate(
            certificate, private_key=private_key
        )
        verified = verify_frozen_baseline_signature(
            certificate, signature, trusted_public_key_pem=_public_pem(private_key)
        )
        assert verified["certificate_sha256"] == certificate["certificate_sha256"]

    def test_wrong_trust_anchor_rejected(self) -> None:
        certificate = build_frozen_baseline_certificate(_material(_payloads()))
        signature = sign_frozen_baseline_certificate(
            certificate, private_key=Ed25519PrivateKey.generate()
        )
        with pytest.raises(FrozenBaselineError) as error:
            verify_frozen_baseline_signature(
                certificate,
                signature,
                trusted_public_key_pem=_public_pem(Ed25519PrivateKey.generate()),
            )
        assert error.value.code == "frozen_baseline_signature_key_mismatch"

    def test_signature_over_tampered_payload_rejected(self) -> None:
        certificate = build_frozen_baseline_certificate(_material(_payloads()))
        private_key = Ed25519PrivateKey.generate()
        signature = dict(
            sign_frozen_baseline_certificate(certificate, private_key=private_key)
        )
        signature["signature_b64"] = signature["signature_b64"][:-4] + "AAA="
        with pytest.raises(FrozenBaselineError) as error:
            verify_frozen_baseline_signature(
                certificate,
                signature,
                trusted_public_key_pem=_public_pem(private_key),
            )
        assert error.value.code == "frozen_baseline_signature_invalid"


class TestBundle:
    def test_publish_and_verify_roundtrip(self, sealed_bundle) -> None:
        root, certificate, _signature, private_key = sealed_bundle
        verified = verify_frozen_baseline(
            root, trusted_public_key_pem=_public_pem(private_key)
        )
        assert verified == certificate

    def test_planned_file_set_is_exact(self, sealed_bundle) -> None:
        root, certificate, _signature, _private_key = sealed_bundle
        observed = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        assert observed == planned_bundle_files(
            certificate, include_signature=True
        )

    def test_unplanned_file_rejected(self, sealed_bundle) -> None:
        root, _certificate, _signature, private_key = sealed_bundle
        _unlock(root)
        extra = root / "measurements" / "extra.json"
        extra.write_bytes(b"{}\n")
        os.chmod(extra, 0o444)
        _relock(root)
        with pytest.raises(FrozenBaselineError) as error:
            verify_frozen_baseline(
                root, trusted_public_key_pem=_public_pem(private_key)
            )
        assert error.value.code == "frozen_baseline_artifact_set_mismatch"

    def test_copy_tamper_rejected(self, sealed_bundle) -> None:
        root, _certificate, _signature, private_key = sealed_bundle
        _unlock(root)
        target = root / "measurements" / "vanilla_eval.json"
        target.write_bytes(b'{"accuracy":0.99}\n')
        _relock(root)
        with pytest.raises(FrozenBaselineError) as error:
            verify_frozen_baseline(
                root, trusted_public_key_pem=_public_pem(private_key)
            )
        assert error.value.code == "frozen_baseline_copy_binding_mismatch"

    def test_writable_file_rejected(self, sealed_bundle) -> None:
        root, _certificate, _signature, private_key = sealed_bundle
        _unlock(root)
        _relock(root)
        os.chmod(root, 0o755)
        os.chmod(root / "measurements", 0o755)
        os.chmod(root / "measurements" / "rlc_eval.json", 0o644)
        with pytest.raises(FrozenBaselineError):
            verify_frozen_baseline(
                root, trusted_public_key_pem=_public_pem(private_key)
            )

    def test_missing_signature_with_trust_anchor_rejected(
        self, tmp_path: Path
    ) -> None:
        payloads = _payloads()
        certificate = build_frozen_baseline_certificate(_material(payloads))
        root = tmp_path / "unsigned"
        publish_frozen_baseline_bundle(
            root, certificate=certificate, file_payloads=payloads
        )
        assert verify_frozen_baseline(root) == certificate
        with pytest.raises(FrozenBaselineError) as error:
            verify_frozen_baseline(
                root,
                trusted_public_key_pem=_public_pem(Ed25519PrivateKey.generate()),
            )
        assert error.value.code == "frozen_baseline_signature_missing"

    def test_payload_binding_mismatch_blocks_publish(self, tmp_path: Path) -> None:
        payloads = _payloads()
        certificate = build_frozen_baseline_certificate(_material(payloads))
        payloads["measurements/rlc_eval.json"] = b'{"accuracy":0.99}\n'
        with pytest.raises(FrozenBaselineError) as error:
            publish_frozen_baseline_bundle(
                tmp_path / "bad", certificate=certificate, file_payloads=payloads
            )
        assert error.value.code == "frozen_baseline_payload_binding_mismatch"
        assert not (tmp_path / "bad").exists()

    def test_existing_root_rejected(self, sealed_bundle, tmp_path: Path) -> None:
        root, certificate, _signature, _private_key = sealed_bundle
        with pytest.raises(FrozenBaselineError) as error:
            publish_frozen_baseline_bundle(
                root, certificate=certificate, file_payloads=_payloads()
            )
        assert error.value.code == "frozen_baseline_root_exists"

    def test_certificate_file_tamper_rejected(self, sealed_bundle) -> None:
        root, certificate, _signature, private_key = sealed_bundle
        _unlock(root)
        tampered = {**certificate, "baseline_id": "spark004-tampered"}
        (root / FROZEN_BASELINE_FILE).write_bytes(
            canonical_json_bytes(tampered) + b"\n"
        )
        os.chmod(root / FROZEN_BASELINE_FILE, 0o444)
        _relock(root)
        with pytest.raises(FrozenBaselineError) as error:
            verify_frozen_baseline(
                root, trusted_public_key_pem=_public_pem(private_key)
            )
        assert error.value.code == "frozen_baseline_certificate_digest_mismatch"

    def test_signature_file_survives_roundtrip(self, sealed_bundle) -> None:
        root, certificate, signature, _private_key = sealed_bundle
        stored = json.loads((root / FROZEN_BASELINE_SIGNATURE_FILE).read_bytes())
        assert stored == signature
        assert stored["certificate_sha256"] == certificate["certificate_sha256"]


class TestExternalChecks:
    def test_source_drift_detected(self, tmp_path: Path) -> None:
        payloads = _payloads()
        material = _material(payloads)
        repo = tmp_path / "repo"
        (repo / "core").mkdir(parents=True)
        (repo / "core" / "task_generator.py").write_bytes(b"generator-source")
        (repo / "core" / "rng_streams.py").write_bytes(b"rng-source")
        certificate = build_frozen_baseline_certificate(material)
        checked = verify_frozen_baseline_sources(certificate, repo_root=repo)
        assert len(checked) == 2
        (repo / "core" / "task_generator.py").write_bytes(b"generator-changed")
        with pytest.raises(FrozenBaselineError) as error:
            verify_frozen_baseline_sources(certificate, repo_root=repo)
        assert error.value.code == "frozen_baseline_source_drift"

    def test_model_drift_detected(self, tmp_path: Path) -> None:
        model_root = tmp_path / "model"
        model_root.mkdir()
        weights = model_root / "model-00001.safetensors"
        weights.write_bytes(b"weight-bytes")
        behavior_payloads = {}
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            payload = f'{{"file":"{name}"}}'.encode("ascii")
            (model_root / name).write_bytes(payload)
            behavior_payloads[name] = payload

        from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
            full_weight_checkpoint_identity,
            model_behavior_bundle_identity,
        )

        payloads = _payloads()
        material = _material(payloads)
        material["model"] = {
            "path": str(model_root),
            "checkpoint": full_weight_checkpoint_identity(model_root),
            "behavior_bundle": model_behavior_bundle_identity(model_root),
        }
        certificate = build_frozen_baseline_certificate(material)
        verified = verify_frozen_baseline_model(certificate, model_root=model_root)
        assert verified["checkpoint"] == material["model"]["checkpoint"]

        weights.write_bytes(b"weight-bytes-changed")
        with pytest.raises(FrozenBaselineError) as error:
            verify_frozen_baseline_model(certificate, model_root=model_root)
        assert error.value.code == "frozen_baseline_model_checkpoint_drift"
