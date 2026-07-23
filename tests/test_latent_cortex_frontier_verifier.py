from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from core.brain.frontier_evidence_v5 import canonical_json_bytes
from core.brain.llm.latent_cortex.frontier_artifacts import (
    RAW_ARTIFACT_MANIFEST_SCHEMA,
    TRIAL_VERIFIER_RECEIPT_SCHEMA,
)
from core.brain.llm.latent_cortex.frontier_certification import (
    INDEPENDENT_ATTESTATION_SCHEMA,
)
from core.brain.llm.latent_cortex.frontier_verifier import (
    ATTESTATION_REQUEST_SCHEMA,
    VERIFICATION_KERNEL_IDENTITY_SCHEMA,
    FrontierVerificationError,
    prepare_independent_attestation_request,
    verifier_implementation_sha256,
    verify_frontier_evidence_package,
)
from core.runtime.subprocess_gateway import SubprocessGateway
from tests.fixtures.latent_frontier import (
    _TASK_ISSUER_PUBLIC_KEY,
    _VERIFIER_ID,
    _VERIFIER_KEY,
    _VERIFIER_PUBLIC_KEY,
    _bundle,
    _refresh_attestation,
    _refresh_task_commitment,
    _trial_accounting,
    _trust_config,
)

_STORE_NAMES = (
    "task_payloads",
    "treatment_outputs",
    "control_outputs",
    "scorer_configs",
    "verifier_receipts",
    "structured_receipts",
)
_EXTERNAL_COMPARISON = "resident_32b_vs_external_frontier"


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _publish_store(package: dict, name: str) -> None:
    payload = _jsonl(package["rows"][name])
    path = package["root"] / "raw" / f"{name}.jsonl"
    path.write_bytes(payload)
    package["manifest"]["stores"][name] = {
        "path": f"raw/{name}.jsonl",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "record_count": len(package["rows"][name]),
    }


def _publish_manifest_and_bundle(package: dict, *, signed: bool) -> None:
    manifest_bytes = canonical_json_bytes(package["manifest"])
    package["manifest_path"].write_bytes(manifest_bytes)
    package["bundle"]["raw_artifact_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    if signed:
        _refresh_attestation(package["bundle"])
    else:
        package["bundle"].pop("independent_verifier", None)
    _write_json(package["bundle_path"], package["bundle"])


def _build_package(
    tmp_path: Path,
    *,
    signed: bool = True,
    comparison_kind: str = "resident_32b_vs_vanilla_same_checkpoint",
) -> dict:
    root = tmp_path / "frontier-package"
    raw = root / "raw"
    raw.mkdir(parents=True)
    external = comparison_kind == _EXTERNAL_COMPARISON
    store_names = (*_STORE_NAMES, "provider_receipts") if external else _STORE_NAMES
    bundle = _bundle(include_attestation=False, comparison_kind=comparison_kind)
    rows = {name: [] for name in store_names}

    for trial in bundle["trials"]:
        trial_id = trial["trial_id"]
        task = canonical_json_bytes({"prompt": f"fresh task {trial_id}"})
        treatment = f"treatment output {trial_id}".encode()
        control = f"control output {trial_id}".encode()
        scorer = canonical_json_bytes(
            {"implementation_sha256": "a" * 64, "mode": "exact", "trial_id": trial_id}
        )
        trial["task_payload_sha256"] = hashlib.sha256(task).hexdigest()
        _, treatment_information = _trial_accounting(trial["task_payload_sha256"])
        _, control_information = _trial_accounting(trial["task_payload_sha256"])
        trial["treatment_information"] = treatment_information
        trial["control_information"] = control_information
        trial["treatment_information_sha256"] = treatment_information[
            "receipt_sha256"
        ]
        trial["control_information_sha256"] = control_information[
            "receipt_sha256"
        ]
        trial["treatment_output_sha256"] = hashlib.sha256(treatment).hexdigest()
        trial["control_output_sha256"] = hashlib.sha256(control).hexdigest()
        trial["scorer_config_sha256"] = hashlib.sha256(scorer).hexdigest()
        verifier_receipt = {
            "schema": TRIAL_VERIFIER_RECEIPT_SCHEMA,
            "trial_id": trial_id,
            "task_payload_sha256": trial["task_payload_sha256"],
            "scorer_config_sha256": trial["scorer_config_sha256"],
            "treatment_output_sha256": trial["treatment_output_sha256"],
            "control_output_sha256": trial["control_output_sha256"],
            "treatment_success": trial["treatment_success"],
            "control_success": trial["control_success"],
            "verifier_blinded": trial["verifier_blinded"],
            "scorer_implementation_sha256": "a" * 64,
            "verified_at": trial["evaluation_started_at"] + 0.5,
        }
        verifier_payload = canonical_json_bytes(verifier_receipt)
        trial["verifier_receipt_sha256"] = hashlib.sha256(verifier_payload).hexdigest()
        payloads = {
            "task_payloads": task,
            "treatment_outputs": treatment,
            "control_outputs": control,
            "scorer_configs": scorer,
            "verifier_receipts": verifier_payload,
        }
        if external:
            provider = canonical_json_bytes(
                {
                    "model_id": trial["control_receipt"]["model_id"],
                    "provider": trial["control_receipt"]["provider"],
                    "request_id": trial["control_receipt"]["request_id"],
                    "response": f"control output {trial_id}",
                }
            )
            trial["control_receipt"]["provider_receipt_sha256"] = hashlib.sha256(
                provider
            ).hexdigest()
            payloads["provider_receipts"] = provider
        for store, payload in payloads.items():
            rows[store].append(
                {
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "trial_id": trial_id,
                }
            )
        rows["structured_receipts"].append(
            {
                "control_compute": trial["control_compute"],
                "control_information": trial["control_information"],
                "control_receipt": trial["control_receipt"],
                "treatment_compute": trial["treatment_compute"],
                "treatment_information": trial["treatment_information"],
                "treatment_receipt": trial["treatment_receipt"],
                "trial_id": trial_id,
            }
        )

    _refresh_task_commitment(bundle)
    manifest = {
        "schema": RAW_ARTIFACT_MANIFEST_SCHEMA,
        "producer_id": bundle["producer_id"],
        "preregistration_sha256": bundle["preregistration_sha256"],
        "task_commitment_sha256": bundle["task_commitment_sha256"],
        "stores": {},
    }
    package = {
        "root": root,
        "bundle": bundle,
        "manifest": manifest,
        "rows": rows,
        "bundle_path": root / "bundle.json",
        "manifest_path": root / "manifest.json",
        "trust_path": root / "trust.json",
    }
    for name in store_names:
        _publish_store(package, name)
    _publish_manifest_and_bundle(package, signed=signed)
    _write_json(package["trust_path"], _trust_config())
    return package


def _verify(package: dict) -> dict:
    return verify_frontier_evidence_package(
        bundle_path=package["bundle_path"],
        manifest_path=package["manifest_path"],
        artifact_root=package["root"],
        trust_path=package["trust_path"],
    )


def test_standalone_verifier_recomputes_complete_raw_package(tmp_path: Path):
    package = _build_package(tmp_path)

    first = _verify(package)
    second = _verify(package)

    assert first == second
    assert first["accepted"] is True, first["reasons"]
    assert first["release_accepted"] is False
    assert first["verification_status"] == "ACCEPTED"
    assert first["claim_tier"] == "PROVEN"
    assert first["statistical_tier"] == "PROVEN"
    assert first["ablation_status"] == "NOT_VERIFIED"
    assert first["artifact_verification"]["trial_count"] == 120
    assert first["artifact_verification"]["artifact_store_count"] == 6
    assert first["core_certificate"]["accepted"] is True
    assert first["certificate_sha256"]


def test_attestation_prepare_and_external_signature_complete_the_flow(tmp_path: Path):
    package = _build_package(tmp_path, signed=False)

    request = prepare_independent_attestation_request(
        bundle_path=package["bundle_path"],
        manifest_path=package["manifest_path"],
        artifact_root=package["root"],
        trust_path=package["trust_path"],
        verifier_id=_VERIFIER_ID,
        verified_at=3000.0,
    )

    assert request["schema"] == ATTESTATION_REQUEST_SCHEMA
    assert request["signed_payload"]["claim_tier"] == "PROVEN"
    payload = request["signed_payload"]
    package["bundle"]["independent_verifier"] = {
        "schema": INDEPENDENT_ATTESTATION_SCHEMA,
        "signed_payload": payload,
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": _VERIFIER_ID,
            "public_key_b64": _VERIFIER_PUBLIC_KEY,
            "signature_b64": base64.b64encode(
                _VERIFIER_KEY.sign(canonical_json_bytes(payload))
            ).decode("ascii"),
        },
    }
    _write_json(package["bundle_path"], package["bundle"])

    certificate = _verify(package)
    assert certificate["accepted"] is True, certificate["reasons"]


@pytest.mark.parametrize("verified_at", [None, True, "invalid", float("nan"), float("inf")])
def test_attestation_prepare_rejects_invalid_direct_timestamps(
    tmp_path: Path,
    verified_at: object,
):
    package = _build_package(tmp_path, signed=False)

    with pytest.raises(FrontierVerificationError, match="attestation_verified_at_invalid"):
        prepare_independent_attestation_request(
            bundle_path=package["bundle_path"],
            manifest_path=package["manifest_path"],
            artifact_root=package["root"],
            trust_path=package["trust_path"],
            verifier_id=_VERIFIER_ID,
            verified_at=verified_at,  # type: ignore[arg-type]
        )


def test_attestation_prepare_rejects_timestamp_before_evidence(tmp_path: Path):
    package = _build_package(tmp_path, signed=False)

    with pytest.raises(
        FrontierVerificationError,
        match="attestation_verified_at_not_after_evidence",
    ):
        prepare_independent_attestation_request(
            bundle_path=package["bundle_path"],
            manifest_path=package["manifest_path"],
            artifact_root=package["root"],
            trust_path=package["trust_path"],
            verifier_id=_VERIFIER_ID,
            verified_at=1300.0,
        )


def test_standalone_verifier_rejects_changed_store_bytes(tmp_path: Path):
    package = _build_package(tmp_path)
    target = package["root"] / "raw" / "treatment_outputs.jsonl"
    target.write_bytes(target.read_bytes() + b"{}\n")

    first = _verify(package)
    second = _verify(package)

    assert first == second
    assert first["accepted"] is False
    assert first["reasons"] == ["treatment_outputs_size_mismatch"]


def test_standalone_verifier_rejects_structured_receipt_divergence(tmp_path: Path):
    package = _build_package(tmp_path)
    package["rows"]["structured_receipts"][0]["treatment_compute"] = {
        "estimated_flops": 7.0,
        "estimator_sha256": "e" * 64,
        "layer_apps": 1,
    }
    _publish_store(package, "structured_receipts")
    _publish_manifest_and_bundle(package, signed=True)

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["structured_receipts_bundle_mismatch"]


def test_standalone_verifier_rejects_missing_trial_record(tmp_path: Path):
    package = _build_package(tmp_path)
    package["rows"]["control_outputs"].pop()
    _publish_store(package, "control_outputs")
    _publish_manifest_and_bundle(package, signed=True)

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["control_outputs_trial_set_mismatch"]


def test_standalone_verifier_rejects_traversal_even_when_rehashed(tmp_path: Path):
    package = _build_package(tmp_path)
    package["manifest"]["stores"]["task_payloads"]["path"] = "../task_payloads.jsonl"
    _publish_manifest_and_bundle(package, signed=True)

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["task_payloads_path_invalid"]


@pytest.mark.parametrize("case", ["symlink", "extra-file"])
def test_standalone_verifier_rejects_unmanifested_or_linked_files(tmp_path: Path, case: str):
    package = _build_package(tmp_path)
    if case == "symlink":
        target = package["root"] / "raw" / "task_payloads.jsonl"
        backup = package["root"] / "task-payload-backup"
        backup.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(backup)
        expected = "raw_artifact_symlink_rejected"
    else:
        (package["root"] / "raw" / "unmanifested.txt").write_text("hidden evidence")
        expected = "raw_artifact_file_set_mismatch"

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == [expected]


def test_standalone_verifier_rejects_duplicate_json_keys(tmp_path: Path):
    package = _build_package(tmp_path)
    package["trust_path"].write_bytes(
        b'{"schema":"aura.latent_cortex.frontier_trust.v1",'
        b'"schema":"duplicate","task_issuers":{},"verifiers":{}}'
    )

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["trust_config_duplicate_json_key"]


def test_standalone_verifier_rejects_cross_role_key_reuse(tmp_path: Path):
    package = _build_package(tmp_path)
    trust = copy.deepcopy(_trust_config())
    trust["verifiers"][_VERIFIER_ID]["public_key_b64"] = _TASK_ISSUER_PUBLIC_KEY
    _write_json(package["trust_path"], trust)

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["trust_role_key_reused"]


def test_standalone_verifier_rejects_unpinned_local_verification_kernel(
    tmp_path: Path,
):
    package = _build_package(tmp_path)
    trust = copy.deepcopy(_trust_config())
    trust["verification_kernel_sha256"] = "0" * 64
    _write_json(package["trust_path"], trust)

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["verification_kernel_trust_pin_mismatch"]


def test_standalone_verifier_accepts_complete_external_frontier_package(tmp_path: Path):
    package = _build_package(tmp_path, comparison_kind=_EXTERNAL_COMPARISON)

    first = _verify(package)
    second = _verify(package)

    assert first == second
    assert first["accepted"] is True, first["reasons"]
    assert first["claim_tier"] == "PROVEN"
    assert first["core_certificate"]["comparison_kind"] == _EXTERNAL_COMPARISON
    assert first["artifact_verification"]["artifact_store_count"] == 7
    assert first["artifact_verification"]["trial_count"] == 120


def test_standalone_verifier_refuses_relabeled_external_comparison_without_provider_receipts(
    tmp_path: Path,
):
    package = _build_package(tmp_path)
    package["bundle"]["preregistration"]["comparison_kind"] = _EXTERNAL_COMPARISON
    _write_json(package["bundle_path"], package["bundle"])

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["raw_artifact_store_set_invalid"]


def test_standalone_verifier_rejects_external_package_missing_provider_store(
    tmp_path: Path,
):
    package = _build_package(tmp_path, comparison_kind=_EXTERNAL_COMPARISON)
    del package["manifest"]["stores"]["provider_receipts"]
    (package["root"] / "raw" / "provider_receipts.jsonl").unlink()
    _publish_manifest_and_bundle(package, signed=True)

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["raw_artifact_store_set_invalid"]


def test_standalone_verifier_rejects_vanilla_package_shipping_provider_store(
    tmp_path: Path,
):
    package = _build_package(tmp_path)
    package["rows"]["provider_receipts"] = [
        {
            "payload_b64": base64.b64encode(b"unexpected provider evidence").decode("ascii"),
            "trial_id": package["bundle"]["trials"][0]["trial_id"],
        }
    ]
    _publish_store(package, "provider_receipts")
    _publish_manifest_and_bundle(package, signed=True)

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["raw_artifact_store_set_invalid"]


def test_standalone_verifier_rejects_forged_provider_receipt_bytes(tmp_path: Path):
    package = _build_package(tmp_path, comparison_kind=_EXTERNAL_COMPARISON)
    package["rows"]["provider_receipts"][0]["payload_b64"] = base64.b64encode(
        b"forged provider response"
    ).decode("ascii")
    _publish_store(package, "provider_receipts")
    _publish_manifest_and_bundle(package, signed=True)

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["provider_receipts_bundle_hash_mismatch"]


def test_standalone_verifier_rejects_missing_provider_receipt_record(tmp_path: Path):
    package = _build_package(tmp_path, comparison_kind=_EXTERNAL_COMPARISON)
    package["rows"]["provider_receipts"].pop()
    _publish_store(package, "provider_receipts")
    _publish_manifest_and_bundle(package, signed=True)

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["reasons"] == ["provider_receipts_trial_set_mismatch"]


def test_standalone_verifier_rejects_unknown_comparison_kind(tmp_path: Path):
    package = _build_package(tmp_path)
    package["bundle"]["preregistration"]["comparison_kind"] = "resident_32b_vs_wishful_thinking"
    _write_json(package["bundle_path"], package["bundle"])

    certificate = _verify(package)

    assert certificate["accepted"] is False
    assert certificate["verification_status"] == "MALFORMED_OR_TAMPERED"
    assert certificate["reasons"] == ["comparison_kind_unsupported"]


def test_external_attestation_prepare_and_signature_complete_the_flow(tmp_path: Path):
    package = _build_package(tmp_path, signed=False, comparison_kind=_EXTERNAL_COMPARISON)

    request = prepare_independent_attestation_request(
        bundle_path=package["bundle_path"],
        manifest_path=package["manifest_path"],
        artifact_root=package["root"],
        trust_path=package["trust_path"],
        verifier_id=_VERIFIER_ID,
        verified_at=3000.0,
    )

    payload = request["signed_payload"]
    package["bundle"]["independent_verifier"] = {
        "schema": INDEPENDENT_ATTESTATION_SCHEMA,
        "signed_payload": payload,
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": _VERIFIER_ID,
            "public_key_b64": _VERIFIER_PUBLIC_KEY,
            "signature_b64": base64.b64encode(
                _VERIFIER_KEY.sign(canonical_json_bytes(payload))
            ).decode("ascii"),
        },
    }
    _write_json(package["bundle_path"], package["bundle"])

    certificate = _verify(package)
    assert certificate["accepted"] is True, certificate["reasons"]
    assert certificate["core_certificate"]["comparison_kind"] == _EXTERNAL_COMPARISON


def test_standalone_cli_returns_truthful_exit_codes(tmp_path: Path):
    package = _build_package(tmp_path)
    tool = Path(__file__).resolve().parents[1] / "tools" / "verify_latent_cortex_frontier.py"
    command = [
        sys.executable,
        str(tool),
        "verify",
        "--bundle",
        str(package["bundle_path"]),
        "--manifest",
        str(package["manifest_path"]),
        "--artifact-root",
        str(package["root"]),
        "--trust",
        str(package["trust_path"]),
    ]

    accepted = SubprocessGateway().run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30.0,
        read_only=True,
        source="proof_tooling:latent_cortex_frontier_cli",
    )
    accepted_payload = json.loads(accepted.stdout)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted_payload["accepted"] is True

    package["bundle"]["trials"][0]["treatment_success"] = False
    _write_json(package["bundle_path"], package["bundle"])
    rejected = SubprocessGateway().run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30.0,
        read_only=True,
        source="proof_tooling:latent_cortex_frontier_cli",
    )
    rejected_payload = json.loads(rejected.stdout)
    assert rejected.returncode == 1
    assert rejected_payload["accepted"] is False


def test_standalone_cli_emits_exact_verification_kernel_fingerprint():
    tool = Path(__file__).resolve().parents[1] / "tools" / "verify_latent_cortex_frontier.py"
    result = SubprocessGateway().run(
        [sys.executable, str(tool), "fingerprint"],
        cwd=Path(__file__).resolve().parents[1],
        timeout=30.0,
        read_only=True,
        source="proof_tooling:latent_cortex_frontier_fingerprint",
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stderr
    assert payload == {
        "schema": VERIFICATION_KERNEL_IDENTITY_SCHEMA,
        "verification_kernel_sha256": verifier_implementation_sha256(),
    }
