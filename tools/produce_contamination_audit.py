#!/usr/bin/env python
"""Signed zero-overlap contamination audit for the paired resident campaign.

The confirmatory campaign refuses to run without a signed audit proving the
frozen evaluation battery shares nothing with the exact training corpus:

    produce  — recompute the campaign's task manifest, sweep every task
               prompt against every training prompt under three methods
               (exact_prompt, normalized_prompt, token_fivegram), write the
               audit body with EXACTLY the schema the campaign verifies,
               and sign it with an Ed25519 key held OUTSIDE the repository.
    verify   — standalone recomputation: regenerate the tasks, recompute
               the overlap from the corpus bytes, and check the signature
               against the trust root — trusting nothing but raw inputs.
    keygen   — create the external Ed25519 keypair (private key 0600).

The audit is honest by construction: an overlap yields status
"failed_overlap" and the campaign will reject it; auditor_independence is
"external" only when both key files physically live outside the repository
tree. Method names and normalization are imported from the SAME modules the
campaign uses, so the audit can never drift from what it certifies.

Usage:
  .venv/bin/python tools/produce_contamination_audit.py keygen \
      --key ~/.aura/trust/contamination_audit_ed25519_private.pem \
      --trust-root ~/.aura/trust/contamination_audit_ed25519_public.pem
  .venv/bin/python tools/produce_contamination_audit.py produce \
      --training-manifest .../adapter/dataset_manifest.json \
      --corpus-name resident_32b_v2_cp139_training_corpus \
      --seeds 20260801,20260802 --domains all --difficulty 2 \
      --task-registry-version 2026.07.18.2 \
      --key ~/.aura/trust/contamination_audit_ed25519_private.pem \
      --trust-root ~/.aura/trust/contamination_audit_ed25519_public.pem \
      --out artifacts/current/contamination_audit_powered.json \
      --report-out artifacts/current/contamination_audit_powered_report.json
  .venv/bin/python tools/produce_contamination_audit.py verify \
      --audit artifacts/current/contamination_audit_powered.json \
      --training-manifest .../adapter/dataset_manifest.json \
      --seeds 20260801,20260802 --domains all --difficulty 2 \
      --task-registry-version 2026.07.18.2 \
      --trust-root ~/.aura/trust/contamination_audit_ed25519_public.pem
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    CONTAMINATION_SAFE_REGISTRY_VERSION,
    CURRENT_REGISTRY_VERSION,
    FRONTIER_DOMAINS,
    REGISTRY_VERSION,
    _normalized_prompt,
    build_task_manifest,
    generate_task_battery,
)
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    CONTAMINATION_AUDIT_SCHEMA,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import (  # noqa: E402
    build_dataset_commitment,
)

AUDIT_METHODS = ("exact_prompt", "normalized_prompt", "token_fivegram")
MAX_MANIFEST_BYTES = 256 * 1024 * 1024


class AuditError(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise AuditError(f"not a regular file: {path}")
    size = resolved.stat().st_size
    if size <= 0 or size > max_bytes:
        raise AuditError(f"file size out of bounds ({size} bytes): {path}")
    return resolved.read_bytes()


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise AuditError(f"symlink output rejected: {path}")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise AuditError(f"existing artifact differs: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise AuditError(f"short write: {temporary}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _load_corpus(
    manifest_path: Path,
) -> tuple[list[str], str, list[dict[str, object]]]:
    """Load a legacy manifest or canonical recurrent-SFT row array."""
    payload = _read_regular_file(manifest_path, max_bytes=MAX_MANIFEST_BYTES)
    snapshot_sha256 = _sha256_bytes(payload)
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AuditError("training manifest is not valid JSON") from exc
    if isinstance(manifest, dict):
        examples = manifest.get("examples")
    else:
        examples = manifest
    if not isinstance(examples, list) or not examples:
        raise AuditError("training manifest has no examples")
    prompts: list[str] = []
    rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        prompt = example.get("prompt") if isinstance(example, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise AuditError(f"training example {index} has no prompt")
        prompts.append(prompt)
        rows.append(dict(example))
    return prompts, snapshot_sha256, rows


def _training_corpus_material(
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, str]], str, int]:
    training_path = Path(args.training_manifest)
    training_prompts, training_sha256, training_rows = _load_corpus(training_path)
    corpus_name = str(getattr(args, "corpus_name", "training") or "training")
    corpora = [
        {
            "name": corpus_name,
            "snapshot_sha256": training_sha256,
        }
    ]
    prompts = list(training_prompts)
    row_count = len(training_rows)
    validation_value = str(getattr(args, "validation_manifest", "") or "").strip()
    if validation_value:
        validation_path = Path(validation_value)
        validation_prompts, validation_sha256, validation_rows = _load_corpus(validation_path)
        prompts.extend(validation_prompts)
        row_count += len(validation_rows)
        corpora.append(
            {
                "name": f"{corpus_name}:validation",
                "snapshot_sha256": validation_sha256,
            }
        )

        def commitment_rows(
            rows: list[dict[str, object]], *, split: str
        ) -> list[dict[str, object]]:
            normalized: list[dict[str, object]] = []
            required = {"task_id", "family", "depth", "prompt", "answer", "ordinal"}
            for index, row in enumerate(rows):
                if set(row) != required or row.get("ordinal") != index:
                    raise AuditError(f"{split} manifest is not a canonical recurrent-SFT row array")
                normalized.append({key: value for key, value in row.items() if key != "ordinal"})
            return normalized

        try:
            dataset_identity = str(
                build_dataset_commitment(
                    commitment_rows(training_rows, split="training"),
                    commitment_rows(validation_rows, split="validation"),
                )["dataset_sha256"]
            )
        except (TypeError, ValueError) as exc:
            raise AuditError(
                "training and validation manifests do not form a canonical "
                "resident recurrent-SFT dataset"
            ) from exc
    else:
        dataset_identity = training_sha256
    expected_identity = str(getattr(args, "training_dataset_identity_sha256", "") or "").strip()
    if expected_identity and expected_identity != dataset_identity:
        raise AuditError(
            "recomputed training dataset identity does not match the declared identity"
        )
    return prompts, corpora, dataset_identity, row_count


def _fivegrams(prompt: str) -> set[str]:
    tokens = _normalized_prompt(prompt).split()
    return {" ".join(tokens[index : index + 5]) for index in range(max(0, len(tokens) - 4))}


def _overlap_report(
    task_prompts: list[tuple[str, str]],
    training_prompts: list[str],
) -> tuple[int, list[dict[str, object]]]:
    """Sweep every campaign prompt against the whole corpus, all methods."""
    exact = {_sha256_bytes(p.encode("utf-8")) for p in training_prompts}
    normalized = {_sha256_bytes(_normalized_prompt(p).encode("utf-8")) for p in training_prompts}
    corpus_fivegrams: set[str] = set()
    for prompt in training_prompts:
        corpus_fivegrams |= _fivegrams(prompt)
    hits: list[dict[str, object]] = []
    contaminated_tasks = 0
    for task_id, prompt in task_prompts:
        methods_hit: list[str] = []
        if _sha256_bytes(prompt.encode("utf-8")) in exact:
            methods_hit.append("exact_prompt")
        if _sha256_bytes(_normalized_prompt(prompt).encode("utf-8")) in normalized:
            methods_hit.append("normalized_prompt")
        shared = _fivegrams(prompt) & corpus_fivegrams
        if shared:
            methods_hit.append("token_fivegram")
        if methods_hit:
            contaminated_tasks += 1
            hits.append(
                {
                    "task_id": task_id,
                    "methods": methods_hit,
                    "shared_fivegrams": sorted(shared)[:20],
                }
            )
    return contaminated_tasks, hits


def _campaign_manifest(args: argparse.Namespace):
    seeds = tuple(int(value) for value in str(args.seeds).split(",") if value)
    if not seeds:
        raise AuditError("at least one seed is required")
    if args.domains.strip().lower() == "all":
        domains = FRONTIER_DOMAINS
    else:
        domains = tuple(value.strip() for value in args.domains.split(",") if value.strip())
    tasks = generate_task_battery(
        seeds,
        domains=domains,
        difficulty=int(args.difficulty),
        registry_version=args.task_registry_version,
    )
    manifest = build_task_manifest(tasks)
    task_prompts = [(record.task_id, record.prompt) for record in manifest.tasks]
    return manifest, task_prompts


def _is_outside_repo(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return True
    return False


def _load_private_key(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    key_bytes = _read_regular_file(path, max_bytes=64 * 1024)
    private_key = serialization.load_pem_private_key(key_bytes, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise AuditError("signing key is not Ed25519")
    return private_key


def _public_der_sha256(public_key) -> tuple[bytes, str]:
    from cryptography.hazmat.primitives import serialization

    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return der, _sha256_bytes(der)


def cmd_keygen(args: argparse.Namespace) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    key_path = Path(args.key).expanduser()
    trust_path = Path(args.trust_root).expanduser()
    for path in (key_path, trust_path):
        if path.exists():
            print(f"refusing to overwrite existing key material: {path}")
            return 1
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _atomic_write(key_path, private_pem, mode=0o600)
    _atomic_write(trust_path, public_pem)
    _, key_id = _public_der_sha256(private_key.public_key())
    location = "external" if _is_outside_repo(key_path) else "INSIDE REPO"
    print(f"keypair created; key_id={key_id}; private key location: {location}")
    if not _is_outside_repo(key_path):
        print(
            "WARNING: the private key is inside the repository tree — audits "
            "signed with it cannot claim external independence"
        )
    return 0


def _build_audit_body(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    training_prompts, corpora, dataset_identity, corpus_size = _training_corpus_material(args)
    manifest, task_prompts = _campaign_manifest(args)
    overlap_count, hits = _overlap_report(task_prompts, training_prompts)
    key_external = _is_outside_repo(Path(args.key)) and _is_outside_repo(Path(args.trust_root))
    body = {
        "schema": CONTAMINATION_AUDIT_SCHEMA,
        "task_manifest_sha256": manifest.manifest_sha256,
        "status": ("passed_zero_overlap" if overlap_count == 0 else "failed_overlap"),
        "overlap_count": overlap_count,
        "auditor_independence": "external" if key_external else "internal",
        "training_dataset_identity_sha256": dataset_identity,
        "corpora": corpora,
        "methods": list(AUDIT_METHODS),
    }
    report = [
        {
            "task_count": len(task_prompts),
            "corpus_examples": corpus_size,
            "registry_version": manifest.registry_version,
            "hits": hits,
        }
    ]
    return body, report


def cmd_produce(args: argparse.Namespace) -> int:
    body, report = _build_audit_body(args)
    private_key = _load_private_key(Path(args.key))
    public_der, key_id = _public_der_sha256(private_key.public_key())
    trust_der, trust_key_id = _trust_root_der(Path(args.trust_root))
    if trust_der != public_der:
        raise AuditError("trust root does not match the signing key")
    signature = private_key.sign(canonical_json_bytes(body))
    audit = {
        **body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": key_id,
            "signature_b64": base64.b64encode(signature).decode("ascii"),
        },
    }
    payload = json.dumps(audit, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_write(Path(args.out), payload)
    if args.report_out:
        report_payload = json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        _atomic_write(Path(args.report_out), report_payload)
    print(
        f"audit written: status={body['status']} overlap_count="
        f"{body['overlap_count']} independence={body['auditor_independence']} "
        f"key_id={key_id}"
    )
    if body["status"] != "passed_zero_overlap":
        print("OVERLAP FOUND — the campaign will reject this audit (see report)")
        return 2
    if body["auditor_independence"] != "external":
        print(
            "WARNING: key material inside the repository — campaign requires external independence"
        )
        return 3
    return 0


def _trust_root_der(path: Path) -> tuple[bytes, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )

    trust_bytes = _read_regular_file(path, max_bytes=64 * 1024)
    public_key = serialization.load_pem_public_key(trust_bytes)
    if not isinstance(public_key, Ed25519PublicKey):
        raise AuditError("trust root is not Ed25519")
    return _public_der_sha256(public_key)


def cmd_verify(args: argparse.Namespace) -> int:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.serialization import (
        load_pem_public_key,
    )

    audit_bytes = _read_regular_file(Path(args.audit), max_bytes=8 * 1024 * 1024)
    audit = json.loads(audit_bytes)
    required = {
        "schema",
        "task_manifest_sha256",
        "status",
        "overlap_count",
        "auditor_independence",
        "training_dataset_identity_sha256",
        "corpora",
        "methods",
        "signature",
    }
    if not isinstance(audit, dict) or set(audit) != required:
        print("FAIL: audit schema keys are wrong")
        return 1
    body = dict(audit)
    signature = body.pop("signature")

    failures: list[str] = []
    if audit.get("schema") != CONTAMINATION_AUDIT_SCHEMA:
        failures.append("audit schema is not the current producer schema")
    # 1. Signature against the trust root, over the canonical body bytes.
    trust_bytes = _read_regular_file(Path(args.trust_root), max_bytes=64 * 1024)
    public_key = load_pem_public_key(trust_bytes)
    _, trust_key_id = _public_der_sha256(public_key)
    if signature.get("algorithm") != "ed25519":
        failures.append("signature algorithm is not ed25519")
    if signature.get("key_id") != trust_key_id:
        failures.append("signer key_id does not match trust root")
    try:
        public_key.verify(
            base64.b64decode(str(signature.get("signature_b64", "")), validate=True),
            canonical_json_bytes(body),
        )
    except (InvalidSignature, ValueError):
        failures.append("signature verification failed")
    # 2. Task manifest recomputed from the declared generation parameters.
    manifest, task_prompts = _campaign_manifest(args)
    if audit["task_manifest_sha256"] != manifest.manifest_sha256:
        failures.append("task manifest hash does not match regenerated battery")
    # 3. Corpus snapshot + overlap recomputed from raw bytes.
    training_prompts, corpora, dataset_identity, _ = _training_corpus_material(args)
    declared = {
        record.get("snapshot_sha256")
        for record in audit.get("corpora", [])
        if isinstance(record, dict)
    }
    expected_snapshots = {record["snapshot_sha256"] for record in corpora}
    if declared != expected_snapshots:
        failures.append("training corpus snapshot hashes do not match the audit")
    if audit.get("training_dataset_identity_sha256") != dataset_identity:
        failures.append("training dataset identity does not match the audit")
    overlap_count, _hits = _overlap_report(task_prompts, training_prompts)
    if overlap_count != audit["overlap_count"]:
        failures.append(f"recomputed overlap {overlap_count} != declared {audit['overlap_count']}")
    expected_status = "passed_zero_overlap" if overlap_count == 0 else "failed_overlap"
    if audit["status"] != expected_status:
        failures.append("status does not match recomputed overlap")
    if not set(AUDIT_METHODS).issubset(audit.get("methods", [])):
        failures.append("required methods missing")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(
        f"VERIFIED: {audit['status']} overlap_count={audit['overlap_count']} "
        f"tasks={len(task_prompts)} signer={trust_key_id[:16]}…"
    )
    return 0


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--validation-manifest", default="")
    parser.add_argument("--training-dataset-identity-sha256", default="")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--domains", default="all")
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument(
        "--task-registry-version",
        default=CURRENT_REGISTRY_VERSION,
        choices=sorted(
            {
                REGISTRY_VERSION,
                CURRENT_REGISTRY_VERSION,
                CONTAMINATION_SAFE_REGISTRY_VERSION,
            }
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    keygen = sub.add_parser("keygen", help="create the external Ed25519 keypair")
    keygen.add_argument("--key", required=True)
    keygen.add_argument("--trust-root", required=True)
    keygen.set_defaults(func=cmd_keygen)

    produce = sub.add_parser("produce", help="produce and sign the audit")
    _add_generation_arguments(produce)
    produce.add_argument("--corpus-name", required=True)
    produce.add_argument("--key", required=True)
    produce.add_argument("--trust-root", required=True)
    produce.add_argument("--out", required=True)
    produce.add_argument("--report-out", default="")
    produce.set_defaults(func=cmd_produce)

    verify = sub.add_parser("verify", help="independently verify an audit")
    _add_generation_arguments(verify)
    verify.add_argument("--audit", required=True)
    verify.add_argument("--trust-root", required=True)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    try:
        return args.func(args)
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
