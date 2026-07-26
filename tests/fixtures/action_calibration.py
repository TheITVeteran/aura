"""Small certificate-shaped value-of-computation fixtures for unit tests."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Mapping
from fractions import Fraction

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from core.brain.llm.latent_cortex.action_calibration import (
    ACTION_CALIBRATION_EVIDENCE_SCHEMA,
    ACTION_CALIBRATION_FINAL_VERIFIER_SCHEMA,
    ACTION_CALIBRATION_WORKER_ADMISSION_SCHEMA,
    ACTION_RESOURCE_DIMENSIONS,
    GLOBAL_BOUND_FAMILY_COUNT,
)
from core.brain.llm.latent_cortex.campaign_journal import (
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    EVIDENCE_VERIFIER,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.value_of_computation import (
    CertifiedActionEvidence,
)


def _rational(value: float) -> dict[str, int]:
    fraction = Fraction(str(value))
    return {
        "numerator": fraction.numerator,
        "denominator": fraction.denominator,
    }


def _private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(f"action-calibration-fixture:{label}".encode()).digest()
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


def _trust_fixture():
    root = _private("root")
    role_keys = {role: _private(role) for role in CAMPAIGN_TRUST_ROLES}
    roles = {}
    for role, key in role_keys.items():
        public = _public_raw(key)
        roles[role] = {
            "signer_id": f"{role}-fixture-signer",
            "organization_id": f"{role}-fixture-organization",
            "public_key_b64": base64.b64encode(public).decode(),
            "key_id": hashlib.sha256(public).hexdigest(),
            "implementation_sha256": hashlib.sha256(
                f"{role}:fixture-implementation".encode()
            ).hexdigest(),
            "release_sha256": hashlib.sha256(f"{role}:fixture-release".encode()).hexdigest(),
            "custody_class": "external_service",
            "custody_evidence_sha256": hashlib.sha256(
                f"{role}:fixture-custody".encode()
            ).hexdigest(),
        }
    now = int(time.time())
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "certified-action-unit-fixture",
        "policy_revision": 1,
        "campaign_name": "certified-action-unit-fixture",
        "protocol_sha256": hashlib.sha256(b"fixture-protocol").hexdigest(),
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": now - 20,
        "not_before_unix": now - 10,
        "expires_at_unix": now + 86_400,
        "roles": roles,
    }
    root_raw = _public_raw(root)
    signed = canonical_json_bytes(body)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode(),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    root_pem = _public_pem(root)
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=root_pem,
        expected_campaign_name=body["campaign_name"],
        now_unix=now,
    )
    return policy, role_keys[EVIDENCE_VERIFIER], root_pem, now


def certified_action_snapshot(
    *,
    bucket: str,
    cells: Mapping[
        OperationKind,
        tuple[float, float, float, float],
    ],
) -> tuple[dict, bytes]:
    """Build strict v2 cells: (gain_lcb, gain_mean, gain_ucb, cost_ucb)."""

    policy, verifier_key, root_pem, now = _trust_fixture()
    candidate_sha256 = ""
    candidate_cells = {}
    complete_cells = {
        action: cells.get(action, (-1.0, -1.0, -1.0, 1.0)) for action in OperationKind
    }
    for action, (
        gain_lcb,
        gain_mean,
        gain_ucb,
        cost_ucb,
    ) in complete_cells.items():
        cell = CertifiedActionEvidence.from_dict(
            {
                "n": 20,
                "unique_task_count": 20,
                "measured": True,
                "gain_mean": gain_mean,
                "gain_lcb": gain_lcb,
                "gain_ucb": gain_ucb,
                "cost_mean": cost_ucb,
                "cost_ucb": cost_ucb,
                "gain_bounds": {
                    "method": ("simultaneous rational Clopper-Pearson contrast bounds"),
                    "family_count": GLOBAL_BOUND_FAMILY_COUNT,
                    "family_alpha": {
                        "numerator": 1,
                        "denominator": 20,
                    },
                    "component_alpha": {
                        "numerator": 1,
                        "denominator": 680,
                    },
                    "simultaneous_coverage_lower": {
                        "numerator": 19,
                        "denominator": 20,
                    },
                    "lower": _rational(gain_lcb),
                    "upper": _rational(gain_ucb),
                    "certified": True,
                },
                "cost_bounds": {
                    "method": "simultaneous Hoeffding upper bound",
                    "family_count": GLOBAL_BOUND_FAMILY_COUNT,
                    "family_alpha": {
                        "numerator": 1,
                        "denominator": 20,
                    },
                    "bounded_interval": [0.0, 1.0],
                    "normalization": ("max fraction of preregistered action-resource caps"),
                    "dimensions": list(ACTION_RESOURCE_DIMENSIONS),
                },
                "calibration_candidate_sha256": "a" * 64,
                "policy_sha256": policy.policy_sha256,
            }
        )
        candidate_cells[action.value] = {
            name: value
            for name, value in cell.to_dict().items()
            if name
            not in {
                "calibration_candidate_sha256",
                "policy_sha256",
            }
        }
    candidate_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "bucket": bucket,
                "cells": candidate_cells,
                "purpose": "certified-action-unit-fixture",
            }
        )
    ).hexdigest()
    normalized = {
        action: {
            **cell,
            "calibration_candidate_sha256": candidate_sha256,
            "policy_sha256": policy.policy_sha256,
        }
        for action, cell in candidate_cells.items()
    }
    pair_count = sum(cell["n"] for cell in candidate_cells.values())
    execution_count = pair_count * 2
    final_payload = {
        "schema": ACTION_CALIBRATION_FINAL_VERIFIER_SCHEMA,
        "accepted": True,
        "candidate_sha256": candidate_sha256,
        "calibration_bucket": bucket,
        "plan_sha256": hashlib.sha256(b"fixture-plan").hexdigest(),
        "policy_sha256": policy.policy_sha256,
        "campaign_manifest_sha256": hashlib.sha256(b"fixture-manifest").hexdigest(),
        "journal_head_sha256": hashlib.sha256(b"fixture-journal").hexdigest(),
        "journal_event_count": 1 + execution_count * 3,
        "observations_sha256": hashlib.sha256(b"fixture-observations").hexdigest(),
        "cells_sha256": hashlib.sha256(canonical_json_bytes(candidate_cells)).hexdigest(),
        "pair_count": pair_count,
        "execution_count": execution_count,
        "frontier_claim_eligible": False,
    }
    admission = {
        "schema": ACTION_CALIBRATION_WORKER_ADMISSION_SCHEMA,
        "campaign_name": policy.document["campaign_name"],
        "policy_validated_at_unix": now,
        "policy_document": policy.document,
        "final_verifier_payload": final_payload,
        "final_verifier_attestation": build_role_attestation(
            policy,
            role=EVIDENCE_VERIFIER,
            payload=final_payload,
            signed_at_unix=now,
            private_key=verifier_key,
        ),
    }
    body = {
        "schema": ACTION_CALIBRATION_EVIDENCE_SCHEMA,
        "bucket": bucket,
        "candidate_sha256": candidate_sha256,
        "policy_sha256": policy.policy_sha256,
        "admission": admission,
        "cells": normalized,
    }
    snapshot = {
        **body,
        "snapshot_sha256": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }
    return snapshot, root_pem


__all__ = ["certified_action_snapshot"]
