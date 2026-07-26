"""Contracts for the resident worker's boot-scoped capture signer."""

from __future__ import annotations

from copy import deepcopy

import pytest

from core.brain.llm.latent_cortex.worker_capture_identity import (
    WorkerCaptureIdentityError,
    build_worker_capture_identity,
    validate_worker_capture_identity,
)


def test_worker_capture_identity_round_trip_is_boot_scoped():
    identity = build_worker_capture_identity(
        worker_boot_id="a" * 32,
        worker_pid=4242,
    )

    assert validate_worker_capture_identity(identity.public_identity) == (identity.public_identity)
    assert identity.public_identity["worker_boot_id"] == "a" * 32
    assert identity.public_identity["worker_pid"] == 4242


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("worker_boot_id", "b" * 32, "hash_mismatch"),
        ("worker_pid", 4243, "hash_mismatch"),
        ("public_key_b64", "A" * 44, "hash_mismatch"),
        ("signature_b64", "A" * 88, "hash_mismatch"),
    ),
)
def test_worker_capture_identity_rejects_public_tampering(
    field: str,
    replacement,
    error: str,
):
    identity = build_worker_capture_identity(
        worker_boot_id="a" * 32,
        worker_pid=4242,
    )
    attacked = deepcopy(identity.public_identity)
    attacked[field] = replacement

    with pytest.raises(WorkerCaptureIdentityError, match=error):
        validate_worker_capture_identity(attacked)


def test_worker_capture_identity_rejects_extra_fields():
    identity = build_worker_capture_identity(
        worker_boot_id="a" * 32,
        worker_pid=4242,
    )
    attacked = {**identity.public_identity, "private_key": "leak"}

    with pytest.raises(WorkerCaptureIdentityError, match="identity_fields"):
        validate_worker_capture_identity(attacked)
