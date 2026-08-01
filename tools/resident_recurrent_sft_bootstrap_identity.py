"""Shared resident bootstrap identities independently replayed at launch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    personality_bundle_identity,
    runtime_environment_identity,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import sha256_json
from tools.validate_structured_sft_tokenization import (
    resident_tokenizer_artifact_identity,
    resident_tokenizer_runtime_identity,
)


def absent_personality_identity() -> dict[str, Any]:
    body = personality_bundle_identity(None)
    return {**body, "identity_sha256": sha256_json(body)}


def resident_bootstrap_tokenizer_identity(
    model_directory: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    artifact = resident_tokenizer_artifact_identity(model_directory)
    runtime = resident_tokenizer_runtime_identity(tokenizer)
    body = {
        "schema": "aura.resident_recurrent_sft_tokenizer_identity.v1",
        "artifact": artifact,
        "runtime": runtime,
        "artifact_sha256": artifact["sha256"],
        "runtime_sha256": runtime["sha256"],
    }
    return {**body, "identity_sha256": sha256_json(body)}


def resident_bootstrap_runtime_identity() -> dict[str, Any]:
    return cast(dict[str, Any], runtime_environment_identity())


__all__ = [
    "absent_personality_identity",
    "resident_bootstrap_runtime_identity",
    "resident_bootstrap_tokenizer_identity",
]
