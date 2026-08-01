"""Shared resident bootstrap identities independently replayed at launch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    personality_bundle_identity,
    runtime_environment_identity,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import sha256_json
from tools.validate_structured_sft_tokenization import (
    TokenizerValidationError,
    resident_tokenizer_artifact_identity,
    resident_tokenizer_runtime_identity,
)


def absent_personality_identity() -> dict[str, Any]:
    body = personality_bundle_identity(None)
    return {**body, "identity_sha256": sha256_json(body)}


def load_resident_bootstrap_tokenizer(model_directory: Path) -> Any:
    """Load only the resident tokenizer under the same EOS contract as MLX-LM."""

    directory = model_directory.expanduser().resolve(strict=True)
    try:
        config = json.loads((directory / "config.json").read_bytes())
        eos_token_ids = config.get("eos_token_id")
        eos_values = eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids]
        if not eos_values or any(
            type(token_id) is not int or not 0 <= token_id < 2**31 for token_id in eos_values
        ):
            raise TokenizerValidationError("tokenizer_eos_contract_invalid")
        from mlx_lm.utils import load_tokenizer

        return load_tokenizer(directory, eos_token_ids=eos_token_ids)
    except TokenizerValidationError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TokenizerValidationError("tokenizer_dependency_unavailable") from exc


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
    "load_resident_bootstrap_tokenizer",
    "resident_bootstrap_runtime_identity",
    "resident_bootstrap_tokenizer_identity",
]
