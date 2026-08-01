"""Shared resident bootstrap identities independently replayed at launch."""

from __future__ import annotations

import hashlib
import json
import os
import sys
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
    runtime = cast(dict[str, Any], runtime_environment_identity())
    runtime_body = dict(runtime)
    runtime_body.pop("identity_sha256")
    executable = Path(os.path.abspath(sys.executable))
    real_executable = executable.resolve(strict=True)
    before = real_executable.stat()
    payload = real_executable.read_bytes()
    after = real_executable.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise TokenizerValidationError("resident_runtime_interpreter_changed")
    runtime_body["interpreter"] = {
        "schema": "aura.resident_recurrent_sft_python.v1",
        "executable": str(executable),
        "real_executable": str(real_executable),
        "sys_prefix": str(Path(sys.prefix).resolve(strict=True)),
        "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    return {**runtime_body, "identity_sha256": sha256_json(runtime_body)}


__all__ = [
    "absent_personality_identity",
    "load_resident_bootstrap_tokenizer",
    "resident_bootstrap_runtime_identity",
    "resident_bootstrap_tokenizer_identity",
]
