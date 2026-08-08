"""Canonical ordinary-decode artifact for monotonic RLC output selection.

The RLC may spend additional compute looking for a better answer, but ordinary
decode remains the incumbent until an independently verified candidate wins.
That floor is only structural when both paths refer to the same generated
artifact. Re-running a nominally identical greedy decode is insufficient:
different decode implementations or numerically close logits can produce
different bytes while every exposed sampler setting still matches.

This module keeps the public, hash-bound receipt separate from the private
token sequence used by the engine. The artifact is immutable and can be
reconstructed and validated without trusting a caller-provided digest.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

INCUMBENT_ARTIFACT_SCHEMA = "aura.rlc.incumbent_artifact.v1"
INCUMBENT_AUTHORITY = "canonical_ordinary_decode"
_TERMINATIONS = frozenset({"contract_complete", "eos", "token_limit"})


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def token_sequence_sha256(tokens: Sequence[int]) -> str:
    return _sha([int(token) for token in tokens])


def input_token_sha256(tokens: Sequence[int]) -> str:
    return token_sequence_sha256(tokens)


@dataclass(frozen=True)
class IncumbentArtifact:
    """Private ordinary output plus its independently reconstructable receipt."""

    text: str
    tokens: tuple[int, ...]
    receipt: dict[str, Any]


def build_incumbent_artifact(
    *,
    input_tokens: Sequence[int],
    output_tokens: Sequence[int],
    output_text: str,
    checkpoint_fingerprint: str,
    checkpoint_fingerprint_method: str,
    max_tokens: int,
    n_layers: int,
    termination: str,
) -> IncumbentArtifact:
    """Build an immutable artifact from one completed ordinary decode."""

    prompt = tuple(int(token) for token in input_tokens)
    output = tuple(int(token) for token in output_tokens)
    if not prompt or any(token < 0 for token in prompt):
        raise ValueError("incumbent input tokens are invalid")
    if not output or any(token < 0 for token in output):
        raise ValueError("incumbent output tokens are invalid")
    if not isinstance(output_text, str) or not output_text:
        raise ValueError("incumbent output text is empty")
    if not _is_sha256(checkpoint_fingerprint):
        raise ValueError("incumbent checkpoint fingerprint is invalid")
    if checkpoint_fingerprint_method != "sha256":
        raise ValueError("incumbent checkpoint fingerprint is not cryptographic")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
        raise ValueError("incumbent max_tokens is outside [1, 8192]")
    if len(output) > max_tokens:
        raise ValueError("incumbent output exceeds its decode budget")
    if type(n_layers) is not int or n_layers <= 0:
        raise ValueError("incumbent layer count is invalid")
    if termination not in _TERMINATIONS:
        raise ValueError("incumbent termination is invalid")

    decode_policy = {
        "contract": "final_answer_v1_stop_only",
        "max_tokens": max_tokens,
        "repetition_penalty": 1.0,
        "sentence_grace_tokens": 0,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    payload = {
        "schema": INCUMBENT_ARTIFACT_SCHEMA,
        "authority": INCUMBENT_AUTHORITY,
        "binding": {
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "checkpoint_fingerprint_method": checkpoint_fingerprint_method,
            "decode_policy_sha256": _sha(decode_policy),
            "input_tokens_sha256": input_token_sha256(prompt),
            "model_layer_count": n_layers,
        },
        "decode_policy": decode_policy,
        "output": {
            "termination": termination,
            "text_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            "token_count": len(output),
            "tokens_sha256": token_sequence_sha256(output),
        },
        "compute": {
            "generated_tokens": len(output),
            "prompt_tokens": len(prompt),
            "transformer_layer_apps": (len(prompt) + len(output)) * n_layers,
        },
    }
    receipt = {**payload, "receipt_sha256": _sha(payload)}
    return IncumbentArtifact(text=output_text, tokens=output, receipt=receipt)


def validate_incumbent_artifact(
    artifact: IncumbentArtifact,
    *,
    input_tokens: Sequence[int],
    checkpoint_fingerprint: str,
    checkpoint_fingerprint_method: str,
    max_tokens: int,
    n_layers: int,
    decode: Callable[[Sequence[int]], str] | None = None,
) -> IncumbentArtifact:
    """Reconstruct and bind an artifact to the current request and checkpoint."""

    if not isinstance(artifact, IncumbentArtifact):
        raise TypeError("incumbent artifact has the wrong type")
    termination = str((artifact.receipt.get("output") or {}).get("termination") or "")
    rebuilt = build_incumbent_artifact(
        input_tokens=input_tokens,
        output_tokens=artifact.tokens,
        output_text=artifact.text,
        checkpoint_fingerprint=checkpoint_fingerprint,
        checkpoint_fingerprint_method=checkpoint_fingerprint_method,
        max_tokens=max_tokens,
        n_layers=n_layers,
        termination=termination,
    )
    if rebuilt.receipt != artifact.receipt:
        raise ValueError("incumbent artifact receipt reconstruction differs")
    if decode is not None and decode(artifact.tokens) != artifact.text:
        raise ValueError("incumbent artifact token/text round trip differs")
    return artifact


def validate_incumbent_receipt(
    value: Mapping[str, Any],
    *,
    checkpoint_fingerprint: str | None = None,
    checkpoint_fingerprint_method: str | None = None,
) -> dict[str, Any]:
    """Validate the public artifact commitment without private output bytes."""

    if not isinstance(value, Mapping):
        raise ValueError("incumbent artifact receipt is missing")
    required = {
        "authority",
        "binding",
        "compute",
        "decode_policy",
        "output",
        "receipt_sha256",
        "schema",
    }
    if set(value) != required:
        raise ValueError("incumbent artifact receipt fields differ")
    if (
        value.get("schema") != INCUMBENT_ARTIFACT_SCHEMA
        or value.get("authority") != INCUMBENT_AUTHORITY
    ):
        raise ValueError("incumbent artifact receipt identity differs")
    binding = value.get("binding")
    policy = value.get("decode_policy")
    output = value.get("output")
    compute = value.get("compute")
    if not all(isinstance(item, Mapping) for item in (binding, policy, output, compute)):
        raise ValueError("incumbent artifact receipt sections are malformed")
    if set(binding) != {
        "checkpoint_fingerprint",
        "checkpoint_fingerprint_method",
        "decode_policy_sha256",
        "input_tokens_sha256",
        "model_layer_count",
    }:
        raise ValueError("incumbent artifact binding fields differ")
    if set(policy) != {
        "contract",
        "max_tokens",
        "repetition_penalty",
        "sentence_grace_tokens",
        "temperature",
        "top_p",
    }:
        raise ValueError("incumbent artifact decode policy fields differ")
    if (
        policy.get("contract") != "final_answer_v1_stop_only"
        or type(policy.get("max_tokens")) is not int
        or not 1 <= policy["max_tokens"] <= 8192
        or policy.get("repetition_penalty") != 1.0
        or policy.get("sentence_grace_tokens") != 0
        or policy.get("temperature") != 0.0
        or policy.get("top_p") != 1.0
        or binding.get("decode_policy_sha256") != _sha(dict(policy))
    ):
        raise ValueError("incumbent artifact decode policy differs")
    if (
        not _is_sha256(binding.get("checkpoint_fingerprint"))
        or binding.get("checkpoint_fingerprint_method") != "sha256"
        or not _is_sha256(binding.get("input_tokens_sha256"))
        or type(binding.get("model_layer_count")) is not int
        or binding["model_layer_count"] <= 0
    ):
        raise ValueError("incumbent artifact binding is invalid")
    if (
        checkpoint_fingerprint is not None
        and binding.get("checkpoint_fingerprint") != checkpoint_fingerprint
    ):
        raise ValueError("incumbent artifact checkpoint differs")
    if (
        checkpoint_fingerprint_method is not None
        and binding.get("checkpoint_fingerprint_method")
        != checkpoint_fingerprint_method
    ):
        raise ValueError("incumbent artifact checkpoint method differs")
    if set(output) != {"termination", "text_sha256", "token_count", "tokens_sha256"}:
        raise ValueError("incumbent artifact output fields differ")
    if (
        output.get("termination") not in _TERMINATIONS
        or not _is_sha256(output.get("text_sha256"))
        or not _is_sha256(output.get("tokens_sha256"))
        or type(output.get("token_count")) is not int
        or not 1 <= output["token_count"] <= policy["max_tokens"]
    ):
        raise ValueError("incumbent artifact output is invalid")
    if set(compute) != {"generated_tokens", "prompt_tokens", "transformer_layer_apps"}:
        raise ValueError("incumbent artifact compute fields differ")
    if (
        compute.get("generated_tokens") != output["token_count"]
        or type(compute.get("prompt_tokens")) is not int
        or compute["prompt_tokens"] <= 0
        or type(compute.get("transformer_layer_apps")) is not int
        or compute["transformer_layer_apps"]
        != (
            compute["prompt_tokens"] + compute["generated_tokens"]
        )
        * binding["model_layer_count"]
    ):
        raise ValueError("incumbent artifact compute accounting is invalid")
    payload = {key: value[key] for key in value if key != "receipt_sha256"}
    if value.get("receipt_sha256") != _sha(payload):
        raise ValueError("incumbent artifact receipt digest differs")
    return dict(value)


def incumbent_artifact_from_value(value: Mapping[str, Any]) -> IncumbentArtifact:
    """Rehydrate a journaled artifact without trusting its public receipt."""

    if not isinstance(value, Mapping) or set(value) != {"receipt", "text", "tokens"}:
        raise ValueError("journaled incumbent artifact is malformed")
    tokens = value.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("journaled incumbent tokens are malformed")
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("journaled incumbent receipt is malformed")
    return IncumbentArtifact(
        text=str(value.get("text") or ""),
        tokens=tuple(int(token) for token in tokens),
        receipt=dict(receipt),
    )


def incumbent_artifact_to_value(artifact: IncumbentArtifact) -> dict[str, Any]:
    return {
        "receipt": dict(artifact.receipt),
        "text": artifact.text,
        "tokens": list(artifact.tokens),
    }


__all__ = [
    "INCUMBENT_ARTIFACT_SCHEMA",
    "INCUMBENT_AUTHORITY",
    "IncumbentArtifact",
    "build_incumbent_artifact",
    "incumbent_artifact_from_value",
    "incumbent_artifact_to_value",
    "input_token_sha256",
    "token_sequence_sha256",
    "validate_incumbent_artifact",
    "validate_incumbent_receipt",
]
