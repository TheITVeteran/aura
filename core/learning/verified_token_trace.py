"""Real-tokenizer generation traces for verified recurrent training.

The contract deliberately treats prompt and generated tokenization differently.
Prompts must reproduce an exact encoding. Generated tokens must reproduce the
recorded response under full-sequence decoding, but are never decoded and then
re-encoded: BPE tokenizations are not necessarily canonical and individual
tokens need not contain a complete UTF-8 character.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Never, Protocol, runtime_checkable

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes

TOKENIZER_BUNDLE_SCHEMA = "aura.verified_tokenizer_bundle.v1"
VERIFIED_TOKEN_TRACE_SCHEMA = "aura.verified_token_trace.v1"

_MAX_TOKEN_ID = (1 << 63) - 1
_MAX_TOKEN_COUNT = 1_000_000
_MAX_TOKENIZER_FILE_BYTES = 4 * 1024 * 1024 * 1024
_SHA256_LENGTH = 64

_BUNDLE_KEYS = {
    "schema",
    "tokenizer_class",
    "tokenizer_files",
    "chat_template",
    "special_token_map",
    "special_token_map_sha256",
    "encode_options",
    "encode_options_sha256",
    "decode_options",
    "decode_options_sha256",
    "implementation_source_sha256",
    "bundle_sha256",
}
_FILE_KEYS = {"path", "size_bytes", "sha256"}
_CHAT_TEMPLATE_KEYS = {"present", "size_bytes", "utf8_sha256"}
_TRACE_KEYS = {
    "schema",
    "tokenizer_bundle",
    "tokenizer_bundle_sha256",
    "prompt",
    "generation",
    "trace_sha256",
}
_PROMPT_KEYS = {
    "text",
    "utf8_b64",
    "token_ids",
    "token_ids_sha256",
}
_GENERATION_KEYS = {
    "response_text",
    "response_utf8_b64",
    "token_ids",
    "token_ids_sha256",
    "behavior_logprobs",
    "behavior_logprobs_sha256",
    "streaming_deltas",
    "streaming_deltas_utf8_b64",
    "streaming_deltas_sha256",
}


class VerifiedTokenTraceError(ValueError):
    """Stable fail-closed error for invalid tokenizer evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTokenTraceError(code)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, role: str) -> str:
    if not _is_sha256(value):
        _fail(f"{role}_sha256_invalid")
    return value


def _canonical_clone(value: Any, *, role: str) -> Any:
    try:
        encoded = canonical_json_bytes(value)
        return json.loads(encoded)
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail(f"{role}_not_canonical_json")


def _require_text(value: Any, *, role: str) -> str:
    if not isinstance(value, str):
        _fail(f"{role}_invalid")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(f"{role}_utf8_invalid")
    return value


def _require_token_ids(
    value: Any,
    *,
    role: str,
    allow_empty: bool,
) -> list[int]:
    if not isinstance(value, (list, tuple)):
        _fail(f"{role}_invalid")
    if (not value and not allow_empty) or len(value) > _MAX_TOKEN_COUNT:
        _fail(f"{role}_invalid")
    result: list[int] = []
    for token in value:
        if type(token) is not int or not 0 <= token <= _MAX_TOKEN_ID:
            _fail(f"{role}_invalid")
        result.append(token)
    return result


def _canonical_logprob(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        _fail("behavior_logprob_invalid")
    if isinstance(value, float) and not math.isfinite(value):
        _fail("behavior_logprob_invalid")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        _fail("behavior_logprob_invalid")
    if not decimal.is_finite() or decimal > 0:
        _fail("behavior_logprob_invalid")
    if decimal == 0:
        return "0"
    return format(decimal.normalize(), "f")


def _require_logprobs(value: Any, *, token_count: int) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) != token_count:
        _fail("behavior_logprobs_invalid")
    return [_canonical_logprob(item) for item in value]


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_b64(value: Any, *, role: str) -> bytes:
    if not isinstance(value, str):
        _fail(f"{role}_invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        _fail(f"{role}_invalid")
    if _b64(decoded) != value:
        _fail(f"{role}_noncanonical")
    return decoded


def tokenizer_adapter_source_sha256() -> str:
    """Hash the production HF/Qwen adapter implementation, not its instance."""

    try:
        source = inspect.getsource(HuggingFaceTokenizerTraceAdapter).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise VerifiedTokenTraceError("tokenizer_adapter_source_unavailable") from exc
    return _sha256_bytes(source)


def _validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail("tokenizer_file_path_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail("tokenizer_file_path_invalid")
    normalized = path.as_posix()
    if normalized != value:
        _fail("tokenizer_file_path_noncanonical")
    return normalized


def _hash_open_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerifiedTokenTraceError("tokenizer_file_open_failed") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail("tokenizer_file_not_regular")
        if not 0 <= metadata.st_size <= _MAX_TOKENIZER_FILE_BYTES:
            _fail("tokenizer_file_size_invalid")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
        ):
            _fail("tokenizer_file_changed_during_hash")
        return metadata.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def tokenizer_file_bindings(
    root: str | Path,
    relative_paths: Sequence[str],
) -> list[dict[str, Any]]:
    """Hash a caller-selected tokenizer bundle without following symlinks."""

    try:
        root_path = Path(root).resolve(strict=True)
    except OSError as exc:
        raise VerifiedTokenTraceError("tokenizer_root_invalid") from exc
    if not root_path.is_dir() or not relative_paths:
        _fail("tokenizer_files_invalid")
    normalized = sorted(_validate_relative_path(value) for value in relative_paths)
    if len(normalized) != len(set(normalized)):
        _fail("tokenizer_file_path_duplicate")
    bindings: list[dict[str, Any]] = []
    for relative in normalized:
        candidate = root_path.joinpath(*PurePosixPath(relative).parts)
        try:
            parent = candidate.parent.resolve(strict=True)
            parent.relative_to(root_path)
        except (OSError, ValueError) as exc:
            raise VerifiedTokenTraceError("tokenizer_file_outside_root") from exc
        size, digest = _hash_open_file(candidate)
        bindings.append({"path": relative, "size_bytes": size, "sha256": digest})
    return bindings


def tokenizer_file_bindings_from_bytes(
    files: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    """In-memory counterpart used by sealed artifact stores and tests."""

    if not isinstance(files, Mapping) or not files:
        _fail("tokenizer_files_invalid")
    bindings: list[dict[str, Any]] = []
    for raw_path in sorted(files):
        path = _validate_relative_path(raw_path)
        content = files[raw_path]
        if not isinstance(content, bytes) or len(content) > _MAX_TOKENIZER_FILE_BYTES:
            _fail("tokenizer_file_content_invalid")
        bindings.append(
            {"path": path, "size_bytes": len(content), "sha256": _sha256_bytes(content)}
        )
    return bindings


def _validate_file_bindings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail("tokenizer_files_invalid")
    result: list[dict[str, Any]] = []
    previous = ""
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _FILE_KEYS:
            _fail("tokenizer_file_binding_invalid")
        path = _validate_relative_path(item.get("path"))
        if path <= previous:
            _fail("tokenizer_file_order_invalid")
        size = item.get("size_bytes")
        if type(size) is not int or not 0 <= size <= _MAX_TOKENIZER_FILE_BYTES:
            _fail("tokenizer_file_size_invalid")
        result.append(
            {
                "path": path,
                "size_bytes": size,
                "sha256": _require_sha256(item.get("sha256"), role="tokenizer_file"),
            }
        )
        previous = path
    return result


def build_tokenizer_bundle_identity(
    *,
    tokenizer_class: str,
    tokenizer_files: Sequence[Mapping[str, Any]],
    chat_template: str | None,
    special_token_map: Mapping[str, Any],
    encode_options: Mapping[str, Any],
    decode_options: Mapping[str, Any],
    implementation_source_sha256: str,
) -> dict[str, Any]:
    """Build the immutable identity consumed by generation-trace validation."""

    if not isinstance(tokenizer_class, str) or not tokenizer_class.strip():
        _fail("tokenizer_class_invalid")
    files = _validate_file_bindings([dict(item) for item in tokenizer_files])
    if chat_template is not None:
        chat_template = _require_text(chat_template, role="chat_template")
        template_bytes = chat_template.encode("utf-8")
        template = {
            "present": True,
            "size_bytes": len(template_bytes),
            "utf8_sha256": _sha256_bytes(template_bytes),
        }
    else:
        template = {
            "present": False,
            "size_bytes": 0,
            "utf8_sha256": _sha256_bytes(b""),
        }
    tokens = _canonical_clone(special_token_map, role="special_token_map")
    if not isinstance(tokens, dict):
        _fail("special_token_map_invalid")
    encode = _canonical_clone(encode_options, role="encode_options")
    decode = _canonical_clone(decode_options, role="decode_options")
    if not isinstance(encode, dict) or not isinstance(decode, dict):
        _fail("tokenizer_options_invalid")
    source_sha256 = _require_sha256(
        implementation_source_sha256,
        role="tokenizer_implementation_source",
    )
    body = {
        "schema": TOKENIZER_BUNDLE_SCHEMA,
        "tokenizer_class": tokenizer_class,
        "tokenizer_files": files,
        "chat_template": template,
        "special_token_map": tokens,
        "special_token_map_sha256": _sha256_json(tokens),
        "encode_options": encode,
        "encode_options_sha256": _sha256_json(encode),
        "decode_options": decode,
        "decode_options_sha256": _sha256_json(decode),
        "implementation_source_sha256": source_sha256,
    }
    return {**body, "bundle_sha256": _sha256_json(body)}


def validate_tokenizer_bundle_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BUNDLE_KEYS:
        _fail("tokenizer_bundle_schema_invalid")
    bundle = _canonical_clone(value, role="tokenizer_bundle")
    if bundle.get("schema") != TOKENIZER_BUNDLE_SCHEMA:
        _fail("tokenizer_bundle_schema_invalid")
    tokenizer_class = bundle.get("tokenizer_class")
    if not isinstance(tokenizer_class, str) or not tokenizer_class.strip():
        _fail("tokenizer_class_invalid")
    _validate_file_bindings(bundle.get("tokenizer_files"))
    template = bundle.get("chat_template")
    if not isinstance(template, Mapping) or set(template) != _CHAT_TEMPLATE_KEYS:
        _fail("chat_template_binding_invalid")
    present = template.get("present")
    size = template.get("size_bytes")
    if type(present) is not bool or type(size) is not int or size < 0:
        _fail("chat_template_binding_invalid")
    _require_sha256(template.get("utf8_sha256"), role="chat_template")
    if not present and (size != 0 or template["utf8_sha256"] != _sha256_bytes(b"")):
        _fail("chat_template_absent_binding_invalid")
    for role in ("special_token_map", "encode_options", "decode_options"):
        document = bundle.get(role)
        if not isinstance(document, dict):
            _fail(f"{role}_invalid")
        if bundle.get(f"{role}_sha256") != _sha256_json(document):
            _fail(f"{role}_digest_mismatch")
    _require_sha256(
        bundle.get("implementation_source_sha256"),
        role="tokenizer_implementation_source",
    )
    observed = bundle.get("bundle_sha256")
    body = {key: bundle[key] for key in bundle if key != "bundle_sha256"}
    if observed != _sha256_json(body):
        _fail("tokenizer_bundle_digest_mismatch")
    return bundle


@runtime_checkable
class TokenizerTraceAdapter(Protocol):
    """Minimal tokenizer-only surface needed to produce and replay a trace."""

    @property
    def bundle_identity(self) -> Mapping[str, Any]: ...

    def encode_prompt(self, prompt_text: str) -> Sequence[int]: ...

    def decode_output(self, token_ids: Sequence[int]) -> str: ...

    def stream_decode_deltas(self, token_ids: Sequence[int]) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenizerTraceAdapter:
    """Tokenizer-only adapter for HF/Qwen-compatible tokenizers.

    Prefix-stable streaming delays text that a tokenizer may revise at a later
    byte boundary. This produces an empty delta for an incomplete UTF-8 token
    and emits it only after the complete character is independently decodable.
    No model weights are loaded or required.
    """

    tokenizer: Any
    _bundle_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        bundle = validate_tokenizer_bundle_identity(self._bundle_identity)
        if bundle["implementation_source_sha256"] != tokenizer_adapter_source_sha256():
            _fail("tokenizer_adapter_source_mismatch")
        if not callable(getattr(self.tokenizer, "encode", None)):
            _fail("tokenizer_encode_missing")
        if not callable(getattr(self.tokenizer, "decode", None)):
            _fail("tokenizer_decode_missing")
        object.__setattr__(self, "_bundle_identity", bundle)

    @property
    def bundle_identity(self) -> Mapping[str, Any]:
        return _canonical_clone(self._bundle_identity, role="tokenizer_bundle")

    def encode_prompt(self, prompt_text: str) -> Sequence[int]:
        options = dict(self._bundle_identity["encode_options"])
        return self.tokenizer.encode(prompt_text, **options)

    def decode_output(self, token_ids: Sequence[int]) -> str:
        options = dict(self._bundle_identity["decode_options"])
        return self.tokenizer.decode(list(token_ids), **options)

    def stream_decode_deltas(self, token_ids: Sequence[int]) -> Sequence[str]:
        tokens = list(token_ids)
        full = _require_text(self.decode_output(tokens), role="tokenizer_full_decode")
        emitted = ""
        deltas: list[str] = []
        for index in range(len(tokens)):
            prefix = _require_text(
                self.decode_output(tokens[: index + 1]),
                role="tokenizer_prefix_decode",
            )
            stable_length = 0
            for left, right in zip(prefix, full, strict=False):
                if left != right:
                    break
                stable_length += 1
            stable = full[:stable_length]
            if not stable.startswith(emitted):
                _fail("tokenizer_prefix_decode_retracted_stable_text")
            deltas.append(stable[len(emitted) :])
            emitted = stable
        if emitted != full:
            _fail("tokenizer_final_prefix_decode_mismatch")
        return deltas


def _adapter_bundle(adapter: TokenizerTraceAdapter) -> dict[str, Any]:
    try:
        return validate_tokenizer_bundle_identity(adapter.bundle_identity)
    except AttributeError as exc:
        raise VerifiedTokenTraceError("tokenizer_adapter_bundle_missing") from exc


def _call_adapter(method: Any, *arguments: Any, role: str) -> Any:
    try:
        return method(*arguments)
    except VerifiedTokenTraceError:
        raise
    except Exception as exc:  # noqa: BLE001 - tokenizer is an evidence boundary
        raise VerifiedTokenTraceError(f"{role}_execution_failed") from exc


def _validate_trace_observations(
    *,
    adapter: TokenizerTraceAdapter,
    prompt_text: str,
    prompt_token_ids: Any,
    output_token_ids: Any,
    behavior_logprobs: Any,
    response_text: str,
    streaming_deltas: Any,
) -> tuple[list[int], list[int], list[str], list[str]]:
    prompt = _require_text(prompt_text, role="prompt_text")
    response = _require_text(response_text, role="response_text")
    prompt_tokens = _require_token_ids(
        prompt_token_ids,
        role="prompt_token_ids",
        allow_empty=False,
    )
    output_tokens = _require_token_ids(
        output_token_ids,
        role="output_token_ids",
        allow_empty=False,
    )
    logprobs = _require_logprobs(behavior_logprobs, token_count=len(output_tokens))
    if not isinstance(streaming_deltas, (list, tuple)) or len(streaming_deltas) != len(
        output_tokens
    ):
        _fail("streaming_deltas_invalid")
    deltas = [
        _require_text(delta, role="streaming_delta") for delta in streaming_deltas
    ]

    encoded_prompt = _call_adapter(
        adapter.encode_prompt,
        prompt,
        role="tokenizer_prompt_encode",
    )
    independently_encoded = _require_token_ids(
        encoded_prompt,
        role="tokenizer_prompt_encoding",
        allow_empty=False,
    )
    if independently_encoded != prompt_tokens:
        _fail("prompt_encoding_mismatch")

    decoded_output = _call_adapter(
        adapter.decode_output,
        output_tokens,
        role="tokenizer_full_decode",
    )
    if _require_text(decoded_output, role="tokenizer_full_decode") != response:
        _fail("full_sequence_decode_mismatch")

    independently_streamed = _call_adapter(
        adapter.stream_decode_deltas,
        output_tokens,
        role="tokenizer_stream_decode",
    )
    if not isinstance(independently_streamed, Sequence) or isinstance(
        independently_streamed, (str, bytes)
    ):
        _fail("tokenizer_streaming_deltas_invalid")
    replayed = [
        _require_text(delta, role="tokenizer_streaming_delta")
        for delta in independently_streamed
    ]
    if len(replayed) != len(output_tokens):
        _fail("tokenizer_streaming_delta_count_mismatch")
    if replayed != deltas:
        _fail("streaming_deltas_mismatch")
    if "".join(deltas) != response:
        _fail("streaming_deltas_response_mismatch")
    return prompt_tokens, output_tokens, logprobs, deltas


def build_verified_token_trace(
    *,
    adapter: TokenizerTraceAdapter,
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    behavior_logprobs: Sequence[str | int | float | Decimal],
    response_text: str,
    streaming_deltas: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build and independently replay a sealed tokenizer generation trace."""

    bundle = _adapter_bundle(adapter)
    output_tokens = _require_token_ids(
        output_token_ids,
        role="output_token_ids",
        allow_empty=False,
    )
    if streaming_deltas is None:
        streaming_deltas = _call_adapter(
            adapter.stream_decode_deltas,
            output_tokens,
            role="tokenizer_stream_decode",
        )
    prompt_tokens, output_tokens, logprobs, deltas = _validate_trace_observations(
        adapter=adapter,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_tokens,
        behavior_logprobs=behavior_logprobs,
        response_text=response_text,
        streaming_deltas=streaming_deltas,
    )
    prompt_bytes = prompt_text.encode("utf-8")
    response_bytes = response_text.encode("utf-8")
    delta_bytes = [delta.encode("utf-8") for delta in deltas]
    prompt_document = {
        "text": prompt_text,
        "utf8_b64": _b64(prompt_bytes),
        "token_ids": prompt_tokens,
        "token_ids_sha256": _sha256_json(prompt_tokens),
    }
    generation_document = {
        "response_text": response_text,
        "response_utf8_b64": _b64(response_bytes),
        "token_ids": output_tokens,
        "token_ids_sha256": _sha256_json(output_tokens),
        "behavior_logprobs": logprobs,
        "behavior_logprobs_sha256": _sha256_json(logprobs),
        "streaming_deltas": deltas,
        "streaming_deltas_utf8_b64": [_b64(delta) for delta in delta_bytes],
        "streaming_deltas_sha256": _sha256_json(
            [_b64(delta) for delta in delta_bytes]
        ),
    }
    body = {
        "schema": VERIFIED_TOKEN_TRACE_SCHEMA,
        "tokenizer_bundle": bundle,
        "tokenizer_bundle_sha256": bundle["bundle_sha256"],
        "prompt": prompt_document,
        "generation": generation_document,
    }
    return {**body, "trace_sha256": _sha256_json(body)}


def validate_verified_token_trace(
    trace: Any,
    *,
    adapter: TokenizerTraceAdapter,
    expected_trace_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay a trace against the exact bound tokenizer-only adapter."""

    if not isinstance(trace, Mapping) or set(trace) != _TRACE_KEYS:
        _fail("verified_token_trace_schema_invalid")
    document = _canonical_clone(trace, role="verified_token_trace")
    if document.get("schema") != VERIFIED_TOKEN_TRACE_SCHEMA:
        _fail("verified_token_trace_schema_invalid")
    body = {key: document[key] for key in document if key != "trace_sha256"}
    observed_digest = _require_sha256(
        document.get("trace_sha256"),
        role="verified_token_trace",
    )
    if observed_digest != _sha256_json(body):
        _fail("verified_token_trace_digest_mismatch")
    if expected_trace_sha256 is not None:
        expected = _require_sha256(expected_trace_sha256, role="expected_token_trace")
        if observed_digest != expected:
            _fail("verified_token_trace_commitment_mismatch")

    bundle = validate_tokenizer_bundle_identity(document.get("tokenizer_bundle"))
    if document.get("tokenizer_bundle_sha256") != bundle["bundle_sha256"]:
        _fail("tokenizer_bundle_reference_mismatch")
    adapter_bundle = _adapter_bundle(adapter)
    if canonical_json_bytes(bundle) != canonical_json_bytes(adapter_bundle):
        _fail("tokenizer_bundle_substitution")

    prompt = document.get("prompt")
    generation = document.get("generation")
    if not isinstance(prompt, Mapping) or set(prompt) != _PROMPT_KEYS:
        _fail("verified_token_prompt_schema_invalid")
    if not isinstance(generation, Mapping) or set(generation) != _GENERATION_KEYS:
        _fail("verified_token_generation_schema_invalid")
    prompt_text = _require_text(prompt.get("text"), role="prompt_text")
    response_text = _require_text(generation.get("response_text"), role="response_text")
    if _decode_b64(prompt.get("utf8_b64"), role="prompt_utf8_b64") != prompt_text.encode(
        "utf-8"
    ):
        _fail("prompt_bytes_text_mismatch")
    if _decode_b64(
        generation.get("response_utf8_b64"),
        role="response_utf8_b64",
    ) != response_text.encode("utf-8"):
        _fail("response_bytes_text_mismatch")

    prompt_tokens = _require_token_ids(
        prompt.get("token_ids"),
        role="prompt_token_ids",
        allow_empty=False,
    )
    output_tokens = _require_token_ids(
        generation.get("token_ids"),
        role="output_token_ids",
        allow_empty=False,
    )
    if prompt.get("token_ids_sha256") != _sha256_json(prompt_tokens):
        _fail("prompt_token_ids_digest_mismatch")
    if generation.get("token_ids_sha256") != _sha256_json(output_tokens):
        _fail("output_token_ids_digest_mismatch")
    logprobs = _require_logprobs(
        generation.get("behavior_logprobs"),
        token_count=len(output_tokens),
    )
    if generation.get("behavior_logprobs_sha256") != _sha256_json(logprobs):
        _fail("behavior_logprobs_digest_mismatch")
    deltas = generation.get("streaming_deltas")
    delta_b64 = generation.get("streaming_deltas_utf8_b64")
    if not isinstance(deltas, list) or not isinstance(delta_b64, list):
        _fail("streaming_deltas_invalid")
    if len(deltas) != len(output_tokens) or len(delta_b64) != len(deltas):
        _fail("streaming_deltas_invalid")
    normalized_deltas = [
        _require_text(delta, role="streaming_delta") for delta in deltas
    ]
    for index, (delta, encoded) in enumerate(zip(normalized_deltas, delta_b64, strict=True)):
        if _decode_b64(encoded, role=f"streaming_delta_{index}_utf8_b64") != delta.encode(
            "utf-8"
        ):
            _fail("streaming_delta_bytes_text_mismatch")
    if generation.get("streaming_deltas_sha256") != _sha256_json(delta_b64):
        _fail("streaming_deltas_digest_mismatch")

    _validate_trace_observations(
        adapter=adapter,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_tokens,
        output_token_ids=output_tokens,
        behavior_logprobs=logprobs,
        response_text=response_text,
        streaming_deltas=normalized_deltas,
    )
    return document


__all__ = [
    "HuggingFaceTokenizerTraceAdapter",
    "TOKENIZER_BUNDLE_SCHEMA",
    "TokenizerTraceAdapter",
    "VERIFIED_TOKEN_TRACE_SCHEMA",
    "VerifiedTokenTraceError",
    "build_tokenizer_bundle_identity",
    "build_verified_token_trace",
    "tokenizer_adapter_source_sha256",
    "tokenizer_file_bindings",
    "tokenizer_file_bindings_from_bytes",
    "validate_tokenizer_bundle_identity",
    "validate_verified_token_trace",
]
