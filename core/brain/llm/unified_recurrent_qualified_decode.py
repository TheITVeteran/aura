"""Domain-qualified answer decoding for certified recurrent controller tissue."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

REQUEST_SCHEMA: Final = "aura.unified_intrinsic.qualified_decode_request.v1"
RESULT_SCHEMA: Final = "aura.unified_intrinsic.qualified_decode_result.v1"
CANARY_AUTHORITY_SCHEMA: Final = "aura.unified_intrinsic.qualified_canary_request.v1"
MAX_PUBLIC_TOKENS: Final = 16_384
MAX_ANSWER_TOKENS: Final = 32
_FAMILIES: Final = frozenset({"khop", "modular", "register_trace"})
_REQUEST_SCOPED_CANARY_MODES: Final = frozenset(
    {"qualified_canary_only", "qualified_typed_pending"}
)
_HEX: Final = frozenset("0123456789abcdef")
_PARSED_FIELDS: Final = {
    "khop": frozenset({"node"}),
    "modular": frozenset({"residue"}),
    "register_trace": frozenset({"r0", "r1", "r2"}),
}
_RESULT_FIELDS: Final = {
    "schema",
    "request_sha256",
    "package_id",
    "controller_sha256",
    "family",
    "task_depth",
    "generated_token_ids",
    "parsed_values",
    "latency_ms",
    "grammar_valid",
    "output_exposed",
    "serving_authority",
    "authority_source",
    "qualified_activation_sha256",
    "result_sha256",
}


class UnifiedRecurrentQualifiedDecodeError(RuntimeError):
    """A request or recurrent answer escaped its certified typed domain."""


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def seal_qualified_decode_request(
    public_token_ids: Sequence[int],
    *,
    package_id: str,
    controller_sha256: str,
    family: str,
    task_depth: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Seal one answer-blind typed request before it crosses worker IPC."""

    body = {
        "schema": REQUEST_SCHEMA,
        "package_id": package_id,
        "controller_sha256": controller_sha256,
        "family": family,
        "task_depth": task_depth,
        "max_tokens": max_tokens,
        "public_token_ids": list(public_token_ids),
    }
    request = {**body, "request_sha256": _sha(body)}
    errors = qualified_decode_request_errors(request)
    if errors:
        raise ValueError(",".join(errors))
    return request


def qualified_decode_request_errors(value: Any) -> list[str]:
    fields = {
        "schema",
        "package_id",
        "controller_sha256",
        "family",
        "task_depth",
        "max_tokens",
        "public_token_ids",
        "request_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        return ["qualified_decode_request_fields_differ"]
    body = {key: item for key, item in value.items() if key != "request_sha256"}
    tokens = value.get("public_token_ids")
    errors: list[str] = []
    if value.get("schema") != REQUEST_SCHEMA or value.get("request_sha256") != _sha(body):
        errors.append("qualified_decode_request_identity_differs")
    if (
        not isinstance(value.get("package_id"), str)
        or not value["package_id"]
        or not isinstance(value.get("controller_sha256"), str)
        or len(value["controller_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["controller_sha256"])
    ):
        errors.append("qualified_decode_package_identity_invalid")
    if value.get("family") not in _FAMILIES:
        errors.append("qualified_decode_family_invalid")
    if type(value.get("task_depth")) is not int or value["task_depth"] < 1:
        errors.append("qualified_decode_task_depth_invalid")
    if type(value.get("max_tokens")) is not int or not 1 <= value["max_tokens"] <= MAX_ANSWER_TOKENS:
        errors.append("qualified_decode_token_budget_invalid")
    if (
        not isinstance(tokens, list)
        or not 1 <= len(tokens) <= MAX_PUBLIC_TOKENS
        or any(type(token_id) is not int or token_id < 0 for token_id in tokens)
    ):
        errors.append("qualified_decode_public_tokens_invalid")
    return errors


def seal_qualified_canary_request_authority(
    *,
    activation_sha256: str,
    battery_sha256: str,
    case_index: int,
    request_sha256: str,
    nonce: str,
    issued_at_unix: float,
    expires_at_unix: float,
) -> dict[str, Any]:
    """Bind provisional authority to one sealed battery case and short lease."""

    body = {
        "schema": CANARY_AUTHORITY_SCHEMA,
        "activation_sha256": activation_sha256,
        "battery_sha256": battery_sha256,
        "case_index": case_index,
        "request_sha256": request_sha256,
        "nonce": nonce,
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
    }
    result = {**body, "authority_sha256": _sha(body)}
    errors = qualified_canary_request_authority_errors(result)
    if errors:
        raise ValueError(",".join(errors))
    return result


def qualified_canary_request_authority_errors(value: Any) -> list[str]:
    fields = {
        "schema",
        "activation_sha256",
        "battery_sha256",
        "case_index",
        "request_sha256",
        "nonce",
        "issued_at_unix",
        "expires_at_unix",
        "authority_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        return ["qualified_canary_authority_fields_differ"]
    body = {key: item for key, item in value.items() if key != "authority_sha256"}
    issued = value.get("issued_at_unix")
    expires = value.get("expires_at_unix")
    if (
        value.get("schema") != CANARY_AUTHORITY_SCHEMA
        or value.get("authority_sha256") != _sha(body)
        or not _is_sha(value.get("activation_sha256"))
        or not _is_sha(value.get("battery_sha256"))
        or not _is_sha(value.get("request_sha256"))
        or type(value.get("case_index")) is not int
        or value["case_index"] < 0
        or not _is_sha(value.get("nonce"))
        or not isinstance(issued, (int, float))
        or isinstance(issued, bool)
        or not isinstance(expires, (int, float))
        or isinstance(expires, bool)
        or not float(issued) < float(expires) <= float(issued) + 3600.0
    ):
        return ["qualified_canary_authority_invalid"]
    return []


def qualified_canary_authority_matches(
    authority: Mapping[str, Any],
    *,
    activation: Mapping[str, Any],
    request: Mapping[str, Any],
    battery: Mapping[str, Any],
    now_unix: float | None = None,
) -> bool:
    """Prove a provisional request is one exact case from the loaded package."""

    if qualified_canary_request_authority_errors(authority):
        return False
    now = time.time() if now_unix is None else float(now_unix)
    cases = battery.get("cases")
    index = authority.get("case_index")
    if (
        not isinstance(cases, list)
        or type(index) is not int
        or not 0 <= index < len(cases)
        or not float(authority["issued_at_unix"]) - 5.0 <= now
        or now > float(authority["expires_at_unix"])
        or authority.get("activation_sha256") != activation.get("activation_sha256")
        or authority.get("battery_sha256") != battery.get("battery_sha256")
        or authority.get("request_sha256") != request.get("request_sha256")
    ):
        return False
    case = cases[index]
    return bool(
        isinstance(case, Mapping)
        and case.get("public_token_ids") == request.get("public_token_ids")
        and case.get("family") == request.get("family")
        and case.get("task_depth") == request.get("task_depth")
        and case.get("max_tokens") == request.get("max_tokens")
    )


def _occurrences(row: Sequence[int], pattern: tuple[int, ...], *, start: int = 0) -> tuple[int, ...]:
    width = len(pattern)
    return tuple(
        index
        for index in range(start, len(row) - width + 1)
        if tuple(row[index : index + width]) == pattern
    )


def _first(row: Sequence[int], pattern: tuple[int, ...], *, start: int = 0) -> int:
    matches = _occurrences(row, pattern, start=start)
    if not matches:
        raise UnifiedRecurrentQualifiedDecodeError("typed public program boundary is absent")
    return matches[0]


def _masked_values(
    values: Sequence[int],
    masks: Sequence[bool],
    start: int,
    stop: int,
) -> tuple[int, ...]:
    return tuple(
        value
        for index, (value, mask) in enumerate(zip(values, masks, strict=True))
        if start <= index < stop and mask
    )


def classify_public_program(
    public_tokens: Sequence[int],
    *,
    answer_contract: Any,
    literal_contract: Any,
    opcode_contract: Any,
) -> tuple[str, int]:
    """Independently recover family and operation depth from public grammar."""

    row = tuple(public_tokens)
    family = answer_contract.family(row)
    if family not in _FAMILIES:
        raise UnifiedRecurrentQualifiedDecodeError("typed public program family is unsupported")
    literal_rows, literal_mask_rows = literal_contract.observe([row])
    opcode_rows, opcode_mask_rows = opcode_contract.observe([row])
    literals, literal_masks = literal_rows[0], literal_mask_rows[0]
    opcodes, opcode_masks = opcode_rows[0], opcode_mask_rows[0]
    contexts = dict(opcode_contract.contexts)
    if family == "khop":
        graph = _first(row, contexts["graph"])
        boundary = _first(row, contexts["graph_edges_end"], start=graph)
        trailing = _masked_values(literals, literal_masks, boundary, len(row))
        if len(trailing) < 2:
            raise UnifiedRecurrentQualifiedDecodeError("graph public depth is absent")
        depth = trailing[1]
    elif family == "modular":
        start = _first(row, contexts["modular_start"]) + len(contexts["modular_start"])
        stop = _first(row, contexts["modular_end"], start=start)
        operations = _masked_values(opcodes, opcode_masks, start, stop)
        operands = _masked_values(literals, literal_masks, start, stop)
        if not operations or len(operations) != len(operands):
            raise UnifiedRecurrentQualifiedDecodeError("modular public program is malformed")
        depth = len(operations)
    else:
        register = _first(row, contexts["register"])
        start = _first(row, contexts["register_ops_start"], start=register) + len(
            contexts["register_ops_start"]
        )
        stop = _first(row, contexts["register_ops_end"], start=start)
        fields = _masked_values(literals, literal_masks, start, stop)
        if not fields or len(fields) % 6:
            raise UnifiedRecurrentQualifiedDecodeError("register public program is malformed")
        depth = len(fields) // 6
    if depth < 1:
        raise UnifiedRecurrentQualifiedDecodeError("typed public program depth is invalid")
    return family, depth


def _consume(
    generated: tuple[int, ...],
    cursor: int,
    pattern: tuple[int, ...],
) -> int:
    if generated[cursor : cursor + len(pattern)] != pattern:
        raise UnifiedRecurrentQualifiedDecodeError("qualified answer syntax differs")
    return cursor + len(pattern)


def _number(
    generated: tuple[int, ...],
    cursor: int,
    *,
    stop_pattern: tuple[int, ...],
    digit_ids: tuple[int, ...],
) -> tuple[int, int]:
    stop = None
    for index in range(cursor + 1, len(generated) + 1):
        if generated[index : index + len(stop_pattern)] == stop_pattern:
            stop = index
            break
    if stop is None:
        raise UnifiedRecurrentQualifiedDecodeError("qualified answer field is unterminated")
    token_to_digit = {token_id: digit for digit, token_id in enumerate(digit_ids)}
    digits = generated[cursor:stop]
    if not 1 <= len(digits) <= 2 or any(token not in token_to_digit for token in digits):
        raise UnifiedRecurrentQualifiedDecodeError("qualified answer digit field differs")
    text = "".join(str(token_to_digit[token]) for token in digits)
    if len(text) > 1 and text.startswith("0"):
        raise UnifiedRecurrentQualifiedDecodeError("qualified answer integer is noncanonical")
    value = int(text)
    if not 0 <= value <= 32:
        raise UnifiedRecurrentQualifiedDecodeError("qualified answer integer exceeds domain")
    return value, stop


def validate_qualified_answer(
    public_tokens: Sequence[int],
    generated_tokens: Sequence[int],
    *,
    answer_contract: Any,
) -> dict[str, int]:
    """Validate the exact package-bound JSON token grammar without a target answer."""

    generated = tuple(generated_tokens)
    if not generated or generated[-1] != answer_contract.eos_token_id:
        raise UnifiedRecurrentQualifiedDecodeError("qualified answer did not terminate at EOS")
    family = answer_contract.family(public_tokens)
    syntax = dict(answer_contract.syntax)
    cursor = 0
    values: dict[str, int] = {}
    if family == "khop":
        cursor = _consume(generated, cursor, syntax["khop"])
        values["node"], cursor = _number(
            generated,
            cursor,
            stop_pattern=syntax["close"],
            digit_ids=answer_contract.digit_token_ids,
        )
    elif family == "modular":
        cursor = _consume(generated, cursor, syntax["modular"])
        values["residue"], cursor = _number(
            generated,
            cursor,
            stop_pattern=syntax["close"],
            digit_ids=answer_contract.digit_token_ids,
        )
    elif family == "register_trace":
        cursor = _consume(generated, cursor, syntax["register_head"])
        for key, boundary in (
            ("r0", syntax["register_mid_r1"]),
            ("r1", syntax["register_mid_r2"]),
            ("r2", syntax["close"]),
        ):
            values[key], cursor = _number(
                generated,
                cursor,
                stop_pattern=boundary,
                digit_ids=answer_contract.digit_token_ids,
            )
            cursor = _consume(generated, cursor, boundary)
        if cursor != len(generated) - 1:
            raise UnifiedRecurrentQualifiedDecodeError("qualified register answer has trailing tokens")
        return values
    else:
        raise UnifiedRecurrentQualifiedDecodeError("qualified answer family is unsupported")
    cursor = _consume(generated, cursor, syntax["close"])
    if cursor != len(generated) - 1:
        raise UnifiedRecurrentQualifiedDecodeError("qualified answer has trailing tokens")
    return values


def qualified_decode_result_errors(
    value: Any,
    *,
    expected_request_sha256: str = "",
    expected_activation_sha256: str = "",
    expected_package_id: str = "",
    expected_controller_sha256: str = "",
    expected_family: str = "",
    expected_task_depth: int | None = None,
    expected_canary_authority: bool = False,
) -> list[str]:
    """Validate a decoded result and its optional serving authority binding."""

    if not isinstance(value, Mapping) or set(value) != _RESULT_FIELDS:
        return ["qualified_decode_result_fields_differ"]
    body = {key: item for key, item in value.items() if key != "result_sha256"}
    errors: list[str] = []
    generated = value.get("generated_token_ids")
    parsed = value.get("parsed_values")
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("result_sha256") != _sha(body)
        or not _is_sha(value.get("request_sha256"))
        or not isinstance(value.get("package_id"), str)
        or not 1 <= len(value["package_id"]) <= 120
        or not _is_sha(value.get("controller_sha256"))
        or value.get("family") not in _FAMILIES
        or type(value.get("task_depth")) is not int
        or value["task_depth"] < 1
        or not isinstance(generated, list)
        or not 1 <= len(generated) <= MAX_ANSWER_TOKENS
        or any(type(token_id) is not int or token_id < 0 for token_id in generated)
        or not isinstance(parsed, Mapping)
        or not parsed
        or set(parsed) != _PARSED_FIELDS.get(value.get("family"), frozenset())
        or any(
            not isinstance(key, str)
            or not key
            or type(item) is not int
            or not 0 <= item <= 32
            for key, item in parsed.items()
        )
        or type(value.get("latency_ms")) is not int
        or value["latency_ms"] < 0
        or value.get("grammar_valid") is not True
        or value.get("output_exposed") is not True
    ):
        errors.append("qualified_decode_result_invalid")
    if expected_request_sha256 and value.get("request_sha256") != expected_request_sha256:
        errors.append("qualified_decode_result_request_differs")
    if any(
        (
            expected_package_id
            and value.get("package_id") != expected_package_id,
            expected_controller_sha256
            and value.get("controller_sha256") != expected_controller_sha256,
            expected_family and value.get("family") != expected_family,
            expected_task_depth is not None
            and value.get("task_depth") != expected_task_depth,
        )
    ):
        errors.append("qualified_decode_result_domain_differs")
    authorized = value.get("serving_authority") is True
    activation_sha = value.get("qualified_activation_sha256")
    authority_source = value.get("authority_source")
    if authorized:
        if (
            not isinstance(activation_sha, str)
            or len(activation_sha) != 64
            or any(character not in "0123456789abcdef" for character in activation_sha)
            or authority_source != "qualified_activation"
            or (expected_activation_sha256 and activation_sha != expected_activation_sha256)
        ):
            errors.append("qualified_decode_result_authority_invalid")
    elif authority_source == "qualified_canary_request":
        if (
            value.get("serving_authority") is not False
            or not _is_sha(activation_sha)
            or not expected_canary_authority
            or (
                expected_activation_sha256
                and activation_sha != expected_activation_sha256
            )
        ):
            errors.append("qualified_decode_result_canary_authority_invalid")
    elif (
        value.get("serving_authority") is not False
        or activation_sha != ""
        or authority_source != "qualified_activation_required_by_caller"
        or expected_activation_sha256
        or expected_canary_authority
    ):
        errors.append("qualified_decode_result_inactive_authority_invalid")
    return errors


def authorize_qualified_decode_result(
    result: Mapping[str, Any],
    activation: Mapping[str, Any],
    *,
    canary_only: bool = False,
) -> dict[str, Any]:
    """Bind one typed result to an independently admitted activation."""

    errors = qualified_decode_result_errors(result)
    if errors:
        raise UnifiedRecurrentQualifiedDecodeError(",".join(errors))
    from core.brain.llm.unified_recurrent_qualified_activation import (
        qualified_activation_errors,
    )

    activation_errors = qualified_activation_errors(activation)
    if activation_errors:
        raise UnifiedRecurrentQualifiedDecodeError(",".join(activation_errors))
    if (
        (
            canary_only
            and (
                activation.get("mode") not in _REQUEST_SCOPED_CANARY_MODES
                or activation.get("serving_authority") is not False
            )
        )
        or (
            not canary_only
            and (
                activation.get("mode") != "qualified_typed_only"
                or activation.get("serving_authority") is not True
            )
        )
        or result.get("package_id") != activation.get("package_id")
        or result.get("controller_sha256") != activation.get("controller_sha256")
        or result.get("family") not in set(activation.get("families") or ())
        or result.get("task_depth") not in set(activation.get("task_depths") or ())
    ):
        raise UnifiedRecurrentQualifiedDecodeError(
            "qualified decode activation identity differs"
        )
    body = {
        key: item
        for key, item in result.items()
        if key not in {"result_sha256", "serving_authority", "authority_source", "qualified_activation_sha256"}
    }
    body.update(
        {
            "serving_authority": not canary_only,
            "authority_source": (
                "qualified_canary_request"
                if canary_only
                else "qualified_activation"
            ),
            "qualified_activation_sha256": activation["activation_sha256"],
        }
    )
    authorized = {**body, "result_sha256": _sha(body)}
    errors = qualified_decode_result_errors(
        authorized,
        expected_request_sha256=str(result.get("request_sha256") or ""),
        expected_activation_sha256=str(activation.get("activation_sha256") or ""),
        expected_canary_authority=canary_only,
    )
    if errors:
        raise UnifiedRecurrentQualifiedDecodeError(",".join(errors))
    return authorized


def run_qualified_decode(
    loaded: Any,
    model: Any,
    request: Mapping[str, Any],
    *,
    cancel_check: Callable[[], bool] | None = None,
    activity: Callable[[], None] | None = None,
    progress: Callable[[Mapping[str, int | str]], None] | None = None,
) -> dict[str, Any]:
    """Decode one certified typed answer with no expected answer in context."""

    errors = qualified_decode_request_errors(request)
    if errors:
        raise UnifiedRecurrentQualifiedDecodeError(",".join(errors))
    receipt = loaded.receipt
    if (
        request["package_id"] != receipt["package_id"]
        or request["controller_sha256"] != receipt["controller_sha256"]
    ):
        raise UnifiedRecurrentQualifiedDecodeError("qualified decode package identity differs")
    literal_contract = getattr(loaded, "literal_contract", None)
    opcode_contract = getattr(loaded, "opcode_contract", None)
    if literal_contract is None or opcode_contract is None:
        raise UnifiedRecurrentQualifiedDecodeError("qualified decode grammar contracts unavailable")
    public_tokens = tuple(request["public_token_ids"])
    vocabulary = int(model.model.embed_tokens.weight.shape[0])
    if any(token_id >= vocabulary for token_id in public_tokens):
        raise UnifiedRecurrentQualifiedDecodeError(
            "qualified decode token exceeds model vocabulary"
        )
    family, task_depth = classify_public_program(
        public_tokens,
        answer_contract=loaded.answer_contract,
        literal_contract=literal_contract,
        opcode_contract=opcode_contract,
    )
    if (
        family != request["family"]
        or task_depth != request["task_depth"]
        or family not in set(receipt["families"])
        or task_depth not in set(receipt["task_depths"])
    ):
        raise UnifiedRecurrentQualifiedDecodeError("qualified decode domain differs")

    generated, _stopped, latency_ms = loaded.decode_recurrent_tokens(
        model,
        public_tokens,
        max_tokens=int(request["max_tokens"]),
        cancel_check=cancel_check,
        activity=activity,
        progress=progress,
    )
    values = validate_qualified_answer(
        public_tokens,
        generated,
        answer_contract=loaded.answer_contract,
    )
    body = {
        "schema": RESULT_SCHEMA,
        "request_sha256": request["request_sha256"],
        "package_id": receipt["package_id"],
        "controller_sha256": receipt["controller_sha256"],
        "family": family,
        "task_depth": task_depth,
        "generated_token_ids": list(generated),
        "parsed_values": values,
        "latency_ms": latency_ms,
        "grammar_valid": True,
        "output_exposed": True,
        "serving_authority": False,
        "authority_source": "qualified_activation_required_by_caller",
        "qualified_activation_sha256": "",
    }
    result = {**body, "result_sha256": _sha(body)}
    errors = qualified_decode_result_errors(result)
    if errors:
        raise UnifiedRecurrentQualifiedDecodeError(",".join(errors))
    return result


__all__ = [
    "MAX_ANSWER_TOKENS",
    "CANARY_AUTHORITY_SCHEMA",
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "UnifiedRecurrentQualifiedDecodeError",
    "authorize_qualified_decode_result",
    "classify_public_program",
    "qualified_decode_request_errors",
    "qualified_canary_authority_matches",
    "qualified_canary_request_authority_errors",
    "qualified_decode_result_errors",
    "run_qualified_decode",
    "seal_qualified_decode_request",
    "seal_qualified_canary_request_authority",
    "validate_qualified_answer",
]
