#!/usr/bin/env python
"""Independent exact scoring kernel for resident paired RLC campaigns.

This module deliberately does not import Aura's production campaign grader,
statistics, response parser, or task scorer. It reconstructs raw campaign
evidence and emits the same semantic grade with a separately implemented,
standard-library-only exact statistics kernel.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Never, cast

GRADE_SCHEMA = "aura.latent_cortex.resident_paired_grade.v2"
CAMPAIGN_SCHEMA = "aura.latent_cortex.resident_paired_campaign.v1"
CONTAMINATION_AUDIT_SCHEMA = "aura.latent_cortex.contamination_audit.v2"
COMPARISON_SCHEMA = "aura.latent_cortex.exact_paired_comparison.v1"
INTERACTION_SCHEMA = "aura.latent_cortex.exact_paired_interaction.v1"
MODEL_PROFILE_SCHEMA = "aura.rlc.model_compute_profile.v1"
RESOURCE_ACCOUNTING_SCHEMA = "aura.rlc.resource_accounting.v1"
INFORMATION_ACCOUNTING_SCHEMA = "aura.rlc.information_accounting.v1"
COMPARISON_ACCOUNTING_SCHEMA = "aura.rlc.comparison_accounting.v1"
RESOURCE_ESTIMATOR_VERSION = "dense_decoder_gqa_structural_flops_v1"
RESOURCE_COUNTERS = (
    "transformer_layer_apps",
    "attention_query_key_pairs",
    "output_head_tokens",
    "tensor_element_reads",
    "tensor_element_writes",
    "tensor_scalar_ops",
    "verifier_calls",
    "verifier_input_bytes",
    "verifier_output_bytes",
    "tool_calls",
    "tool_input_bytes",
    "tool_result_bytes",
    "external_model_calls",
    "external_model_input_tokens",
    "external_model_output_tokens",
    "host_scalar_ops",
)
NON_NEURAL_PARITY_COUNTERS = (
    "tensor_element_reads",
    "tensor_element_writes",
    "verifier_calls",
    "verifier_input_bytes",
    "verifier_output_bytes",
    "tool_calls",
    "tool_input_bytes",
    "tool_result_bytes",
    "external_model_calls",
    "external_model_input_tokens",
    "external_model_output_tokens",
)
BOUND_CERTIFICATE_VERSION = (
    "aura.latent_cortex.exact_paired_effect_bounds.v1"
)
NONINFERIORITY_POWER_SCHEMA = (
    "aura.latent_cortex.exact_noninferiority_power.v1"
)
BOUND_METHOD = (
    "four one-sided Clopper-Pearson marginal bounds; Bonferroni over "
    "win/loss x lower/upper x declared families; dyadic outward rounding "
    "with exact binomial-tail witnesses"
)
GRADE_METHOD = (
    "exact paired binomial + exact Holm + simultaneous rational "
    "Clopper-Pearson effect bounds"
)
INTERACTION_METHOD = (
    "exact task-paired 2x2 difference-in-differences sign flip + "
    "simultaneous contrast-composed Clopper-Pearson bounds"
)
SIGN_FLIP_ASSUMPTION = (
    "task draws are independent and arm labels are exchangeable under the "
    "sharp no-interaction null"
)

PROVEN = "PROVEN"
SUPPORTED = "SUPPORTED"
CONJECTURE = "CONJECTURE"
REFUTED = "REFUTED"

BASE_VANILLA = "base_vanilla"
BASE_RLC = "base_rlc"
ADAPTER_VANILLA = "adapter_vanilla"
ADAPTER_RLC = "adapter_rlc"
BASE_EQUAL_COMPUTE = "base_equal_compute"
ADAPTER_EQUAL_COMPUTE = "adapter_equal_compute"
PRIMARY_ARMS = (BASE_VANILLA, BASE_RLC, ADAPTER_VANILLA, ADAPTER_RLC)
FULL_ARMS = (*PRIMARY_ARMS, BASE_EQUAL_COMPUTE, ADAPTER_EQUAL_COMPUTE)
FRONTIER_DOMAINS = (
    "novel_algorithms",
    "mathematics",
    "coding",
    "scientific_inference",
    "long_horizon_planning",
    "calibration",
    "misleading_premise",
)

MIN_DOMAIN_TRIALS = 20
BOUND_PRECISION_BITS = 40
MAX_RESPONSE_BYTES = 32_000
MAX_ANSWER_PAYLOAD_BYTES = 8_000
MAX_JSON_DEPTH = 8
MAX_JSON_NODES = 256
MAX_JSON_STRING_BYTES = 2_048
MAX_RATIONAL_BITS = 262_144
MAX_BOUND_OBSERVATIONS = 4_096
MAX_SIGN_FLIP_OBSERVATIONS = 4_096
MAX_SIGN_FLIP_TOTAL_MAGNITUDE = 100_000
MAX_SIGN_FLIP_STATES = 200_001
MAX_SIGN_FLIP_TRANSITIONS = 100_000_000
MAX_COMPUTE_UNITS = (1 << 63) - 1
FINAL_MARKER = "FINAL_ANSWER:"
_ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")
_ED25519_FIELD = (1 << 255) - 19
_ED25519_ORDER = (
    (1 << 252) + 27742317777372353535851937790883648493
)
_ED25519_D = (
    -121665 * pow(121666, _ED25519_FIELD - 2, _ED25519_FIELD)
) % _ED25519_FIELD
_ED25519_SQRT_M1 = pow(
    2, (_ED25519_FIELD - 1) // 4, _ED25519_FIELD
)
_ED25519_IDENTITY = (0, 1, 1, 0)


class IndependentScoringError(ValueError):
    """Raw evidence cannot be independently and exactly certified."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise IndependentScoringError(code)


@dataclass(frozen=True, slots=True)
class _Q:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            _fail("independent_rational_type_invalid")
        if self.denominator <= 0:
            _fail("independent_rational_denominator_invalid")
        if (
            self.numerator.bit_length() > MAX_RATIONAL_BITS
            or self.denominator.bit_length() > MAX_RATIONAL_BITS
        ):
            _fail("independent_rational_resource_limit")
        if self.numerator == 0:
            object.__setattr__(self, "denominator", 1)
            return
        divisor = math.gcd(abs(self.numerator), self.denominator)
        object.__setattr__(self, "numerator", self.numerator // divisor)
        object.__setattr__(self, "denominator", self.denominator // divisor)


ZERO = _Q(0, 1)
ONE = _Q(1, 1)
NEGATIVE_ONE = _Q(-1, 1)
ALPHA = _Q(1, 20)
MINIMUM_EFFECT = _Q(1, 50)


def _q_payload(value: _Q) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _q_add(left: _Q, right: _Q) -> _Q:
    return _Q(
        left.numerator * right.denominator
        + right.numerator * left.denominator,
        left.denominator * right.denominator,
    )


def _q_subtract(left: _Q, right: _Q) -> _Q:
    return _Q(
        left.numerator * right.denominator
        - right.numerator * left.denominator,
        left.denominator * right.denominator,
    )


def _q_scale(value: _Q, factor: int) -> _Q:
    return _Q(value.numerator * factor, value.denominator)


def _q_less(left: _Q, right: _Q) -> bool:
    return (
        left.numerator * right.denominator
        < right.numerator * left.denominator
    )


def _q_less_equal(left: _Q, right: _Q) -> bool:
    return (
        left.numerator * right.denominator
        <= right.numerator * left.denominator
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise IndependentScoringError(
            "independent_noncanonical_value"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_object(payload: str) -> dict[str, Any]:
    if len(payload.encode("utf-8")) > MAX_ANSWER_PAYLOAD_BYTES:
        _fail("independent_answer_too_large")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("independent_answer_duplicate_key")
            result[key] = value
        return result

    def parse_int(raw: str) -> int:
        digits = raw.removeprefix("-")
        if not digits or len(digits) > 19:
            _fail("independent_answer_integer_out_of_bounds")
        value = int(raw)
        if not -(1 << 63) <= value <= (1 << 63) - 1:
            _fail("independent_answer_integer_out_of_bounds")
        return value

    def reject_float(_raw: str) -> Never:
        _fail("independent_answer_floating_point_forbidden")

    def reject_constant(_raw: str) -> Never:
        _fail("independent_answer_non_finite_number")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs_hook,
            parse_int=parse_int,
            parse_float=reject_float,
            parse_constant=reject_constant,
        )
    except IndependentScoringError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise IndependentScoringError(
            "independent_answer_invalid_json"
        ) from exc
    if not isinstance(value, dict):
        _fail("independent_answer_not_object")
    _validate_json_tree(value)
    return cast(dict[str, Any], value)


def _validate_json_tree(
    value: Any,
    *,
    depth: int = 0,
    nodes: list[int] | None = None,
) -> None:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > MAX_JSON_NODES:
        _fail("independent_answer_too_complex")
    if depth > MAX_JSON_DEPTH:
        _fail("independent_answer_too_deep")
    if value is None or type(value) in {bool, int}:
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES or "\x00" in value:
            _fail("independent_answer_string_invalid")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_tree(item, depth=depth + 1, nodes=nodes)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key.encode("utf-8")) > 128
            ):
                _fail("independent_answer_key_invalid")
            _validate_json_tree(item, depth=depth + 1, nodes=nodes)
        return
    _fail("independent_answer_type_invalid")


def _parse_terminal_answer(response: Any) -> dict[str, Any]:
    if (
        not isinstance(response, str)
        or not response.strip()
        or len(response.encode("utf-8")) > MAX_RESPONSE_BYTES
        or "\x00" in response
    ):
        _fail("independent_response_invalid")
    if response.count(FINAL_MARKER) != 1:
        _fail("independent_answer_marker_count")
    lines = response.rstrip().splitlines()
    if (
        not lines
        or not lines[-1].startswith(FINAL_MARKER)
        or any(FINAL_MARKER in line for line in lines[:-1])
    ):
        _fail("independent_answer_not_terminal")
    encoded = lines[-1][len(FINAL_MARKER) :].strip()
    if not encoded:
        _fail("independent_answer_missing")
    return _strict_json_object(encoded)


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _score_response(task: Any, response: Any) -> dict[str, Any]:
    try:
        answer = _parse_terminal_answer(response)
    except IndependentScoringError as exc:
        return {
            "parsed": False,
            "correct": False,
            "reason": exc.code,
            "normalized_answer_sha256": None,
        }
    try:
        blinded = task.reveal_for_verifier()
    except Exception as exc:  # noqa: BLE001 - independent trust boundary
        raise IndependentScoringError(
            "independent_answer_reveal_failed"
        ) from exc
    expected = blinded.get("expected") if isinstance(blinded, Mapping) else None
    if not isinstance(expected, dict):
        _fail("independent_expected_answer_invalid")
    correct = _strict_equal(answer, expected)
    return {
        "parsed": True,
        "correct": correct,
        "reason": "correct" if correct else "incorrect_or_schema_mismatch",
        "normalized_answer_sha256": _sha256(answer),
    }


def _ed25519_decode_point(encoded: bytes) -> tuple[int, int, int, int]:
    if len(encoded) != 32:
        _fail("independent_ed25519_point_invalid")
    y = int.from_bytes(encoded, "little") & ((1 << 255) - 1)
    sign = encoded[31] >> 7
    if y >= _ED25519_FIELD:
        _fail("independent_ed25519_point_invalid")
    y_squared = y * y % _ED25519_FIELD
    denominator = (_ED25519_D * y_squared + 1) % _ED25519_FIELD
    if denominator == 0:
        _fail("independent_ed25519_point_invalid")
    x_squared = (
        (y_squared - 1)
        * pow(denominator, _ED25519_FIELD - 2, _ED25519_FIELD)
    ) % _ED25519_FIELD
    x = pow(x_squared, (_ED25519_FIELD + 3) // 8, _ED25519_FIELD)
    if (x * x - x_squared) % _ED25519_FIELD:
        x = x * _ED25519_SQRT_M1 % _ED25519_FIELD
    if (x * x - x_squared) % _ED25519_FIELD:
        _fail("independent_ed25519_point_invalid")
    if x & 1 != sign:
        x = (-x) % _ED25519_FIELD
    if x == 0 and sign:
        _fail("independent_ed25519_point_invalid")
    return x, y, 1, x * y % _ED25519_FIELD


def _ed25519_add(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = (y1 - x1) * (y2 - x2) % _ED25519_FIELD
    b = (y1 + x1) * (y2 + x2) % _ED25519_FIELD
    c = 2 * _ED25519_D * t1 * t2 % _ED25519_FIELD
    d = 2 * z1 * z2 % _ED25519_FIELD
    e = (b - a) % _ED25519_FIELD
    f = (d - c) % _ED25519_FIELD
    g = (d + c) % _ED25519_FIELD
    h = (b + a) % _ED25519_FIELD
    return (
        e * f % _ED25519_FIELD,
        g * h % _ED25519_FIELD,
        f * g % _ED25519_FIELD,
        e * h % _ED25519_FIELD,
    )


def _ed25519_multiply(
    point: tuple[int, int, int, int],
    scalar: int,
) -> tuple[int, int, int, int]:
    if type(scalar) is not int or scalar < 0:
        _fail("independent_ed25519_scalar_invalid")
    result = _ED25519_IDENTITY
    addend = point
    while scalar:
        if scalar & 1:
            result = _ed25519_add(result, addend)
        addend = _ed25519_add(addend, addend)
        scalar >>= 1
    return result


def _ed25519_equal(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return (
        (left[0] * right[2] - right[0] * left[2]) % _ED25519_FIELD == 0
        and (left[1] * right[2] - right[1] * left[2])
        % _ED25519_FIELD
        == 0
    )


@lru_cache(maxsize=1)
def _ed25519_basepoint() -> tuple[int, int, int, int]:
    y = 4 * pow(5, _ED25519_FIELD - 2, _ED25519_FIELD) % _ED25519_FIELD
    return _ed25519_decode_point(y.to_bytes(32, "little"))


def _verify_ed25519(
    public_der: bytes,
    signature: bytes,
    message: bytes,
) -> bool:
    if (
        len(public_der) != len(_ED25519_SPKI_PREFIX) + 32
        or not public_der.startswith(_ED25519_SPKI_PREFIX)
        or len(signature) != 64
    ):
        return False
    public_raw = public_der[len(_ED25519_SPKI_PREFIX) :]
    encoded_r = signature[:32]
    scalar_s = int.from_bytes(signature[32:], "little")
    if scalar_s >= _ED25519_ORDER:
        return False
    try:
        public_point = _ed25519_decode_point(public_raw)
        point_r = _ed25519_decode_point(encoded_r)
    except IndependentScoringError:
        return False
    if (
        not _ed25519_equal(
            _ed25519_multiply(public_point, _ED25519_ORDER),
            _ED25519_IDENTITY,
        )
        or not _ed25519_equal(
            _ed25519_multiply(point_r, _ED25519_ORDER),
            _ED25519_IDENTITY,
        )
    ):
        return False
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + public_raw + message).digest(),
        "little",
    ) % _ED25519_ORDER
    left = _ed25519_multiply(_ed25519_basepoint(), scalar_s)
    right = _ed25519_add(
        point_r,
        _ed25519_multiply(public_point, challenge),
    )
    return _ed25519_equal(left, right)


def _contamination_metadata_valid(
    audit: Any,
    *,
    task_manifest_sha256: Any,
    trusted_root_sha256: str | None,
) -> bool:
    if not isinstance(audit, Mapping) or set(audit) != {
        "schema",
        "task_manifest_sha256",
        "status",
        "overlap_count",
        "auditor_independence",
        "corpora",
        "methods",
        "signature",
    }:
        return False
    signature = audit.get("signature")
    corpora = audit.get("corpora")
    methods = audit.get("methods")
    if (
        audit.get("schema") != CONTAMINATION_AUDIT_SCHEMA
        or audit.get("task_manifest_sha256") != task_manifest_sha256
        or audit.get("status") != "passed_zero_overlap"
        or type(audit.get("overlap_count")) is not int
        or audit.get("overlap_count") != 0
        or audit.get("auditor_independence") != "external"
        or not isinstance(corpora, list)
        or not corpora
        or any(
            not isinstance(record, Mapping)
            or set(record) != {"name", "snapshot_sha256"}
            or not isinstance(record.get("name"), str)
            or not _is_sha256(record.get("snapshot_sha256"))
            for record in corpora
        )
        or not isinstance(methods, list)
        or not {"exact_prompt", "normalized_prompt", "token_fivegram"}.issubset(
            methods
        )
        or not isinstance(signature, Mapping)
        or set(signature)
        != {
            "algorithm",
            "key_id",
            "signature_b64",
            "signed_payload_sha256",
            "public_key_der_b64",
            "trust_root_sha256",
            "verified",
        }
        or signature.get("algorithm") != "ed25519"
        or signature.get("verified") is not True
        or not _is_sha256(signature.get("key_id"))
        or not _is_sha256(signature.get("signed_payload_sha256"))
        or not _is_sha256(signature.get("trust_root_sha256"))
        or signature.get("trust_root_sha256") != trusted_root_sha256
    ):
        return False
    body = {key: value for key, value in audit.items() if key != "signature"}
    if signature.get("signed_payload_sha256") != _sha256(body):
        return False
    try:
        public_der = base64.b64decode(
            cast(str, signature.get("public_key_der_b64")), validate=True
        )
        signature_bytes = base64.b64decode(
            cast(str, signature.get("signature_b64")), validate=True
        )
    except (TypeError, ValueError, binascii.Error):
        return False
    public_sha256 = hashlib.sha256(public_der).hexdigest()
    return (
        public_sha256 == trusted_root_sha256
        and signature.get("key_id") == public_sha256
        and _verify_ed25519(
            public_der,
            signature_bytes,
            _canonical_bytes(body),
        )
    )


def _resource_counter(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 10**30:
        _fail("independent_resource_counter_invalid")
    return value


def _resource_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 160
    ):
        _fail("independent_resource_name_invalid")
    return value


def _validate_model_profile(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "estimator_version",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "head_dim",
        "dense_flops_per_token_layer",
        "flops_per_attention_pair",
        "flops_per_output_head_token",
        "profile_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema") != MODEL_PROFILE_SCHEMA
        or value.get("estimator_version") != RESOURCE_ESTIMATOR_VERSION
    ):
        _fail("independent_model_compute_profile_invalid")
    _resource_name(value.get("model_type"))
    dimensions = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "head_dim",
    )
    if any(
        type(value.get(name)) is not int or not 1 <= value[name] <= 10_000_000
        for name in dimensions
    ):
        _fail("independent_model_compute_profile_invalid")
    hidden = value["hidden_size"]
    heads = value["num_attention_heads"]
    kv_heads = value["num_key_value_heads"]
    head_dim = value["head_dim"]
    if hidden != heads * head_dim or kv_heads > heads:
        _fail("independent_model_compute_profile_invalid")
    kv_width = kv_heads * head_dim
    expected_derived = {
        "dense_flops_per_token_layer": (
            2 * hidden * (hidden + kv_width + kv_width + hidden)
            + 6 * hidden * value["intermediate_size"]
            + 18 * hidden
            + 4 * (hidden + 2 * kv_width)
        ),
        "flops_per_attention_pair": 4 * heads * head_dim,
        "flops_per_output_head_token": 2 * hidden * value["vocab_size"] + 4 * hidden,
    }
    if any(value.get(name) != expected for name, expected in expected_derived.items()):
        _fail("independent_model_compute_profile_invalid")
    body = {key: value[key] for key in required - {"profile_sha256"}}
    if value.get("profile_sha256") != _sha256(body):
        _fail("independent_model_compute_profile_invalid")
    return dict(value)


def _validate_resource_accounting(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "estimator_version",
        "model_profile",
        "operations",
        "totals",
        "estimated_flops",
        "unknown_operations",
        "accounting_complete",
        "receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema") != RESOURCE_ACCOUNTING_SCHEMA
        or value.get("estimator_version") != RESOURCE_ESTIMATOR_VERSION
        or type(value.get("accounting_complete")) is not bool
    ):
        _fail("independent_resource_accounting_invalid")
    body = {key: value[key] for key in required - {"receipt_sha256"}}
    if value.get("receipt_sha256") != _sha256(body):
        _fail("independent_resource_accounting_invalid")
    profile = _validate_model_profile(value.get("model_profile"))
    operations = value.get("operations")
    unknown = value.get("unknown_operations")
    if (
        not isinstance(operations, Mapping)
        or not isinstance(unknown, list)
        or any(not isinstance(name, str) for name in operations)
        or any(not isinstance(item, str) for item in unknown)
        or unknown != sorted(set(unknown))
    ):
        _fail("independent_resource_accounting_invalid")
    totals = {name: 0 for name in RESOURCE_COUNTERS}
    normalized_operations: dict[str, dict[str, int]] = {}
    for operation in sorted(operations):
        name = _resource_name(operation)
        counters = operations[operation]
        if not isinstance(counters, Mapping) or set(counters) != set(RESOURCE_COUNTERS):
            _fail("independent_resource_accounting_invalid")
        row = {
            counter: _resource_counter(counters[counter])
            for counter in RESOURCE_COUNTERS
        }
        normalized_operations[name] = row
        for counter in RESOURCE_COUNTERS:
            totals[counter] = _resource_counter(totals[counter] + row[counter])
    normalized_unknown = sorted(_resource_name(item) for item in unknown)
    dense_flops = (
        totals["transformer_layer_apps"]
        * profile["dense_flops_per_token_layer"]
        + totals["attention_query_key_pairs"] * profile["flops_per_attention_pair"]
        + totals["output_head_tokens"] * profile["flops_per_output_head_token"]
        + totals["tensor_scalar_ops"]
        + totals["host_scalar_ops"]
    )
    rebuilt_body = {
        "schema": RESOURCE_ACCOUNTING_SCHEMA,
        "estimator_version": RESOURCE_ESTIMATOR_VERSION,
        "model_profile": profile,
        "operations": normalized_operations,
        "totals": totals,
        "estimated_flops": dense_flops,
        "unknown_operations": normalized_unknown,
        "accounting_complete": not normalized_unknown,
    }
    rebuilt = {**rebuilt_body, "receipt_sha256": _sha256(rebuilt_body)}
    if not _strict_equal(dict(value), rebuilt):
        _fail("independent_resource_accounting_invalid")
    return rebuilt


def _validate_information_accounting(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "sources",
        "policies",
        "unknown_accesses",
        "accounting_complete",
        "source_set_sha256",
        "receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema") != INFORMATION_ACCOUNTING_SCHEMA
        or not isinstance(value.get("sources"), list)
        or not isinstance(value.get("policies"), Mapping)
        or not isinstance(value.get("unknown_accesses"), list)
        or any(not isinstance(name, str) for name in value.get("policies", {}))
        or any(
            not isinstance(item, str) for item in value.get("unknown_accesses", [])
        )
    ):
        _fail("independent_information_accounting_invalid")
    sources: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for raw in value["sources"]:
        if not isinstance(raw, Mapping) or set(raw) != {
            "source_id",
            "kind",
            "content_sha256",
            "byte_count",
            "token_count",
        }:
            _fail("independent_information_accounting_invalid")
        source_id = _resource_name(raw.get("source_id"))
        if source_id in source_ids or not _is_sha256(raw.get("content_sha256")):
            _fail("independent_information_accounting_invalid")
        source_ids.add(source_id)
        sources.append(
            {
                "source_id": source_id,
                "kind": _resource_name(raw.get("kind")),
                "content_sha256": raw["content_sha256"],
                "byte_count": _resource_counter(raw.get("byte_count")),
                "token_count": _resource_counter(raw.get("token_count")),
            }
        )
    sources.sort(key=lambda row: (row["source_id"], row["kind"]))
    policies: dict[str, str] = {}
    for raw_name in sorted(value["policies"]):
        name = _resource_name(raw_name)
        digest = value["policies"][raw_name]
        if not _is_sha256(digest):
            _fail("independent_information_accounting_invalid")
        policies[name] = digest
    unknown = sorted({_resource_name(item) for item in value["unknown_accesses"]})
    source_set_sha256 = _sha256({"sources": sources, "policies": policies})
    rebuilt_body = {
        "schema": INFORMATION_ACCOUNTING_SCHEMA,
        "sources": sources,
        "policies": policies,
        "unknown_accesses": unknown,
        "accounting_complete": not unknown,
        "source_set_sha256": source_set_sha256,
    }
    rebuilt = {**rebuilt_body, "receipt_sha256": _sha256(rebuilt_body)}
    if not _strict_equal(dict(value), rebuilt):
        _fail("independent_information_accounting_invalid")
    return rebuilt


def _comparison_resource_match(left: int, right: int, tolerance: _Q) -> bool:
    if left == right:
        return True
    if left == 0 or right == 0:
        return False
    return (
        abs(left - right) * tolerance.denominator
        <= max(left, right) * tolerance.numerator
    )


def _comparison_accounting(
    *,
    treatment_resource: Mapping[str, Any],
    control_resource: Mapping[str, Any],
    treatment_information: Mapping[str, Any],
    control_information: Mapping[str, Any],
    tolerance: _Q,
    require_compute: bool,
) -> dict[str, Any]:
    treatment = _validate_resource_accounting(treatment_resource)
    control = _validate_resource_accounting(control_resource)
    treatment_info = _validate_information_accounting(treatment_information)
    control_info = _validate_information_accounting(control_information)
    reasons: list[str] = []
    if not treatment["accounting_complete"]:
        reasons.append("treatment_resource_accounting_incomplete")
    if not control["accounting_complete"]:
        reasons.append("control_resource_accounting_incomplete")
    if not treatment_info["accounting_complete"]:
        reasons.append("treatment_information_accounting_incomplete")
    if not control_info["accounting_complete"]:
        reasons.append("control_information_accounting_incomplete")
    information_matched = (
        treatment_info["source_set_sha256"] == control_info["source_set_sha256"]
    )
    if not information_matched:
        reasons.append("information_or_policy_mismatch")
    if (
        treatment["model_profile"]["profile_sha256"]
        != control["model_profile"]["profile_sha256"]
    ):
        reasons.append("compute_estimator_profile_mismatch")
    pairs = {
        "estimated_flops": (
            treatment["estimated_flops"],
            control["estimated_flops"],
        ),
        **{
            name: (treatment["totals"][name], control["totals"][name])
            for name in NON_NEURAL_PARITY_COUNTERS
        },
    }
    dimensions: dict[str, dict[str, Any]] = {}
    for name, (left, right) in pairs.items():
        matched = _comparison_resource_match(left, right, tolerance)
        dimensions[name] = {
            "treatment": left,
            "control": right,
            "within_tolerance": matched,
        }
        if require_compute and not matched:
            reasons.append(f"resource_mismatch:{name}")
    body = {
        "schema": COMPARISON_ACCOUNTING_SCHEMA,
        "require_compute_parity": require_compute,
        "tolerance_numerator": tolerance.numerator,
        "tolerance_denominator": tolerance.denominator,
        "treatment_resource_sha256": treatment["receipt_sha256"],
        "control_resource_sha256": control["receipt_sha256"],
        "treatment_information_sha256": treatment_info["receipt_sha256"],
        "control_information_sha256": control_info["receipt_sha256"],
        "information_matched": information_matched,
        "resource_dimensions": dimensions,
        "reasons": sorted(set(reasons)),
        "admitted": not reasons,
    }
    return {**body, "certificate_sha256": _sha256(body)}


@dataclass(frozen=True, slots=True)
class _CampaignMaterial:
    rows: dict[
        str,
        dict[str, tuple[str, bool, int, dict[str, Any], dict[str, Any]]],
    ]
    arms: tuple[str, ...]
    expected_task_count: int
    expected_cell_count: int
    claim_eligible: bool
    plan_sha256: str


def _extract_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    plan: Any,
    issuer_tasks: Sequence[Any],
    trusted_contamination_root_sha256: str | None,
    trusted_campaign_policy_sha256: str | None,
) -> _CampaignMaterial:
    try:
        document = plan.to_dict()
        plan_sha256 = plan.plan_sha256
        cell_ids = tuple(plan.cell_ids)
    except (AttributeError, TypeError, ValueError) as exc:
        raise IndependentScoringError("independent_plan_invalid") from exc
    metadata = document.get("metadata") if isinstance(document, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema") != CAMPAIGN_SCHEMA
    ):
        _fail("independent_plan_metadata_invalid")
    raw_arms = metadata.get("arms")
    task_manifest = metadata.get("task_manifest")
    model_identity = metadata.get("model_identity")
    adapter_identity = metadata.get("adapter_identity")
    execution_config = metadata.get("execution_config")
    if (
        not isinstance(raw_arms, list)
        or not isinstance(task_manifest, Mapping)
        or not isinstance(model_identity, Mapping)
        or not isinstance(adapter_identity, Mapping)
        or not isinstance(execution_config, Mapping)
    ):
        _fail("independent_plan_metadata_invalid")
    arms = tuple(raw_arms)
    if (
        any(not isinstance(arm, str) for arm in arms)
        or len(set(arms)) != len(arms)
        or not set(PRIMARY_ARMS).issubset(arms)
        or any(arm not in FULL_ARMS for arm in arms)
    ):
        _fail("independent_plan_arms_invalid")
    public_tasks = task_manifest.get("tasks")
    if not isinstance(public_tasks, list) or not public_tasks:
        _fail("independent_plan_tasks_invalid")
    task_records: dict[str, Mapping[str, Any]] = {}
    for task in public_tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("task_id"), str):
            _fail("independent_plan_tasks_invalid")
        task_id = cast(str, task["task_id"])
        if task_id in task_records:
            _fail("independent_plan_task_duplicate")
        task_records[task_id] = task
    issuer_by_id: dict[str, Any] = {}
    if (
        isinstance(issuer_tasks, (str, bytes))
        or not isinstance(issuer_tasks, Sequence)
        or not issuer_tasks
    ):
        _fail("independent_issuer_tasks_invalid")
    for task in issuer_tasks:
        task_id = getattr(task, "task_id", None)
        if not isinstance(task_id, str) or task_id in issuer_by_id:
            _fail("independent_issuer_tasks_invalid")
        public = getattr(task, "public", None)
        if public is None or not _strict_equal(
            public.to_dict(),
            dict(task_records.get(task_id, {})),
        ):
            _fail("independent_issuer_task_mismatch")
        issuer_by_id[task_id] = task
    if set(issuer_by_id) != set(task_records):
        _fail("independent_issuer_task_mismatch")

    claim_eligible = metadata.get("claim_eligible")
    if type(claim_eligible) is not bool:
        _fail("independent_claim_eligibility_invalid")
    if claim_eligible:
        campaign_trust = metadata.get("campaign_trust")
        contamination_root = metadata.get("contamination_trust_root_sha256")
        audit = metadata.get("contamination_audit")
        if (
            execution_config.get("worker_origin_protocol")
            != "detached_supervisor_staged_arm_import_v3"
            or type(execution_config.get("worker_origin_attempt_slots")) is not int
            or execution_config.get("worker_origin_attempt_slots", 0) <= 0
            or not isinstance(campaign_trust, Mapping)
            or campaign_trust.get("prelaunch_verified") is not True
            or campaign_trust.get("externally_custodied") is not True
            or not _is_sha256(campaign_trust.get("policy_sha256"))
            or campaign_trust.get("policy_sha256")
            != trusted_campaign_policy_sha256
            or contamination_root != trusted_contamination_root_sha256
            or not _contamination_metadata_valid(
                audit,
                task_manifest_sha256=task_manifest.get("manifest_sha256"),
                trusted_root_sha256=trusted_contamination_root_sha256,
            )
            or arms != FULL_ARMS
        ):
            _fail("independent_claim_trust_invalid")
        planned_domains: set[str] = set()
        for task in task_records.values():
            domain = task.get("domain")
            if not isinstance(domain, str):
                _fail("independent_plan_tasks_invalid")
            planned_domains.add(domain)
        if planned_domains != set(FRONTIER_DOMAINS):
            _fail("independent_claim_eligibility_invalid")
        configured_domains = execution_config.get("domains")
        generation_seed_count = execution_config.get("generation_seed_count")
        observed_power = execution_config.get("exact_statistical_power")
        domain_counts: dict[str, int] = defaultdict(int)
        for task in task_records.values():
            domain_counts[cast(str, task["domain"])] += 1
        if (
            not isinstance(configured_domains, list)
            or not configured_domains
            or any(
                not isinstance(domain, str) or not domain
                for domain in configured_domains
            )
            or len(set(configured_domains)) != len(configured_domains)
            or set(configured_domains) != set(domain_counts)
            or type(generation_seed_count) is not int
            or generation_seed_count <= 0
            or any(
                count != generation_seed_count
                for count in domain_counts.values()
            )
            or not isinstance(observed_power, Mapping)
        ):
            _fail("independent_claim_power_invalid")
        expected_power = _exact_campaign_power_plan(
            domain_count=len(domain_counts),
            comparison_count=4
            + int(BASE_EQUAL_COMPUTE in arms)
            + int(ADAPTER_EQUAL_COMPUTE in arms),
            arm_count=len(arms),
            planned_observations_per_domain=generation_seed_count,
        )
        if (
            _canonical_bytes(dict(observed_power))
            != _canonical_bytes(expected_power)
            or expected_power["powered_for_zero_loss_noninferiority"]
            is not True
        ):
            _fail("independent_claim_power_invalid")
    arm_execution_order = metadata.get("arm_execution_order")
    if (
        not isinstance(arm_execution_order, list)
        or len(arm_execution_order) != len(arms)
        or any(not isinstance(arm, str) for arm in arm_execution_order)
        or set(arm_execution_order) != set(arms)
    ):
        _fail("independent_plan_arm_order_invalid")

    expected_pairs = {
        (task_id, arm) for task_id in task_records for arm in arms
    }
    planned_pairs: set[tuple[str, str]] = set()
    execution_ordinals: dict[str, set[int]] = defaultdict(set)
    expected_task_count = len(task_records)
    for cell_id in cell_ids:
        definition = plan.cell_definition(cell_id)
        task_id = definition.get("task_id")
        arm = definition.get("arm")
        task = task_records.get(task_id) if isinstance(task_id, str) else None
        ordinal = definition.get("execution_ordinal_within_arm")
        if (
            task is None
            or arm not in arms
            or definition.get("domain") != task.get("domain")
            or definition.get("task_payload_sha256")
            != task.get("task_payload_sha256")
            or type(ordinal) is not int
            or not 0 <= ordinal < expected_task_count
        ):
            _fail("independent_plan_cell_invalid")
        pair = (cast(str, task_id), cast(str, arm))
        if pair in planned_pairs:
            _fail("independent_plan_cell_invalid")
        planned_pairs.add(pair)
        execution_ordinals[cast(str, arm)].add(ordinal)
    if planned_pairs != expected_pairs or any(
        execution_ordinals[arm] != set(range(expected_task_count))
        for arm in arms
    ):
        _fail("independent_plan_coverage_invalid")

    runtime_bundle = model_identity.get("runtime_bundle")
    implementation_sha256 = execution_config.get("implementation_sha256")
    adapter_receipt = adapter_identity.get("identity_receipt")
    if (
        not isinstance(runtime_bundle, Mapping)
        or not isinstance(implementation_sha256, Mapping)
        or not isinstance(adapter_receipt, Mapping)
    ):
        _fail("independent_runtime_plan_identity_invalid")

    rows: dict[
        str,
        dict[str, tuple[str, bool, int, dict[str, Any], dict[str, Any]]],
    ] = defaultdict(dict)
    seen_cells: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            _fail("independent_record_invalid")
        cell_id = record.get("cell_id")
        definition = record.get("definition")
        result = record.get("result")
        verification = record.get("verification")
        commit = record.get("commit")
        if (
            not isinstance(cell_id, str)
            or cell_id not in cell_ids
            or cell_id in seen_cells
            or not isinstance(definition, Mapping)
            or not isinstance(result, Mapping)
            or not isinstance(verification, Mapping)
            or not isinstance(commit, Mapping)
        ):
            _fail("independent_record_shape_invalid")
        seen_cells.add(cell_id)
        expected_definition = plan.cell_definition(cell_id)
        if not _strict_equal(dict(definition), expected_definition):
            _fail("independent_record_definition_mismatch")
        task_id = expected_definition.get("task_id")
        domain = expected_definition.get("domain")
        arm = expected_definition.get("arm")
        if (
            not isinstance(task_id, str)
            or not isinstance(domain, str)
            or arm not in arms
            or result.get("arm") != arm
        ):
            _fail("independent_record_identity_invalid")
        text = result.get("text")
        output_sha256 = result.get("output_sha256")
        layer_apps = result.get("layer_apps")
        if (
            not isinstance(text, str)
            or not isinstance(output_sha256, str)
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != output_sha256
            or type(layer_apps) is not int
            or layer_apps <= 0
            or layer_apps > MAX_COMPUTE_UNITS
        ):
            _fail("independent_result_invalid")
        runtime = result.get("runtime_model_identity")
        planned_personality = model_identity.get("personality_adapter")
        planned_effective_stack = model_identity.get(
            "effective_stack_sha256"
        )
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("worker_model_path") != model_identity.get("model_path")
            or not _strict_equal(
                runtime.get("worker_model_parameter_count"),
                runtime_bundle.get("logical_parameter_count"),
            )
            or runtime.get("worker_model_parameter_count_basis")
            != runtime_bundle.get("logical_parameter_count_basis")
            or runtime.get("worker_weight_fingerprint")
            != model_identity.get("fingerprint")
            or runtime.get("worker_weight_fingerprint_method")
            != model_identity.get("method")
            or not _strict_equal(
                runtime.get("worker_weight_file_count"),
                model_identity.get("files"),
            )
            or runtime.get("worker_runtime_bundle_sha256")
            != runtime_bundle.get("bundle_sha256")
            or runtime.get("worker_load_boundary_verified") is not True
            or runtime.get("worker_source_sha256")
            != implementation_sha256.get(
                "tools/run_latent_cortex_paired_campaign.py"
            )
            or (
                planned_personality is not None
                and runtime.get("worker_personality_adapter")
                != planned_personality
            )
            or (
                planned_effective_stack is not None
                and runtime.get("worker_effective_stack_sha256")
                != planned_effective_stack
            )
        ):
            _fail("independent_runtime_model_identity_mismatch")
        if arm.startswith("adapter_"):
            if (
                result.get("adapter_identity_sha256")
                != adapter_receipt.get("composite_identity_sha256")
                or not _strict_equal(
                    result.get("adapter_wrapped_projections"),
                    adapter_receipt.get("wrapped_projection_count"),
                )
                or not isinstance(result.get("runtime_adapter_identity"), Mapping)
                or not _strict_equal(
                    dict(result["runtime_adapter_identity"]),
                    dict(adapter_receipt),
                )
            ):
                _fail("independent_adapter_activation_mismatch")
        elif (
            result.get("adapter_identity_sha256") is not None
            or type(result.get("adapter_wrapped_projections")) is not int
            or result.get("adapter_wrapped_projections") != 0
            or result.get("runtime_adapter_identity") is not None
        ):
            _fail("independent_base_arm_adapter_contaminated")
        resource_accounting = _validate_resource_accounting(
            result.get("resource_accounting")
        )
        information_accounting = _validate_information_accounting(
            result.get("information_accounting")
        )
        if (
            resource_accounting["accounting_complete"] is not True
            or information_accounting["accounting_complete"] is not True
        ):
            _fail("independent_record_accounting_incomplete")
        if arm.endswith("_rlc"):
            episode_receipt = result.get("episode_receipt")
            episode_budget = (
                episode_receipt.get("budget")
                if isinstance(episode_receipt, Mapping)
                else None
            )
            if (
                not isinstance(episode_budget, Mapping)
                or not _strict_equal(
                    episode_budget.get("resource_accounting"),
                    resource_accounting,
                )
                or not _strict_equal(
                    episode_budget.get("information_accounting"),
                    information_accounting,
                )
            ):
                _fail("independent_episode_accounting_binding_invalid")
        independent = _score_response(issuer_by_id[task_id], text)
        score_receipt = verification.get("score_receipt")
        score_reason = score_receipt.get("reason") if isinstance(
            score_receipt, Mapping
        ) else None
        if (
            type(verification.get("correct")) is not bool
            or verification.get("correct") is not independent["correct"]
            or not isinstance(score_receipt, Mapping)
            or score_receipt.get("parsed") is not independent["parsed"]
            or score_receipt.get("correct") is not independent["correct"]
            or score_receipt.get("normalized_answer_sha256")
            != independent["normalized_answer_sha256"]
            or not isinstance(score_reason, str)
            or verification.get("answer_commitment_sha256")
            != task_records[task_id].get("answer_commitment_sha256")
        ):
            _fail("independent_score_disagrees_with_receipt")
        if (
            commit.get("result_sha256") != _sha256(dict(result))
            or commit.get("verification_sha256")
            != _sha256(dict(verification))
        ):
            _fail("independent_commitment_mismatch")
        if arm in rows[task_id]:
            _fail("independent_task_arm_duplicate")
        rows[task_id][cast(str, arm)] = (
            domain,
            cast(bool, independent["correct"]),
            layer_apps,
            resource_accounting,
            information_accounting,
        )
    return _CampaignMaterial(
        rows=dict(rows),
        arms=arms,
        expected_task_count=expected_task_count,
        expected_cell_count=expected_task_count * len(arms),
        claim_eligible=claim_eligible,
        plan_sha256=cast(str, plan_sha256),
    )


def _paired_tail(wins: int, losses: int) -> _Q:
    if (
        type(wins) is not int
        or type(losses) is not int
        or wins < 0
        or losses < 0
        or wins + losses > 4096
    ):
        _fail("independent_paired_tail_invalid")
    discordant = wins + losses
    if discordant == 0:
        return ONE
    return _Q(
        sum(
            math.comb(discordant, count)
            for count in range(wins, discordant + 1)
        ),
        1 << discordant,
    )


def _holm(pvalues: Mapping[str, _Q]) -> tuple[dict[str, _Q], list[dict[str, Any]]]:
    if not pvalues or len(pvalues) > 1024:
        _fail("independent_holm_invalid")
    if any(
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or len(name) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or type(value) is not _Q
        or _q_less(value, ZERO)
        or _q_less(ONE, value)
        for name, value in pvalues.items()
    ):
        _fail("independent_holm_invalid")
    from functools import cmp_to_key

    def compare(left: tuple[str, _Q], right: tuple[str, _Q]) -> int:
        cross_left = left[1].numerator * right[1].denominator
        cross_right = right[1].numerator * left[1].denominator
        if cross_left != cross_right:
            return -1 if cross_left < cross_right else 1
        return -1 if left[0] < right[0] else (1 if left[0] > right[0] else 0)

    ordered = sorted(pvalues.items(), key=cmp_to_key(compare))
    adjusted: dict[str, _Q] = {}
    entries: list[dict[str, Any]] = []
    running = ZERO
    family_size = len(ordered)
    for index, (name, raw) in enumerate(ordered):
        scaled = _q_scale(raw, family_size - index)
        if _q_less(ONE, scaled):
            scaled = ONE
        if _q_less(running, scaled):
            running = scaled
        adjusted[name] = running
        entries.append(
            {
                "hypothesis": name,
                "rank": index + 1,
                "raw": _q_payload(raw),
                "adjusted": _q_payload(running),
            }
        )
    return adjusted, entries


def _within_compute(treatment: int, control: int, tolerance: _Q) -> bool:
    if (
        type(treatment) is not int
        or type(control) is not int
        or treatment <= 0
        or control <= 0
        or treatment > MAX_COMPUTE_UNITS
        or control > MAX_COMPUTE_UNITS
    ):
        _fail("independent_compute_invalid")
    return (
        abs(treatment - control) * tolerance.denominator
        <= control * tolerance.numerator
    )


def _binomial_counts(
    trials: int,
    successes: int,
    probability: _Q,
    *,
    upper_tail: bool,
) -> tuple[int, int]:
    p = probability.numerator
    q = probability.denominator - p
    denominator = pow(probability.denominator, trials)
    if p == 0:
        if upper_tail:
            return (denominator if successes == 0 else 0), denominator
        return denominator, denominator
    if q == 0:
        if upper_tail:
            return denominator, denominator
        return (
            denominator if successes == trials else 0
        ), denominator
    first, last = (successes, trials) if upper_tail else (0, successes)
    term = pow(q, trials)
    numerator = 0
    for count in range(trials + 1):
        if first <= count <= last:
            numerator += term
        if count == trials:
            break
        next_numerator = term * (trials - count) * p
        next_denominator = (count + 1) * q
        term, remainder = divmod(next_numerator, next_denominator)
        if remainder:
            _fail("independent_binomial_recurrence_invalid")
    return numerator, denominator


def _counts_at_most(
    counts: tuple[int, int],
    threshold: _Q,
) -> bool:
    return (
        counts[0] * threshold.denominator
        <= threshold.numerator * counts[1]
    )


@lru_cache(maxsize=16_384)
def _proportion_bound(
    component: str,
    successes: int,
    trials: int,
    bound_kind: str,
    component_alpha: _Q,
    precision_bits: int,
) -> dict[str, Any]:
    if bound_kind == "lower" and successes == 0:
        return _component_payload(
            component,
            bound_kind,
            "exact-boundary",
            successes,
            trials,
            ZERO,
            None,
            component_alpha,
            None,
            None,
            precision_bits,
        )
    if bound_kind == "upper" and successes == trials:
        return _component_payload(
            component,
            bound_kind,
            "exact-boundary",
            successes,
            trials,
            ONE,
            None,
            component_alpha,
            None,
            None,
            precision_bits,
        )
    scale = 1 << precision_bits
    lower_index, upper_index = 0, scale
    upper_tail = bound_kind == "lower"
    while lower_index + 1 < upper_index:
        middle = (lower_index + upper_index) // 2
        counts = _binomial_counts(
            trials,
            successes,
            _Q(middle, scale),
            upper_tail=upper_tail,
        )
        satisfies = _counts_at_most(counts, component_alpha)
        if bound_kind == "lower":
            if satisfies:
                lower_index = middle
            else:
                upper_index = middle
        elif satisfies:
            upper_index = middle
        else:
            lower_index = middle
    selected_index, adjacent_index = (
        (lower_index, upper_index)
        if bound_kind == "lower"
        else (upper_index, lower_index)
    )
    bound = _Q(selected_index, scale)
    adjacent = _Q(adjacent_index, scale)
    selected_counts = _binomial_counts(
        trials, successes, bound, upper_tail=upper_tail
    )
    adjacent_counts = _binomial_counts(
        trials, successes, adjacent, upper_tail=upper_tail
    )
    if (
        not _counts_at_most(selected_counts, component_alpha)
        or _counts_at_most(adjacent_counts, component_alpha)
    ):
        _fail("independent_bound_witness_invalid")
    return _component_payload(
        component,
        bound_kind,
        "upper" if upper_tail else "lower",
        successes,
        trials,
        bound,
        _Q(*selected_counts),
        component_alpha,
        adjacent,
        _Q(*adjacent_counts),
        precision_bits,
    )


def _component_payload(
    component: str,
    bound_kind: str,
    tail_kind: str,
    successes: int,
    trials: int,
    bound: _Q,
    tail_probability: _Q | None,
    component_alpha: _Q,
    adjacent_bound: _Q | None,
    adjacent_tail_probability: _Q | None,
    precision_bits: int,
) -> dict[str, Any]:
    return {
        "component": component,
        "bound_kind": bound_kind,
        "tail_kind": tail_kind,
        "successes": successes,
        "trials": trials,
        "bound": _q_payload(bound),
        "tail_probability": (
            _q_payload(tail_probability)
            if tail_probability is not None
            else None
        ),
        "component_alpha": _q_payload(component_alpha),
        "adjacent_bound": (
            _q_payload(adjacent_bound) if adjacent_bound is not None else None
        ),
        "adjacent_tail_probability": (
            _q_payload(adjacent_tail_probability)
            if adjacent_tail_probability is not None
            else None
        ),
        "precision_bits": precision_bits,
        "certified": True,
    }


@lru_cache(maxsize=4096)
def _effect_bounds(
    wins: int,
    losses: int,
    ties: int,
    family_count: int,
) -> dict[str, Any]:
    observations = wins + losses + ties
    if (
        any(type(value) is not int or value < 0 for value in (wins, losses, ties))
        or not 1 <= observations <= MAX_BOUND_OBSERVATIONS
        or type(family_count) is not int
        or not 1 <= family_count <= 1024
    ):
        _fail("independent_effect_bounds_invalid")
    component_alpha = _Q(
        ALPHA.numerator,
        ALPHA.denominator * 4 * family_count,
    )
    components = [
        _proportion_bound(
            "win_lower",
            wins,
            observations,
            "lower",
            component_alpha,
            BOUND_PRECISION_BITS,
        ),
        _proportion_bound(
            "win_upper",
            wins,
            observations,
            "upper",
            component_alpha,
            BOUND_PRECISION_BITS,
        ),
        _proportion_bound(
            "loss_lower",
            losses,
            observations,
            "lower",
            component_alpha,
            BOUND_PRECISION_BITS,
        ),
        _proportion_bound(
            "loss_upper",
            losses,
            observations,
            "upper",
            component_alpha,
            BOUND_PRECISION_BITS,
        ),
    ]
    by_name = {item["component"]: item for item in components}
    win_lower = _Q(**by_name["win_lower"]["bound"])
    win_upper = _Q(**by_name["win_upper"]["bound"])
    loss_lower = _Q(**by_name["loss_lower"]["bound"])
    loss_upper = _Q(**by_name["loss_upper"]["bound"])
    lower = _q_subtract(win_lower, loss_upper)
    upper = _q_subtract(win_upper, loss_lower)
    if _q_less(lower, NEGATIVE_ONE):
        lower = NEGATIVE_ONE
    if _q_less(ONE, upper):
        upper = ONE
    if _q_less(upper, lower):
        _fail("independent_effect_bounds_inverted")
    grid = _Q(1, 1 << BOUND_PRECISION_BITS)
    return {
        "certificate_version": BOUND_CERTIFICATE_VERSION,
        "method": BOUND_METHOD,
        "certified": True,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "observations": observations,
        "lower": _q_payload(lower),
        "upper": _q_payload(upper),
        "family_count": family_count,
        "family_alpha": _q_payload(ALPHA),
        "component_alpha": _q_payload(component_alpha),
        "simultaneous_coverage_lower": _q_payload(_Q(19, 20)),
        "precision_bits": BOUND_PRECISION_BITS,
        "grid_step": _q_payload(grid),
        "endpoint_max_outward_rounding": _q_payload(_q_scale(grid, 2)),
        "components": components,
    }


@lru_cache(maxsize=64)
def _minimum_zero_loss_noninferiority_observations(
    global_bound_family_count: int,
) -> dict[str, Any]:
    if (
        type(global_bound_family_count) is not int
        or global_bound_family_count <= 0
    ):
        _fail("independent_power_contract_invalid")
    negative_margin = _Q(-MINIMUM_EFFECT.numerator, MINIMUM_EFFECT.denominator)

    def powered(observations: int) -> tuple[bool, dict[str, Any]]:
        bounds = _effect_bounds(
            0,
            0,
            observations,
            global_bound_family_count,
        )
        return (
            _q_less(negative_margin, _Q(**bounds["lower"])),
            bounds,
        )

    low = 0
    high = 1
    while high < MAX_BOUND_OBSERVATIONS:
        attainable, _bounds = powered(high)
        if attainable:
            break
        low = high
        high = min(high * 2, MAX_BOUND_OBSERVATIONS)
    attainable, _bounds = powered(high)
    if not attainable:
        _fail("independent_power_unattainable")
    while low + 1 < high:
        middle = (low + high) // 2
        passes, _bounds = powered(middle)
        if passes:
            high = middle
        else:
            low = middle
    passes, selected = powered(high)
    if not passes:
        _fail("independent_power_minimum_invalid")
    if high > 1:
        prior_passes, prior = powered(high - 1)
        if prior_passes:
            _fail("independent_power_minimum_invalid")
        prior_lower = prior["lower"]
        prior_upper = prior["upper"]
    else:
        prior_lower = None
        prior_upper = None
    return {
        "schema": NONINFERIORITY_POWER_SCHEMA,
        "certified": True,
        "global_bound_family_count": global_bound_family_count,
        "margin": _q_payload(MINIMUM_EFFECT),
        "minimum_observations": high,
        "selected_lower": selected["lower"],
        "selected_upper": selected["upper"],
        "prior_observations": high - 1 if high > 1 else None,
        "prior_lower": prior_lower,
        "prior_upper": prior_upper,
        "precision_bits": BOUND_PRECISION_BITS,
        "resource_max_observations": MAX_BOUND_OBSERVATIONS,
    }


def _exact_campaign_power_plan(
    *,
    domain_count: int,
    comparison_count: int,
    arm_count: int,
    planned_observations_per_domain: int,
) -> dict[str, Any]:
    if (
        type(domain_count) is not int
        or domain_count <= 0
        or type(comparison_count) is not int
        or comparison_count <= 0
        or type(arm_count) is not int
        or arm_count <= 0
        or type(planned_observations_per_domain) is not int
        or planned_observations_per_domain <= 0
    ):
        _fail("independent_power_contract_invalid")
    global_count = comparison_count * (domain_count + 1) + 2
    receipt = _minimum_zero_loss_noninferiority_observations(global_count)
    planned_tasks = planned_observations_per_domain * domain_count
    return {
        **receipt,
        "domain_count": domain_count,
        "comparison_count": comparison_count,
        "planned_observations_per_domain": planned_observations_per_domain,
        "planned_total_tasks": planned_tasks,
        "planned_total_cells": planned_tasks * arm_count,
        "powered_for_zero_loss_noninferiority": (
            planned_observations_per_domain >= receipt["minimum_observations"]
        ),
    }


def _sign_flip(values: Sequence[int]) -> tuple[_Q, dict[str, int]]:
    materialized = tuple(values)
    if (
        not materialized
        or len(materialized) > MAX_SIGN_FLIP_OBSERVATIONS
        or any(type(value) is not int for value in materialized)
        or sum(abs(value) for value in materialized)
        > MAX_SIGN_FLIP_TOTAL_MAGNITUDE
    ):
        _fail("independent_sign_flip_invalid")
    histogram: dict[int, int] = {0: 1}
    transitions = 0
    for value in materialized:
        transitions += 2 * len(histogram)
        if transitions > MAX_SIGN_FLIP_TRANSITIONS:
            _fail("independent_sign_flip_resource_limit")
        updated: dict[int, int] = {}
        for total, multiplicity in histogram.items():
            updated[total + value] = (
                updated.get(total + value, 0) + multiplicity
            )
            updated[total - value] = (
                updated.get(total - value, 0) + multiplicity
            )
        if len(updated) > MAX_SIGN_FLIP_STATES:
            _fail("independent_sign_flip_resource_limit")
        histogram = updated
    assignments = 1 << len(materialized)
    if sum(histogram.values()) != assignments:
        _fail("independent_sign_flip_mass_invalid")
    observed_sum = sum(materialized)
    numerator = sum(
        multiplicity
        for total, multiplicity in histogram.items()
        if total >= observed_sum
    )
    return _Q(numerator, assignments), {
        "threshold": observed_sum,
        "observations": len(materialized),
        "observed_sum": observed_sum,
        "total_assignments": assignments,
    }


def _comparison(
    rows: Mapping[
        str,
        Mapping[str, tuple[str, bool, int, dict[str, Any], dict[str, Any]]],
    ],
    *,
    treatment: str,
    control: str,
    require_compute: bool,
    compute_tolerance: _Q,
    global_bound_family_count: int,
) -> dict[str, Any]:
    by_domain: dict[str, list[tuple[str, bool, bool, int, int]]] = defaultdict(
        list
    )
    accounting_certificates: list[dict[str, Any]] = []
    for task_id in sorted(rows):
        arms = rows[task_id]
        if treatment not in arms or control not in arms:
            _fail("independent_comparison_incomplete")
        (
            treatment_domain,
            treatment_ok,
            treatment_cost,
            treatment_resource,
            treatment_information,
        ) = arms[treatment]
        (
            control_domain,
            control_ok,
            control_cost,
            control_resource,
            control_information,
        ) = arms[control]
        if treatment_domain != control_domain:
            _fail("independent_domain_drift")
        certificate = _comparison_accounting(
            treatment_resource=treatment_resource,
            control_resource=control_resource,
            treatment_information=treatment_information,
            control_information=control_information,
            tolerance=compute_tolerance,
            require_compute=require_compute,
        )
        accounting_certificates.append(
            {"task_id": task_id, "family": treatment_domain, **certificate}
        )
        by_domain[treatment_domain].append(
            (task_id, treatment_ok, control_ok, treatment_cost, control_cost)
        )
    if not by_domain:
        _fail("independent_comparison_empty")

    family_material: dict[str, dict[str, Any]] = {}
    family_bounds: dict[str, dict[str, Any]] = {}
    raw_pvalues: dict[str, _Q] = {}
    invalid_compute: list[str] = []
    underpowered: list[str] = []
    pooled: list[int] = []
    for domain in sorted(by_domain):
        observations = by_domain[domain]
        differences = [int(row[1]) - int(row[2]) for row in observations]
        wins = differences.count(1)
        losses = differences.count(-1)
        ties = len(differences) - wins - losses
        bounds = _effect_bounds(
            wins, losses, ties, global_bound_family_count
        )
        pvalue = _paired_tail(wins, losses)
        mismatches = [
            row[0]
            for row in observations
            if not _within_compute(row[3], row[4], compute_tolerance)
        ]
        if require_compute and mismatches:
            invalid_compute.append(domain)
        if len(observations) < MIN_DOMAIN_TRIALS:
            underpowered.append(domain)
        else:
            raw_pvalues[domain] = pvalue
        family_bounds[domain] = bounds
        family_material[domain] = {
            "n": len(observations),
            "treatment_wins": wins,
            "control_wins": losses,
            "ties": ties,
            "paired_effect": _q_payload(
                _Q(sum(differences), len(differences))
            ),
            "effect_bounds": bounds,
            "one_sided_exact_p": _q_payload(pvalue),
            "holm_adjusted_p": None,
            "missing_compute": False,
            "nonpositive_compute": False,
            "compute_mismatch_task_ids": mismatches,
        }
        pooled.extend(differences)
    if raw_pvalues:
        adjusted, holm_entries = _holm(raw_pvalues)
        for name, value in adjusted.items():
            family_material[name]["holm_adjusted_p"] = _q_payload(value)
        holm_size = len(adjusted)
    else:
        adjusted = {}
        holm_entries = []
        holm_size = 0
    positive = [
        domain
        for domain in sorted(by_domain)
        if domain in adjusted
        and _q_less(adjusted[domain], ALPHA)
        and _q_less(
            MINIMUM_EFFECT, _Q(**family_bounds[domain]["lower"])
        )
        and domain not in invalid_compute
    ]
    negative_minimum = _Q(-1, 50)
    regressed = [
        domain
        for domain in sorted(by_domain)
        if _q_less(
            _Q(**family_bounds[domain]["upper"]), negative_minimum
        )
    ]
    noninferior = [
        domain
        for domain in sorted(by_domain)
        if _q_less(
            negative_minimum, _Q(**family_bounds[domain]["lower"])
        )
    ]
    pooled_wins = pooled.count(1)
    pooled_losses = pooled.count(-1)
    pooled_ties = len(pooled) - pooled_wins - pooled_losses
    pooled_bounds = _effect_bounds(
        pooled_wins,
        pooled_losses,
        pooled_ties,
        global_bound_family_count,
    )
    pooled_p = _paired_tail(pooled_wins, pooled_losses)
    pooled_positive = (
        len(pooled) >= MIN_DOMAIN_TRIALS
        and _q_less(pooled_p, ALPHA)
        and _q_less(MINIMUM_EFFECT, _Q(**pooled_bounds["lower"]))
    )
    required_positive = max(2, (2 * len(by_domain) + 2) // 3)
    if (
        len(positive) >= required_positive
        and pooled_positive
        and not regressed
    ):
        tier = PROVEN
    elif positive and pooled_positive and not regressed:
        tier = SUPPORTED
    elif regressed or _q_less_equal(
        _Q(**pooled_bounds["upper"]), ZERO
    ):
        tier = REFUTED
    else:
        tier = CONJECTURE
    evidence = {
        "schema": COMPARISON_SCHEMA,
        "method": GRADE_METHOD,
        "treatment": treatment,
        "control": control,
        "alpha": _q_payload(ALPHA),
        "minimum_effect": _q_payload(MINIMUM_EFFECT),
        "require_compute": require_compute,
        "compute_tolerance": _q_payload(compute_tolerance),
        "global_bound_family_count": global_bound_family_count,
        "bound_precision_bits": BOUND_PRECISION_BITS,
        "families": family_material,
        "holm": {
            "method": "Holm step-down, exact rational",
            "family_size": holm_size,
            "ordered": holm_entries,
        },
        "positive_families": positive,
        "noninferior_families": noninferior,
        "all_families_noninferior": len(noninferior) == len(by_domain),
        "regressed_families": regressed,
        "underpowered_families": underpowered,
        "invalid_compute_families": invalid_compute,
        "required_positive_families": required_positive,
        "pooled": {
            "n": len(pooled),
            "treatment_wins": pooled_wins,
            "control_wins": pooled_losses,
            "ties": pooled_ties,
            "paired_effect": _q_payload(_Q(sum(pooled), len(pooled))),
            "effect_bounds": pooled_bounds,
            "one_sided_exact_p": _q_payload(pooled_p),
        },
    }
    accounting_admitted = all(
        certificate["admitted"] for certificate in accounting_certificates
    )
    evidence["resource_accounting_required"] = True
    evidence["comparison_accounting_admitted"] = accounting_admitted
    evidence["comparison_accounting"] = accounting_certificates
    if not accounting_admitted:
        tier = CONJECTURE
    return {
        "experiment": f"{treatment}_vs_{control}",
        "statement": f"{treatment} improves over {control}",
        "tier": tier,
        "evidence": evidence,
    }


def _interaction(
    adapter: Sequence[int],
    base: Sequence[int],
    *,
    global_bound_family_count: int,
) -> dict[str, Any]:
    if not adapter or len(adapter) != len(base):
        _fail("independent_interaction_invalid")
    interaction = [
        adapter_value - base_value
        for adapter_value, base_value in zip(adapter, base, strict=True)
    ]
    adapter_bounds = _effect_bounds(
        adapter.count(1),
        adapter.count(-1),
        adapter.count(0),
        global_bound_family_count,
    )
    base_bounds = _effect_bounds(
        base.count(1),
        base.count(-1),
        base.count(0),
        global_bound_family_count,
    )
    lower = _q_subtract(
        _Q(**adapter_bounds["lower"]), _Q(**base_bounds["upper"])
    )
    upper = _q_subtract(
        _Q(**adapter_bounds["upper"]), _Q(**base_bounds["lower"])
    )
    pvalue, sign_flip = _sign_flip(interaction)
    return {
        "schema": INTERACTION_SCHEMA,
        "method": INTERACTION_METHOD,
        "n": len(interaction),
        "mean": _q_payload(_Q(sum(interaction), len(interaction))),
        "lower": _q_payload(lower),
        "upper": _q_payload(upper),
        "alpha": _q_payload(ALPHA),
        "minimum_effect": _q_payload(MINIMUM_EFFECT),
        "global_bound_family_count": global_bound_family_count,
        "simultaneous_coverage_lower": _q_payload(_Q(19, 20)),
        "one_sided_exact_sign_flip_p": _q_payload(pvalue),
        "sign_flip_assumption": SIGN_FLIP_ASSUMPTION,
        "sign_flip_assumption_preregistered": True,
        "sign_flip": sign_flip,
        "adapter_contrast_bounds": adapter_bounds,
        "base_contrast_bounds": base_bounds,
        "interaction_values": interaction,
    }


def _semantic_grade(material: _CampaignMaterial) -> dict[str, Any]:
    rows = material.rows
    observed_cells = sum(len(task_arms) for task_arms in rows.values())
    complete = (
        len(rows) == material.expected_task_count
        and observed_cells == material.expected_cell_count
        and all(
            set(task_arms) == set(material.arms)
            for task_arms in rows.values()
        )
    )
    if not complete:
        body = {
            "schema": GRADE_SCHEMA,
            "verdict": "incomplete",
            "claim_tier": CONJECTURE,
            "expected_task_count": material.expected_task_count,
            "expected_cell_count": material.expected_cell_count,
            "observed_task_count": len(rows),
            "observed_cell_count": observed_cells,
            "frontier_claim_eligible": False,
            "same_checkpoint_gain_claim_eligible": material.claim_eligible,
            "plan_sha256": material.plan_sha256,
            "reasons": ["campaign_incomplete"],
        }
        return {**body, "grade_sha256": _sha256(body)}

    domain_counts: dict[str, int] = defaultdict(int)
    for task_arms in rows.values():
        domain_counts[task_arms[BASE_VANILLA][0]] += 1
    comparison_count = 4
    if BASE_EQUAL_COMPUTE in material.arms:
        comparison_count += 1
    if ADAPTER_EQUAL_COMPUTE in material.arms:
        comparison_count += 1
    global_bound_family_count = (
        comparison_count * (len(domain_counts) + 1) + 2
    )
    comparisons = {
        "base_rlc_gain": _comparison(
            rows,
            treatment=BASE_RLC,
            control=BASE_VANILLA,
            require_compute=False,
            compute_tolerance=ONE,
            global_bound_family_count=global_bound_family_count,
        ),
        "adapter_rlc_gain": _comparison(
            rows,
            treatment=ADAPTER_RLC,
            control=ADAPTER_VANILLA,
            require_compute=False,
            compute_tolerance=ONE,
            global_bound_family_count=global_bound_family_count,
        ),
        "adapter_effect_under_rlc": _comparison(
            rows,
            treatment=ADAPTER_RLC,
            control=BASE_RLC,
            require_compute=False,
            compute_tolerance=ONE,
            global_bound_family_count=global_bound_family_count,
        ),
        "adapter_effect_under_vanilla": _comparison(
            rows,
            treatment=ADAPTER_VANILLA,
            control=BASE_VANILLA,
            require_compute=False,
            compute_tolerance=ONE,
            global_bound_family_count=global_bound_family_count,
        ),
    }
    if BASE_EQUAL_COMPUTE in material.arms:
        comparisons["base_equal_compute"] = _comparison(
            rows,
            treatment=BASE_RLC,
            control=BASE_EQUAL_COMPUTE,
            require_compute=True,
            compute_tolerance=_Q(1, 5),
            global_bound_family_count=global_bound_family_count,
        )
    if ADAPTER_EQUAL_COMPUTE in material.arms:
        comparisons["adapter_equal_compute"] = _comparison(
            rows,
            treatment=ADAPTER_RLC,
            control=ADAPTER_EQUAL_COMPUTE,
            require_compute=True,
            compute_tolerance=_Q(1, 5),
            global_bound_family_count=global_bound_family_count,
        )
    adapter_differences: list[int] = []
    base_differences: list[int] = []
    for task_arms in rows.values():
        adapter_differences.append(
            int(task_arms[ADAPTER_RLC][1])
            - int(task_arms[ADAPTER_VANILLA][1])
        )
        base_differences.append(
            int(task_arms[BASE_RLC][1])
            - int(task_arms[BASE_VANILLA][1])
        )
    interaction = _interaction(
        adapter_differences,
        base_differences,
        global_bound_family_count=global_bound_family_count,
    )
    underpowered = sorted(
        domain
        for domain, count in domain_counts.items()
        if count < MIN_DOMAIN_TRIALS
    )
    required_claims = ["adapter_rlc_gain", "adapter_effect_under_rlc"]
    if ADAPTER_EQUAL_COMPUTE in material.arms:
        required_claims.append("adapter_equal_compute")
    statistically_proven = (
        not underpowered
        and all(
            comparisons[name]["tier"] == PROVEN for name in required_claims
        )
        and _q_less(MINIMUM_EFFECT, _Q(**interaction["lower"]))
        and _q_less(
            _Q(**interaction["one_sided_exact_sign_flip_p"]), ALPHA
        )
        and comparisons["adapter_effect_under_vanilla"]["evidence"].get(
            "all_families_noninferior"
        )
    )
    refuted = (
        comparisons["adapter_rlc_gain"]["tier"] == REFUTED
        or _q_less_equal(_Q(**interaction["upper"]), ZERO)
    )
    if statistically_proven and material.claim_eligible:
        verdict, tier, reasons = (
            "gain_preverified",
            CONJECTURE,
            ["independent_final_verifier_required"],
        )
    elif statistically_proven:
        verdict, tier, reasons = (
            "gain_observed_preflight",
            CONJECTURE,
            ["campaign_not_claim_eligible"],
        )
    elif refuted:
        verdict, tier, reasons = (
            "gain_refuted",
            REFUTED,
            ["gain_gate_failed"],
        )
    elif underpowered:
        verdict, tier, reasons = (
            "incomplete_underpowered",
            CONJECTURE,
            [f"underpowered_domain:{domain}" for domain in underpowered],
        )
    else:
        verdict, tier, reasons = (
            "inconclusive",
            CONJECTURE,
            ["gain_not_proven"],
        )
    body = {
        "schema": GRADE_SCHEMA,
        "verdict": verdict,
        "claim_tier": tier,
        "expected_task_count": material.expected_task_count,
        "expected_cell_count": material.expected_cell_count,
        "observed_task_count": len(rows),
        "observed_cell_count": observed_cells,
        "plan_sha256": material.plan_sha256,
        "domain_counts": dict(sorted(domain_counts.items())),
        "statistical_policy": {
            "alpha": _q_payload(ALPHA),
            "minimum_effect": _q_payload(MINIMUM_EFFECT),
            "minimum_domain_observations": MIN_DOMAIN_TRIALS,
            "minimum_domain_count": 2,
            "global_bound_family_count": global_bound_family_count,
            "bound_precision_bits": BOUND_PRECISION_BITS,
        },
        "comparisons": comparisons,
        "interaction": interaction,
        "frontier_claim_eligible": False,
        "same_checkpoint_gain_claim_eligible": material.claim_eligible,
        "reasons": reasons,
    }
    return {**body, "grade_sha256": _sha256(body)}


def independent_grade_campaign(
    records: Iterable[Mapping[str, Any]],
    *,
    plan: Any,
    issuer_tasks: Sequence[Any],
    trusted_contamination_root_sha256: str | None = None,
    trusted_campaign_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an independently reconstructed exact semantic grade."""

    material = _extract_rows(
        records,
        plan=plan,
        issuer_tasks=issuer_tasks,
        trusted_contamination_root_sha256=trusted_contamination_root_sha256,
        trusted_campaign_policy_sha256=trusted_campaign_policy_sha256,
    )
    semantic_grade = _semantic_grade(material)
    return {
        "semantic_grade": semantic_grade,
        "semantic_grade_canonical_sha256": hashlib.sha256(
            _canonical_bytes(semantic_grade)
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


__all__ = [
    "IndependentScoringError",
    "independent_grade_campaign",
]
