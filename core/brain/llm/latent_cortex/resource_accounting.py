"""Claim-grade resource and information accounting for RLC comparisons.

Token-layer applications remain useful as an admission-control budget, but
they are not a scientific compute currency: one decode token attends over a
different context than one prefill token, and tensor operators, verifiers, or
tools can otherwise disappear from the comparison.  This module keeps exact
native counters, derives neural FLOP estimates from a bound model profile,
and refuses an equal-resource certificate when either arm has unmetered work
or unequal information/policy access.

The estimator is intentionally structural.  It does not claim hardware time,
energy, or quantized-kernel bit operations.  Those remain separate deployment
measurements.  Its purpose is narrower: the same operation graph, estimator,
and information envelope must be applied to every scientific arm.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

MODEL_PROFILE_SCHEMA = "aura.rlc.model_compute_profile.v1"
RESOURCE_ACCOUNTING_SCHEMA = "aura.rlc.resource_accounting.v1"
INFORMATION_ACCOUNTING_SCHEMA = "aura.rlc.information_accounting.v1"
COMPARISON_ACCOUNTING_SCHEMA = "aura.rlc.comparison_accounting.v1"
ESTIMATOR_VERSION = "dense_decoder_gqa_structural_flops_v1"

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
_MAX_COUNTER = 10**30


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _counter(value: Any, *, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_COUNTER:
        raise ValueError(f"{field_name} must be a bounded non-negative integer")
    return value


def _bounded_name(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 160
    ):
        raise ValueError(f"{field_name} must be a bounded non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ModelComputeProfile:
    """Architecture fields required by the structural FLOP estimator."""

    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    head_dim: int
    estimator_version: str = ESTIMATOR_VERSION

    def __post_init__(self) -> None:
        _bounded_name(self.model_type, field_name="model_type")
        if self.estimator_version != ESTIMATOR_VERSION:
            raise ValueError("unsupported compute estimator version")
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "head_dim",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 10_000_000:
                raise ValueError(f"{name} must be a bounded positive integer")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads cannot exceed attention heads")

    @classmethod
    def from_model(cls, model: Any) -> ModelComputeProfile:
        inner = getattr(model, "model", None)
        args = getattr(model, "args", None) or getattr(inner, "args", None)
        layers = getattr(inner, "layers", None)
        if args is None or not layers:
            raise ValueError("model does not expose a decoder compute profile")
        hidden = int(args.hidden_size)
        heads = int(args.num_attention_heads)
        head_dim = int(getattr(args, "head_dim", 0) or hidden // heads)
        return cls(
            model_type=str(getattr(args, "model_type", type(model).__name__)),
            hidden_size=hidden,
            intermediate_size=int(args.intermediate_size),
            num_hidden_layers=len(layers),
            num_attention_heads=heads,
            num_key_value_heads=int(getattr(args, "num_key_value_heads", heads)),
            vocab_size=int(args.vocab_size),
            head_dim=head_dim,
        )

    @property
    def dense_flops_per_token_layer(self) -> int:
        hidden = self.hidden_size
        kv_width = self.num_key_value_heads * self.head_dim
        # Q/K/V/O linear projections plus SwiGLU gate/up/down projections.
        projections = 2 * hidden * (hidden + kv_width + kv_width + hidden)
        mlp = 6 * hidden * self.intermediate_size
        # Two RMS norms, rotary application, and elementwise activation/gating.
        elementwise = 18 * hidden + 4 * (hidden + 2 * kv_width)
        return projections + mlp + elementwise

    @property
    def flops_per_attention_pair(self) -> int:
        # QK dot product and attention-value product, multiply + add each.
        return 4 * self.num_attention_heads * self.head_dim

    @property
    def flops_per_output_head_token(self) -> int:
        return 2 * self.hidden_size * self.vocab_size + 4 * self.hidden_size

    def estimate_neural_flops(
        self,
        *,
        transformer_layer_apps: int,
        attention_query_key_pairs: int,
        output_head_tokens: int,
    ) -> int:
        layer_apps = _counter(
            transformer_layer_apps, field_name="transformer_layer_apps"
        )
        pairs = _counter(
            attention_query_key_pairs, field_name="attention_query_key_pairs"
        )
        head_tokens = _counter(output_head_tokens, field_name="output_head_tokens")
        return (
            layer_apps * self.dense_flops_per_token_layer
            + pairs * self.flops_per_attention_pair
            + head_tokens * self.flops_per_output_head_token
        )

    def to_receipt(self) -> dict[str, Any]:
        body = {
            "schema": MODEL_PROFILE_SCHEMA,
            "estimator_version": self.estimator_version,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "head_dim": self.head_dim,
            "dense_flops_per_token_layer": self.dense_flops_per_token_layer,
            "flops_per_attention_pair": self.flops_per_attention_pair,
            "flops_per_output_head_token": self.flops_per_output_head_token,
        }
        return {**body, "profile_sha256": _canonical_sha256(body)}

    @classmethod
    def from_receipt(cls, value: Any) -> ModelComputeProfile:
        if not isinstance(value, Mapping):
            raise ValueError("model compute profile must be a mapping")
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
        if set(value) != required or value.get("schema") != MODEL_PROFILE_SCHEMA:
            raise ValueError("model compute profile schema is invalid")
        body = {key: value[key] for key in required - {"profile_sha256"}}
        if value.get("profile_sha256") != _canonical_sha256(body):
            raise ValueError("model compute profile digest differs")
        profile = cls(
            model_type=value["model_type"],
            hidden_size=value["hidden_size"],
            intermediate_size=value["intermediate_size"],
            num_hidden_layers=value["num_hidden_layers"],
            num_attention_heads=value["num_attention_heads"],
            num_key_value_heads=value["num_key_value_heads"],
            vocab_size=value["vocab_size"],
            head_dim=value["head_dim"],
            estimator_version=value["estimator_version"],
        )
        expected = profile.to_receipt()
        if dict(value) != expected:
            raise ValueError("model compute profile derived constants differ")
        return profile


def _empty_counters() -> dict[str, int]:
    return {name: 0 for name in RESOURCE_COUNTERS}


@dataclass(slots=True)
class ResourceLedger:
    """Aggregated, digest-bound operation ledger for one arm or episode."""

    profile: ModelComputeProfile | None = None
    operations: dict[str, dict[str, int]] = field(default_factory=dict)
    unknown_operations: set[str] = field(default_factory=set)

    def bind_profile(self, profile: ModelComputeProfile) -> None:
        if not isinstance(profile, ModelComputeProfile):
            raise TypeError("resource ledger profile has an invalid type")
        if self.profile is not None and self.profile != profile:
            raise ValueError("resource ledger model profile changed after binding")
        self.profile = profile

    def _row(self, operation: str) -> dict[str, int]:
        name = _bounded_name(operation, field_name="operation")
        return self.operations.setdefault(name, _empty_counters())

    def charge(self, operation: str, **counters: int) -> None:
        unknown = set(counters) - set(RESOURCE_COUNTERS)
        if unknown:
            raise ValueError(f"unknown resource counters: {sorted(unknown)}")
        row = self._row(operation)
        for name, raw in counters.items():
            value = _counter(raw, field_name=name)
            updated = row[name] + value
            if updated > _MAX_COUNTER:
                raise ValueError(f"resource counter overflow: {name}")
            row[name] = updated

    def mark_unknown(self, operation: str) -> None:
        self.unknown_operations.add(_bounded_name(operation, field_name="operation"))

    def totals(self) -> dict[str, int]:
        totals = _empty_counters()
        for row in self.operations.values():
            for name in RESOURCE_COUNTERS:
                totals[name] += row[name]
        return totals

    def estimated_flops(self) -> int | None:
        if self.profile is None:
            return None
        totals = self.totals()
        return self.profile.estimate_neural_flops(
            transformer_layer_apps=totals["transformer_layer_apps"],
            attention_query_key_pairs=totals["attention_query_key_pairs"],
            output_head_tokens=totals["output_head_tokens"],
        ) + totals["tensor_scalar_ops"] + totals["host_scalar_ops"]

    def to_receipt(self) -> dict[str, Any]:
        operations = {
            name: dict(self.operations[name]) for name in sorted(self.operations)
        }
        body = {
            "schema": RESOURCE_ACCOUNTING_SCHEMA,
            "estimator_version": ESTIMATOR_VERSION,
            "model_profile": self.profile.to_receipt() if self.profile else None,
            "operations": operations,
            "totals": self.totals(),
            "estimated_flops": self.estimated_flops(),
            "unknown_operations": sorted(self.unknown_operations),
            "accounting_complete": bool(
                self.profile is not None and not self.unknown_operations
            ),
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}

    @classmethod
    def aggregate(cls, receipts: Sequence[Mapping[str, Any]]) -> ResourceLedger:
        if not receipts:
            raise ValueError("resource aggregation requires at least one receipt")
        ledger = cls()
        for index, value in enumerate(receipts):
            validated = validate_resource_receipt(value)
            profile = ModelComputeProfile.from_receipt(validated["model_profile"])
            ledger.bind_profile(profile)
            for operation, counters in validated["operations"].items():
                ledger.charge(f"sample_{index}:{operation}", **counters)
            for operation in validated["unknown_operations"]:
                ledger.mark_unknown(f"sample_{index}:{operation}")
        return ledger


def validate_resource_receipt(value: Any) -> dict[str, Any]:
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
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("resource accounting receipt schema is invalid")
    if (
        value.get("schema") != RESOURCE_ACCOUNTING_SCHEMA
        or value.get("estimator_version") != ESTIMATOR_VERSION
        or type(value.get("accounting_complete")) is not bool
    ):
        raise ValueError("resource accounting receipt version is invalid")
    body = {key: value[key] for key in required - {"receipt_sha256"}}
    if value.get("receipt_sha256") != _canonical_sha256(body):
        raise ValueError("resource accounting receipt digest differs")
    profile = ModelComputeProfile.from_receipt(value.get("model_profile"))
    operations = value.get("operations")
    unknown = value.get("unknown_operations")
    if (
        not isinstance(operations, Mapping)
        or not isinstance(unknown, list)
        or unknown != sorted(set(unknown))
    ):
        raise ValueError("resource accounting operations are invalid")
    reconstructed = ResourceLedger(profile=profile)
    for operation in sorted(operations):
        _bounded_name(operation, field_name="operation")
        counters = operations[operation]
        if not isinstance(counters, Mapping) or set(counters) != set(RESOURCE_COUNTERS):
            raise ValueError("resource accounting operation counters are invalid")
        reconstructed.charge(operation, **dict(counters))
    for operation in unknown:
        reconstructed.mark_unknown(operation)
    expected = reconstructed.to_receipt()
    if dict(value) != expected:
        raise ValueError("resource accounting derived totals differ")
    return expected


def build_information_receipt(
    *,
    sources: Sequence[Mapping[str, Any]],
    policies: Mapping[str, str],
    unknown_accesses: Sequence[str] = (),
) -> dict[str, Any]:
    normalized_sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in sources:
        if not isinstance(raw, Mapping) or set(raw) != {
            "source_id",
            "kind",
            "content_sha256",
            "byte_count",
            "token_count",
        }:
            raise ValueError("information source schema is invalid")
        source_id = _bounded_name(raw["source_id"], field_name="source_id")
        kind = _bounded_name(raw["kind"], field_name="kind")
        if source_id in seen or not _is_sha256(raw["content_sha256"]):
            raise ValueError("information source identity is invalid")
        seen.add(source_id)
        normalized_sources.append(
            {
                "source_id": source_id,
                "kind": kind,
                "content_sha256": raw["content_sha256"],
                "byte_count": _counter(raw["byte_count"], field_name="byte_count"),
                "token_count": _counter(raw["token_count"], field_name="token_count"),
            }
        )
    normalized_sources.sort(key=lambda row: (row["source_id"], row["kind"]))
    normalized_policies: dict[str, str] = {}
    for name in sorted(policies):
        _bounded_name(name, field_name="policy name")
        digest = policies[name]
        if not _is_sha256(digest):
            raise ValueError("information policy commitments must be sha256 digests")
        normalized_policies[name] = digest
    unknown = sorted(
        {
            _bounded_name(item, field_name="unknown information access")
            for item in unknown_accesses
        }
    )
    body = {
        "schema": INFORMATION_ACCOUNTING_SCHEMA,
        "sources": normalized_sources,
        "policies": normalized_policies,
        "unknown_accesses": unknown,
        "accounting_complete": not unknown,
    }
    body["source_set_sha256"] = _canonical_sha256(
        {"sources": normalized_sources, "policies": normalized_policies}
    )
    return {**body, "receipt_sha256": _canonical_sha256(body)}


def validate_information_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("information accounting receipt must be a mapping")
    required = {
        "schema",
        "sources",
        "policies",
        "unknown_accesses",
        "accounting_complete",
        "source_set_sha256",
        "receipt_sha256",
    }
    if set(value) != required or value.get("schema") != INFORMATION_ACCOUNTING_SCHEMA:
        raise ValueError("information accounting receipt schema is invalid")
    rebuilt = build_information_receipt(
        sources=value["sources"],
        policies=value["policies"],
        unknown_accesses=value["unknown_accesses"],
    )
    if dict(value) != rebuilt:
        raise ValueError("information accounting receipt differs from its sources")
    return rebuilt


def _within_tolerance(left: int, right: int, *, numerator: int, denominator: int) -> bool:
    _counter(left, field_name="left resource")
    _counter(right, field_name="right resource")
    if type(numerator) is not int or type(denominator) is not int:
        raise TypeError("resource tolerance must use integers")
    if not 0 <= numerator <= denominator or denominator <= 0:
        raise ValueError("resource tolerance must be inside [0, 1]")
    if left == right:
        return True
    if left == 0 or right == 0:
        return False
    return abs(left - right) * denominator <= max(left, right) * numerator


def certify_comparison_accounting(
    *,
    treatment_resource: Mapping[str, Any],
    control_resource: Mapping[str, Any],
    treatment_information: Mapping[str, Any],
    control_information: Mapping[str, Any],
    tolerance_numerator: int = 1,
    tolerance_denominator: int = 20,
    require_compute_parity: bool = True,
) -> dict[str, Any]:
    """Validate both ledgers and emit a fail-closed comparison certificate."""

    if type(require_compute_parity) is not bool:
        raise TypeError("require_compute_parity must be boolean")
    treatment = validate_resource_receipt(treatment_resource)
    control = validate_resource_receipt(control_resource)
    treatment_info = validate_information_receipt(treatment_information)
    control_info = validate_information_receipt(control_information)
    reasons: list[str] = []
    if not treatment["accounting_complete"]:
        reasons.append("treatment_resource_accounting_incomplete")
    if not control["accounting_complete"]:
        reasons.append("control_resource_accounting_incomplete")
    if not treatment_info["accounting_complete"]:
        reasons.append("treatment_information_accounting_incomplete")
    if not control_info["accounting_complete"]:
        reasons.append("control_information_accounting_incomplete")
    if treatment_info["source_set_sha256"] != control_info["source_set_sha256"]:
        reasons.append("information_or_policy_mismatch")
    treatment_profile = treatment["model_profile"]
    control_profile = control["model_profile"]
    if treatment_profile["profile_sha256"] != control_profile["profile_sha256"]:
        reasons.append("compute_estimator_profile_mismatch")

    dimensions: dict[str, dict[str, Any]] = {}
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
    for name, (left, right) in pairs.items():
        matched = (
            type(left) is int
            and type(right) is int
            and _within_tolerance(
                left,
                right,
                numerator=tolerance_numerator,
                denominator=tolerance_denominator,
            )
        )
        dimensions[name] = {
            "treatment": left,
            "control": right,
            "within_tolerance": matched,
        }
        if require_compute_parity and not matched:
            reasons.append(f"resource_mismatch:{name}")

    body = {
        "schema": COMPARISON_ACCOUNTING_SCHEMA,
        "require_compute_parity": require_compute_parity,
        "tolerance_numerator": tolerance_numerator,
        "tolerance_denominator": tolerance_denominator,
        "treatment_resource_sha256": treatment["receipt_sha256"],
        "control_resource_sha256": control["receipt_sha256"],
        "treatment_information_sha256": treatment_info["receipt_sha256"],
        "control_information_sha256": control_info["receipt_sha256"],
        "information_matched": (
            treatment_info["source_set_sha256"]
            == control_info["source_set_sha256"]
        ),
        "resource_dimensions": dimensions,
        "reasons": sorted(set(reasons)),
        "admitted": not reasons,
    }
    return {**body, "certificate_sha256": _canonical_sha256(body)}


def validate_comparison_accounting_certificate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("comparison accounting certificate must be a mapping")
    required = {
        "schema",
        "require_compute_parity",
        "tolerance_numerator",
        "tolerance_denominator",
        "treatment_resource_sha256",
        "control_resource_sha256",
        "treatment_information_sha256",
        "control_information_sha256",
        "information_matched",
        "resource_dimensions",
        "reasons",
        "admitted",
        "certificate_sha256",
    }
    if set(value) != required or value.get("schema") != COMPARISON_ACCOUNTING_SCHEMA:
        raise ValueError("comparison accounting certificate schema is invalid")
    for name in (
        "treatment_resource_sha256",
        "control_resource_sha256",
        "treatment_information_sha256",
        "control_information_sha256",
    ):
        if not _is_sha256(value.get(name)):
            raise ValueError("comparison accounting commitment is invalid")
    if (
        type(value.get("require_compute_parity")) is not bool
        or type(value.get("information_matched")) is not bool
        or type(value.get("admitted")) is not bool
        or type(value.get("tolerance_numerator")) is not int
        or type(value.get("tolerance_denominator")) is not int
        or not 0
        <= value["tolerance_numerator"]
        <= value["tolerance_denominator"]
        or value["tolerance_denominator"] <= 0
        or not isinstance(value.get("resource_dimensions"), Mapping)
        or not isinstance(value.get("reasons"), list)
        or value["reasons"] != sorted(set(value["reasons"]))
    ):
        raise ValueError("comparison accounting verdict fields are invalid")
    expected_dimensions = {"estimated_flops", *NON_NEURAL_PARITY_COUNTERS}
    allowed_reasons = {
        "treatment_resource_accounting_incomplete",
        "control_resource_accounting_incomplete",
        "treatment_information_accounting_incomplete",
        "control_information_accounting_incomplete",
        "information_or_policy_mismatch",
        "compute_estimator_profile_mismatch",
        *(f"resource_mismatch:{name}" for name in expected_dimensions),
    }
    if any(
        not isinstance(reason, str) or reason not in allowed_reasons
        for reason in value["reasons"]
    ):
        raise ValueError("comparison accounting reason is invalid")
    dimensions = value["resource_dimensions"]
    if set(dimensions) != expected_dimensions:
        raise ValueError("comparison accounting dimensions are invalid")
    for name in sorted(expected_dimensions):
        row = dimensions[name]
        if not isinstance(row, Mapping) or set(row) != {
            "treatment",
            "control",
            "within_tolerance",
        }:
            raise ValueError("comparison accounting dimension row is invalid")
        left = _counter(row["treatment"], field_name=f"{name} treatment")
        right = _counter(row["control"], field_name=f"{name} control")
        matched = _within_tolerance(
            left,
            right,
            numerator=value["tolerance_numerator"],
            denominator=value["tolerance_denominator"],
        )
        if row["within_tolerance"] is not matched:
            raise ValueError("comparison accounting tolerance verdict differs")
        mismatch_reason = f"resource_mismatch:{name}"
        if value["require_compute_parity"] is True:
            if (mismatch_reason in value["reasons"]) is matched:
                raise ValueError("comparison accounting resource reason differs")
        elif mismatch_reason in value["reasons"]:
            raise ValueError("comparison accounting contains an inapplicable mismatch")
    information_reason = "information_or_policy_mismatch"
    if (information_reason in value["reasons"]) is value["information_matched"]:
        raise ValueError("comparison accounting information reason differs")
    body = {key: value[key] for key in required - {"certificate_sha256"}}
    if value.get("certificate_sha256") != _canonical_sha256(body):
        raise ValueError("comparison accounting certificate digest differs")
    expected_admitted = not value["reasons"]
    if value["admitted"] is not expected_admitted:
        raise ValueError("comparison accounting verdict contradicts reasons")
    return dict(value)


def policy_sha256(value: Any) -> str:
    """Public helper for committing a JSON-safe policy description."""

    return _canonical_sha256(value)


def triangular_attention_pairs(token_count: int, *, context_tokens: int = 0) -> int:
    tokens = _counter(token_count, field_name="token_count")
    context = _counter(context_tokens, field_name="context_tokens")
    return tokens * context + tokens * (tokens + 1) // 2


__all__ = [
    "COMPARISON_ACCOUNTING_SCHEMA",
    "ESTIMATOR_VERSION",
    "INFORMATION_ACCOUNTING_SCHEMA",
    "MODEL_PROFILE_SCHEMA",
    "NON_NEURAL_PARITY_COUNTERS",
    "RESOURCE_ACCOUNTING_SCHEMA",
    "RESOURCE_COUNTERS",
    "ModelComputeProfile",
    "ResourceLedger",
    "build_information_receipt",
    "certify_comparison_accounting",
    "policy_sha256",
    "triangular_attention_pairs",
    "validate_comparison_accounting_certificate",
    "validate_information_receipt",
    "validate_resource_receipt",
]
