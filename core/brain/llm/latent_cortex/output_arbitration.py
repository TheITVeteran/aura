"""Sealed authority for choosing between ordinary and recurrent output.

The Recursive Latent Cortex is an enhancement lane.  Process signals such as
convergence, branch agreement, verifier-shaped reward, or a valid response
schema are not authority to overwrite an ordinary answer: they do not prove
that the recurrent answer is correct.  This module makes that boundary a pure,
replayable decision.

Candidate text remains private.  The public receipt binds text and tokens by
hash, binds both candidates to the same request/model/seed/contract, and records
the independent verifier interval that actually authorized the selection.
Unmeasured evidence, partial coverage, ties, and tampering retain the ordinary
incumbent.  A recurrent candidate can be selected only when it independently
rescues an inadmissible incumbent or its lower bound clears the incumbent's
upper bound by the configured margin.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

OUTPUT_ARBITRATION_SCHEMA = "aura.rlc.output_arbitration.v1"
DEFAULT_ARBITRATION_MARGIN = 0.05
MAX_ARBITRATION_OUTPUT_TOKENS = 12_288
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class CandidateSource(StrEnum):
    VANILLA = "vanilla"
    RLC = "rlc"


class ArbitrationDecision(StrEnum):
    RETAIN_VANILLA = "retain_vanilla"
    SELECT_RLC = "select_rlc"
    NO_ADMISSIBLE_OUTPUT = "no_admissible_output"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _bound(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be finite in [0, 1]")
    return round(float(value), 10)


def _margin(value: Any) -> float:
    result = _bound(value, field="arbitration margin")
    if result >= 1.0:
        raise ValueError("arbitration margin must be less than 1")
    return result


@dataclass(frozen=True)
class OutputCandidate:
    """Private output plus the evidence allowed to grant serving authority."""

    source: CandidateSource
    text: str
    tokens: tuple[int, ...]
    request_sha256: str
    model_sha256: str
    seed: int
    contract_sha256: str
    contract_valid: bool
    product_valid: bool | None
    verifier_independent: bool
    full_span_coverage: bool
    quality_lower_bound: float
    quality_upper_bound: float
    verifier_receipt_sha256: str = ""
    material_regression: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, CandidateSource):
            raise ValueError("candidate source must be a CandidateSource")
        if not isinstance(self.text, str):
            raise ValueError("candidate text must be a string")
        if not isinstance(self.tokens, tuple) or any(
            type(token) is not int or token < 0 for token in self.tokens
        ):
            raise ValueError("candidate tokens must be a tuple of non-negative integers")
        if len(self.tokens) > MAX_ARBITRATION_OUTPUT_TOKENS:
            raise ValueError("candidate exceeds the arbitration output-token limit")
        if bool(self.text) is not bool(self.tokens):
            raise ValueError("candidate text and tokens must be present or absent together")
        _require_sha256(self.request_sha256, field="request_sha256")
        _require_sha256(self.model_sha256, field="model_sha256")
        _require_sha256(self.contract_sha256, field="contract_sha256")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("candidate seed must be a non-negative integer")
        for field_name in (
            "contract_valid",
            "verifier_independent",
            "full_span_coverage",
            "material_regression",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be boolean")
        if self.product_valid is not None and type(self.product_valid) is not bool:
            raise ValueError("product_valid must be true, false, or unmeasured")
        lower = _bound(self.quality_lower_bound, field="quality_lower_bound")
        upper = _bound(self.quality_upper_bound, field="quality_upper_bound")
        if lower > upper:
            raise ValueError("candidate quality interval is inverted")
        if self.product_valid is not None:
            if not self.verifier_independent:
                raise ValueError("measured product validity requires an independent verifier")
            _require_sha256(
                self.verifier_receipt_sha256,
                field="verifier_receipt_sha256",
            )
        else:
            _require_sha256(
                self.verifier_receipt_sha256,
                field="verifier_receipt_sha256",
                allow_empty=True,
            )
        if self.product_valid is True and not self.full_span_coverage:
            raise ValueError("positive product validity requires full-span coverage")
        if self.product_valid is False and upper != 0.0:
            raise ValueError("a refuted product must have a zero upper bound")

    @property
    def admissible(self) -> bool:
        return bool(
            self.text
            and self.tokens
            and self.contract_valid
            and self.product_valid is not False
            and not self.material_regression
        )

    @property
    def authoritative(self) -> bool:
        return bool(
            self.admissible
            and self.product_valid is True
            and self.verifier_independent
            and self.full_span_coverage
            and self.verifier_receipt_sha256
        )

    @property
    def text_sha256(self) -> str:
        return _text_sha256(self.text)

    @property
    def tokens_sha256(self) -> str:
        return _canonical_sha256(list(self.tokens))

    def public_evidence(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "request_sha256": self.request_sha256,
            "model_sha256": self.model_sha256,
            "seed": self.seed,
            "contract_sha256": self.contract_sha256,
            "output": {
                "text_sha256": self.text_sha256,
                "tokens_sha256": self.tokens_sha256,
                "token_count": len(self.tokens),
            },
            "contract_valid": self.contract_valid,
            "product_valid": self.product_valid,
            "verifier_independent": self.verifier_independent,
            "full_span_coverage": self.full_span_coverage,
            "quality_interval": {
                "lower_bound": _bound(
                    self.quality_lower_bound,
                    field="quality_lower_bound",
                ),
                "upper_bound": _bound(
                    self.quality_upper_bound,
                    field="quality_upper_bound",
                ),
            },
            "verifier_receipt_sha256": self.verifier_receipt_sha256,
            "material_regression": self.material_regression,
            "admissible": self.admissible,
            "authoritative": self.authoritative,
        }


def _validate_pair(vanilla: OutputCandidate, rlc: OutputCandidate) -> None:
    if vanilla.source is not CandidateSource.VANILLA:
        raise ValueError("incumbent candidate must be vanilla")
    if rlc.source is not CandidateSource.RLC:
        raise ValueError("challenger candidate must be rlc")
    vanilla_binding = (
        vanilla.request_sha256,
        vanilla.model_sha256,
        vanilla.seed,
        vanilla.contract_sha256,
    )
    rlc_binding = (
        rlc.request_sha256,
        rlc.model_sha256,
        rlc.seed,
        rlc.contract_sha256,
    )
    if vanilla_binding != rlc_binding:
        raise ValueError("candidate request/model/seed/contract bindings differ")


def _decide(
    vanilla: OutputCandidate,
    rlc: OutputCandidate,
    *,
    margin: float,
) -> tuple[ArbitrationDecision, str]:
    if vanilla.admissible:
        if vanilla.text_sha256 == rlc.text_sha256 and rlc.admissible:
            return ArbitrationDecision.RETAIN_VANILLA, "equivalent_output_keeps_incumbent"
        if rlc.authoritative and (
            _bound(rlc.quality_lower_bound, field="rlc lower bound")
            > _bound(vanilla.quality_upper_bound, field="vanilla upper bound") + margin
        ):
            return (
                ArbitrationDecision.SELECT_RLC,
                "rlc_authoritative_lower_bound_dominates_vanilla",
            )
        if not rlc.contract_valid:
            reason = "rlc_contract_invalid"
        elif rlc.product_valid is False:
            reason = "rlc_product_refuted"
        elif rlc.material_regression:
            reason = "rlc_material_regression"
        elif not rlc.authoritative:
            reason = "rlc_lacks_independent_full_span_authority"
        else:
            reason = "rlc_dominance_not_proven"
        return ArbitrationDecision.RETAIN_VANILLA, reason
    if rlc.authoritative:
        return ArbitrationDecision.SELECT_RLC, "rlc_authoritative_rescue"
    return ArbitrationDecision.NO_ADMISSIBLE_OUTPUT, "no_admissible_verified_output"


def build_output_arbitration_receipt(
    vanilla: OutputCandidate,
    rlc: OutputCandidate,
    *,
    margin: float = DEFAULT_ARBITRATION_MARGIN,
) -> tuple[dict[str, Any], str, tuple[int, ...]]:
    """Build a public decision receipt and return the selected private output."""

    _validate_pair(vanilla, rlc)
    normalized_margin = _margin(margin)
    decision, reason = _decide(vanilla, rlc, margin=normalized_margin)
    selected = (
        vanilla
        if decision is ArbitrationDecision.RETAIN_VANILLA
        else rlc
        if decision is ArbitrationDecision.SELECT_RLC
        else None
    )
    selected_text = selected.text if selected is not None else ""
    selected_tokens = selected.tokens if selected is not None else ()
    payload = {
        "schema": OUTPUT_ARBITRATION_SCHEMA,
        "binding": {
            "request_sha256": vanilla.request_sha256,
            "model_sha256": vanilla.model_sha256,
            "seed": vanilla.seed,
            "contract_sha256": vanilla.contract_sha256,
        },
        "policy": {
            "margin": normalized_margin,
            "incumbent": CandidateSource.VANILLA.value,
            "challenger": CandidateSource.RLC.value,
            "selection_rule": "independent_full_span_lcb_gt_incumbent_ucb_plus_margin",
            "unmeasured_policy": "retain_incumbent",
            "proxy_score_authority": "none",
            "material_regression_policy": "reject_challenger",
        },
        "candidates": {
            CandidateSource.VANILLA.value: vanilla.public_evidence(),
            CandidateSource.RLC.value: rlc.public_evidence(),
        },
        "decision": decision.value,
        "reason": reason,
        "selected_source": selected.source.value if selected is not None else "none",
        "selected_output": {
            "text_sha256": _text_sha256(selected_text) if selected is not None else "",
            "tokens_sha256": (
                _canonical_sha256(list(selected_tokens)) if selected is not None else ""
            ),
            "token_count": len(selected_tokens),
        },
        "authority": "sealed_output_arbitration",
    }
    receipt = {**payload, "receipt_sha256": _canonical_sha256(payload)}
    assert_output_arbitration_no_regression(receipt)
    return receipt, selected_text, selected_tokens


def assert_output_arbitration_no_regression(receipt: Mapping[str, Any]) -> None:
    """Reject any receipt whose RLC selection lacks the required authority."""

    if not isinstance(receipt, Mapping):
        raise ValueError("output arbitration receipt is missing")
    candidates = receipt.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("output arbitration candidates are missing")
    if receipt.get("decision") != ArbitrationDecision.SELECT_RLC.value:
        return
    vanilla = candidates.get(CandidateSource.VANILLA.value)
    rlc = candidates.get(CandidateSource.RLC.value)
    policy = receipt.get("policy")
    if not all(isinstance(value, Mapping) for value in (vanilla, rlc, policy)):
        raise ValueError("RLC selection evidence is malformed")
    vanilla_admissible = vanilla.get("admissible") is True
    rlc_authoritative = rlc.get("authoritative") is True
    rlc_interval = rlc.get("quality_interval")
    vanilla_interval = vanilla.get("quality_interval")
    if not rlc_authoritative:
        raise ValueError("RLC selection lacks independent full-span authority")
    if vanilla_admissible:
        if not isinstance(rlc_interval, Mapping) or not isinstance(
            vanilla_interval,
            Mapping,
        ):
            raise ValueError("RLC dominance intervals are missing")
        lower = _bound(rlc_interval.get("lower_bound"), field="RLC lower bound")
        upper = _bound(
            vanilla_interval.get("upper_bound"),
            field="vanilla upper bound",
        )
        margin = _margin(policy.get("margin"))
        if not lower > upper + margin:
            raise ValueError("RLC selection does not dominate the vanilla incumbent")


def validate_output_arbitration_receipt(
    receipt: Any,
    *,
    vanilla: OutputCandidate,
    rlc: OutputCandidate,
    expected_margin: float = DEFAULT_ARBITRATION_MARGIN,
    expected_output_text: str | None = None,
    expected_output_tokens: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Rebuild the decision from private candidates in the validating domain."""

    if not isinstance(receipt, Mapping):
        raise ValueError("output arbitration receipt is missing")
    expected, selected_text, selected_tokens = build_output_arbitration_receipt(
        vanilla,
        rlc,
        margin=expected_margin,
    )
    if dict(receipt) != expected:
        raise ValueError("output arbitration receipt reconstruction differs")
    if expected_output_text is not None and expected_output_text != selected_text:
        raise ValueError("output arbitration selected text differs")
    if expected_output_tokens is not None and tuple(expected_output_tokens) != selected_tokens:
        raise ValueError("output arbitration selected tokens differ")
    assert_output_arbitration_no_regression(receipt)
    return dict(receipt)


__all__ = [
    "ArbitrationDecision",
    "CandidateSource",
    "DEFAULT_ARBITRATION_MARGIN",
    "MAX_ARBITRATION_OUTPUT_TOKENS",
    "OUTPUT_ARBITRATION_SCHEMA",
    "OutputCandidate",
    "assert_output_arbitration_no_regression",
    "build_output_arbitration_receipt",
    "validate_output_arbitration_receipt",
]
