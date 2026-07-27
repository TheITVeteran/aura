"""Proof that the answer came out of the latent path, and not past it.

SPARK-066 asks for the selected architecture to run inside the resident
checkpoint's penultimate/latent execution, with exact model and adapter
identity bound, and for successful output to be provably *not* an ordinary
generation or a shallow orchestration fallback wearing the latent path's name.

This repository has already paid for that last clause. CP227's accuracy gate
was voided after the fact because `_decode` ran outside
`recurrence_adapter_scope`: both arms decoded the bare base model, the "on" and
"off" numbers matched exactly, and the negative result meant nothing. Nothing
lied. Nobody checked that the treatment was attached, and an unchecked
treatment reported as a run treatment is indistinguishable from a real null.

So this receipt makes the four ways that failure happens refusable:

1. **The adapter that was never attached.** An adapter declared attached must
   name the blocks it activated in, and those must be exactly the blocks it was
   expected to activate in. An empty activation list with `attached: true` is
   invalid, not "presumably fine".
2. **The depth that never recurred.** Passes carry per-pass state digests. If
   several passes report identical states, no recurrence happened, and the
   receipt cannot claim depth it did not execute.
3. **The state that was computed and then dropped.** The decoded state must
   *be* the final pass's state. A run that recurred beautifully and then
   decoded from something else did not use the latent path.
4. **The fallback wearing the latent name.** The mechanism is declared, and a
   receipt that records a fallback cannot also claim `recurrent_latent`.

The module is a strict validator over data with no model imports, so the
independent verifier can refuse a campaign's latent claim without loading
twenty gigabytes of weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

PENULTIMATE_RECEIPT_SCHEMA: Final = "aura.rlc.penultimate_execution.v1"

RECURRENT_LATENT: Final = "recurrent_latent"
ORDINARY_GENERATION: Final = "ordinary_generation"
SHALLOW_ORCHESTRATION: Final = "shallow_orchestration"
MECHANISMS: Final = (RECURRENT_LATENT, ORDINARY_GENERATION, SHALLOW_ORCHESTRATION)

PROVEN: Final = "PROVEN"
REFUSED: Final = "REFUSED"

_SHA256_PATTERN: Final = re.compile(r"\A[0-9a-f]{64}\Z")
_IDENTITY_FIELDS: Final = frozenset(
    {
        "checkpoint_sha256",
        "tokenizer_sha256",
        "parameter_count",
        "quantization",
        "layer_count",
    }
)
_ADAPTER_FIELDS: Final = frozenset(
    {"adapter_sha256", "attached", "expected_blocks", "activated_blocks"}
)
_PASS_FIELDS: Final = frozenset({"ordinal", "state_sha256", "delta_l2"})
_EXECUTION_FIELDS: Final = frozenset(
    {"layer_index", "passes", "decode_state_sha256", "decoded_token_count"}
)
_FALLBACK_FIELDS: Final = frozenset({"occurred", "reason"})
_MAX_PASSES: Final = 256
_MAX_BLOCKS: Final = 1024


class PenultimateReceiptError(ValueError):
    """A penultimate-execution receipt is malformed."""


def _fail(code: str) -> Never:
    raise PenultimateReceiptError(str(code or "penultimate_receipt_invalid"))


def _sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise PenultimateReceiptError("penultimate_receipt_noncanonical_value") from exc
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.match(value))


def _count(value: Any, code: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(code)
    return value


def _blocks(value: Any, code: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(code)
    if len(value) > _MAX_BLOCKS:
        _fail(code)
    seen: set[int] = set()
    for item in value:
        index = _count(item, code)
        if index in seen:
            _fail(code)
        seen.add(index)
    return sorted(seen)


def model_identity(
    *,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
    parameter_count: int,
    quantization: str,
    layer_count: int,
) -> dict[str, Any]:
    """Bind exactly which weights ran."""

    if not _is_sha256(checkpoint_sha256) or not _is_sha256(tokenizer_sha256):
        _fail("penultimate_receipt_model_identity_invalid")
    if not isinstance(quantization, str) or not quantization.strip():
        _fail("penultimate_receipt_model_identity_invalid")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "parameter_count": _count(
            parameter_count, "penultimate_receipt_model_identity_invalid", minimum=1
        ),
        "quantization": quantization,
        "layer_count": _count(
            layer_count, "penultimate_receipt_model_identity_invalid", minimum=2
        ),
    }


def adapter_binding(
    *,
    adapter_sha256: str | None,
    attached: bool,
    expected_blocks: Sequence[int],
    activated_blocks: Sequence[int],
) -> dict[str, Any]:
    """Bind whether the treatment was actually in the forward path.

    This is the CP227 repair. ``attached`` is not taken at its word: an adapter
    that claims attachment must name the blocks it fired in, and they must be
    exactly the blocks it was expected to fire in. Silence is refused.
    """

    if type(attached) is not bool:
        _fail("penultimate_receipt_adapter_invalid")
    expected = _blocks(expected_blocks, "penultimate_receipt_adapter_blocks_invalid")
    activated = _blocks(activated_blocks, "penultimate_receipt_adapter_blocks_invalid")

    if attached:
        if adapter_sha256 is None or not _is_sha256(adapter_sha256):
            _fail("penultimate_receipt_adapter_identity_invalid")
        if not expected:
            _fail("penultimate_receipt_adapter_expected_blocks_missing")
        if activated != expected:
            # An adapter present in the process but absent from the forward
            # pass is exactly the CP227 failure. It is refused here rather
            # than discovered after a campaign's verdict is published.
            _fail("penultimate_receipt_adapter_did_not_activate")
    else:
        if adapter_sha256 is not None or activated:
            _fail("penultimate_receipt_adapter_invalid")

    return {
        "adapter_sha256": adapter_sha256,
        "attached": attached,
        "expected_blocks": expected,
        "activated_blocks": activated,
    }


def recurrent_pass(*, ordinal: int, state_sha256: str, delta_l2: float) -> dict[str, Any]:
    if not _is_sha256(state_sha256):
        _fail("penultimate_receipt_pass_state_invalid")
    if isinstance(delta_l2, bool) or not isinstance(delta_l2, (int, float)):
        _fail("penultimate_receipt_pass_delta_invalid")
    delta = round(float(delta_l2), 9)
    if delta != delta or delta < 0.0:
        _fail("penultimate_receipt_pass_delta_invalid")
    return {
        "ordinal": _count(ordinal, "penultimate_receipt_pass_ordinal_invalid"),
        "state_sha256": state_sha256,
        "delta_l2": delta,
    }


def penultimate_execution_receipt(
    *,
    mechanism: str,
    identity: Mapping[str, Any],
    adapter: Mapping[str, Any],
    layer_index: int,
    passes: Sequence[Mapping[str, Any]],
    decode_state_sha256: str,
    decoded_token_count: int,
    answer_sha256: str,
    fallback_occurred: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    """Build one receipt for one execution, refusing incoherent claims."""

    if mechanism not in MECHANISMS:
        _fail("penultimate_receipt_mechanism_unknown")
    if type(fallback_occurred) is not bool:
        _fail("penultimate_receipt_fallback_invalid")
    if fallback_occurred:
        if not isinstance(fallback_reason, str) or not fallback_reason.strip():
            _fail("penultimate_receipt_fallback_reason_missing")
        if mechanism == RECURRENT_LATENT:
            # A run that fell back did not run the latent path, whatever the
            # caller would prefer the receipt to say.
            _fail("penultimate_receipt_fallback_claims_latent")
    elif fallback_reason is not None:
        _fail("penultimate_receipt_fallback_invalid")

    normalized_identity = model_identity(
        checkpoint_sha256=identity.get("checkpoint_sha256"),
        tokenizer_sha256=identity.get("tokenizer_sha256"),
        parameter_count=identity.get("parameter_count"),
        quantization=identity.get("quantization"),
        layer_count=identity.get("layer_count"),
    )
    if not isinstance(adapter, Mapping) or set(adapter) != _ADAPTER_FIELDS:
        _fail("penultimate_receipt_adapter_fields_differ")
    normalized_adapter = adapter_binding(
        adapter_sha256=adapter.get("adapter_sha256"),
        attached=adapter.get("attached"),
        expected_blocks=adapter.get("expected_blocks"),
        activated_blocks=adapter.get("activated_blocks"),
    )

    index = _count(layer_index, "penultimate_receipt_layer_invalid")
    if index != normalized_identity["layer_count"] - 2:
        # "Penultimate" is a position, not a label a caller may reassign.
        _fail("penultimate_receipt_layer_is_not_penultimate")

    if not isinstance(passes, Sequence) or isinstance(passes, (str, bytes)):
        _fail("penultimate_receipt_passes_invalid")
    if not passes or len(passes) > _MAX_PASSES:
        _fail("penultimate_receipt_passes_invalid")
    rows: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(passes):
        if not isinstance(raw, Mapping) or set(raw) != _PASS_FIELDS:
            _fail("penultimate_receipt_pass_fields_differ")
        row = recurrent_pass(
            ordinal=raw["ordinal"],
            state_sha256=raw["state_sha256"],
            delta_l2=raw["delta_l2"],
        )
        if row["ordinal"] != ordinal:
            _fail("penultimate_receipt_pass_out_of_order")
        rows.append(row)

    if not _is_sha256(decode_state_sha256) or not _is_sha256(answer_sha256):
        _fail("penultimate_receipt_decode_invalid")

    body = {
        "schema": PENULTIMATE_RECEIPT_SCHEMA,
        "mechanism": mechanism,
        "model_identity": normalized_identity,
        "adapter": normalized_adapter,
        "execution": {
            "layer_index": index,
            "passes": rows,
            "decode_state_sha256": decode_state_sha256,
            "decoded_token_count": _count(
                decoded_token_count, "penultimate_receipt_decode_invalid", minimum=1
            ),
        },
        "answer_sha256": answer_sha256,
        "fallback": {"occurred": fallback_occurred, "reason": fallback_reason},
    }
    return {**body, "receipt_sha256": _sha256(body)}


def validate_penultimate_receipt(value: Any) -> dict[str, Any]:
    """Re-derive a receipt from its own fields and refuse any drift."""

    if not isinstance(value, Mapping) or value.get("schema") != PENULTIMATE_RECEIPT_SCHEMA:
        _fail("penultimate_receipt_invalid")
    execution = value.get("execution")
    fallback = value.get("fallback")
    if (
        not isinstance(execution, Mapping)
        or set(execution) != _EXECUTION_FIELDS
        or not isinstance(fallback, Mapping)
        or set(fallback) != _FALLBACK_FIELDS
        or not isinstance(value.get("model_identity"), Mapping)
        or set(value["model_identity"]) != _IDENTITY_FIELDS
    ):
        _fail("penultimate_receipt_fields_differ")
    passes = execution.get("passes")
    if not isinstance(passes, Sequence) or isinstance(passes, (str, bytes)):
        _fail("penultimate_receipt_passes_invalid")

    normalized = penultimate_execution_receipt(
        mechanism=value.get("mechanism"),
        identity=value["model_identity"],
        adapter=value.get("adapter"),
        layer_index=execution.get("layer_index"),
        passes=[dict(row) if isinstance(row, Mapping) else row for row in passes],
        decode_state_sha256=execution.get("decode_state_sha256"),
        decoded_token_count=execution.get("decoded_token_count"),
        answer_sha256=value.get("answer_sha256"),
        fallback_occurred=fallback.get("occurred"),
        fallback_reason=fallback.get("reason"),
    )
    if dict(value) != normalized:
        _fail("penultimate_receipt_differs")
    return normalized


def latent_execution_verdict(
    receipt: Mapping[str, Any],
    *,
    require_adapter: bool,
    minimum_passes: int = 1,
) -> dict[str, Any]:
    """Decide whether this receipt actually proves latent execution.

    Returns a verdict either way, naming every reason. A refusal here is the
    intended outcome for an ordinary generation -- it is not an error, it is
    the receipt correctly declining to claim something it did not do.
    """

    normalized = validate_penultimate_receipt(receipt)
    reasons: list[dict[str, Any]] = []

    if normalized["mechanism"] != RECURRENT_LATENT:
        reasons.append(
            {"reason": "mechanism_is_not_latent", "mechanism": normalized["mechanism"]}
        )
    if normalized["fallback"]["occurred"]:
        reasons.append(
            {
                "reason": "fallback_occurred",
                "detail": normalized["fallback"]["reason"],
            }
        )
    if require_adapter and not normalized["adapter"]["attached"]:
        reasons.append({"reason": "treatment_adapter_not_attached"})

    passes = normalized["execution"]["passes"]
    floor = _count(minimum_passes, "penultimate_receipt_minimum_passes_invalid", minimum=1)
    if len(passes) < floor:
        reasons.append(
            {
                "reason": "fewer_passes_than_required",
                "passes": len(passes),
                "required": floor,
            }
        )

    # Several passes that all landed on the same state did not recur. This is
    # the T=1 identity wearing a depth label.
    distinct = {row["state_sha256"] for row in passes}
    if len(passes) > 1 and len(distinct) == 1:
        reasons.append(
            {"reason": "no_recurrence_occurred", "passes": len(passes), "distinct": 1}
        )

    # The decoded state must be the state the recurrence produced. A run that
    # recurred and then decoded from something else did not use its own work.
    if normalized["execution"]["decode_state_sha256"] != passes[-1]["state_sha256"]:
        reasons.append(
            {
                "reason": "decode_did_not_consume_final_state",
                "final_pass_state_sha256": passes[-1]["state_sha256"],
                "decode_state_sha256": normalized["execution"]["decode_state_sha256"],
            }
        )

    return {
        "verdict": REFUSED if reasons else PROVEN,
        "receipt_sha256": normalized["receipt_sha256"],
        "mechanism": normalized["mechanism"],
        "passes": len(passes),
        "distinct_pass_states": len(distinct),
        "adapter_attached": normalized["adapter"]["attached"],
        "reasons": reasons,
    }


__all__ = [
    "MECHANISMS",
    "ORDINARY_GENERATION",
    "PENULTIMATE_RECEIPT_SCHEMA",
    "PROVEN",
    "RECURRENT_LATENT",
    "REFUSED",
    "SHALLOW_ORCHESTRATION",
    "PenultimateReceiptError",
    "adapter_binding",
    "latent_execution_verdict",
    "model_identity",
    "penultimate_execution_receipt",
    "recurrent_pass",
    "validate_penultimate_receipt",
]
