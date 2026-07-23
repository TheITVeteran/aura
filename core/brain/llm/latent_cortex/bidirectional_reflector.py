"""Read-only full-trace reflection over recurrent hidden-state transitions."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256

BIDIRECTIONAL_REFLECTOR_RECEIPT_SCHEMA = (
    "aura.rlc.bidirectional_reflector_receipt.v1"
)
REFLECTOR_OBSERVATION_SCHEMA = "aura.rlc.reflector_transition_observation.v1"
MAX_STATE_WIDTH = 16_384
MAX_BUCKETS = 64
MAX_SLOT_BUCKETS = 8
MAX_SLOTS = 64
MAX_TRANSITIONS = 256


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _vector_sha256(value: Sequence[float]) -> str:
    return hashlib.sha256(
        ",".join(f"{float(item):.8f}" for item in value).encode("ascii")
    ).hexdigest()


def _finite_vector(
    value: Sequence[float],
    *,
    width: int | None = None,
) -> list[float]:
    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("reflector hidden vector must be a sequence")
    vector = [round(float(item), 8) for item in value]
    if (
        not vector
        or len(vector) > MAX_STATE_WIDTH
        or (width is not None and len(vector) != width)
        or any(not math.isfinite(item) for item in vector)
    ):
        raise ValueError("reflector hidden vector is invalid")
    return vector


def _hidden_sketch(value: Sequence[float]) -> list[float]:
    """Bounded block mean/RMS sketch in which every hidden dimension contributes."""

    vector = [math.asinh(item) for item in _finite_vector(value)]
    width = len(vector)
    buckets = min(MAX_BUCKETS, width)
    means: list[float] = []
    rms_values: list[float] = []
    for index in range(buckets):
        start = index * width // buckets
        end = (index + 1) * width // buckets
        block = vector[start:end]
        means.append(round(sum(block) / len(block), 8))
        rms_values.append(
            round(math.sqrt(sum(item * item for item in block) / len(block)), 8)
        )
    return means + rms_values


def _slot_sketch(value: Sequence[float]) -> list[float]:
    vector = [math.asinh(item) for item in _finite_vector(value)]
    width = len(vector)
    buckets = min(MAX_SLOT_BUCKETS, width)
    means: list[float] = []
    rms_values: list[float] = []
    for index in range(buckets):
        start = index * width // buckets
        end = (index + 1) * width // buckets
        block = vector[start:end]
        means.append(round(sum(block) / len(block), 8))
        rms_values.append(
            round(math.sqrt(sum(item * item for item in block) / len(block)), 8)
        )
    return means + rms_values


def position_hidden_sketch(value: Sequence[float]) -> list[float]:
    """Public training/runtime map for one latent sequence position."""

    return _slot_sketch(value)


def _pooled_hidden(state: Any) -> list[float]:
    import mlx.core as mx

    if (
        state.ndim < 1
        or not 1 <= int(state.shape[-1]) <= MAX_STATE_WIDTH
        or int(state.size) < int(state.shape[-1])
        or not bool(mx.all(mx.isfinite(state)).item())
    ):
        raise ValueError("reflector hidden state is incompatible or non-finite")
    axes = tuple(range(state.ndim - 1))
    pooled = mx.mean(state.astype(mx.float32), axis=axes)
    mx.eval(pooled)
    return _finite_vector(pooled.tolist())


def _position_hidden(state: Any) -> list[list[float]]:
    import mlx.core as mx

    if (
        state.ndim < 2
        or not 1 <= int(state.shape[-2]) <= MAX_SLOTS
        or not 1 <= int(state.shape[-1]) <= MAX_STATE_WIDTH
        or not bool(mx.all(mx.isfinite(state)).item())
    ):
        raise ValueError("reflector position state is incompatible or non-finite")
    axes = tuple(range(state.ndim - 2))
    positions = (
        mx.mean(state.astype(mx.float32), axis=axes)
        if axes
        else state.astype(mx.float32)
    )
    mx.eval(positions)
    values = positions.tolist()
    if not isinstance(values, list) or len(values) != int(state.shape[-2]):
        raise ValueError("reflector position state is invalid")
    return [_finite_vector(row, width=int(state.shape[-1])) for row in values]


def observe_reflector_vectors(
    prior_hidden: Sequence[float],
    proposal_hidden: Sequence[float],
    admitted_hidden: Sequence[float],
    *,
    branch_index: int,
    branch_step: int,
    prior_state_sha256: str,
    proposal_state_sha256: str,
    admitted_state_sha256: str,
    accepted: bool,
    prior_positions: Sequence[Sequence[float]] | None = None,
    proposal_positions: Sequence[Sequence[float]] | None = None,
    admitted_positions: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    if (
        type(branch_index) is not int
        or branch_index < 0
        or type(branch_step) is not int
        or not 0 <= branch_step < MAX_TRANSITIONS
        or type(accepted) is not bool
        or not all(
            _is_sha256(value)
            for value in (
                prior_state_sha256,
                proposal_state_sha256,
                admitted_state_sha256,
            )
        )
    ):
        raise ValueError("reflector observation identity is invalid")
    prior = _finite_vector(prior_hidden)
    proposal = _finite_vector(proposal_hidden, width=len(prior))
    admitted = _finite_vector(admitted_hidden, width=len(prior))
    prior_sketch = _hidden_sketch(prior)
    proposal_sketch = _hidden_sketch(proposal)
    admitted_sketch = _hidden_sketch(admitted)
    prior_rows = (
        [prior] if prior_positions is None else list(prior_positions)
    )
    proposal_rows = (
        [proposal]
        if proposal_positions is None
        else list(proposal_positions)
    )
    admitted_rows = (
        [admitted]
        if admitted_positions is None
        else list(admitted_positions)
    )
    if (
        not 1 <= len(prior_rows) <= MAX_SLOTS
        or len(proposal_rows) != len(prior_rows)
        or len(admitted_rows) != len(prior_rows)
    ):
        raise ValueError("reflector position coverage differs")
    prior_position_sketches = [
        _slot_sketch(_finite_vector(row, width=len(prior))) for row in prior_rows
    ]
    proposal_position_sketches = [
        _slot_sketch(_finite_vector(row, width=len(prior)))
        for row in proposal_rows
    ]
    admitted_position_sketches = [
        _slot_sketch(_finite_vector(row, width=len(prior)))
        for row in admitted_rows
    ]
    position_sketch_width = len(prior_position_sketches[0])
    payload = {
        "schema": REFLECTOR_OBSERVATION_SCHEMA,
        "branch_index": branch_index,
        "branch_step": branch_step,
        "state_width": len(prior),
        "sketch_width": len(prior_sketch),
        "position_count": len(prior_rows),
        "position_sketch_width": position_sketch_width,
        "prior_reasoning_sha256": prior_state_sha256,
        "proposal_reasoning_sha256": proposal_state_sha256,
        "admitted_reasoning_sha256": admitted_state_sha256,
        "accepted": accepted,
        "prior_sketch": prior_sketch,
        "prior_sketch_sha256": _vector_sha256(prior_sketch),
        "proposal_sketch": proposal_sketch,
        "proposal_sketch_sha256": _vector_sha256(proposal_sketch),
        "admitted_sketch": admitted_sketch,
        "admitted_sketch_sha256": _vector_sha256(admitted_sketch),
        "prior_position_sketches": prior_position_sketches,
        "prior_position_sketches_sha256": canonical_sha256(
            prior_position_sketches
        ),
        "proposal_position_sketches": proposal_position_sketches,
        "proposal_position_sketches_sha256": canonical_sha256(
            proposal_position_sketches
        ),
        "admitted_position_sketches": admitted_position_sketches,
        "admitted_position_sketches_sha256": canonical_sha256(
            admitted_position_sketches
        ),
    }
    return {**payload, "observation_sha256": canonical_sha256(payload)}


def observe_reflector_transition(
    prior_state: Any,
    proposal_state: Any,
    admitted_state: Any,
    **identity: Any,
) -> dict[str, Any]:
    return observe_reflector_vectors(
        _pooled_hidden(prior_state),
        _pooled_hidden(proposal_state),
        _pooled_hidden(admitted_state),
        prior_positions=_position_hidden(prior_state),
        proposal_positions=_position_hidden(proposal_state),
        admitted_positions=_position_hidden(admitted_state),
        **identity,
    )


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("reflector context cannot be empty")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("reflector context widths differ")
    return [
        round(sum(float(vector[index]) for vector in vectors) / len(vectors), 8)
        for index in range(width)
    ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(item) ** 2 for item in left))
    right_norm = math.sqrt(sum(float(item) ** 2 for item in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return round(max(-1.0, min(1.0, dot / (left_norm * right_norm))), 10)


def _delta_rms(left: Sequence[float], right: Sequence[float]) -> float:
    return round(
        math.sqrt(
            sum(
                (float(a) - float(b)) ** 2
                for a, b in zip(left, right, strict=True)
            )
            / len(left)
        ),
        10,
    )


def _reflection_rows(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= len(observations) <= MAX_TRANSITIONS:
        raise ValueError("reflector trace length is outside bounds")
    premise = observations[0]["prior_sketch"]
    conclusion = observations[-1]["admitted_sketch"]
    admitted = [row["admitted_sketch"] for row in observations]
    rows: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        prior = observation["prior_sketch"]
        proposal = observation["proposal_sketch"]
        admitted_state = observation["admitted_sketch"]
        prefix = _mean(admitted[: index + 1])
        suffix = _mean(admitted[index:])
        reflected = (
            list(prior)
            + list(proposal)
            + list(admitted_state)
            + prefix
            + suffix
            + list(premise)
            + list(conclusion)
        )
        metrics = {
            "local_proposal_delta_rms": _delta_rms(prior, proposal),
            "local_admitted_delta_rms": _delta_rms(prior, admitted_state),
            "proposal_admitted_delta_rms": _delta_rms(
                proposal, admitted_state
            ),
            "proposal_to_premise_cosine": _cosine(proposal, premise),
            "proposal_to_conclusion_cosine": _cosine(proposal, conclusion),
            "proposal_to_prefix_cosine": _cosine(proposal, prefix),
            "proposal_to_suffix_cosine": _cosine(proposal, suffix),
            "prefix_to_suffix_cosine": _cosine(prefix, suffix),
        }
        rows.append(
            {
                "ordinal": index,
                "branch_step": observation["branch_step"],
                "past_context_count": index,
                "future_context_count": len(observations) - index - 1,
                "uses_past_context": index > 0,
                "uses_future_context": index < len(observations) - 1,
                "prefix_context_sha256": _vector_sha256(prefix),
                "suffix_context_sha256": _vector_sha256(suffix),
                "reflected_state_sha256": _vector_sha256(reflected),
                "metrics": metrics,
            }
        )
    summary = {
        "trace_length": len(observations),
        "premise_sketch_sha256": _vector_sha256(premise),
        "conclusion_sketch_sha256": _vector_sha256(conclusion),
        "premise_conclusion_cosine": _cosine(premise, conclusion),
        "premise_conclusion_delta_rms": _delta_rms(premise, conclusion),
        "rows_with_past_context": sum(row["uses_past_context"] for row in rows),
        "rows_with_future_context": sum(row["uses_future_context"] for row in rows),
    }
    return rows, summary


def _validate_observation(
    value: Any,
    *,
    branch_index: int,
    source_transition: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema",
        "branch_index",
        "branch_step",
        "state_width",
        "sketch_width",
        "position_count",
        "position_sketch_width",
        "prior_reasoning_sha256",
        "proposal_reasoning_sha256",
        "admitted_reasoning_sha256",
        "accepted",
        "prior_sketch",
        "prior_sketch_sha256",
        "proposal_sketch",
        "proposal_sketch_sha256",
        "admitted_sketch",
        "admitted_sketch_sha256",
        "prior_position_sketches",
        "prior_position_sketches_sha256",
        "proposal_position_sketches",
        "proposal_position_sketches_sha256",
        "admitted_position_sketches",
        "admitted_position_sketches_sha256",
        "observation_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("reflector observation fields differ")
    row = dict(value)
    payload = {key: row[key] for key in fields - {"observation_sha256"}}
    if (
        row["schema"] != REFLECTOR_OBSERVATION_SCHEMA
        or row["branch_index"] != branch_index
        or row["branch_step"] != source_transition.get("branch_step")
        or row["prior_reasoning_sha256"]
        != source_transition.get("prior_reasoning_sha256")
        or row["proposal_reasoning_sha256"]
        != source_transition.get("proposal_reasoning_sha256")
        or row["admitted_reasoning_sha256"]
        != source_transition.get("admitted_reasoning_sha256")
        or row["accepted"] is not source_transition.get("accepted")
        or row["observation_sha256"] != canonical_sha256(payload)
        or type(row["state_width"]) is not int
        or not 1 <= row["state_width"] <= MAX_STATE_WIDTH
        or type(row["sketch_width"]) is not int
        or row["sketch_width"]
        != 2 * min(MAX_BUCKETS, row["state_width"])
        or type(row["position_count"]) is not int
        or not 1 <= row["position_count"] <= MAX_SLOTS
        or row["position_sketch_width"]
        != 2 * min(MAX_SLOT_BUCKETS, row["state_width"])
    ):
        raise ValueError("reflector observation identity is invalid")
    for prefix in ("prior", "proposal", "admitted"):
        vector = row[f"{prefix}_sketch"]
        if (
            not isinstance(vector, list)
            or len(vector) != row["sketch_width"]
            or _finite_vector(vector, width=row["sketch_width"]) != vector
            or row[f"{prefix}_sketch_sha256"] != _vector_sha256(vector)
        ):
            raise ValueError("reflector observation sketch is invalid")
        positions = row[f"{prefix}_position_sketches"]
        if (
            not isinstance(positions, list)
            or len(positions) != row["position_count"]
            or any(
                not isinstance(position, list)
                or len(position) != row["position_sketch_width"]
                or _finite_vector(
                    position,
                    width=row["position_sketch_width"],
                )
                != position
                for position in positions
            )
            or row[f"{prefix}_position_sketches_sha256"]
            != canonical_sha256(positions)
        ):
            raise ValueError("reflector position sketches are invalid")
    if (
        row["accepted"]
        and row["proposal_reasoning_sha256"] != row["admitted_reasoning_sha256"]
    ):
        raise ValueError("accepted reflector transition did not admit proposal")
    if (
        not row["accepted"]
        and row["prior_reasoning_sha256"] != row["admitted_reasoning_sha256"]
    ):
        raise ValueError("rejected reflector transition did not retain prior")
    return row


def build_bidirectional_reflector_receipt(
    *,
    branches: list[Any],
    update_acceptance: Mapping[str, Any],
    selected_branch: int,
    budget: Any | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(update_acceptance, Mapping)
        or not _is_sha256(update_acceptance.get("receipt_sha256"))
        or type(selected_branch) is not int
        or not 0 <= selected_branch < len(branches)
    ):
        raise ValueError("reflector receipt source is invalid")
    branch_rows: list[dict[str, Any]] = []
    for branch in branches:
        observations = [dict(row) for row in branch.reflector_trace]
        reflections, summary = _reflection_rows(observations)
        branch_rows.append(
            {
                "branch_index": int(branch.index),
                "observations": observations,
                "reflections": reflections,
                **summary,
            }
        )
        if budget is not None:
            sketch_width = int(observations[0]["sketch_width"])
            position_width = int(observations[0]["position_sketch_width"])
            position_count = int(observations[0]["position_count"])
            budget.charge_tensor_work(
                "bidirectional_reflector_review",
                element_reads=len(observations)
                * 3
                * (sketch_width + position_count * position_width),
                host_scalar_ops=len(observations)
                * 24
                * (sketch_width + position_count * position_width),
            )
    payload = {
        "schema": BIDIRECTIONAL_REFLECTOR_RECEIPT_SCHEMA,
        "review_mode": "full_sequence_bidirectional_hidden_trace",
        "update_acceptance_sha256": update_acceptance["receipt_sha256"],
        "branches": branch_rows,
        "observation_count": sum(
            row["trace_length"] for row in branch_rows
        ),
        "rows_with_past_context": sum(
            row["rows_with_past_context"] for row in branch_rows
        ),
        "rows_with_future_context": sum(
            row["rows_with_future_context"] for row in branch_rows
        ),
        "selected_branch": selected_branch,
        "selected_branch_premise_conclusion": {
            "cosine": branch_rows[selected_branch][
                "premise_conclusion_cosine"
            ],
            "delta_rms": branch_rows[selected_branch][
                "premise_conclusion_delta_rms"
            ],
        },
        "complete_trace_inspected": True,
        "hidden_trace_only": True,
        "answer_text_consumed": False,
        "state_mutation_authorized": False,
        "selection_authorized": False,
        "repair_authorized": False,
        "attention_perturbation_authorized": False,
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_bidirectional_reflector_receipt(
        receipt,
        update_acceptance=update_acceptance,
        expected_n_branches=len(branches),
    )


def validate_bidirectional_reflector_receipt(
    value: Any,
    *,
    update_acceptance: Mapping[str, Any],
    expected_n_branches: int,
) -> dict[str, Any]:
    fields = {
        "schema",
        "review_mode",
        "update_acceptance_sha256",
        "branches",
        "observation_count",
        "rows_with_past_context",
        "rows_with_future_context",
        "selected_branch",
        "selected_branch_premise_conclusion",
        "complete_trace_inspected",
        "hidden_trace_only",
        "answer_text_consumed",
        "state_mutation_authorized",
        "selection_authorized",
        "repair_authorized",
        "attention_perturbation_authorized",
        "receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or not isinstance(update_acceptance, Mapping)
        or not _is_sha256(update_acceptance.get("receipt_sha256"))
    ):
        raise ValueError("reflector receipt fields/source differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    branches = receipt["branches"]
    source_branches = update_acceptance.get("branches")
    if (
        receipt["schema"] != BIDIRECTIONAL_REFLECTOR_RECEIPT_SCHEMA
        or receipt["review_mode"]
        != "full_sequence_bidirectional_hidden_trace"
        or receipt["update_acceptance_sha256"]
        != update_acceptance["receipt_sha256"]
        or receipt["receipt_sha256"] != canonical_sha256(payload)
        or type(expected_n_branches) is not int
        or expected_n_branches < 1
        or not isinstance(branches, list)
        or len(branches) != expected_n_branches
        or not isinstance(source_branches, list)
        or len(source_branches) != expected_n_branches
    ):
        raise ValueError("reflector receipt identity is invalid")
    observation_count = past_count = future_count = 0
    for branch_index, branch in enumerate(branches):
        branch_fields = {
            "branch_index",
            "observations",
            "reflections",
            "trace_length",
            "premise_sketch_sha256",
            "conclusion_sketch_sha256",
            "premise_conclusion_cosine",
            "premise_conclusion_delta_rms",
            "rows_with_past_context",
            "rows_with_future_context",
        }
        if (
            not isinstance(branch, Mapping)
            or set(branch) != branch_fields
            or branch["branch_index"] != branch_index
            or not isinstance(branch["observations"], list)
            or not isinstance(branch["reflections"], list)
        ):
            raise ValueError("reflector branch evidence is invalid")
        source = source_branches[branch_index]
        transitions = (
            source.get("transitions") if isinstance(source, Mapping) else None
        )
        if (
            not isinstance(transitions, list)
            or not 1 <= len(transitions) <= MAX_TRANSITIONS
            or len(branch["observations"]) != len(transitions)
        ):
            raise ValueError("reflector trace coverage differs")
        observations = [
            _validate_observation(
                observation,
                branch_index=branch_index,
                source_transition=transitions[index],
            )
            for index, observation in enumerate(branch["observations"])
        ]
        reflections, summary = _reflection_rows(observations)
        if branch["reflections"] != reflections or any(
            branch[key] != expected for key, expected in summary.items()
        ):
            raise ValueError("reflector branch reconstruction failed")
        observation_count += summary["trace_length"]
        past_count += summary["rows_with_past_context"]
        future_count += summary["rows_with_future_context"]
    selected = receipt["selected_branch"]
    if (
        receipt["observation_count"] != observation_count
        or receipt["rows_with_past_context"] != past_count
        or receipt["rows_with_future_context"] != future_count
        or type(selected) is not int
        or not 0 <= selected < expected_n_branches
        or receipt["selected_branch_premise_conclusion"]
        != {
            "cosine": branches[selected]["premise_conclusion_cosine"],
            "delta_rms": branches[selected]["premise_conclusion_delta_rms"],
        }
        or receipt["complete_trace_inspected"] is not True
        or receipt["hidden_trace_only"] is not True
        or receipt["answer_text_consumed"] is not False
        or any(
            receipt[name] is not False
            for name in (
                "state_mutation_authorized",
                "selection_authorized",
                "repair_authorized",
                "attention_perturbation_authorized",
            )
        )
    ):
        raise ValueError("reflector aggregate evidence differs")
    return receipt


__all__ = [
    "BIDIRECTIONAL_REFLECTOR_RECEIPT_SCHEMA",
    "REFLECTOR_OBSERVATION_SCHEMA",
    "build_bidirectional_reflector_receipt",
    "observe_reflector_transition",
    "observe_reflector_vectors",
    "position_hidden_sketch",
    "validate_bidirectional_reflector_receipt",
]
