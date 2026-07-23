"""Pinned full-trace contradiction evidence over recurrent latent positions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.learning.contradiction_tensor import (
    ContradictionTensorHead,
    contradiction_channels,
    contradiction_features,
)

CONTRADICTION_TENSOR_RECEIPT_SCHEMA = (
    "aura.rlc.contradiction_tensor_receipt.v1"
)
UNAVAILABLE = "unavailable"
LEARNED = "learned"
MAX_TRANSITIONS = 256
MAX_POSITIONS = 64


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_vector(
    value: Any,
    *,
    width: int | None = None,
) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or (width is not None and len(value) != width)
    ):
        raise ValueError("contradiction tensor vector is invalid")
    vector = [round(float(item), 8) for item in value]
    if any(not math.isfinite(item) for item in vector):
        raise ValueError("contradiction tensor vector is non-finite")
    return vector


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("contradiction tensor context cannot be empty")
    width = len(vectors[0])
    rows = [_finite_vector(vector, width=width) for vector in vectors]
    return [
        round(sum(row[index] for row in rows) / len(rows), 8)
        for index in range(width)
    ]


class ContradictionTensorRuntime:
    """Pinned admitted tensor head; unavailable mode invents no evidence."""

    def __init__(
        self,
        *,
        mode: str,
        head: ContradictionTensorHead | None = None,
        head_sha256: str = "",
    ) -> None:
        if mode not in {UNAVAILABLE, LEARNED}:
            raise ValueError("contradiction tensor mode is invalid")
        if mode == UNAVAILABLE and (head is not None or head_sha256):
            raise ValueError("unavailable contradiction tensor carries a head")
        if mode == LEARNED and (
            head is None or not head.admitted or not _is_sha256(head_sha256)
        ):
            raise ValueError(
                "learned contradiction tensor requires an admitted pinned head"
            )
        if head is not None:
            head.validate()
        self.mode = mode
        self.head = head
        self.head_sha256 = head_sha256

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, Any] | None,
    ) -> ContradictionTensorRuntime:
        config = dict(value or {})
        mode = str(config.get("mode", UNAVAILABLE))
        if mode == UNAVAILABLE:
            if set(config) - {"mode"}:
                raise ValueError(
                    "unavailable contradiction tensor carries a head"
                )
            return cls(mode=UNAVAILABLE)
        if mode != LEARNED or set(config) != {
            "mode",
            "head_path",
            "head_sha256",
        }:
            raise ValueError("contradiction tensor config fields differ")
        path = config.get("head_path")
        digest = config.get("head_sha256")
        if (
            not isinstance(path, str)
            or not path.strip()
            or not _is_sha256(digest)
        ):
            raise ValueError("learned contradiction tensor config is incomplete")
        try:
            head = ContradictionTensorHead.load(
                Path(path).expanduser(),
                expected_sha256=digest,
            )
        except OSError as exc:
            raise ValueError(
                "contradiction tensor artifact is unreadable"
            ) from exc
        return cls(mode=LEARNED, head=head, head_sha256=digest)

    @property
    def manifest(self) -> dict[str, Any]:
        return {} if self.head is None else self.head.manifest()


def _branch_tensor(
    branch: Mapping[str, Any],
    *,
    runtime: ContradictionTensorRuntime,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], Any, Any]:
    observations = branch.get("observations")
    if (
        not isinstance(observations, list)
        or not 1 <= len(observations) <= MAX_TRANSITIONS
    ):
        raise ValueError("contradiction tensor trace is unavailable")
    if runtime.head is None:
        return [], [], None, None
    first = observations[0]
    if not isinstance(first, Mapping):
        raise ValueError("contradiction tensor observation is invalid")
    position_count = first.get("position_count")
    position_width = first.get("position_sketch_width")
    if (
        type(position_count) is not int
        or not 1 <= position_count <= MAX_POSITIONS
        or type(position_width) is not int
        or position_width < 1
    ):
        raise ValueError("contradiction tensor position shape is invalid")
    admitted_by_position: list[list[list[float]]] = [
        [] for _ in range(position_count)
    ]
    normalized: list[dict[str, Any]] = []
    for observation in observations:
        if (
            not isinstance(observation, Mapping)
            or observation.get("position_count") != position_count
            or observation.get("position_sketch_width") != position_width
        ):
            raise ValueError("contradiction tensor trace shape changed")
        row = dict(observation)
        for name in ("prior", "proposal", "admitted"):
            positions = row.get(f"{name}_position_sketches")
            if (
                not isinstance(positions, list)
                or len(positions) != position_count
            ):
                raise ValueError("contradiction tensor position coverage differs")
            row[f"{name}_position_sketches"] = [
                _finite_vector(position, width=position_width)
                for position in positions
            ]
        for index, position in enumerate(
            row["admitted_position_sketches"]
        ):
            admitted_by_position[index].append(position)
        normalized.append(row)
    premise = normalized[0]["prior_position_sketches"]
    conclusion = normalized[-1]["admitted_position_sketches"]
    tensor: list[list[dict[str, Any]]] = []
    step_feature_rows: list[list[tuple[float, ...]]] = []
    flat_probabilities: list[float] = []
    flat_coordinates: list[tuple[int, int]] = []
    for transition_index, observation in enumerate(normalized):
        transition_fraction = transition_index / max(1, len(normalized) - 1)
        cells: list[dict[str, Any]] = []
        cell_features: list[tuple[float, ...]] = []
        for position_index in range(position_count):
            position_fraction = position_index / max(1, position_count - 1)
            prior = observation["prior_position_sketches"][position_index]
            proposal = observation["proposal_position_sketches"][
                position_index
            ]
            admitted = observation["admitted_position_sketches"][
                position_index
            ]
            prefix = _mean(
                admitted_by_position[position_index][: transition_index + 1]
            )
            suffix = _mean(
                admitted_by_position[position_index][transition_index:]
            )
            features = contradiction_features(
                prior,
                proposal,
                admitted,
                premise[position_index],
                conclusion[position_index],
                prefix,
                suffix,
                accepted=bool(observation["accepted"]),
                transition_fraction=transition_fraction,
                position_fraction=position_fraction,
            )
            probability = (
                None
                if runtime.head is None
                else round(runtime.head.probability(features), 10)
            )
            channels = contradiction_channels(
                prior,
                proposal,
                admitted,
                premise[position_index],
                conclusion[position_index],
                prefix,
                suffix,
            )
            cell = {
                "transition_index": transition_index,
                "branch_step": observation["branch_step"],
                "position_index": position_index,
                "position_kind": "latent_workspace_sequence_position",
                "decoded_token_index": None,
                "features_sha256": canonical_sha256(
                    [round(value, 10) for value in features]
                ),
                "channels": channels,
                "contradiction_probability": probability,
            }
            cells.append(cell)
            cell_features.append(features)
            if probability is not None:
                flat_probabilities.append(probability)
                flat_coordinates.append((transition_index, position_index))
        tensor.append(cells)
        step_feature_rows.append(cell_features)
    step_rows: list[dict[str, Any]] = []
    for transition_index, cells in enumerate(tensor):
        probabilities = [
            cell["contradiction_probability"]
            for cell in cells
            if cell["contradiction_probability"] is not None
        ]
        if probabilities:
            best_position = max(
                range(len(probabilities)), key=probabilities.__getitem__
            )
            probability = round(
                runtime.head.step_probability(
                    step_feature_rows[transition_index]
                ),
                10,
            )
        else:
            best_position = None
            probability = None
        step_rows.append(
            {
                "transition_index": transition_index,
                "branch_step": normalized[transition_index]["branch_step"],
                "max_position_index": best_position,
                "contradiction_probability": probability,
            }
        )
    candidate = candidate_probability = None
    if runtime.head is not None and flat_probabilities:
        best = max(
            range(len(flat_probabilities)),
            key=flat_probabilities.__getitem__,
        )
        candidate_probability = flat_probabilities[best]
        if candidate_probability >= runtime.head.threshold:
            candidate = {
                "transition_index": flat_coordinates[best][0],
                "position_index": flat_coordinates[best][1],
            }
    return tensor, step_rows, candidate, candidate_probability


def build_contradiction_tensor_receipt(
    *,
    reflector: Mapping[str, Any],
    runtime: ContradictionTensorRuntime,
    selected_branch: int,
    budget: Any | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(reflector, Mapping)
        or not _is_sha256(reflector.get("receipt_sha256"))
        or not isinstance(reflector.get("branches"), list)
        or type(selected_branch) is not int
        or not 0 <= selected_branch < len(reflector["branches"])
    ):
        raise ValueError("contradiction tensor source is invalid")
    branches: list[dict[str, Any]] = []
    for branch_index, source in enumerate(reflector["branches"]):
        tensor, steps, candidate, probability = _branch_tensor(
            source,
            runtime=runtime,
        )
        branches.append(
            {
                "branch_index": branch_index,
                "tensor": tensor,
                "step_probabilities": steps,
                "candidate": candidate,
                "candidate_probability": probability,
            }
        )
        if budget is not None and runtime.head is not None:
            cell_count = sum(len(row) for row in tensor)
            hidden_width = int(runtime.head.input_bias.shape[0])
            budget.charge_tensor_work(
                "contradiction_tensor_head",
                element_reads=cell_count * runtime.head.feature_width,
                host_scalar_ops=cell_count
                * (
                    runtime.head.feature_width * hidden_width
                    + hidden_width
                ),
            )
    payload = {
        "schema": CONTRADICTION_TENSOR_RECEIPT_SCHEMA,
        "mode": runtime.mode,
        "head_sha256": runtime.head_sha256,
        "head_manifest": runtime.manifest,
        "reflector_sha256": reflector["receipt_sha256"],
        "branches": branches,
        "cell_count": sum(
            sum(len(row) for row in branch["tensor"]) for branch in branches
        ),
        "candidate_count": sum(
            branch["candidate"] is not None for branch in branches
        ),
        "selected_branch": selected_branch,
        "selected_branch_candidate": branches[selected_branch]["candidate"],
        "calibrated": runtime.mode == LEARNED,
        "complete_trace_consumed": runtime.mode == LEARNED,
        "latent_positions_consumed": runtime.mode == LEARNED,
        "decoded_answer_consumed": False,
        "diagnostic_only": True,
        "state_mutation_authorized": False,
        "selection_authorized": False,
        "repair_authorized": False,
        "attention_perturbation_authorized": False,
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_contradiction_tensor_receipt(
        receipt,
        expected_runtime=runtime,
        reflector=reflector,
        expected_n_branches=len(branches),
    )


def validate_contradiction_tensor_receipt(
    value: Any,
    *,
    expected_runtime: ContradictionTensorRuntime,
    reflector: Mapping[str, Any],
    expected_n_branches: int,
) -> dict[str, Any]:
    fields = {
        "schema",
        "mode",
        "head_sha256",
        "head_manifest",
        "reflector_sha256",
        "branches",
        "cell_count",
        "candidate_count",
        "selected_branch",
        "selected_branch_candidate",
        "calibrated",
        "complete_trace_consumed",
        "latent_positions_consumed",
        "decoded_answer_consumed",
        "diagnostic_only",
        "state_mutation_authorized",
        "selection_authorized",
        "repair_authorized",
        "attention_perturbation_authorized",
        "receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or not isinstance(reflector, Mapping)
        or not _is_sha256(reflector.get("receipt_sha256"))
    ):
        raise ValueError("contradiction tensor receipt fields/source differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    branches = receipt["branches"]
    source_branches = reflector.get("branches")
    if (
        receipt["schema"] != CONTRADICTION_TENSOR_RECEIPT_SCHEMA
        or receipt["mode"] != expected_runtime.mode
        or receipt["head_sha256"] != expected_runtime.head_sha256
        or receipt["head_manifest"] != expected_runtime.manifest
        or receipt["reflector_sha256"] != reflector["receipt_sha256"]
        or receipt["receipt_sha256"] != canonical_sha256(payload)
        or type(expected_n_branches) is not int
        or expected_n_branches < 1
        or not isinstance(branches, list)
        or len(branches) != expected_n_branches
        or not isinstance(source_branches, list)
        or len(source_branches) != expected_n_branches
    ):
        raise ValueError("contradiction tensor receipt identity is invalid")
    cell_count = candidate_count = 0
    candidates: list[Any] = []
    for branch_index, branch in enumerate(branches):
        branch_fields = {
            "branch_index",
            "tensor",
            "step_probabilities",
            "candidate",
            "candidate_probability",
        }
        if (
            not isinstance(branch, Mapping)
            or set(branch) != branch_fields
            or branch["branch_index"] != branch_index
        ):
            raise ValueError("contradiction tensor branch evidence is invalid")
        tensor, steps, candidate, probability = _branch_tensor(
            source_branches[branch_index],
            runtime=expected_runtime,
        )
        if (
            branch["tensor"] != tensor
            or branch["step_probabilities"] != steps
            or branch["candidate"] != candidate
            or branch["candidate_probability"] != probability
        ):
            raise ValueError("contradiction tensor reconstruction failed")
        cell_count += sum(len(row) for row in tensor)
        candidate_count += int(candidate is not None)
        candidates.append(candidate)
    selected = receipt["selected_branch"]
    if (
        receipt["cell_count"] != cell_count
        or receipt["candidate_count"] != candidate_count
        or type(selected) is not int
        or not 0 <= selected < expected_n_branches
        or receipt["selected_branch_candidate"] != candidates[selected]
        or receipt["calibrated"] is not (expected_runtime.mode == LEARNED)
        or receipt["complete_trace_consumed"]
        is not (expected_runtime.mode == LEARNED)
        or receipt["latent_positions_consumed"]
        is not (expected_runtime.mode == LEARNED)
        or receipt["decoded_answer_consumed"] is not False
        or receipt["diagnostic_only"] is not True
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
        raise ValueError("contradiction tensor aggregate evidence differs")
    return receipt


__all__ = [
    "CONTRADICTION_TENSOR_RECEIPT_SCHEMA",
    "LEARNED",
    "UNAVAILABLE",
    "ContradictionTensorRuntime",
    "build_contradiction_tensor_receipt",
    "validate_contradiction_tensor_receipt",
]
